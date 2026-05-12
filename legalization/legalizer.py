"""Legalization pass for PCB placement.

This pass runs after optimization to produce a legal, grid-snapped
placement with boundary enforcement, overlap resolution, and cleanup.

IMPORTANT: Run legalization as a single post-optimization pass,
NOT iteratively inside the optimizer loop — it would corrupt
gradient/energy signals.
"""

from __future__ import annotations

import re
import math
from typing import Optional

from models.board_model import BoardModel, Component, BoardOutline
from engine.constraint_evaluator import evaluate_constraint_penalties, _build_decoupling_map


def legalize(
    model: BoardModel,
    grid_mm: float = 0.1,
    max_iterations: int = 300,
    push_strength: float = 1,
    verbose: bool = False,
    interior_bbox: tuple[float, float, float, float] | None = None,
) -> BoardModel:
    """Full legalization pipeline.

    Args:
        model: Board model with optimized but potentially illegal positions
        grid_mm: Grid size in mm (0.1mm or 0.05mm typical for KiCad)
        max_iterations: Max iterations for overlap resolution (300 for dense designs)
        push_strength: How far to push overlapping components (fraction of overlap, 0.8 is aggressive)
        verbose: Print progress information
        interior_bbox: Optional (x_min, y_min, x_max, y_max) to clamp interior
            components inside, keeping them away from edge connectors.

    Returns:
        BoardModel with legalized component positions
    """
    if verbose:
        overlaps_before = _count_overlaps(model)
        oob_before = _count_oob(model)
        print(f"Legalization input: {overlaps_before} overlaps, {oob_before} out-of-bounds")

    # Step 1: Snap to grid (overlap-aware)
    _snap_to_grid(model, grid_mm)

    # Step 2: Enforce board boundary
    _enforce_boundary(model, interior_bbox)

    # Step 3: Resolve overlaps (smart push-apart + greedy fallback)
    _resolve_overlaps(model, max_iterations, push_strength, grid_mm, verbose, interior_bbox)

    # Step 4: Greedy cleanup for any remaining overlaps
    remaining = _count_overlaps(model)
    if remaining > 0:
        _greedy_resolve(model, grid_mm, verbose, interior_bbox)

    # Step 5: Final grid snap and boundary check
    _snap_to_grid(model, grid_mm)
    _enforce_boundary(model, interior_bbox)

    # Step 6: Final greedy cleanup (grid snap may have created overlaps)
    remaining = _count_overlaps(model)
    if remaining > 0:
        _greedy_resolve(model, grid_mm, verbose, interior_bbox)

    # Step 7: Final legality pass. The last greedy cleanup can move parts
    # back out of bounds, so enforce boundaries again before returning.
    _enforce_boundary(model, interior_bbox)
    remaining = _count_overlaps(model)
    if remaining > 0:
        _greedy_resolve(model, grid_mm, verbose, interior_bbox)
        _enforce_boundary(model, interior_bbox)

    # Step 8: Constraint-preserving nudge — push decoupling caps back
    # toward their ICs without creating new overlaps.  The overlap
    # resolution steps above can push caps far from their ICs, which
    # defeats the decoupling_proximity constraint that the optimizer
    # worked hard to satisfy.
    rules = getattr(model, 'active_rules', None) or []
    if rules:
        _nudge_caps_to_ics(model, rules, interior_bbox, verbose)

    if verbose:
        overlaps_after = _count_overlaps(model)
        oob_after = _count_oob(model)
        print(f"Legalization output: {overlaps_after} overlaps, {oob_after} out-of-bounds")

    return model


def _snap_to_grid(model: BoardModel, grid_mm: float) -> None:
    """Round all component positions to the nearest grid point.

    v11: Overlap-aware grid snapping. After snapping each component,
    check if it now overlaps with any previously-snapped component.
    If so, try adjacent grid points to find a non-overlapping snap.
    """
    components = list(model.components)
    snapped_positions = []  # list of (comp, old_x, old_y) for already-snapped

    for comp in components:
        # Skip fixed components AND edge connectors
        if comp.is_fixed or comp.is_edge_connector:
            continue

        old_x, old_y = comp.x, comp.y
        new_x = round(old_x / grid_mm) * grid_mm
        new_y = round(old_y / grid_mm) * grid_mm
        comp.rotation = round(comp.rotation / 90.0) * 90.0

        # Check if snapping created overlaps with already-snapped components
        comp.x = new_x
        comp.y = new_y

        has_overlap = any(
            comp.overlaps(other)
            for other, _, _ in snapped_positions
        )

        if has_overlap:
            # Try adjacent grid points
            best_x, best_y = old_x, old_y  # fallback: keep original
            best_overlaps = 999

            for dx in [-grid_mm, 0, grid_mm]:
                for dy in [-grid_mm, 0, grid_mm]:
                    if dx == 0 and dy == 0:
                        continue
                    trial_x = new_x + dx
                    trial_y = new_y + dy
                    comp.x = trial_x
                    comp.y = trial_y
                    overlap_count = sum(
                        1 for other, _, _ in snapped_positions
                        if comp.overlaps(other)
                    )
                    if overlap_count < best_overlaps:
                        best_overlaps = overlap_count
                        best_x, best_y = trial_x, trial_y
                        if overlap_count == 0:
                            break
                if best_overlaps == 0:
                    break

            if best_overlaps > 0:
                # No adjacent grid point avoids overlap — keep original position
                comp.x = old_x
                comp.y = old_y
            else:
                comp.x = best_x
                comp.y = best_y

        snapped_positions.append((comp, old_x, old_y))


def _enforce_boundary(
    model: BoardModel,
    interior_bbox: tuple[float, float, float, float] | None = None,
) -> None:
    """Clamp component positions so their bounding boxes stay within bounds.

    v11: Overlap-aware boundary enforcement. When clamping would create
    a new overlap, try sliding along the boundary to find a non-overlapping
    position. If no such position exists, still clamp (out-of-bounds is
    worse than overlapping).
    """
    board = model.board
    components = list(model.components)

    for comp in model.components:
        if comp.is_fixed or comp.is_edge_connector:
            continue

        old_x, old_y = comp.x, comp.y
        _enforce_boundary_single(comp, interior_bbox, board)

        # Check if clamping created new overlaps
        if comp.x != old_x or comp.y != old_y:
            overlap_count = sum(1 for other in components
                                if other is not comp and comp.overlaps(other))
            if overlap_count > 0:
                # Try sliding along boundary to reduce overlaps
                best_x, best_y = comp.x, comp.y
                best_overlaps = overlap_count

                # Try small adjustments along the boundary
                for delta in [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0]:
                    for dx_dir, dy_dir in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        trial_x = comp.x + dx_dir * delta
                        trial_y = comp.y + dy_dir * delta

                        # Keep within bounds
                        half_w = comp.effective_width / 2.0
                        half_h = comp.effective_height / 2.0
                        if interior_bbox:
                            x_min = interior_bbox[0] + half_w
                            x_max = interior_bbox[2] - half_w
                            y_min = interior_bbox[1] + half_h
                            y_max = interior_bbox[3] - half_h
                        else:
                            x_min = board.x_min + half_w
                            x_max = board.x_max - half_w
                            y_min = board.y_min + half_h
                            y_max = board.y_max - half_h

                        trial_x = max(x_min, min(trial_x, x_max))
                        trial_y = max(y_min, min(trial_y, y_max))

                        comp.x = trial_x
                        comp.y = trial_y
                        trial_overlaps = sum(1 for other in components
                                             if other is not comp and comp.overlaps(other))
                        if trial_overlaps < best_overlaps:
                            best_overlaps = trial_overlaps
                            best_x, best_y = trial_x, trial_y
                            if trial_overlaps == 0:
                                break
                    if best_overlaps == 0:
                        break

                comp.x = best_x
                comp.y = best_y


def _enforce_boundary_single(
    comp: Component,
    interior_bbox: tuple[float, float, float, float] | None,
    board: BoardOutline,
) -> None:
    """Clamp a single component to board/interior bounds."""
    if comp.is_fixed or comp.is_edge_connector:
        return
    half_w = comp.effective_width / 2.0
    half_h = comp.effective_height / 2.0

    if interior_bbox:
        x_min = interior_bbox[0] + half_w
        x_max = interior_bbox[2] - half_w
        y_min = interior_bbox[1] + half_h
        y_max = interior_bbox[3] - half_h
    else:
        x_min = board.x_min + half_w
        x_max = board.x_max - half_w
        y_min = board.y_min + half_h
        y_max = board.y_max - half_h

    comp.x = max(x_min, min(comp.x, x_max))
    comp.y = max(y_min, min(comp.y, y_max))


def _count_overlaps_involving(
    comp: Component,
    components: list[Component],
) -> int:
    """Count overlaps involving comp with any other component."""
    count = 0
    for other in components:
        if other is comp:
            continue
        if comp.overlaps(other):
            count += 1
    return count


def _legalizer_score(
    model: BoardModel,
    moved: list[Component],
    original_positions: dict[int, tuple[float, float]],
) -> tuple[float, int, int]:
    """Score legalization candidates — lower is better.

    Keep overlap count dominant, use overlap area as a tiebreaker.

    NOTE: HPWL is intentionally excluded from this scoring function.
    The legalizer's primary job is resolving overlaps and enforcing
    boundaries — HPWL optimization is handled by the annealer/greedy
    before legalization. Including total_hpwl() here made the legalizer
    extremely slow because it's called inside tight inner loops
    (nudge search: 8 dirs x 13 distances = 104 calls per component;
     grid sweep: potentially hundreds of calls per component), and
    each total_hpwl() call does a full O(NxP) recomputation of all
    nets. With HPWL weight 5.0 vs overlap weight 100000.0 (a 20000:1
    ratio), HPWL had virtually zero influence on legalizer decisions
    but dominated runtime.
    """
    overlaps = _count_overlaps(model)
    oob = _count_oob(model)
    overlap_area = _compute_total_overlap_area(model)
    rules = getattr(model, 'active_rules', None) or []
    if rules:
        constraint_total, _ = evaluate_constraint_penalties(model, rules)
    else:
        constraint_total = 0.0
    displacement = 0.0
    for comp in moved:
        ox, oy = original_positions.get(id(comp), (comp.x, comp.y))
        displacement += abs(comp.x - ox) + abs(comp.y - oy)
    score = (100000.0 * overlaps + 25000.0 * oob + 50.0 * overlap_area
             + 10.0 * constraint_total + displacement)
    return score, overlaps, oob


def _count_pair_overlaps_involving(
    c1: Component,
    c2: Component,
    components: list[Component],
) -> int:
    """Count overlaps involving c1 or c2 with any other component."""
    count = 0
    for other in components:
        if other is c1 or other is c2:
            continue
        if c1.overlaps(other):
            count += 1
        if c2.overlaps(other):
            count += 1
    # Also count c1-c2 overlap
    if c1.overlaps(c2):
        count += 1
    return count


def _resolve_overlaps(
    model: BoardModel,
    max_iterations: int,
    push_strength: float,
    grid_mm: float,
    verbose: bool,
    interior_bbox: tuple[float, float, float, float] | None = None,
) -> None:
    """Iteratively resolve component overlaps by pushing components apart.

    v11: Smart push-apart that avoids creating new overlaps.
    - Sorts overlaps by area (smallest first — easier to resolve)
    - Only accepts pushes that reduce local overlap count
    - Tries reduced strength if full push creates new overlaps
    - Falls back to greedy resolution for stubborn overlaps
    """
    board = model.board
    prev_overlap_count = float("inf")
    stall_iterations = 0
    adaptive_strength = push_strength

    for iteration in range(max_iterations):
        # Collect all overlapping pairs with their overlap areas
        overlap_pairs = []
        components = list(model.components)
        components.sort(key=lambda c: (c.x, c.y))

        for i, c1 in enumerate(components):
            for c2 in components[i + 1:]:
                if not c1.overlaps(c2):
                    continue

                # Cannot resolve if both are fixed or both are edge connectors
                c1_fixed = c1.is_fixed or c1.is_edge_connector
                c2_fixed = c2.is_fixed or c2.is_edge_connector
                if c1_fixed and c2_fixed:
                    continue

                area = c1.overlap_area(c2)
                overlap_pairs.append((area, c1, c2, c1_fixed, c2_fixed))

        overlap_count = len(overlap_pairs)

        if overlap_count == 0:
            if verbose:
                print(f"  Overlap resolution converged in {iteration + 1} iterations")
            break

        # Sort by overlap area (smallest first — easier to resolve, less cascade)
        overlap_pairs.sort(key=lambda x: x[0])

        resolved_this_pass = 0
        for area, c1, c2, c1_fixed, c2_fixed in overlap_pairs:
            # Re-check if they still overlap (previous resolves may have fixed this)
            if not c1.overlaps(c2):
                continue

            # Save positions before push
            old_x1, old_y1 = c1.x, c1.y
            old_x2, old_y2 = c2.x, c2.y

            # Count overlaps involving c1 and c2 before push
            local_before = _count_pair_overlaps_involving(c1, c2, components)
            moved = [c for c in (c1, c2) if not c.is_fixed and not c.is_edge_connector]
            original_positions = {id(c1): (old_x1, old_y1), id(c2): (old_x2, old_y2)}
            base_score, _, _ = _legalizer_score(model, moved, original_positions)

            # Try push-apart with full strength
            if c1_fixed:
                _push_apart_one(c2, c1, adaptive_strength, grid_mm, board=board)
            elif c2_fixed:
                _push_apart_one(c1, c2, adaptive_strength, grid_mm, board=board)
            else:
                _push_apart(c1, c2, adaptive_strength, grid_mm)

            # Apply boundary enforcement for pushed components only
            _enforce_boundary_single(c1, interior_bbox, board)
            _enforce_boundary_single(c2, interior_bbox, board)

            # Count overlaps involving c1 and c2 after push
            local_after = _count_pair_overlaps_involving(c1, c2, components)
            trial_score, _, _ = _legalizer_score(model, moved, original_positions)

            if local_after <= local_before and trial_score <= base_score:
                # Push improved or maintained the situation — accept
                resolved_this_pass += 1
            else:
                # Push made things worse — try with reduced strength
                c1.x, c1.y = old_x1, old_y1
                c2.x, c2.y = old_x2, old_y2

                reduced_strength = adaptive_strength * 0.5
                if c1_fixed:
                    _push_apart_one(c2, c1, reduced_strength, grid_mm, board=board)
                elif c2_fixed:
                    _push_apart_one(c1, c2, reduced_strength, grid_mm, board=board)
                else:
                    _push_apart(c1, c2, reduced_strength, grid_mm)

                _enforce_boundary_single(c1, interior_bbox, board)
                _enforce_boundary_single(c2, interior_bbox, board)

                local_after_reduced = _count_pair_overlaps_involving(c1, c2, components)
                trial_score_reduced, _, _ = _legalizer_score(model, moved, original_positions)

                if local_after_reduced <= local_before and trial_score_reduced <= base_score:
                    # Reduced push worked
                    resolved_this_pass += 1
                else:
                    # Still worse — undo completely and skip
                    c1.x, c1.y = old_x1, old_y1
                    c2.x, c2.y = old_x2, old_y2
                    continue

        # Global boundary enforcement
        _enforce_boundary(model, interior_bbox)

        # Count actual overlaps
        actual_overlap_count = _count_overlaps(model)

        # Detect stalling and increase push strength
        if actual_overlap_count >= prev_overlap_count:
            stall_iterations += 1
            if stall_iterations >= 8:
                adaptive_strength = min(2.0, adaptive_strength * 1.3)
                stall_iterations = 0
        else:
            stall_iterations = 0

        prev_overlap_count = actual_overlap_count

        if actual_overlap_count == 0:
            if verbose:
                print(f"  Overlap resolution converged in {iteration + 1} iterations")
            break

        # If no progress after many iterations, break early — greedy will handle it
        if stall_iterations >= 20:
            if verbose:
                print(f"  Push-apart stalled at iteration {iteration + 1} "
                      f"({actual_overlap_count} overlaps remaining, switching to greedy)")
            break
    else:
        if verbose:
            print(f"  Overlap resolution: max iterations ({max_iterations}) reached, "
                  f"{_count_overlaps(model)} overlaps remaining")


def _greedy_resolve(
    model: BoardModel,
    grid_mm: float,
    verbose: bool,
    interior_bbox: tuple[float, float, float, float] | None = None,
) -> None:
    """Greedy overlap resolution: move overlapping components to nearest free positions.

    This is the fallback when push-apart can't resolve all overlaps.
    For each overlapping component, search nearby positions for a spot
    that minimizes overlaps. Includes a grid-sweep search for finding
    free positions on dense boards.

    Strategy:
    1. Find the component with the most overlaps
    2. Try nudging it in all directions at various distances
    3. If nudging fails, try sweeping a grid of positions across the board
    4. Accept the position that minimizes its overlap count
    5. Repeat until no overlaps remain or no improvement possible
    """
    board = model.board
    components = list(model.components)
    movable = [c for c in components if not c.is_fixed and not c.is_edge_connector]
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1),
                  (1, 1), (-1, 1), (1, -1), (-1, -1)]
    nudge_dists = [grid_mm, grid_mm * 2, grid_mm * 5, grid_mm * 10,
                   0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0]

    # Compute board bounds for grid sweep
    if interior_bbox:
        bx_min = interior_bbox[0]
        bx_max = interior_bbox[2]
        by_min = interior_bbox[1]
        by_max = interior_bbox[3]
    else:
        bx_min = board.x_min
        bx_max = board.x_max
        by_min = board.y_min
        by_max = board.y_max

    prev_total = _count_overlaps(model)
    for outer in range(50):  # More passes for dense boards
        total_overlaps = _count_overlaps(model)
        if total_overlaps == 0:
            break

        # Find overlapping components and their overlap counts
        overlap_counts: dict[int, int] = {}
        for idx, c in enumerate(movable):
            count = _count_overlaps_involving(c, components)
            if count > 0:
                overlap_counts[id(c)] = count

        if not overlap_counts:
            break

        # Sort by overlap count (most overlapping first)
        movable_with_overlaps = [c for c in movable if id(c) in overlap_counts]
        movable_with_overlaps.sort(key=lambda c: overlap_counts[id(c)], reverse=True)

        improved = False
        for comp in movable_with_overlaps:
            old_x, old_y = comp.x, comp.y
            old_overlaps = overlap_counts.get(id(comp), 0)
            if old_overlaps == 0:
                continue

            best_x, best_y = old_x, old_y
            best_overlaps = old_overlaps
            base_score, _, _ = _legalizer_score(model, [comp], {id(comp): (old_x, old_y)})
            best_score = base_score

            # Strategy 1: Try nudging in all directions
            for dx_dir, dy_dir in directions:
                for dist in nudge_dists:
                    comp.x = old_x + dx_dir * dist
                    comp.y = old_y + dy_dir * dist
                    _enforce_boundary_single(comp, interior_bbox, board)

                    new_overlaps = _count_overlaps_involving(comp, components)
                    trial_score, _, _ = _legalizer_score(model, [comp], {id(comp): (old_x, old_y)})
                    if new_overlaps < best_overlaps or (new_overlaps == best_overlaps and trial_score < best_score):
                        best_overlaps = new_overlaps
                        best_score = trial_score
                        best_x, best_y = comp.x, comp.y
                        if new_overlaps == 0:
                            break
                if best_overlaps == 0:
                    break

            # Strategy 2: If nudging didn't resolve, try grid sweep
            # Search a coarse grid across the board for a free position
            if best_overlaps > 0:
                comp.x, comp.y = old_x, old_y  # reset
                half_w = comp.effective_width / 2.0
                half_h = comp.effective_height / 2.0
                grid_step = max(comp.effective_width * 0.5, comp.effective_height * 0.5, 0.5)

                gx = bx_min + half_w
                while gx <= bx_max - half_w:
                    gy = by_min + half_h
                    while gy <= by_max - half_h:
                        comp.x = gx
                        comp.y = gy
                        new_overlaps = _count_overlaps_involving(comp, components)
                        trial_score, _, _ = _legalizer_score(model, [comp], {id(comp): (old_x, old_y)})
                        if new_overlaps < best_overlaps or (new_overlaps == best_overlaps and trial_score < best_score):
                            best_overlaps = new_overlaps
                            best_score = trial_score
                            best_x, best_y = comp.x, comp.y
                            if new_overlaps == 0:
                                break
                        gy += grid_step
                    if best_overlaps == 0:
                        break
                    gx += grid_step

            comp.x, comp.y = best_x, best_y
            if best_overlaps < old_overlaps or best_score < base_score:
                improved = True
            else:
                comp.x, comp.y = old_x, old_y

        current_total = _count_overlaps(model)
        if current_total >= prev_total and not improved:
            break
        prev_total = current_total

    if verbose:
        final = _count_overlaps(model)
        if final > 0:
            print(f"  Greedy resolution: {final} overlaps remaining (board may be too dense)")


def _push_apart_one(
    movable: Component,
    fixed: Component,
    strength: float,
    grid_mm: float,
    board: BoardOutline | None = None,
) -> None:
    """Push the movable component away from a fixed component (edge connector).

    Args:
        movable: Component to move (interior component)
        fixed: Fixed component to move away from (edge connector on perimeter)
        strength: Multiplier for push distance
        grid_mm: Minimum movement (1.5x grid unit)
    """
    # Compute overlap on each axis
    ax1, ay1, ax2, ay2 = movable.bbox
    bx1, by1, bx2, by2 = fixed.bbox

    overlap_x = min(ax2, bx2) - max(ax1, bx1)
    overlap_y = min(ay2, by2) - max(ay1, by1)

    if overlap_x <= 0 or overlap_y <= 0:
        return  # No actual overlap

    # Ensure minimum push distance = 1.5x grid unit to guarantee progress
    push_mm = max(grid_mm * 1.5, 0.15)

    # Edge connectors live on the perimeter, so move interior parts toward the
    # board center instead of letting the generic minimum-overlap axis push them
    # sideways into another perimeter part.
    if board is not None and fixed.is_edge_connector:
        board_cx = (board.x_min + board.x_max) / 2.0
        board_cy = (board.y_min + board.y_max) / 2.0
        center_dx = 1 if board_cx >= fixed.x else -1
        center_dy = 1 if board_cy >= fixed.y else -1

        if abs(board_cx - fixed.x) >= abs(board_cy - fixed.y):
            push_x = max(overlap_x * strength, push_mm)
            movable.x += center_dx * push_x
        else:
            push_y = max(overlap_y * strength, push_mm)
            movable.y += center_dy * push_y
        return

    # Push along axis of minimum separation for minimal displacement
    if overlap_x <= overlap_y:
        push_x = max(overlap_x * strength, push_mm)
        push_y = 0
    else:
        push_x = 0
        push_y = max(overlap_y * strength, push_mm)

    # Determine push direction: move movable away from fixed
    dx = 1 if movable.x >= fixed.x else -1
    dy = 1 if movable.y >= fixed.y else -1

    movable.x += dx * push_x
    movable.y += dy * push_y


def _push_apart(c1: Component, c2: Component, strength: float, grid_mm: float) -> None:
    """Push two overlapping components apart along axis of minimum separation.

    The direction is chosen to minimize displacement (push along axis
    where overlap is smallest). For severe overlaps, pushes on both axes.

    Args:
        c1, c2: Overlapping components
        strength: Multiplier for push distance
        grid_mm: Minimum movement (1.5x grid unit)
    """
    # Compute overlap on each axis
    ax1, ay1, ax2, ay2 = c1.bbox
    bx1, by1, bx2, by2 = c2.bbox

    overlap_x = min(ax2, bx2) - max(ax1, bx1)
    overlap_y = min(ay2, by2) - max(ay1, by1)

    if overlap_x <= 0 or overlap_y <= 0:
        return  # No actual overlap

    # For severe overlaps (>70% of smaller dimension), push on both axes
    c1_min = min(c1.effective_width, c1.effective_height)
    c2_min = min(c2.effective_width, c2.effective_height)
    severity_threshold = 0.7 * min(c1_min, c2_min)

    push_both = (overlap_x > severity_threshold) or (overlap_y > severity_threshold)

    # Ensure minimum push distance = 1.5x grid unit to guarantee progress
    push_mm = max(grid_mm * 1.5, 0.15)

    # Calculate push distances
    if push_both:
        # Push along diagonal (both axes) for severe overlap
        push_x = max(overlap_x * strength * 0.5, push_mm)
        push_y = max(overlap_y * strength * 0.5, push_mm)
    elif overlap_x <= overlap_y:
        # Push along X axis (smaller overlap)
        push_x = max(overlap_x * strength, push_mm)
        push_y = 0
    else:
        # Push along Y axis (smaller overlap)
        push_x = 0
        push_y = max(overlap_y * strength, push_mm)

    # Determine push direction based on relative positions
    dx = 1 if c2.x >= c1.x else -1
    dy = 1 if c2.y >= c1.y else -1

    # Apply push
    if c1.is_fixed and not c2.is_fixed:
        c2.x += dx * push_x
        c2.y += dy * push_y
    elif c2.is_fixed and not c1.is_fixed:
        c1.x -= dx * push_x
        c1.y -= dy * push_y
    else:
        # Split push between both components
        c1.x -= dx * push_x / 2.0
        c1.y -= dy * push_y / 2.0
        c2.x += dx * push_x / 2.0
        c2.y += dy * push_y / 2.0


def _nudge_caps_to_ics(
    model: BoardModel,
    rules: list,
    interior_bbox: tuple[float, float, float, float] | None = None,
    verbose: bool = False,
) -> None:
    """Nudge decoupling caps toward their associated ICs after legalization.

    Legalization resolves overlaps by pushing components apart, which often
    moves decoupling caps far from their ICs.  This pass moves caps back
    toward their ICs in small steps, stopping before creating overlaps
    or boundary violations.

    The algorithm:
    1. Build the decoupling map (IC → caps).
    2. For each cap that's farther than max_distance_mm from its IC,
       move it toward the IC in small increments.
    3. Stop if an overlap or boundary violation would occur.
    4. Repeat for up to 5 passes (caps may need to "wait their turn"
       as other caps move).
    """
    board = model.board
    components = list(model.components)

    # Find the decoupling_proximity rule params
    max_dist = 5.0  # default
    for rule in rules:
        if rule.name == 'decoupling_proximity':
            max_dist = rule.params.get('max_distance_mm', 5.0)
            break
    else:
        return  # no decoupling rule — nothing to do

    decap_map = _build_decoupling_map(model)
    if not decap_map:
        return

    # Build movable set for overlap checks
    step_sizes = [2.0, 1.0, 0.5, 0.25, 0.1]

    for pass_num in range(5):
        any_moved = False

        for ic_ref, cap_refs in decap_map.items():
            ic = model.get_component(ic_ref)
            if not ic:
                continue

            for cap_ref in cap_refs:
                cap = model.get_component(cap_ref)
                if not cap:
                    continue
                if cap.is_fixed or cap.is_edge_connector:
                    continue

                dist = math.hypot(ic.x - cap.x, ic.y - cap.y)
                if dist <= max_dist:
                    continue  # already close enough

                # Direction from cap toward IC
                dx = ic.x - cap.x
                dy = ic.y - cap.y
                if dx == 0 and dy == 0:
                    continue
                length = math.hypot(dx, dy)
                dx /= length
                dy /= length

                # Try moving in decreasing step sizes
                for step in step_sizes:
                    new_x = cap.x + dx * step
                    new_y = cap.y + dy * step

                    # Clamp to bounds
                    half_w = cap.effective_width / 2.0
                    half_h = cap.effective_height / 2.0
                    if interior_bbox:
                        x_min = interior_bbox[0] + half_w
                        x_max = interior_bbox[2] - half_w
                        y_min = interior_bbox[1] + half_h
                        y_max = interior_bbox[3] - half_h
                    else:
                        x_min = board.x_min + half_w
                        x_max = board.x_max - half_w
                        y_min = board.y_min + half_h
                        y_max = board.y_max - half_h

                    new_x = max(x_min, min(new_x, x_max))
                    new_y = max(y_min, min(new_y, y_max))

                    # Check if this move creates overlaps
                    old_x, old_y = cap.x, cap.y
                    cap.x = new_x
                    cap.y = new_y

                    has_overlap = any(
                        cap.overlaps(other)
                        for other in components
                        if other is not cap and not other.is_fixed
                    )

                    if not has_overlap:
                        new_dist = math.hypot(ic.x - new_x, ic.y - new_y)
                        if new_dist < dist:
                            any_moved = True
                            break  # accept this step
                    else:
                        cap.x = old_x
                        cap.y = old_y

        if not any_moved:
            break

    if verbose:
        # Report remaining violations
        violations = 0
        for ic_ref, cap_refs in decap_map.items():
            ic = model.get_component(ic_ref)
            if not ic:
                continue
            for cap_ref in cap_refs:
                cap = model.get_component(cap_ref)
                if not cap:
                    continue
                dist = math.hypot(ic.x - cap.x, ic.y - cap.y)
                if dist > max_dist:
                    violations += 1
        if violations > 0:
            print(f"  Cap-IC nudge: {violations} caps still beyond {max_dist:.0f}mm threshold")
        else:
            print(f"  Cap-IC nudge: all caps within {max_dist:.0f}mm of their ICs")


def _count_overlaps(model: BoardModel) -> int:
    """Count overlapping component pairs."""
    count = 0
    for i, c1 in enumerate(model.components):
        for c2 in model.components[i + 1:]:
            if c1.overlaps(c2):
                count += 1
    return count


def _count_oob(model: BoardModel) -> int:
    """Count out-of-bounds components, excluding edge connectors."""
    count = 0
    board = model.board
    for comp in model.components:
        if comp.is_edge_connector:
            continue
        x1, y1, x2, y2 = comp.bbox
        if x1 < board.x_min or x2 > board.x_max or y1 < board.y_min or y2 > board.y_max:
            count += 1
    return count


def _compute_total_overlap_area(model: BoardModel) -> float:
    """Compute total overlap area across all overlapping component pairs.

    Used by _legalizer_score to prevent creating fewer but deeper overlaps.
    The old score (100000 * overlaps) optimized for count only, causing the
    legalizer to create 7 overlaps at 309mm² vs 30 at 130mm².
    """
    total = 0.0
    components = list(model.components)
    for i, c1 in enumerate(components):
        for c2 in components[i + 1:]:
            if c1.overlaps(c2):
                total += c1.overlap_area(c2)
    return total
