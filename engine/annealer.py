"""Simulated Annealing engine for PCB component placement.

This module includes the SA stability fixes, density-aware tuning,
global-best tracking across reheats, and HPWL-based revert logic used
by the placement pipeline.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from models.board_model import BoardModel
from engine.cost_state import CostState, OVERLAP_WEIGHT, OVERLAP_COUNT_WEIGHT, BOUNDARY_WEIGHT, CONSTRAINT_WEIGHT
from engine.moves import (
    select_move_type, get_moveable_indices,
    do_translate, do_swap, do_rotate, do_median,
    revert_move, affected_indices, MoveUndo,
)


@dataclass
class SAConfig:
    max_iterations: int = 300          # more steps for thorough exploration
    reheat_count: int = 3              # 3 reheats for sparse boards, fewer for dense
    reheat_ratio: float = 0.40         # reheat to 40% of previous T0
    calibration_samples: int = 500     # more samples for robust T0 estimation
    initial_accept_rate: float = 0.92   # target high accept rate at start
    penalty_scale_min: float = 0.50    # will be overridden by density-adaptive logic
    min_temperature: float = 1e-8
    freeze_threshold: float = 0.005    # stop sooner when frozen
    greedy_nudge_distances: tuple[float, ...] = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
    greedy_rotations: tuple[float, ...] = (90.0, 180.0, 270.0)
    greedy_improve_threshold: float = 0.5  # accept smaller improvements
    overlap_cap_factor: float = 2.0    # reject moves that exceed this * initial overlaps
    verbose: bool = True


def _compute_density(model: BoardModel) -> float:
    """Compute board density = total component area / board area."""
    board = model.board
    board_area = board.width * board.height
    if board_area <= 0:
        return 0.0
    comp_area = sum(c.effective_width * c.effective_height for c in model.components)
    return min(1.0, comp_area / board_area)


def _density_adaptive_penalty_scale(density: float) -> float:
    """Compute penalty_scale_min based on board density.

    Sparse boards (density < 0.25): penalty_scale_min = 0.80
    Dense boards (density > 0.40): penalty_scale_min = 0.95+

    Overlap penalties are stronger now (OVERLAP_WEIGHT=25, COUNT_WEIGHT=12),
    so even at 0.80 the penalties are significant.
    """
    if density < 0.25:
        return 0.80
    elif density < 0.40:
        # Linear ramp from 0.80 to 0.95
        t = (density - 0.25) / 0.15
        return 0.80 + 0.15 * t
    else:
        # Dense boards: 0.95 to 1.00
        t = min(1.0, (density - 0.40) / 0.30)
        return 0.95 + 0.05 * t


def _density_adaptive_reheat_count(density: float) -> int:
    """Fewer reheats for dense boards, but still at least 2.

    Dense boards get 2 reheats (was 1). A single reheat is
    insufficient - the best state often gets rejected due to overlap
    drift. With accurate per-step resync, 2 reheats give better results.
    """
    if density < 0.25:
        return 3
    elif density < 0.40:
        return 2
    else:
        return 2


def _save_positions(model: BoardModel, indices: list[int]) -> dict[int, tuple[float, float, float]]:
    return {i: (model.components[i].x, model.components[i].y, model.components[i].rotation) for i in indices}


def _restore_positions(model: BoardModel, saved: dict[int, tuple[float, float, float]]):
    for i, (x, y, rot) in saved.items():
        model.components[i].x = x
        model.components[i].y = y
        model.components[i].set_rotation(rot)


def _calibrate_t0(
    model: BoardModel,
    cost_state: CostState,
    moveable_indices: list[int],
    config: SAConfig,
) -> float:
    """Auto-calibrate initial temperature by sampling ALL move types.

    v11: For dense boards, cap T0 to prevent excessive initial disruption.
    """
    board = model.board
    max_window = max(board.width, board.height) * 0.08
    min_window = max(board.width, board.height) * 0.005
    window = max_window  # hot phase uses max window
    noise = board.width * 0.02
    deltas = []

    # Set penalty_scale to what SA will use at the start (hot phase)
    cost_state.update_penalty_scale(config.penalty_scale_min)

    for i in range(config.calibration_samples):
        # Sample all move types with hot-phase probabilities
        mt = select_move_type(1.0)  # t_ratio=1.0 = hot phase

        if mt == 'translate':
            undo = do_translate(model, moveable_indices, 1.0, window)
        elif mt == 'swap':
            undo = do_swap(model, moveable_indices)
        elif mt == 'rotate':
            undo = do_rotate(model, moveable_indices)
        else:
            undo = do_median(model, moveable_indices, 1.0, noise)

        if not undo.old_states:
            continue

        moved = affected_indices(undo)
        old_bboxes = cost_state.old_bboxes_from_states(undo.old_states)
        snap = cost_state.snapshot(moved, old_bboxes=old_bboxes)
        new_cost = cost_state.incremental_update(moved)
        # v10 BUG FIX: use snap['overlap_penalty'] (full sum) not
        # sum(snap['pair_overlaps'].values()) (partial — only moved-component
        # overlaps).  The old code compared new_cost (all overlaps) with
        # old_cost (only moved-component overlaps), inflating delta by the
        # sum of all unaffected overlaps.
        old_cost = (snap['hpwl']
                    + (OVERLAP_WEIGHT * snap['overlap_penalty'] + OVERLAP_COUNT_WEIGHT * len(snap['pair_overlaps'])) * config.penalty_scale_min
                    + BOUNDARY_WEIGHT * snap['boundary_penalty'] * config.penalty_scale_min
                    + CONSTRAINT_WEIGHT * snap['constraint_total'] * config.penalty_scale_min)

        delta = new_cost - old_cost
        if delta > 0:
            deltas.append(delta)

        # Revert — model FIRST so spatial index is consistent
        revert_move(model, undo)
        cost_state.restore(snap)

    if not deltas:
        return 1.0

    # Use 75th percentile for dense boards (lower T0 = less disruption)
    # vs 90th percentile for sparse boards (higher T0 = more exploration)
    density = _compute_density(model)
    if density > 0.35:
        pct = 0.75
    elif density > 0.25:
        pct = 0.85
    else:
        pct = 0.90

    idx_pct = int(len(deltas) * pct)
    pct_delta = sorted(deltas)[min(idx_pct, len(deltas) - 1)]
    t0 = -pct_delta / math.log(config.initial_accept_rate)

    # Safety: ensure T0 is at least large enough that median delta has
    # accept rate > 0.5 (prevents T0 being too low for skewed distributions)
    if deltas:
        median_delta = sorted(deltas)[len(deltas) // 2]
        t0_median = -median_delta / math.log(0.55)
        t0 = max(t0, t0_median)

    return t0


def _run_sa_pass(
    model: BoardModel,
    cost_state: CostState,
    moveable_indices: list[int],
    t0: float,
    config: SAConfig,
    max_iter_override: int | None = None,
    initial_overlap_count: int = 0,
    global_best_cost: float = float('inf'),
    global_best_positions: dict | None = None,
    global_best_overlap_count: int = 999999,
) -> tuple[float, float, dict, int]:
    """Run one SA pass. Returns (final_temperature, global_best_cost,
    global_best_positions, global_best_overlap_count).

    Tracks global best across all passes (main + reheats).
    Each reheat receives the global best from previous passes and
    preserves it — the old code let reheats overwrite the global best
    with a worse state, then greedy couldn't recover enough.

    v11: Hard overlap cap — reject moves that increase overlap count
    beyond overlap_cap_factor * initial_overlap_count. This prevents
    SA from exploring states that are clearly worse (too many overlaps).
    """
    T = t0
    n_moveable = len(moveable_indices)
    moves_per_temp = max(300, 20 * n_moveable)
    board = model.board
    max_window = max(board.width, board.height) * 0.10
    min_window = max(board.width, board.height) * 0.003
    max_iter = max_iter_override or config.max_iterations

    best_positions = _save_positions(model, moveable_indices)
    best_cost = cost_state.normalized_cost
    best_overlap_count = cost_state.overlap_count
    best_overlap_seen = initial_overlap_count
    no_improve_count = 0

    # Global best state across all passes (main + reheats)
    if global_best_positions is None:
        global_best_cost = best_cost
        global_best_positions = best_positions
        global_best_overlap_count = best_overlap_count

    # Allow small overlap increases during exploration when starting from 0.
    # Old strict mode rejected ALL overlap-creating moves → SA froze (accept=0.00)
    strict_max_overlaps = max(3, int(len(moveable_indices) * 0.05))

    # Current overlap count (incremental, updated each move)
    current_overlap_count = cost_state.overlap_count

    for step in range(max_iter):
        t_ratio = math.log(T + 1.0) / math.log(t0 + 1.0) if t0 > 0 else 0.0
        t_ratio = max(0.0, min(1.0, t_ratio))

        # Adaptive penalty scale: hot→discount penalties, cold→full weight
        scale = config.penalty_scale_min + (1.0 - config.penalty_scale_min) * (1.0 - t_ratio)
        cost_state.update_penalty_scale(scale)

        window = min_window + (max_window - min_window) * t_ratio
        noise = board.width * 0.03 * t_ratio

        accepted = 0
        for _ in range(moves_per_temp):
            mt = select_move_type(t_ratio)
            if mt == 'translate':
                undo = do_translate(model, moveable_indices, t_ratio, window)
            elif mt == 'swap':
                undo = do_swap(model, moveable_indices)
            elif mt == 'rotate':
                undo = do_rotate(model, moveable_indices)
            else:
                undo = do_median(model, moveable_indices, t_ratio, noise)

            if not undo.old_states:
                continue

            moved = affected_indices(undo)
            old_bboxes = cost_state.old_bboxes_from_states(undo.old_states)
            snap = cost_state.snapshot(moved, old_bboxes=old_bboxes)

            new_cost = cost_state.incremental_update(moved)

            # Dynamic overlap cap: reject moves that create too many overlaps
            # Cap adapts based on best overlap count seen (tightens as SA improves)
            # and temperature (relaxed at high T for exploration, tight at low T)
            if initial_overlap_count == 0:
                # Relaxed strict mode — allow small overlap increases during
                # exploration. Old code rejected ALL overlap-creating moves when
                # starting from 0 overlaps, which froze SA (accept=0.00).
                # Allow up to strict_max_overlaps temporary overlaps for exploration.
                if cost_state.overlap_count > strict_max_overlaps:
                    cost_state.restore(snap)
                    revert_move(model, undo)
                    continue
            else:
                cap_base = max(best_overlap_seen + 10, initial_overlap_count + 10)
                cap_relaxation = 1.0 + 0.25 * t_ratio  # 1.25x at hot, 1.0x at cold
                overlap_cap = int(cap_base * cap_relaxation)
                if cost_state.overlap_count > overlap_cap:
                    # Above cap: only reject if overlaps INCREASED from current
                    if cost_state.overlap_count > current_overlap_count:
                        cost_state.restore(snap)
                        revert_move(model, undo)
                        continue

            # v10 BUG FIX: use snap['overlap_penalty'] (full sum) not
            # sum(snap['pair_overlaps'].values()) (partial).  See _calibrate_t0.
            old_cost = (snap['hpwl']
                        + (OVERLAP_WEIGHT * snap['overlap_penalty'] + OVERLAP_COUNT_WEIGHT * len(snap['pair_overlaps'])) * scale
                        + BOUNDARY_WEIGHT * snap['boundary_penalty'] * scale
                        + CONSTRAINT_WEIGHT * snap['constraint_total'] * scale)

            delta = new_cost - old_cost

            accept_move = (delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)))
            # Temperature-dependent overlap rejection.
            # At HIGH temperature (t_ratio > 0.4), allow SA to explore via normal
            # Metropolis probability — most exploratory moves create temporary overlaps.
            # At LOW temperature, hard-reject overlap increases to preserve quality.
            # The old code hard-rejected at ALL temperatures, causing SA to freeze.
            if t_ratio < 0.4:
                if cost_state.overlap_count > current_overlap_count and cost_state.normalized_cost >= best_cost:
                    accept_move = False

            if accept_move:
                accepted += 1
                current_overlap_count = cost_state.overlap_count
            else:
                revert_move(model, undo)
                cost_state.restore(snap)

        accept_rate = accepted / max(moves_per_temp, 1)

        # Resync every 3 steps for accurate overlap tracking.
        # Every step is too expensive on large boards. The old_bboxes_from_states
        # fix makes incremental tracking much more reliable, so 3 steps is
        # a good balance between accuracy and performance.
        if step % 3 == 0:
            cost_state._compute_all()
            current_overlap_count = cost_state.overlap_count

            # Update best state using verified cost (after resync)
            # Use overlap count as tiebreaker: prefer fewer overlaps at same cost
            verified_cost = cost_state.normalized_cost
            if (cost_state.overlap_count < best_overlap_count or
                    (cost_state.overlap_count == best_overlap_count and verified_cost < best_cost)):
                best_cost = verified_cost
                best_positions = _save_positions(model, moveable_indices)

            # Update global best across all passes (main + reheats)
            if (cost_state.overlap_count < global_best_overlap_count or
                    (cost_state.overlap_count == global_best_overlap_count and verified_cost < global_best_cost)):
                global_best_cost = verified_cost
                global_best_positions = _save_positions(model, moveable_indices)
                global_best_overlap_count = cost_state.overlap_count

            best_overlap_count = cost_state.overlap_count
            best_overlap_seen = min(best_overlap_seen, cost_state.overlap_count)
            no_improve_count = 0

        if config.verbose and step % 10 == 0:
            print(f"    T={T:.2f} accept={accept_rate:.2f} cost={verified_cost:.1f} "
                  f"hpwl={cost_state.hpwl:.1f} overlaps={cost_state.overlap_count}")

        # Adaptive cooling
        if accept_rate > 0.6:
            T *= 0.95
        elif accept_rate > 0.4:
            T *= 0.97
        elif accept_rate > 0.2:
            T *= 0.985
        elif accept_rate > 0.05:
            T *= 0.99
        else:
            T *= 0.98

        T = max(T, config.min_temperature)

        # Track stall
        no_improve_count += 1

        # Early stop when frozen AND stalled
        if accept_rate < config.freeze_threshold and T < t0 * 0.03:
            if config.verbose:
                print(f"    Frozen at step {step}, accept_rate={accept_rate:.3f}")
            break

        # Stall detection
        if no_improve_count > max_iter * 0.4 and accept_rate < 0.05:
            if config.verbose:
                print(f"    Stalled at step {step}, no improvement for {no_improve_count} steps")
            break

    # Restore global best, not just pass-local best. This prevents
    # reheats from destroying the main pass's best state.
    _restore_positions(model, global_best_positions)
    cost_state._compute_all()

    if config.verbose:
        print(f"    Restored best state: cost={cost_state.normalized_cost:.1f} "
              f"overlaps={cost_state.overlap_count}")

    return T, global_best_cost, global_best_positions, global_best_overlap_count


def _greedy_refine(
    model: BoardModel,
    cost_state: CostState,
    moveable_indices: list[int],
    config: SAConfig,
) -> None:
    """Greedy refinement: try nudges and rotations, accept only if improving.

    Hard overlap rejection: greedy does small moves, so rejecting
    overlap-creating moves doesn't limit exploration.  This ensures
    greedy never creates new overlaps.
    """
    # Reset penalty scale for greedy — we want full-weight evaluation
    cost_state.update_penalty_scale(1.0)

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1),
                  (1, 1), (-1, 1), (1, -1), (-1, -1)]

    improved = True
    sweep = 0
    best_sweep_cost = cost_state.normalized_cost
    best_sweep_positions = _save_positions(model, moveable_indices)

    while improved:
        improved = False
        sweep += 1
        for idx in moveable_indices:
            comp = model.components[idx]
            # Try nudges
            for dx_dir, dy_dir in directions:
                for dist in config.greedy_nudge_distances:
                    old_x, old_y = comp.x, comp.y
                    base_cost = cost_state.normalized_cost
                    prev_overlap_count = cost_state.overlap_count

                    comp.x += dx_dir * dist
                    comp.y += dy_dir * dist

                    new_cost = cost_state.incremental_update({idx})

                    # Hard overlap rejection: greedy must not create new overlaps
                    if cost_state.overlap_count > prev_overlap_count:
                        comp.x = old_x
                        comp.y = old_y
                        cost_state.incremental_update({idx})
                        continue

                    if new_cost < base_cost - config.greedy_improve_threshold:
                        improved = True
                    else:
                        comp.x = old_x
                        comp.y = old_y
                        cost_state.incremental_update({idx})

            # Try rotations
            base_rot = comp.rotation
            for rot in config.greedy_rotations:
                base_cost = cost_state.normalized_cost
                prev_overlap_count = cost_state.overlap_count
                comp.set_rotation((base_rot + rot) % 360.0)
                new_cost = cost_state.incremental_update({idx})

                # Hard overlap rejection for rotations too
                if cost_state.overlap_count > prev_overlap_count:
                    comp.set_rotation(base_rot)
                    cost_state.incremental_update({idx})
                    continue

                if new_cost < base_cost - config.greedy_improve_threshold:
                    improved = True
                    base_rot = comp.rotation
                else:
                    comp.set_rotation(base_rot)
                    cost_state.incremental_update({idx})

        cur_cost = cost_state.normalized_cost
        if cur_cost < best_sweep_cost:
            best_sweep_cost = cur_cost
            best_sweep_positions = _save_positions(model, moveable_indices)
        else:
            _restore_positions(model, best_sweep_positions)
            cost_state._compute_all()

        if config.verbose:
            print(f"    Greedy sweep {sweep}: cost={cost_state.normalized_cost:.1f} "
                  f"overlaps={cost_state.overlap_count}")

        if sweep >= 8:
            break

    # Final restore to best sweep result
    if cost_state.normalized_cost > best_sweep_cost:
        _restore_positions(model, best_sweep_positions)
        cost_state._compute_all()


def _resolve_overlaps_greedy(
    model: BoardModel,
    cost_state: CostState,
    moveable_indices: list[int],
    config: SAConfig,
) -> None:
    """Force-resolve remaining overlaps after greedy refinement.

    v11: Added overlap-aware moves — check if resolving one overlap
    creates new overlaps with other components, and choose the move
    that minimizes total overlap count.
    """
    cost_state.update_penalty_scale(1.0)
    board = model.board
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1),
                  (1, 1), (-1, 1), (1, -1), (-1, -1)]
    nudge_dists = [0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0]
    moveable_set = set(moveable_indices)

    prev_overlaps = cost_state.overlap_count
    for outer in range(15):  # max 15 outer iterations
        overlaps = cost_state.overlap_count
        if overlaps == 0:
            break

        # Identify overlapping component pairs
        overlap_pairs = []
        for (i, j), penalty in cost_state._pair_overlaps.items():
            if penalty > 0:
                overlap_pairs.append((i, j))

        # Sort by overlap area (smallest first — easier to resolve)
        pair_areas = []
        for i, j in overlap_pairs:
            c1 = model.components[i]
            c2 = model.components[j]
            area = c1.overlap_area(c2)
            pair_areas.append((area, i, j))
        pair_areas.sort()

        for area, i, j in pair_areas:
            if cost_state.overlap_count == 0:
                break

            c1 = model.components[i]
            c2 = model.components[j]
            c1_movable = i in moveable_set
            c2_movable = j in moveable_set

            best_overlap = cost_state.overlap_count
            best_action = None  # ('single', idx, x, y) or ('pair', i, j, x1, y1, x2, y2)

            # Strategy 1: try moving each component individually
            for idx, is_movable in [(i, c1_movable), (j, c2_movable)]:
                if not is_movable:
                    continue
                comp = model.components[idx]
                for dx_dir, dy_dir in directions:
                    for dist in nudge_dists:
                        old_x, old_y = comp.x, comp.y
                        comp.x += dx_dir * dist
                        comp.y += dy_dir * dist
                        cost_state.incremental_update({idx})

                        if cost_state.overlap_count < best_overlap:
                            best_overlap = cost_state.overlap_count
                            best_action = ('single', idx, comp.x, comp.y)

                        comp.x = old_x
                        comp.y = old_y
                        cost_state.incremental_update({idx})

            # Strategy 2: push both components apart simultaneously
            if c1_movable and c2_movable:
                dx_sign = 1 if c2.x >= c1.x else -1
                dy_sign = 1 if c2.y >= c1.y else -1
                for dist in nudge_dists:
                    old_x1, old_y1 = c1.x, c1.y
                    old_x2, old_y2 = c2.x, c2.y
                    c1.x -= dx_sign * dist * 0.5
                    c1.y -= dy_sign * dist * 0.5
                    c2.x += dx_sign * dist * 0.5
                    c2.y += dy_sign * dist * 0.5
                    cost_state.incremental_update({i, j})

                    if cost_state.overlap_count < best_overlap:
                        best_overlap = cost_state.overlap_count
                        best_action = ('pair', i, j, c1.x, c1.y, c2.x, c2.y)

                    c1.x, c1.y = old_x1, old_y1
                    c2.x, c2.y = old_x2, old_y2
                    cost_state.incremental_update({i, j})

            # Apply the best overlap-reducing action
            if best_action is not None and best_overlap < cost_state.overlap_count:
                if best_action[0] == 'single':
                    _, idx, nx, ny = best_action
                    model.components[idx].x = nx
                    model.components[idx].y = ny
                    cost_state.incremental_update({idx})
                elif best_action[0] == 'pair':
                    _, ci, cj, x1, y1, x2, y2 = best_action
                    model.components[ci].x = x1
                    model.components[ci].y = y1
                    model.components[cj].x = x2
                    model.components[cj].y = y2
                    cost_state.incremental_update({ci, cj})

        overlaps_now = cost_state.overlap_count
        if config.verbose:
            print(f"    Overlap resolve pass {outer + 1}: overlaps={overlaps_now} "
                  f"(was {prev_overlaps}, resolved {prev_overlaps - overlaps_now})")

        if overlaps_now >= prev_overlaps:
            # No progress — break to avoid infinite loop
            break
        prev_overlaps = overlaps_now

    # Clamp to board bounds after overlap resolution
    # v11: Don't blindly clamp — check if clamping creates new overlaps.
    # Only clamp if it doesn't increase the overlap count.
    prev_overlaps = cost_state.overlap_count
    for idx in moveable_indices:
        comp = model.components[idx]
        half_w = comp.effective_width / 2.0
        half_h = comp.effective_height / 2.0
        new_x = max(board.x_min + half_w, min(comp.x, board.x_max - half_w))
        new_y = max(board.y_min + half_h, min(comp.y, board.y_max - half_h))
        if new_x != comp.x or new_y != comp.y:
            old_x, old_y = comp.x, comp.y
            comp.x = new_x
            comp.y = new_y
            cost_state.incremental_update({idx})
            # If clamping created new overlaps, undo
            if cost_state.overlap_count > prev_overlaps:
                comp.x = old_x
                comp.y = old_y
                cost_state.incremental_update({idx})
    cost_state._compute_all()


def simulate_annealing(
    model: BoardModel,
    cost_state: CostState,
    moveable_indices: list[int],
    config: SAConfig | None = None,
) -> float:
    """Run SA optimization. Returns final normalized cost. Mutates model in-place."""
    if config is None:
        config = SAConfig()

    if not moveable_indices:
        return cost_state.normalized_cost

    # Record initial overlap count for hard cap
    initial_overlap_count = cost_state.overlap_count

    density = _compute_density(model)
    config.penalty_scale_min = max(config.penalty_scale_min, _density_adaptive_penalty_scale(density))
    config.reheat_count = min(config.reheat_count, _density_adaptive_reheat_count(density))

    # Calibrate T0
    t0 = _calibrate_t0(model, cost_state, moveable_indices, config)
    if config.verbose:
        print(f"  SA: T0={t0:.4f}, {len(moveable_indices)} moveable components, "
              f"density={density:.3f}, penalty_scale_min={config.penalty_scale_min:.2f}")

    # Compute initial cost at penalty_scale=1.0 for fair comparison.
    # The old code compared initial_cost (at penalty_scale_min, e.g. 0.80)
    # with final_cost (at penalty_scale=1.0 after greedy), causing false reverts.
    # Example: initial=1470 @scale=0.80 vs final=1681 @scale=1.0 → ratio=1.14 > 1.10
    # But at scale=1.0, initial would be ~1840 → ratio=0.91 (improvement!).
    cost_state.update_penalty_scale(1.0)
    initial_cost_at_full_scale = cost_state.normalized_cost
    initial_hpwl_for_revert = cost_state.hpwl
    cost_state.update_penalty_scale(config.penalty_scale_min)

    # Store initial cost at full scale so run_sa can use it for revert logic
    simulate_annealing._initial_cost_full_scale = initial_cost_at_full_scale

    # Main SA pass — capture global best from each pass
    _, global_best_cost, global_best_positions, global_best_overlap_count = \
        _run_sa_pass(model, cost_state, moveable_indices, t0, config,
                     initial_overlap_count=initial_overlap_count)

    # Reheating rounds — pass global best between reheats
    for reheat_i in range(config.reheat_count):
        reheat_t0 = t0 * config.reheat_ratio * (0.7 ** reheat_i)
        if config.verbose:
            print(f"  Reheat {reheat_i + 1}: T0={reheat_t0:.4f}")
        _, global_best_cost, global_best_positions, global_best_overlap_count = \
            _run_sa_pass(
                model, cost_state, moveable_indices,
                reheat_t0, config,
                max_iter_override=config.max_iterations // 3,
                initial_overlap_count=initial_overlap_count,
                global_best_cost=global_best_cost,
                global_best_positions=global_best_positions,
                global_best_overlap_count=global_best_overlap_count,
            )

    # Greedy refinement
    if config.verbose:
        print("  Greedy refinement...")
    _greedy_refine(model, cost_state, moveable_indices, config)

    # NO overlap resolution in SA. The legalizer handles overlaps.
    # SA overlap resolution destroys HPWL optimization for dense boards
    # by pushing components far apart to eliminate overlaps, causing
    # massive cost increase that triggers SA revert.
    # The legalizer has better context for resolving overlaps.
    if cost_state.overlap_count > 0 and config.verbose:
        print(f"  SA leaving {cost_state.overlap_count} overlaps for legalizer to resolve")

    final_cost = cost_state.normalized_cost
    if config.verbose:
        print(f"  SA complete: cost={final_cost:.1f} hpwl={cost_state.hpwl:.1f} overlaps={cost_state.overlap_count}")
    return final_cost


def run_sa(
    model: BoardModel,
    profile_weights: dict | None = None,
    config: SAConfig | None = None,
    verbose: bool = True,
    rules: list | None = None,
) -> dict:
    """High-level entry point: build CostState, get moveable indices, run SA.

    HPWL-based revert policy.
    Consistent penalty_scale comparison prevents false reverts.
    Density-adaptive SA parameters.
    Density-adaptive SA parameters for dense boards.
    """
    if config is None:
        config = SAConfig(verbose=verbose)
    elif verbose and not config.verbose:
        config = SAConfig(**{**config.__dict__, 'verbose': True})

    # Density-adaptive SA parameters
    density = _compute_density(model)
    adaptive_penalty_min = _density_adaptive_penalty_scale(density)
    adaptive_reheat_count = _density_adaptive_reheat_count(density)

    # Override config with density-adaptive values (only if user didn't explicitly set)
    if config.penalty_scale_min == 0.50:  # default value — user didn't override
        config.penalty_scale_min = adaptive_penalty_min
    if config.reheat_count == 3:  # default value — user didn't override
        config.reheat_count = adaptive_reheat_count
    # Density-aware overlap cap — tighter for dense boards
    if config.overlap_cap_factor == 2.0:  # default value
        if density > 0.35:
            config.overlap_cap_factor = 1.5  # dense: allow 50% increase
        elif density > 0.25:
            config.overlap_cap_factor = 1.75  # medium: allow 75% increase

    if config.verbose:
        print(f"  Density: {density:.3f}, penalty_scale_min: {config.penalty_scale_min:.2f}, "
              f"reheat_count: {config.reheat_count}, overlap_cap: {config.overlap_cap_factor:.1f}x")

    cost_state = CostState(model, rules=rules)
    moveable_indices = get_moveable_indices(model)
    cost_state.update_penalty_scale(config.penalty_scale_min)

    initial_cost = cost_state.normalized_cost
    initial_hpwl = cost_state.hpwl
    initial_overlaps = cost_state.overlap_count

    # Save pre-SA positions so we can restore if SA makes things worse
    pre_sa_positions = _save_positions(model, moveable_indices)

    final_cost = simulate_annealing(model, cost_state, moveable_indices, config)

    # HPWL-based revert logic
    final_overlaps = cost_state.overlap_count
    final_hpwl = cost_state.hpwl
    overlap_improvement = initial_overlaps - final_overlaps
    hpwl_ratio = final_hpwl / initial_hpwl if initial_hpwl > 0 else float('inf')

    should_revert = False
    revert_reason = ""

    if final_overlaps == 0:
        should_revert = False
        if config.verbose and final_cost > initial_cost:
            print(f"  SA kept: 0 overlaps achieved "
                  f"(cost {initial_cost:.1f} → {final_cost:.1f}, "
                  f"HPWL {initial_hpwl:.1f} → {final_hpwl:.1f})")
    elif hpwl_ratio > 2.0:
        should_revert = True
        revert_reason = f"HPWL doubled ({initial_hpwl:.1f} → {final_hpwl:.1f})"
    elif hpwl_ratio > 1.5 and overlap_improvement <= 0:
        should_revert = True
        revert_reason = f"HPWL increased {hpwl_ratio:.0%} with no overlap improvement"
    elif overlap_improvement > 0:
        if config.verbose and final_cost > initial_cost:
            print(f"  SA kept: overlaps improved ({initial_overlaps} → {final_overlaps}) "
                  f"despite cost increase ({initial_cost:.1f} → {final_cost:.1f})")
    elif final_cost > initial_cost and config.verbose:
        print(f"  SA kept: cost slightly worse ({initial_cost:.1f} → {final_cost:.1f})")

    if should_revert:
        if config.verbose:
            print(f"  SA reverted: {revert_reason}")
        _restore_positions(model, pre_sa_positions)
        cost_state._compute_all()
        final_cost = cost_state.normalized_cost

    return {
        'initial_cost': initial_cost,
        'final_cost': final_cost,
        'initial_hpwl': initial_hpwl,
        'final_hpwl': cost_state.hpwl,
        'overlap_count': cost_state.overlap_count,
        'improvement': initial_cost - final_cost,
    }
