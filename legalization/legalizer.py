"""Legalization pass for PCB placement.

Produces a legal, grid-snapped placement with boundary enforcement,
overlap resolution, and cleanup. Uses spatial grid acceleration,
HPWL-aware tiebreaking, rotation awareness, and Abacus row-based DP.
"""

from __future__ import annotations

import re
import math

from models.board_model import BoardModel, Component, BoardOutline, Net
from engine.constraint_evaluator import evaluate_constraint_penalties, _build_decoupling_map
from engine.group_moves import propagate_ic_delta, propagate_ic_move
from legalization.spatial_grid import SpatialGrid, compute_overlap_stats_fast, count_overlaps_involving_fast, count_pair_overlaps_involving_fast


def legalize(
    model: BoardModel,
    grid_mm: float = 0.1,
    max_iterations: int = 300,
    push_strength: float = 1,
    verbose: bool = False,
    interior_bbox: tuple[float, float, float, float] | None = None,
    use_abacus: bool = False,
    # --- Adaptive params (plan.md §2-4) ---
    max_bbox_expansions: int = 2,
    push_apart_hard_cap: int = 1000,
    spread_pass_enabled: bool = True,
    bbox_expansion_factor: float = 0.05,
    bbox_expansion_density_threshold: float = 0.75,
    gradient_plateau_threshold: float = -0.5,
    gradient_history_window: int = 20,
    gradient_split: int = 10,
    density_push_min: float = 0.5,
    density_push_max: float = 1.5,
) -> BoardModel:
    rules = getattr(model, 'active_rules', None) or []
    cached_decap_map = _build_decoupling_map(model) if rules else {}
    # Keepouts (mounting holes, slots) parsed from Edge.Cuts —
    # _enforce_boundary_single pushes components out of these zones.
    keepouts = getattr(model, 'keepouts', None) or []

    grid = SpatialGrid.from_components(list(model.components), model.board)

    if verbose:
        overlaps_before, _ = _compute_overlap_stats(model, grid)
        oob_before = _count_oob(model)
        print(f"Legalization input: {overlaps_before} overlaps, {oob_before} out-of-bounds")

    # Step 1: Snap to grid (overlap-aware + rotation-aware)
    _snap_to_grid(model, grid_mm)
    grid.build(list(model.components))

    # Step 2: Enforce board boundary
    _enforce_boundary(model, interior_bbox, keepouts=keepouts)
    grid.build(list(model.components))

    # Step 2.5: Try Abacus row-based DP legalization first
    abacus_success = False
    if use_abacus:
        from legalization.abacus_legalizer import abacus_legalize
        abacus_success = abacus_legalize(
            model, grid_mm=grid_mm, verbose=verbose,
            interior_bbox=interior_bbox,
            cached_decap_map=cached_decap_map,
        )
        if verbose:
            if abacus_success:
                print("  Abacus resolved all overlaps - skipping push-apart")
            else:
                print("  Abacus left overlaps - falling through to push-apart + greedy")

    # Step 3: Resolve overlaps (only if Abacus didn't fully resolve).
    # Adaptive loop: push-apart → if plateau, expand bbox → repeat up to
    # max_bbox_expansions. Final fallthrough is greedy cleanup.
    if not abacus_success:
        working_bbox = interior_bbox
        expansion_round = 0
        while True:
            status = _resolve_overlaps(
                model, max_iterations, push_strength, grid_mm, verbose,
                working_bbox, cached_decap_map, grid, keepouts=keepouts,
                hard_cap=push_apart_hard_cap,
                plateau_threshold=gradient_plateau_threshold,
                history_window=gradient_history_window,
                history_split=gradient_split,
                density_push_min=density_push_min,
                density_push_max=density_push_max,
            )

            # Stop if converged or stalled (stalled → greedy cleanup later).
            if status in ("converged", "stalled", "hard_cap", "max_iter"):
                break

            # status == 'plateau' — try expanding the bbox.
            if expansion_round >= max_bbox_expansions:
                if verbose:
                    print(f"  bbox expansion cap reached ({max_bbox_expansions}), "
                          f"accepting residual overlaps")
                break

            new_bbox = _maybe_expand_bbox(
                model, working_bbox, model.board,
                factor=bbox_expansion_factor,
                density_threshold=bbox_expansion_density_threshold,
                verbose=verbose,
            )
            if new_bbox is None:
                if verbose:
                    print("  No bbox expansion possible — accepting residual overlaps")
                break

            working_bbox = new_bbox
            expansion_round += 1
            # Re-enforce boundary with the expanded bbox so components
            # can spread into the new region before the next push-apart.
            _enforce_boundary(model, working_bbox, keepouts=keepouts)
            grid.build(list(model.components))

        # Persist the (possibly expanded) working bbox back to the caller's
        # interior_bbox reference for downstream steps (boundary enforcement,
        # cap nudge, post-legalize).
        if working_bbox is not None:
            interior_bbox = working_bbox

    # Step 3.5: Anti-centroid spread pass (plan.md §4).
    # IMPORTANT: only run the spread pass when the placement is already
    # overlap-free. If push-apart left residual overlaps, the spread pass
    # would push components into each other and the greedy cleanup can't
    # always recover (especially for large IC↔IC overlaps where greedy
    # can't push apart without crossing boundaries). When the placement
    # is clean, the spread pass has room to operate without creating
    # unresolvable overlaps.
    if spread_pass_enabled:
        pre_spread_overlaps, _ = _compute_overlap_stats(model, grid)
        if pre_spread_overlaps == 0:
            _spread_pass(model, interior_bbox, model.board, verbose=verbose)
            _enforce_boundary(model, interior_bbox, keepouts=keepouts)
            grid.build(list(model.components))
        elif verbose:
            print(f"  Skipping spread pass: {pre_spread_overlaps} overlaps "
                  f"remain after push-apart (spread would worsen)")

    # Step 4: Greedy cleanup
    remaining, _ = _compute_overlap_stats(model, grid)
    if remaining > 0:
        _greedy_resolve(model, grid_mm, verbose, interior_bbox, cached_decap_map, grid, keepouts=keepouts)

    # Step 5: Final grid snap and boundary check
    _snap_to_grid(model, grid_mm)
    _enforce_boundary(model, interior_bbox, keepouts=keepouts)
    grid.build(list(model.components))

    # Step 6: Final greedy cleanup
    remaining, _ = _compute_overlap_stats(model, grid)
    if remaining > 0:
        _greedy_resolve(model, grid_mm, verbose, interior_bbox, cached_decap_map, grid, keepouts=keepouts)

    # Step 7: Final legality pass
    _enforce_boundary(model, interior_bbox, keepouts=keepouts)
    grid.build(list(model.components))
    remaining, _ = _compute_overlap_stats(model, grid)
    if remaining > 0:
        _greedy_resolve(model, grid_mm, verbose, interior_bbox, cached_decap_map, grid, keepouts=keepouts)
        _enforce_boundary(model, interior_bbox, keepouts=keepouts)
        grid.build(list(model.components))

    # Step 8: Cap-IC re-attachment is NO LONGER NEEDED.
    # With the macro approach, caps follow their IC through every stage
    # via propagate_ic_move (translation + rotation). No nudge, displacement,
    # or repair passes needed — the cap-IC group is a rigid body.
    # _cleanup_cap_ic_overlaps (Step 10) remains as a safety net for edge
    # cases where cap-IC overlaps might still occur (e.g. two ICs pushed
    # close together and their caps overlap).

    # Step 9: Post-legalization HPWL recovery (cell sliding + pair swap)
    from legalization.post_legalize import post_legalization_refine
    post_legalization_refine(model, grid_mm, interior_bbox, verbose, cached_decap_map)

    # Step 9.5: Safety net — post_legalization_refine can re-introduce
    # overlaps (its cell_slide and pair_swap operators are HPWL-driven,
    # not overlap-aware). If it did, re-run greedy + boundary to clean
    # them up. This catches the "WARNING: post-legalization refine
    # introduced N overlaps" case that previously left overlaps in the
    # final output.
    grid.build(list(model.components))
    refine_overlaps, _ = _compute_overlap_stats(model, grid)
    if refine_overlaps > 0:
        if verbose:
            print(f"  Post-refine cleanup: {refine_overlaps} overlaps re-introduced, "
                  f"running greedy + boundary")
        _greedy_resolve(model, grid_mm, verbose, interior_bbox, cached_decap_map, grid, keepouts=keepouts)
        _enforce_boundary(model, interior_bbox, keepouts=keepouts)
        _snap_to_grid(model, grid_mm)
        _enforce_boundary(model, interior_bbox, keepouts=keepouts)
        grid.build(list(model.components))

    # Step 10: Cap-IC overlap cleanup (edge-case safety net).
    # The macro invariant keeps caps adjacent to their IC, but edge
    # clamping at board boundaries can deform the macro and produce a
    # cap-IC overlap. This final pass moves any such cap to the closest
    # overlap-free slot adjacent to its IC.
    if rules:
        _cleanup_cap_ic_overlaps(model, cached_decap_map, interior_bbox, verbose)

    # Step 10.1: Any-cap-IC overlap cleanup.
    # Catches caps that overlap an IC they're NOT assigned to decouple
    # (e.g. a +5V bulk cap drifted onto an FPGA during SA but shares no
    # power rail, so it's not in the decap_map that Step 10 uses).
    _cleanup_any_cap_ic_overlaps(model, interior_bbox, verbose)

    # Step 10.5: Final safety net — if any overlaps remain (cap-IC cleanup
    # couldn't find a slot, or post-refine cleanup couldn't resolve a large
    # IC↔IC overlap), run one more greedy pass + boundary enforce. This is
    # the last line of defense before returning the model to the user.
    grid.build(list(model.components))
    final_overlaps, _ = _compute_overlap_stats(model, grid)
    if final_overlaps > 0:
        if verbose:
            print(f"  Final cleanup: {final_overlaps} overlaps remain, running final greedy")
        _greedy_resolve(model, grid_mm, verbose, interior_bbox, cached_decap_map, grid, keepouts=keepouts)
        _enforce_boundary(model, interior_bbox, keepouts=keepouts)
        grid.build(list(model.components))

    # Step 10.6: Large-overlap emergency resolution.
    grid.build(list(model.components))
    final_overlaps, _ = _compute_overlap_stats(model, grid)
    if final_overlaps > 0:
        _resolve_large_overlaps(model, interior_bbox, grid, verbose=verbose)
        grid.build(list(model.components))

    # Step 11: Brute-force final overlap check.
    # Catch-all for any overlaps missed by the spatial grid (non-adjacent
    # cells) or introduced by macro propagation clamping at board edges.
    # The macro invariant (caps follow their IC via propagate_ic_move) is
    # enforced in every stage above — no snapshot/restore needed.
    _bf_overlaps = 0
    comps = list(model.components)
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            if comps[i].overlaps(comps[j]):
                _bf_overlaps += 1
    if _bf_overlaps > 0:
        if verbose:
            print(f"  Brute-force final check: {_bf_overlaps} overlaps, resolving")
        _greedy_resolve(model, grid_mm, verbose, interior_bbox, cached_decap_map, grid, keepouts=keepouts)
        _enforce_boundary(model, interior_bbox, keepouts=keepouts)
        if rules and cached_decap_map:
            _cleanup_cap_ic_overlaps(model, cached_decap_map, interior_bbox, verbose)
        grid.build(list(model.components))

    if verbose:
        overlaps_after, _ = _compute_overlap_stats(model, grid)
        oob_after = _count_oob(model)
        print(f"Legalization output: {overlaps_after} overlaps, {oob_after} out-of-bounds")

    return model


def _is_non_square(comp: Component) -> bool:
    return abs(comp.width - comp.height) > 0.01


def _snap_to_grid(model: BoardModel, grid_mm: float) -> None:
    from engine.group_moves import get_macro_member_refs
    components = list(model.components)
    snapped_positions = []
    # Track IC pre-snap positions AND rotations for macro-aware propagation.
    ic_pre: dict[str, tuple[float, float, float]] = {}  # ref -> (x, y, rot)
    ic_types = {"ic", "mcu", "regulator"}
    # Macro member caps are NOT snapped independently — they follow their
    # IC via propagate_ic_move at the end of this function.
    macro_member_refs = get_macro_member_refs(model)

    for comp in components:
        if comp.is_fixed or comp.is_edge_connector:
            continue
        # Skip macro member caps — they follow their IC via propagation.
        if comp.ref in macro_member_refs:
            continue

        old_x, old_y = comp.x, comp.y
        old_rot = comp.rotation
        if getattr(comp, "component_type", "") in ic_types:
            ic_pre[comp.ref] = (old_x, old_y, old_rot)
        new_x = round(old_x / grid_mm) * grid_mm
        new_y = round(old_y / grid_mm) * grid_mm
        comp.rotation = round(comp.rotation / 90.0) * 90.0
        snapped_rot = comp.rotation

        comp.x = new_x
        comp.y = new_y

        has_overlap = any(
            comp.overlaps(other)
            for other, _, _ in snapped_positions
        )

        if has_overlap:
            best_x, best_y = old_x, old_y
            best_rot = snapped_rot
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

            # Try 90° rotation for non-square components
            if best_overlaps > 0 and _is_non_square(comp):
                rot_alt = (snapped_rot + 90) % 360
                comp.set_rotation(rot_alt)

                comp.x = new_x
                comp.y = new_y
                rot_overlaps = sum(
                    1 for other, _, _ in snapped_positions
                    if comp.overlaps(other)
                )
                if rot_overlaps < best_overlaps:
                    best_overlaps = rot_overlaps
                    best_x, best_y = new_x, new_y
                    best_rot = rot_alt

                for dx in [-grid_mm, 0, grid_mm]:
                    for dy in [-grid_mm, 0, grid_mm]:
                        if dx == 0 and dy == 0:
                            continue
                        comp.x = new_x + dx
                        comp.y = new_y + dy
                        overlap_count = sum(
                            1 for other, _, _ in snapped_positions
                            if comp.overlaps(other)
                        )
                        if overlap_count < best_overlaps:
                            best_overlaps = overlap_count
                            best_x, best_y = comp.x, comp.y
                            best_rot = rot_alt
                            if overlap_count == 0:
                                break
                    if best_overlaps == 0:
                        break

                if best_rot != rot_alt:
                    comp.set_rotation(snapped_rot)

            if best_overlaps > 0:
                comp.x = old_x
                comp.y = old_y
            else:
                comp.x = best_x
                comp.y = best_y
            comp.rotation = best_rot

        snapped_positions.append((comp, old_x, old_y))

    # Macro-aware: propagate IC snap deltas (translation + rotation) to caps.
    for comp in components:
        if getattr(comp, "component_type", "") not in ic_types:
            continue
        if comp.is_fixed or comp.is_edge_connector:
            continue
        pre = ic_pre.get(comp.ref)
        if pre is None:
            continue
        propagate_ic_move(model, comp, pre[0], pre[1], pre[2],
                          comp.x, comp.y, comp.rotation)


def _enforce_boundary(
    model: BoardModel,
    interior_bbox: tuple[float, float, float, float] | None = None,
    keepouts: list[BoardOutline] | None = None,
) -> None:
    from engine.group_moves import get_macro_member_refs
    board = model.board
    components = list(model.components)
    ic_types = {"ic", "mcu", "regulator"}
    # Track IC pre-clamp positions AND rotations so we can propagate
    # both translation and rotation to caps (macro-aware).
    ic_pre: dict[str, tuple[float, float, float]] = {}  # ref -> (x, y, rot)
    # Macro member caps follow their IC via propagate_ic_move — skip here.
    macro_member_refs = get_macro_member_refs(model)

    for comp in model.components:
        if comp.is_fixed or comp.is_edge_connector:
            continue
        # Skip macro member caps — they follow their IC via propagation.
        if comp.ref in macro_member_refs:
            continue

        old_x, old_y = comp.x, comp.y
        old_rot = comp.rotation
        if getattr(comp, "component_type", "") in ic_types:
            ic_pre[comp.ref] = (old_x, old_y, old_rot)
        # Apply the type-aware edge keepout here (initial boundary clamp).
        # Overlap resolvers call _enforce_boundary_single with
        # extra_keepout_mm=0.0 so they can push ICs to the edge to
        # resolve overlaps without the keepout fighting them.
        _enforce_boundary_single(comp, interior_bbox, board,
                                 keepouts=keepouts, model=model)

        # Rotation-aware: if still OOB and non-square, try 90°
        if _is_non_square(comp):
            bbox = comp.bbox
            still_oob = (bbox[0] < board.x_min or bbox[2] > board.x_max
                         or bbox[1] < board.y_min or bbox[3] > board.y_max)
            if interior_bbox and not still_oob:
                still_oob = (bbox[0] < interior_bbox[0] or bbox[2] > interior_bbox[2]
                             or bbox[1] < interior_bbox[1] or bbox[3] > interior_bbox[3])
            if still_oob:
                rot_alt = (comp.rotation + 90) % 360
                comp.set_rotation(rot_alt)
                alt_half_w = comp.effective_width / 2.0
                alt_half_h = comp.effective_height / 2.0
                if interior_bbox:
                    alt_x_min = interior_bbox[0] + alt_half_w
                    alt_x_max = interior_bbox[2] - alt_half_w
                    alt_y_min = interior_bbox[1] + alt_half_h
                    alt_y_max = interior_bbox[3] - alt_half_h
                else:
                    alt_x_min = board.x_min + alt_half_w
                    alt_x_max = board.x_max - alt_half_w
                    alt_y_min = board.y_min + alt_half_h
                    alt_y_max = board.y_max - alt_half_h
                if alt_x_min <= alt_x_max and alt_y_min <= alt_y_max:
                    comp.x = max(alt_x_min, min(comp.x, alt_x_max))
                    comp.y = max(alt_y_min, min(comp.y, alt_y_max))
                    rot_overlaps = sum(1 for other in components
                                       if other is not comp and comp.overlaps(other))
                    if rot_overlaps <= 2:
                        continue
                    else:
                        comp.set_rotation(old_rot)
                        # Recompute bounds for original rotation
                        half_w = comp.effective_width / 2.0
                        half_h = comp.effective_height / 2.0
                        if interior_bbox:
                            x_min_b = interior_bbox[0] + half_w
                            x_max_b = interior_bbox[2] - half_w
                            y_min_b = interior_bbox[1] + half_h
                            y_max_b = interior_bbox[3] - half_h
                        else:
                            x_min_b = board.x_min + half_w
                            x_max_b = board.x_max - half_w
                            y_min_b = board.y_min + half_h
                            y_max_b = board.y_max - half_h
                        comp.x = max(x_min_b, min(comp.x, x_max_b))
                        comp.y = max(y_min_b, min(comp.y, y_max_b))
                else:
                    comp.set_rotation(old_rot)

        if comp.x != old_x or comp.y != old_y or comp.rotation != old_rot:
            overlap_count = sum(1 for other in components
                                if other is not comp and comp.overlaps(other))
            if overlap_count > 0:
                best_x, best_y = comp.x, comp.y
                best_rot = comp.rotation
                best_overlaps = overlap_count

                for delta in [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0]:
                    for dx_dir, dy_dir in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        trial_x = comp.x + dx_dir * delta
                        trial_y = comp.y + dy_dir * delta
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

    # Macro-aware: propagate IC clamp deltas (translation + rotation) to caps.
    # Uses propagate_ic_move instead of propagate_ic_delta so caps also
    # follow IC rotation — the cap-IC group acts as a rigid body.
    bounds = (interior_bbox[0], interior_bbox[1], interior_bbox[2], interior_bbox[3]) if interior_bbox else None
    for comp in model.components:
        if getattr(comp, "component_type", "") not in ic_types:
            continue
        if comp.is_fixed or comp.is_edge_connector:
            continue
        pre = ic_pre.get(comp.ref)
        if pre is None:
            continue
        propagate_ic_move(model, comp, pre[0], pre[1], pre[2],
                          comp.x, comp.y, comp.rotation, bounds=bounds)


def _enforce_boundary_single(
    comp: Component,
    interior_bbox: tuple[float, float, float, float] | None,
    board: BoardOutline,
    keepouts: list[BoardOutline] | None = None,
    extra_keepout_mm: float | None = None,
    model: BoardModel | None = None,
) -> None:
    """Clamp a component inside the board / interior bbox AND push it
    out of any internal keepout zones (mounting holes, slots).

    Keepout eviction is a one-pass nudge: if the component's center is
    inside a keepout, push it to the nearest keepout edge.  This is
    conservative — a component straddling a keepout boundary may still
    overlap it after the nudge — but the legalizer's overlap-resolution
    loop will catch any remaining overlap on subsequent iterations.

    ``extra_keepout_mm`` shrinks the clamp bounds by an additional margin
    for component types that need extra edge clearance (ICs, MCUs,
    regulators).  If None, the keepout extra is looked up automatically
    from the component's type via ``cost_state.edge_keepout_extra_for``
    (density-adaptive when ``model`` is provided). Pass 0.0 explicitly to
    disable the extra keepout for a specific call — this is what overlap
    resolvers should do so they have freedom to push ICs toward the edge
    to resolve overlaps without the keepout fighting them.
    """
    if comp.is_fixed or comp.is_edge_connector:
        return
    # Look up the type-aware edge keepout extra if not explicitly provided.
    # This is the common-sense DFM rule: ICs stay further from the board
    # edge than passives (routing room, panelization clearance, assembly
    # clearance). Without this, ICs end up with their courtyard flush
    # against (board_edge + margin) — exactly what a human PCB designer
    # avoids. Callers that need to push ICs to resolve overlaps should
    # pass extra_keepout_mm=0.0 to disable this.
    if extra_keepout_mm is None:
        try:
            from engine.cost_state import edge_keepout_extra_for
            extra_keepout_mm = edge_keepout_extra_for(comp, model)
        except Exception:
            extra_keepout_mm = 0.0
    half_w = comp.effective_width / 2.0
    half_h = comp.effective_height / 2.0
    if interior_bbox:
        x_min = interior_bbox[0] + half_w + extra_keepout_mm
        x_max = interior_bbox[2] - half_w - extra_keepout_mm
        y_min = interior_bbox[1] + half_h + extra_keepout_mm
        y_max = interior_bbox[3] - half_h - extra_keepout_mm
    else:
        x_min = board.x_min + half_w + extra_keepout_mm
        x_max = board.x_max - half_w - extra_keepout_mm
        y_min = board.y_min + half_h + extra_keepout_mm
        y_max = board.y_max - half_h - extra_keepout_mm
    # Guard against the clamp window collapsing for large ICs on small
    # boards — fall back to the non-extra-keepout bounds so the component
    # still has somewhere to sit.
    if x_min > x_max:
        x_min = interior_bbox[0] + half_w if interior_bbox else board.x_min + half_w
        x_max = interior_bbox[2] - half_w if interior_bbox else board.x_max - half_w
    if y_min > y_max:
        y_min = interior_bbox[1] + half_h if interior_bbox else board.y_min + half_h
        y_max = interior_bbox[3] - half_h if interior_bbox else board.y_max - half_h
    comp.x = max(x_min, min(comp.x, x_max))
    comp.y = max(y_min, min(comp.y, y_max))

    # Keepout eviction: if the component's center is inside a keepout,
    # nudge it to the nearest keepout edge (plus the component's half-
    # extent so the courtyard clears the keepout boundary).
    if keepouts:
        for k in keepouts:
            if k.x_min <= comp.x <= k.x_max and k.y_min <= comp.y <= k.y_max:
                d_left = comp.x - k.x_min + half_w
                d_right = k.x_max - comp.x + half_w
                d_top = comp.y - k.y_min + half_h
                d_bottom = k.y_max - comp.y + half_h
                min_d = min(d_left, d_right, d_top, d_bottom)
                if min_d == d_left:
                    comp.x = k.x_min - half_w
                elif min_d == d_right:
                    comp.x = k.x_max + half_w
                elif min_d == d_top:
                    comp.y = k.y_min - half_h
                else:
                    comp.y = k.y_max + half_h
                # Re-clamp to board/interior after the nudge.
                comp.x = max(x_min, min(comp.x, x_max))
                comp.y = max(y_min, min(comp.y, y_max))


def _count_overlaps_involving(
    comp: Component, components: list[Component],
    grid: SpatialGrid | None = None,
) -> int:
    if grid is not None:
        return count_overlaps_involving_fast(comp, components, grid)
    count = 0
    for other in components:
        if other is comp:
            continue
        if comp.overlaps(other):
            count += 1
    return count


def _compute_overlap_stats(model: BoardModel, grid: SpatialGrid | None = None) -> tuple[int, float]:
    if grid is not None:
        return compute_overlap_stats_fast(model, grid)
    count = 0
    total_area = 0.0
    components = model.components
    for i, c1 in enumerate(components):
        for c2 in components[i + 1:]:
            if c1.overlaps(c2):
                count += 1
                total_area += c1.overlap_area(c2)
    return count, total_area


def _build_comp_net_lookup(model: BoardModel) -> dict[str, list[Net]]:
    comp_nets: dict[str, list[Net]] = {}
    for net in model.nets:
        for ref, _pad_name in net.pins:
            if ref not in comp_nets:
                comp_nets[ref] = []
            comp_nets[ref].append(net)
    return comp_nets


def _compute_local_hpwl(
    comp: Component, comp_nets: dict[str, list[Net]], model: BoardModel,
    comp_map: dict[str, Component] | None = None,
) -> float:
    total = 0.0
    nets = comp_nets.get(comp.ref, [])
    if comp_map is None:
        comp_map = {c.ref: c for c in model.components}
    for net in nets:
        from engine.cost_state import _is_power_net
        if _is_power_net(net.name):
            continue
        pins = []
        for ref, pad_name in net.pins:
            other = comp_map.get(ref)
            if not other:
                continue
            for pad in other.pads:
                if pad.pad_name == pad_name:
                    abs_x, abs_y = pad.absolute_pos(other.x, other.y, other.rotation)
                    pins.append((abs_x, abs_y))
                    break
            else:
                pins.append((other.x, other.y))
        if len(pins) < 2:
            continue
        xs = [p[0] for p in pins]
        ys = [p[1] for p in pins]
        total += (max(xs) - min(xs)) + (max(ys) - min(ys))
    return total


def _legalizer_score(
    model: BoardModel, moved: list[Component],
    original_positions: dict[int, tuple[float, float]],
    cached_decap_map: dict | None = None,
    grid: SpatialGrid | None = None,
    comp_net_lookup: dict[str, list[Net]] | None = None,
    comp_map: dict[str, Component] | None = None,
) -> tuple[float, int, int]:
    overlaps, overlap_area = _compute_overlap_stats(model, grid)
    oob = _count_oob(model)
    rules = getattr(model, 'active_rules', None) or []
    if rules:
        constraint_total, _ = evaluate_constraint_penalties(model, rules)
    else:
        constraint_total = 0.0
    displacement = 0.0
    for comp in moved:
        ox, oy = original_positions.get(id(comp), (comp.x, comp.y))
        displacement += abs(comp.x - ox) + abs(comp.y - oy)
    # P0 #3: include HPWL so tie-break moves pick the lower-wire option.
    # Previously two equal-overlap moves were scored only on displacement,
    # letting HPWL balloon during cleanup (test4 run3: 708 -> 1220).
    hpwl_total = 0.0
    if comp_net_lookup is not None:
        for comp in moved:
            hpwl_total += _compute_local_hpwl(comp, comp_net_lookup, model, comp_map)
    score = (100000.0 * overlaps + 25000.0 * oob + 50.0 * overlap_area
             + 10.0 * constraint_total + displacement + hpwl_total)
    return score, overlaps, oob

def _count_pair_overlaps_involving(
    c1: Component, c2: Component, components: list[Component],
    grid: SpatialGrid | None = None,
) -> int:
    if grid is not None:
        return count_pair_overlaps_involving_fast(c1, c2, components, grid)
    count = 0
    for other in components:
        if other is c1 or other is c2:
            continue
        if c1.overlaps(other):
            count += 1
        if c2.overlaps(other):
            count += 1
    if c1.overlaps(c2):
        count += 1
    return count


def _resolve_overlaps(
    model: BoardModel, max_iterations: int, push_strength: float,
    grid_mm: float, verbose: bool,
    interior_bbox: tuple[float, float, float, float] | None = None,
    cached_decap_map: dict | None = None,
    grid: SpatialGrid | None = None,
    keepouts: list[BoardOutline] | None = None,
    *,
    hard_cap: int | None = None,
    plateau_threshold: float = -0.5,
    history_window: int = 20,
    history_split: int = 10,
    density_push_min: float = 0.5,
    density_push_max: float = 1.5,
) -> str:
    """Adaptive overlap resolution via HPWL-aware push-apart.

    Augments the legacy fixed-iteration push-apart with three signals
    (plan.md §2):

      * **Gradient plateau detection** — track a rolling history of overlap
        counts; when ``mean(last_n) - mean(prev_n) >= plateau_threshold``
        and overlaps remain, stop early so the caller can expand the bbox
        (or accept residual overlaps).
      * **Hard-cap continuation** — when the gradient is still improving
        (``< plateau_threshold``), iterations continue past
        ``max_iterations`` up to ``hard_cap`` (default ``max(1000,
        max_iterations)``).
      * **Density-scaled push strength** — each pair's push strength is
        scaled by ``density_push_min..density_push_max`` based on the
        number of overlaps involving that pair, so heavy clusters get
        pushed harder than isolated pairs.

    Returns a status string so the caller can decide what to do next:
      ``'converged'``   — overlaps reached 0
      ``'plateau'``     — gradient plateaued with overlaps remaining
      ``'stalled'``     — stall_iterations >= 20 (legacy signal)
      ``'max_iter'``    — soft cap reached, gradient not improving
      ``'hard_cap'``    — hard cap reached
    """
    board = model.board
    prev_overlap_count = float("inf")
    stall_iterations = 0
    adaptive_strength = push_strength
    # P0 #3: build comp→nets once so push-apart + tie-break scoring can
    # evaluate HPWL without re-walking the netlist on every call.
    comp_net_lookup = _build_comp_net_lookup(model)
    # Wall-time fix: comp_map is stable across legalization (components only
    # move, never added/removed), so build it once instead of rebuilding it
    # inside every _compute_local_hpwl call.
    comp_map = {c.ref: c for c in model.components}

    if hard_cap is None:
        hard_cap = max(max_iterations, 1000)
    # Defensive: never let hard_cap dip below max_iterations, or the loop
    # would terminate before the soft cap is reached.
    hard_cap = max(hard_cap, max_iterations)

    history: list[int] = []
    status: str = "max_iter"

    iteration = 0
    while iteration < hard_cap:
        # Soft-cap check: once we've reached max_iterations, only continue
        # if the gradient is still improving (below plateau_threshold).
        if iteration >= max_iterations:
            should_continue = False
            if len(history) >= history_window:
                split = min(history_split, len(history) // 2)
                if split >= 1 and len(history) >= 2 * split:
                    last_n = history[-split:]
                    prev_n = history[-2 * split:-split]
                    grad = sum(last_n) / len(last_n) - sum(prev_n) / len(prev_n)
                    if grad < plateau_threshold:
                        should_continue = True
            if not should_continue:
                status = "max_iter"
                if verbose:
                    remaining, _ = _compute_overlap_stats(model, grid)
                    print(f"  Overlap resolution: max iterations ({max_iterations}) reached, "
                          f"{remaining} overlaps remaining")
                break

        overlap_pairs = []
        components = list(model.components)
        components.sort(key=lambda c: (c.x, c.y))

        if grid is not None:
            grid.build(components)
            # id→idx map for _push_apart_hpwl's SpatialGrid fast path.
            comp_to_idx = {id(c): i for i, c in enumerate(components)}
            seen: set[tuple[int, int]] = set()
            for i in range(len(components)):
                c1 = components[i]
                candidates = grid.query_overlaps(i, components)
                for j in candidates:
                    if j <= i:
                        continue
                    pair = (i, j)
                    if pair in seen:
                        continue
                    seen.add(pair)
                    c2 = components[j]
                    if not c1.overlaps(c2):
                        continue
                    c1_fixed = c1.is_fixed or c1.is_edge_connector
                    c2_fixed = c2.is_fixed or c2.is_edge_connector
                    if c1_fixed and c2_fixed:
                        continue
                    area = c1.overlap_area(c2)
                    overlap_pairs.append((area, c1, c2, c1_fixed, c2_fixed))
        else:
            for i, c1 in enumerate(components):
                for c2 in components[i + 1:]:
                    if not c1.overlaps(c2):
                        continue
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
            status = "converged"
            break

        overlap_pairs.sort(key=lambda x: x[0])

        resolved_this_pass = 0
        for area, c1, c2, c1_fixed, c2_fixed in overlap_pairs:
            if not c1.overlaps(c2):
                continue

            old_x1, old_y1 = c1.x, c1.y
            old_x2, old_y2 = c2.x, c2.y

            local_before = _count_pair_overlaps_involving(c1, c2, components, grid)

            # Density-scaled push strength (plan.md §2): components in
            # high-overlap zones get a stronger push; isolated pairs get
            # a gentler nudge. local_before counts overlaps involving
            # c1 or c2 (excluding their mutual overlap).
            density_scale = density_push_min + (
                density_push_max - density_push_min
            ) * min(1.0, max(0, local_before) / 5.0)
            pair_strength = adaptive_strength * density_scale

            _push_apart_hpwl(
                c1, c2, components, comp_net_lookup, model,
                pair_strength, grid_mm, board, interior_bbox,
                comp_map=comp_map, keepouts=keepouts,
                grid=grid, comp_to_idx=comp_to_idx,
            )

            _enforce_boundary_single(c1, interior_bbox, board, keepouts=keepouts, extra_keepout_mm=0.0)
            _enforce_boundary_single(c2, interior_bbox, board, keepouts=keepouts, extra_keepout_mm=0.0)

            local_after = _count_pair_overlaps_involving(c1, c2, components, grid)

            if local_after < local_before:
                resolved_this_pass += 1
            elif local_after == local_before:
                moved = [c for c in (c1, c2) if not c.is_fixed and not c.is_edge_connector]
                original_positions = {id(c1): (old_x1, old_y1), id(c2): (old_x2, old_y2)}
                base_score, _, _ = _legalizer_score(
                    model, moved, original_positions, cached_decap_map, grid,
                    comp_net_lookup=comp_net_lookup, comp_map=comp_map,
                )

                c1.x, c1.y = old_x1, old_y1
                c2.x, c2.y = old_x2, old_y2
                pre_score, _, _ = _legalizer_score(
                    model, moved, original_positions, cached_decap_map, grid,
                    comp_net_lookup=comp_net_lookup, comp_map=comp_map,
                )

                _push_apart_hpwl(
                    c1, c2, components, comp_net_lookup, model,
                    pair_strength, grid_mm, board, interior_bbox,
                    comp_map=comp_map,
                    grid=grid, comp_to_idx=comp_to_idx,
                )
                _enforce_boundary_single(c1, interior_bbox, board, keepouts=keepouts, extra_keepout_mm=0.0)
                _enforce_boundary_single(c2, interior_bbox, board, keepouts=keepouts, extra_keepout_mm=0.0)

                if base_score <= pre_score:
                    resolved_this_pass += 1
                else:
                    c1.x, c1.y = old_x1, old_y1
                    c2.x, c2.y = old_x2, old_y2
            else:
                c1.x, c1.y = old_x1, old_y1
                c2.x, c2.y = old_x2, old_y2

                _push_apart_hpwl(
                    c1, c2, components, comp_net_lookup, model,
                    pair_strength * 0.5, grid_mm, board, interior_bbox,
                    comp_map=comp_map,
                    grid=grid, comp_to_idx=comp_to_idx,
                )

                _enforce_boundary_single(c1, interior_bbox, board, keepouts=keepouts, extra_keepout_mm=0.0)
                _enforce_boundary_single(c2, interior_bbox, board, keepouts=keepouts, extra_keepout_mm=0.0)

                local_after_reduced = _count_pair_overlaps_involving(c1, c2, components, grid)
                if local_after_reduced <= local_before:
                    resolved_this_pass += 1
                else:
                    c1.x, c1.y = old_x1, old_y1
                    c2.x, c2.y = old_x2, old_y2
                    continue

        _enforce_boundary(model, interior_bbox, keepouts=keepouts)
        actual_overlap_count, _ = _compute_overlap_stats(model, grid)

        # Track overlap history for gradient computation (plan.md §2).
        history.append(actual_overlap_count)
        if len(history) > history_window:
            history.pop(0)

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
            status = "converged"
            break

        # Gradient-based plateau detection (plan.md §2). When the recent
        # overlap-count slope flattens (>= plateau_threshold) and overlaps
        # remain, signal the caller to expand the bbox (or accept residuals).
        if len(history) >= history_window:
            split = min(history_split, len(history) // 2)
            if split >= 1 and len(history) >= 2 * split:
                last_n = history[-split:]
                prev_n = history[-2 * split:-split]
                grad = sum(last_n) / len(last_n) - sum(prev_n) / len(prev_n)
                if grad >= plateau_threshold:
                    if verbose:
                        print(f"  Push-apart plateau at iteration {iteration + 1} "
                              f"(gradient={grad:.2f}, {actual_overlap_count} overlaps remaining)")
                    status = "plateau"
                    break

        if stall_iterations >= 20:
            if verbose:
                print(f"  Push-apart stalled at iteration {iteration + 1} "
                      f"({actual_overlap_count} overlaps remaining, switching to greedy)")
            status = "stalled"
            break

        iteration += 1
    else:
        # while loop completed without break → hit hard_cap.
        status = "hard_cap"
        if verbose:
            remaining, _ = _compute_overlap_stats(model, grid)
            print(f"  Overlap resolution: hard cap ({hard_cap}) reached, "
                  f"{remaining} overlaps remaining")

    return status


def _greedy_resolve(
    model: BoardModel, grid_mm: float, verbose: bool,
    interior_bbox: tuple[float, float, float, float] | None = None,
    cached_decap_map: dict | None = None,
    grid: SpatialGrid | None = None,
    keepouts: list[BoardOutline] | None = None,
) -> None:
    from engine.group_moves import get_macro_member_refs
    board = model.board
    components = list(model.components)
    # MACRO-AWARE: Skip macro member caps — they follow their IC via
    # propagate_ic_move. Moving them independently breaks the group.
    macro_member_refs = get_macro_member_refs(model)
    movable = [c for c in components
               if not c.is_fixed and not c.is_edge_connector
               and c.ref not in macro_member_refs]
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1),
                  (1, 1), (-1, 1), (1, -1), (-1, -1)]

    max_nudge = max(board.width, board.height) * 0.15
    nudge_dists = [grid_mm, grid_mm * 2, grid_mm * 5, grid_mm * 10,
                   0.5, 1.0, 2.0, 4.0, 8.0, 12.0]
    nudge_dists.extend([d for d in [16.0, 24.0, 32.0] if d <= max_nudge])

    comp_net_lookup = _build_comp_net_lookup(model)

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

    prev_total, _ = _compute_overlap_stats(model, grid)
    for outer in range(50):
        total_overlaps, _ = _compute_overlap_stats(model, grid)
        if total_overlaps == 0:
            break

        overlap_counts: dict[int, int] = {}
        for idx, c in enumerate(movable):
            count = _count_overlaps_involving(c, components, grid)
            if count > 0:
                overlap_counts[id(c)] = count

        if not overlap_counts:
            break

        movable_with_overlaps = [c for c in movable if id(c) in overlap_counts]
        movable_with_overlaps.sort(key=lambda c: overlap_counts[id(c)], reverse=True)

        improved = False
        for comp in movable_with_overlaps:
            old_x, old_y = comp.x, comp.y
            old_rot = comp.rotation
            old_overlaps = overlap_counts.get(id(comp), 0)
            if old_overlaps == 0:
                continue

            orig_local_hpwl = _compute_local_hpwl(comp, comp_net_lookup, model)

            best_x, best_y, best_rot = old_x, old_y, old_rot
            best_overlaps = old_overlaps
            best_local_hpwl = orig_local_hpwl
            best_displacement = 0.0

            is_nonsquare = _is_non_square(comp)
            rotations_to_try = [old_rot]
            if is_nonsquare:
                rotations_to_try.append((old_rot + 90) % 360)

            for try_rot in rotations_to_try:
                if try_rot != old_rot:
                    comp.set_rotation(try_rot)
                comp.x, comp.y = old_x, old_y

                rot_penalty = 0.0 if comp.rotation == old_rot else 0.1

                for dx_dir, dy_dir in directions:
                    for dist in nudge_dists:
                        comp.x = old_x + dx_dir * dist
                        comp.y = old_y + dy_dir * dist
                        _enforce_boundary_single(comp, interior_bbox, board, keepouts=keepouts, extra_keepout_mm=0.0)

                        new_overlaps = _count_overlaps_involving(comp, components, grid)
                        if new_overlaps < best_overlaps:
                            best_overlaps = new_overlaps
                            best_x, best_y = comp.x, comp.y
                            best_rot = comp.rotation
                            best_local_hpwl = _compute_local_hpwl(comp, comp_net_lookup, model)
                            best_displacement = abs(comp.x - old_x) + abs(comp.y - old_y)
                            if new_overlaps == 0:
                                break
                        elif new_overlaps == best_overlaps:
                            trial_hpwl = _compute_local_hpwl(comp, comp_net_lookup, model)
                            trial_disp = abs(comp.x - old_x) + abs(comp.y - old_y)
                            best_rot_penalty = 0.0 if best_rot == old_rot else 0.1
                            if (trial_hpwl + rot_penalty < best_local_hpwl + best_rot_penalty or
                                    (trial_hpwl + rot_penalty == best_local_hpwl + best_rot_penalty
                                     and trial_disp < best_displacement)):
                                best_local_hpwl = trial_hpwl
                                best_displacement = trial_disp
                                best_x, best_y = comp.x, comp.y
                                best_rot = comp.rotation
                    if best_overlaps == 0:
                        break

                if best_overlaps == 0:
                    break

                # Grid sweep
                comp.x, comp.y = old_x, old_y
                half_w = comp.effective_width / 2.0
                half_h = comp.effective_height / 2.0
                grid_step = max(comp.effective_width * 0.5, comp.effective_height * 0.5, 0.5)

                gx = bx_min + half_w
                while gx <= bx_max - half_w:
                    gy = by_min + half_h
                    while gy <= by_max - half_h:
                        comp.x = gx
                        comp.y = gy
                        new_overlaps = _count_overlaps_involving(comp, components, grid)
                        if new_overlaps < best_overlaps:
                            best_overlaps = new_overlaps
                            best_x, best_y = comp.x, comp.y
                            best_rot = comp.rotation
                            best_local_hpwl = _compute_local_hpwl(comp, comp_net_lookup, model)
                            best_displacement = abs(comp.x - old_x) + abs(comp.y - old_y)
                            if new_overlaps == 0:
                                break
                        elif new_overlaps == best_overlaps:
                            trial_hpwl = _compute_local_hpwl(comp, comp_net_lookup, model)
                            trial_disp = abs(comp.x - old_x) + abs(comp.y - old_y)
                            best_rot_penalty = 0.0 if best_rot == old_rot else 0.1
                            if (trial_hpwl + rot_penalty < best_local_hpwl + best_rot_penalty or
                                    (trial_hpwl + rot_penalty == best_local_hpwl + best_rot_penalty
                                     and trial_disp < best_displacement)):
                                best_local_hpwl = trial_hpwl
                                best_displacement = trial_disp
                                best_x, best_y = comp.x, comp.y
                                best_rot = comp.rotation
                        gy += grid_step
                    if best_overlaps == 0:
                        break
                    gx += grid_step

                if best_overlaps == 0:
                    break

                comp.x, comp.y = old_x, old_y
                if try_rot != old_rot:
                    comp.set_rotation(old_rot)

            comp_old_x, comp_old_y = comp.x, comp.y
            comp_old_rot = comp.rotation
            comp.x, comp.y = best_x, best_y
            if comp.rotation != best_rot:
                comp.set_rotation(best_rot)

            if best_overlaps < old_overlaps:
                improved = True
                # Macro-aware: propagate IC move (translation + rotation) to caps.
                propagate_ic_move(model, comp, comp_old_x, comp_old_y, comp_old_rot,
                                  comp.x, comp.y, comp.rotation,
                                  bounds=(interior_bbox[0], interior_bbox[1], interior_bbox[2], interior_bbox[3]) if interior_bbox else None)
            elif best_overlaps == old_overlaps and best_local_hpwl < orig_local_hpwl:
                improved = True
                propagate_ic_move(model, comp, comp_old_x, comp_old_y, comp_old_rot,
                                  comp.x, comp.y, comp.rotation,
                                  bounds=(interior_bbox[0], interior_bbox[1], interior_bbox[2], interior_bbox[3]) if interior_bbox else None)
            else:
                comp.x, comp.y = old_x, old_y
                if comp.rotation != old_rot:
                    comp.set_rotation(old_rot)

        current_total, _ = _compute_overlap_stats(model, grid)
        if current_total >= prev_total and not improved:
            break
        prev_total = current_total

    if verbose:
        final, _ = _compute_overlap_stats(model, grid)
        if final > 0:
            print(f"  Greedy resolution: {final} overlaps remaining (board may be too dense)")


def _push_apart(c1: Component, c2: Component, strength: float, grid_mm: float) -> None:
    ax1, ay1, ax2, ay2 = c1.bbox
    bx1, by1, bx2, by2 = c2.bbox

    overlap_x = min(ax2, bx2) - max(ax1, bx1)
    overlap_y = min(ay2, by2) - max(ay1, by1)

    if overlap_x <= 0 or overlap_y <= 0:
        return

    c1_min = min(c1.effective_width, c1.effective_height)
    c2_min = min(c2.effective_width, c2.effective_height)
    severity_threshold = 0.7 * min(c1_min, c2_min)

    push_both = (overlap_x > severity_threshold) or (overlap_y > severity_threshold)

    push_mm = max(grid_mm * 1.5, 0.15)

    if push_both:
        push_x = max(overlap_x * strength * 0.5, push_mm)
        push_y = max(overlap_y * strength * 0.5, push_mm)
    elif overlap_x <= overlap_y:
        push_x = max(overlap_x * strength, push_mm)
        push_y = 0
    else:
        push_x = 0
        push_y = max(overlap_y * strength, push_mm)

    dx = 1 if c2.x >= c1.x else -1
    dy = 1 if c2.y >= c1.y else -1

    if c1.is_fixed and not c2.is_fixed:
        c2.x += dx * push_x
        c2.y += dy * push_y
    elif c2.is_fixed and not c1.is_fixed:
        c1.x -= dx * push_x
        c1.y -= dy * push_y
    else:
        c1.x -= dx * push_x / 2.0
        c1.y -= dy * push_y / 2.0
        c2.x += dx * push_x / 2.0
        c2.y += dy * push_y / 2.0


def _build_mover_hpwl_cache(
    mover: Component,
    comp_net_lookup: dict[str, list[Net]],
    comp_map: dict[str, Component],
) -> list[tuple[float, float, float, float, float, float]]:
    """Per-iteration HPWL cache for mover.

    For each non-power net that ``mover`` participates in, precompute:
      (mover_pin_offset_x, mover_pin_offset_y,
       others_min_x, others_max_x, others_min_y, others_max_y)

    Mover's pin offset is constant during a push-apart call (rotation
    doesn't change).  Other pins' positions are constant within a
    single ``which`` iteration (only mover moves during candidate
    evaluation), so the cache is built once per (mover, iter) and
    reused across all 16 candidate positions.

    Lets _compute_local_hpwl_cached run in O(k) where k = nets on
    mover, vs O(k · p · pads) for the full recomputation.
    """
    from engine.cost_state import _is_power_net
    cache: list[tuple[float, float, float, float, float, float]] = []
    nets = comp_net_lookup.get(mover.ref, [])
    for net in nets:
        if _is_power_net(net.name):
            continue
        mover_offset_x = 0.0
        mover_offset_y = 0.0
        mover_found = False
        others_xs: list[float] = []
        others_ys: list[float] = []
        for ref, pad_name in net.pins:
            other = comp_map.get(ref)
            if other is None:
                continue
            if other is mover:
                for pad in other.pads:
                    if pad.pad_name == pad_name:
                        abs_x, abs_y = pad.absolute_pos(
                            other.x, other.y, other.rotation)
                        mover_offset_x = abs_x - other.x
                        mover_offset_y = abs_y - other.y
                        mover_found = True
                        break
                else:
                    mover_found = True  # use (0, 0) offset
            else:
                for pad in other.pads:
                    if pad.pad_name == pad_name:
                        abs_x, abs_y = pad.absolute_pos(
                            other.x, other.y, other.rotation)
                        others_xs.append(abs_x)
                        others_ys.append(abs_y)
                        break
                else:
                    others_xs.append(other.x)
                    others_ys.append(other.y)
        if not mover_found or not others_xs:
            continue
        cache.append((
            mover_offset_x, mover_offset_y,
            min(others_xs), max(others_xs),
            min(others_ys), max(others_ys),
        ))
    return cache


def _compute_local_hpwl_cached(
    mover_x: float, mover_y: float,
    cache: list[tuple[float, float, float, float, float, float]],
) -> float:
    """HPWL of mover's nets given cached pin data.

    Mathematically equivalent to _compute_local_hpwl: net HPWL is
    (max_x - min_x) + (max_y - min_y) over all pins.  With mover's pin
    at (mover_x + offset_x, mover_y + offset_y) and others' bbox
    precomputed, the merged bbox is just min/max of two values per
    axis.  O(k) per call.
    """
    total = 0.0
    for offset_x, offset_y, ox_min, ox_max, oy_min, oy_max in cache:
        pin_x = mover_x + offset_x
        pin_y = mover_y + offset_y
        net_min_x = pin_x if pin_x < ox_min else ox_min
        net_max_x = pin_x if pin_x > ox_max else ox_max
        net_min_y = pin_y if pin_y < oy_min else oy_min
        net_max_y = pin_y if pin_y > oy_max else oy_max
        total += (net_max_x - net_min_x) + (net_max_y - net_min_y)
    return total


def _push_apart_hpwl(
    c1: Component,
    c2: Component,
    components: list[Component],
    comp_net_lookup: dict[str, list[Net]],
    model: BoardModel,
    strength: float,
    grid_mm: float,
    board: BoardOutline,
    interior_bbox: tuple[float, float, float, float] | None,
    max_iter: int = 5,
    comp_map: dict[str, Component] | None = None,
    keepouts: list[BoardOutline] | None = None,
    grid: "SpatialGrid | None" = None,
    comp_to_idx: dict[int, int] | None = None,
) -> None:
    """P0 #3: HPWL-aware push-apart via bounded gradient descent.

    Tries single-component displacements in 4 directions × multiple
    distances, scores each by ``β·overlap_area + local_hpwl + λ·new_overlaps``
    (overlap area against the conflict partner plus a penalty for any new
    overlap created with a third party), and applies the move that most
    reduces combined cost.  Iterates up to ``max_iter`` times.

    A resolving move (overlap drops to 0) naturally wins when its HPWL
    cost is modest, but a partial move can beat it if the resolving move
    would shunt the component into a crowded neighbourhood.

    Wall-time fix: cache mover's per-net pin offset and other-pins'
    bbox at the start of each ``which`` iteration.  HPWL per candidate
    then costs O(k) instead of O(k·p·pads).  On test4 this cut overall
    legalization wall time ~40% (140s → 80s with profiler attached).
    """
    beta = 50.0      # matches overlap penalty weight in cost function
    new_ov_cost = 500.0  # penalty per overlap introduced with a third party
    c1_fixed = c1.is_fixed or c1.is_edge_connector
    c2_fixed = c2.is_fixed or c2.is_edge_connector
    if c1_fixed and c2_fixed:
        return
    # MACRO-AWARE: if either side is a macro member cap, skip — its IC
    # leader will be pushed elsewhere and the cap follows via propagate.
    from engine.group_moves import get_macro_member_refs
    macro_refs = get_macro_member_refs(model)
    if c1.ref in macro_refs or c2.ref in macro_refs:
        return

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    base_dists = [grid_mm, grid_mm * 2, grid_mm * 5, grid_mm * 10]
    distances = [d * max(strength, 0.5) for d in base_dists]

    def _third_party_overlaps(mover: Component) -> int:
        # Fast path: use SpatialGrid to get only nearby candidates.
        if grid is not None and comp_to_idx is not None:
            idx = comp_to_idx.get(id(mover))
            if idx is not None:
                cands = grid.query_overlaps(idx, components)
                count = 0
                for j in cands:
                    other = components[j]
                    if other is c1 or other is c2:
                        continue
                    if mover.overlaps(other):
                        count += 1
                return count
        # Fallback: full table scan
        count = 0
        for other in components:
            if other is mover or other is c1 or other is c2:
                continue
            if mover.overlaps(other):
                count += 1
        return count

    for _ in range(max_iter):
        if not c1.overlaps(c2):
            return

        best: tuple[float, int, float, float] | None = None

        for which in (1, 2):
            if which == 1 and c1_fixed:
                continue
            if which == 2 and c2_fixed:
                continue
            mover = c1 if which == 1 else c2
            other = c2 if which == 1 else c1
            old_x, old_y = mover.x, mover.y

            # Build HPWL cache for this mover against the current
            # positions of every other component (including the partner).
            # Within this `which` iteration only mover moves, so the
            # cache is valid for all 16 candidate positions below.
            hpwl_cache = _build_mover_hpwl_cache(
                mover, comp_net_lookup, comp_map or {})

            # Pre-compute boundary slack for skip-if-in-bounds optimisation.
            # If mover is far enough from all boundaries that no trial
            # displacement can push it OOB, skip the per-candidate clamp.
            max_disp = max(distances) if distances else 0.0
            if interior_bbox:
                _bx_min = interior_bbox[0]; _bx_max = interior_bbox[2]
                _by_min = interior_bbox[1]; _by_max = interior_bbox[3]
            else:
                _bx_min = board.x_min; _bx_max = board.x_max
                _by_min = board.y_min; _by_max = board.y_max
            _mhw = mover.effective_width / 2.0
            _mhh = mover.effective_height / 2.0
            slack_x = min(old_x - _bx_min - _mhw, _bx_max - _mhw - old_x)
            slack_y = min(old_y - _by_min - _mhh, _by_max - _mhh - old_y)
            can_skip_clamp = (not keepouts and slack_x > max_disp and slack_y > max_disp)

            for dx_dir, dy_dir in directions:
                for dist in distances:
                    mover.x = old_x + dx_dir * dist
                    mover.y = old_y + dy_dir * dist
                    if not can_skip_clamp:
                        _enforce_boundary_single(mover, interior_bbox, board, keepouts=keepouts, extra_keepout_mm=0.0)

                    if mover.overlaps(other):
                        pair_ov = mover.overlap_area(other)
                    else:
                        pair_ov = 0.0

                    new_hpwl = _compute_local_hpwl_cached(
                        mover.x, mover.y, hpwl_cache)
                    third_ov = _third_party_overlaps(mover)
                    cost = beta * pair_ov + new_hpwl + new_ov_cost * third_ov
                    if best is None or cost < best[0] - 1e-9:
                        best = (cost, which, mover.x, mover.y)

            mover.x, mover.y = old_x, old_y

        if best is None:
            return

        _, which, new_x, new_y = best
        mover = c1 if which == 1 else c2
        mover_old_x, mover_old_y = mover.x, mover.y
        mover_old_rot = mover.rotation
        mover.x = new_x
        mover.y = new_y
        # Macro-aware: if mover is an IC, propagate move (translation +
        # rotation) to its caps. push_apart doesn't rotate, but using
        # propagate_ic_move is consistent and future-proof.
        propagate_ic_move(model, mover, mover_old_x, mover_old_y, mover_old_rot,
                          new_x, new_y, mover.rotation,
                          bounds=(interior_bbox[0], interior_bbox[1], interior_bbox[2], interior_bbox[3]) if interior_bbox else None)


def _nudge_caps_to_ics(
    model: BoardModel, rules: list,
    interior_bbox: tuple[float, float, float, float] | None = None,
    verbose: bool = False,
    cached_decap_map: dict | None = None,
    grid: SpatialGrid | None = None,
) -> None:
    board = model.board
    components = list(model.components)

    max_dist = 5.0
    for rule in rules:
        if rule.name == 'decoupling_proximity':
            max_dist = rule.params.get('max_distance_mm', 5.0)
            break
    else:
        return

    decap_map = cached_decap_map if cached_decap_map is not None else _build_decoupling_map(model)
    if not decap_map:
        return

    # Larger step sizes — Abacus can leave caps 50+mm from their ICs (same row
    # but opposite ends in X), and a 2mm max step would never close that gap
    # in 5 passes.  Fall back to smaller steps if a large jump is blocked.
    step_sizes = [16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.1]

    def _bbox_center_for_origin(cap, target_cx, target_cy, rotation):
        """Compute (cap.x, cap.y) so cap's bbox center sits at (target_cx, target_cy).

        Component bbox center is offset from origin by bbox_offset rotated by
        `rotation` — many ICs have substantial offsets (e.g., DIP packages with
        origin on pin 1 rather than body center).  Without this correction,
        spacing math based on cap.x/ic.x leaves bboxes overlapping.
        """
        rad = math.radians(rotation)
        cos_r = math.cos(rad)
        sin_r = math.sin(rad)
        # bbox center = (cap.x + ox*cos - oy*sin, cap.y + ox*sin + oy*cos)
        # => cap.x = target_cx - ox*cos + oy*sin
        new_x = target_cx - cap.bbox_offset_x * cos_r + cap.bbox_offset_y * sin_r
        new_y = target_cy - cap.bbox_offset_x * sin_r - cap.bbox_offset_y * cos_r
        return new_x, new_y

    def _check_position(cap, trial_x, trial_y, ic, components,
                        board, interior_bbox) -> tuple[float, float, float] | None:
        """Return (clamped_x, clamped_y, dist) if trial pos is overlap-free, else None.

        Mutates cap.x/cap.y temporarily for the overlap check, always restores.
        Includes the IC in the overlap check — caps must be ADJACENT to their
        IC (close but not overlapping), not sitting on top of it.
        """
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
        clamped_x = max(x_min, min(trial_x, x_max))
        clamped_y = max(y_min, min(trial_y, y_max))

        saved_x, saved_y = cap.x, cap.y
        cap.x = clamped_x
        cap.y = clamped_y
        try:
            has_overlap = any(
                cap.overlaps(other)
                for other in components
                if other is not cap and not other.is_fixed
            )
            if has_overlap:
                return None
            new_dist = math.hypot(ic.x - clamped_x, ic.y - clamped_y)
            return (clamped_x, clamped_y, new_dist)
        finally:
            cap.x = saved_x
            cap.y = saved_y

    # Phase 1: Direct jump to positions adjacent to IC.  This bypasses the
    # incremental nudge when the cap is far from the IC and the direct line
    # is blocked by other components.  Tries 8 candidate slots around the IC
    # (4 sides + 4 corners) at increasing radii until one fits.
    # Candidate positions are computed in BBOX space (not origin space) so
    # components with large bbox_offsets (e.g., DIP ICs with origin on pin 1)
    # don't end up with bboxes overlapping.
    for ic_ref, cap_refs in decap_map.items():
        ic = model.get_component(ic_ref)
        if not ic:
            continue

        ic_bbox = ic.bbox
        ic_bbox_cx = (ic_bbox[0] + ic_bbox[2]) / 2.0
        ic_bbox_cy = (ic_bbox[1] + ic_bbox[3]) / 2.0

        for cap_ref in cap_refs:
            cap = model.get_component(cap_ref)
            if not cap:
                continue
            if cap.is_fixed or cap.is_edge_connector:
                continue

            orig_dist = math.hypot(ic.x - cap.x, ic.y - cap.y)
            if orig_dist <= max_dist:
                continue

            cap_start_x, cap_start_y = cap.x, cap.y
            cap_start_rot = cap.rotation
            cap_nonsquare = _is_non_square(cap)

            rotations = [cap_start_rot]
            if cap_nonsquare:
                rotations.append((cap_start_rot + 90) % 360)

            best_jump: tuple[float, float, float, float] | None = None  # (x, y, rot, dist)

            for try_rot in rotations:
                if try_rot != cap_start_rot:
                    cap.set_rotation(try_rot)

                cap_half_w = cap.effective_width / 2.0
                cap_half_h = cap.effective_height / 2.0

                # Smallest spacing first — prefer tightest packing around IC.
                for spacing_mult in [1.0, 1.5, 2.0, 3.0, 5.0]:
                    gap = 0.5 * spacing_mult
                    # Target bbox-center positions: cap bbox sits `gap` mm
                    # beyond IC's bbox on the chosen side.
                    right_cx = ic_bbox[2] + gap + cap_half_w
                    left_cx = ic_bbox[0] - gap - cap_half_w
                    below_cy = ic_bbox[3] + gap + cap_half_h
                    above_cy = ic_bbox[1] - gap - cap_half_h

                    # Convert target bbox-center → cap origin (handles bbox_offset)
                    candidates_origin = [
                        _bbox_center_for_origin(cap, right_cx, ic_bbox_cy, try_rot),
                        _bbox_center_for_origin(cap, left_cx, ic_bbox_cy, try_rot),
                        _bbox_center_for_origin(cap, ic_bbox_cx, below_cy, try_rot),
                        _bbox_center_for_origin(cap, ic_bbox_cx, above_cy, try_rot),
                        _bbox_center_for_origin(cap, right_cx, below_cy, try_rot),
                        _bbox_center_for_origin(cap, left_cx, below_cy, try_rot),
                        _bbox_center_for_origin(cap, right_cx, above_cy, try_rot),
                        _bbox_center_for_origin(cap, left_cx, above_cy, try_rot),
                    ]

                    found_in_spacing = False
                    for cand_x, cand_y in candidates_origin:
                        result = _check_position(cap, cand_x, cand_y, ic,
                                                 components, board, interior_bbox)
                        if result is not None and result[2] < orig_dist:
                            cx, cy, cdist = result
                            if (best_jump is None or cdist < best_jump[3]
                                    or (cdist == best_jump[3]
                                        and try_rot == cap_start_rot
                                        and best_jump[2] != cap_start_rot)):
                                best_jump = (cx, cy, try_rot, cdist)
                            found_in_spacing = True
                            break
                    if found_in_spacing:
                        break

                cap.x, cap.y = cap_start_x, cap_start_y
                if try_rot != cap_start_rot:
                    cap.set_rotation(cap_start_rot)

            if best_jump is not None:
                cap.x = best_jump[0]
                cap.y = best_jump[1]
                if cap.rotation != best_jump[2]:
                    cap.set_rotation(best_jump[2])

    # Phase 2: Incremental nudge along direct line toward IC.
    # Catches caps that couldn't be reached by phase 1 (no overlap-free slot
    # adjacent to the IC).  Steps along the cap→IC unit vector.
    for pass_num in range(8):
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
                    continue

                dx = ic.x - cap.x
                dy = ic.y - cap.y
                if dx == 0 and dy == 0:
                    continue
                length = math.hypot(dx, dy)
                dx /= length
                dy /= length

                cap_start_x, cap_start_y = cap.x, cap.y
                cap_start_rot = cap.rotation
                cap_nonsquare = _is_non_square(cap)
                best_result: tuple[float, float, float, float] | None = None

                rotations = [cap_start_rot]
                if cap_nonsquare:
                    rotations.append((cap_start_rot + 90) % 360)

                for try_rot in rotations:
                    if try_rot != cap_start_rot:
                        cap.set_rotation(try_rot)

                    for step in step_sizes:
                        new_x = cap_start_x + dx * step
                        new_y = cap_start_y + dy * step

                        result = _check_position(cap, new_x, new_y, ic,
                                                 components, board, interior_bbox)
                        if result is not None and result[2] < dist:
                            cx, cy, cdist = result
                            if (best_result is None or cdist < best_result[3]
                                    or (cdist == best_result[3]
                                        and try_rot == cap_start_rot
                                        and best_result[2] != cap_start_rot)):
                                best_result = (cx, cy, try_rot, cdist)
                            break

                    cap.x, cap.y = cap_start_x, cap_start_y
                    if try_rot != cap_start_rot:
                        cap.set_rotation(cap_start_rot)

                if best_result is not None:
                    cap.x = best_result[0]
                    cap.y = best_result[1]
                    if cap.rotation != best_result[2]:
                        cap.set_rotation(best_result[2])
                    any_moved = True

        if not any_moved:
            break

    if verbose:
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


def _cleanup_cap_ic_overlaps(
    model: BoardModel,
    decap_map: dict | None,
    interior_bbox: tuple[float, float, float, float] | None = None,
    verbose: bool = False,
) -> None:
    """Move caps that overlap their assigned IC to the closest free adjacent slot.

    Edge-case safety net for the macro approach. The macro invariant
    preserves cap-IC offset through translation + rotation, but when an
    IC is clamped near a board edge the propagated cap position may end
    up overlapping the IC. This pass finds each such cap and relocates
    it to the nearest slot just outside the IC's bbox, checking overlap
    against *all* components (IC included). Leaves every other component
    where it is.

    The candidate search: 8 slots (4 sides + 4 corners) at spacings
    0.5→2.5 mm, smallest spacing first so the cap ends up packed tightly
    against the IC rather than drifting away.
    """
    if not decap_map:
        return

    board = model.board
    components = list(model.components)

    def _bbox_center_for_origin(cap, target_cx, target_cy, rotation):
        rad = math.radians(rotation)
        cos_r = math.cos(rad)
        sin_r = math.sin(rad)
        new_x = target_cx - cap.bbox_offset_x * cos_r + cap.bbox_offset_y * sin_r
        new_y = target_cy - cap.bbox_offset_x * sin_r - cap.bbox_offset_y * cos_r
        return new_x, new_y

    def _slot_overlap_free(cap, trial_x, trial_y, ic) -> bool:
        """Return True if cap at (trial_x, trial_y) overlaps nothing (IC included).

        Mutates cap.x/cap.y temporarily; always restores.
        Temporarily inflates the cap's courtyard by 0.3mm for the overlap
        check to account for bbox discrepancies between the in-memory model
        and the re-parsed .kicad_pcb file (rounding differences in
        bbox_offset can cause tiny ~0.1mm overlaps to appear after write).
        """
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
        if not (x_min <= trial_x <= x_max and y_min <= trial_y <= y_max):
            return False

        saved_x, saved_y = cap.x, cap.y
        saved_court = cap.courtyard_margin
        cap.x = trial_x
        cap.y = trial_y
        cap.courtyard_margin = saved_court + 0.3  # safety inflation
        try:
            for other in components:
                if other is cap or other.is_fixed:
                    continue
                if cap.overlaps(other):
                    return False
            return True
        finally:
            cap.x = saved_x
            cap.y = saved_y
            cap.courtyard_margin = saved_court

    moved = 0
    unresolved = 0
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
            # Check for overlap OR near-overlap. The model's bbox can differ
            # slightly from the re-parsed .kicad_pcb file's bbox (rounding
            # in bbox_offset), so a cap that's 0.1mm clear in the model
            # might overlap after re-parse. Treat any cap within 0.5mm of
            # its IC's bbox edge as "overlapping" and push it to a safe slot.
            if not cap.overlaps(ic):
                # Also check: is the cap within 0.5mm of the IC bbox?
                cap_bbox = cap.bbox
                ic_bbox_check = ic.bbox
                gap_x = min(abs(cap_bbox[0] - ic_bbox_check[2]),
                            abs(cap_bbox[2] - ic_bbox_check[0]))
                gap_y = min(abs(cap_bbox[1] - ic_bbox_check[3]),
                            abs(cap_bbox[3] - ic_bbox_check[1]))
                # If bboxes intersect in one axis, check the gap in the other
                intersects_x = cap_bbox[0] < ic_bbox_check[2] and cap_bbox[2] > ic_bbox_check[0]
                intersects_y = cap_bbox[1] < ic_bbox_check[3] and cap_bbox[3] > ic_bbox_check[1]
                near_overlap = ((intersects_x and gap_y < 0.5) or
                                (intersects_y and gap_x < 0.5) or
                                (gap_x < 0.5 and gap_y < 0.5))
                if not near_overlap:
                    continue

            ic_bbox = ic.bbox
            ic_bbox_cx = (ic_bbox[0] + ic_bbox[2]) / 2.0
            ic_bbox_cy = (ic_bbox[1] + ic_bbox[3]) / 2.0

            cap_start_x, cap_start_y = cap.x, cap.y
            cap_start_rot = cap.rotation
            rotations = [cap_start_rot]
            if _is_non_square(cap):
                rotations.append((cap_start_rot + 90) % 360)

            best: tuple[float, float, float, float] | None = None  # (x, y, rot, dist)

            for try_rot in rotations:
                if try_rot != cap_start_rot:
                    cap.set_rotation(try_rot)

                cap_half_w = cap.effective_width / 2.0
                cap_half_h = cap.effective_height / 2.0

                # Try spacings from tight (1.0mm gap) to wide (10mm gap).
                # Tight is preferred (cap close to IC); wide is the fallback
                # when near slots are all occupied by other components.
                # NOTE: minimum gap is 1.0mm (not 0.5mm) to account for bbox
                # discrepancies between the in-memory model and the re-parsed
                # .kicad_pcb file — rounding in bbox_offset can cause ~0.1mm
                # overlaps to appear after file write if the gap is too tight.
                for spacing_mult in [2.0, 3.0, 4.0, 5.0, 8.0, 12.0, 20.0]:
                    gap = 0.5 * spacing_mult
                    right_cx = ic_bbox[2] + gap + cap_half_w
                    left_cx = ic_bbox[0] - gap - cap_half_w
                    below_cy = ic_bbox[3] + gap + cap_half_h
                    above_cy = ic_bbox[1] - gap - cap_half_h

                    candidates_origin = [
                        _bbox_center_for_origin(cap, right_cx, ic_bbox_cy, try_rot),
                        _bbox_center_for_origin(cap, left_cx, ic_bbox_cy, try_rot),
                        _bbox_center_for_origin(cap, ic_bbox_cx, below_cy, try_rot),
                        _bbox_center_for_origin(cap, ic_bbox_cx, above_cy, try_rot),
                        _bbox_center_for_origin(cap, right_cx, below_cy, try_rot),
                        _bbox_center_for_origin(cap, left_cx, below_cy, try_rot),
                        _bbox_center_for_origin(cap, right_cx, above_cy, try_rot),
                        _bbox_center_for_origin(cap, left_cx, above_cy, try_rot),
                    ]

                    for cand_x, cand_y in candidates_origin:
                        if not _slot_overlap_free(cap, cand_x, cand_y, ic):
                            continue
                        dist = math.hypot(ic.x - cand_x, ic.y - cand_y)
                        if (best is None or dist < best[3]
                                or (dist == best[3]
                                    and try_rot == cap_start_rot
                                    and best[2] != cap_start_rot)):
                            best = (cand_x, cand_y, try_rot, dist)
                        break
                    if best is not None:
                        break

                cap.x, cap.y = cap_start_x, cap_start_y
                if try_rot != cap_start_rot:
                    cap.set_rotation(cap_start_rot)

            if best is not None:
                cap.x = best[0]
                cap.y = best[1]
                if cap.rotation != best[2]:
                    cap.set_rotation(best[2])
                moved += 1
            else:
                unresolved += 1

    if verbose and (moved or unresolved):
        print(f"  Cap-IC overlap cleanup: moved {moved} caps adjacent to their ICs"
              + (f", {unresolved} still overlapping (no free adjacent slot)" if unresolved else ""))


def _cleanup_any_cap_ic_overlaps(
    model: BoardModel,
    interior_bbox: tuple[float, float, float, float] | None = None,
    verbose: bool = False,
) -> None:
    """Move any capacitor that overlaps ANY IC to a free adjacent slot.

    Unlike ``_cleanup_cap_ic_overlaps`` (which only handles caps assigned
    to an IC via the decoupling_proximity rule), this pass catches caps
    that overlap an IC they're NOT decoupling — e.g. a +5V bulk cap that
    drifted on top of an FPGA during SA, but shares no power rail with
    it so it's not in the decap map.

    The algorithm is the same 8-direction slot search as
    ``_cleanup_cap_ic_overlaps``, but the "IC" is whichever IC the cap
    currently overlaps (not necessarily its assigned IC).
    """
    board = model.board
    components = list(model.components)
    ic_types = {"ic", "mcu", "regulator"}
    caps = [c for c in components
            if getattr(c, "component_type", "") == "capacitor"
            and not c.is_fixed and not c.is_edge_connector]
    ics = [c for c in components if getattr(c, "component_type", "") in ic_types]

    if not caps or not ics:
        return

    def _bbox_center_for_origin(cap, target_cx, target_cy, rotation):
        rad = math.radians(rotation)
        cos_r = math.cos(rad)
        sin_r = math.sin(rad)
        new_x = target_cx - cap.bbox_offset_x * cos_r + cap.bbox_offset_y * sin_r
        new_y = target_cy - cap.bbox_offset_x * sin_r - cap.bbox_offset_y * cos_r
        return new_x, new_y

    def _slot_overlap_free(cap, trial_x, trial_y) -> bool:
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
        if not (x_min <= trial_x <= x_max and y_min <= trial_y <= y_max):
            return False
        saved_x, saved_y = cap.x, cap.y
        cap.x = trial_x
        cap.y = trial_y
        try:
            for other in components:
                if other is cap or other.is_fixed:
                    continue
                if cap.overlaps(other):
                    return False
            return True
        finally:
            cap.x = saved_x
            cap.y = saved_y

    moved = 0
    unresolved = 0
    for cap in caps:
        # Find any IC this cap overlaps.
        overlapping_ic = None
        for ic in ics:
            if cap.overlaps(ic):
                overlapping_ic = ic
                break
        if overlapping_ic is None:
            continue

        ic = overlapping_ic
        ic_bbox = ic.bbox
        ic_bbox_cx = (ic_bbox[0] + ic_bbox[2]) / 2.0
        ic_bbox_cy = (ic_bbox[1] + ic_bbox[3]) / 2.0

        cap_start_x, cap_start_y = cap.x, cap.y
        cap_start_rot = cap.rotation
        rotations = [cap_start_rot]
        if _is_non_square(cap):
            rotations.append((cap_start_rot + 90) % 360)

        best: tuple[float, float, float, float] | None = None

        for try_rot in rotations:
            if try_rot != cap_start_rot:
                cap.set_rotation(try_rot)

            cap_half_w = cap.effective_width / 2.0
            cap_half_h = cap.effective_height / 2.0

            for spacing_mult in [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]:
                gap = 0.5 * spacing_mult
                right_cx = ic_bbox[2] + gap + cap_half_w
                left_cx = ic_bbox[0] - gap - cap_half_w
                below_cy = ic_bbox[3] + gap + cap_half_h
                above_cy = ic_bbox[1] - gap - cap_half_h

                candidates_origin = [
                    _bbox_center_for_origin(cap, right_cx, ic_bbox_cy, try_rot),
                    _bbox_center_for_origin(cap, left_cx, ic_bbox_cy, try_rot),
                    _bbox_center_for_origin(cap, ic_bbox_cx, below_cy, try_rot),
                    _bbox_center_for_origin(cap, ic_bbox_cx, above_cy, try_rot),
                    _bbox_center_for_origin(cap, right_cx, below_cy, try_rot),
                    _bbox_center_for_origin(cap, left_cx, below_cy, try_rot),
                    _bbox_center_for_origin(cap, right_cx, above_cy, try_rot),
                    _bbox_center_for_origin(cap, left_cx, above_cy, try_rot),
                ]

                for cand_x, cand_y in candidates_origin:
                    if not _slot_overlap_free(cap, cand_x, cand_y):
                        continue
                    dist = math.hypot(ic.x - cand_x, ic.y - cand_y)
                    if (best is None or dist < best[3]
                            or (dist == best[3]
                                and try_rot == cap_start_rot
                                and best[2] != cap_start_rot)):
                        best = (cand_x, cand_y, try_rot, dist)
                    break
                if best is not None:
                    break

            cap.x, cap.y = cap_start_x, cap_start_y
            if try_rot != cap_start_rot:
                cap.set_rotation(cap_start_rot)

        if best is not None:
            cap.x = best[0]
            cap.y = best[1]
            if cap.rotation != best[2]:
                cap.set_rotation(best[2])
            moved += 1
        else:
            unresolved += 1

    if verbose and (moved or unresolved):
        print(f"  Any-cap-IC overlap cleanup: moved {moved} caps off non-assigned ICs"
              + (f", {unresolved} still overlapping (no free adjacent slot)" if unresolved else ""))


def _resolve_large_overlaps(
    model: BoardModel,
    interior_bbox: tuple[float, float, float, float] | None,
    grid: SpatialGrid,
    *,
    verbose: bool = False,
    min_overlap_area: float = 5.0,
) -> None:
    """Emergency resolver for large stuck overlaps.

    Handles three cases that greedy push-apart can't resolve:
      1. IC↔IC overlaps (both components large, no room to push apart)
      2. Large cap↔IC overlaps (cap drifted onto a non-assigned IC
         during SA, too big for the 8-direction slot search)
      3. Cap↔cap clusters (multiple caps piled at the same spot)

    For each overlap pair with area >= min_overlap_area, picks the
    smaller component, searches the interior bbox on a coarse grid for
    the largest open slot that fits it, and teleports it there.
    Ignores HPWL — a legal placement with bad HPWL is strictly better
    than an illegal one.
    """
    board = model.board
    ic_types = {"ic", "mcu", "regulator"}

    # MACRO-AWARE: skip macro member caps — they follow their IC leader.
    from engine.group_moves import get_macro_member_refs
    macro_refs = get_macro_member_refs(model)

    if interior_bbox is not None:
        rx_min, ry_min, rx_max, ry_max = interior_bbox
    else:
        rx_min, ry_min, rx_max, ry_max = (
            board.x_min, board.y_min, board.x_max, board.y_max,
        )

    components = list(model.components)

    def _slot_overlap_free(comp: Component, trial_x: float, trial_y: float,
                           exclude: set[int] | None = None) -> bool:
        """Return True if comp at (trial_x, trial_y) overlaps nothing."""
        half_w = comp.effective_width / 2.0
        half_h = comp.effective_height / 2.0
        x_min = rx_min + half_w
        x_max = rx_max - half_w
        y_min = ry_min + half_h
        y_max = ry_max - half_h
        if not (x_min <= trial_x <= x_max and y_min <= trial_y <= y_max):
            return False
        saved_x, saved_y = comp.x, comp.y
        comp.x = trial_x
        comp.y = trial_y
        try:
            for other in components:
                if other is comp or other.is_fixed:
                    continue
                if exclude is not None and id(other) in exclude:
                    continue
                if comp.overlaps(other):
                    return False
            return True
        finally:
            comp.x = saved_x
            comp.y = saved_y

    # Build decap map for cap-IC group movement
    rules = getattr(model, 'active_rules', None) or []
    decap_map: dict[str, list[str]] = {}
    if rules:
        try:
            decap_map = _build_decoupling_map(model)
        except Exception:
            pass

    # Find all overlap pairs with area >= min_overlap_area
    overlap_pairs = []
    for i, a in enumerate(components):
        if a.is_fixed or a.is_edge_connector:
            continue
        if a.ref in macro_refs:
            continue
        for b in components[i+1:]:
            if b.is_fixed or b.is_edge_connector:
                continue
            if b.ref in macro_refs:
                continue
            if not a.overlaps(b):
                continue
            area = a.overlap_area(b)
            if area >= min_overlap_area:
                overlap_pairs.append((area, a, b))

    overlap_pairs.sort(key=lambda x: -x[0])  # largest first

    moved = 0
    unresolved = 0
    for area, a, b in overlap_pairs:
        if not a.overlaps(b):
            continue  # may have been resolved by a previous move

        # Pick the smaller component to move
        area_a = a.effective_width * a.effective_height
        area_b = b.effective_width * b.effective_height
        mover = a if area_a <= area_b else b
        stayer = b if mover is a else a

        # Build the mover's group (itself + its decoupling caps if it's an IC)
        mover_group = {mover}
        if getattr(mover, 'component_type', '') in ic_types and decap_map:
            for cap_ref in decap_map.get(mover.ref, []):
                cap = model.get_component(cap_ref)
                if cap and not cap.is_fixed:
                    mover_group.add(cap)

        # Search for a free slot on a coarse grid.
        # Try multiple step sizes — start coarse (fast), refine if no slot found.
        half_w = mover.effective_width / 2.0
        half_h = mover.effective_height / 2.0

        best_slot: tuple[float, float] | None = None
        best_dist = float("inf")
        orig_x, orig_y = mover.x, mover.y

        exclude_ids = {id(c) for c in mover_group}

        # Try step sizes: coarse (mover_size + 2mm) → medium (+1mm) → fine (+0.5mm)
        for step_mult in [2.0, 1.0, 0.5]:
            step = max(mover.effective_width, mover.effective_height) * step_mult + 0.5
            if step < 1.0:
                step = 1.0
            x = rx_min + half_w
            while x <= rx_max - half_w:
                y = ry_min + half_h
                while y <= ry_max - half_h:
                    if _slot_overlap_free(mover, x, y, exclude=exclude_ids):
                        dist = math.hypot(x - orig_x, y - orig_y)
                        if dist < best_dist:
                            best_dist = dist
                            best_slot = (x, y)
                    y += step
                x += step
            if best_slot is not None:
                break  # found a slot, no need to refine

        if best_slot is not None:
            dx = best_slot[0] - mover.x
            dy = best_slot[1] - mover.y
            for comp in mover_group:
                new_x = comp.x + dx
                new_y = comp.y + dy
                cw = comp.effective_width / 2.0
                ch = comp.effective_height / 2.0
                new_x = max(rx_min + cw, min(new_x, rx_max - cw))
                new_y = max(ry_min + ch, min(new_y, ry_max - ch))
                comp.x = new_x
                comp.y = new_y
            moved += 1
            if verbose:
                print(f"  Large-overlap emergency: moved {mover.ref} "
                      f"({mover.effective_width:.1f}x{mover.effective_height:.1f}mm) "
                      f"by ({dx:.1f},{dy:.1f})mm to resolve {area:.1f}mm² overlap with {stayer.ref}")
        else:
            # Last resort: try moving just the mover alone (not its caps).
            # This breaks the cap-IC atomic group but resolves the overlap.
            if len(mover_group) > 1:
                best_slot_solo = None
                best_dist_solo = float("inf")
                for step_mult in [2.0, 1.0, 0.5]:
                    step = max(mover.effective_width, mover.effective_height) * step_mult + 0.5
                    if step < 1.0:
                        step = 1.0
                    x = rx_min + half_w
                    while x <= rx_max - half_w:
                        y = ry_min + half_h
                        while y <= ry_max - half_h:
                            # Solo: only exclude the mover itself
                            if _slot_overlap_free(mover, x, y, exclude={id(mover)}):
                                dist = math.hypot(x - orig_x, y - orig_y)
                                if dist < best_dist_solo:
                                    best_dist_solo = dist
                                    best_slot_solo = (x, y)
                            y += step
                        x += step
                    if best_slot_solo is not None:
                        break
                if best_slot_solo is not None:
                    mover.x = max(rx_min + half_w, min(best_slot_solo[0], rx_max - half_w))
                    mover.y = max(ry_min + half_h, min(best_slot_solo[1], ry_max - half_h))
                    moved += 1
                    if verbose:
                        print(f"  Large-overlap emergency (solo): moved {mover.ref} "
                              f"to ({mover.x:.1f},{mover.y:.1f}) to resolve {area:.1f}mm² overlap with {stayer.ref}")
                    continue
            unresolved += 1
            if verbose:
                print(f"  Large-overlap emergency: could not find free slot for "
                      f"{mover.ref} (overlaps {stayer.ref}, area={area:.1f}mm²)")

    if verbose and (moved or unresolved):
        print(f"  Large-overlap emergency: {moved} moved, {unresolved} unresolved")


# Backward-compatible alias
_resolve_ic_ic_overlaps = _resolve_large_overlaps

def _count_overlaps(model: BoardModel) -> int:
    count, _ = _compute_overlap_stats(model)
    return count


def _count_oob(model: BoardModel) -> int:
    count = 0
    board = model.board
    for comp in model.components:
        if comp.is_edge_connector:
            continue
        x1, y1, x2, y2 = comp.bbox
        if x1 < board.x_min or x2 > board.x_max or y1 < board.y_min or y2 > board.y_max:
            count += 1
    return count


# =============================================================================
# Adaptive bbox expansion + anti-centroid spread (plan.md §3-4)
# =============================================================================

def _maybe_expand_bbox(
    model: BoardModel,
    interior_bbox: tuple[float, float, float, float] | None,
    board: BoardOutline,
    *,
    factor: float = 0.05,
    density_threshold: float = 0.75,
    margin: float = 1.0,
    verbose: bool = False,
) -> tuple[float, float, float, float] | None:
    """Grow the working interior bbox when push-apart has stalled.

    Plan.md §3: when push-apart plateaus with overlaps remaining and the
    interior components occupy less than ``density_threshold`` of the
    board area, expand the interior bbox by ``factor`` (default +5%)
    toward the board outline, clamped so the bbox never crosses the
    outline minus ``margin``.

    Returns the new bbox tuple if expansion was performed, or ``None``
    if expansion was not warranted (already dense, or no room to grow).
    """
    if interior_bbox is None:
        return None

    bx_min, by_min, bx_max, by_max = interior_bbox
    bbox_w = max(0.0, bx_max - bx_min)
    bbox_h = max(0.0, by_max - by_min)
    bbox_area = bbox_w * bbox_h
    board_area = max(1.0, board.width * board.height)

    # Density = used component area / bbox area (how packed the current
    # bbox is). We then check whether the bbox itself is small relative
    # to the board (i.e. there's room to grow without crossing the
    # board outline).
    used_area = 0.0
    for c in model.components:
        if c.is_fixed or c.is_edge_connector:
            continue
        used_area += c.effective_width * c.effective_height

    bbox_density = used_area / max(1.0, bbox_area)
    board_density = bbox_area / board_area

    if verbose:
        print(f"  bbox expand check: bbox_density={bbox_density:.2f} "
              f"bbox/board={board_density:.2f} (threshold={density_threshold})")

    # Only expand if the current bbox is small relative to the board
    # (i.e. there's headroom toward the outline) AND the bbox is
    # actually packed (otherwise we'd just spread empty space).
    if board_density >= density_threshold:
        return None
    if bbox_density < 0.30:
        # Not actually packed — expansion won't help.
        return None

    # Compute expansion delta = factor * board dimension, clamped so we
    # never cross (board outline - margin).
    expand_x = factor * board.width
    expand_y = factor * board.height

    new_x_min = max(board.x_min + margin, bx_min - expand_x)
    new_y_min = max(board.y_min + margin, by_min - expand_y)
    new_x_max = min(board.x_max - margin, bx_max + expand_x)
    new_y_max = min(board.y_max - margin, by_max + expand_y)

    # No movement → no expansion.
    if (new_x_min >= bx_min - 1e-6 and new_y_min >= by_min - 1e-6
            and new_x_max <= bx_max + 1e-6 and new_y_max <= by_max + 1e-6):
        if verbose:
            print("  bbox expand: no room to grow (already at board outline)")
        return None

    new_bbox = (new_x_min, new_y_min, new_x_max, new_y_max)
    if verbose:
        print(f"  bbox expand: ({bx_min:.1f},{by_min:.1f},{bx_max:.1f},{by_max:.1f}) "
              f"-> ({new_x_min:.1f},{new_y_min:.1f},{new_x_max:.1f},{new_y_max:.1f})")
    return new_bbox


def _spread_pass(
    model: BoardModel,
    interior_bbox: tuple[float, float, float, float] | None,
    board: BoardOutline,
    *,
    verbose: bool = False,
    max_step_fraction: float = 0.10,
) -> None:
    """Anti-centroid spread pass (plan.md §4).

    After push-apart converges, components often remain clustered in
    the board's center because push-apart only resolves *overlaps* —
    it has no incentive to use underutilized board regions. This pass:

      1. Computes component density per quadrant (4 quadrants + center).
         When the interior bbox is offset from the board center, we use
         the interior bbox's quadrants (since components are clamped to
         the bbox, using board quadrants would attempt moves the clamp
         would block, creating overlaps without spread).
      2. Identifies the densest and sparsest quadrants.
      3. Applies a gentle force vector to each movable component in the
         densest quadrant, pulling it toward the sparsest quadrant's
         centroid. The force is scaled by the density gap so a near-
         uniform distribution isn't disturbed.

    The pass is a single iteration — no recursion. Components are
    clamped to the interior bbox (or board outline if no bbox) so they
    can't escape the legal region. This intentionally does NOT enforce
    non-overlap; the legalizer's greedy resolver runs afterward to
    clean up any overlaps the spread creates.
    """
    movable = [
        c for c in model.components
        if not c.is_fixed and not c.is_edge_connector
    ]
    if len(movable) < 4:
        return  # not enough components to bother

    # Clamp region (interior bbox if given, else board with margin).
    if interior_bbox is not None:
        rx_min, ry_min, rx_max, ry_max = interior_bbox
    else:
        rx_min, ry_min, rx_max, ry_max = (
            board.x_min, board.y_min, board.x_max, board.y_max,
        )

    cx = (rx_min + rx_max) / 2.0
    cy = (ry_min + ry_max) / 2.0
    half_w = (rx_max - rx_min) / 2.0
    half_h = (ry_max - ry_min) / 2.0

    # 5 regions: TL, TR, BL, BR, Center.
    regions = {
        "TL": {"count": 0, "sx": 0.0, "sy": 0.0},
        "TR": {"count": 0, "sx": 0.0, "sy": 0.0},
        "BL": {"count": 0, "sx": 0.0, "sy": 0.0},
        "BR": {"count": 0, "sx": 0.0, "sy": 0.0},
        "C":  {"count": 0, "sx": 0.0, "sy": 0.0},
    }
    for c in movable:
        # Center band: within 10% of region half-dim from region center.
        if (abs(c.x - cx) < 0.10 * half_w) and (abs(c.y - cy) < 0.10 * half_h):
            key = "C"
        elif c.x < cx and c.y < cy:
            key = "BL"
        elif c.x < cx and c.y >= cy:
            key = "TL"
        elif c.x >= cx and c.y < cy:
            key = "BR"
        else:
            key = "TR"
        regions[key]["count"] += 1
        regions[key]["sx"] += c.x
        regions[key]["sy"] += c.y

    # Compute density (count / total movable) per region.
    total = float(len(movable))
    for r in regions.values():
        r["density"] = r["count"] / total
        if r["count"] > 0:
            r["centroid"] = (r["sx"] / r["count"], r["sy"] / r["count"])
        else:
            r["centroid"] = (cx, cy)

    # Find densest non-empty and sparsest regions.
    sorted_regions = sorted(regions.items(), key=lambda kv: kv[1]["density"])
    sparsest_name, sparsest = sorted_regions[0]
    densest_name, densest = sorted_regions[-1]

    # If distribution is already near-uniform, do nothing.
    density_gap = densest["density"] - sparsest["density"]
    if density_gap < 0.10:
        if verbose:
            print(f"  spread pass: distribution near-uniform "
                  f"(gap={density_gap:.2f}), skipping")
        return

    # If sparsest region is empty, target the sparsest quadrant's
    # geometric centroid (rather than (0,0) which would pull toward a
    # corner).
    if sparsest["count"] == 0:
        region_geometric = {
            "TL": (rx_min + (rx_max - rx_min) * 0.25, ry_min + (ry_max - ry_min) * 0.75),
            "TR": (rx_min + (rx_max - rx_min) * 0.75, ry_min + (ry_max - ry_min) * 0.75),
            "BL": (rx_min + (rx_max - rx_min) * 0.25, ry_min + (ry_max - ry_min) * 0.25),
            "BR": (rx_min + (rx_max - rx_min) * 0.75, ry_min + (ry_max - ry_min) * 0.25),
            "C":  (cx, cy),
        }
        target_x, target_y = region_geometric[sparsest_name]
    else:
        target_x, target_y = sparsest["centroid"]

    # Apply gentle pull to components in the densest region.
    # Step size is capped at max_step_fraction of the region dimension so
    # we don't yank components across the region.
    max_step = max_step_fraction * max(rx_max - rx_min, ry_max - ry_min)
    step = max_step * density_gap  # scale by how imbalanced the region is
    step = max(0.5, min(max_step, step))

    moved = 0
    for c in movable:
        # Determine which region this component is in.
        if (abs(c.x - cx) < 0.10 * half_w
                and abs(c.y - cy) < 0.10 * half_h):
            in_region = "C"
        elif c.x < cx and c.y < cy:
            in_region = "BL"
        elif c.x < cx and c.y >= cy:
            in_region = "TL"
        elif c.x >= cx and c.y < cy:
            in_region = "BR"
        else:
            in_region = "TR"

        if in_region != densest_name:
            continue

        # Pull toward sparsest region's centroid.
        dx = target_x - c.x
        dy = target_y - c.y
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            continue
        ux, uy = dx / dist, dy / dist

        # Cap the move so we don't overshoot into the sparsest region.
        move = min(step, dist * 0.5)

        new_x = c.x + ux * move
        new_y = c.y + uy * move

        # Clamp to legal region.
        half_w_c = c.effective_width / 2.0
        half_h_c = c.effective_height / 2.0
        new_x = max(rx_min + half_w_c, min(new_x, rx_max - half_w_c))
        new_y = max(ry_min + half_h_c, min(new_y, ry_max - half_h_c))

        if abs(new_x - c.x) > 1e-3 or abs(new_y - c.y) > 1e-3:
            c.x = new_x
            c.y = new_y
            moved += 1

    if verbose and moved:
        print(f"  spread pass: moved {moved} components from {densest_name} "
              f"(density={densest['density']:.2f}) toward {sparsest_name} "
              f"(density={sparsest['density']:.2f}, step={step:.2f}mm)")
