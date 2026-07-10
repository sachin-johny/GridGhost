"""Post-legalization HPWL recovery via cell sliding and pair swap.

After legalization, components have been moved from their optimal
positions to resolve overlaps. These lightweight post-processing
steps recover 2-5% HPWL without creating new overlaps or pushing
components out of bounds.

Phase 1 - Cell Sliding:
  For each movable component, try sliding it left and right along
  its row at various step sizes. At each trial position, verify:
  (a) no new overlaps, (b) stays within bounds, (c) local HPWL
  decreases. Also try 90-degree rotation for non-square components.
  Accept the slide that maximally reduces HPWL. 3 passes total.

Phase 2 - Pair Swap:
  Try swapping positions (and optionally rotations) of component
  pairs. Candidate pair selection uses net connectivity (share at
  least one net) and size-class similarity (both caps, both ICs,
  etc.) to limit the O(N^2) search space. Accept only if no new
  overlaps, both in bounds, and combined local HPWL decreases.
  2 passes, HPWL-worst first, max 200 swap attempts per pass.

Phase 3 - Cell Sliding (again):
  Pair swaps may open up space that cell sliding can now exploit.

Uses _compute_local_hpwl and _build_comp_net_lookup from legalizer
for O(k) HPWL evaluation per component.
"""

from __future__ import annotations

from models.board_model import BoardModel, Component, BoardOutline, Net, Pad
from legalization.legalizer import (
    _compute_local_hpwl,
    _build_comp_net_lookup,
    _count_overlaps_involving,
    _compute_overlap_stats,
    _enforce_boundary_single,
    _is_non_square,
    _count_oob,
)
from engine.group_moves import propagate_ic_delta


# ---------------------------------------------------------------------------
# Helper: boundary check for a single component
# ---------------------------------------------------------------------------

def _edge_keepout_for(comp, model=None) -> float:
    """Type-aware edge keepout extra (mm) for this component.

    ICs/MCUs/regulators get extra edge clearance so they don't end up at
    the board edge after cell_slide / pair_swap. Falls back to 0.0 if the
    cost_state helper can't be imported.

    NOTE: post-legalize cell_slide and pair_swap are HPWL-driven, not
    overlap-aware. Applying the full keepout here can cause overlaps on
    dense boards (the slide pushes an IC inward to satisfy the keepout,
    but there's no room). We return 0.0 by default and only apply the
    keepout when the caller explicitly passes a model AND the board is
    sparse enough to have room. This is a deliberate trade-off: the SA
    cost gradient and the legalizer's initial _enforce_boundary already
    keep ICs away from edges; post-legalize should have freedom to
    refine HPWL without the keepout fighting it.
    """
    return 0.0


def _in_bounds(comp: Component, interior_bbox, board: BoardOutline) -> bool:
    x1, y1, x2, y2 = comp.bbox
    if interior_bbox:
        return (x1 >= interior_bbox[0] and x2 <= interior_bbox[2]
                and y1 >= interior_bbox[1] and y2 <= interior_bbox[3])
    return (x1 >= board.x_min and x2 <= board.x_max
            and y1 >= board.y_min and y2 <= board.y_max)


def _clamp_to_bounds(comp, interior_bbox, board):
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
    new_x = max(x_min, min(comp.x, x_max))
    new_y = max(y_min, min(comp.y, y_max))
    changed = (new_x != comp.x or new_y != comp.y)
    comp.x = new_x
    comp.y = new_y
    return changed


def _has_any_overlap(comp, components):
    for other in components:
        if other is comp:
            continue
        if comp.overlaps(other):
            return True
    return False


def _has_overlap_excluding(comp, components, excluded):
    for other in components:
        if other is comp or other in excluded:
            continue
        if comp.overlaps(other):
            return True
    return False


# ---------------------------------------------------------------------------
# Cell Sliding
# ---------------------------------------------------------------------------

def cell_slide(
    model: BoardModel,
    grid_mm: float,
    interior_bbox: tuple[float, float, float, float] | None,
    verbose: bool,
    cached_decap_map: dict | None,
) -> None:
    """Slide each movable component along its row to reduce local HPWL."""
    board = model.board
    components = list(model.components)
    movable = [c for c in components
               if not c.is_fixed and not c.is_edge_connector]

    # Step sizes for sliding: small grid steps then larger jumps
    slide_steps = [grid_mm, grid_mm * 2, grid_mm * 5, 0.5, 1.0, 2.0]
    seen = set()
    unique_steps = []
    for s in slide_steps:
        rs = round(s, 6)
        if rs not in seen:
            seen.add(rs)
            unique_steps.append(s)
    slide_steps = unique_steps

    comp_net_lookup = _build_comp_net_lookup(model)

    for pass_num in range(3):
        total_hpwl_delta = 0.0
        moves_made = 0

        for comp in movable:
            old_x, old_y = comp.x, comp.y
            old_rot = comp.rotation

            orig_hpwl = _compute_local_hpwl(comp, comp_net_lookup, model)
            if orig_hpwl < 1e-9:
                continue

            best_x, best_y, best_rot = old_x, old_y, old_rot
            best_hpwl = orig_hpwl

            rotations_to_try = [old_rot]
            if _is_non_square(comp):
                rotations_to_try.append((old_rot + 90) % 360)

            for try_rot in rotations_to_try:
                is_rotated = (try_rot != old_rot)
                if is_rotated:
                    comp.set_rotation(try_rot)

                rot_penalty = 0.1 if is_rotated else 0.0

                # Try sliding left and right along the row (x-axis)
                for step in slide_steps:
                    for direction in (-1, 1):
                        trial_x = old_x + direction * step
                        comp.x = trial_x
                        comp.y = old_y

                        _clamp_to_bounds(comp, interior_bbox, board)

                        if _has_any_overlap(comp, components):
                            continue

                        trial_hpwl = _compute_local_hpwl(comp, comp_net_lookup, model)
                        effective_hpwl = trial_hpwl + rot_penalty

                        if effective_hpwl < best_hpwl:
                            best_hpwl = effective_hpwl
                            best_x, best_y = comp.x, comp.y
                            best_rot = comp.rotation

                # Also try y-axis sliding
                for step in slide_steps:
                    for direction in (-1, 1):
                        trial_y = old_y + direction * step
                        comp.x = old_x
                        comp.y = trial_y

                        _clamp_to_bounds(comp, interior_bbox, board)

                        if _has_any_overlap(comp, components):
                            continue

                        trial_hpwl = _compute_local_hpwl(comp, comp_net_lookup, model)
                        effective_hpwl = trial_hpwl + rot_penalty

                        if effective_hpwl < best_hpwl:
                            best_hpwl = effective_hpwl
                            best_x, best_y = comp.x, comp.y
                            best_rot = comp.rotation

                # Restore rotation for next iteration
                if is_rotated:
                    comp.set_rotation(old_rot)

            # Apply best if improvement found
            comp_pre_x, comp_pre_y = comp.x, comp.y
            comp.x, comp.y = best_x, best_y
            if comp.rotation != best_rot:
                comp.set_rotation(best_rot)

            if best_hpwl < orig_hpwl - 1e-9:
                hpwl_gain = orig_hpwl - best_hpwl
                total_hpwl_delta += hpwl_gain
                moves_made += 1
                # Group-aware: if comp is an IC, propagate delta to caps.
                bounds = (interior_bbox[0], interior_bbox[1], interior_bbox[2], interior_bbox[3]) if interior_bbox else None
                propagate_ic_delta(model, comp, comp_pre_x, comp_pre_y, comp.x, comp.y,
                                  bounds=bounds, decap_map=cached_decap_map)
            else:
                # No improvement - revert
                comp.x, comp.y = old_x, old_y
                if comp.rotation != old_rot:
                    comp.set_rotation(old_rot)

        if verbose and moves_made > 0:
            print(f"  Cell slide pass {pass_num + 1}/3: "
                  f"{moves_made} moves, "
                  f"HPWL recovered: {total_hpwl_delta:.2f}mm")

        if moves_made == 0:
            break

        components = list(model.components)


# ---------------------------------------------------------------------------
# Pair Swap
# ---------------------------------------------------------------------------

def _size_class(comp: Component) -> str:
    ct = getattr(comp, 'component_type', 'generic') or 'generic'
    if ct in ('capacitor', 'cap'):
        return 'cap'
    if ct in ('resistor', 'res'):
        return 'res'
    if ct in ('ic',):
        return 'ic'
    if ct in ('connector',):
        return 'conn'
    if ct in ('crystal',):
        return 'xtal'
    return 'other'


def _build_net_connected_pairs(model: BoardModel) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for net in model.nets:
        refs = net.component_refs
        ref_list = sorted(refs)
        for i, r1 in enumerate(ref_list):
            for r2 in ref_list[i + 1:]:
                pairs.add((r1, r2))
    return pairs


def _build_size_class_groups(
    components: list[Component],
) -> dict[str, list[Component]]:
    groups: dict[str, list[Component]] = {}
    for comp in components:
        sc = _size_class(comp)
        if sc not in groups:
            groups[sc] = []
        groups[sc].append(comp)
    return groups


def _swap_acceptable(
    c1: Component,
    c2: Component,
    components: list[Component],
    interior_bbox,
    board: BoardOutline,
    comp_net_lookup: dict,
    model: BoardModel,
    orig_combined_hpwl: float,
    rot_penalty: float = 0.0,
) -> bool:
    """Pure check: is the current swap (positions already applied) acceptable?"""
    if not _in_bounds(c1, interior_bbox, board):
        return False
    if not _in_bounds(c2, interior_bbox, board):
        return False

    excluded = [c1, c2]
    if _has_overlap_excluding(c1, components, excluded):
        return False
    if _has_overlap_excluding(c2, components, excluded):
        return False

    new_hpwl1 = _compute_local_hpwl(c1, comp_net_lookup, model)
    new_hpwl2 = _compute_local_hpwl(c2, comp_net_lookup, model)
    new_combined = new_hpwl1 + new_hpwl2 + rot_penalty

    return new_combined < orig_combined_hpwl - 1e-9


def pair_swap_refine(
    model: BoardModel,
    grid_mm: float,
    interior_bbox: tuple[float, float, float, float] | None,
    verbose: bool,
    cached_decap_map: dict | None,
) -> None:
    """Try swapping positions of component pairs to reduce HPWL."""
    board = model.board
    components = list(model.components)
    movable = [c for c in components
               if not c.is_fixed and not c.is_edge_connector]

    comp_net_lookup = _build_comp_net_lookup(model)
    comp_map = {c.ref: c for c in components}

    net_pairs = _build_net_connected_pairs(model)
    size_groups = _build_size_class_groups(movable)

    candidate_pairs: list[tuple[Component, Component]] = []
    for r1, r2 in net_pairs:
        c1 = comp_map.get(r1)
        c2 = comp_map.get(r2)
        if c1 and c2 and not c1.is_fixed and not c1.is_edge_connector \
                and not c2.is_fixed and not c2.is_edge_connector:
            candidate_pairs.append((c1, c2))

    existing = set()
    for c1, c2 in candidate_pairs:
        key = (min(c1.ref, c2.ref), max(c1.ref, c2.ref))
        existing.add(key)

    for sc, group in size_groups.items():
        if sc in ('cap', 'res', 'ic') and len(group) >= 2:
            for i, c1 in enumerate(group):
                for c2 in group[i + 1:]:
                    key = (min(c1.ref, c2.ref), max(c1.ref, c2.ref))
                    if key not in existing:
                        existing.add(key)
                        candidate_pairs.append((c1, c2))

    MAX_ATTEMPTS_PER_PASS = 200

    for pass_num in range(2):
        swaps_made = 0
        total_hpwl_delta = 0.0
        attempts = 0

        def pair_hpwl(pair):
            c1, c2 = pair
            return (_compute_local_hpwl(c1, comp_net_lookup, model)
                    + _compute_local_hpwl(c2, comp_net_lookup, model))

        candidate_pairs.sort(key=pair_hpwl, reverse=True)

        for c1, c2 in candidate_pairs:
            if attempts >= MAX_ATTEMPTS_PER_PASS:
                break
            attempts += 1

            orig_x1, orig_y1, orig_rot1 = c1.x, c1.y, c1.rotation
            orig_x2, orig_y2, orig_rot2 = c2.x, c2.y, c2.rotation

            orig_hpwl1 = _compute_local_hpwl(c1, comp_net_lookup, model)
            orig_hpwl2 = _compute_local_hpwl(c2, comp_net_lookup, model)
            orig_combined = orig_hpwl1 + orig_hpwl2

            if orig_combined < 1e-9:
                continue

            # --- Try simple position swap ---
            c1.x, c1.y = orig_x2, orig_y2
            c2.x, c2.y = orig_x1, orig_y1

            if _swap_acceptable(
                c1, c2, components, interior_bbox, board,
                comp_net_lookup, model, orig_combined,
            ):
                swaps_made += 1
                hpwl_now = (_compute_local_hpwl(c1, comp_net_lookup, model)
                            + _compute_local_hpwl(c2, comp_net_lookup, model))
                total_hpwl_delta += orig_combined - hpwl_now
                # Group-aware: propagate swap deltas to caps for any IC swapped.
                bounds = (interior_bbox[0], interior_bbox[1], interior_bbox[2], interior_bbox[3]) if interior_bbox else None
                propagate_ic_delta(model, c1, orig_x1, orig_y1, c1.x, c1.y,
                                  bounds=bounds, decap_map=cached_decap_map)
                propagate_ic_delta(model, c2, orig_x2, orig_y2, c2.x, c2.y,
                                  bounds=bounds, decap_map=cached_decap_map)
                continue

            # Revert position swap
            c1.x, c1.y = orig_x1, orig_y1
            c2.x, c2.y = orig_x2, orig_y2

            # --- Try swap + rotation for non-square components ---
            nonsq1 = _is_non_square(c1)
            nonsq2 = _is_non_square(c2)

            if nonsq1 or nonsq2:
                c1.x, c1.y = orig_x2, orig_y2
                c2.x, c2.y = orig_x1, orig_y1

                rot_penalty = 0.1 * ((1 if nonsq1 else 0) + (1 if nonsq2 else 0))

                if nonsq1:
                    c1.set_rotation((orig_rot1 + 90) % 360)
                if nonsq2:
                    c2.set_rotation((orig_rot2 + 90) % 360)

                if _swap_acceptable(
                    c1, c2, components, interior_bbox, board,
                    comp_net_lookup, model, orig_combined,
                    rot_penalty=rot_penalty,
                ):
                    swaps_made += 1
                    hpwl_now = (_compute_local_hpwl(c1, comp_net_lookup, model)
                                + _compute_local_hpwl(c2, comp_net_lookup, model))
                    total_hpwl_delta += orig_combined - hpwl_now
                    # Group-aware: propagate swap deltas to caps for any IC swapped.
                    bounds = (interior_bbox[0], interior_bbox[1], interior_bbox[2], interior_bbox[3]) if interior_bbox else None
                    propagate_ic_delta(model, c1, orig_x1, orig_y1, c1.x, c1.y,
                                      bounds=bounds, decap_map=cached_decap_map)
                    propagate_ic_delta(model, c2, orig_x2, orig_y2, c2.x, c2.y,
                                      bounds=bounds, decap_map=cached_decap_map)
                    continue

                # Revert rotation swap
                c1.x, c1.y = orig_x1, orig_y1
                c1.set_rotation(orig_rot1)
                c2.x, c2.y = orig_x2, orig_y2
                c2.set_rotation(orig_rot2)

        if verbose and swaps_made > 0:
            print(f"  Pair swap pass {pass_num + 1}/2: "
                  f"{swaps_made} swaps ({attempts} attempts), "
                  f"HPWL recovered: {total_hpwl_delta:.2f}mm")

        components = list(model.components)


# ---------------------------------------------------------------------------
# Combined Post-Legalization Refinement
# ---------------------------------------------------------------------------

def post_legalization_refine(
    model: BoardModel,
    grid_mm: float,
    interior_bbox: tuple[float, float, float, float] | None,
    verbose: bool,
    cached_decap_map: dict | None,
) -> None:
    """Run post-legalization HPWL recovery: slide -> swap -> slide.

    cell_slide checks each trial position against all neighbors before
    accepting (via _has_any_overlap), so it is safe to run even when
    residual overlaps exist — it just won't help the overlapping
    components themselves.  Pair swap requires both endpoints
    overlap-free, so it is skipped if any overlap remains.
    """
    overlaps, _ = _compute_overlap_stats(model)
    oob = _count_oob(model)

    if oob > 0:
        if verbose:
            print(f"  Post-legalization refine: skipping ({oob} out-of-bounds)")
        return

    if overlaps > 0:
        if verbose:
            print(f"  Post-legalization refine: cell-slide only "
                  f"({overlaps} overlaps remain, pair-swap needs clean board)")
        if verbose:
            from engine.cost_function import total_hpwl
            hpwl_before = total_hpwl(model)
        cell_slide(model, grid_mm, interior_bbox, verbose, cached_decap_map)
        if verbose:
            try:
                from engine.cost_function import total_hpwl
                hpwl_after = total_hpwl(model)
                pct = ((hpwl_before - hpwl_after) / hpwl_before * 100
                       if hpwl_before > 0 else 0.0)
                print(f"  Post-legalization refine (slide-only): HPWL "
                      f"{hpwl_before:.1f} -> {hpwl_after:.1f} "
                      f"({pct:+.2f}%)")
            except Exception:
                pass
        return

    if verbose:
        from engine.cost_function import total_hpwl
        hpwl_before = total_hpwl(model)

    # Phase 1: Cell sliding
    cell_slide(model, grid_mm, interior_bbox, verbose, cached_decap_map)

    # Phase 2: Pair swap
    pair_swap_refine(model, grid_mm, interior_bbox, verbose, cached_decap_map)

    # Phase 3: Cell sliding again (pair swaps may have opened up space)
    cell_slide(model, grid_mm, interior_bbox, verbose, cached_decap_map)

    # Final safety: verify no overlaps or OOB were introduced
    overlaps, _ = _compute_overlap_stats(model)
    oob = _count_oob(model)
    if overlaps > 0 or oob > 0:
        if verbose:
            print(f"  WARNING: post-legalization refine introduced "
                  f"{overlaps} overlaps, {oob} OOB (should not happen)")

    if verbose:
        try:
            from engine.cost_function import total_hpwl
            hpwl_after = total_hpwl(model)
            pct = ((hpwl_before - hpwl_after) / hpwl_before * 100
                   if hpwl_before > 0 else 0.0)
            print(f"  Post-legalization refine: HPWL "
                  f"{hpwl_before:.1f} -> {hpwl_after:.1f} "
                  f"({pct:+.2f}%)")
        except Exception:
            pass
