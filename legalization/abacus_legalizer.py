"""Abacus-style row-based DP legalization with net-topology awareness.

Based on "Faster Optimal Slot Assignment" by Brenner (2005).
Components assigned to rows by y-coordinate, with net-connected components
preferentially merged into the same row to minimize HPWL increase.
Within each row, the Abacus cluster-merge DP finds minimum squared-displacement
positions subject to non-overlap constraints.
"""

from __future__ import annotations

import math
from models.board_model import BoardModel, Component, BoardOutline


def abacus_legalize(
    model: BoardModel,
    grid_mm: float = 0.1,
    verbose: bool = False,
    interior_bbox: tuple[float, float, float, float] | None = None,
    cached_decap_map: dict | None = None,
) -> bool:
    board = model.board
    components = list(model.components)
    movable = [c for c in components if not c.is_fixed and not c.is_edge_connector]

    if not movable:
        return True

    orig_positions = {id(c): (c.x, c.y, c.rotation) for c in movable}

    rows = _assign_rows(movable, grid_mm, board, interior_bbox, model=model,
                        cached_decap_map=cached_decap_map)

    if verbose:
        print(f"  Abacus: {len(rows)} rows, {len(movable)} movable components")

    total_displacement = 0.0
    for row_idx, (row_y, row_height, row_comps) in enumerate(rows):
        disp = _legalize_row(row_comps, row_y, row_height, board, grid_mm,
                             interior_bbox, components)
        total_displacement += disp

    if verbose:
        print(f"  Abacus: total displacement = {total_displacement:.2f} mm")

    for comp in movable:
        _clamp_to_bounds(comp, board, interior_bbox)

    remaining = _count_overlaps(movable, components)
    if remaining > 0 and verbose:
        print(f"  Abacus: {remaining} cross-row overlaps, resolving...")

    if remaining > 0:
        _resolve_cross_row_overlaps(movable, components, board, grid_mm, interior_bbox)

    final_overlaps = _count_overlaps(movable, components)
    if verbose:
        print(f"  Abacus: {final_overlaps} overlaps remaining after row DP")

    return final_overlaps == 0


# ---------------------------------------------------------------------------
# Row assignment with net-topology awareness
# ---------------------------------------------------------------------------

def _assign_rows(
    components: list[Component],
    grid_mm: float,
    board: BoardOutline,
    interior_bbox: tuple[float, float, float, float] | None = None,
    model: BoardModel | None = None,
    cached_decap_map: dict | None = None,
) -> list[tuple[float, float, list[Component]]]:
    if not components:
        return []

    max_height = max(c.effective_height for c in components)
    row_pitch = max(math.ceil(max_height / grid_mm) * grid_mm, grid_mm)

    if interior_bbox:
        y_min = interior_bbox[1]
        y_max = interior_bbox[3]
    else:
        y_min = board.y_min
        y_max = board.y_max

    row_map: dict[int, list[Component]] = {}
    comp_row: dict[int, int] = {}
    for comp in components:
        row_idx = round((comp.y - y_min - row_pitch / 2) / row_pitch)
        if row_idx not in row_map:
            row_map[row_idx] = []
        row_map[row_idx].append(comp)
        comp_row[id(comp)] = row_idx

    comp_map = {c.ref: c for c in components}

    # Net-topology-aware row merging: components sharing signal nets
    # preferentially placed in the same row if displacement is small.
    if model is not None:
        from engine.cost_state import _is_power_net
        net_comps: dict[int, list[Component]] = {}
        for net_idx, net in enumerate(model.nets):
            if _is_power_net(net.name):
                continue
            comps_on_net = []
            for ref, _ in net.pins:
                if ref in comp_map:
                    comps_on_net.append(comp_map[ref])
            if len(comps_on_net) >= 2:
                net_comps[net_idx] = comps_on_net

        for net_idx, comps in net_comps.items():
            if not comps:
                continue
            row_votes: dict[int, int] = {}
            for c in comps:
                r = comp_row.get(id(c))
                if r is not None:
                    row_votes[r] = row_votes.get(r, 0) + 1
            if not row_votes:
                continue
            target_row = max(row_votes, key=row_votes.get)

            for c in comps:
                current_row = comp_row.get(id(c))
                if current_row is None or current_row == target_row:
                    continue
                target_y = y_min + target_row * row_pitch + row_pitch / 2
                displacement = abs(target_y - c.y)
                if displacement < 2.0 * row_pitch:
                    if current_row in row_map and c in row_map[current_row]:
                        row_map[current_row].remove(c)
                    if target_row not in row_map:
                        row_map[target_row] = []
                    row_map[target_row].append(c)
                    comp_row[id(c)] = target_row

    # Decoupling topology: pull each cap into its assigned IC's row so the
    # legalizer doesn't put them on opposite sides of the board.  Power nets
    # are excluded from the signal-net merging above, so without this pass
    # decoupling caps end up wherever their initial Y placed them — typically
    # far from the IC after SA condenses the cluster.
    # Use a larger displacement tolerance than signal-net merging (4× row_pitch
    # vs 2×) since cap-IC proximity is the whole point of the constraint rule.
    if cached_decap_map:
        for ic_ref, cap_refs in cached_decap_map.items():
            ic = comp_map.get(ic_ref)
            if ic is None:
                continue
            ic_row_idx = comp_row.get(id(ic))
            if ic_row_idx is None:
                continue
            for cap_ref in cap_refs:
                cap = comp_map.get(cap_ref)
                if cap is None:
                    continue
                cap_row_idx = comp_row.get(id(cap))
                if cap_row_idx is None or cap_row_idx == ic_row_idx:
                    continue
                target_y = y_min + ic_row_idx * row_pitch + row_pitch / 2
                displacement = abs(target_y - cap.y)
                if displacement < 4.0 * row_pitch:
                    if cap_row_idx in row_map and cap in row_map[cap_row_idx]:
                        row_map[cap_row_idx].remove(cap)
                    if ic_row_idx not in row_map:
                        row_map[ic_row_idx] = []
                    row_map[ic_row_idx].append(cap)
                    comp_row[id(cap)] = ic_row_idx

    row_map = {k: v for k, v in row_map.items() if v}

    rows = []
    for row_idx in sorted(row_map.keys()):
        row_comps = row_map[row_idx]
        row_y = y_min + row_idx * row_pitch + row_pitch / 2
        row_height = max(c.effective_height for c in row_comps)
        row_comps.sort(key=lambda c: c.x)
        rows.append((row_y, row_height, row_comps))

    return rows


# ---------------------------------------------------------------------------
# Row legalization (core Abacus DP)
# ---------------------------------------------------------------------------

def _legalize_row(
    row_comps: list[Component],
    row_y: float,
    row_height: float,
    board: BoardOutline,
    grid_mm: float,
    interior_bbox: tuple[float, float, float, float] | None,
    all_components: list[Component],
) -> float:
    if not row_comps:
        return 0.0

    if interior_bbox:
        x_min_bound = interior_bbox[0]
        x_max_bound = interior_bbox[2]
    else:
        x_min_bound = board.x_min
        x_max_bound = board.x_max

    orig_xs = {id(c): c.x for c in row_comps}

    snapped_row_y = round(row_y / grid_mm) * grid_mm
    for comp in row_comps:
        comp.y = snapped_row_y

    sorted_comps = sorted(row_comps, key=lambda c: orig_xs[id(c)])

    gap = grid_mm
    clusters: list[dict] = []

    for comp in sorted_comps:
        ew = comp.effective_width
        opt_x = round(orig_xs[id(comp)] / grid_mm) * grid_mm

        new_cluster = _make_cluster(comp, opt_x, ew)

        while clusters:
            prev = clusters[-1]
            if _clusters_overlap(prev, new_cluster):
                new_cluster = _merge_clusters(prev, new_cluster, gap)
                clusters.pop()
            else:
                break

        clusters.append(new_cluster)

    total_displacement = 0.0
    for cluster in clusters:
        positions = _place_cluster(cluster, grid_mm, x_min_bound, x_max_bound)
        for comp, placed_x in positions:
            old_x = orig_xs[id(comp)]
            total_displacement += abs(placed_x - old_x)
            comp.x = placed_x

    return total_displacement


def _make_cluster(comp: Component, opt_x: float, ew: float) -> dict:
    d = [0.0]
    e_val = opt_x - d[0]
    return {
        'comps': [comp],
        'opt_xs': [opt_x],
        'ews': [ew],
        'd': d,
        'e_sum': e_val,
        'n': 1,
        'x1_opt': e_val,
    }


def _merge_clusters(left: dict, right: dict, gap: float) -> dict:
    offset = left['d'][-1] + (left['ews'][-1] + right['ews'][0]) / 2 + gap

    merged_d = left['d'] + [offset + d_val for d_val in right['d']]

    merged_e_sum = left['e_sum']
    for j in range(right['n']):
        merged_e_sum += right['opt_xs'][j] - (offset + right['d'][j])

    merged_n = left['n'] + right['n']
    x1_opt = merged_e_sum / merged_n

    return {
        'comps': left['comps'] + right['comps'],
        'opt_xs': left['opt_xs'] + right['opt_xs'],
        'ews': left['ews'] + right['ews'],
        'd': merged_d,
        'e_sum': merged_e_sum,
        'n': merged_n,
        'x1_opt': x1_opt,
    }


def _clusters_overlap(left: dict, right: dict) -> bool:
    left_right_edge = left['x1_opt'] + left['d'][-1] + left['ews'][-1] / 2
    right_left_edge = right['x1_opt'] - right['ews'][0] / 2
    return right_left_edge < left_right_edge


def _place_cluster(
    cluster: dict,
    grid_mm: float,
    x_min_bound: float,
    x_max_bound: float,
) -> list[tuple[Component, float]]:
    n = cluster['n']
    if n == 0:
        return []

    x1 = cluster['x1_opt']
    x1 = round(x1 / grid_mm) * grid_mm

    first_ew = cluster['ews'][0]
    last_ew = cluster['ews'][-1]
    last_d = cluster['d'][-1]

    x1_min = x_min_bound + first_ew / 2
    x1_max = x_max_bound - last_d - last_ew / 2

    if x1_min > x1_max:
        x1 = x1_min
    else:
        x1 = max(x1_min, min(x1, x1_max))

    positions = []
    for j in range(n):
        placed_x = x1 + cluster['d'][j]
        placed_x = round(placed_x / grid_mm) * grid_mm
        comp_ew = cluster['ews'][j]
        placed_x = max(x_min_bound + comp_ew / 2,
                       min(placed_x, x_max_bound - comp_ew / 2))
        positions.append((cluster['comps'][j], placed_x))

    return positions


# ---------------------------------------------------------------------------
# Cross-row overlap resolution
# ---------------------------------------------------------------------------

def _resolve_cross_row_overlaps(
    movable: list[Component],
    all_components: list[Component],
    board: BoardOutline,
    grid_mm: float,
    interior_bbox: tuple[float, float, float, float] | None,
) -> None:
    max_nudge = max(board.width, board.height) * 0.15
    nudge_dists = [grid_mm, grid_mm * 2, grid_mm * 5, grid_mm * 10,
                   0.5, 1.0, 2.0, 4.0, 8.0]
    nudge_dists.extend([d for d in [12.0, 16.0] if d <= max_nudge])

    row_pitch = max(c.effective_height for c in movable)
    row_pitch = max(math.ceil(row_pitch / grid_mm) * grid_mm, grid_mm)

    # Wider Y search (Fix 1.7 step 2): include ±0.5×row_pitch and ±1.5×row_pitch
    # candidates for boards where rows are sparse but components are tall.
    # Ordered near → far so smaller displacements are preferred.
    y_candidates = [
        -row_pitch * 0.5, row_pitch * 0.5,
        -row_pitch, row_pitch,
        -row_pitch * 1.5, row_pitch * 1.5,
        -row_pitch * 2, row_pitch * 2,
    ]

    # Lazy import to avoid circular dependency (legalizer.py imports abacus_legalizer).
    from legalization.legalizer import _push_apart

    for outer in range(20):
        overlap_pairs = _find_overlap_pairs(movable)
        if not overlap_pairs:
            break

        progress = False
        for c1, c2 in overlap_pairs:
            if not c1.overlaps(c2):
                continue

            c1_overlaps = _count_overlaps_single(c1, all_components)
            c2_overlaps = _count_overlaps_single(c2, all_components)

            if c1_overlaps >= c2_overlaps and not c2.is_fixed:
                mover = c2
            elif not c1.is_fixed:
                mover = c1
            else:
                continue

            old_x, old_y = mover.x, mover.y
            initial_overlaps = _count_overlaps_single(mover, all_components)
            best_x, best_y = old_x, old_y
            best_overlaps = initial_overlaps

            for delta in nudge_dists:
                for dx in [-delta, delta]:
                    trial_x = old_x + dx
                    trial_x = _clamp_x(trial_x, mover, board, interior_bbox)
                    mover.x = trial_x
                    mover.y = old_y
                    new_overlaps = _count_overlaps_single(mover, all_components)
                    if new_overlaps < best_overlaps:
                        best_overlaps = new_overlaps
                        best_x = trial_x
                        best_y = old_y
                        if new_overlaps == 0:
                            break
                if best_overlaps == 0:
                    break

            if best_overlaps > 0:
                for dy in y_candidates:
                    trial_y = old_y + dy
                    mover.x = old_x
                    mover.y = trial_y
                    _clamp_to_bounds(mover, board, interior_bbox)
                    new_overlaps = _count_overlaps_single(mover, all_components)
                    if new_overlaps < best_overlaps:
                        best_overlaps = new_overlaps
                        best_x = mover.x
                        best_y = mover.y
                        if new_overlaps == 0:
                            break

            mover.x = best_x
            mover.y = best_y
            if best_overlaps < initial_overlaps:
                # Single-mover search reduced overlaps (even if not to zero).
                progress = True

            # Bidirectional push (Fix 1.7 step 1): only fire when single-mover
            # search stalled AND the pair still overlaps. Strict acceptance
            # (pair_after < pair_before) — accepting equal would just shuffle
            # overlaps around without converging.
            if best_overlaps == 0 or not c1.overlaps(c2):
                continue

            c1_old_x, c1_old_y = c1.x, c1.y
            c2_old_x, c2_old_y = c2.x, c2.y
            pair_before = _count_pair_overlaps_involving(c1, c2, all_components)
            _push_apart(c1, c2, 1.0, grid_mm)
            _clamp_to_bounds(c1, board, interior_bbox)
            _clamp_to_bounds(c2, board, interior_bbox)
            pair_after = _count_pair_overlaps_involving(c1, c2, all_components)
            if pair_after < pair_before:
                progress = True
            else:
                # No strict improvement — revert. The legalizer fallback will
                # handle residual overlaps via _resolve_overlaps + _greedy_resolve.
                c1.x, c1.y = c1_old_x, c1_old_y
                c2.x, c2.y = c2_old_x, c2_old_y

        if not progress:
            break


def _count_pair_overlaps_involving(
    c1: Component, c2: Component, all_components: list[Component],
) -> int:
    """Count overlaps involving c1 or c2 (including the c1↔c2 pair once)."""
    count = 0
    for other in all_components:
        if other is c1 or other is c2:
            continue
        if c1.overlaps(other):
            count += 1
        if c2.overlaps(other):
            count += 1
    if c1.overlaps(c2):
        count += 1
    return count


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _clamp_to_bounds(
    comp: Component,
    board: BoardOutline,
    interior_bbox: tuple[float, float, float, float] | None,
) -> None:
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


def _clamp_x(
    x: float, comp: Component, board: BoardOutline,
    interior_bbox: tuple[float, float, float, float] | None,
) -> float:
    half_w = comp.effective_width / 2.0
    if interior_bbox:
        x_min = interior_bbox[0] + half_w
        x_max = interior_bbox[2] - half_w
    else:
        x_min = board.x_min + half_w
        x_max = board.x_max - half_w
    return max(x_min, min(x, x_max))


def _count_overlaps(movable: list[Component], all_components: list[Component]) -> int:
    count = 0
    for c1 in movable:
        for c2 in all_components:
            if c2 is c1:
                continue
            if c1.overlaps(c2):
                count += 1
    return count


def _count_overlaps_single(comp: Component, all_components: list[Component]) -> int:
    count = 0
    for other in all_components:
        if other is comp:
            continue
        if comp.overlaps(other):
            count += 1
    return count


def _find_overlap_pairs(movable: list[Component]) -> list[tuple[Component, Component]]:
    pairs = []
    for i, c1 in enumerate(movable):
        for c2 in movable[i + 1:]:
            if c1.overlaps(c2):
                pairs.append((c1, c2))
    return pairs
