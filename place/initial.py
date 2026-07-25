"""Macro interior placement — Phase A (space-filling) + Phase B (bounded nudge).

Issue 1 of PLACEMENT_FIX_PLAN.md splits interior seeding into two
independently-testable phases instead of one blended formula that had to be
simultaneously space-filling and connectivity-aware (and collapsed to the
board center whenever a high-fan-out net like GND dominated the average):

  * **Phase A — space-filling seed (connectivity-blind positioning).**
    Cluster macros by net connectivity (connected macros share a shelf-pack
    block), order the *clusters* by descending max-member height for
    shelf-pack area efficiency, then shelf-pack the blocks to FILL the
    interior — rows are spread vertically and blocks within a row are spread
    horizontally so the whole interior is used.  No attractor logic at all:
    worst case (zero useful connectivity signal) still yields a well-spread
    layout, not a collapsed clump.

  * **Phase B — bounded connectivity nudge** (``place.cluster.compute_attractor_nudges``).
    A small, per-net-centroid, clique-net-weighted (``1/(k-1)``), capped
    nudge toward genuinely informative attractors.  Bounded by construction
    so it can only pull a cluster part-way off its Phase-A home — SA + the
    legalizer do the real fine-grained HPWL optimization from there.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from place.cluster import (
    order_macros_by_connectivity,
    collect_attractor_positions,
    compute_attractor_nudges,
)

if TYPE_CHECKING:
    from models.board_model import BoardModel, BoardOutline
    from models.macro import Macro


def _macro_w(m: "Macro") -> float:
    return m.bbox[2] - m.bbox[0]


def _macro_h(m: "Macro") -> float:
    return m.bbox[3] - m.bbox[1]


def _clamp_macros_to_bounds(
    macros: list["Macro"],
    bounds: tuple[float, float, float, float],
) -> None:
    """Best-effort translate every macro so its bbox sits inside ``bounds``.

    Unlike the legalizer's ``boundary_clamp`` (which *rejects* a move that
    leaves the bbox out of bounds — leaving over-dense macros OOB), this
    forces macros in unconditionally so SA starts from a legal pose:
    macros that fit are clamped to the nearest in-bounds position; macros
    bigger than the bounds (a 62-macro board in a 49×48mm interior) are
    *centered* so the overflow is symmetric and minimal.  The overlaps this
    creates are exactly what the legalizer's push-apart is for — but no
    macro starts outside the board, which is the regression this prevents.
    """
    x_min, y_min, x_max, y_max = bounds
    bw = x_max - x_min
    bh = y_max - y_min
    for m in macros:
        bx1, by1, bx2, by2 = m.bbox
        mw = bx2 - bx1
        mh = by2 - by1
        if mw <= bw:
            if bx1 < x_min:
                dx = x_min - bx1
            elif bx2 > x_max:
                dx = x_max - bx2
            else:
                dx = 0.0
        else:
            dx = (x_min + x_max) / 2 - (bx1 + bx2) / 2  # center oversized
        if mh <= bh:
            if by1 < y_min:
                dy = y_min - by1
            elif by2 > y_max:
                dy = y_max - by2
            else:
                dy = 0.0
        else:
            dy = (y_min + y_max) / 2 - (by1 + by2) / 2  # center oversized
        if abs(dx) > 1e-9 or abs(dy) > 1e-9:
            m.translate(dx, dy)  # bounds=None → always applies; followers follow


def _pack_cluster_block(
    cluster_macros: list["Macro"],
    min_gap: float,
    gap_factor: float,
) -> tuple[list[tuple["Macro", float, float]], float, float]:
    """Shelf-pack one cluster's macros into a block anchored at (0, 0).

    Returns ``(layout, block_w, block_h)`` where ``layout`` is a list of
    ``(macro, origin_rel_x, origin_rel_y)`` — the leader *origin* position
    relative to the block's top-left corner.  Macros are sorted by descending
    height for tidy rows; per-macro gap scales with macro size so SA has room.

    Origin-vs-bbox note: ``origin_rel`` is derived from each macro's current
    bbox (``leader.x - bbox_left``), the macro-level equivalent of
    ``Component.set_bbox_center``.  Macros are placed at rotation 0, which
    matches their parse-time rotation, so the offset is consistent.  The
    legalizer clamps anything that spills.
    """
    if not cluster_macros:
        return [], 0.0, 0.0

    sorted_macros = sorted(cluster_macros, key=lambda m: -_macro_h(m))
    total_area = sum(_macro_w(m) * _macro_h(m) for m in sorted_macros)
    target_width = max(math.sqrt(total_area) * 1.3, 10.0)

    rows: list[tuple[list[tuple["Macro", float, float, float, float]], float]] = []
    current_row: list[tuple["Macro", float, float, float, float]] = []
    cursor_x = 0.0
    row_height = 0.0
    for m in sorted_macros:
        mw, mh = _macro_w(m), _macro_h(m)
        g = max(min_gap, max(mw, mh) * gap_factor)
        if cursor_x + mw > target_width and current_row:
            rows.append((current_row, row_height))
            current_row = []
            cursor_x = 0.0
            row_height = 0.0
        # cursor_x is this macro's bbox-left offset within the row
        current_row.append((m, mw, mh, g, cursor_x))
        cursor_x += mw + g
        row_height = max(row_height, mh)
    if current_row:
        rows.append((current_row, row_height))

    inter_row_gap = min_gap
    layout: list[tuple["Macro", float, float]] = []
    block_w = 0.0
    cur_y = 0.0
    for row_items, rh in rows:
        row_w = 0.0
        for m, mw, mh, g, bx_left_rel in row_items:
            bx1, by1, _, _ = m.bbox
            lox = m.leader.x - bx1  # leader origin offset from macro bbox-left
            loy = m.leader.y - by1  # leader origin offset from macro bbox-top
            layout.append((m, bx_left_rel + lox, cur_y + loy))
            row_w = max(row_w, bx_left_rel + mw)
        block_w = max(block_w, row_w)
        cur_y += rh + inter_row_gap
    block_h = max(0.0, cur_y - inter_row_gap) if rows else 0.0
    return layout, block_w, block_h


def place_interior_phase_a(
    model: "BoardModel",
    macros: list["Macro"],
    interior_bbox: tuple[float, float, float, float],
    *,
    min_gap: float = 2.5,
    gap_factor: float = 0.5,
) -> list[list["Macro"]]:
    """Phase A: height-ordered shelf-pack of net-clusters to fill the interior.

    Returns the cluster list (for Phase B).  Macros are placed at rotation 0
    with no bounds — initial placement accepts any position; the legalizer
    clamps.  This is the connectivity-blind space-filling baseline.
    """
    if not macros:
        return []

    x_min, y_min, x_max, y_max = interior_bbox
    interior_w = x_max - x_min
    interior_h = y_max - y_min

    clusters = order_macros_by_connectivity(model, macros)

    # Pack each cluster into a block.
    blocks: list[dict] = []
    for cl in clusters:
        layout, w, h = _pack_cluster_block(cl, min_gap, gap_factor)
        blocks.append({"layout": layout, "w": w, "h": h, "cluster": cl})

    # Shelf-pack blocks into rows (height-sorted for area efficiency).
    # Target a number of rows that fills the interior HEIGHT, not just the
    # width. Without this, a board with 6 tall blocks (47mm each) on a
    # 115mm-tall interior ends up with 1 row of 6 blocks + 1 row of 2
    # small blocks — 47mm used, 68mm of vertical emptiness.
    #
    # The target row height is based on the MEDIAN block height (not
    # interior_h / n_target_rows) because the blocks have a fixed height
    # distribution — a 47mm-tall IC macro can't shrink to fit a 38mm
    # target row. Using the median ensures the target is achievable:
    # roughly half the blocks fit in one row, the other half trigger a
    # new row. This naturally produces 2-3 rows for typical IC+cap
    # macro sets instead of 1 mega-row.
    #
    # DISABLE on dense boards (macro density > 50%): no vertical slack
    # to spread into, and forcing extra rows creates overlaps the
    # legalizer can't resolve.
    blocks.sort(key=lambda b: -b["h"])
    n_blocks = len(blocks)
    total_block_area = sum(b["w"] * b["h"] for b in blocks)
    interior_area = max(interior_w * interior_h, 1.0)
    density = total_block_area / interior_area
    height_overflow_enabled = density < 0.50

    if n_blocks > 0 and height_overflow_enabled:
        # Target row height = median block height × 1.5 (allow 2 rows of
        # median-height blocks per target row, so small blocks don't
        # trigger premature row breaks but tall blocks do).
        #
        # Use the TRUE median: for even n, average the two middle values
        # (sorted_heights[n//2 - 1] + sorted_heights[n//2]) / 2.
        # The previous sorted_heights[n // 2] picked the upper median
        # for even n, biasing target_row_h upward by one rank and
        # triggering premature row breaks on small even-count boards.
        sorted_heights = sorted(b["h"] for b in blocks)
        if n_blocks % 2 == 1:
            median_h = sorted_heights[n_blocks // 2]
        else:
            median_h = (sorted_heights[n_blocks // 2 - 1]
                        + sorted_heights[n_blocks // 2]) / 2.0
        target_row_h = max(median_h * 1.5, 1.0)
    else:
        target_row_h = math.inf  # never trigger height_overflow

    rows: list[tuple[list[dict], float]] = []
    cur_row: list[dict] = []
    cur_w = 0.0
    row_h = 0.0
    for b in blocks:
        width_overflow = cur_row and cur_w + b["w"] > max(interior_w, 1.0)
        height_overflow = (
            cur_row and row_h >= target_row_h
            and b["h"] > 0.5 * target_row_h
        )
        if width_overflow or height_overflow:
            rows.append((cur_row, row_h))
            cur_row = []
            cur_w = 0.0
            row_h = 0.0
        cur_row.append(b)
        cur_w += b["w"] + min_gap
        row_h = max(row_h, b["h"])
    if cur_row:
        rows.append((cur_row, row_h))

    # Spread rows vertically to fill the interior height; center the stack.
    n_rows = len(rows)
    total_row_h = sum(rh for _, rh in rows)
    v_slack = max(0.0, interior_h - total_row_h) / max(1, n_rows - 1) if n_rows > 1 else 0.0
    total_with_v = total_row_h + v_slack * max(0, n_rows - 1)
    cur_y = y_min + max(0.0, interior_h - total_with_v) / 2

    for row_blocks, rh in rows:
        # Spread blocks horizontally to fill the interior width; center the row.
        n_in_row = len(row_blocks)
        total_bw = sum(b["w"] for b in row_blocks)
        h_slack = max(0.0, interior_w - total_bw) / max(1, n_in_row - 1) if n_in_row > 1 else 0.0
        total_with_h = total_bw + h_slack * max(0, n_in_row - 1)
        cur_x = x_min + max(0.0, interior_w - total_with_h) / 2

        for b in row_blocks:
            for m, rx, ry in b["layout"]:
                m.set_pose(cur_x + rx, cur_y + ry, 0.0)
            cur_x += b["w"] + h_slack
        cur_y += rh + v_slack

    # Best-effort clamp into the interior so SA starts in-bounds. On sparse
    # boards this is a no-op (the spread already fits); on over-dense boards
    # it pulls edge macros in and centers oversized ones, preventing the OOB
    # regression at the cost of overlaps the legalizer then resolves.
    _clamp_macros_to_bounds(macros, interior_bbox)

    return clusters


def apply_connectivity_nudges(
    model: "BoardModel",
    clusters: list[list["Macro"]],
    interior_bbox: tuple[float, float, float, float],
    *,
    nudge_fraction: float = 0.25,
    verbose: bool = False,
) -> float:
    """Phase B: apply bounded, per-net-weighted connectivity nudges.

    Returns the mean cluster displacement (mm) — printed by the pipeline so a
    future collapse regression shows up in the log instead of needing an SVG.
    """
    if not clusters:
        return 0.0

    x_min, y_min, x_max, y_max = interior_bbox
    # Cap nudge at ~one grid cell (interior / n_clusters) per the plan.
    grid_cell = min(x_max - x_min, y_max - y_min) / max(1, len(clusters))
    attractor_positions = collect_attractor_positions(model)
    nudges = compute_attractor_nudges(
        model, clusters, attractor_positions,
        nudge_fraction=nudge_fraction, max_nudge_mm=grid_cell,
    )

    total_disp = 0.0
    for idx, (dx, dy) in nudges.items():
        for m in clusters[idx]:
            m.translate(dx, dy)  # bounds=None → always applies; followers follow
        total_disp += math.hypot(dx, dy)

    avg = total_disp / len(clusters)
    if verbose:
        print(f"  Phase B (connectivity nudge): {len(nudges)}/{len(clusters)} clusters "
              f"moved, avg displacement={avg:.2f}mm (cap {grid_cell:.1f}mm)")
    return avg


def place_interior_clustered(
    model: "BoardModel",
    macros: list["Macro"],
    interior_bbox: tuple[float, float, float, float],
    *,
    min_gap: float = 2.5,
    gap_factor: float = 0.5,
    # Accepted for backward compatibility; the old global attractor_pull
    # constant is replaced by per-net weighting in compute_attractor_nudges.
    attractor_pull: float | None = None,
) -> None:
    """Cluster-aware interior placement (Phase A + Phase B).

    Thin wrapper kept for callers that want one call.  The pipeline calls
    ``place_interior_phase_a`` and ``apply_connectivity_nudges`` directly so
    the two phases show up as separate steps in verbose output.
    """
    clusters = place_interior_phase_a(
        model, macros, interior_bbox, min_gap=min_gap, gap_factor=gap_factor,
    )
    apply_connectivity_nudges(model, clusters, interior_bbox)


def compute_interior_bbox(
    macros: list["Macro"],
    board: "BoardOutline",
    margin: float,
    connector_reserve: float = 0.0,
) -> tuple[float, float, float, float]:
    """Compute the usable interior region.

    Shrinks the board by ``margin`` plus ``connector_reserve`` (room
    reserved on each edge for connectors).
    """
    x_min = board.x_min + margin + connector_reserve
    y_min = board.y_min + margin + connector_reserve
    x_max = board.x_max - margin - connector_reserve
    y_max = board.y_max - margin - connector_reserve
    return (x_min, y_min, x_max, y_max)
