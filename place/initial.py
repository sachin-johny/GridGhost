"""Net-aware initial interior placement.

Cluster macros by net connectivity (reusing engine/net_clustering),
then shelf-pack each cluster around a target point computed from the
cluster's attractors:

- If the cluster shares nets with fixed components or edge connectors,
  the target is a blend of the attractor centroid and a unique grid
  cell. Blending prevents multiple clusters that share a connector
  from piling on the same point — each gets pulled toward the I/O it
  talks to, but also keeps a unique home in the grid.
- Clusters with no attractor fall back to the grid cell alone.

Each cluster is shelf-packed with slack proportional to macro size
(``gap = max(min_gap, max_macro_dim * gap_factor)``), so SA has room
to make moves that actually improve cost instead of being jammed
against a neighbor from the start.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from engine.net_clustering import cluster_components

if TYPE_CHECKING:
    from models.board_model import BoardModel, BoardOutline
    from models.macro import Macro


def place_interior_clustered(
    model: "BoardModel",
    macros: list["Macro"],
    interior_bbox: tuple[float, float, float, float],
    *,
    min_gap: float = 2.5,
    gap_factor: float = 0.5,
    attractor_pull: float = 0.6,
) -> None:
    """Cluster-aware shelf-pack with connector-attractor pull.

    Steps:
      1. ``cluster_components(model)`` groups all movable components by
         net connectivity.
      2. Each cluster is mapped to its leader macros (cap followers
         stay with their leaders — they're not separate macros).
      3. For each cluster, compute a target point as a blend between:
         - the centroid of fixed/placed-connector components sharing
           nets with the cluster (the attractor), and
         - a unique grid cell in ``interior_bbox``.
         ``attractor_pull`` ∈ [0, 1] sets the blend: 1.0 fully pulled
         to attractor (risks pile-up when clusters share attractors),
         0.0 fully on grid cells (no net awareness). Default 0.6.
      4. Shelf-pack the cluster's macros around its target with
         proportional slack so SA has room to move.
    """
    if not macros:
        return

    x_min, y_min, x_max, y_max = interior_bbox

    # Map ref → macro (leaders only). Cap followers are part of their
    # leader's macro; they should NOT appear as separate clusters.
    ref_to_macro: dict[str, "Macro"] = {m.leader.ref: m for m in macros}

    # Group macros by net cluster.
    macro_clusters: list[list["Macro"]] = []
    seen: set[str] = set()
    for cluster_refs in cluster_components(model):
        group: list["Macro"] = []
        for ref in cluster_refs:
            m = ref_to_macro.get(ref)
            if m is None or m.leader.ref in seen:
                continue
            group.append(m)
            seen.add(m.leader.ref)
        if group:
            macro_clusters.append(group)

    # Safety net: any macro that didn't surface in a cluster goes last.
    orphans = [m for m in macros if m.leader.ref not in seen]
    if orphans:
        macro_clusters.append(orphans)

    # Grid fallback layout for clusters (also used as the "home" cell
    # that gets blended with the attractor).
    n_clusters = len(macro_clusters)
    cols = max(1, int(math.ceil(math.sqrt(n_clusters))))
    rows = max(1, int(math.ceil(n_clusters / cols)))
    region_w = (x_max - x_min) / cols
    region_h = (y_max - y_min) / rows

    attractor_positions = _collect_attractor_positions(model)

    for idx, cluster_macros in enumerate(macro_clusters):
        col = idx % cols
        row = idx // cols
        grid_cx = x_min + (col + 0.5) * region_w
        grid_cy = y_min + (row + 0.5) * region_h

        attractors = _cluster_attractors(model, cluster_macros, attractor_positions)
        if attractors:
            ax = sum(p[0] for p in attractors) / len(attractors)
            ay = sum(p[1] for p in attractors) / len(attractors)
            cx = ax * attractor_pull + grid_cx * (1.0 - attractor_pull)
            cy = ay * attractor_pull + grid_cy * (1.0 - attractor_pull)
        else:
            cx, cy = grid_cx, grid_cy

        # Clamp to interior so the cluster center stays in-bounds; the
        # shelf-packer will still let macros spill if the cluster is
        # bigger than the interior (legalizer will mop up).
        pad = 1.0
        cx = max(x_min + pad, min(x_max - pad, cx))
        cy = max(y_min + pad, min(y_max - pad, cy))

        _shelf_pack_around_point(
            cluster_macros, cx, cy, interior_bbox,
            min_gap=min_gap, gap_factor=gap_factor,
        )


def _shelf_pack_around_point(
    macros: list["Macro"],
    cx: float,
    cy: float,
    interior_bbox: tuple[float, float, float, float],
    *,
    min_gap: float,
    gap_factor: float,
) -> None:
    """Shelf-pack macros centered on (cx, cy).

    Same row-major mechanic as a top-left shelf-pack, but:
      - Sorted within cluster by descending height for tidy rows.
      - Per-macro gap = ``max(min_gap, max_macro_dim * gap_factor)`` —
        bigger macros get more breathing room.
      - Block is centered on (cx, cy) instead of starting at the
        top-left corner of the interior.
    """
    if not macros:
        return

    x_min, y_min, x_max, y_max = interior_bbox
    sorted_macros = sorted(macros, key=lambda m: -(m.bbox[3] - m.bbox[1]))

    # Target row width: ~sqrt of cluster area * 1.3 for slack. Clamped
    # to interior width so single-cluster boards don't sprawl off-edge.
    total_area = sum(
        (m.bbox[2] - m.bbox[0]) * (m.bbox[3] - m.bbox[1])
        for m in sorted_macros
    )
    target_width = min(x_max - x_min, max(math.sqrt(total_area) * 1.3, 10.0))

    def gap_for(m: "Macro") -> float:
        bx1, by1, bx2, by2 = m.bbox
        return max(min_gap, max(bx2 - bx1, by2 - by1) * gap_factor)

    # First pass: build rows of (macro, width, height, gap).
    rows: list[tuple[list[tuple["Macro", float, float, float]], float]] = []
    current_row: list[tuple["Macro", float, float, float]] = []
    cursor_x = 0.0
    row_height = 0.0
    for m in sorted_macros:
        bx1, by1, bx2, by2 = m.bbox
        mw = bx2 - bx1
        mh = by2 - by1
        g = gap_for(m)

        if cursor_x + mw > target_width and current_row:
            rows.append((current_row, row_height))
            current_row = []
            cursor_x = 0.0
            row_height = 0.0

        current_row.append((m, mw, mh, g))
        cursor_x += mw + g
        row_height = max(row_height, mh)
    if current_row:
        rows.append((current_row, row_height))

    # Compute total block height (with inter-row gap = min_gap).
    inter_row_gap = min_gap
    total_h = sum(rh for _, rh in rows) + inter_row_gap * max(0, len(rows) - 1)

    # Second pass: place each row centered horizontally on cx; stack
    # rows vertically centered on cy.
    cur_y = cy - total_h / 2
    for row, rh in rows:
        # Row width = sum of macro widths + inter-macro gaps.
        row_width = sum(w + g for _, w, _, g in row) - (row[-1][3] if row else 0.0)
        cur_x = cx - row_width / 2

        for m, mw, mh, g in row:
            bx1, by1, _, _ = m.bbox
            lox = m.leader.x - bx1
            loy = m.leader.y - by1
            target_x = cur_x + lox
            target_y = cur_y + loy
            # No bounds here — initial placement must accept any position;
            # legalizer will clamp. Bounds-rejecting would crash on dense boards.
            m.set_pose(target_x, target_y, 0.0)
            cur_x += mw + g

        cur_y += rh + inter_row_gap


def _collect_attractor_positions(
    model: "BoardModel",
) -> dict[str, list[tuple[float, float]]]:
    """Map net name → list of (x, y) attractor positions for that net.

    Attractors are components that are either fixed (e.g., pre-placed
    mounting holes or fixed ICs) OR already-placed connectors. We use
    the model's current state at call time — so if the pipeline places
    connectors before calling us, those positions count.
    """
    out: dict[str, list[tuple[float, float]]] = {}
    for net in model.nets:
        for ref in net.component_refs:
            comp = model.get_component(ref)
            if comp is None:
                continue
            if getattr(comp, "is_fixed", False) or comp.component_type == "connector":
                out.setdefault(net.name, []).append((comp.x, comp.y))
    return out


def _cluster_attractors(
    model: "BoardModel",
    cluster_macros: list["Macro"],
    attractor_positions: dict[str, list[tuple[float, float]]],
) -> list[tuple[float, float]]:
    """Collect attractor positions for any net touching this cluster."""
    attractors: list[tuple[float, float]] = []
    cluster_refs = {m.leader.ref for m in cluster_macros}
    # Include cap followers — they're part of the cluster's connectivity too.
    for m in cluster_macros:
        for f in m.followers:
            cluster_refs.add(f.ref)

    for net in model.nets:
        if not any(r in cluster_refs for r in net.component_refs):
            continue
        attractors.extend(attractor_positions.get(net.name, []))
    return attractors


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
