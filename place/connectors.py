"""Connector perimeter placement — simplified.

Place edge connectors on the board perimeter, facing outward. For each
connector, compute the mating direction from pad geometry (long-axis
vs short-axis), then pick the rotation that best aligns with each
edge's outward normal.

Simplified vs the existing engine/smart_placement.py: no chain pairing,
no family grouping, no capacity splitting. Connectors distribute by
along-edge extent with a fixed gap. If they don't fit, we shrink the
gap; if they still don't fit, they overlap (the legalizer will catch
this and surface it as an error).
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.board_model import Component, BoardOutline


# Connector families where the mating face is on a SHORT edge of the
# body's long axis (parallel to the pad column). For these, mating
# direction = body long axis, on the side opposite the pad centroid.
_END_MATING_MARKERS = (
    "barreljack", "barrel_jack", "dcjack", "dc_jack",
    "usb", "usb_a", "usb_b", "usb_c", "micro_usb", "miniusb",
    "rj45", "ethernet", "hdmi", "dsub", "db9", "db25", "vga",
    "dvi", "displayport", "jack_dc",
)


def _compute_pad_facing_direction(comp: "Component") -> tuple[float, float]:
    """Return (fx, fy) unit vector in local coords pointing toward the mating face.

    Two cases:
    1. End-mating (USB, barrel jack, RJ45, HDMI, D-Sub): mate at the
       END of the body's long axis, opposite from pad centroid.
    2. Face-mating (terminal blocks, pin headers): mate perpendicular
       to the pad column, on the side opposite the pad centroid.
    """
    if len(comp.pads) < 2:
        return (1.0, 0.0)

    fp = (comp.footprint or "").lower()
    val = (comp.value or "").lower()
    name = fp + " " + val
    is_end_mating = any(m in name for m in _END_MATING_MARKERS)

    xs = [p.x for p in comp.pads]
    ys = [p.y for p in comp.pads]
    spread_x = max(xs) - min(xs)
    spread_y = max(ys) - min(ys)

    if is_end_mating:
        if comp.height >= comp.width:
            body_cy = getattr(comp, "bbox_offset_y", 0.0)
            pad_cy = sum(ys) / len(ys)
            return (0.0, -1.0 if pad_cy < body_cy else 1.0)
        body_cx = getattr(comp, "bbox_offset_x", 0.0)
        pad_cx = sum(xs) / len(xs)
        return (-1.0 if pad_cx < body_cx else 1.0, 0.0)

    # Face-mating heuristic
    if spread_x >= spread_y:
        pad_cy = sum(ys) / len(ys)
        return (0.0, -1.0 if pad_cy > 0 else 1.0)
    pad_cx = sum(xs) / len(xs)
    return (-1.0 if pad_cx > 0 else 1.0, 0.0)


def _compute_connector_rotation(comp: "Component", edge: str) -> float:
    """Pick the rotation (0/90/180/270) that best aligns the mating face with the edge's outward normal."""
    fx, fy = _compute_pad_facing_direction(comp)
    edge_outward = {
        "right": (1.0, 0.0),
        "top": (0.0, -1.0),
        "left": (-1.0, 0.0),
        "bottom": (0.0, 1.0),
    }
    ox, oy = edge_outward[edge]

    best_theta = 0.0
    best_dot = -2.0
    for theta in (0.0, 90.0, 180.0, 270.0):
        rad = math.radians(theta)
        cos_r = math.cos(rad)
        sin_r = math.sin(rad)
        # KiCad CW: rotated vec = (fx*cos + fy*sin, -fx*sin + fy*cos)
        rx = fx * cos_r + fy * sin_r
        ry = -fx * sin_r + fy * cos_r
        dot = rx * ox + ry * oy
        if dot > best_dot:
            best_dot = dot
            best_theta = float(theta)
    return best_theta


def _along_edge_extent(comp: "Component", edge: str, mating_margin: float = 5.0) -> float:
    """Center-to-center spacing needed along the edge for this connector."""
    rot = _compute_connector_rotation(comp, edge)
    old_rot = comp.rotation
    comp.set_rotation(rot)
    w = comp.effective_width
    h = comp.effective_height
    comp.set_rotation(old_rot)
    along = w if edge in ("bottom", "top") else h
    return along + mating_margin


def place_connectors_perimeter(
    connectors: list["Component"],
    board: "BoardOutline",
    margin: float,
    min_gap: float = 2.0,
    mating_margin: float = 5.0,
) -> None:
    """Distribute connectors across the four board edges, facing outward.

    Each edge gets ~1/4 of the connectors, proportionally adjusted by
    edge length. Within each edge, connectors are spaced evenly with
    at least ``min_gap`` mm between adjacent bodies.
    """
    if not connectors:
        return

    edges = ["bottom", "right", "top", "left"]
    edge_lengths = {
        "bottom": board.x_max - board.x_min - 2 * margin,
        "right": board.y_max - board.y_min - 2 * margin,
        "top": board.x_max - board.x_min - 2 * margin,
        "left": board.y_max - board.y_min - 2 * margin,
    }

    # Sort by ref for determinism, then split into 4 edges by total extent.
    sorted_conns = sorted(connectors, key=lambda c: c.ref)
    total_extent = sum(
        _along_edge_extent(c, "bottom", mating_margin) for c in sorted_conns
    )
    if total_extent == 0:
        return

    # Distribute proportional to edge length.
    conn_per_edge: dict[str, list["Component"]] = {e: [] for e in edges}
    cumulative = 0.0
    cumulative_target = 0.0
    edge_idx = 0
    total_edge_length = sum(edge_lengths.values())
    for c in sorted_conns:
        ext = _along_edge_extent(c, edges[edge_idx], mating_margin)
        cumulative += ext
        cumulative_target = (
            sum(edge_lengths[e] for e in edges[: edge_idx + 1])
            / total_edge_length
        ) * total_extent
        conn_per_edge[edges[edge_idx]].append(c)
        if cumulative >= cumulative_target and edge_idx < len(edges) - 1:
            edge_idx += 1

    # Place each edge's connectors
    for edge in edges:
        conns = conn_per_edge[edge]
        if not conns:
            continue
        _place_on_edge(conns, edge, board, margin, min_gap, mating_margin)


def _place_on_edge(
    conns: list["Component"],
    edge: str,
    board: "BoardOutline",
    margin: float,
    min_gap: float,
    mating_margin: float,
) -> None:
    """Place connectors along a single edge, evenly spaced."""
    if not conns:
        return

    extents = [_along_edge_extent(c, edge, mating_margin) for c in conns]
    total_ext = sum(extents) + min_gap * (len(conns) - 1)

    if edge in ("bottom", "top"):
        edge_len = board.x_max - board.x_min - 2 * margin
    else:
        edge_len = board.y_max - board.y_min - 2 * margin

    # Shrink gap if total exceeds edge length
    if total_ext > edge_len and len(conns) > 1:
        available_gap = max(0.0, (edge_len - sum(extents)) / (len(conns) - 1))
        gap = available_gap
        total_ext = sum(extents) + gap * (len(conns) - 1)
    else:
        gap = min_gap

    # Start position so connectors are centered along the edge
    start_offset = max(0.0, (edge_len - total_ext) / 2)

    for conn, ext in zip(conns, extents):
        rot = _compute_connector_rotation(conn, edge)
        conn.set_rotation(rot)

        # Center along edge at start_offset + ext/2 + accumulated
        center_offset = start_offset + ext / 2
        start_offset += ext + gap

        if edge == "bottom":
            conn.x = board.x_min + margin + center_offset
            conn.y = board.y_max - margin - mating_margin / 2
        elif edge == "top":
            conn.x = board.x_min + margin + center_offset
            conn.y = board.y_min + margin + mating_margin / 2
        elif edge == "left":
            conn.x = board.x_min + margin + mating_margin / 2
            conn.y = board.y_min + margin + center_offset
        elif edge == "right":
            conn.x = board.x_max - margin - mating_margin / 2
            conn.y = board.y_min + margin + center_offset
