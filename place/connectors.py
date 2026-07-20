"""Connector perimeter placement — family-aware.

Place edge connectors on the board perimeter, facing outward. For each
connector, compute the mating direction from pad geometry (long-axis
vs short-axis), then pick the rotation that best aligns with each
edge's outward normal.

Connectors are *grouped* before placement so a human-designer layout
results: PinHeaders by footprint, SMAs by signal stem, power by family.
Groups are then assigned to edges in a **family-exclusive,
load-balanced** way — each edge holds a single family, and groups fan
out across edges so connector counts stay even (4/4/4/4 for 16
connectors, not 6/6/4/0). Within an edge, connectors are ordered by
(family rank, signal stem, channel number, along-edge centroid).

This is a port of the legacy engine/smart_placement.py grouping logic
(lines ~1410-2209), adapted to the board-perimeter frame used by the
macro-first pipeline. Chain-pair / signal-flow co-location is
intentionally NOT ported: it needs an engine.subcircuit_patterns
dependency, and on cbb every chain is cross-family so family-exclusivity
already forbids edge-sharing anyway. TODO: same-family chains (e.g.
SMA_in -> IC -> SMA_out) would not get endpoints co-located.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set

if TYPE_CHECKING:
    from models.board_model import BoardModel, Component, BoardOutline


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


# =============================================================================
# CONNECTOR GROUPING
# =============================================================================

@dataclass
class ConnectorGroup:
    """Group of related connectors that should share an edge."""
    group_id: str
    category: str  # "power", or a footprint family ("SMA", "PinHeader", ...)
    connectors: List["Component"] = field(default_factory=list)
    priority: int = 0


def _connector_name(c: "Component") -> str:
    """Get the best name for categorization: value first, then ref."""
    return c.value if c.value and c.value.strip() else c.ref


def _footprint_family(c: "Component") -> str:
    """Extract footprint family from a connector's footprint string.

    "Connector_Coaxial:SMA_Amphenol_132289_EdgeMount" -> "SMA"
    "Connector_PinHeader_2.54mm:PinHeader_1x06_P2.54mm_Horizontal" -> "PinHeader"
    "Connector_PinSocket_2.54mm:PinSocket_1x03_P2.54mm_Vertical" -> "PinSocket"
    """
    fp = getattr(c, "footprint", "") or ""
    if ":" in fp:
        fp = fp.split(":", 1)[1]
    m = re.match(r"^([A-Za-z]+)", fp)
    return m.group(1) if m else "unknown"


def signal_stem(name: str) -> str:
    """Normalize a value to its signal-role stem.

    Strip digits and non-alpha chars so channel/bus indices don't
    fragment the group: "ADC1_in" -> "adcin", "Buffer2_out" -> "bufferout".
    Shared by grouping and the within-edge sort.
    """
    s = re.sub(r"[^a-zA-Z]", "", name).lower()
    return s or "misc"


# Order in which connector families should appear along an edge so that
# visually-similar connectors cluster together. Lower rank = placed first
# (leftmost / topmost). Families not in the map get rank 9.
_FAMILY_EDGE_RANK = {
    "PinHeader":    0,
    "PinSocket":    1,
    "USB":          2,
    "BarrelJack":   3,
    "TerminalBlock":4,
    "SMA":          5,
    "Coaxial":      6,
    "BNC":          7,
    "power":        8,
}


def _family_rank(family: str) -> int:
    """Within-edge ordering rank for a connector family."""
    return _FAMILY_EDGE_RANK.get(family, 9)


def _group_connectors(connectors: List["Component"]) -> List[ConnectorGroup]:
    """Group connectors for perimeter placement.

    Strategy:
      1. Power — connectors whose value/name matches a power keyword
         (PWR, VCC, GND, +/-V rails, ...), sub-grouped by footprint family
         so a +12V SMA and a +12V PinHeader land in different groups (and
         thus can claim different edges). Each family keeps priority 100.
      2. PinHeader / PinSocket / USB families — one group per family. All
         PinHeaders together regardless of signal role.
      3. SMA / Coaxial / other coax-style families — sub-group by the stem
         of the connector value (digits/channel indices stripped). So
         "ADC1_in"..."ADC4_in" cluster together while "Buffer1_out"... form
         a separate group. This is what spreads ADC inputs and Buffer
         outputs across distinct edges.

    Downstream edge assignment prefers empty edges so groups fan out across
    all four board edges when there are <=4 groups.
    """
    groups: List[ConnectorGroup] = []
    grouped: Set[str] = set()

    def is_power(name: str) -> bool:
        kw = ["PWR", "POWER", "VCC", "VDD", "VSS", "GND", "GROUND",
              "VIN", "VOUT", "VBAT", "MAIN", "SUPPLY", "+3", "+5",
              "+12", "+24", "-5", "-12", "3V3", "5V", "12V"]
        return any(k in name.upper() for k in kw)

    # 1. Power — sub-group by family so a +12V SMA and a +12V PinHeader
    #    don't share a group (and thus don't share an edge).
    power = [c for c in connectors if is_power(_connector_name(c))]
    if power:
        power_by_fam: Dict[str, List["Component"]] = defaultdict(list)
        for c in power:
            power_by_fam[_footprint_family(c)].append(c)
        for fam, conns in sorted(power_by_fam.items()):
            groups.append(ConnectorGroup(
                group_id=f"power_{fam}",
                category="power",
                connectors=sorted(conns, key=lambda x: x.ref),
                priority=100,
            ))
        grouped.update(c.ref for c in power)

    # 2./3. Remaining connectors bucketed by footprint family.
    families: Dict[str, List["Component"]] = defaultdict(list)
    for c in connectors:
        if c.ref in grouped:
            continue
        families[_footprint_family(c)].append(c)

    family_priorities = {
        "PinHeader": 70,
        "PinSocket": 68,
        "USB":       65,
        "SMA":       60,
        "Coaxial":   58,
    }

    # Families where footprint dominates signal role — keep as one group.
    footprint_dominant = {"PinHeader", "PinSocket", "USB"}

    for family, conns in families.items():
        pri = family_priorities.get(family, 50)

        if family in footprint_dominant or len(conns) == 1:
            groups.append(ConnectorGroup(
                group_id=f"family_{family}",
                category=family,
                connectors=sorted(conns, key=lambda x: x.ref),
                priority=pri,
            ))
        else:
            # SMA/Coaxial/etc.: sub-group by signal stem so different signal
            # roles (ADC in vs Buffer out) land on different edges.
            stems: Dict[str, List["Component"]] = defaultdict(list)
            for c in conns:
                stems[signal_stem(_connector_name(c))].append(c)
            for stem, stem_conns in sorted(stems.items()):
                groups.append(ConnectorGroup(
                    group_id=f"{family}_{stem}",
                    category=family,
                    connectors=sorted(stem_conns, key=lambda x: x.ref),
                    priority=pri,
                ))

    return sorted(groups, key=lambda g: g.priority, reverse=True)


def _group_family_for_edge(g: "ConnectorGroup") -> str:
    """Return the footprint family of a group, for edge-exclusivity.

    Power groups carry their family in the group_id ("power_SMA" -> "SMA")
    so power and signal SMAs CAN share an edge on boards with >4 of a
    family. Non-power groups use their category directly.
    """
    cat = g.category or ""
    if cat == "power":
        gid = g.group_id
        if gid.startswith("power_"):
            return gid.split("_", 1)[1]
        return "power"
    return cat


def _resolve_corners(
    connectors: List["Component"],
    comp_edge: Dict[str, str],
    board: "BoardOutline",
    margin: float,
    gap: float,
) -> None:
    """Push connectors at corners apart to resolve overlaps."""
    adjacent = {frozenset(["top", "left"]), frozenset(["top", "right"]),
                frozenset(["bottom", "left"]), frozenset(["bottom", "right"])}

    for _ in range(30):
        resolved = True

        for i, c1 in enumerate(connectors):
            e1 = comp_edge.get(c1.ref)
            if not e1:
                continue

            for c2 in connectors[i + 1:]:
                e2 = comp_edge.get(c2.ref)
                if not e2 or not c1.overlaps(c2):
                    continue
                if frozenset([e1, e2]) not in adjacent:
                    continue

                a = c1.bbox
                b = c2.bbox
                ox = min(a[2], b[2]) - max(a[0], b[0])
                oy = min(a[3], b[3]) - max(a[1], b[1])

                # Push in bbox-center space via set_bbox_center so the BODY
                # (not the origin) moves and stays clamped inside the board.
                for comp, edge, other in [(c1, e1, c2), (c2, e2, c1)]:
                    cb = comp.bbox
                    bcx = (cb[0] + cb[2]) / 2
                    bcy = (cb[1] + cb[3]) / 2
                    ob = other.bbox
                    if edge in ("top", "bottom"):
                        sign = 1.0 if bcx > (ob[0] + ob[2]) / 2 else -1.0
                        new_bcx = max(
                            board.x_min + comp.effective_width / 2,
                            min(bcx + (ox + gap) * sign,
                                board.x_max - comp.effective_width / 2),
                        )
                        comp.set_bbox_center(new_bcx, bcy, comp.rotation)
                    else:
                        sign = 1.0 if bcy > (ob[1] + ob[3]) / 2 else -1.0
                        new_bcy = max(
                            board.y_min + comp.effective_height / 2,
                            min(bcy + (oy + gap) * sign,
                                board.y_max - comp.effective_height / 2),
                        )
                        comp.set_bbox_center(bcx, new_bcy, comp.rotation)

                resolved = False

        if resolved:
            break


# =============================================================================
# CONNECTOR PERIMETER PLACEMENT
# =============================================================================

def place_connectors_perimeter(
    model: "BoardModel",
    connectors: list["Component"],
    board: "BoardOutline",
    margin: float,
    min_gap: float = 2.0,
    mating_margin: float = 5.0,
) -> None:
    """Distribute connectors across the four board edges, facing outward.

    Connectors are grouped (PinHeaders by footprint, SMAs by signal stem,
    power by family) and assigned to edges in a family-exclusive,
    load-balanced way: each edge gets ONE family — never mix SMA with
    PinHeader — and same-family groups fan out across edges so connector
    counts stay even (4/4/4/4 for 16 connectors). Within an edge,
    connectors are ordered by (family rank, signal stem, channel number,
    along-edge net-centroid) so logically-related connectors cluster.
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
    gap = max(1.5, min_gap)

    # ── 1. Group connectors ──────────────────────────────────────────
    groups = _group_connectors(connectors)
    if not groups:
        return

    # ── 2. Capacity-split any group wider than the longest edge ──────
    max_edge_len = max(edge_lengths.values()) if edge_lengths else 0.0

    def _max_per_connector_extent(c: "Component") -> float:
        return max(_along_edge_extent(c, e, mating_margin) for e in edges)

    def _split_for_capacity(group: "ConnectorGroup") -> List["ConnectorGroup"]:
        if not group.connectors:
            return [group]
        total = sum(_max_per_connector_extent(c) for c in group.connectors) \
                + gap * (len(group.connectors) - 1)
        if total <= max_edge_len + 0.5:
            return [group]
        # Sort by family first (keep same-family in the same chunk), then ref.
        sorted_conns = sorted(group.connectors,
                              key=lambda c: (_footprint_family(c), c.ref))
        chunks: List[List["Component"]] = []
        current: List["Component"] = []
        current_ext = 0.0
        for c in sorted_conns:
            c_ext = _max_per_connector_extent(c)
            add = c_ext + (gap if current else 0.0)
            if current and current_ext + add > max_edge_len:
                chunks.append(current)
                current = [c]
                current_ext = c_ext
            else:
                current.append(c)
                current_ext += add
        if current:
            chunks.append(current)
        return [ConnectorGroup(
            group_id=f"{group.group_id}_part{i + 1}",
            category=group.category,
            connectors=chunk,
            priority=group.priority,
        ) for i, chunk in enumerate(chunks)]

    split_groups: List[ConnectorGroup] = []
    for group in groups:
        split_groups.extend(_split_for_capacity(group))
    groups = split_groups

    # ── 3. Net-weighted centroid per connector (board-center fallback) ──
    board_cx = (board.x_min + board.x_max) / 2
    board_cy = (board.y_min + board.y_max) / 2

    conn_centroids: Dict[str, tuple[float, float]] = {}
    for comp in connectors:
        cx, cy, count = 0.0, 0.0, 0
        for net in model.nets:
            if comp.ref not in net.component_refs:
                continue
            for ref, _ in net.pins:
                if ref == comp.ref:
                    continue
                other = model.get_component(ref)
                if other:
                    cx += other.x
                    cy += other.y
                    count += 1
        if count > 0:
            conn_centroids[comp.ref] = (cx / count, cy / count)
        else:
            conn_centroids[comp.ref] = (board_cx, board_cy)

    # ── 4. Edge midpoints from the board outline ─────────────────────
    edge_midpoints = {
        "bottom": ((board.x_min + board.x_max) / 2, board.y_max),
        "right": (board.x_max, (board.y_min + board.y_max) / 2),
        "top": ((board.x_min + board.x_max) / 2, board.y_min),
        "left": (board.x_min, (board.y_min + board.y_max) / 2),
    }

    def _group_extent(group: "ConnectorGroup", edge: str) -> float:
        """Total along-edge extent of a group on a given edge, incl. intra-group gaps."""
        if not group.connectors:
            return 0.0
        total = sum(_along_edge_extent(c, edge, mating_margin)
                    for c in group.connectors)
        total += gap * (len(group.connectors) - 1)
        return total

    def _group_centroid(group: "ConnectorGroup") -> tuple[float, float]:
        if not group.connectors:
            return (board_cx, board_cy)
        sx = sum(conn_centroids[c.ref][0] for c in group.connectors)
        sy = sum(conn_centroids[c.ref][1] for c in group.connectors)
        n = len(group.connectors)
        return (sx / n, sy / n)

    # ── 5. Family-exclusive + load-balanced edge assignment ──────────
    # Two human-designer rules:
    #   1. Each edge gets ONE family — never mix SMA with PinHeader.
    #   2. Connectors distribute EVENLY across edges.
    # Among family-compatible edges (empty OR same-family) that fit, pick
    # the one with the fewest connectors (tiebreak: HPWL proximity). Lock
    # the chosen edge to that family. Overflow to the least-loaded edge
    # only as a last resort.
    edge_assigned_groups: Dict[str, List[int]] = {e: [] for e in edges}
    edge_used: Dict[str, float] = {e: 0.0 for e in edges}
    edge_conn_count: Dict[str, int] = {e: 0 for e in edges}
    edge_family: Dict[str, Optional[str]] = {e: None for e in edges}

    group_order = sorted(
        range(len(groups)),
        key=lambda gi: (groups[gi].priority, len(groups[gi].connectors)),
        reverse=True,
    )

    for gi in group_order:
        group = groups[gi]
        gcx, gcy = _group_centroid(group)
        group_n = len(group.connectors)
        group_fam = _group_family_for_edge(group)

        def _prox(edge: str) -> float:
            return math.hypot(gcx - edge_midpoints[edge][0],
                              gcy - edge_midpoints[edge][1])

        def _fits(edge: str) -> bool:
            return _group_extent(group, edge) <= edge_lengths[edge] - edge_used[edge] + 0.5

        def _family_ok(edge: str) -> bool:
            ef = edge_family[edge]
            return ef is None or ef == group_fam

        # Family-compatible edges (empty OR same family) that fit.
        compat = [e for e in edges if _family_ok(e) and _fits(e)]
        if compat:
            # Primary: connector count (load balance); Secondary: proximity.
            compat.sort(key=lambda e: (edge_conn_count[e], _prox(e)))
            assigned_edge: str = compat[0]
        else:
            # Last resort: overflow to least-loaded edge (may break exclusivity).
            assigned_edge = min(edges, key=lambda e: edge_used[e])

        edge_used[assigned_edge] += _group_extent(group, assigned_edge) + gap
        edge_assigned_groups[assigned_edge].append(gi)
        edge_conn_count[assigned_edge] += group_n
        if edge_family[assigned_edge] is None:
            edge_family[assigned_edge] = group_fam

    # ── 6. Per-edge placement: single logical sort, then _place_on_edge ─
    comp_edge: Dict[str, str] = {}
    for edge in edges:
        assigned_groups = edge_assigned_groups[edge]
        if not assigned_groups:
            continue

        comps_on_edge: List["Component"] = []
        for gi in assigned_groups:
            comps_on_edge.extend(groups[gi].connectors)

        # One sort by a logical key: (family rank, signal stem, channel
        # number, along-edge net-centroid). Produces ADC1,ADC2,ADC3,ADC4
        # together, then Buffer1..4 together — no interleaving.
        def _signal_sort_key(c: "Component") -> tuple:
            name = _connector_name(c)
            stem = signal_stem(name)
            num_match = re.search(r"\d+", name)
            chan = int(num_match.group()) if num_match else 999
            fam = _footprint_family(c)
            centroid = conn_centroids[c.ref][0] if edge in ("bottom", "top") \
                else conn_centroids[c.ref][1]
            return (_family_rank(fam), stem, chan, centroid)

        comps_on_edge.sort(key=_signal_sort_key)

        for c in comps_on_edge:
            comp_edge[c.ref] = edge

        _place_on_edge(comps_on_edge, edge, board, margin, gap, mating_margin)

    # ── 7. Resolve corner overlaps ───────────────────────────────────
    _resolve_corners(connectors, comp_edge, board, margin, gap)


def _place_on_edge(
    conns: list["Component"],
    edge: str,
    board: "BoardOutline",
    margin: float,
    min_gap: float,
    mating_margin: float,
) -> None:
    """Place connectors along a single edge, evenly spaced (order preserved)."""
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

        # Along-edge bbox-center position. ``ext`` already includes
        # ``mating_margin``, so centering on ``start_offset + ext/2`` leaves
        # ``mating_margin/2`` clearance to each slot boundary — adjacent
        # connector *bodies* end up ``mating_margin`` apart (plus ``gap``).
        center_offset = start_offset + ext / 2
        start_offset += ext + gap

        # Perpendicular: edge connectors OVERHANG the board edge — the body
        # extends OUTSIDE the outline and only the pad/lead area sits inside.
        # Place the bbox so its inward face is ``mating_margin`` inside the
        # board (pads on the board), letting the body overhang past the edge.
        # This is why connectors consume almost no interior room: their
        # inside footprint is just the pad depth, not the full body. Position
        # via set_bbox_center so the BODY lands here, not the origin (pin
        # headers / edge-mount SMAs have origin ≠ bbox center).
        if edge == "bottom":
            target_cx = board.x_min + margin + center_offset
            target_cy = board.y_max - mating_margin + conn.effective_height / 2
        elif edge == "top":
            target_cx = board.x_min + margin + center_offset
            target_cy = board.y_min + mating_margin - conn.effective_height / 2
        elif edge == "left":
            target_cx = board.x_min + mating_margin - conn.effective_width / 2
            target_cy = board.y_min + margin + center_offset
        elif edge == "right":
            target_cx = board.x_max - mating_margin + conn.effective_width / 2
            target_cy = board.y_min + margin + center_offset

        conn.set_bbox_center(target_cx, target_cy, rot)
