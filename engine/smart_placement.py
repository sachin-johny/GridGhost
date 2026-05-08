"""Smart PCB placement engine.

Places interior components first, then positions connectors
on the perimeter of the interior component cluster, facing outward.

Orientation is computed from pad geometry rather than hardcoded,
ensuring connectors face the correct direction on every edge.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from models.board_model import BoardModel, Component, BoardOutline


# =============================================================================
# INTERIOR BBOX COMPUTATION
# =============================================================================

def _compute_interior_bbox(
    interior: List["Component"],
    board: "BoardOutline",
    margin: float,
) -> tuple[float, float, float, float]:
    """Bounding box around all placed interior components.

    Falls back to board outline minus margin if no interior components.
    """
    if not interior:
        return (board.x_min + margin, board.y_min + margin,
                board.x_max - margin, board.y_max - margin)

    ib_x_min = min(c.bbox[0] for c in interior)
    ib_y_min = min(c.bbox[1] for c in interior)
    ib_x_max = max(c.bbox[2] for c in interior)
    ib_y_max = max(c.bbox[3] for c in interior)

    return (ib_x_min, ib_y_min, ib_x_max, ib_y_max)


# =============================================================================
# PAD-BASED FACING DIRECTION
# =============================================================================

def _compute_pad_facing_direction(comp: "Component") -> tuple[float, float]:
    """Analyze pad positions to find connector's facing direction in local coords.

    The pad column is the axis with more pad spread. The connector faces
    perpendicular to the pad column (toward the mating interface).
    """
    if len(comp.pads) < 2:
        return (1.0, 0.0)

    xs = [p.x for p in comp.pads]
    ys = [p.y for p in comp.pads]
    spread_x = max(xs) - min(xs)
    spread_y = max(ys) - min(ys)

    if spread_x >= spread_y:
        # Pad column along X, face in Y direction
        pad_cy = sum(ys) / len(ys)
        fy = -1.0 if pad_cy > 0 else 1.0
        return (0.0, fy)
    else:
        # Pad column along Y, face in X direction
        pad_cx = sum(xs) / len(xs)
        fx = -1.0 if pad_cx > 0 else 1.0
        return (fx, 0.0)


# =============================================================================
# CONNECTOR ROTATION COMPUTATION
# =============================================================================

def _compute_connector_rotation(comp: "Component", edge: str) -> float:
    """Compute rotation so connector's facing direction points outward.

    Tries all 4 discrete rotations (0/90/180/270) and picks the one
    that best aligns the local facing direction with the edge's outward normal.
    """
    fx, fy = _compute_pad_facing_direction(comp)

    # Outward normals for each edge of the interior bbox
    edge_outward = {
        "right":  (1.0, 0.0),
        "top":    (0.0, -1.0),   # y_min side, outward = -Y
        "left":   (-1.0, 0.0),
        "bottom": (0.0, 1.0),    # y_max side, outward = +Y
    }
    ox, oy = edge_outward[edge]

    best_theta = 0.0
    best_dot = -2.0

    for theta in [0, 90, 180, 270]:
        rad = math.radians(theta)
        cos_r = math.cos(rad)
        sin_r = math.sin(rad)
        # CW rotation: [[cos, sin], [-sin, cos]]
        rx = fx * cos_r + fy * sin_r
        ry = -fx * sin_r + fy * cos_r
        dot = rx * ox + ry * oy
        if dot > best_dot:
            best_dot = dot
            best_theta = float(theta)

    return best_theta


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def smart_grid_place(
    model: "BoardModel",
    margin: float = 5.0,
    spacing_factor: float = 1.3,
) -> "BoardModel":
    """Smart PCB placement: interior first, connectors around interior bbox perimeter.

    Args:
        model: BoardModel with components to place
        margin: Margin from board edges in mm
        spacing_factor: Spacing multiplier for interior components

    Returns:
        BoardModel with all components placed
    """
    # Separate components
    connectors = [c for c in model.components
                  if getattr(c, 'component_type', '') == "connector" and not c.is_fixed]
    interior = [c for c in model.components
                if getattr(c, 'component_type', '') != "connector" and not c.is_fixed]

    # Step 1: Place interior components (full board area)
    if interior:
        _place_interior(model, interior, margin, spacing_factor)

    # Step 2: Compute bounding box around interior components
    ib = _compute_interior_bbox(interior, model.board, margin)

    # Step 3: Place connectors on perimeter of interior bbox
    if connectors:
        _place_connectors_perimeter(model, connectors, margin, ib)

    # Step 4: Resolve any remaining overlaps (connectors fixed, interior pushed inward)
    _resolve_all_overlaps(model, margin)

    return model


# =============================================================================
# INTERIOR PLACEMENT
# =============================================================================

def _place_interior(
    model: "BoardModel",
    interior: List["Component"],
    margin: float,
    spacing_factor: float,
) -> None:
    """Place interior components in efficient grid layout across full board area."""
    board = model.board

    x_min = board.x_min + margin
    x_max = board.x_max - margin
    y_min = board.y_min + margin
    y_max = board.y_max - margin

    n = len(interior)
    region_w = x_max - x_min
    region_h = y_max - y_min

    # Aspect-ratio aware grid
    aspect = region_w / max(region_h, 1e-9)
    cols = max(1, min(n, int(round(math.sqrt(n * aspect)))))
    rows = max(1, int(math.ceil(n / cols)))

    cell_w = region_w / cols
    cell_h = region_h / rows

    # Sort by size (larger first for centering)
    interior_sorted = sorted(interior, key=lambda c: c.effective_width * c.effective_height, reverse=True)

    for idx, comp in enumerate(interior_sorted):
        col = idx % cols
        row = idx // cols

        # Center of cell
        cx = x_min + (col + 0.5) * cell_w
        cy = y_min + (row + 0.5) * cell_h

        # Offset for odd rows
        if row % 2 == 1:
            cx += cell_w * 0.08

        # Clamp to bounds
        w2 = comp.effective_width / 2
        h2 = comp.effective_height / 2
        comp.x = max(x_min + w2, min(cx, x_max - w2))
        comp.y = max(y_min + h2, min(cy, y_max - h2))

    # Apply repulsion
    _apply_repulsion(interior, x_min, x_max, y_min, y_max, spacing_factor)


def _apply_repulsion(
    components: List["Component"],
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    spacing_factor: float,
) -> None:
    """Push components apart to reduce overlaps."""
    for iteration in range(100):
        moved = False

        for i, ca in enumerate(components):
            for cb in components[i + 1:]:
                dx = cb.x - ca.x
                dy = cb.y - ca.y
                dist = math.sqrt(dx * dx + dy * dy)

                min_dist = max(
                    ca.effective_width, ca.effective_height,
                    cb.effective_width, cb.effective_height
                ) * spacing_factor * 0.5

                if dist < min_dist:
                    if dist < 0.1:
                        angle = (ca.x + ca.y) * 0.5
                        dx, dy = math.cos(angle), math.sin(angle)
                    else:
                        dx, dy = dx / dist, dy / dist

                    push = (min_dist - dist) * 0.5 + 0.3

                    ca.x = max(x_min + ca.effective_width/2,
                              min(ca.x - push * dx, x_max - ca.effective_width/2))
                    ca.y = max(y_min + ca.effective_height/2,
                              min(ca.y - push * dy, y_max - ca.effective_height/2))
                    cb.x = max(x_min + cb.effective_width/2,
                              min(cb.x + push * dx, x_max - cb.effective_width/2))
                    cb.y = max(y_min + cb.effective_height/2,
                              min(cb.y + push * dy, y_max - cb.effective_height/2))
                    moved = True

        if not moved:
            break


# =============================================================================
# CONNECTOR GROUPING
# =============================================================================

@dataclass
class ConnectorGroup:
    """Group of related connectors."""
    group_id: str
    category: str  # "power", "input", "output", "data", "other"
    connectors: List["Component"] = field(default_factory=list)
    priority: int = 0

    @property
    def total_width(self) -> float:
        if not self.connectors:
            return 0.0
        gap = 1.5
        return sum(max(c.effective_width, c.effective_height)
                   for c in self.connectors) + gap * (len(self.connectors) - 1)


def _connector_name(c: "Component") -> str:
    """Get the best name for categorization: value first, then ref."""
    return c.value if c.value and c.value.strip() else c.ref


def _group_connectors(connectors: List["Component"]) -> List[ConnectorGroup]:
    """Group connectors by category and relationships.

    Uses component value (e.g. "+12V OPA2227P", "ADC3_in") for categorization,
    falling back to ref (e.g. "J1") if value is empty.
    """
    groups = []
    grouped: Set[str] = set()

    def get_voltage(name: str) -> Optional[float]:
        m = re.search(r'([+-]?\d+(?:\.\d+)?)\s*V', name, re.IGNORECASE)
        return float(m.group(1)) if m else None

    def get_polarity(name: str) -> Optional[str]:
        if re.search(r'[+]\s*\d', name): return "positive"
        if re.search(r'[-]\s*\d', name): return "negative"
        return None

    def is_power(name: str) -> bool:
        kw = ['PWR', 'POWER', 'VCC', 'VDD', 'VSS', 'GND', 'GROUND',
              'VIN', 'VOUT', 'VBAT', 'MAIN', 'SUPPLY', '+3', '+5',
              '+12', '+24', '-5', '-12', '3V3', '5V', '12V']
        return any(k in name.upper() for k in kw)

    def is_input(name: str) -> bool:
        name_u = name.upper()
        if re.search(r'_IN\b', name_u): return True
        if re.search(r'\bINPUT\b', name_u): return True
        if re.search(r'\bRX\b', name_u): return True
        return False

    def is_output(name: str) -> bool:
        name_u = name.upper()
        if re.search(r'_OUT\b', name_u): return True
        if re.search(r'\bOUTPUT\b', name_u): return True
        if re.search(r'\bTX\b', name_u): return True
        return False

    def is_data(name: str) -> bool:
        kw = ['DATA', 'SDA', 'SCL', 'SPI', 'I2C', 'UART', 'USB', 'CAN', 'ETH', 'JTAG',
              'ADC', 'DAC']
        return any(k in name.upper() for k in kw)

    # 1. Power pairs (+V/-V)
    voltage_map: Dict[float, List] = defaultdict(list)
    for c in connectors:
        name = _connector_name(c)
        v, p = get_voltage(name), get_polarity(name)
        if v is not None and p:
            voltage_map[abs(v)].append((c, p))

    for v, conns in voltage_map.items():
        pos = [c for c, p in conns if p == "positive"]
        neg = [c for c, p in conns if p == "negative"]
        for p, n in zip(pos, neg):
            groups.append(ConnectorGroup(
                group_id=f"power_pair_{v}V",
                category="power",
                connectors=[p, n],
                priority=100
            ))
            grouped.update([p.ref, n.ref])

    # 2. Remaining power
    power = [c for c in connectors if c.ref not in grouped and is_power(_connector_name(c))]
    if power:
        groups.append(ConnectorGroup("power_other", "power", sorted(power, key=lambda x: x.ref), 90))
        grouped.update(c.ref for c in power)

    # 3. Inputs
    inputs = [c for c in connectors if c.ref not in grouped and is_input(_connector_name(c))]
    if inputs:
        groups.append(ConnectorGroup("inputs", "input", sorted(inputs, key=lambda x: x.ref), 80))
        grouped.update(c.ref for c in inputs)

    # 4. Outputs
    outputs = [c for c in connectors if c.ref not in grouped and is_output(_connector_name(c))]
    if outputs:
        groups.append(ConnectorGroup("outputs", "output", sorted(outputs, key=lambda x: x.ref), 70))
        grouped.update(c.ref for c in outputs)

    # 5. Data
    data = [c for c in connectors if c.ref not in grouped and is_data(_connector_name(c))]
    if data:
        groups.append(ConnectorGroup("data", "data", sorted(data, key=lambda x: x.ref), 60))
        grouped.update(c.ref for c in data)

    # 6. Other
    other = [c for c in connectors if c.ref not in grouped]
    if other:
        groups.append(ConnectorGroup("other", "other", sorted(other, key=lambda x: x.ref), 50))

    return sorted(groups, key=lambda g: g.priority, reverse=True)


# =============================================================================
# CONNECTOR PERIMETER PLACEMENT
# =============================================================================

def _place_connectors_perimeter(
    model: "BoardModel",
    connectors: List["Component"],
    margin: float,
    interior_bbox: tuple[float, float, float, float],
) -> None:
    """Place connectors on perimeter of interior bbox, facing outward.

    Strategy:
    1. Group connectors by category (power pairs, input, output, data, etc.)
    2. Compute net-weighted center for each group
    3. Divide into 4 batches, assign each to nearest edge
    4. Place with per-connector rotation from pad analysis
    """
    ib_x_min, ib_y_min, ib_x_max, ib_y_max = interior_bbox
    board = model.board
    gap = 1.5
    conn_margin = 5.0

    groups = _group_connectors(connectors)
    if not groups:
        return

    edges = ["bottom", "right", "top", "left"]
    edge_lengths = {
        "bottom": ib_x_max - ib_x_min,
        "right": ib_y_max - ib_y_min,
        "top": ib_x_max - ib_x_min,
        "left": ib_y_max - ib_y_min,
    }

    # Edge midpoints on interior bbox for proximity scoring
    edge_midpoints = {
        "bottom": ((ib_x_min + ib_x_max) / 2, ib_y_max),
        "right": (ib_x_max, (ib_y_min + ib_y_max) / 2),
        "top": ((ib_x_min + ib_x_max) / 2, ib_y_min),
        "left": (ib_x_min, (ib_y_min + ib_y_max) / 2),
    }

    # Compute net-weighted center for each group
    group_centers = []
    for group in groups:
        cx, cy, count = 0.0, 0.0, 0
        for comp in group.connectors:
            for net in model.nets:
                conn_refs = {c.ref for c in group.connectors}
                has_conn = any(ref in conn_refs for ref, _ in net.pins)
                if not has_conn:
                    continue
                for ref, _ in net.pins:
                    if ref in conn_refs:
                        continue
                    other = model.get_component(ref)
                    if other:
                        cx += other.x
                        cy += other.y
                        count += 1
                        break
        if count > 0:
            group_centers.append((cx / count, cy / count))
        else:
            group_centers.append(((ib_x_min + ib_x_max) / 2,
                                  (ib_y_min + ib_y_max) / 2))

    # Divide groups into 4 batches of ~N/4 connectors
    sorted_groups = sorted(enumerate(groups),
                           key=lambda x: len(x[1].connectors), reverse=True)

    batches: List[List[int]] = [[] for _ in range(4)]
    batch_counts = [0] * 4

    for gi, group in sorted_groups:
        g_count = len(group.connectors)
        best_batch = min(range(4), key=lambda b: batch_counts[b])
        batches[best_batch].append(gi)
        batch_counts[best_batch] += g_count

    # Assign each batch to the best edge based on net proximity
    edge_taken: Set[int] = set()
    batch_edge: Dict[int, str] = {}

    batch_centers = []
    for bi, batch in enumerate(batches):
        if not batch:
            batch_centers.append(None)
            continue
        tcx, tcy, tc = 0.0, 0.0, 0
        for gi in batch:
            gcx, gcy = group_centers[gi]
            tcx += gcx * len(groups[gi].connectors)
            tcy += gcy * len(groups[gi].connectors)
            tc += len(groups[gi].connectors)
        batch_centers.append((tcx / tc, tcy / tc) if tc > 0 else
                             ((ib_x_min + ib_x_max) / 2,
                              (ib_y_min + ib_y_max) / 2))

    for bi in sorted(range(4), key=lambda b: -batch_counts[b]):
        if batch_centers[bi] is None:
            continue
        bcx, bcy = batch_centers[bi]

        scored = []
        for ei, edge in enumerate(edges):
            if ei in edge_taken:
                continue
            ex, ey = edge_midpoints[edge]
            dist = math.hypot(bcx - ex, bcy - ey)
            scored.append((dist, ei, edge))

        scored.sort()
        for _, ei, edge in scored:
            batch_width = sum(groups[gi].total_width for gi in batches[bi]) + gap * max(0, len(batches[bi]) - 1)
            if batch_width <= edge_lengths[edge] or not edge_taken:
                batch_edge[bi] = edge
                edge_taken.add(ei)
                break
        else:
            for ei, edge in enumerate(edges):
                if ei not in edge_taken:
                    batch_edge[bi] = edge
                    edge_taken.add(ei)
                    break

    # Handle empty batches that didn't get assigned
    for bi in range(4):
        if bi not in batch_edge and batches[bi]:
            for ei, edge in enumerate(edges):
                if ei not in edge_taken:
                    batch_edge[bi] = edge
                    edge_taken.add(ei)
                    break

    # Place connectors on assigned edges
    comp_edge: Dict[str, str] = {}

    for bi, batch in enumerate(batches):
        if not batch or bi not in batch_edge:
            continue

        edge = batch_edge[bi]
        available = edge_lengths[edge]

        batch_groups = [groups[gi] for gi in batch]
        total_needed = sum(g.total_width for g in batch_groups) + gap * max(0, len(batch_groups) - 1)

        offset = max(0.0, (available - total_needed) / 2.0)
        pos = offset

        for group in batch_groups:
            for comp in group.connectors:
                # Compute per-connector rotation from pad geometry
                rot = _compute_connector_rotation(comp, edge)
                comp.set_rotation(rot)

                w = comp.effective_width   # X-extent after rotation
                h = comp.effective_height  # Y-extent after rotation

                if edge == "bottom":
                    cx = ib_x_min + pos + w / 2
                    cy = ib_y_max + conn_margin + h / 2
                elif edge == "top":
                    cx = ib_x_min + pos + w / 2
                    cy = ib_y_min - conn_margin - h / 2
                elif edge == "left":
                    cx = ib_x_min - conn_margin - w / 2
                    cy = ib_y_min + pos + h / 2
                else:  # right
                    cx = ib_x_max + conn_margin + w / 2
                    cy = ib_y_min + pos + h / 2

                # Clamp to board bounds
                comp.x = max(board.x_min + w / 2, min(cx, board.x_max - w / 2))
                comp.y = max(board.y_min + h / 2, min(cy, board.y_max - h / 2))

                comp_edge[comp.ref] = edge

                # Advance position by the along-edge extent
                if edge in ("bottom", "top"):
                    pos += w + gap
                else:
                    pos += h + gap

            pos += gap * 0.5

    _resolve_corners(connectors, comp_edge, board, margin, gap)


def _resolve_corners(
    connectors: List["Component"],
    comp_edge: Dict[str, str],
    board,
    margin: float,
    gap: float,
) -> None:
    """Push connectors at corners to resolve overlaps."""
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

                for comp, edge, other in [(c1, e1, c2), (c2, e2, c1)]:
                    if edge in ("top", "bottom"):
                        push = (ox + gap) * (1 if comp.x > other.x else -1)
                        new_x = comp.x + push
                        comp.x = max(board.x_min + comp.effective_width/2,
                                    min(new_x, board.x_max - comp.effective_width/2))
                    else:
                        push = (oy + gap) * (1 if comp.y > other.y else -1)
                        new_y = comp.y + push
                        comp.y = max(board.y_min + comp.effective_height/2,
                                    min(new_y, board.y_max - comp.effective_height/2))

                resolved = False

        if resolved:
            break


# =============================================================================
# FINAL OVERLAP RESOLUTION
# =============================================================================

def _resolve_all_overlaps(model: "BoardModel", margin: float) -> None:
    """Final pass to resolve overlaps. Connectors are fixed, only interior components move."""
    connector_ids = {id(c) for c in model.components
                     if getattr(c, 'component_type', '') == "connector"}
    board = model.board

    for iteration in range(50):
        overlap_found = False

        for i, c1 in enumerate(model.components):
            if c1.is_fixed:
                continue
            c1_is_conn = id(c1) in connector_ids

            for c2 in model.components[i + 1:]:
                if c2.is_fixed:
                    continue
                if not c1.overlaps(c2):
                    continue

                c2_is_conn = id(c2) in connector_ids
                if c1_is_conn and c2_is_conn:
                    continue

                overlap_found = True
                dx = c2.x - c1.x
                dy = c2.y - c1.y
                dist = math.sqrt(dx * dx + dy * dy)

                if dist < 0.1:
                    dx, dy = 1.0, 0.0
                else:
                    dx, dy = dx / dist, dy / dist

                push = 0.5

                if c1_is_conn:
                    c2.x = max(board.x_min + c2.effective_width/2,
                              min(c2.x + push * dx * 2, board.x_max - c2.effective_width/2))
                    c2.y = max(board.y_min + c2.effective_height/2,
                              min(c2.y + push * dy * 2, board.y_max - c2.effective_height/2))
                elif c2_is_conn:
                    c1.x = max(board.x_min + c1.effective_width/2,
                              min(c1.x - push * dx * 2, board.x_max - c1.effective_width/2))
                    c1.y = max(board.y_min + c1.effective_height/2,
                              min(c1.y - push * dy * 2, board.y_max - c1.effective_height/2))
                else:
                    c1.x = max(board.x_min + c1.effective_width/2,
                              min(c1.x - push * dx, board.x_max - c1.effective_width/2))
                    c1.y = max(board.y_min + c1.effective_height/2,
                              min(c1.y - push * dy, board.y_max - c1.effective_height/2))
                    c2.x = max(board.x_min + c2.effective_width/2,
                              min(c2.x + push * dx, board.x_max - c2.effective_width/2))
                    c2.y = max(board.y_min + c2.effective_height/2,
                              min(c2.y + push * dy, board.y_max - c2.effective_height/2))

        if not overlap_found:
            break


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ["smart_grid_place"]
