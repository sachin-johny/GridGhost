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
from typing import Optional, List, Dict, Set, Tuple, TYPE_CHECKING

from engine.net_clustering import cluster_components
from engine.cost_state import _is_power_net
from engine.quadratic_placement import quadratic_place
from engine.congestion import rudy_congestion_penalty

if TYPE_CHECKING:
    from models.board_model import BoardModel, Component, BoardOutline


# =============================================================================
# VERTICAL CONNECTOR DETECTION
# =============================================================================

def _is_vertical_connector(comp: "Component") -> bool:
    """Return True if connector should be treated as interior (vertical/THT)."""
    fp  = getattr(comp, 'footprint', '') or ''
    val = getattr(comp, 'value', '')    or ''
    name = fp + ' ' + val
    if re.search(r'Horizontal|Angled|Side', name, re.IGNORECASE):
        return False
    if re.search(r'Vertical|THT', name, re.IGNORECASE):
        return True
    return not re.search(r'Horizontal|Angled|Edge|Side', name, re.IGNORECASE)


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

def pad_bbox(comp: "Component", pad_margin: float = 0.2) -> tuple[float, float, float, float]:
    """Bounding box around all pads in board coords, with small margin.

    Used for connectors: pads must stay inside board, body can overhang.
    """
    if not comp.pads:
        # Fall back to component bbox if no pads
        return comp.bbox

    xs, ys = [], []
    for p in comp.pads:
        px, py = p.absolute_pos(comp.x, comp.y, comp.rotation)
        xs.append(px)
        ys.append(py)
    
    # xs and ys provide pad centers; add offset for edge location
    pad_edge_margin = 3.5

    min_x = min(xs) - pad_edge_margin - pad_margin
    max_x = max(xs) + pad_edge_margin + pad_margin
    min_y = min(ys) - pad_edge_margin - pad_margin
    max_y = max(ys) + pad_edge_margin + pad_margin

    return (min_x, min_y, max_x, max_y)

# =============================================================================
# ROTATION-AWARE CONNECTOR SPACING
# =============================================================================

def _connector_along_edge_extent(
    comp: "Component", edge: str, mating_margin: float = 5.0,
) -> float:
    """Compute the center-to-center spacing a connector needs along an edge.

    Accounts for rotation: after computing the best rotation for the edge,
    the along-edge extent is the dimension parallel to that edge, plus
    mating_margin for physical clearance.
    """
    rot = _compute_connector_rotation(comp, edge)
    old_rot = getattr(comp, 'rotation', 0.0)
    comp.set_rotation(rot)
    w = comp.effective_width
    h = comp.effective_height
    comp.set_rotation(old_rot)

    along = w if edge in ("bottom", "top") else h
    return along + mating_margin


def _estimate_min_edge_space(
    connectors: List["Component"], mating_margin: float = 5.0,
) -> float:
    """Estimate minimum space needed on one edge for N/4 connectors.

    Uses average along-edge extent (sampling all 4 edges) since exact
    edge assignment isn't known yet.
    """
    if not connectors:
        return 0.0
    total = 0.0
    for c in connectors:
        extents = [_connector_along_edge_extent(c, e, mating_margin)
                   for e in ("bottom", "right", "top", "left")]
        total += sum(extents) / 4.0
    return total / 2


def _max_connector_depth(connectors: List["Component"]) -> float:
    """Max perpendicular extent across all connectors on any edge.

    Tries each connector at each edge rotation to find the worst-case
    depth the connector zone must reserve.
    """
    max_depth = 0.0
    for conn in connectors:
        for edge in ("bottom", "top", "left", "right"):
            rot = _compute_connector_rotation(conn, edge)
            old = conn.rotation
            conn.set_rotation(rot)
            w = conn.effective_width
            h = conn.effective_height
            conn.set_rotation(old)
            depth = h if edge in ("bottom", "top") else w
            max_depth = max(max_depth, depth)
    return max_depth


def _expand_interior_for_connectors(
    ib: tuple[float, float, float, float],
    connectors: List["Component"],
    board: "BoardOutline",
    margin: float,
    min_gap: float = 2.0,
    mating_margin: float = 5.0,
) -> tuple[float, float, float, float]:
    """Expand interior bbox so connectors have room on each edge.

    When SA condenses interior components, the resulting bbox can be too small
    for connectors to fit with proper mating clearance. This expands the bbox
    symmetrically within board bounds, using rotation-aware spacing.
    """
    if not connectors:
        return ib

    ib_x_min, ib_y_min, ib_x_max, ib_y_max = ib

    min_per_edge = _estimate_min_edge_space(connectors, mating_margin)
    extra_edge_margin = 2.0  # Additional buffer beyond estimated connector space to avoid edge overlaps (later need to added to config)
    min_per_edge += extra_edge_margin

    width = ib_x_max - ib_x_min
    height = ib_y_max - ib_y_min

    if width < min_per_edge:
        expand = (min_per_edge - width) / 2
        ib_x_min = max(board.x_min + margin, ib_x_min - expand)
        ib_x_max = min(board.x_max - margin, ib_x_max + expand)

    if height < min_per_edge:
        expand = (min_per_edge - height) / 2
        ib_y_min = max(board.y_min + margin, ib_y_min - expand)
        ib_y_max = min(board.y_max - margin, ib_y_max + expand)

    return (ib_x_min, ib_y_min, ib_x_max, ib_y_max)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def smart_grid_place(
    model: "BoardModel",
    margin: float = 5.0,
    spacing_factor: float = 1.3,
    sa_iterations: int = 2000,
    min_connector_gap: float = 2.0,
    rules: list | None = None,
) -> "BoardModel":
    """Smart PCB placement: interior first, connectors around interior bbox perimeter.

    Args:
        model: BoardModel with components to place
        margin: Margin from board edges in mm
        spacing_factor: Spacing multiplier for interior components
        sa_iterations: SA iterations for interior optimization
        min_connector_gap: Minimum gap between adjacent connectors
        rules: Optional list of ConstraintRule objects.  When provided (typical
            case: from board profile), the interior SA optimizer includes
            constraint penalties in its cost function, so decoupling caps
            stay near their ICs, connectors respect edge rules, etc.

    Returns:
        BoardModel with all components placed
    """
    # Phase 1: Classify — vertical connectors go with interior
    edge_connectors = [
        c for c in model.components
        if getattr(c, 'component_type', '') == "connector"
        and not c.is_fixed
        and not _is_vertical_connector(c)
    ]
    interior = [
        c for c in model.components
        if not c.is_fixed and (
            getattr(c, 'component_type', '') != "connector"
            or _is_vertical_connector(c)
        )
    ]

    # Compute effective margin: reserve perimeter for connectors on user-defined boards
    effective_margin = margin
    if getattr(model, 'user_defined_outline', False) and edge_connectors:
        conn_depth = _max_connector_depth(edge_connectors)
        effective_margin = margin + conn_depth + 1.0
        min_dim = min(model.board.width, model.board.height)
        max_allowed = min_dim * 0.35
        if effective_margin > max_allowed:
            print(f"  Note: capping connector reserve to {max_allowed:.1f}mm "
                  f"(board {min_dim:.1f}mm too tight for {effective_margin:.1f}mm)")
            effective_margin = max(margin, max_allowed)
        print(f"  Connector-aware margin: {margin:.1f} -> {effective_margin:.1f}mm "
              f"(depth reserve {conn_depth:.1f}mm)")

    # Phase 2: Net-cluster-based interior placement
    if interior:
        _place_interior(model, interior, effective_margin, spacing_factor)

    # Phase 2.5: Quadratic analytical placement
    if interior:
        quadratic_place(model, margin=effective_margin, n_iterations=3, verbose=False)

    # Phase 2.7: Rough-place connectors so SA sees interior↔connector nets
    if edge_connectors and interior:
        _rough_place_connectors(model, edge_connectors, interior, margin)

    # Phase 3: SA optimization — interior only moves, but HPWL includes
    # fixed connector positions so interior components are pulled toward
    # their connected connectors (prevents center collapse).
    if interior:
        board = model.board
        board_area = board.width * board.height
        if board_area > 0:
            comp_area = sum(c.effective_width * c.effective_height for c in interior)
            density = min(1.0, comp_area / board_area)
        else:
            density = 0.0
        if density > 0.30:
            sa_n_iter = sa_iterations // 4
        elif density < 0.20:
            sa_n_iter = sa_iterations // 8
        else:
            t = (density - 0.20) / 0.10
            sa_n_iter = int(sa_iterations // 8 + t * (sa_iterations // 4 - sa_iterations // 8))
            sa_n_iter = max(sa_n_iter, sa_iterations // 8)
        conn_refs = {c.ref for c in edge_connectors} if edge_connectors else None
        _optimize_interior_sa(model, interior, effective_margin, n_iter=sa_n_iter,
                              rules=rules, fixed_refs=conn_refs)

    # Phase 3.5: recentre interior cluster in usable board area
    if interior:
        _recentre_interior_cluster(model, interior, effective_margin)

    # Phase 4: Interior bbox computed AFTER optimization
    ib = _compute_interior_bbox(interior, model.board, margin)

    # Phase 4.5: Expand bbox so connectors have room on edges
    if edge_connectors:
        ib = _expand_interior_for_connectors(
            ib, edge_connectors, model.board, margin, min_connector_gap)

    # Phase 5: Place edge connectors on exterior of interior bbox
    if edge_connectors:
        _place_connectors_perimeter(
            model, edge_connectors, margin, ib,
            min_connector_gap=min_connector_gap,
        )

    # Phase 6: Overlap resolution — push inward, clamp to interior bbox
    _resolve_all_overlaps(model, interior, margin, ib)

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
    """Net-cluster based interior placement.

    Groups ICs with their decoupling caps/resistors via cluster_components(),
    assigns each cluster its own sub-region. Within each cluster, components
    are ordered so each IC is immediately followed by its associated passives.
    """
    board = model.board
    x_min = board.x_min + margin
    x_max = board.x_max - margin
    y_min = board.y_min + margin
    y_max = board.y_max - margin
    interior_refs = {c.ref for c in interior}
    ref_to_comp = {c.ref: c for c in interior}

    # Get cluster ref-lists and compute sub-regions
    clusters = None
    regions: Dict[int, tuple] = {}

    try:
        clusters = cluster_components(model)
        clusters = [
            [r for r in cl if r in interior_refs]
            for cl in clusters
        ]
        clusters = [cl for cl in clusters if cl]

        # Merge cap-only clusters into clusters with their power-domain ICs
        _merge_orphan_caps(clusters, model, interior_refs)

        n_cl = len(clusters)
        cols = max(1, int(math.ceil(math.sqrt(n_cl))))
        rows = max(1, int(math.ceil(n_cl / cols)))
        rw = (x_max - x_min) / cols
        rh = (y_max - y_min) / rows
        for idx in range(n_cl):
            c = idx % cols
            r = idx // cols
            regions[idx] = (x_min + c * rw, y_min + r * rh,
                           x_min + (c + 1) * rw, y_min + (r + 1) * rh)
    except Exception:
        clusters = None

    placed_refs: Set[str] = set()

    if clusters:
        for idx, cl_refs in enumerate(clusters):
            rx_min, ry_min, rx_max, ry_max = regions.get(
                idx, (x_min, y_min, x_max, y_max))
            comps = [ref_to_comp[r] for r in cl_refs if r in ref_to_comp]
            if not comps:
                continue

            grouped = _group_by_ic_affinity(model, comps)
            _place_cluster_grid(grouped, rx_min, ry_min, rx_max, ry_max)
            placed_refs.update(c.ref for c in comps)

    # Fallback: any component not covered by a cluster
    unplaced = [c for c in interior if c.ref not in placed_refs]
    if unplaced:
        grouped = _group_by_ic_affinity(model, unplaced)
        _place_cluster_grid(grouped, x_min, y_min, x_max, y_max)

    _apply_repulsion(interior, x_min, x_max, y_min, y_max, spacing_factor,
                     max_iterations=30)


def _merge_orphan_caps(
    clusters: List[List[str]],
    model: "BoardModel",
    interior_refs: Set[str],
) -> None:
    """Move caps in clusters without ICs into clusters with their power-domain ICs."""
    ic_types = {'ic', 'mcu', 'regulator'}

    # Identify which clusters have ICs
    cluster_has_ics: Dict[int, bool] = {}
    cluster_ic_refs: Dict[int, Set[str]] = {}
    for i, cl in enumerate(clusters):
        ic_refs = set()
        for r in cl:
            comp = model.get_component(r)
            if comp and getattr(comp, 'component_type', '') in ic_types:
                ic_refs.add(r)
        cluster_has_ics[i] = bool(ic_refs)
        cluster_ic_refs[i] = ic_refs

    # Build power domains across all interior components
    domains = _build_power_domains(model, interior_refs)

    # For each cap-only cluster, find the best IC cluster via power domains
    orphan_indices = [i for i, has_ics in cluster_has_ics.items() if not has_ics]

    for orphan_idx in orphan_indices:
        caps_to_move: Dict[str, int] = {}  # cap_ref -> target_cluster_idx
        remaining_refs = list(clusters[orphan_idx])

        for cap_ref in remaining_refs:
            comp = model.get_component(cap_ref)
            if not comp or getattr(comp, 'component_type', '') != 'capacitor':
                continue
            # Find which cluster has ICs on the same power rail as this cap
            best_cluster = None
            best_count = 0
            for net_name, (domain_ics, domain_caps) in domains.items():
                if cap_ref not in domain_caps:
                    continue
                # This cap belongs to this power domain — find cluster with most domain ICs
                for ci, has_ics in cluster_has_ics.items():
                    if not has_ics:
                        continue
                    overlap = len(cluster_ic_refs[ci] & set(domain_ics))
                    if overlap > best_count:
                        best_count = overlap
                        best_cluster = ci
                break  # cap found its domain

            if best_cluster is not None:
                caps_to_move[cap_ref] = best_cluster

        # Move caps to their target clusters
        for cap_ref, target_idx in caps_to_move.items():
            clusters[orphan_idx].remove(cap_ref)
            clusters[target_idx].append(cap_ref)

    # Remove now-empty clusters
    clusters[:] = [cl for cl in clusters if cl]


def _build_power_domains(
    model: "BoardModel",
    comp_refs: Set[str],
) -> Dict[str, Tuple[List[str], List[str]]]:
    """Identify power domains: non-GND power nets → (IC refs, cap refs).

    Returns {net_name: ([ic_refs], [cap_refs])} for domains that have
    both ICs and capacitors. GND is excluded since every component
    shares it — no discriminative power.
    """
    ic_types = {'ic', 'mcu', 'regulator'}
    ref_type: Dict[str, str] = {}
    for c in model.components:
        if c.ref in comp_refs:
            ref_type[c.ref] = getattr(c, 'component_type', '')

    # net_name → sets of refs
    net_ics: Dict[str, List[str]] = defaultdict(list)
    net_caps: Dict[str, List[str]] = defaultdict(list)

    for net in model.nets:
        name = getattr(net, 'name', '') or ''
        if not _is_power_net(name):
            continue
        # Exclude GND — everything shares it
        clean = name.lstrip('/').upper()
        if any(clean.startswith(p) for p in ('GND', 'AGND', 'DGND', 'PGND', 'SGND', 'VSS')):
            continue

        for ref, _ in net.pins:
            if ref not in comp_refs:
                continue
            ctype = ref_type.get(ref, '')
            if ctype in ic_types:
                net_ics[name].append(ref)
            elif ctype == 'capacitor':
                net_caps[name].append(ref)

    domains: Dict[str, Tuple[List[str], List[str]]] = {}
    for name in net_ics:
        if name in net_caps and net_ics[name] and net_caps[name]:
            domains[name] = (net_ics[name], net_caps[name])
    return domains


def _group_by_ic_affinity(
    model: "BoardModel",
    comps: List["Component"],
) -> List["Component"]:
    """Order components so each IC is immediately followed by its closest passives.

    Phase A: Power-domain round-robin assigns decoupling caps to ICs sharing
    the same rail (one cap per IC per rail, balanced).
    Phase B: Remaining passives use shared-net affinity scoring.

    Output: [IC1, passive1a, passive1b, IC2, passive2a, ...]
    """
    ic_types = {'ic', 'mcu', 'regulator'}
    ics = [c for c in comps if getattr(c, 'component_type', '') in ic_types]
    passives = [c for c in comps if getattr(c, 'component_type', '') not in ic_types]

    if not ics:
        return comps

    comp_refs = {c.ref for c in comps}
    ic_affinity: Dict[str, List["Component"]] = {ic.ref: [ic] for ic in ics}
    ref_to_comp = {c.ref: c for c in comps}
    assigned_passives: Set[str] = set()

    # Phase A: Power-domain round-robin
    domains = _build_power_domains(model, comp_refs)
    for _net_name, (domain_ics, domain_caps) in domains.items():
        # Sort ICs by fewest assigned caps first (balanced distribution)
        domain_ics_sorted = sorted(
            domain_ics,
            key=lambda r: sum(1 for p in ic_affinity.get(r, [])
                              if getattr(p, 'component_type', '') == 'capacitor'),
        )
        cap_idx = 0
        for cap_ref in domain_caps:
            if cap_ref not in comp_refs or cap_ref in assigned_passives:
                continue
            # Round-robin: assign to IC with fewest caps
            target_ic = domain_ics_sorted[cap_idx % len(domain_ics_sorted)]
            cap_comp = ref_to_comp.get(cap_ref)
            if cap_comp and target_ic in ic_affinity:
                ic_affinity[target_ic].append(cap_comp)
                assigned_passives.add(cap_ref)
            cap_idx += 1

    # Phase B: Shared-net affinity for remaining passives
    remaining = [p for p in passives if p.ref not in assigned_passives]

    ref_nets: Dict[str, Set[int]] = {}
    for ni, net in enumerate(model.nets):
        for ref, _ in net.pins:
            ref_nets.setdefault(ref, set()).add(ni)

    ic_refs = [ic.ref for ic in ics]
    unassigned: List["Component"] = []

    for p in remaining:
        p_nets = ref_nets.get(p.ref, set())
        p_total = max(len(p_nets), 1)
        best_ic = None
        best_score = -1.0
        for ic_ref in ic_refs:
            shared = len(p_nets & ref_nets.get(ic_ref, set()))
            score = shared / p_total
            if score > best_score or (score == best_score and best_ic is not None
                                       and len(ic_affinity[ic_ref]) < len(ic_affinity[best_ic])):
                best_score = score
                best_ic = ic_ref
        if best_ic and best_score > 0:
            ic_affinity[best_ic].append(p)
        else:
            unassigned.append(p)

    sorted_ics = sorted(ic_refs, key=lambda r: len(ref_nets.get(r, set())), reverse=True)

    result = []
    for ic_ref in sorted_ics:
        result.extend(ic_affinity[ic_ref])
    result.extend(unassigned)
    return result


def _place_cluster_grid(
    comps: List["Component"],
    rx_min: float, ry_min: float,
    rx_max: float, ry_max: float,
) -> None:
    """Grid layout for one cluster's components in a sub-region.

    Components are ordered by IC-affinity (IC followed by its passives).
    ICs are placed in a center-first grid; passives are placed in a tight
    ring around their parent IC within the IC's cell.
    """
    n = len(comps)
    if n == 0:
        return
    rw = rx_max - rx_min
    rh = ry_max - ry_min
    if rw <= 0 or rh <= 0:
        return
    if n == 1:
        c = comps[0]
        c.x = (rx_min + rx_max) / 2
        c.y = (ry_min + ry_max) / 2
        return

    ic_types = {'ic', 'mcu', 'regulator'}

    # Build IC groups: [(ic, [passive1, passive2, ...]), ...]
    ic_groups: List[tuple] = []
    current_ic = None
    current_passives: List["Component"] = []
    standalone: List["Component"] = []

    for comp in comps:
        if getattr(comp, 'component_type', '') in ic_types:
            if current_ic is not None:
                ic_groups.append((current_ic, current_passives))
            current_ic = comp
            current_passives = []
        else:
            if current_ic is not None:
                current_passives.append(comp)
            else:
                standalone.append(comp)

    if current_ic is not None:
        ic_groups.append((current_ic, current_passives))

    # Add standalone components (no IC in group) as single-slot entries
    for comp in standalone:
        ic_groups.append((comp, []))

    n_groups = len(ic_groups)
    aspect = rw / max(rh, 1e-9)
    cols   = max(1, min(n_groups, int(round(math.sqrt(n_groups * aspect)))))
    rows   = max(1, math.ceil(n_groups / cols))
    cell_w = rw / cols
    cell_h = rh / rows

    center_col = cols // 2
    center_row = rows // 2
    order = sorted(
        range(cols * rows),
        key=lambda slot: abs(slot % cols - center_col) + abs(slot // cols - center_row),
    )

    for group_idx, (ic_comp, passives) in enumerate(ic_groups):
        slot = order[group_idx] if group_idx < len(order) else group_idx
        col  = slot % cols
        row  = slot // cols

        # IC goes at cell center
        cx = rx_min + (col + 0.5) * cell_w
        cy = ry_min + (row + 0.5) * cell_h
        ic_comp.x = cx
        ic_comp.y = cy

        # Place passives in a tight ring around the IC
        if passives:
            ic_radius = max(ic_comp.effective_width, ic_comp.effective_height) / 2
            ring_r = ic_radius + 4.0  # 4mm gap from IC edge for manual tuning room
            for pi, p in enumerate(passives):
                angle = 2 * math.pi * pi / len(passives)
                px = cx + ring_r * math.cos(angle)
                py = cy + ring_r * math.sin(angle)
                w2 = p.effective_width / 2
                h2 = p.effective_height / 2
                # Clamp to cell
                p.x = max(rx_min + col * cell_w + w2,
                          min(px, rx_min + (col + 1) * cell_w - w2))
                p.y = max(ry_min + row * cell_h + h2,
                          min(py, ry_min + (row + 1) * cell_h - h2))


def _apply_repulsion(
    components: List["Component"],
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    spacing_factor: float,
    max_iterations: int = 150,
) -> None:
    """Push components apart to reduce overlaps."""
    for iteration in range(max_iterations):
        moved = False

        for i, ca in enumerate(components):
            for cb in components[i + 1:]:
                dx = cb.x - ca.x
                dy = cb.y - ca.y
                dist = math.sqrt(dx * dx + dy * dy)

                min_dist = max(
                    ca.effective_width, ca.effective_height,
                    cb.effective_width, cb.effective_height
                ) * spacing_factor * 0.8

                if dist < min_dist:
                    if dist < 0.1:
                        angle = (ca.x + ca.y) * 0.5
                        dx, dy = math.cos(angle), math.sin(angle)
                    else:
                        dx, dy = dx / dist, dy / dist

                    push = (min_dist - dist) * 0.6 + 0.5

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
# SA OPTIMIZATION
# =============================================================================

def _compute_hpwl(model: "BoardModel", interior_refs: Set[str] | None = None) -> float:
    """Half-perimeter wire length across all non-power nets.

    When interior_refs is provided, only counts positions of components in
    that set — excludes edge connectors whose positions aren't finalized yet.
    Power nets are skipped — their HPWL is nearly constant regardless of placement.
    """
    total = 0.0
    for net in model.nets:
        if _is_power_net(net.name):
            continue
        xs, ys = [], []
        for ref, _ in net.pins:
            if interior_refs is not None and ref not in interior_refs:
                continue
            c = model.get_component(ref)
            if c:
                xs.append(c.x)
                ys.append(c.y)
        if len(xs) >= 2:
            total += (max(xs) - min(xs)) + (max(ys) - min(ys))
    return total


def _compute_overlap_cost(components: List["Component"]) -> float:
    """Total overlap area between components — penalizes condensation."""
    total = 0.0
    for i, ca in enumerate(components):
        for cb in components[i + 1:]:
            if ca.overlaps(cb):
                total += ca.overlap_area(cb)
    return total


def _optimize_interior_sa(
    model: "BoardModel",
    interior: List["Component"],
    margin: float,
    T_start: float = 5.0,
    T_end: float = 0.1,
    n_iter: int = 500,
    overlap_weight: float = 10.0,
    rules: list | None = None,
    fixed_refs: set[str] | None = None,
) -> None:
    """Simulated annealing to minimize HPWL + overlap + constraint penalty for interior components.

    v9: optimized to use incremental HPWL and overlap computation instead of
    O(n²) full recomputation per iteration.  This makes the interior SA much
    faster, especially for boards with many components.

    When fixed_refs is provided (connector refs with pre-placed positions),
    those refs are included in HPWL computation so SA pulls interior
    components toward connected connectors.
    """
    import random
    board = model.board
    x_min, x_max = board.x_min + margin, board.x_max - margin
    y_min, y_max = board.y_min + margin, board.y_max - margin

    # HPWL refs: interior + fixed connectors (so SA sees interior↔connector nets)
    interior_only_refs = {c.ref for c in interior}
    hpwl_refs = set(interior_only_refs)
    if fixed_refs:
        hpwl_refs |= fixed_refs
    interior_set = set(id(c) for c in interior)

    # Weighted HPWL: interior-only + fraction of connector↔interior nets
    # Using full connector HPWL dominates SA and hurts interior optimization.
    # A 30% weight gives a gentle pull toward connectors without starving
    # interior↔interior optimization.
    _conn_weight = 0.3 if fixed_refs else 0.0

    def _fast_hpwl():
        if _conn_weight > 0:
            ih = _compute_hpwl(model, interior_only_refs)
            fh = _compute_hpwl(model, hpwl_refs)
            return ih + _conn_weight * (fh - ih)
        return _compute_hpwl(model, hpwl_refs)

    def _fast_overlap():
        total = 0.0
        for i, ca in enumerate(interior):
            for cb in interior[i + 1:]:
                if ca.overlaps(cb):
                    total += ca.overlap_area(cb)
        return total

    hpwl = _fast_hpwl()
    overlap = _fast_overlap()

    # Constraint penalty for interior SA
    constraint = 0.0
    constraint_weight = 4.0
    if rules:
        from engine.constraint_evaluator import evaluate_constraint_penalties
        constraint_raw, _ = evaluate_constraint_penalties(model, rules)
        constraint = constraint_raw

    cost = hpwl + overlap_weight * overlap + constraint_weight * constraint
    rudy_active = len(interior) >= 100
    rudy_weight = 0.2
    rudy_penalty = 0.0
    if rudy_active:
        try:
            rudy_penalty, _rpeak, _ravg, _roverflow = rudy_congestion_penalty(model)
        except Exception:
            rudy_active = False
            rudy_penalty = 0.0
    rudy_cost = rudy_weight * rudy_penalty if rudy_active else 0.0
    cost += rudy_cost
    T       = T_start
    cooling = (T_end / T_start) ** (1.0 / max(n_iter, 1))
    rng     = random.Random(42)

    # Track best solution
    best_cost = cost
    best_positions = {c.ref: (c.x, c.y) for c in interior}

    for it in range(n_iter):
        comp        = rng.choice(interior)
        old_x, old_y = comp.x, comp.y

        step    = T * 2.0
        comp.x  = max(x_min + comp.effective_width  / 2,
                      min(old_x + rng.uniform(-step, step),
                          x_max - comp.effective_width  / 2))
        comp.y  = max(y_min + comp.effective_height / 2,
                      min(old_y + rng.uniform(-step, step),
                          y_max - comp.effective_height / 2))

        # Incremental: only recompute what changed
        new_hpwl = _fast_hpwl()
        new_overlap = _fast_overlap()
        new_constraint = 0.0
        if rules:  # evaluate every iteration (was every 5th — stale constraints caused caps to drift)
            from engine.constraint_evaluator import evaluate_constraint_penalties
            new_constraint_raw, _ = evaluate_constraint_penalties(model, rules)
            new_constraint = new_constraint_raw
        new_cost = new_hpwl + overlap_weight * new_overlap + constraint_weight * new_constraint
        new_rudy_cost = rudy_cost
        if rudy_active and it % 50 == 0:
            try:
                new_rudy_penalty, _, _, _ = rudy_congestion_penalty(model)
                new_rudy_cost = rudy_weight * new_rudy_penalty
            except Exception:
                new_rudy_cost = rudy_cost
        new_cost += new_rudy_cost
        delta       = new_cost - cost

        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
            cost = new_cost
            rudy_cost = new_rudy_cost
            constraint = new_constraint
            if cost < best_cost:
                best_cost = cost
                best_positions[comp.ref] = (comp.x, comp.y)
        else:
            comp.x, comp.y = old_x, old_y

        T *= cooling

    # Restore best positions
    for c in interior:
        if c.ref in best_positions:
            c.x, c.y = best_positions[c.ref]


# =============================================================================
# ROUGH CONNECTOR PLACEMENT (pre-SA anchors)
# =============================================================================

def _rough_place_connectors(
    model: "BoardModel",
    connectors: List["Component"],
    interior: List["Component"],
    margin: float,
) -> None:
    """Rough-place connectors on perimeter based on net connectivity.

    Uses post-quadratic interior positions to determine each connector's
    best edge.  These positions serve as fixed anchors during interior SA
    so HPWL includes interior↔connector nets and pulls interior components
    toward their connected connectors.
    """
    if not connectors or not interior:
        return

    board = model.board
    ib = _compute_interior_bbox(interior, board, margin)
    ib_x_min, ib_y_min, ib_x_max, ib_y_max = ib

    edge_midpoints = {
        "bottom": ((ib_x_min + ib_x_max) / 2, ib_y_max),
        "right":  (ib_x_max, (ib_y_min + ib_y_max) / 2),
        "top":    ((ib_x_min + ib_x_max) / 2, ib_y_min),
        "left":   (ib_x_min, (ib_y_min + ib_y_max) / 2),
    }

    interior_refs = {c.ref for c in interior}
    conn_margin = 3.0

    for conn in connectors:
        connected_x: list[float] = []
        connected_y: list[float] = []

        for net in model.nets:
            if _is_power_net(net.name):
                continue
            has_conn = conn.ref in {r for r, _ in net.pins}
            if not has_conn:
                continue
            for ref, _ in net.pins:
                if ref in interior_refs:
                    comp = model.get_component(ref)
                    if comp:
                        connected_x.append(comp.x)
                        connected_y.append(comp.y)

        if connected_x:
            target_x = sum(connected_x) / len(connected_x)
            target_y = sum(connected_y) / len(connected_y)
        else:
            target_x = (ib_x_min + ib_x_max) / 2
            target_y = (ib_y_min + ib_y_max) / 2

        best_edge = min(
            edge_midpoints,
            key=lambda e: math.hypot(
                target_x - edge_midpoints[e][0],
                target_y - edge_midpoints[e][1]))

        half_w = conn.effective_width / 2
        half_h = conn.effective_height / 2

        if best_edge == "bottom":
            conn.x = max(ib_x_min + half_w, min(target_x, ib_x_max - half_w))
            conn.y = ib_y_max + conn_margin + half_h
        elif best_edge == "top":
            conn.x = max(ib_x_min + half_w, min(target_x, ib_x_max - half_w))
            conn.y = ib_y_min - conn_margin - half_h
        elif best_edge == "left":
            conn.x = ib_x_min - conn_margin - half_w
            conn.y = max(ib_y_min + half_h, min(target_y, ib_y_max - half_h))
        else:  # right
            conn.x = ib_x_max + conn_margin + half_w
            conn.y = max(ib_y_min + half_h, min(target_y, ib_y_max - half_h))

        # Clamp to board bounds
        conn.x = max(board.x_min + half_w, min(conn.x, board.x_max - half_w))
        conn.y = max(board.y_min + half_h, min(conn.y, board.y_max - half_h))


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

def _recentre_interior_cluster(
    model: "BoardModel",
    interior: list["Component"],
    margin: float,
) -> None:
    """Shift all interior components so their centroid is near the board center.

    This reduces cases where the SA-condensed interior cluster hugs a corner
    of the usable area, which in turn makes connector placement/corner
    handling more stable.
    """
    if not interior:
        return

    board = model.board

    # Usable area for interiors (same as in _place_interior)
    ux_min = board.x_min + margin
    ux_max = board.x_max - margin
    uy_min = board.y_min + margin
    uy_max = board.y_max - margin

    # Current centroid of interior components
    sx = sy = 0.0
    n = 0
    for c in interior:
        sx += c.x
        sy += c.y
        n += 1
    if n == 0:
        return

    cx = sx / n
    cy = sy / n

    # Target center = center of usable area
    tx = (ux_min + ux_max) / 2.0
    ty = (uy_min + uy_max) / 2.0

    dx = tx - cx
    dy = ty - cy

    if abs(dx) < 1e-3 and abs(dy) < 1e-3:
        return  # already centered enough

    # Translate all interior components by (dx, dy), clamped to usable area
    for c in interior:
        new_x = c.x + dx
        new_y = c.y + dy

        # Clamp component centers so their courtyards stay inside usable area
        half_w = c.effective_width / 2.0
        half_h = c.effective_height / 2.0

        c.x = max(ux_min + half_w, min(new_x, ux_max - half_w))
        c.y = max(uy_min + half_h, min(new_y, uy_max - half_h))

# =============================================================================
# CONNECTOR PERIMETER PLACEMENT
# =============================================================================

def _place_connectors_perimeter(
    model: "BoardModel",
    connectors: List["Component"],
    margin: float,
    interior_bbox: tuple[float, float, float, float],
    min_connector_gap: float = 2.0,
    mating_margin: float = 5.0,
) -> None:
    """Place connectors on perimeter of interior bbox, facing outward.

    Strategy:
    1. Group connectors by category (power pairs, input, output, data, etc.)
    2. Compute net-weighted center for each group
    3. Divide into 4 batches, assign each to nearest edge
    4. Place with per-connector rotation from pad analysis
    5. Space using post-rotation along-edge extent + mating margin
    """
    ib_x_min, ib_y_min, ib_x_max, ib_y_max = interior_bbox
    board = model.board
    gap = max(1.5, min_connector_gap)

    # Adaptive margin per edge: use less if space between interior bbox and
    # board edge is tight. Ensures connectors stay within board bounds.
    default_conn_margin = 5.0
    edge_space = {
        "bottom": board.y_max - ib_y_max,
        "right":  board.x_max - ib_x_max,
        "top":    ib_y_min - board.y_min,
        "left":   ib_x_min - board.x_min,
    }
    conn_margin = {}
    for edge_name, space in edge_space.items():
        conn_margin[edge_name] = min(default_conn_margin, max(1.0, space * 0.4))

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

    # Compute proportional target counts per edge based on edge lengths
    total_perimeter = sum(edge_lengths[e] for e in edges)
    n_connectors = len(connectors)
    targets = []
    for edge in edges:
        t = round(n_connectors * edge_lengths[edge] / total_perimeter)
        targets.append(t)

    # Round-off correction: adjust longest edges first
    diff = n_connectors - sum(targets)
    if diff != 0:
        edge_order = sorted(range(4), key=lambda i: edge_lengths[edges[i]],
                            reverse=True)
        for i in edge_order:
            if diff == 0:
                break
            if diff > 0:
                targets[i] += 1
                diff -= 1
            else:
                targets[i] -= 1
                diff += 1

    # Assign groups to edges proportionally, preferring net proximity
    sorted_groups = sorted(enumerate(groups),
                           key=lambda x: len(x[1].connectors), reverse=True)

    batches: List[List[int]] = [[] for _ in range(4)]
    batch_counts = [0] * 4

    for gi, group in sorted_groups:
        g_count = len(group.connectors)
        gcx, gcy = group_centers[gi]

        remaining = [targets[i] - batch_counts[i] for i in range(4)]
        available = [i for i in range(4) if remaining[i] > 0]

        if not available:
            best_batch = max(range(4), key=lambda b: remaining[b])
        else:
            best_batch = min(
                available,
                key=lambda i: math.hypot(
                    gcx - edge_midpoints[edges[i]][0],
                    gcy - edge_midpoints[edges[i]][1]))

        batches[best_batch].append(gi)
        batch_counts[best_batch] += g_count

    # Direct mapping: batch i → edge i (built proportionally by edge length)
    batch_edge: Dict[int, str] = {i: edges[i] for i in range(4) if batches[i]}

    # Place connectors on assigned edges
    comp_edge: Dict[str, str] = {}

    for bi, batch in enumerate(batches):
        if not batch or bi not in batch_edge:
            continue

        edge = batch_edge[bi]
        available = edge_lengths[edge]

        batch_groups = [groups[gi] for gi in batch]
        # Compute total space needed using rotation-aware per-connector spacing
        total_needed = 0.0
        for g in batch_groups:
            for comp in g.connectors:
                total_needed += _connector_along_edge_extent(comp, edge, mating_margin)
            total_needed += gap
        total_needed = max(0, total_needed - gap)  # no trailing gap

        offset = max(0.0, (available - total_needed) / 2.0)
        pos = offset

        for group in batch_groups:
            for comp in group.connectors:
                # Compute per-connector rotation from pad geometry
                rot = _compute_connector_rotation(comp, edge)
                comp.set_rotation(rot)

                w = comp.effective_width   # X-extent after rotation
                h = comp.effective_height  # Y-extent after rotation
                em = conn_margin[edge]

                # Along-edge extent = dimension parallel to the edge
                along = w if edge in ("bottom", "top") else h

                if edge == "bottom":
                    cx = ib_x_min + pos + along / 2
                    cy = ib_y_max + em + h / 2
                elif edge == "top":
                    cx = ib_x_min + pos + along / 2
                    cy = ib_y_min - em - h / 2
                elif edge == "left":
                    cx = ib_x_min - em - w / 2
                    cy = ib_y_min + pos + along / 2
                else:  # right
                    cx = ib_x_max + em + w / 2
                    cy = ib_y_min + pos + along / 2

                comp.x = cx
                comp.y = cy

                # Bbox-aware clamp: adjust position so actual bbox stays
                # within board (accounts for bbox_offset from footprint origin)
                if getattr(comp, "component_type", "") == "connector":
                    b = pad_bbox(comp)
                else:
                    b = comp.bbox

                if b[0] < board.x_min:
                    comp.x += board.x_min - b[0]
                elif b[2] > board.x_max:
                    comp.x -= b[2] - board.x_max
                # b = comp.bbox
                if b[1] < board.y_min:
                    comp.y += board.y_min - b[1]
                elif b[3] > board.y_max:
                    comp.y -= b[3] - board.y_max

                comp_edge[comp.ref] = edge

                # Advance by along-edge extent + mating margin
                # mating_margin ensures physical space for cable/mating
                pos += along + mating_margin

            pos += gap

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

def _resolve_all_overlaps(
    model: "BoardModel",
    interior: List["Component"],
    margin: float,
    ib: tuple[float, float, float, float],
    ib_margin: float = 1.0,
) -> None:
    """Resolve overlaps. Interior components are pushed toward interior bbox center."""
    connector_ids = {
        id(c) for c in model.components
        if getattr(c, 'component_type', '') == "connector"
        and not _is_vertical_connector(c)
    }

    ib_x_min, ib_y_min, ib_x_max, ib_y_max = ib
    clamp_x_min = ib_x_min + ib_margin
    clamp_x_max = ib_x_max - ib_margin
    clamp_y_min = ib_y_min + ib_margin
    clamp_y_max = ib_y_max - ib_margin
    ib_cx = (ib_x_min + ib_x_max) / 2
    ib_cy = (ib_y_min + ib_y_max) / 2

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
                dx   = c2.x - c1.x
                dy   = c2.y - c1.y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < 0.1:
                    dx, dy = 1.0, 0.0
                else:
                    dx, dy = dx / dist, dy / dist

                push = 0.5

                if c1_is_conn:
                    tx = ib_cx - c2.x
                    ty = ib_cy - c2.y
                    td = math.sqrt(tx * tx + ty * ty)
                    tx, ty = (tx / td, ty / td) if td > 0.1 else (-dx, -dy)
                    c2.x += push * 2 * tx
                    c2.y += push * 2 * ty
                    _clamp_to_interior(c2, clamp_x_min, clamp_x_max,
                                           clamp_y_min, clamp_y_max)

                elif c2_is_conn:
                    tx = ib_cx - c1.x
                    ty = ib_cy - c1.y
                    td = math.sqrt(tx * tx + ty * ty)
                    tx, ty = (tx / td, ty / td) if td > 0.1 else (dx, dy)
                    c1.x += push * 2 * tx
                    c1.y += push * 2 * ty
                    _clamp_to_interior(c1, clamp_x_min, clamp_x_max,
                                           clamp_y_min, clamp_y_max)

                else:
                    c1.x = max(clamp_x_min + c1.effective_width  / 2,
                               min(c1.x - push * dx,
                                   clamp_x_max - c1.effective_width  / 2))
                    c1.y = max(clamp_y_min + c1.effective_height / 2,
                               min(c1.y - push * dy,
                                   clamp_y_max - c1.effective_height / 2))
                    c2.x = max(clamp_x_min + c2.effective_width  / 2,
                               min(c2.x + push * dx,
                                   clamp_x_max - c2.effective_width  / 2))
                    c2.y = max(clamp_y_min + c2.effective_height / 2,
                               min(c2.y + push * dy,
                                   clamp_y_max - c2.effective_height / 2))

        if not overlap_found:
            break


def _clamp_to_interior(
    comp: "Component",
    x_min: float, x_max: float,
    y_min: float, y_max: float,
) -> None:
    """Clamp a component's center to stay within the interior bbox."""
    w2 = comp.effective_width  / 2
    h2 = comp.effective_height / 2
    comp.x = max(x_min + w2, min(comp.x, x_max - w2))
    comp.y = max(y_min + h2, min(comp.y, y_max - h2))


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ["smart_grid_place"]
