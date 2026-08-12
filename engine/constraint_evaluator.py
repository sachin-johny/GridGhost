"""Constraint rule evaluator for PCB placement.

Implements the constraint penalty computation.

Each rule type produces a non-negative penalty proportional to how badly
the current placement violates that constraint.  The total constraint
penalty is:

    constraint = Σ (rule.weight x rule_penalty)

Rule penalties are *continuous* (not binary) so the SA gradient can
actually optimize them — a cap 5 mm beyond its threshold penalises more
than one 1 mm beyond.

"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Optional

from models.board_model import BoardModel, Component
from profiles.board_profiles import ConstraintRule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_POWER_PREFIXES_TUPLE = (
    'GND', 'AGND', 'DGND', 'PGND', 'SGND',
    'VSS', 'VCC', 'VDD', 'VEE', 'VBAT', 'VBUS',
)
_GND_PREFIXES_TUPLE = ('GND', 'AGND', 'DGND', 'PGND', 'SGND', 'VSS')
# Compile the voltage regex once at module load (was re.match per call).
_VOLTAGE_RE = re.compile(r'^[+\-]\d[\d.]*V', re.IGNORECASE)


def _is_power_net(name: str) -> bool:
    """Check if a net name is a power/ground net."""
    n = name.lstrip('/').upper()
    if n.startswith(_POWER_PREFIXES_TUPLE):
        return True
    return bool(_VOLTAGE_RE.match(n))


def _center_distance(a: Component, b: Component) -> float:
    """Euclidean distance between component centers."""
    return math.hypot(a.x - b.x, a.y - b.y)


def _distance_to_nearest_edge(comp: Component, board) -> float:
    """Minimum distance from component center to any board edge."""
    dx = min(comp.x - board.x_min, board.x_max - comp.x)
    dy = min(comp.y - board.y_min, board.y_max - comp.y)
    return max(0.0, min(dx, dy))


# ---------------------------------------------------------------------------
# Decoupling proximity rule
# ---------------------------------------------------------------------------

def _build_decoupling_map(
    model: BoardModel,
) -> dict[str, list[str]]:
    """Map each IC ref → list of decoupling cap refs that share a power net.

    Distributes caps evenly across ICs sharing a power rail.

    When multiple ICs share the same power net (e.g., four op-amps on
    +3V3), the old code assigned ALL caps on that net to a single IC
    (whichever came first alphabetically).  This left 3 out of 4 ICs
    with zero decoupling caps and zero proximity penalty, so the SA
    had no incentive to keep caps near those ICs.

    The new algorithm:
    1. For each cap, collect the set of ICs sharing its power net(s).
    2. Distribute caps round-robin among those ICs, prioritising ICs
       that currently have the fewest caps assigned.  This ensures
       every IC gets at least one nearby decoupling cap.
    """
    ic_types = {'ic', 'mcu', 'regulator'}
    ics = [c for c in model.components if getattr(c, 'component_type', '') in ic_types]
    caps = [c for c in model.components if getattr(c, 'component_type', '') == 'capacitor']

    if not ics or not caps:
        return {}

    ref_to_type = {c.ref: getattr(c, 'component_type', '') for c in model.components}

    # Build net → {refs} for non-GND power nets
    power_nets: dict[str, set[str]] = {}
    for net in model.nets:
        if not _is_power_net(net.name):
            continue
        clean = net.name.lstrip('/').upper()
        # Exclude GND — everything shares it, no discriminative power
        if clean.startswith(_GND_PREFIXES_TUPLE):
            continue
        refs = net.component_refs
        power_nets[net.name] = refs

    # For each cap, find which ICs share power nets
    ic_refs = {ic.ref for ic in ics}
    # cap → set of IC refs that share at least one power net
    cap_to_ics: dict[str, set[str]] = defaultdict(set)

    # Iterate power_nets and net_caps in sorted order so the
    # cap_to_ics insertion order (and downstream min() tie-breaking in
    # the round-robin) is deterministic across runs. Without this, set
    # iteration order varies with id() under ASLR, leading to different
    # cap→IC assignments and different SA trajectories.
    for net_name in sorted(power_nets.keys()):
        refs = power_nets[net_name]
        net_ics = refs & ic_refs
        # Check if non-IC refs are actually capacitors via ref→type lookup
        net_caps = {r for r in refs if r not in ic_refs and ref_to_type.get(r) == 'capacitor'}
        for cap_ref in sorted(net_caps):
            cap_to_ics[cap_ref].update(net_ics)

    # Distribute caps round-robin: each cap goes to the IC with the
    # fewest caps currently assigned (among ICs that share its power net).
    # This ensures every IC gets decoupling caps, not just one.
    ic_caps: dict[str, list[str]] = defaultdict(list)
    for cap_ref in sorted(cap_to_ics.keys()):
        candidate_ics = cap_to_ics[cap_ref]
        if not candidate_ics:
            continue
        # Pick the IC with fewest caps assigned so far. Sort to break
        # ties deterministically (min() returns the first minimum).
        best_ic = min(sorted(candidate_ics), key=lambda r: len(ic_caps[r]))
        ic_caps[best_ic].append(cap_ref)

    return dict(ic_caps)


def penalty_decoupling_proximity(
    model: BoardModel,
    rule: ConstraintRule,
    decap_map: dict[str, list[str]] | None = None,
    decap_pairs: list[tuple] | None = None,
) -> float:
    """Penalty for decoupling caps too far from their IC.

    For each (IC, cap) pair sharing a power net, penalty is:
        max(0, distance - max_distance_mm)

    where max_distance_mm defaults to 5.0 mm.  The penalty is quadratic
    beyond the threshold so SA sees increasing urgency:

        penalty = Σ  (excess)^2

    Returns the *raw* penalty (before rule.weight multiplication).

    Performance: pass ``decap_pairs`` (a list of ``(ic_component,
    cap_component)`` tuples) to skip the per-call ``model.get_component``
    lookups.  This is the SA hot path — materialising the pairs once at
    SA start eliminates ~2.5M dict lookups on a 68-component board.
    """
    max_dist = rule.params.get('max_distance_mm', 5.0)

    # Fast path: pre-materialised Component pairs (SA hot path)
    if decap_pairs is not None:
        total = 0.0
        for ic, cap in decap_pairs:
            dist = math.hypot(ic.x - cap.x, ic.y - cap.y)
            excess = dist - max_dist
            if excess > 0.0:
                total += excess * excess
        return total

    # Cold path: build from decap_map (string refs → Component lookups)
    if decap_map is None:
        decap_map = _build_decoupling_map(model)

    total = 0.0
    for ic_ref, cap_refs in decap_map.items():
        ic = model.get_component(ic_ref)
        if not ic:
            continue
        for cap_ref in cap_refs:
            cap = model.get_component(cap_ref)
            if not cap:
                continue
            dist = _center_distance(ic, cap)
            excess = max(0.0, dist - max_dist)
            total += excess * excess

    return total


# ---------------------------------------------------------------------------
# Crystal-MCU proximity rule
# ---------------------------------------------------------------------------

def _find_crystal_mcu_pairs(model: BoardModel) -> list[tuple[str, str]]:
    """Pair each crystal with the MCU/IC it shares the most nets with."""
    crystals = [c for c in model.components if getattr(c, 'component_type', '') == 'crystal']
    if not crystals:
        return []

    ic_types = {'ic', 'mcu', 'regulator'}
    ics = [c for c in model.components if getattr(c, 'component_type', '') in ic_types]

    # Build ref → set of net indices
    ref_nets: dict[str, set[int]] = defaultdict(set)
    for ni, net in enumerate(model.nets):
        for ref, _ in net.pins:
            ref_nets[ref].add(ni)

    pairs = []
    for crystal in crystals:
        best_ic = None
        best_shared = 0
        for ic in ics:
            shared = len(ref_nets.get(crystal.ref, set()) & ref_nets.get(ic.ref, set()))
            if shared > best_shared:
                best_shared = shared
                best_ic = ic.ref
        if best_ic:
            pairs.append((crystal.ref, best_ic))

    return pairs


def penalty_crystal_mcu(
    model: BoardModel,
    rule: ConstraintRule,
    crystal_pairs: list[tuple[str, str]] | None = None,
) -> float:
    """Penalty for crystals too far from their associated MCU/IC."""
    max_dist = rule.params.get('max_distance_mm', 10.0)
    if crystal_pairs is None:
        crystal_pairs = _find_crystal_mcu_pairs(model)

    total = 0.0
    for crystal_ref, ic_ref in crystal_pairs:
        crystal = model.get_component(crystal_ref)
        ic = model.get_component(ic_ref)
        if not crystal or not ic:
            continue
        dist = _center_distance(crystal, ic)
        excess = max(0.0, dist - max_dist)
        total += excess

    return total


# ---------------------------------------------------------------------------
# Connector-edge proximity rule
# ---------------------------------------------------------------------------

def penalty_connector_edge(
    model: BoardModel,
    rule: ConstraintRule,
    connectors: list[Component] | None = None,
) -> float:
    """Penalty for connectors far from board edges.

    Edge distance is measured from component center to nearest board edge.
    A connector that is `d` mm from the nearest edge with threshold
    `max_edge_distance_mm` (default 15.0 mm) is penalised:

        excess = max(0, d - max_edge_distance_mm)
        penalty = Σ excess^2

    Optional ``connectors`` lets callers pre-filter the connector list
    once and reuse it across SA moves. Connector identity (which components
    are connectors) is invariant during SA — only positions change.
    Pass the actual Component objects (not refs) to avoid O(N) lookups
    per call.
    """
    max_edge_dist = rule.params.get('max_edge_distance_mm', 15.0)
    board = model.board

    total = 0.0
    if connectors is None:
        for comp in model.components:
            if getattr(comp, 'component_type', '') != 'connector':
                continue
            if comp.is_fixed:
                continue
            d = _distance_to_nearest_edge(comp, board)
            excess = max(0.0, d - max_edge_dist)
            total += excess
    else:
        for comp in connectors:
            d = _distance_to_nearest_edge(comp, board)
            excess = max(0.0, d - max_edge_dist)
            total += excess

    return total


# ---------------------------------------------------------------------------
# Thermal grouping rule
# ---------------------------------------------------------------------------

def _thermal_types() -> set[str]:
    """Component types that generate or dissipate significant heat."""
    return {'regulator', 'mosfet', 'transistor', 'diode', 'led'}


def penalty_thermal_grouping(
    model: BoardModel,
    rule: ConstraintRule,
) -> float:
    """Penalty for thermally-related components being spread apart.

    Components of thermal types (regulators, MOSFETs, etc.) that share
    nets should be grouped together.  Penalty is the sum of squared
    distances between all thermal-component pairs on the same net,
    scaled down so it doesn't dominate.

    Only non-power nets are considered (power nets connect everything).
    """
    max_dist = rule.params.get('max_distance_mm', 20.0)
    thermal_set = _thermal_types()

    # Build net → thermal components
    net_thermals: dict[str, list[str]] = defaultdict(list)
    for net in model.nets:
        if _is_power_net(net.name):
            continue
        for ref, _ in net.pins:
            comp = model.get_component(ref)
            if comp and getattr(comp, 'component_type', '') in thermal_set:
                net_thermals[net.name].append(ref)

    total = 0.0
    seen_pairs: set[frozenset[str]] = set()
    for net_name, refs in net_thermals.items():
        for i, r1 in enumerate(refs):
            for r2 in refs[i + 1:]:
                pair = frozenset({r1, r2})
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                c1 = model.get_component(r1)
                c2 = model.get_component(r2)
                if not c1 or not c2:
                    continue
                dist = _center_distance(c1, c2)
                excess = max(0.0, dist - max_dist)
                total += excess

    return total


# ---------------------------------------------------------------------------
# Thermal separation rule (Phase 2.4 — DFM)
# ---------------------------------------------------------------------------

def penalty_thermal_separation(
    model: BoardModel,
    rule: ConstraintRule,
) -> float:
    """Penalty for hot parts on DIFFERENT nets being too close together.

    Complementary to ``thermal_grouping``: ``thermal_grouping`` keeps hot
    parts on the SAME net close (they're a functional block — e.g. a
    regulator + its pass transistor sharing a feedback net).  This rule
    keeps hot parts on DIFFERENT nets apart, so two independent heat
    sources don't stack and create a thermal hotspot.

    Real-world DFM context (from the brief): two regulators each
    dissipating 1W, placed 2mm apart, will heat each other far more than
    if they were 5-10mm apart — the local ambient temperature around
    each one rises, reducing efficiency and lifetime.  The ~5mm minimum
    separation is the same order of distance the existing
    ``thermal_grouping`` rule uses (20mm for shared-net grouping), just
    applied in the opposite direction.

    Pairs that share ANY non-power net are skipped — ``thermal_grouping``
    handles those (they SHOULD be close).  Power nets (GND, VCC, etc.)
    don't count as "sharing" because every component connects to them.

    Penalty: linear excess beyond ``min_distance_mm`` (default 5.0mm):
        penalty = Σ max(0, min_distance_mm - dist(c1, c2))

    Params:
        min_distance_mm: minimum allowed center-to-center distance
                         between two hot parts on different nets (default 5.0).
    """
    min_dist = rule.params.get('min_distance_mm', 5.0)
    thermal_set = _thermal_types()

    # Collect all thermal components.
    thermals = [
        c for c in model.components
        if getattr(c, 'component_type', '') in thermal_set and not c.is_fixed
    ]
    if len(thermals) < 2:
        return 0.0

    # Build ref → set of non-power net names for each thermal component.
    # Used to skip pairs that share a non-power net (thermal_grouping
    # handles those — they SHOULD be close).
    ref_to_signal_nets: dict[str, set[str]] = {}
    for c in thermals:
        nets = set()
        for net in model.nets:
            if _is_power_net(net.name):
                continue
            if c.ref in net.component_refs:
                nets.add(net.name)
        ref_to_signal_nets[c.ref] = nets

    total = 0.0
    for i in range(len(thermals)):
        c1 = thermals[i]
        n1 = ref_to_signal_nets.get(c1.ref, set())
        for j in range(i + 1, len(thermals)):
            c2 = thermals[j]
            n2 = ref_to_signal_nets.get(c2.ref, set())
            # Skip pairs that share any non-power signal net —
            # thermal_grouping handles those.
            if n1 & n2:
                continue
            dist = _center_distance(c1, c2)
            excess = max(0.0, min_dist - dist)
            total += excess

    return total


# ---------------------------------------------------------------------------
# High-current path rule
# ---------------------------------------------------------------------------

def _is_high_current_net(name: str) -> bool:
    """Heuristic: net names suggesting high current paths."""
    keywords = ('VOUT', 'VIN', 'VBAT', 'VBUS', 'PWR', 'POWER',
                'SUPPLY', 'MAIN', 'BATT', 'CHARGE', 'MOTOR', 'HEATER')
    n = name.lstrip('/').upper()
    return any(k in n for k in keywords)


def penalty_high_current_path(
    model: BoardModel,
    rule: ConstraintRule,
) -> float:
    """Penalty proportional to HPWL on high-current nets.

    High-current nets need short, thick traces.  This rule adds extra
    penalty for HPWL on nets whose names suggest high current, beyond
    what the standard HPWL term already charges.
    """
    scale = rule.params.get('hpwl_weight', 2.0)

    total = 0.0
    comp_map = {c.ref: c for c in model.components}
    for net in model.nets:
        if not _is_high_current_net(net.name):
            continue
        xs, ys = [], []
        for ref, _ in net.pins:
            comp = comp_map.get(ref)
            if comp:
                xs.append(comp.x)
                ys.append(comp.y)
        if len(xs) >= 2:
            hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys))
            total += hpwl * scale

    return total


# ---------------------------------------------------------------------------
# Bulk capacitor near input rule
# ---------------------------------------------------------------------------

def _is_bulk_cap(comp: Component) -> bool:
    """Heuristic: large-value or physically large capacitors are bulk caps."""
    if getattr(comp, 'component_type', '') != 'capacitor':
        return False
    if comp.effective_width >= 5.0 or comp.effective_height >= 5.0:
        return True
    val = getattr(comp, 'value', '') or ''
    val_u = val.upper()
    if any(u in val_u for u in ('UF', 'F')) and not any(u in val_u for u in ('PF', 'NF')):
        return True
    return False


def penalty_bulk_cap_input(
    model: BoardModel,
    rule: ConstraintRule,
) -> float:
    """Penalty for bulk capacitors far from power-input connectors."""
    max_dist = rule.params.get('max_distance_mm', 15.0)

    power_input_connectors: list[str] = []
    for comp in model.components:
        if getattr(comp, 'component_type', '') != 'connector':
            continue
        for net in model.nets:
            if comp.ref not in net.component_refs:
                continue
            if _is_power_net(net.name):
                power_input_connectors.append(comp.ref)
                break

    if not power_input_connectors:
        return 0.0

    bulk_caps = [c for c in model.components if _is_bulk_cap(c)]
    if not bulk_caps:
        return 0.0

    total = 0.0
    for cap in bulk_caps:
        min_dist = float('inf')
        for conn_ref in power_input_connectors:
            conn = model.get_component(conn_ref)
            if conn:
                d = _center_distance(cap, conn)
                min_dist = min(min_dist, d)
        excess = max(0.0, min_dist - max_dist)
        total += excess

    return total


# ---------------------------------------------------------------------------
# Antenna keepout rule
# ---------------------------------------------------------------------------

def penalty_antenna_keepout(
    model: BoardModel,
    rule: ConstraintRule,
) -> float:
    """Penalty for components within the antenna keepout zone."""
    kx_min = rule.params.get('keepout_x_min', 0.0)
    ky_min = rule.params.get('keepout_y_min', 0.0)
    kx_max = rule.params.get('keepout_x_max', 0.0)
    ky_max = rule.params.get('keepout_y_max', 0.0)

    if kx_max <= kx_min or ky_max <= ky_min:
        return 0.0

    total = 0.0
    for comp in model.components:
        if getattr(comp, 'component_type', '') == 'antenna':
            continue
        bx1, by1, bx2, by2 = comp.bbox
        ox1 = max(bx1, kx_min)
        oy1 = max(by1, ky_min)
        ox2 = min(bx2, kx_max)
        oy2 = min(by2, ky_max)
        if ox2 > ox1 and oy2 > oy1:
            total += (ox2 - ox1) * (oy2 - oy1)

    return total


# ---------------------------------------------------------------------------
# Analog/digital separation rule
# ---------------------------------------------------------------------------

def _domain_for_comp(comp: Component) -> Optional[str]:
    """Heuristic: classify component as 'analog' or 'digital' from net names."""
    nets = getattr(comp, 'nets', [])
    if not nets:
        return None
    analog_kw = ('ADC', 'DAC', 'ANA', 'OPA', 'AMP', 'SENSOR', 'AUDIO', 'IN-', 'OUT-')
    digital_kw = ('SPI', 'I2C', 'UART', 'USB', 'CAN', 'SDIO', 'GPIO', 'CLK', 'DATA', 'MOSI', 'MISO', 'SCK')

    a_score = sum(1 for n in nets if any(k in n.upper() for k in analog_kw))
    d_score = sum(1 for n in nets if any(k in n.upper() for k in digital_kw))

    if a_score > d_score:
        return 'analog'
    elif d_score > a_score:
        return 'digital'
    return None


def penalty_analog_digital_separation(
    model: BoardModel,
    rule: ConstraintRule,
) -> float:
    """Penalty for analog and digital components being too close.

    Uses a simple X-axis partition: components classified as analog
    should be on one side of the board, digital on the other.
    """
    axis = rule.params.get('separation_axis', 'x')
    separation_margin = rule.params.get('margin_mm', 5.0)

    analogs = []
    digitals = []
    for comp in model.components:
        if comp.is_fixed:
            continue
        domain = _domain_for_comp(comp)
        if domain == 'analog':
            analogs.append(comp)
        elif domain == 'digital':
            digitals.append(comp)

    if not analogs or not digitals:
        return 0.0

    a_cx = sum(c.x for c in analogs) / len(analogs)
    a_cy = sum(c.y for c in analogs) / len(analogs)
    d_cx = sum(c.x for c in digitals) / len(digitals)
    d_cy = sum(c.y for c in digitals) / len(digitals)

    if axis == 'x':
        if a_cx <= d_cx:
            analog_x, digital_x = a_cx, d_cx
        else:
            analog_x, digital_x = d_cx, a_cx
        midline = (analog_x + digital_x) / 2.0
        total = 0.0
        for comp in analogs:
            excess = max(0.0, comp.x - midline - separation_margin)
            total += excess
        for comp in digitals:
            excess = max(0.0, midline - separation_margin - comp.x)
            total += excess
    else:
        if a_cy <= d_cy:
            analog_y, digital_y = a_cy, d_cy
        else:
            analog_y, digital_y = d_cy, a_cy
        midline = (analog_y + digital_y) / 2.0
        total = 0.0
        for comp in analogs:
            excess = max(0.0, comp.y - midline - separation_margin)
            total += excess
        for comp in digitals:
            excess = max(0.0, midline - separation_margin - comp.y)
            total += excess

    return total


# ---------------------------------------------------------------------------
# Matched-length rule (placeholder — needs net-pair metadata)
# ---------------------------------------------------------------------------

def penalty_matched_length(
    model: BoardModel,
    rule: ConstraintRule,
) -> float:
    """Penalty for length mismatch between matched signal pairs.

    Requires params: 'pairs' — list of (net_name_1, net_name_2) tuples.
    Since net pair metadata isn't typically available from KiCad parsing,
    this is a placeholder that returns 0.0 unless pairs are specified.
    """
    pairs = rule.params.get('pairs', [])
    if not pairs:
        return 0.0

    comp_map = {c.ref: c for c in model.components}
    total = 0.0

    for n1_name, n2_name in pairs:
        net1 = model.get_net(n1_name)
        net2 = model.get_net(n2_name)
        if not net1 or not net2:
            continue

        def _net_hpwl(net):
            xs, ys = [], []
            for ref, _ in net.pins:
                comp = comp_map.get(ref)
                if comp:
                    xs.append(comp.x)
                    ys.append(comp.y)
            if len(xs) < 2:
                return 0.0
            return (max(xs) - min(xs)) + (max(ys) - min(ys))

        hpwl1 = _net_hpwl(net1)
        hpwl2 = _net_hpwl(net2)
        mismatch = abs(hpwl1 - hpwl2)
        total += mismatch * mismatch

    return total


# ---------------------------------------------------------------------------
# Ground-plane clearance rule (placeholder — needs zone metadata)
# ---------------------------------------------------------------------------

def penalty_ground_plane_clearance(
    model: BoardModel,
    rule: ConstraintRule,
) -> float:
    """Penalty for components overlapping ground plane split boundaries.

    Requires params: 'zones' — list of (x_min, y_min, x_max, y_max) tuples.
    Without zone data, returns 0.0.
    """
    zones = rule.params.get('zones', [])
    if not zones:
        return 0.0

    total = 0.0
    for comp in model.components:
        if comp.is_fixed:
            continue
        bx1, by1, bx2, by2 = comp.bbox
        for zx1, zy1, zx2, zy2 in zones:
            ox1 = max(bx1, zx1)
            oy1 = max(by1, zy1)
            ox2 = min(bx2, zx2)
            oy2 = min(by2, zy2)
            if ox2 > ox1 and oy2 > oy1:
                total += (ox2 - ox1) * (oy2 - oy1)

    return total


# ---------------------------------------------------------------------------
# Routing congestion rule (RUDY — Rectangular Uniform wire DensitY)
# ---------------------------------------------------------------------------

def penalty_routing_congestion(
    model: BoardModel,
    rule: ConstraintRule,
) -> float:
    """RUDY-based routing congestion penalty.

    Wraps ``engine.congestion.rudy_congestion_penalty`` so the existing
    RUDY implementation (already used as a SA translate-bias and
    best-state tiebreaker) becomes a profile-composable ConstraintRule.

    For each non-power net, RUDY distributes a unit of wire density
    uniformly across the grid cells covered by the net's pin bbox.
    The penalty is ``peak * (peak - target) + overflow`` where target
    is 1.5x the average congestion — penalises both localised hotspots
    and widespread overflow.  See ``engine/congestion.py`` for the full
    derivation and the Spindler & Johannes DATE 2007 reference.

    Brief context (Phase 2.1): HPWL is a distance proxy; it doesn't
    tell you WHERE wires will pile up.  Two placements with identical
    HPWL can have very different routability — one spreads nets
    uniformly, the other funnels 6 nets through a 2mm gap between two
    ICs.  RUDY catches the choke point that HPWL misses.

    Default off — opt in via a profile's ConstraintRule list.  The
    annealer's hardcoded RUDY bias (``SAConfig.rudy_weight``) stays as
    a default-on background spreading force; this rule makes the
    *acceptance cost* profile-controllable.

    Params:
        grid_resolution_mm: RUDY grid cell size in mm (default 2.0).
    """
    from engine.congestion import rudy_congestion_penalty
    grid_res = rule.params.get('grid_resolution_mm', 2.0)
    penalty, _peak, _avg, _overflow = rudy_congestion_penalty(model, grid_res)
    return penalty


def penalty_signal_flow_grouping(
    model: BoardModel,
    rule: ConstraintRule,
    chain_pairs: list[tuple] | None = None,
) -> float:
    """Penalty for signal-flow chain members being too far apart.

    For each detected signal-flow chain, penalty is::

        Σ  max(0, dist(adjacent_pair) − max_step_mm)²

    where ``max_step_mm`` defaults to 15 mm.  The penalty is quadratic
    beyond the threshold so SA sees increasing urgency to keep chain
    members close.

    Returns the *raw* penalty (before rule.weight multiplication).

    Performance: pass ``chain_pairs`` (a list of ``(ic_comp, cap_comp)
    or more generally ``(comp_a, comp_b)`` tuples for each adjacent pair
    in each chain) to skip the per-call pattern detection + component
    lookups.  Built once at SA start by ``CostState.__init__``.
    """
    max_step = rule.params.get('max_step_mm', 15.0)

    # Fast path: pre-materialised adjacent pairs
    if chain_pairs is not None:
        total = 0.0
        for ca, cb in chain_pairs:
            dist = math.hypot(ca.x - cb.x, ca.y - cb.y)
            excess = dist - max_step
            if excess > 0.0:
                total += excess * excess
        return total

    # Cold path: detect chains and build pairs
    from engine.subcircuit_patterns import detect_subcircuit_patterns
    total = 0.0
    for p in detect_subcircuit_patterns(model):
        if p.motif_type != 'signal_flow_chain':
            continue
        chain = p.metadata.get('chain', [])
        if len(chain) < 2:
            continue
        for i in range(len(chain) - 1):
            ca = model.get_component(chain[i])
            cb = model.get_component(chain[i + 1])
            if not ca or not cb:
                continue
            dist = math.hypot(ca.x - cb.x, ca.y - cb.y)
            excess = max(0.0, dist - max_step)
            total += excess * excess
    return total


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_RULE_HANDLERS = {
    'decoupling_proximity':       penalty_decoupling_proximity,
    'crystal_mcu':                penalty_crystal_mcu,
    'connector_edge':             penalty_connector_edge,
    'thermal_grouping':           penalty_thermal_grouping,
    'thermal_separation':         penalty_thermal_separation,
    'high_current_path':          penalty_high_current_path,
    'bulk_cap_input':             penalty_bulk_cap_input,
    'antenna_keepout':            penalty_antenna_keepout,
    'analog_digital_separation':  penalty_analog_digital_separation,
    'matched_length':             penalty_matched_length,
    'ground_plane_clearance':     penalty_ground_plane_clearance,
    'routing_congestion':         penalty_routing_congestion,
    'signal_flow_grouping':       penalty_signal_flow_grouping,
}


def evaluate_constraint_penalties(
    model: BoardModel,
    rules: list[ConstraintRule],
    decap_map: dict[str, list[str]] | None = None,
    crystal_pairs: list[tuple[str, str]] | None = None,
    connectors: list[Component] | None = None,
    decap_pairs: list[tuple] | None = None,
    chain_adjacent_pairs: list[tuple] | None = None,
) -> tuple[float, dict[str, float]]:
    """Evaluate all enabled constraint rules.

    Returns:
        (total_penalty, breakdown) where breakdown maps
        rule.name → raw penalty (before weight multiplication).

    Optional ``decap_map``, ``crystal_pairs``, and ``connectors`` let
    callers cache the topology (which ICs/caps share power nets, which
    crystals pair with which ICs, which components are connectors) since
    SA only moves components — it never changes the net connectivity.
    Skipping the rebuild is a 10-30× SA speedup.

    ``decap_pairs`` is a further optimisation: a pre-materialised list of
    ``(ic_component, cap_component)`` tuples that lets
    ``penalty_decoupling_proximity`` skip ``model.get_component`` lookups
    entirely.  Built once at SA start from ``decap_map``.

    ``chain_adjacent_pairs`` is the signal-flow-chain analog: a list of
    ``(comp_a, comp_b)`` tuples for each adjacent pair in each detected
    signal-flow chain, used by ``penalty_signal_flow_grouping``.
    """
    total = 0.0
    breakdown: dict[str, float] = {}

    for rule in rules:
        if not rule.enabled:
            breakdown[rule.name] = 0.0
            continue

        handler = _RULE_HANDLERS.get(rule.name)
        if handler is None:
            breakdown[rule.name] = 0.0
            continue

        if rule.name == 'decoupling_proximity':
            penalty = handler(model, rule, decap_map=decap_map, decap_pairs=decap_pairs)
        elif rule.name == 'signal_flow_grouping':
            penalty = handler(model, rule, chain_pairs=chain_adjacent_pairs)
        elif rule.name == 'crystal_mcu':
            penalty = handler(model, rule, crystal_pairs=crystal_pairs)
        elif rule.name == 'connector_edge':
            penalty = handler(model, rule, connectors=connectors)
        else:
            penalty = handler(model, rule)
        breakdown[rule.name] = penalty
        total += rule.weight * penalty

    return total, breakdown