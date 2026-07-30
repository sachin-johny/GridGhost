"""Phase 3.2 — Sub-circuit pattern recognition.

Recognizes common PCB sub-circuit motifs from component-type + shared-net
adjacency (not full graph isomorphism — kept cheap).  Each detected motif
is tagged as a rigid sub-group and feeds strong edges into the clustering
hypergraph so SA keeps motif members together.

Motifs recognized:
  - crystal_oscillator: crystal + 2 load caps (one on each oscillator net)
  - regulator: regulator IC + input cap + output cap (LDO/buck/boost)
  - signal_flow_chain: connector → passives → IC → IC → passives → connector
    (the "ADC in → op-amp → buffer → ADC out" pattern)

The "which side things go on" aspect (canonical relative arrangement) is
deferred to a follow-up — this commit focuses on detection + clustering
edges.  See AUDIT_FINAL_REPORT.md §3.2 for context.

Reference: brief item 3.2 — "certain component-type + netlist patterns
recur constantly and have a known canonical relative layout that any
PCB designer would recognize on sight".
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from models.board_model import BoardModel, Component, Net
from engine.cost_state import _is_power_net


# Weight for subcircuit-pattern clustering edges.  Stronger than sheet
# edges (2.0) and signal edges (1.0) — these are RIGID sub-groups where
# a human PCB designer would NEVER scatter the members.  Tunable.
SUBCIRCUIT_EDGE_WEIGHT = 3.0

# Weight for signal-flow chain edges.  Stronger than subcircuit (3.0)
# because chains span MORE components (J→R→U→U→R→J = 7 members) and
# need extra pull to keep the whole signal path together against the
# connector-perimeter placement which tries to split endpoints across
# different edges.
SIGNAL_FLOW_EDGE_WEIGHT = 4.0


@dataclass
class SubcircuitPattern:
    """A detected sub-circuit motif (e.g. crystal + 2 load caps).

    Attributes:
        motif_type: 'crystal_oscillator' | 'regulator' | ...
        anchor_ref: ref of the anchor component (the crystal / regulator IC)
        member_refs: refs of the other motif members (caps, inductor, etc.)
        all_refs: anchor + members, convenience for clustering edge creation
        metadata: motif-specific info (e.g. input/output net names for
                  regulator, oscillator net names for crystal)
    """
    motif_type: str
    anchor_ref: str
    member_refs: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def all_refs(self) -> list[str]:
        return [self.anchor_ref] + list(self.member_refs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_ref_to_nets(model: BoardModel) -> dict[str, set[str]]:
    """Map each component ref → set of net names it's on (excluding power nets)."""
    ref_nets: dict[str, set[str]] = defaultdict(set)
    for net in model.nets:
        if _is_power_net(net.name):
            continue
        for ref, _ in net.pins:
            ref_nets[ref].add(net.name)
    return ref_nets


def _build_net_to_refs(model: BoardModel) -> dict[str, set[str]]:
    """Map each net name → set of component refs on it (excluding power nets)."""
    net_refs: dict[str, set[str]] = defaultdict(set)
    for net in model.nets:
        if _is_power_net(net.name):
            continue
        for ref, _ in net.pins:
            net_refs[net.name].add(ref)
    return net_refs


def _build_ref_to_comp(model: BoardModel) -> dict[str, Component]:
    """Map each component ref → Component object."""
    return {c.ref: c for c in model.components}


# ---------------------------------------------------------------------------
# Motif detectors
# ---------------------------------------------------------------------------

def _detect_crystal_oscillators(
    model: BoardModel,
    ref_nets: dict[str, set[str]],
    net_refs: dict[str, set[str]],
    ref_comp: dict[str, Component],
    used_refs: set[str],
) -> list[SubcircuitPattern]:
    """Detect crystal + 2 load caps motifs.

    Pattern signature:
      - A crystal (component_type == 'crystal')
      - Connected to exactly 2 non-power signal nets (OSC_IN, OSC_OUT)
      - Each of those nets has exactly one capacitor on it (the load cap)
        that also connects to GND

    The crystal is the anchor; the 2 load caps are members.
    """
    patterns = []
    crystals = [
        c for c in model.components
        if getattr(c, 'component_type', '') == 'crystal'
        and c.ref not in used_refs
    ]

    for crystal in crystals:
        # Crystal should be on exactly 2 non-power signal nets.
        crystal_nets = ref_nets.get(crystal.ref, set())
        if len(crystal_nets) != 2:
            continue
        # ``crystal_nets`` is a ``set[str]`` — sort before iterating so
        # the metadata ``osc_nets`` list and the ``load_caps`` member order
        # are deterministic across PYTHONHASHSEED values.
        osc_nets = sorted(crystal_nets)

        # Each oscillator net should have exactly one capacitor on it
        # (the load cap), which also connects to a power net (GND).
        load_caps = []
        valid = True
        for osc_net in osc_nets:
            refs_on_net = net_refs.get(osc_net, set())
            # Find capacitors on this oscillator net (excluding the crystal itself).
            # ``refs_on_net`` is a ``set[str]`` — sort so that the "first"
            # cap picked via ``caps_on_net[0]`` below is deterministic when
            # (incorrectly) more than one cap is present, and so the member
            # order doesn't drift with PYTHONHASHSEED.
            caps_on_net = [
                r for r in sorted(refs_on_net)
                if r != crystal.ref
                and getattr(ref_comp.get(r), 'component_type', '') == 'capacitor'
            ]
            if len(caps_on_net) != 1:
                valid = False
                break
            load_caps.append(caps_on_net[0])

        if not valid:
            continue

        # All 3 components (crystal + 2 caps) must not already be used
        # in another motif.
        all_members = {crystal.ref, *load_caps}
        if all_members & used_refs:
            continue

        patterns.append(SubcircuitPattern(
            motif_type='crystal_oscillator',
            anchor_ref=crystal.ref,
            member_refs=load_caps,
            metadata={
                'osc_nets': osc_nets,
                # Pick the first IC on the first oscillator net as a metadata
                # hint.  ``net_refs.get(...)`` returns a ``set[str]`` — sort
                # it so the chosen ``ic_ref`` is deterministic across runs.
                'ic_ref': next(
                    (r for r in sorted(net_refs.get(osc_nets[0], set()))
                     if r != crystal.ref and r not in load_caps
                     and getattr(ref_comp.get(r), 'component_type', '') in {'ic', 'mcu', 'regulator'}),
                    None,
                ),
            },
        ))
        used_refs.update(all_members)

    return patterns


def _detect_regulators(
    model: BoardModel,
    ref_nets: dict[str, set[str]],
    net_refs: dict[str, set[str]],
    ref_comp: dict[str, Component],
    used_refs: set[str],
) -> list[SubcircuitPattern]:
    """Detect regulator + input cap + output cap motifs.

    Pattern signature:
      - A regulator (component_type == 'regulator')
      - Connected to at least 2 non-GND power nets (input rail, output rail)
      - Each of those power rails has exactly one capacitor on it
        (input bulk cap, output cap)

    The regulator is the anchor; the input cap and output cap are members.
    """
    patterns = []
    regulators = [
        c for c in model.components
        if getattr(c, 'component_type', '') == 'regulator'
        and c.ref not in used_refs
    ]

    # Build a map of power net → set of capacitor refs on it (excluding GND).
    # We need power nets here (regulator rails are power nets), so we look
    # at ALL nets, not just the ref_nets map (which excludes power nets).
    power_net_caps: dict[str, set[str]] = defaultdict(set)
    for net in model.nets:
        if not _is_power_net(net.name):
            continue
        clean = net.name.lstrip('/').upper()
        # Exclude GND — every component shares it.
        if any(clean.startswith(p) for p in ('GND', 'AGND', 'DGND', 'PGND', 'SGND', 'VSS')):
            continue
        for ref, _ in net.pins:
            if getattr(ref_comp.get(ref), 'component_type', '') == 'capacitor':
                power_net_caps[net.name].add(ref)

    for reg in regulators:
        # Find all non-GND power nets the regulator is on.
        reg_power_nets = []
        for net in model.nets:
            if not _is_power_net(net.name):
                continue
            clean = net.name.lstrip('/').upper()
            if any(clean.startswith(p) for p in ('GND', 'AGND', 'DGND', 'PGND', 'SGND', 'VSS')):
                continue
            if reg.ref in net.component_refs:
                reg_power_nets.append(net.name)

        # Need at least 2 power rails (input + output) to form a regulator motif.
        if len(reg_power_nets) < 2:
            continue

        # For each power rail, find the cap(s) on it.  Prefer exactly one
        # cap per rail (the canonical input/output cap pattern).  If a rail
        # has multiple caps, pick the closest one to the regulator (by
        # center distance) as the canonical member.
        member_caps = []
        valid = True
        for rail_net in reg_power_nets:
            caps_on_rail = power_net_caps.get(rail_net, set())
            if not caps_on_rail:
                # This rail has no cap — skip it, but don't fail the whole motif.
                continue
            # Pick the closest cap to the regulator (canonical member).
            # ``caps_on_rail`` is a ``set[str]``. ``min()`` returns the FIRST
            # element achieving the minimum key, so without a tiebreaker,
            # equidistant caps would be chosen by hash-seed-dependent set
            # iteration order — leaking nondeterminism into motif MEMBERSHIP.
            # The tuple key (distance, ref) makes ties resolve to the
            # lexicographically smaller ref, deterministically.
            reg_comp_obj = ref_comp.get(reg.ref)
            if reg_comp_obj is None:
                valid = False
                break
            closest_cap = min(
                caps_on_rail,
                key=lambda r: (
                    _center_distance(reg_comp_obj, ref_comp[r])
                    if ref_comp.get(r) else float('inf'),
                    r,  # deterministic tiebreaker on equidistant caps
                ),
            )
            member_caps.append(closest_cap)

        if not valid or len(member_caps) < 2:
            continue

        # Deduplicate — if the same cap is on multiple rails (unusual), only count once.
        member_caps = list(dict.fromkeys(member_caps))
        if len(member_caps) < 2:
            continue

        all_members = {reg.ref, *member_caps}
        if all_members & used_refs:
            continue

        patterns.append(SubcircuitPattern(
            motif_type='regulator',
            anchor_ref=reg.ref,
            member_refs=member_caps,
            metadata={
                'power_rails': reg_power_nets,
            },
        ))
        used_refs.update(all_members)

    return patterns


def _center_distance(a: Component, b: Component) -> float:
    """Euclidean distance between component centers."""
    import math
    return math.hypot(a.x - b.x, a.y - b.y)


# ---------------------------------------------------------------------------
# Signal-flow chain detector (Phase 3.3)
# ---------------------------------------------------------------------------

IC_TYPES = frozenset({'ic', 'mcu', 'regulator'})
PASSIVE_TYPES = frozenset({'resistor', 'capacitor', 'inductor', 'diode', 'generic'})
# Max BFS depth (hops through components).  A typical chain is
# J→R→U→R→U→R→J = 6 hops.  Allow 8 for longer paths.
_MAX_CHAIN_DEPTH = 8
# Min ICs in a chain.  1 accepts J→U→J (simple pass-through).
# 2 requires J→U→U→J (multi-stage, like ADC→op-amp→buffer→out).
_MIN_CHAIN_ICS = 1


def _detect_signal_flow_chains(
    model: BoardModel,
    ref_nets: dict[str, set[str]],
    net_refs: dict[str, set[str]],
    ref_comp: dict[str, Component],
    used_refs: set[str],
) -> list[SubcircuitPattern]:
    """Detect connector-to-connector signal-flow chains.

    A signal-flow chain is a path through signal nets from one connector
    to another, passing through at least one IC.  Example::

        J5 (ADC1_in) → R2 → U1 → R17 → U5 → R29 → J9 (Buffer1_out)

    This is the "ADC connector → IC → IC → output connector" pattern that
    human PCB designers place as a unit.  Detection uses BFS from each
    connector through non-power signal nets, recording the shortest path
    to each other connector.

    Chains can share ICs (e.g. a quad op-amp serves multiple channels) —
    only connectors are marked as used to prevent duplicate chains.

    Determinism: every iteration over a ``set`` returned by ``ref_nets`` /
    ``net_refs`` MUST be ``sorted()``-wrapped.  These sets hold refs and net
    names whose iteration order is governed by ``PYTHONHASHSEED``; leaving
    them unsorted leaks hash-seed noise into BFS traversal order, which in
    turn flips two downstream first-wins tie-breaks ("first path recorded
    for each endpoint" and "first endpoint with max ic_count").  See the
    docstring of the legacy detector's BFS loop below for the exact lines.
    """
    patterns: list[SubcircuitPattern] = []
    connectors = [
        c for c in model.components
        if getattr(c, 'component_type', '') == 'connector'
        and c.ref not in used_refs
        and not c.is_fixed
    ]

    for start_conn in connectors:
        # Skip if this connector was already used as a chain endpoint.
        if start_conn.ref in used_refs:
            continue

        # BFS from start_conn through signal nets.
        # State: (current_ref, path_list, depth)
        # Track visited per-BFS to avoid cycles.
        visited = {start_conn.ref}
        queue: list[tuple[str, list[str], int]] = [(start_conn.ref, [start_conn.ref], 0)]
        # Record shortest path to each connector found.
        found_endpoints: dict[str, list[str]] = {}

        while queue:
            curr_ref, path, depth = queue.pop(0)
            if depth >= _MAX_CHAIN_DEPTH:
                continue

            # Get signal nets for current component (non-power).
            # ``curr_nets`` and ``refs_on_net`` are ``set[str]`` — their
            # iteration order depends on PYTHONHASHSEED.  We MUST sort them
            # so the BFS tree grows in a fixed order; otherwise the
            # first-wins ``next_ref not in found_endpoints`` rule below
            # records different paths for the same endpoint across runs,
            # and the strict-``>`` tie-break on ``ic_count`` then picks a
            # different "best" chain.  This was the root cause of the
            # hash-seed nondeterminism in signal-flow chain detection.
            curr_nets = ref_nets.get(curr_ref, set())
            for net_name in sorted(curr_nets):
                refs_on_net = net_refs.get(net_name, set())
                for next_ref in sorted(refs_on_net):
                    if next_ref in visited:
                        continue
                    next_comp = ref_comp.get(next_ref)
                    if next_comp is None:
                        continue

                    next_type = getattr(next_comp, 'component_type', '')

                    # Found another connector — record the path.
                    if next_type == 'connector' and next_ref != start_conn.ref:
                        if next_ref not in found_endpoints:
                            found_endpoints[next_ref] = path + [next_ref]
                        continue  # don't BFS past a connector

                    # Continue through ICs and passives.
                    if next_type in IC_TYPES or next_type in PASSIVE_TYPES:
                        visited.add(next_ref)
                        queue.append((next_ref, path + [next_ref], depth + 1))

        # Filter: keep chains with ≥ _MIN_CHAIN_ICS ICs, prefer more ICs.
        # Iterate ``found_endpoints`` in sorted-endpoint order so that the
        # strict-``>`` tie-break below is deterministic: when two endpoints
        # tie on ``ic_count``, the lexicographically smaller ``end_ref`` wins.
        # (Without this, the first-inserted endpoint wins, but insertion
        # order depends on BFS traversal order, which previously leaked
        # PYTHONHASHSEED noise via the unsorted set iterations above.)
        best_chain: tuple[str, list[str], int] | None = None  # (end_ref, path, ic_count)
        for end_ref, chain_path in sorted(found_endpoints.items()):
            if end_ref in used_refs:
                continue
            ic_count = sum(
                1 for r in chain_path
                if getattr(ref_comp.get(r), 'component_type', '') in IC_TYPES
            )
            if ic_count < _MIN_CHAIN_ICS:
                continue
            # Prefer chains with more ICs (main signal flow > tap connections).
            if best_chain is None or ic_count > best_chain[2]:
                best_chain = (end_ref, chain_path, ic_count)

        if best_chain is not None:
            end_ref, chain_path, ic_count = best_chain
            patterns.append(SubcircuitPattern(
                motif_type='signal_flow_chain',
                anchor_ref=start_conn.ref,
                member_refs=chain_path[1:],
                metadata={
                    'chain': chain_path,
                    'end_connector': end_ref,
                    'ic_count': ic_count,
                    'hop_count': len(chain_path) - 1,
                },
            ))
            # Mark both endpoints so we don't get duplicate reverse chains.
            used_refs.add(start_conn.ref)
            used_refs.add(end_ref)

    return patterns


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def detect_subcircuit_patterns(model: BoardModel) -> list[SubcircuitPattern]:
    """Detect all recognizable sub-circuit motifs on the board.

    Returns a list of SubcircuitPattern objects.  Each component is
    used in at most one motif (greedy — first match wins).  Order of
    detection: crystal_oscillator first (most specific signature),
    then regulator (more generic, could overlap), then signal_flow_chain
    (connector-to-connector paths through ICs).

    Cheap by design: O(nets × components) per motif type, no graph
    isomorphism.  Designed to run once per placement pipeline invocation,
    not per SA move.
    """
    ref_nets = _build_ref_to_nets(model)
    net_refs = _build_net_to_refs(model)
    ref_comp = _build_ref_to_comp(model)
    used_refs: set[str] = set()

    patterns: list[SubcircuitPattern] = []

    # Detect in order of specificity — crystal_oscillator has the most
    # constraining signature (exactly 2 signal nets, exactly 1 cap per net),
    # so it's least likely to false-positive.  Regulator is more generic.
    # Signal_flow_chain runs last and only marks connectors as used.
    patterns.extend(_detect_crystal_oscillators(model, ref_nets, net_refs, ref_comp, used_refs))
    patterns.extend(_detect_regulators(model, ref_nets, net_refs, ref_comp, used_refs))
    patterns.extend(_detect_signal_flow_chains(model, ref_nets, net_refs, ref_comp, used_refs))

    return patterns


def add_subcircuit_edges(G, model: BoardModel) -> None:
    """Add clique edges between every pair of components in each detected
    subcircuit pattern.

    Called by build_net_hypergraph after sheet edges are added.  These
    edges are STRONGER than sheet edges (2.0) and signal edges (1.0) —
    subcircuit members are rigid sub-groups that should never scatter
    across clusters.

    Signal-flow chains use SIGNAL_FLOW_EDGE_WEIGHT (4.0) — stronger than
    subcircuit motifs (3.0) because chains span more components and need
    extra pull to keep the whole signal path together.
    """
    patterns = detect_subcircuit_patterns(model)
    for pattern in patterns:
        refs = pattern.all_refs
        # Only add edges between refs that are nodes in the graph
        # (movable components).
        movable_refs = [r for r in refs if G.has_node(r)]
        if len(movable_refs) < 2:
            continue
        # Weight depends on motif type.
        weight = SIGNAL_FLOW_EDGE_WEIGHT if pattern.motif_type == 'signal_flow_chain' else SUBCIRCUIT_EDGE_WEIGHT
        # Sorted for deterministic edge insertion order.
        movable_refs.sort()
        for i, r1 in enumerate(movable_refs):
            for r2 in movable_refs[i + 1:]:
                if G.has_edge(r1, r2):
                    G[r1][r2]["weight"] += weight
                else:
                    G.add_edge(r1, r2, weight=weight)
