"""Phase 3.2 — Sub-circuit pattern recognition.

Recognizes common PCB sub-circuit motifs from component-type + shared-net
adjacency (not full graph isomorphism — kept cheap).  Each detected motif
is tagged as a rigid sub-group and feeds strong edges into the clustering
hypergraph so SA keeps motif members together.

Motifs recognized:
  - crystal_oscillator: crystal + 2 load caps (one on each oscillator net)
  - regulator: regulator IC + input cap + output cap (LDO/buck/boost)
  - (future: opamp_feedback, connector_esd_termination)

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
        osc_nets = list(crystal_nets)

        # Each oscillator net should have exactly one capacitor on it
        # (the load cap), which also connects to a power net (GND).
        load_caps = []
        valid = True
        for osc_net in osc_nets:
            refs_on_net = net_refs.get(osc_net, set())
            # Find capacitors on this oscillator net (excluding the crystal itself)
            caps_on_net = [
                r for r in refs_on_net
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
                'ic_ref': next(
                    (r for r in net_refs.get(osc_nets[0], set())
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
            reg_comp_obj = ref_comp.get(reg.ref)
            if reg_comp_obj is None:
                valid = False
                break
            closest_cap = min(
                caps_on_rail,
                key=lambda r: _center_distance(reg_comp_obj, ref_comp[r])
                if ref_comp.get(r) else float('inf'),
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
# Main entry point
# ---------------------------------------------------------------------------

def detect_subcircuit_patterns(model: BoardModel) -> list[SubcircuitPattern]:
    """Detect all recognizable sub-circuit motifs on the board.

    Returns a list of SubcircuitPattern objects.  Each component is
    used in at most one motif (greedy — first match wins).  Order of
    detection: crystal_oscillator first (most specific signature),
    then regulator (more generic, could overlap).

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
    patterns.extend(_detect_crystal_oscillators(model, ref_nets, net_refs, ref_comp, used_refs))
    patterns.extend(_detect_regulators(model, ref_nets, net_refs, ref_comp, used_refs))

    return patterns


def add_subcircuit_edges(G, model: BoardModel) -> None:
    """Add clique edges (weight SUBCIRCUIT_EDGE_WEIGHT) between every pair
    of components in each detected subcircuit pattern.

    Called by build_net_hypergraph after sheet edges are added.  These
    edges are STRONGER than sheet edges (2.0) and signal edges (1.0) —
    subcircuit members are rigid sub-groups that should never scatter
    across clusters.
    """
    patterns = detect_subcircuit_patterns(model)
    for pattern in patterns:
        refs = pattern.all_refs
        # Only add edges between refs that are nodes in the graph
        # (movable components).
        movable_refs = [r for r in refs if G.has_node(r)]
        if len(movable_refs) < 2:
            continue
        # Sorted for deterministic edge insertion order.
        movable_refs.sort()
        for i, r1 in enumerate(movable_refs):
            for r2 in movable_refs[i + 1:]:
                if G.has_edge(r1, r2):
                    G[r1][r2]["weight"] += SUBCIRCUIT_EDGE_WEIGHT
                else:
                    G.add_edge(r1, r2, weight=SUBCIRCUIT_EDGE_WEIGHT)
