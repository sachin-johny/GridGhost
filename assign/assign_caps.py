"""Classify caps and assign each decoupling cap to exactly one IC.

Four cap classes:

- **Decoupling (rigid)** — real decoupling caps on a power rail whose
  pins on the IC sit in one tight physical cluster (see
  ``RAIL_PAD_SPREAD_THRESHOLD_MM``). ALL caps the netlist actually put on
  that (IC, rail) pair become rigid macro followers — the cap count comes
  from the schematic, not a manual cap. These stay within
  ``MAX_CAP_IC_GAP_MM`` (edge-to-edge) of their leader: the "inner ring"
  of critical bypass caps that must be as close as physically possible
  to the IC power pins.

- **Decoupling (rail-adjacent)** — real decoupling caps on a rail whose
  pins are physically scattered across the IC footprint (e.g. a BGA
  power net landing on pads in opposite corners). There is no single
  point to glue a tight rigid ring to, so these become standalone
  macros; SA holds them near their assigned IC via the cap→IC attraction
  cost term instead of rigid geometry. This is a *geometric* decision,
  not a count-based one — a rail with 6 caps on one tightly clustered pin
  group stays fully rigid, while a rail with only 2 caps on pins spread
  across the die goes rail-adjacent.

- **Bulk** — large value (>10uF) OR on a power rail that no IC shares.
  These are rail-level filters (regulator output, board input rail, etc.)
  and are placed standalone; they do not belong to any one IC.

- **Coupling / signal** — only on signal nets (crystal loads, shields,
  USB D+/D-, reset filters). No power rail at all. Standalone.

Decoupling caps are distributed round-robin across the ICs sharing a
rail so each IC gets roughly equal decoupling, then grouped per
(IC, rail) pair — because one IC commonly has several *independent*
power domains (e.g. core, PLL, I/O), each with its own pin cluster and
its own rigid-vs-rail-adjacent verdict. A single IC can end up with,
say, 6 rigid followers on one rail and 2 rail-adjacent on another; there
is no per-IC ceiling.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.board_model import BoardModel, Component

IC_TYPES = frozenset({"ic", "mcu", "regulator"})

# Caps larger than this are "bulk", not decoupling followers.
# 10µF covers common 4.7/10µF IC bulk-decoupling caps; the previous 1µF
# threshold mis-classified them as decoupling followers (which get paired
# tightly to a single IC via a rigid macro).
DECAP_MAX_VALUE_F = 10e-6  # 10uF

# A (IC, rail) pin group is "tight" — and its real caps stay rigid — when
# the max pairwise distance between that IC's pads on that net is at or
# below this. Roughly: two caps flanking one BGA ball, or a handful of
# balls in one corner of the footprint, is tight; a rail landing on pads
# in opposite corners of the package is not (its caps can't all sit close
# to their own pin without overlapping each other or a neighboring rail's
# ring, so they're better held by soft attraction than rigid geometry).
# Calibrated against test4/U30: VCCPLL0/VCCPLL1 are single-pad (0mm
# spread, stay rigid); +1V2's 4 pads sit at opposite BGA corners (4.53mm
# spread, goes rail-adjacent).
RAIL_PAD_SPREAD_THRESHOLD_MM = 3.0

# Net-name patterns that mark power rails.
_POWER_PREFIXES = (
    "GND", "AGND", "DGND", "PGND", "SGND",
    "VSS", "VCC", "VDD", "VEE", "VBAT", "VBUS",
)
_GND_PREFIXES = ("GND", "AGND", "DGND", "PGND", "SGND", "VSS")
_POWER_VOLTAGE_RE = re.compile(r"^[+\-]\d[\d.]*V", re.IGNORECASE)

# Two common KiCad naming patterns hide real power rails from a plain
# prefix/regex check on the full net name:
#
# 1. Hierarchical/local labels, e.g. "/Buck/VIN" or "/Battery Protect/P+"
#    — the meaningful token is the LAST path segment, not the sheet name
#    prefixed onto it.
# 2. Auto-generated names for unlabeled nets, e.g. "Net-(U1-VIN)" — KiCad
#    falls back to "Net-(<ref>-<pin>)" when no net label/power-flag was
#    placed in the schematic. The meaningful token is the pin name.
#
# Both cases are common for a regulator's own VIN/VOUT pin or a local
# battery rail that the schematic author never gave a global label —
# electrically still a power rail, just not named like one. We extract
# the trailing token and check it against a small set of power-pin-name
# conventions. Deliberately conservative: control/feedback/switching
# pins on the same IC (EN, FB, COMP, SW, BOOT, VG, RT/CLK) are NOT power
# rails even though they're electrically adjacent, so they're excluded.
_HIER_LABEL_RE = re.compile(r"^/(?:.*/)?([^/]+)$")
_AUTO_NET_RE = re.compile(r"^Net-\([^-()]+-([^()]+)\)$")
_POWER_PIN_TOKENS = frozenset({
    "VIN", "VOUT", "PVIN", "PVOUT", "PVDD", "PVCC", "AVDD", "AVCC", "DVDD",
    # Battery pack/protection positive rail. NOT the negative side (P-,
    # B-, BAT-, PACK-) — in standard single-supply battery-protection
    # topology those are the ground return, not a second power node; see
    # _GROUND_PIN_TOKENS.
    "P+", "B+", "BAT+", "PACK+",
})
# Same idea as _POWER_PIN_TOKENS, for ground return paths that only show
# up in a hidden (hierarchical/auto-named) net name — e.g. an unlabeled
# "Net-(U1-GND)" pin, a local "/Analog/AGND" label, or a battery pack's
# negative terminal ("/Battery Protect/P-").
_GROUND_PIN_TOKENS = frozenset({
    "GND", "AGND", "DGND", "PGND", "SGND", "VSS", "0V", "COM",
    "P-", "B-", "BAT-", "PACK-",
})


def _hidden_rail_token(name: str) -> str | None:
    """Extract the trailing token from a hierarchical local label
    ("/Buck/VIN" -> "VIN") or KiCad's auto-generated fallback name
    ("Net-(U1-VIN)" -> "VIN"). Returns None if neither pattern matches
    (i.e. the name is already a plain, directly-labeled net)."""
    m = _HIER_LABEL_RE.match(name) or _AUTO_NET_RE.match(name)
    return m.group(1).upper() if m else None

# Cap value unit multipliers.
_UNIT_MULT = {
    "f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6,
    "m": 1e-3, "": 1.0, "k": 1e3, "meg": 1e6,
}


def parse_cap_value(value: str) -> float | None:
    """Parse a cap value string to Farads. Return None if unparseable.

    Handles KiCad notations: "100n", "u1", "4u7", "9p", "150u",
    "0.1u", "0.1uF", "4.7uF", "1u".
    """
    if not value:
        return None
    s = value.strip().lstrip("+").lower()
    # Strip trailing "f" / "farads" — the Farad unit suffix. Assumes nobody
    # is labeling caps in femtofarads (fF), which is never the case in
    # practical PCB work.
    s = re.sub(r"f(arads)?$", "", s)
    if not s:
        return None

    m = re.fullmatch(r"(\d*\.?\d*)([fpnumµ]|meg)?(\d*)", s)
    if not m:
        try:
            return float(s)
        except ValueError:
            return None

    mant_str, unit_str, frac_str = m.group(1), m.group(2), m.group(3)
    mult = _UNIT_MULT.get(unit_str or "")
    if mult is None:
        return None

    mant = float(mant_str) if mant_str else 0.0
    if frac_str:
        mant += int(frac_str) / (10 ** len(frac_str))
    return mant * mult


def is_power_net(name: str) -> bool:
    n = name.lstrip("/").upper()
    if any(n.startswith(p) for p in _POWER_PREFIXES):
        return True
    if _POWER_VOLTAGE_RE.match(n):
        return True
    token = _hidden_rail_token(name)
    return token in _POWER_PIN_TOKENS if token else False


def is_ground_net(name: str) -> bool:
    n = name.lstrip("/").upper()
    if any(n.startswith(p) for p in _GND_PREFIXES):
        return True
    token = _hidden_rail_token(name)
    return token in _GROUND_PIN_TOKENS if token else False


def _build_power_net_map(model: "BoardModel") -> dict[str, set[str]]:
    """Return {net_name: set_of_component_refs} for non-GND power nets."""
    power_nets: dict[str, set[str]] = {}
    for net in model.nets:
        if not is_power_net(net.name):
            continue
        if is_ground_net(net.name):
            continue
        power_nets[net.name] = set(net.component_refs)
    return power_nets


def _rail_pad_spread(ic: "Component", net_name: str) -> float | None:
    """Max pairwise distance (mm) between ``ic``'s pads on ``net_name``.

    Returns 0.0 for a single pad (trivially tight), and ``None`` when no
    pad-level net data is available for this component (e.g. a
    schematic-only model with no footprint) — callers treat unknown
    geometry as tight by default, since there's no evidence it's spread
    out.
    """
    pts = [(p.x, p.y) for p in getattr(ic, "pads", None) or [] if p.net == net_name]
    if not pts:
        return None
    if len(pts) == 1:
        return 0.0
    return max(
        math.dist(a, b)
        for i, a in enumerate(pts)
        for b in pts[i + 1:]
    )


def _classify_caps_impl(
    model: "BoardModel",
    rigid_spread_threshold_mm: float,
) -> tuple[dict[str, list[str]], dict[str, list[str]], list[str], list[str]]:
    """Shared classification core. Returns
    ``(decap_map, rail_adjacent_map, bulk_refs, coupling_refs)``.

    ``rail_adjacent_map`` groups the rail-adjacent (scattered-pin-group)
    caps by the IC they were round-robin assigned to — the per-IC
    pairing the public 4-tuple flattens and the Phase-A seeder consumes.
    """
    ics = [c for c in model.components if c.component_type in IC_TYPES]
    caps = [c for c in model.components if c.component_type == "capacitor"]
    if not ics or not caps:
        return {}, {}, [], []

    ref_to_type = {c.ref: c.component_type for c in model.components}
    ic_refs = {c.ref for c in ics}
    power_nets = _build_power_net_map(model)

    # Per-cap eligible IC set: {cap_ref: set_of_ic_refs_sharing_a_power_rail}
    cap_eligible_ics: dict[str, set[str]] = defaultdict(set)
    # Per-cap power-net names (for classification)
    cap_power_nets: dict[str, list[str]] = defaultdict(list)

    for net_name, refs in power_nets.items():
        net_ics = refs & ic_refs
        net_caps = {
            r for r in refs
            if r not in ic_refs and ref_to_type.get(r) == "capacitor"
        }
        for cap_ref in net_caps:
            cap_power_nets[cap_ref].append(net_name)
            if net_ics:
                cap_eligible_ics[cap_ref].update(net_ics)

    bulk_refs: list[str] = []
    coupling_refs: list[str] = []
    decap_to_ics: dict[str, set[str]] = {}

    for cap in caps:
        pnets = cap_power_nets.get(cap.ref, [])
        eligible = cap_eligible_ics.get(cap.ref, set())

        # No power net at all → coupling / signal cap.
        if not pnets:
            coupling_refs.append(cap.ref)
            continue

        # On a power rail but no IC shares it → bulk rail filter.
        if not eligible:
            bulk_refs.append(cap.ref)
            continue

        # On an IC-shared power rail. Classify by value. Unparseable
        # values default to BULK (conservative) — pairing an unknown
        # cap tightly to an IC as a rigid decoupling follower is riskier
        # than treating it as a standalone bulk cap.
        val_f = parse_cap_value(getattr(cap, "value", "") or "")
        if val_f is None or val_f > DECAP_MAX_VALUE_F:
            bulk_refs.append(cap.ref)
        else:
            decap_to_ics[cap.ref] = eligible

    # Round-robin: each decoupling cap → IC with the fewest decaps so far
    # (load-balanced across ICs sharing a rail), grouped by (ic, net) —
    # not just ic — because one IC can have several independent power
    # domains, each with its own pin cluster and its own rigid verdict.
    ref_to_component = {c.ref: c for c in model.components}
    ic_load: dict[str, int] = defaultdict(int)
    net_ic_caps: dict[tuple[str, str], list[str]] = defaultdict(list)
    for cap_ref in sorted(decap_to_ics):
        candidates = decap_to_ics[cap_ref]
        best_ic = min(sorted(candidates), key=lambda r: ic_load[r])
        ic_load[best_ic] += 1
        # Which of this cap's power nets is the one shared with best_ic?
        # Normally exactly one; for a cap spanning multiple rails, pick
        # the first that actually connects to the chosen IC.
        net_for_pair = next(
            (n for n in cap_power_nets[cap_ref] if best_ic in power_nets[n]),
            cap_power_nets[cap_ref][0],
        )
        net_ic_caps[(best_ic, net_for_pair)].append(cap_ref)

    # Rigid vs rail-adjacent is decided per (ic, net) pin group, not by
    # any per-IC cap count: if that rail's pads on the IC form one tight
    # cluster, ALL of that rail's real decoupling caps stay rigid — the
    # cap count is whatever the schematic actually put there. If the
    # pads are physically scattered, the whole group goes rail-adjacent
    # (held near the IC by soft attraction instead of rigid geometry).
    decap_map: dict[str, list[str]] = defaultdict(list)
    rail_adjacent_map: dict[str, list[str]] = defaultdict(list)
    for (ic_ref, net_name), cap_refs in net_ic_caps.items():
        ic = ref_to_component.get(ic_ref)
        spread = _rail_pad_spread(ic, net_name) if ic is not None else None
        is_tight = spread is None or spread <= rigid_spread_threshold_mm
        target = decap_map if is_tight else rail_adjacent_map
        target[ic_ref].extend(cap_refs)

    decap_map = {k: sorted(v) for k, v in decap_map.items()}
    rail_adjacent_map = {k: sorted(v) for k, v in rail_adjacent_map.items()}

    return decap_map, rail_adjacent_map, bulk_refs, coupling_refs


def classify_caps(
    model: "BoardModel",
    rigid_spread_threshold_mm: float = RAIL_PAD_SPREAD_THRESHOLD_MM,
) -> tuple[dict[str, list[str]], list[str], list[str], list[str]]:
    """Classify caps and assign decoupling caps to ICs.

    Returns ``(decap_map, rail_adjacent_refs, bulk_refs, coupling_refs)``
    where:

    - ``decap_map``: ``{ic_ref: [cap_ref, ...]}`` — rigid followers.
      Every real decoupling cap on a (IC, rail) pin group whose pads sit
      within ``rigid_spread_threshold_mm`` of each other stays rigid —
      there is no per-IC cap; an IC with several tight power domains can
      have a large rigid set. These become Macro followers in
      ``build_macros``.
    - ``rail_adjacent_refs``: decoupling caps on a (IC, rail) pin group
      whose pads are physically scattered beyond the threshold. They
      become standalone macros (not rigid followers) so SA can move them
      freely; the cap→IC attraction cost term holds them near the IC
      instead of rigid geometry. Use ``rail_adjacent_to_ic()`` to recover
      which IC each was assigned to.
    - ``bulk_refs``: caps that are bulk rail filters (large value, or on
      rails with no IC sharing). Standalone macros.
    - ``coupling_refs``: caps with no power net at all (signal-only).
      Standalone macros.
    """
    decap_map, rail_adjacent_map, bulk_refs, coupling_refs = _classify_caps_impl(
        model, rigid_spread_threshold_mm,
    )
    rail_adjacent_refs = [
        c for caps in rail_adjacent_map.values() for c in caps
    ]
    return decap_map, rail_adjacent_refs, bulk_refs, coupling_refs


def rail_adjacent_to_ic(
    model: "BoardModel",
    rigid_spread_threshold_mm: float = RAIL_PAD_SPREAD_THRESHOLD_MM,
) -> dict[str, str]:
    """Map each rail-adjacent cap to the IC it was round-robin assigned to.

    ``classify_caps`` splits each (IC, rail) pin group by pad-cluster
    tightness: tight groups become rigid followers, scattered groups
    become rail-adjacent. This function re-runs the same deterministic
    classification and returns the scattered-group pairing as
    ``{cap_ref: ic_ref}``.

    The Phase-A seeder uses this to place freed caps around their
    ASSIGNED IC (not just any rail peer). The assignment is load-balanced
    round-robin, so honoring it spreads excess caps across the rail's
    ICs instead of dumping them all at one.
    """
    _, rail_map, _, _ = _classify_caps_impl(model, rigid_spread_threshold_mm)
    return {cap: ic for ic, caps in rail_map.items() for cap in caps}


def assign_caps(model: "BoardModel") -> dict[str, list[str]]:
    """Backward-compatible wrapper: returns the rigid decoupling map only.

    Discards rail-adjacent/bulk/coupling info. Use ``classify_caps()`` for
    the full picture. Caps not in the returned dict become standalone
    macros naturally in ``pipeline.build_macros``.
    """
    decap_map, _, _, _ = classify_caps(model)
    return decap_map
