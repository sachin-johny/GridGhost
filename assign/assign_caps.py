"""Classify caps and assign each decoupling cap to exactly one IC.

Four cap classes:

- **Decoupling (rigid)** — the first ``max_decaps_per_ic`` caps assigned
  to each IC become macro followers and stay within
  ``MAX_CAP_IC_GAP_MM`` (edge-to-edge) of their leader. These are the
  "inner ring" of critical bypass caps that must be as close as
  physically possible to the IC power pins.

- **Decoupling (rail-adjacent)** — excess caps beyond
  ``max_decaps_per_ic`` per IC. These are small-value caps on the same
  power rail, but there isn't room for all of them in the rigid fan
  around the IC. They become standalone macros (not rigid followers); SA
  can move them freely. This avoids rigidly gluing a large cap fan to one
  IC on dense shared-rail boards (e.g. test4: 10 caps on U30's +3V3
  rail), which starves SA of degrees of freedom and produces a bulky
  macro block that is hard for the legalizer to place without cap-IC
  overlap.

- **Bulk** — large value (>10uF) OR on a power rail that no IC shares.
  These are rail-level filters (regulator output, board input rail, etc.)
  and are placed standalone; they do not belong to any one IC.

- **Coupling / signal** — only on signal nets (crystal loads, shields,
  USB D+/D-, reset filters). No power rail at all. Standalone.

Decoupling caps are distributed round-robin across the ICs sharing the
rail so each IC gets roughly equal decoupling. The first N per IC
(default 2) become rigid followers; the rest become rail-adjacent.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.board_model import BoardModel

IC_TYPES = frozenset({"ic", "mcu", "regulator"})

# Caps larger than this are "bulk", not decoupling followers.
# 10µF covers common 4.7/10µF IC bulk-decoupling caps; the previous 1µF
# threshold mis-classified them as decoupling followers (which get paired
# tightly to a single IC via a rigid macro).
DECAP_MAX_VALUE_F = 10e-6  # 10uF

# Maximum decoupling caps per IC that become rigid macro followers.
# The rest become standalone "rail-adjacent" caps that SA can move freely.
# 2 is the standard bypass config: one 100nF (high-freq) + one bulk/low-ESL
# (mid-freq) per IC power pin pair. Rigidly gluing more than 2 creates a
# large rigid fan around the IC: it starves SA of degrees of freedom and
# produces a bulky macro that is hard for the legalizer to place without
# cap-IC overlap (observed on test4: 10 caps on U30).
MAX_DECAPS_PER_IC = 2

# Net-name patterns that mark power rails.
_POWER_PREFIXES = (
    "GND", "AGND", "DGND", "PGND", "SGND",
    "VSS", "VCC", "VDD", "VEE", "VBAT", "VBUS",
)
_GND_PREFIXES = ("GND", "AGND", "DGND", "PGND", "SGND", "VSS")
_POWER_VOLTAGE_RE = re.compile(r"^[+\-]\d[\d.]*V", re.IGNORECASE)

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
    return bool(_POWER_VOLTAGE_RE.match(n))


def is_ground_net(name: str) -> bool:
    n = name.lstrip("/").upper()
    return any(n.startswith(p) for p in _GND_PREFIXES)


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


def _classify_caps_impl(
    model: "BoardModel",
    max_decaps_per_ic: int,
) -> tuple[dict[str, list[str]], dict[str, list[str]], list[str], list[str]]:
    """Shared classification core. Returns
    ``(decap_map, rail_adjacent_map, bulk_refs, coupling_refs)``.

    ``rail_adjacent_map`` groups the rail-adjacent (excess) caps by the
    IC they were round-robin assigned to — the per-IC pairing the public
    4-tuple flattens and the Phase-A seeder consumes.
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

    # Round-robin: each decoupling cap → IC with the fewest decaps so far.
    # This produces the FULL assignment (including caps that will later
    # become rail-adjacent).
    full_ic_caps: dict[str, list[str]] = defaultdict(list)
    for cap_ref in sorted(decap_to_ics):
        candidates = decap_to_ics[cap_ref]
        best_ic = min(sorted(candidates), key=lambda r: len(full_ic_caps[r]))
        full_ic_caps[best_ic].append(cap_ref)

    # Split into rigid followers (≤ max_decaps_per_ic per IC) and
    # rail-adjacent (the rest), KEPT AS A PER-IC MAP so the Phase-A
    # seeder knows which IC each freed cap was assigned to.
    decap_map: dict[str, list[str]] = {}
    rail_adjacent_map: dict[str, list[str]] = {}
    for ic_ref, all_caps in full_ic_caps.items():
        rigid = all_caps[:max_decaps_per_ic]
        excess = all_caps[max_decaps_per_ic:]
        if rigid:
            decap_map[ic_ref] = rigid
        if excess:
            rail_adjacent_map[ic_ref] = excess

    return decap_map, rail_adjacent_map, bulk_refs, coupling_refs


def classify_caps(
    model: "BoardModel",
    max_decaps_per_ic: int = MAX_DECAPS_PER_IC,
) -> tuple[dict[str, list[str]], list[str], list[str], list[str]]:
    """Classify caps and assign decoupling caps to ICs.

    Returns ``(decap_map, rail_adjacent_refs, bulk_refs, coupling_refs)``
    where:

    - ``decap_map``: ``{ic_ref: [cap_ref, ...]}`` — rigid followers,
      at most ``max_decaps_per_ic`` caps per IC. These become Macro
      followers in ``build_macros``.
    - ``rail_adjacent_refs``: decoupling caps that exceeded the per-IC
      rigid limit. They become standalone macros (not rigid followers)
      so SA can move them freely. They share the rail, so HPWL keeps
      them near the IC cluster without rigidly locking them. Use
      ``rail_adjacent_to_ic()`` to recover which IC each was assigned to.
    - ``bulk_refs``: caps that are bulk rail filters (large value, or on
      rails with no IC sharing). Standalone macros.
    - ``coupling_refs``: caps with no power net at all (signal-only).
      Standalone macros.
    """
    decap_map, rail_adjacent_map, bulk_refs, coupling_refs = _classify_caps_impl(
        model, max_decaps_per_ic,
    )
    rail_adjacent_refs = [
        c for caps in rail_adjacent_map.values() for c in caps
    ]
    return decap_map, rail_adjacent_refs, bulk_refs, coupling_refs


def rail_adjacent_to_ic(
    model: "BoardModel",
    max_decaps_per_ic: int = MAX_DECAPS_PER_IC,
) -> dict[str, str]:
    """Map each rail-adjacent cap to the IC it was round-robin assigned to.

    ``classify_caps`` splits each IC's full assignment at
    ``max_decaps_per_ic``: the head becomes rigid followers, the tail
    becomes rail-adjacent. This function re-runs the same deterministic
    classification and returns the tail pairing as ``{cap_ref: ic_ref}``.

    The Phase-A seeder uses this to place freed caps around their
    ASSIGNED IC (not just any rail peer). The assignment is load-balanced
    round-robin, so honoring it spreads excess caps across the rail's
    ICs instead of dumping them all at one.
    """
    _, rail_map, _, _ = _classify_caps_impl(model, max_decaps_per_ic)
    return {cap: ic for ic, caps in rail_map.items() for cap in caps}


def assign_caps(model: "BoardModel") -> dict[str, list[str]]:
    """Backward-compatible wrapper: returns the rigid decoupling map only.

    Discards rail-adjacent/bulk/coupling info. Use ``classify_caps()`` for
    the full picture. Caps not in the returned dict become standalone
    macros naturally in ``pipeline.build_macros``.
    """
    decap_map, _, _, _ = classify_caps(model)
    return decap_map
