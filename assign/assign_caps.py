"""Classify caps and assign each decoupling cap to exactly one IC.

Three cap classes:

- **Decoupling** — small value (<=1uF) on a non-GND power rail shared with
  at least one IC. These become macro followers and stay within
  ``MAX_CAP_IC_GAP_MM`` (edge-to-edge) of their leader.

- **Bulk** — large value (>1uF) OR on a power rail that no IC shares.
  These are rail-level filters (regulator output, board input rail, etc.)
  and are placed standalone; they do not belong to any one IC.

- **Coupling / signal** — only on signal nets (crystal loads, shields,
  USB D+/D-, reset filters). No power rail at all. Standalone.

Decoupling caps are distributed round-robin across the ICs sharing the
rail so each IC gets roughly equal decoupling.
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


def classify_caps(
    model: "BoardModel",
) -> tuple[dict[str, list[str]], list[str], list[str]]:
    """Classify caps and assign decoupling caps to ICs.

    Returns ``(decap_map, bulk_refs, coupling_refs)`` where:

    - ``decap_map``: ``{ic_ref: [cap_ref, ...]}`` — each cap in exactly
      one IC's list, distributed round-robin by fewest-caps-so-far.
    - ``bulk_refs``: caps that are bulk rail filters (large value, or on
      rails with no IC sharing). Standalone macros.
    - ``coupling_refs``: caps with no power net at all (signal-only).
      Standalone macros.
    """
    ics = [c for c in model.components if c.component_type in IC_TYPES]
    caps = [c for c in model.components if c.component_type == "capacitor"]
    if not ics or not caps:
        return {}, [], []

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
    ic_caps: dict[str, list[str]] = defaultdict(list)
    for cap_ref in sorted(decap_to_ics):
        candidates = decap_to_ics[cap_ref]
        best_ic = min(sorted(candidates), key=lambda r: len(ic_caps[r]))
        ic_caps[best_ic].append(cap_ref)

    return dict(ic_caps), bulk_refs, coupling_refs


def assign_caps(model: "BoardModel") -> dict[str, list[str]]:
    """Backward-compatible wrapper: returns the decoupling map only.

    Discards bulk/coupling info. Use ``classify_caps()`` for the full
    picture. Caps not in the returned dict become standalone macros
    naturally in ``pipeline.build_macros``.
    """
    decap_map, _, _ = classify_caps(model)
    return decap_map
