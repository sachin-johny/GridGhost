"""Assign each decoupling cap to exactly one IC.

For each cap on a power rail (+3V3, +12V, etc.), find ICs sharing that
rail and distribute round-robin (each cap to the IC with the fewest
caps so far). Every cap ends up assigned to exactly one IC; ICs sharing
a rail get roughly equal decoupling.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.board_model import BoardModel


IC_TYPES = frozenset({"ic", "mcu", "regulator"})

# Net-name patterns that mark power rails. Anything matching is treated
# as a power net; non-GND power nets are the cap-IC coupling signal.
_POWER_PREFIXES = (
    "GND", "AGND", "DGND", "PGND", "SGND",
    "VSS", "VCC", "VDD", "VEE", "VBAT", "VBUS",
)
_GND_PREFIXES = ("GND", "AGND", "DGND", "PGND", "SGND", "VSS")
_POWER_VOLTAGE_RE = re.compile(r"^[+\-]\d[\d.]*V", re.IGNORECASE)


def is_power_net(name: str) -> bool:
    n = name.lstrip("/").upper()
    if any(n.startswith(p) for p in _POWER_PREFIXES):
        return True
    return bool(_POWER_VOLTAGE_RE.match(n))


def is_ground_net(name: str) -> bool:
    n = name.lstrip("/").upper()
    return any(n.startswith(p) for p in _GND_PREFIXES)


def assign_caps(model: "BoardModel") -> dict[str, list[str]]:
    """Return ``{ic_ref: [cap_ref, ...]}``.

    Each cap appears in exactly one IC's list. Caps not on any shared
    power rail with an IC are not assigned (they will be placed as
    standalone macros).
    """
    ics = [c for c in model.components if c.component_type in IC_TYPES]
    caps = [c for c in model.components if c.component_type == "capacitor"]
    if not ics or not caps:
        return {}

    ref_to_type = {c.ref: c.component_type for c in model.components}
    ic_refs = {c.ref for c in ics}

    # net name → set of component refs on that net (non-GND power nets only).
    power_nets: dict[str, set[str]] = {}
    for net in model.nets:
        if not is_power_net(net.name):
            continue
        if is_ground_net(net.name):
            continue
        power_nets[net.name] = set(net.component_refs)

    # cap → set of IC refs sharing at least one non-GND power net
    cap_to_ics: dict[str, set[str]] = defaultdict(set)
    for net_name in sorted(power_nets):
        refs = power_nets[net_name]
        net_ics = refs & ic_refs
        if not net_ics:
            continue
        net_caps = {
            r for r in refs
            if r not in ic_refs and ref_to_type.get(r) == "capacitor"
        }
        for cap_ref in sorted(net_caps):
            cap_to_ics[cap_ref].update(net_ics)

    # Round-robin: each cap → IC with the fewest caps so far.
    ic_caps: dict[str, list[str]] = defaultdict(list)
    for cap_ref in sorted(cap_to_ics):
        candidates = cap_to_ics[cap_ref]
        if not candidates:
            continue
        best_ic = min(sorted(candidates), key=lambda r: len(ic_caps[r]))
        ic_caps[best_ic].append(cap_ref)

    return dict(ic_caps)
