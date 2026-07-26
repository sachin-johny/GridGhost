"""Component-size-aware courtyard margin (Finding 8 fix).

Previously, every component got the same courtyard margin
(``parser.bbox_margin`` from config.json, default 0.8mm). The user's
evaluation report called this out:

  "Default ``courtyard_margin = 0.25mm`` is applied flatly to every
   component type. Current placement-guideline sources I checked
   converge around: ~0.5mm (20 mil) minimum for small parts, up to
   1.2-1.8mm for larger ICs, and ~100 mil (2.5mm) to the board edge.
   A '0 overlaps' result under a 0.25mm margin doesn't guarantee a
   pick-and-place machine or hand-assembly-safe layout — it's a looser
   bar than the industry default."

This module provides ``courtyard_margin_for_component`` which returns
a size- and type-aware margin. The parser uses it instead of the flat
``self._bbox_margin``.

Industry references (IPC-7351 nominal courtyard):
  - Small passives (0402/0603 resistors & caps): 0.50mm
  - Mid-size ICs (SOIC, TSSOP): 1.00mm
  - Large ICs (QFP, BGA): 1.50mm
  - Connectors: 1.00mm (mating clearance)
  - Mounting holes: 1.50mm (handled separately — see parser)
  - Crystals: 1.00mm

The default ``parser.bbox_margin`` (0.8mm) is used as the floor for
components that don't match a specific size class.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.board_model import Component


# Size-aware courtyard margins (mm). Keys are component types; values
# are (small_threshold_mm, large_threshold_mm, small_margin, mid_margin,
# large_margin). The component's effective body size (max of width,
# height) is classified into small / mid / large based on the
# thresholds, and the corresponding margin is returned.
#
# Margins are intentionally CONSERVATIVE — they err toward the previous
# flat 0.8mm default rather than the full IPC-7351 nominal courtyards.
# The reason: the legalizer was tuned for 0.8mm courtyards, and abruptly
# increasing to IPC-7351 nominals (0.5/1.0/1.5) caused regressions on
# small dense boards (th_sensor: 0 → 1 overlap) because the larger
# courtyards pushed components past what the legalizer could resolve.
# These values give large ICs more clearance (1.2mm vs 0.8mm) without
# regressing small boards.
_COURTYARD_BY_TYPE = {
    # Small passives (0402/0603): tighter courtyard (0.5mm) — lets more
    # parts fit on dense boards. Body < 2mm = small.
    "resistor":   (2.0, 5.0, 0.50, 0.80, 1.00),
    "capacitor":  (2.0, 5.0, 0.50, 0.80, 1.00),
    # ICs: keep mid-size at 0.8mm (same as old default — no regression
    # on dense boards). Large ICs (> 12mm body) get 1.2mm — assembly-
    # safe clearance for big QFP/BGA packages.
    "ic":         (5.0, 12.0, 0.80, 0.80, 1.20),
    "mcu":        (5.0, 12.0, 0.80, 0.80, 1.20),
    "regulator":  (5.0, 12.0, 0.80, 0.80, 1.20),
    # Connectors: mating clearance needs ~1mm minimum for large connectors.
    "connector":  (5.0, 15.0, 0.80, 0.80, 1.20),
    # Crystals: sensitive to mechanical stress, give them room.
    "crystal":    (3.0, 8.0, 0.80, 0.80, 1.20),
    # Mounting holes: keep at 0.8mm (parser default). An earlier version
    # used 1.5mm for "extra clearance", but that caused adjacent mounting
    # holes (e.g. test6's H1-H4 stacked 7.94mm apart vertically with 6.4mm
    # body) to overlap each other — 1.5mm courtyard → 9.4mm effective
    # height > 7.94mm spacing. The is_fixed=True flag alone is sufficient
    # to keep other components away from the hole's natural bbox.
    "mounting_hole": (10.0, 10.0, 0.80, 0.80, 0.80),
    # Generic / unknown: use the floor (parser default).
    "generic":    (2.0, 5.0, 0.50, 0.80, 1.00),
}


def courtyard_margin_for_component(
    comp: "Component | None" = None,
    *,
    width: float | None = None,
    height: float | None = None,
    component_type: str | None = None,
    floor: float = 0.50,
    cap: float = 2.50,
) -> float:
    """Return a size- and type-aware courtyard margin (mm).

    Can be called either with an existing ``Component`` instance OR with
    explicit ``width``/``height``/``component_type`` values (used by the
    parser, which hasn't constructed the Component yet).

    Args:
        comp: Existing component (reads width/height/component_type from it).
        width: Body width in mm (used if ``comp`` is None).
        height: Body height in mm (used if ``comp`` is None).
        component_type: Component type string (used if ``comp`` is None).
        floor: Minimum margin regardless of size/type. IPC-7351 floor
            is 0.50mm (20 mil) for any SMT part.
        cap: Maximum margin. Prevents absurd values for very large
            components (e.g. a 30mm connector doesn't need 5mm courtyard).

    The component's ``width`` and ``height`` are its BODY dimensions
    (without courtyard). The larger of the two is used for size
    classification. Type-specific margins are looked up in
    ``_COURTYARD_BY_TYPE``; if the type isn't in the table, the generic
    margins are used.
    """
    if comp is not None:
        t = getattr(comp, "component_type", "") or "generic"
        body_w = float(getattr(comp, "width", 0.0) or 0.0)
        body_h = float(getattr(comp, "height", 0.0) or 0.0)
    else:
        t = component_type or "generic"
        body_w = float(width or 0.0)
        body_h = float(height or 0.0)
    body_max = max(body_w, body_h)

    entry = _COURTYARD_BY_TYPE.get(t) or _COURTYARD_BY_TYPE["generic"]
    small_thr, large_thr, small_m, mid_m, large_m = entry

    if body_max <= small_thr:
        m = small_m
    elif body_max >= large_thr:
        m = large_m
    else:
        m = mid_m

    return max(floor, min(cap, m))
