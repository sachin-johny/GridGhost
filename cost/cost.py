"""Single cost function: HPWL (with power rails) + overlap + boundary.

Today's pipeline fails because HPWL EXCLUDES power nets (the comment
"they're nearly constant" is true for routing but wrong for placement
coupling — caps share power rails with ICs, so power-net HPWL is the
ONLY gradient signal keeping caps near their assigned ICs).

This module includes power nets by default. Decoupling becomes a
first-class concern of the cost function instead of a soft constraint
that gets discounted during hot SA.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.board_model import BoardModel, Net
    from models.macro import Macro

# Soft target edge-gap for rail-adjacent (freed) caps around their assigned
# IC. Rigid followers are held at 4.0mm edge-to-edge by macro construction
# (assign_caps.MAX_CAP_IC_GAP_MM); freed caps get this looser SOFT target.
# The attraction term has a deadband: a cap inside the target gap costs
# nothing, so the term never fights the overlap penalty (which pushes
# macros apart) — it only charges drift BEYOND the target.
CAP_ATTRACTION_TARGET_GAP_MM = 5.0

# Default minimum edge-to-edge clearance between macro bboxes (the routing
# halo). Component bboxes already include type-aware courtyard margins
# (0.5-1.5mm, utils/courtyard.py), so this is clearance ON TOP of courtyard
# separation — roughly one trace lane + clearance class between neighbors.
# Set to 0.0 to disable the clearance term entirely.
CLEARANCE_TARGET_MM = 1.0

# Per-component-type clearance targets (mm, edge-to-edge on bbox).
# A pair's effective target is the MIN of the two members' targets. Pairs
# where EITHER member is a TestPoint are entirely exempt (see below).
#
# Rationale:
#   - mounting_hole / fiducial / test_coupon: 0mm (exempt — mechanical
#     features; pad+via stacks legitimately interleave, can't be moved
#     by SA, exemption matches _is_overlap_exempt).
#   - TestPoint (via footprint detection): 0mm (exempt — probe access
#     pads intentionally placed on/near IC pins; the 1mm uniform target
#     fought net-HPWL pull that pins them to the IC pin and lost, parking
#     them at 0.000mm gap on test4 with no gradient signal to improve).
#     Exempting them removes the false-positive charge and lets SA spend
#     its clearance budget on actual routing-halo pairs.
#   - All other types (ic, mcu, regulator, connector, crystal, capacitor,
#     resistor, generic): fall back to the caller's target_mm (default
#     1.0mm = the historical uniform target). Preserves behavior for
#     the heterogeneous `generic` class (LEDs, diodes, inductors of
#     widely varying sizes) and avoids the regression where reducing
#     cap/resistor targets let HPWL pull a diode onto a mounting hole.
#
# Originally this table also relaxed cap/resistor targets to 0.3mm under
# the reasoning that "small SMT passives don't need a 1mm halo". That was
# INVERTED: lowering the target reduces the deficit (max(0, target - gap))
# for the same gap, which WEAKENS SA's push-apart gradient — so cap-cap
# pairs ended up at the same gap or tighter, not looser. The right way to
# tighten DRC min-gap on cap-cap pairs is a higher target (more pressure)
# or a separate hard-DRC term, not a lower one.
_COMPONENT_CLEARANCE_TARGETS_MM: dict[str, float] = {
    "mounting_hole": 0.0,
    "fiducial":      0.0,
    "test_coupon":   0.0,
    # All other types fall back to the caller's target_mm (default 1.0).
}

# Footprint prefixes that identify test points (probe access pads).
# TestPoints are classified as `component_type="generic"` by the parser
# (no leading letter distinguishes them from R/C/U), so we need a
# footprint-name check to exempt them from the clearance term.
_TESTPOINT_FP_PREFIXES = (
    "TestPoint", "testpoint",
    "MeasurementPoint",
)


def _is_testpoint(comp: "object") -> bool:
    """Return True if comp's footprint identifies it as a probe test point."""
    fp = getattr(comp, "footprint", "") or ""
    # Strip "Library:" prefix if present (KiCad convention)
    if ":" in fp:
        fp = fp.split(":", 1)[1]
    return any(fp.startswith(p) for p in _TESTPOINT_FP_PREFIXES)


def _clearance_target_for_pair(
    a: "Macro",
    b: "Macro",
    default_mm: float,
) -> float:
    """Return the appropriate clearance target (mm) for this pair.

    0.0 means the pair is exempt from the clearance term entirely.
    The ``default_mm`` is used as a fallback for component types not in
    the per-class table.
    """
    from models.macro import _is_overlap_exempt

    # Mechanical-vs-mechanical pairs (mounting holes, fiducials) — exempt
    if _is_overlap_exempt(a, b):
        return 0.0

    a_lead = a.leader
    b_lead = b.leader

    # TestPoint pairs — exempt (intentional tight placement for probing)
    if _is_testpoint(a_lead) or _is_testpoint(b_lead):
        return 0.0

    a_target = _COMPONENT_CLEARANCE_TARGETS_MM.get(
        getattr(a_lead, "component_type", "") or "", default_mm)
    b_target = _COMPONENT_CLEARANCE_TARGETS_MM.get(
        getattr(b_lead, "component_type", "") or "", default_mm)
    # Use MIN of the two: the less-demanding component wins. A cap next to
    # an IC gets the cap's 0.3mm, not the IC's 1.0mm — the cap doesn't
    # need a full routing halo.
    return min(a_target, b_target)


def clearance_pair_charge(
    a_bbox: tuple[float, float, float, float],
    b_bbox: tuple[float, float, float, float],
    target_mm: float = CLEARANCE_TARGET_MM,
) -> float:
    """Linear clearance deficit between two bboxes.

    Returns ``max(0, target_mm − edge_gap)`` where ``edge_gap`` is the
    axis-aligned separation between the two boxes (negative when they
    intersect — clamped so the deficit equals ``target_mm`` at touching
    and never exceeds it: pairs that actually overlap are the overlap
    term's (β) job, not this term's; double-charging both would change
    the effective β mid-run).

    This is the pairwise primitive for ``clearance_deficit`` — see the
    discussion there for why this is a separate term rather than bbox
    inflation (``Macro.bbox`` feeds bounds checks, legalizer slot math,
    and density calculations; inflating it would corrupt all three).
    """
    ax1, ay1, ax2, ay2 = a_bbox
    bx1, by1, bx2, by2 = b_bbox
    dx = max(ax1 - bx2, bx1 - ax2)  # x separation (negative = overlap)
    dy = max(ay1 - by2, by1 - ay2)
    if dx < 0 and dy < 0:
        return target_mm  # bboxes intersect — full deficit, β handles depth
    return max(0.0, target_mm - max(dx, dy))


def clearance_deficit(
    macros: list["Macro"],
    target_mm: float = CLEARANCE_TARGET_MM,
) -> float:
    """Sum of pairwise clearance deficits (routing halo) across macros.

    ``β·overlap`` alone is discontinuous at touching: a 0.01mm gap and a
    1.5mm gap cost exactly the same (zero), so SA's gradient drives every
    pair to just-barely-not-overlapping and parks it there — the measured
    result on test4 was a 0.00-0.01mm minimum pair gap with ~3% of pairs
    under 1mm, i.e. no room to route between neighbors. This term adds
    the missing gradient: charge ``max(0, target − gap)`` per pair —
    linear ramp below the target, flat (zero) beyond it, so there is no
    incentive to spread further than one routing lane.

    Uses the same mechanical-feature exemption as the overlap terms via
    ``_is_overlap_exempt`` (mounting-hole pad stacks shouldn't be pushed
    apart — they're fixed and their bboxes legitimately interleave).

    Per-pair target selection (see ``_clearance_target_for_pair``):
      - Mechanical-mechanical pairs: exempt (0mm target).
      - Pairs with ANY TestPoint member: exempt (intentional tight
        placement for probe access).
      - Cap-cap / cap-resistor / resistor-resistor pairs: 0.3mm target
        (DRC min + one trace lane; small SMT passives don't need the
        full 1mm halo the IC class does).
      - IC-IC / IC-MCU pairs: 1.0mm (full routing halo).
      - Other pairs: per-component-class MIN of the two members, falling
        back to ``target_mm`` for unclassified types.

    The previous uniform 1.0mm target caused two distinct pathologies:
      1. TestPoint ↔ IC pairs at 0.000mm (U1↔TP16 on test4) — SA fought
         net-HPWL pull that pinned the test point to the IC pin,
         couldn't win, and parked at touching with no gradient signal
         to do better. Exempting them removes the false-positive charge.
      2. Cap-cap pairs at 0.130mm (C6↔C23 on test4, two 0402s) — the
         1mm target was so far above what's physically needed that SA
         had no gradient to push them apart to DRC-min (0.15mm). A
         0.3mm target gives SA 0.17mm of useful gradient without
         demanding a 1mm halo the small passives don't need.
    """
    from models.macro import _is_overlap_exempt

    total = 0.0
    n = len(macros)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = macros[i], macros[j]
            pair_target = _clearance_target_for_pair(a, b, target_mm)
            if pair_target <= 0.0:
                continue  # exempt pair
            total += clearance_pair_charge(a.bbox, b.bbox, pair_target)
    return total


def hpwl_net(net: "Net", ref_map: dict[str, "object"], weight: float = 1.0) -> float:
    """Standard half-perimeter wirelength for one net.

    Uses the bounding-box model: (max_x - min_x) + (max_y - min_y)
    over all pin positions on the net. Tight for 2-pin nets, lower
    bound for multi-pin nets.  ``weight`` scales the contribution — used
    by Issue 3's signal-flow-chain net weighting to pull chain members
    together.
    """
    if len(net.pins) < 2:
        return 0.0
    xs: list[float] = []
    ys: list[float] = []
    for ref, _ in net.pins:
        c = ref_map.get(ref)
        if c is None:
            continue
        xs.append(c.x)
        ys.append(c.y)
    if len(xs) < 2:
        return 0.0
    return ((max(xs) - min(xs)) + (max(ys) - min(ys))) * weight


def total_hpwl(
    model: "BoardModel",
    include_power: bool = True,
    exclude_nets: set[str] | None = None,
    net_weights: dict[str, float] | None = None,
) -> float:
    """Total HPWL across all nets.

    ``include_power=True`` includes power and ground nets — this is
    the key change vs the existing pipeline. Power-net HPWL is what
    gives SA gradient signal to keep caps near their assigned ICs.

    Ground nets can optionally be excluded via ``exclude_nets`` (they
    are nearly constant for HPWL because every component touches
    ground, but including them is harmless).

    ``net_weights`` optionally upweights specific nets (e.g. signal-flow-
    chain internal nets, Issue 3) by a multiplier; nets absent from the
    dict use weight 1.0.
    """
    ref_map = {c.ref: c for c in model.components}
    total = 0.0
    for net in model.nets:
        if exclude_nets and net.name in exclude_nets:
            continue
        w = net_weights.get(net.name, 1.0) if net_weights else 1.0
        total += hpwl_net(net, ref_map, weight=w)
    return total


def macro_overlap_area(a: "Macro", b: "Macro") -> float:
    """Intersection area of two macros' bboxes.

    Delegates to ``Macro.overlap_area`` so the mechanical-feature
    exemption (mounting holes, fiducials, test coupons) is applied
    consistently — otherwise the cost function sees phantom overlaps
    for mounting-hole pad+via stacks that the legalizer cannot resolve
    (they're fixed) and SA wastes move budget trying to push them apart.
    See ``models/macro.py:_is_overlap_exempt`` for the full rationale.
    """
    # ``Macro.overlap_area`` handles the exemption; fall back to raw
    # bbox math only if a non-Macro duck-typed object was passed (tests
    # sometimes use lightweight stand-ins).
    if hasattr(a, "overlap_area") and hasattr(b, "bbox"):
        return a.overlap_area(b)
    ax1, ay1, ax2, ay2 = a.bbox
    bx1, by1, bx2, by2 = b.bbox
    ox1 = max(ax1, bx1)
    oy1 = max(ay1, by1)
    ox2 = min(ax2, bx2)
    oy2 = min(ay2, by2)
    if ox2 <= ox1 or oy2 <= oy1:
        return 0.0
    return (ox2 - ox1) * (oy2 - oy1)


def total_macro_overlap(macros: list["Macro"]) -> float:
    """Sum of pairwise macro bbox-overlap areas.

    Uses ``Macro.overlap_area`` (via ``macro_overlap_area``) so the
    mechanical-feature exemption is applied uniformly across the
    cost function and the legalizer — see ``models/macro.py``.
    """
    total = 0.0
    n = len(macros)
    for i in range(n):
        for j in range(i + 1, n):
            total += macro_overlap_area(macros[i], macros[j])
    return total


def total_boundary(model: "BoardModel") -> float:
    """Total out-of-bounds distance for all components.

    Edge connectors (intentional overhang) are excluded. Returns the
    linear sum of how far each component's bbox pokes past the board
    outline — a smooth gradient SA can follow.

    Polygon-aware via ``BoardOutline.bbox_overflow``: for a rectangular
    outline this is exactly the old left+right+top+bottom overflow sum
    (preserved byte-for-byte by ``bbox_overflow``); for a non-rectangular
    outline (notches, mouse-bites, cutouts, holes) it charges each bbox
    corner that has strayed off the TRUE board for its distance back to
    the nearest outline edge — so SA on the default macro-v2 path gets a
    gradient away from concavities, not just the outer AABB.
    """
    b = model.board
    total = 0.0
    for c in model.components:
        if getattr(c, "is_edge_connector", False):
            continue
        total += b.bbox_overflow(c.bbox)
    return total


def cap_attraction_penalty(
    model: "BoardModel",
    cap_to_ic: dict[str, str],
    target_gap_mm: float = CAP_ATTRACTION_TARGET_GAP_MM,
) -> float:
    """Sum of deadband-linear drift for rail-adjacent caps.

    For each ``(cap, ic)`` pair, charge ``max(0, dist − target_gap)`` where
    ``dist`` is the Euclidean center distance between the cap and its
    assigned IC. The deadband makes this purely a drift penalty:

    - inside ``target_gap`` the term is flat — it never fights the overlap
      penalty (β, which pushes macros apart) or the congestion terms;
    - beyond it the gradient is constant-strength toward the IC, so SA has
      a signal to pull shared-rail caps back even when rail-bbox HPWL is
      flat w.r.t. the cap's position (the diagnosed root cause of cap drift
      on boards like test4: a cap on +3V3 can wander inside the 4-IC rail
      span with zero HPWL change).

    Distances are center-to-center (not edge-to-edge) — cheap and adequate
    for a soft term; the deadband absorbs the body sizes.
    """
    if not cap_to_ic:
        return 0.0
    ref_map = {c.ref: c for c in model.components}
    total = 0.0
    for cap_ref, ic_ref in cap_to_ic.items():
        cap = ref_map.get(cap_ref)
        ic = ref_map.get(ic_ref)
        if cap is None or ic is None:
            continue
        d = math.hypot(cap.x - ic.x, cap.y - ic.y)
        if d > target_gap_mm:
            total += d - target_gap_mm
    return total


def total_keepout_overlap(model: "BoardModel") -> float:
    """Total area of component bboxes intersecting internal keepouts.

    Sums the intersection area of each non-exempt component's bbox with
    every keepout in ``model.keepouts``. Returns 0.0 when the board has
    no internal cutouts (the common case — 5 of 6 bundled boards have
    no Edge.Cuts at all, cbbwO has only an outer rect).

    Edge connectors are exempt (a connector's body may legitimately
    overhang a mounting-hole zone near the board edge).

    Acts as a γ-style linear penalty: SA gets a smooth gradient pushing
    macros OUT of internal cutouts (mounting slots, milled pockets,
    non-plated through-holes). Without this term, SA is blind to
    keepouts — the legalizer's `_keepout_clamp` would have to do all the
    work reactively, fighting the SA gradient instead of reinforcing it.
    See IMPROVEMENTS §2.1.
    """
    if not model.keepouts:
        return 0.0
    total = 0.0
    for c in model.components:
        if getattr(c, "is_edge_connector", False):
            continue
        cx1, cy1, cx2, cy2 = c.bbox
        for k in model.keepouts:
            ox1 = max(cx1, k.x_min)
            oy1 = max(cy1, k.y_min)
            ox2 = min(cx2, k.x_max)
            oy2 = min(cy2, k.y_max)
            if ox2 > ox1 and oy2 > oy1:
                total += (ox2 - ox1) * (oy2 - oy1)
    return total


def evaluate(
    model: "BoardModel",
    macros: list["Macro"],
    *,
    alpha: float = 1.0,
    beta: float = 25.0,
    gamma: float = 8.0,
    include_power: bool = True,
    exclude_nets: set[str] | None = None,
    net_weights: dict[str, float] | None = None,
    rudy_weight: float = 0.0,
    rudy_penalty: float | None = None,
    pin_density_weight: float = 0.0,
    pin_density_penalty: float | None = None,
    keepout_weight: float | None = None,
    delta: float = 0.0,
    rules: list | None = None,
    constraint_penalty: float | None = None,
    cap_attraction_weight: float = 0.0,
    cap_pairs: dict[str, str] | None = None,
    clearance_weight: float = 0.0,
    clearance_target_mm: float = CLEARANCE_TARGET_MM,
) -> dict[str, float]:
    """Total placement cost.

    Args:
        alpha: HPWL weight (typically 1.0).
        beta: Overlap penalty weight (strong — SA must avoid overlaps).
        gamma: Boundary penalty weight.
        include_power: Whether to include power/ground nets in HPWL.
            Default True — required for cap-IC coupling.
        exclude_nets: Optional set of net names to drop from HPWL
            entirely (e.g. a single global ground net, which is nearly
            constant and adds no gradient signal but costs an O(n) scan
            every evaluation). Previously only reachable by calling
            ``total_hpwl`` directly; now threaded through so pipeline/SA
            callers can actually use it.
        net_weights: Optional per-net HPWL multipliers (e.g. signal-flow-chain
            internal nets from Issue 3).  Nets absent from the dict use 1.0.
        rudy_weight: RUDY congestion penalty weight (default 0 = disabled).
            When > 0, the RUDY penalty is added to the total cost so SA
            gets gradient signal to spread components away from routing
            choke points. Finding 7 fix — the legacy smart_placement path
            had RUDY wired in; the macro-v2 path didn't.
        rudy_penalty: Pre-computed RUDY penalty (avoids re-computing the
            RUDY map on every cost evaluation). If None and rudy_weight >
            0, the penalty is computed here (slower). Callers that
            evaluate cost in a tight SA loop should pre-compute the
            penalty every N steps and pass it in.
        pin_density_weight: Pin-density congestion penalty weight
            (default 0 = disabled). Complementary to ``rudy_weight``:
            RUDY sees wire density from net bounding boxes; pin density
            sees local pin-escape demand (a tight cluster of small
            passives next to a QFN has high pin density even if every
            net is short and contributes little to RUDY). Both default
            ON via config.json so the placer produces a routable result
            out of the box; pass weight 0 to disable either signal.
        pin_density_penalty: Pre-computed pin-density penalty (avoids
            re-computing the pin map on every cost evaluation). Same
            caching pattern as ``rudy_penalty``.
        keepout_weight: Internal-keepout overlap penalty weight. When
            None (default), uses ``gamma`` — internal cutouts are
            conceptually a boundary penalty (the macro is somewhere
            it can't be), so reusing γ gives a consistent gradient
            strength. Pass 0.0 to disable keepout enforcement in the
            cost function (the legalizer still clamps to keepouts via
            ``_keepout_clamp``; this just removes SA's gradient signal
            toward that outcome). See IMPROVEMENTS §2.1.
        delta: Constraint penalty weight (default 0 = disabled).
            When > 0 AND ``rules`` is provided, the cost function
            includes ``delta · constraint_penalty(model, rules)`` —
            the same constraint evaluator the legacy path uses
            (``engine/constraint_evaluator.py:evaluate_constraint_penalties``).
            Crystal-MCU proximity, thermal grouping/separation,
            analog/digital separation, decoupling proximity, etc.
            Default 0 preserves the current macro-v2 behavior (no
            constraint penalties) so this is a strictly additive
            opt-in. See IMPROVEMENTS §2.4.
        rules: List of ``ConstraintRule`` objects (from a board profile).
            Required when ``delta > 0``; ignored otherwise.
        constraint_penalty: Pre-computed constraint penalty (avoids
            re-computing the rule evaluation on every cost call). Same
            caching pattern as ``rudy_penalty``. SA callers should
            pass this in.
        cap_attraction_weight: Weight for the rail-adjacent cap→IC
            attraction term (default 0 = disabled). When > 0 AND
            ``cap_pairs`` is provided, adds
            ``cap_attraction_weight · cap_attraction_penalty`` — a
            deadband-linear charge on freed caps drifting beyond
            ``CAP_ATTRACTION_TARGET_GAP_MM`` from their assigned IC.
            Root-cause fix for shared-rail cap drift: on a rail like
            +3V3 shared by ICs spread across the board, rail-bbox HPWL
            is flat w.r.t. a freed cap's position, so SA has no signal
            keeping it near its assigned IC (measured on test4: seed
            8.8mm → 32.3mm post-SA). Dedicated-rail caps (+1V2,
            VCCPLL*) don't need it — their rail HPWL already pins them.
        cap_pairs: ``{cap_ref: ic_ref}`` assignment for rail-adjacent
            caps (from ``assign_caps.rail_adjacent_to_ic``). Required
            when ``cap_attraction_weight > 0``; ignored otherwise.
        clearance_weight: Weight for the pairwise clearance (routing
            halo) deficit. When > 0, adds ``clearance_weight ·
            clearance_deficit`` — a linear charge on macro pairs closer
            than ``clearance_target_mm`` edge-to-edge. Gives SA the
            gradient β·overlap lacks (overlap cost is 0 the instant
            bboxes stop intersecting, so pairs park at just-barely-
            touching with no trace lane between them). Pairs that
            actually overlap contribute a fixed deficit (the overlap
            term charges their depth) so the two terms never fight.
        clearance_target_mm: Target edge-to-edge gap between macro
            bboxes. Bboxes include courtyard margins, so this is a true
            routing lane on top of courtyard separation.

    Returns dict with hpwl, overlap, boundary, keepout, rudy, pin_density,
    constraint, cap_attraction, clearance, and total components.
    """
    h = total_hpwl(model, include_power=include_power, exclude_nets=exclude_nets,
                    net_weights=net_weights)
    o = total_macro_overlap(macros)
    b = total_boundary(model)
    k = total_keepout_overlap(model)
    # Keepout overlap defaults to γ (boundary weight) — same gradient
    # strength as the outer boundary penalty.
    kw = gamma if keepout_weight is None else keepout_weight
    r = 0.0
    if rudy_weight > 0:
        if rudy_penalty is None:
            try:
                from engine.congestion import rudy_congestion_penalty
                r, _peak, _avg, _overflow = rudy_congestion_penalty(model)
            except Exception:
                r = 0.0
        else:
            r = rudy_penalty
    p = 0.0
    if pin_density_weight > 0:
        if pin_density_penalty is None:
            try:
                from engine.congestion import pin_density_penalty as _pdp
                p, _peak, _avg, _overflow = _pdp(model)
            except Exception:
                p = 0.0
        else:
            p = pin_density_penalty
    # Constraint penalties — only computed when delta > 0 AND rules are
    # provided. Default delta=0 means this is a no-op on the macro-v2
    # path unless the caller explicitly opts in (see IMPROVEMENTS §2.4).
    c = 0.0
    if delta > 0 and rules:
        if constraint_penalty is None:
            try:
                from engine.constraint_evaluator import evaluate_constraint_penalties
                c, _breakdown = evaluate_constraint_penalties(model, rules)
            except Exception:
                c = 0.0
        else:
            c = constraint_penalty
    a = 0.0
    if cap_attraction_weight > 0 and cap_pairs:
        a = cap_attraction_penalty(model, cap_pairs)
    clr = 0.0
    if clearance_weight > 0:
        clr = clearance_deficit(macros, clearance_target_mm)
    return {
        "hpwl": h,
        "overlap": o,
        "boundary": b,
        "keepout": k,
        "rudy": r,
        "pin_density": p,
        "constraint": c,
        "cap_attraction": a,
        "clearance": clr,
        "total": (alpha * h + beta * o + gamma * b + kw * k
                  + rudy_weight * r
                  + pin_density_weight * p
                  + delta * c
                  + cap_attraction_weight * a
                  + clearance_weight * clr),
    }
