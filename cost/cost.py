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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.board_model import BoardModel, Net
    from models.macro import Macro


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

    Returns dict with hpwl, overlap, boundary, keepout, rudy, pin_density,
    constraint, and total components.
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
    return {
        "hpwl": h,
        "overlap": o,
        "boundary": b,
        "keepout": k,
        "rudy": r,
        "pin_density": p,
        "constraint": c,
        "total": (alpha * h + beta * o + gamma * b + kw * k
                  + rudy_weight * r
                  + pin_density_weight * p
                  + delta * c),
    }
