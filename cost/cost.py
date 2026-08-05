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
    """Intersection area of two macros' bboxes."""
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
    """Sum of pairwise macro bbox-overlap areas."""
    total = 0.0
    n = len(macros)
    for i in range(n):
        for j in range(i + 1, n):
            total += macro_overlap_area(macros[i], macros[j])
    return total


def total_boundary(model: "BoardModel") -> float:
    """Total out-of-bounds distance for all components.

    Edge connectors (intentional overhang) are excluded. Returns the
    linear sum of how far each component's bbox pokes past each board
    edge — a smooth gradient SA can follow.
    """
    b = model.board
    total = 0.0
    for c in model.components:
        if getattr(c, "is_edge_connector", False):
            continue
        x1, y1, x2, y2 = c.bbox
        if x1 < b.x_min:
            total += b.x_min - x1
        if y1 < b.y_min:
            total += b.y_min - y1
        if x2 > b.x_max:
            total += x2 - b.x_max
        if y2 > b.y_max:
            total += y2 - b.y_max
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

    Returns dict with hpwl, overlap, boundary, rudy, pin_density, and
    total components.
    """
    h = total_hpwl(model, include_power=include_power, exclude_nets=exclude_nets,
                    net_weights=net_weights)
    o = total_macro_overlap(macros)
    b = total_boundary(model)
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
    return {
        "hpwl": h,
        "overlap": o,
        "boundary": b,
        "rudy": r,
        "pin_density": p,
        "total": (alpha * h + beta * o + gamma * b
                  + rudy_weight * r
                  + pin_density_weight * p),
    }
