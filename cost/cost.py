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


def hpwl_net(net: "Net", ref_map: dict[str, "object"]) -> float:
    """Standard half-perimeter wirelength for one net.

    Uses the bounding-box model: (max_x - min_x) + (max_y - min_y)
    over all pin positions on the net. Tight for 2-pin nets, lower
    bound for multi-pin nets.
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
    return (max(xs) - min(xs)) + (max(ys) - min(ys))


def total_hpwl(
    model: "BoardModel",
    include_power: bool = True,
    exclude_nets: set[str] | None = None,
) -> float:
    """Total HPWL across all nets.

    ``include_power=True`` includes power and ground nets — this is
    the key change vs the existing pipeline. Power-net HPWL is what
    gives SA gradient signal to keep caps near their assigned ICs.

    Ground nets can optionally be excluded via ``exclude_nets`` (they
    are nearly constant for HPWL because every component touches
    ground, but including them is harmless).
    """
    ref_map = {c.ref: c for c in model.components}
    total = 0.0
    for net in model.nets:
        if exclude_nets and net.name in exclude_nets:
            continue
        total += hpwl_net(net, ref_map)
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
) -> dict[str, float]:
    """Total placement cost.

    Args:
        alpha: HPWL weight (typically 1.0).
        beta: Overlap penalty weight (strong — SA must avoid overlaps).
        gamma: Boundary penalty weight.
        include_power: Whether to include power/ground nets in HPWL.
            Default True — required for cap-IC coupling.

    Returns dict with hpwl, overlap, boundary, and total components.
    """
    h = total_hpwl(model, include_power=include_power)
    o = total_macro_overlap(macros)
    b = total_boundary(model)
    return {
        "hpwl": h,
        "overlap": o,
        "boundary": b,
        "total": alpha * h + beta * o + gamma * b,
    }
