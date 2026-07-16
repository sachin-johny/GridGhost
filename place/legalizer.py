"""Single-pass legalizer.

Three steps:
1. Grid snap macro leaders to ``grid_mm`` (followers stay at their
   fixed offsets relative to the leader).
2. Push apart overlapping macros (greedy, macro-aware). Each push is
   rigid — the whole macro moves, followers come along.
3. Boundary clamp.

The <8mm cap-IC distance is preserved automatically because macros
are rigid throughout. If push-apart or boundary clamping can't place
a macro without breaking it, the legalizer reports the failure and
leaves the macro at the best available position.

There is no spread pass, no abacus DP, no reheat rounds, no "brute-
force safety net". The SA + this legalize pass is the whole story.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from models.macro import Macro, MAX_CAP_IC_DISTANCE_MM

if TYPE_CHECKING:
    from models.board_model import BoardModel, BoardOutline


def _snap_to_grid(value: float, grid_mm: float) -> float:
    return round(value / grid_mm) * grid_mm


def grid_snap(macros: list["Macro"], grid_mm: float, bounds: tuple[float, float, float, float]) -> None:
    """Snap each macro's leader to the grid; followers follow rigidly."""
    for m in macros:
        old = (m.leader.x, m.leader.y, m.leader.rotation)
        gx = _snap_to_grid(m.leader.x, grid_mm)
        gy = _snap_to_grid(m.leader.y, grid_mm)
        if not m.set_pose(gx, gy, m.leader.rotation, bounds=bounds):
            # Grid snap failed bounds check — keep original position
            m.leader.x, m.leader.y = old[0], old[1]
            m.leader.set_rotation(old[2])
            m.apply_offsets()


def push_apart_overlapping(
    macros: list["Macro"],
    bounds: tuple[float, float, float, float],
    max_passes: int = 200,
) -> int:
    """Greedy macro-macro overlap repair.

    For each overlapping pair, push the smaller macro along the cheaper
    axis (the one with less overlap depth). If that push is rejected by
    bounds, try the other macro, then the perpendicular axis, then
    fractional pushes (half, quarter, eighth). Returns residual overlap
    count after ``max_passes`` passes (or earlier when no progress).
    """
    for _ in range(max_passes):
        any_overlap = False
        any_resolved = False

        for i in range(len(macros)):
            for j in range(i + 1, len(macros)):
                a = macros[i]
                b = macros[j]
                if not a.overlaps(b):
                    continue
                any_overlap = True

                ax1, ay1, ax2, ay2 = a.bbox
                bx1, by1, bx2, by2 = b.bbox
                ox = min(ax2, bx2) - max(ax1, bx1)
                oy = min(ay2, by2) - max(ay1, by1)

                a_area = (ax2 - ax1) * (ay2 - ay1)
                b_area = (bx2 - bx1) * (by2 - by1)
                mover = a if a_area < b_area else b
                other = b if mover is a else a

                # Cheaper axis first: lower overlap depth = easier escape.
                # If tied, prefer X (arbitrary).
                if ox <= oy:
                    cheap_axis, cheap_depth = "x", ox
                    steep_axis, steep_depth = "y", oy
                else:
                    cheap_axis, cheap_depth = "y", oy
                    steep_axis, steep_depth = "x", ox

                attempts = [
                    (mover, cheap_axis, cheap_depth, 1.0),
                    (other, cheap_axis, cheap_depth, 1.0),
                    (mover, steep_axis, steep_depth, 1.0),
                    (other, steep_axis, steep_depth, 1.0),
                    (mover, cheap_axis, cheap_depth, 0.5),
                    (other, cheap_axis, cheap_depth, 0.5),
                    (mover, cheap_axis, cheap_depth, 0.25),
                    (other, cheap_axis, cheap_depth, 0.25),
                    (mover, cheap_axis, cheap_depth, 0.125),
                    (other, cheap_axis, cheap_depth, 0.125),
                ]
                for who, axis, depth, frac in attempts:
                    if _try_push(who, other if who is mover else mover, axis, depth, frac, bounds):
                        any_resolved = True
                        break

        if not any_overlap:
            return 0
        if not any_resolved:
            break

    residual = 0
    for i in range(len(macros)):
        for j in range(i + 1, len(macros)):
            if macros[i].overlaps(macros[j]):
                residual += 1
    return residual


def _try_push(
    mover: "Macro",
    other: "Macro",
    axis: str,
    depth: float,
    frac: float,
    bounds: tuple[float, float, float, float],
) -> bool:
    """Push `mover` out of `other` along `axis` ('x' or 'y') by depth*frac.

    Direction is set by which side of `other` the `mover` is currently on.
    Returns True on success, restores mover on failure.
    """
    snap = mover._snapshot()
    mx1, my1, _, _ = mover.bbox
    ox1, oy1, _, _ = other.bbox
    mag = depth * frac + 0.01

    if axis == "x":
        sign = -1.0 if mx1 < ox1 else 1.0
        ok = mover.translate(sign * mag, 0.0, bounds=bounds)
    else:
        sign = -1.0 if my1 < oy1 else 1.0
        ok = mover.translate(0.0, sign * mag, bounds=bounds)

    if ok:
        return True
    mover._restore(snap)
    return False


def boundary_clamp(macros: list["Macro"], bounds: tuple[float, float, float, float]) -> int:
    """Clamp each macro inside bounds. Returns count of macros that couldn't fit.

    A macro that can't fit at any position (e.g., macro bbox larger
    than bounds) is left at its current pose; the caller must handle.
    """
    x_min, y_min, x_max, y_max = bounds
    failed = 0
    for m in macros:
        bx1, by1, bx2, by2 = m.bbox
        dx_left = x_min - bx1
        dx_right = x_max - bx2
        dy_top = y_min - by1
        dy_bottom = y_max - by2

        dx = 0.0
        dy = 0.0
        if dx_left > 0:
            dx = dx_left
        elif dx_right < 0:
            dx = dx_right
        if dy_top > 0:
            dy = dy_top
        elif dy_bottom < 0:
            dy = dy_bottom

        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            continue

        if not m.translate(dx, dy, bounds=bounds):
            # Try only x or only y
            if abs(dx) > 1e-9 and m.translate(dx, 0.0, bounds=bounds):
                continue
            if abs(dy) > 1e-9 and m.translate(0.0, dy, bounds=bounds):
                continue
            failed += 1
    return failed


def legalize(
    model: "BoardModel",
    macros: list["Macro"],
    bounds: tuple[float, float, float, float],
    *,
    grid_mm: float = 1.0,
    max_push_passes: int = 200,
    max_rounds: int = 5,
    verbose: bool = False,
) -> dict[str, int]:
    """Iterative legalization: snap → (push apart → clamp) repeated.

    Each round repairs the overlaps created by the previous clamp, then
    clamps the resulting positions back inside bounds. Stops when a
    round makes no progress or after ``max_rounds`` iterations.

    Returns a dict with overlap count, boundary-failure count, and
    cap-IC distance violation count.
    """
    grid_snap(macros, grid_mm, bounds)

    residual = push_apart_overlapping(macros, bounds, max_passes=max_push_passes)
    failed = boundary_clamp(macros, bounds)

    # Iterate: clamp creates overlaps, push-apart fixes them but may push
    # something back OOB, clamp fixes that, etc. Continue until stable.
    for round_idx in range(max_rounds):
        new_residual = push_apart_overlapping(macros, bounds, max_passes=max_push_passes)
        new_failed = boundary_clamp(macros, bounds)
        if new_residual == residual and new_failed == failed:
            # No progress this round.
            residual, failed = new_residual, new_failed
            break
        residual, failed = new_residual, new_failed
        if residual == 0 and failed == 0:
            break

    # One last push-apart to clean up overlaps introduced by the final clamp.
    if failed > 0:
        residual = push_apart_overlapping(macros, bounds, max_passes=max_push_passes)

    cap_ic_violations = 0
    for m in macros:
        if not m.followers:
            continue
        for f in m.followers:
            d = math.hypot(f.x - m.leader.x, f.y - m.leader.y)
            if d > MAX_CAP_IC_DISTANCE_MM + 1e-6:
                cap_ic_violations += 1

    if verbose:
        print(
            f"  Legalize: {residual} residual overlaps, "
            f"{failed} boundary failures, {cap_ic_violations} cap-IC distance violations"
        )

    return {
        "residual_overlaps": residual,
        "boundary_failures": failed,
        "cap_ic_violations": cap_ic_violations,
    }
