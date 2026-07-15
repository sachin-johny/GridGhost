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
    max_passes: int = 50,
) -> int:
    """Greedy macro-macro overlap repair.

    For each pair of overlapping macros, push the smaller one along
    the cheaper axis (the one with less overlap depth). Moves are
    rigid; followers come along. Returns the number of overlaps
    remaining after ``max_passes`` passes.
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

                # Compute overlap depth on each axis
                ax1, ay1, ax2, ay2 = a.bbox
                bx1, by1, bx2, by2 = b.bbox
                ox = min(ax2, bx2) - max(ax1, bx1)
                oy = min(ay2, by2) - max(ay1, by1)

                # Push along the axis with LESS overlap (cheaper escape)
                # Move the smaller macro (by area)
                a_area = (ax2 - ax1) * (ay2 - ay1)
                b_area = (bx2 - bx1) * (by2 - by1)
                mover = a if a_area < b_area else b
                other = b if mover is a else a

                snap = mover._snapshot()
                if ox < oy:
                    # Push along X
                    mx1, _, mx2, _ = mover.bbox
                    ox1, _, ox2, _ = other.bbox
                    if mx1 < ox1:
                        dx = -(ox + 0.01)
                    else:
                        dx = ox + 0.01
                    ok = mover.translate(dx, 0.0, bounds=bounds)
                else:
                    # Push along Y
                    _, my1, _, my2 = mover.bbox
                    _, oy1, _, oy2 = other.bbox
                    if my1 < oy1:
                        dy = -(oy + 0.01)
                    else:
                        dy = oy + 0.01
                    ok = mover.translate(0.0, dy, bounds=bounds)

                if not ok:
                    # Push rejected (bounds). Try the other macro.
                    snap2 = other._snapshot()
                    if ox < oy:
                        mx1, _, mx2, _ = other.bbox
                        ox1, _, ox2, _ = mover.bbox
                        if mx1 < ox1:
                            dx = -(ox + 0.01)
                        else:
                            dx = ox + 0.01
                        ok = other.translate(dx, 0.0, bounds=bounds)
                    else:
                        _, my1, _, my2 = other.bbox
                        _, oy1, _, oy2 = mover.bbox
                        if my1 < oy1:
                            dy = -(oy + 0.01)
                        else:
                            dy = oy + 0.01
                        ok = other.translate(0.0, dy, bounds=bounds)
                    if not ok:
                        # Neither can move; restore both and give up on this pair
                        mover._restore(snap)
                        continue
                    else:
                        mover._restore(snap)
                any_resolved = True

        if not any_overlap:
            return 0
        if not any_resolved:
            # Made no progress; further passes won't help
            break

    # Count residual overlaps
    residual = 0
    for i in range(len(macros)):
        for j in range(i + 1, len(macros)):
            if macros[i].overlaps(macros[j]):
                residual += 1
    return residual


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
    max_push_passes: int = 50,
    verbose: bool = False,
) -> dict[str, int]:
    """Single-pass legalization: snap → push apart → clamp.

    Returns a dict with overlap count and failed-macro count.
    """
    grid_snap(macros, grid_mm, bounds)
    residual = push_apart_overlapping(macros, bounds, max_passes=max_push_passes)
    failed = boundary_clamp(macros, bounds)

    # Verify cap-IC distances are preserved
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
