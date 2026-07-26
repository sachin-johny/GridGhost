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

from models.macro import Macro

if TYPE_CHECKING:
    from models.board_model import BoardModel, BoardOutline


def _snap_to_grid(value: float, grid_mm: float) -> float:
    return round(value / grid_mm) * grid_mm


def expand_bounds_to_fit(
    macros: list["Macro"],
    bounds: tuple[float, float, float, float],
    *,
    extra_padding: float = 1.0,
    target_density: float | None = None,
) -> tuple[float, float, float, float]:
    """Grow ``bounds`` just enough that macros can pack without overlap.

    Used on over-dense boards (e.g. test4: 62 macros in a 49×48mm
    interior at 65% density) where the SA + push-apart cannot fit all
    macros without overlap because the bounds themselves are too small.
    Without this, the legalizer rejects push-apart moves (every push
    sends something OOB), and 20+ residual overlaps remain.

    Algorithm (area-density based, scales uniformly):
      1. Compute total macro area.
      2. Compute target bounds area = total_area / target_density.
         55% density is a safe 2D packing target for irregular rectangles
         (literature: 50-70% is achievable for arbitrary rectangle
         bin-packing; we pick 55% to leave push-apart room).
      3. Scale the bounds uniformly (preserving aspect ratio) about its
         center until its area reaches the target. Aspect-ratio
         preservation matters: anisotropic scaling distorts the board
         layout — a 49×48mm board should grow to ~57×56mm, not 80×40mm.
      4. Add ``extra_padding`` (mm) on each side so push-apart has room.

    Cap: never expand beyond 2× the original bounds per side — if even
    that isn't enough, the legalizer will report residual overlaps and
    the user needs a bigger board.

    Returns the (possibly expanded) bounds. The macro positions are
    not modified — push_apart_overlapping + boundary_clamp will pack
    them into the new bounds.

    Finding 6 fix: ``target_density`` now defaults to the shared
    ``utils.density.target_pack_density()`` value (0.55) so it agrees
    with the outline-inference routine. Pass an explicit float to
    override.
    """
    if not macros:
        return bounds

    if target_density is None:
        from utils.density import target_pack_density
        target_density = target_pack_density()

    x_min, y_min, x_max, y_max = bounds
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    cur_w = x_max - x_min
    cur_h = y_max - y_min
    cur_area = cur_w * cur_h
    if cur_area <= 0:
        return bounds

    # Total macro area
    total_area = sum(
        max(0.0, m.bbox[2] - m.bbox[0]) * max(0.0, m.bbox[3] - m.bbox[1])
        for m in macros
    )

    # Current density
    cur_density = total_area / cur_area

    # If we're already below the target density, no expansion needed
    if cur_density <= target_density:
        return bounds

    # Scale uniformly so new area = total_area / target_density
    target_area = total_area / target_density
    scale = math.sqrt(target_area / cur_area)

    # Cap at 2x per side (4x area) — if macros need more than that,
    # report residual overlaps.
    scale = min(scale, 2.0)

    new_w = cur_w * scale + 2 * extra_padding
    new_h = cur_h * scale + 2 * extra_padding

    return (cx - new_w / 2, cy - new_h / 2, cx + new_w / 2, cy + new_h / 2)


def grid_snap(macros: list["Macro"], grid_mm: float, bounds: tuple[float, float, float, float]) -> None:
    """Snap each macro's leader to the grid; followers follow rigidly.

    Fixed macros (e.g. already-placed connectors) are NOT snapped —
    their positions are intentional and snapping could move them off
    their perimeter edge.
    """
    for m in macros:
        if m.is_fixed:
            continue
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

    Fixed macros (``is_fixed=True``) are NEVER chosen as the mover —
    they're treated as immovable obstacles. When a non-fixed macro
    overlaps a fixed one, only the non-fixed macro is pushed. When two
    fixed macros overlap (shouldn't happen — connectors are placed on
    the perimeter with spacing), the overlap is reported but not
    resolved.
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

                # Choose mover: prefer the smaller macro, but NEVER a
                # fixed macro. If both are fixed, neither can move —
                # the overlap is unresolvable by push-apart.
                if a.is_fixed and b.is_fixed:
                    continue  # both fixed — can't resolve here
                elif a.is_fixed:
                    mover, other = b, a
                elif b.is_fixed:
                    mover, other = a, b
                else:
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

                # Build attempts — only the non-fixed mover is a
                # candidate. If one macro is fixed, `other` is fixed
                # and we shouldn't try to push it; just try the mover
                # along both axes at multiple fractions.
                if a.is_fixed or b.is_fixed:
                    # One is fixed — only mover can be pushed.
                    attempts = [
                        (mover, cheap_axis, cheap_depth, 1.0),
                        (mover, steep_axis, steep_depth, 1.0),
                        (mover, cheap_axis, cheap_depth, 0.5),
                        (mover, steep_axis, steep_depth, 0.5),
                        (mover, cheap_axis, cheap_depth, 0.25),
                        (mover, steep_axis, steep_depth, 0.25),
                        (mover, cheap_axis, cheap_depth, 0.125),
                        (mover, steep_axis, steep_depth, 0.125),
                    ]
                else:
                    # Neither fixed — try both as mover.
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

    Fixed macros refuse to move — returns False immediately. This is a
    safety net: the caller (`push_apart_overlapping`) should already
    avoid passing a fixed macro as `mover`, but `_try_push` is also
    called from other contexts (e.g. the keepout post-loop) where the
    mover might be fixed.
    """
    if mover.is_fixed:
        return False
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


def force_spread_overlapping(
    macros: list["Macro"],
    bounds: tuple[float, float, float, float],
    max_passes: int = 50,
    step_size: float = 2.0,
) -> int:
    """Force-directed spread pass — escapes greedy push-apart local minima.

    The greedy `push_apart_overlapping` processes overlapping pairs one at
    a time and accepts a push only if it doesn't create a new overlap with
    the *immediate* neighbor. This gets stuck in local minima: a macro
    surrounded by 3-4 overlapping neighbors can't move in any direction
    without creating a new overlap, even when there's plenty of empty
    space elsewhere in the bounds.

    This pass takes a global view: for each macro, compute the NET
    repulsion vector from ALL overlapping neighbors (sum of unit vectors
    pointing away from each neighbor, weighted by overlap depth), then
    move the macro along that vector. The move is accepted if it reduces
    the macro's total overlap count — not requiring zero new overlaps,
    just net improvement. This escapes local minima by allowing temporary
    local increases in overlap if the net global overlap decreases.

    Fixed macros (``is_fixed=True``) are never moved — they're treated as
    immovable obstacles that contribute to the repulsion field but don't
    move themselves.

    Returns the residual overlap count after ``max_passes`` passes.
    """
    bounds_cx = (bounds[0] + bounds[2]) / 2
    bounds_cy = (bounds[1] + bounds[3]) / 2

    def _count_overlaps_for(m: "Macro") -> int:
        return sum(1 for o in macros if o is not m and m.overlaps(o))

    for _ in range(max_passes):
        any_moved = False
        # Snapshot current overlap counts per macro
        overlap_counts = {id(m): _count_overlaps_for(m) for m in macros}

        for m in macros:
            if m.is_fixed:
                continue
            if overlap_counts[id(m)] == 0:
                continue  # no overlaps — skip

            # Compute net repulsion from all overlapping neighbors
            mx1, my1, mx2, my2 = m.bbox
            mcx = (mx1 + mx2) / 2
            mcy = (my1 + my2) / 2
            fx, fy = 0.0, 0.0
            for o in macros:
                if o is m or not m.overlaps(o):
                    continue
                ox1, oy1, ox2, oy2 = o.bbox
                ocx = (ox1 + ox2) / 2
                ocy = (oy1 + oy2) / 2
                # Direction from other to me (repulsion pushes me away)
                dx = mcx - ocx
                dy = mcy - ocy
                dist = math.hypot(dx, dy)
                if dist < 1e-6:
                    # Macros exactly co-located — push in a random-ish
                    # direction (use ref hash for determinism)
                    dx, dy = 1.0 if hash(m.leader.ref) % 2 else -1.0, 1.0
                    dist = math.hypot(dx, dy)
                # Weight by overlap depth (deeper overlap = stronger push)
                ox_depth = min(mx2, ox2) - max(mx1, ox1)
                oy_depth = min(my2, oy2) - max(my1, oy1)
                weight = (ox_depth + oy_depth) / dist
                fx += (dx / dist) * weight
                fy += (dy / dist) * weight

            if abs(fx) < 1e-6 and abs(fy) < 1e-6:
                continue

            # Normalize the force vector and scale to step_size
            fmag = math.hypot(fx, fy)
            if fmag < 1e-6:
                continue
            dx = (fx / fmag) * step_size
            dy = (fy / fmag) * step_size

            # Add a tiny pull toward bounds center to prevent drift to corners
            pull_x = (bounds_cx - mcx) * 0.05
            pull_y = (bounds_cy - mcy) * 0.05
            dx += pull_x
            dy += pull_y

            # Try the move; accept if it reduces this macro's overlap count
            snap = m._snapshot()
            ok = m.translate(dx, dy, bounds=bounds)
            if not ok:
                # Try axis-only moves (one of dx/dy may be the OOB direction)
                if abs(dx) > 1e-6 and m.translate(dx, 0.0, bounds=bounds):
                    pass
                elif abs(dy) > 1e-6 and m.translate(0.0, dy, bounds=bounds):
                    pass
                else:
                    m._restore(snap)
                    continue

            new_count = _count_overlaps_for(m)
            if new_count < overlap_counts[id(m)]:
                # Net improvement — keep the move
                any_moved = True
                # Update overlap counts for the next iteration
                overlap_counts[id(m)] = new_count
            else:
                # No improvement — revert
                m._restore(snap)

        if not any_moved:
            break

    # Final residual count
    residual = 0
    for i in range(len(macros)):
        for j in range(i + 1, len(macros)):
            if macros[i].overlaps(macros[j]):
                residual += 1
    return residual


def displace_to_clear_slots(
    macros: list["Macro"],
    bounds: tuple[float, float, float, float],
    *,
    grid_mm: float = 1.0,
    max_candidate_slots: int = 400,
    verbose: bool = False,
    per_macro_keepout: "callable[[Macro], float] | None" = None,
) -> int:
    """Tetris-style global cleanup — jump overlapping macros to the nearest empty slot.

    For each non-fixed macro that overlaps at least one other macro,
    search a grid of candidate positions across ``bounds`` and jump to
    the nearest one that clears all overlaps. This is the global
    recovery the greedy push-apart + force-spread can't do — they only
    make LOCAL moves (push by overlap depth, spread by net repulsion
    vector). When a macro is jammed in a corner with 3-4 overlapping
    neighbors and no local move helps, the only escape is to JUMP to a
    different region of the board.

    Algorithm:
      1. Compute the set of "overlapping" macros (non-fixed, ≥1 overlap).
      2. For each, search a grid of candidate slots (centered on a
         ``grid_mm`` lattice across the bounds). For each slot, check
         whether placing the macro there would overlap any other macro
         AND whether the macro's bbox fits inside the keepout-shrunk
         bounds (so the subsequent boundary_clamp pass doesn't push it
         back into overlap). Pick the nearest valid slot.
      3. Apply the jump. If no valid slot exists in the candidate grid,
         leave the macro at its current position (it's truly stuck).

    The candidate grid is capped at ``max_candidate_slots`` positions
    to bound the worst-case O(slots × macros) cost. For a 100×100mm
    board at 1mm grid, that's 10,000 slots — we cap at 400 and sample
    sparsely (every ~5mm) to keep the cost reasonable.

    Fixed macros (``is_fixed=True``) are NEVER moved — they're treated
    as immovable obstacles. Macros that overlap ONLY fixed obstacles
    are also skipped (no clear slot would resolve the overlap — the
    fixed obstacle is in the way everywhere).

    If ``per_macro_keepout`` is provided, each macro's candidate slots
    are filtered to those where the macro's bbox fits inside the
    keepout-shrunk bounds (bounds inset by ``per_macro_keepout(m)``
    on each side). This prevents the Tetris cleanup from placing a
    macro at a slot that ``boundary_clamp`` would later push inward
    (creating new overlaps).

    Returns the residual overlap count after cleanup.
    """
    if not macros:
        return 0

    x_min, y_min, x_max, y_max = bounds
    bounds_w = x_max - x_min
    bounds_h = y_max - y_min
    if bounds_w <= 0 or bounds_h <= 0:
        return _count_residual_overlaps(macros)

    # Build a sparse candidate-slot grid. Aim for ~max_candidate_slots
    # evenly-spaced positions across the bounds. The slot step is
    # max(grid_mm, ceil(bounds_dim / sqrt(max_slots))).
    import math
    target_slots_per_axis = max(2, int(math.sqrt(max_candidate_slots)))
    step_x = max(grid_mm, bounds_w / target_slots_per_axis)
    step_y = max(grid_mm, bounds_h / target_slots_per_axis)

    # Snap step to grid_mm multiple for grid-consistent candidate positions.
    step_x = max(math.ceil(step_x / grid_mm) * grid_mm, grid_mm)
    step_y = max(math.ceil(step_y / grid_mm) * grid_mm, grid_mm)

    candidate_x = []
    x = x_min + step_x / 2
    while x < x_max - step_x / 2 + 1e-9:
        candidate_x.append(x)
        x += step_x
    candidate_y = []
    y = y_min + step_y / 2
    while y < y_max - step_y / 2 + 1e-9:
        candidate_y.append(y)
        y += step_y

    if not candidate_x or not candidate_y:
        return _count_residual_overlaps(macros)

    def _overlaps_at(m: "Macro", new_x: float, new_y: float,
                     others: list["Macro"], keepout: float = 0.0) -> bool:
        """Would macro ``m`` overlap any other macro if its leader were at (new_x, new_y)?

        Also checks that the macro's bbox fits inside the keepout-shrunk
        bounds (bounds inset by ``keepout`` on each side). If it doesn't
        fit, returns True (treats it as an "overlap" so the slot is
        rejected).

        Computes the macro's bbox at the proposed new leader position
        (preserving current rotation) WITHOUT modifying the macro, then
        checks pairwise overlap with every other macro.
        """
        snap = m._snapshot()
        dx = new_x - m.leader.x
        dy = new_y - m.leader.y
        m.leader.x += dx
        m.leader.y += dy
        m.apply_offsets()
        try:
            # Keepout-shrunk bounds check first — if the macro's bbox
            # doesn't fit, reject this slot (subsequent boundary_clamp
            # would push it inward, creating new overlaps).
            if keepout > 0:
                kx_min = x_min + keepout
                ky_min = y_min + keepout
                kx_max = x_max - keepout
                ky_max = y_max - keepout
                for c in m.members:
                    cx1, cy1, cx2, cy2 = c.bbox
                    if (cx1 < kx_min - 1e-6 or cy1 < ky_min - 1e-6 or
                        cx2 > kx_max + 1e-6 or cy2 > ky_max + 1e-6):
                        return True  # treat as overlap → reject slot
            for o in others:
                if o is m:
                    continue
                if m.overlaps(o):
                    return True
            return False
        finally:
            m._restore(snap)

    # Identify overlapping macros (non-fixed, with at least one non-fixed
    # overlapping neighbor). Macros that overlap ONLY fixed obstacles
    # can't be helped by jumping — the fixed obstacle is in the way.
    def _has_movable_overlap(m: "Macro") -> bool:
        for o in macros:
            if o is m:
                continue
            if not m.overlaps(o):
                continue
            if not o.is_fixed:
                return True
        return False

    overlapping_macros = [m for m in macros
                          if not m.is_fixed and _has_movable_overlap(m)]

    if not overlapping_macros:
        return _count_residual_overlaps(macros)

    # Sort by overlap count (most-overlapping first — biggest win per jump).
    def _overlap_count(m: "Macro") -> int:
        return sum(1 for o in macros if o is not m and m.overlaps(o))
    overlapping_macros.sort(key=lambda m: -_overlap_count(m))

    if verbose:
        print(f"  Tetris: {len(overlapping_macros)} macros with overlaps, "
              f"searching {len(candidate_x) * len(candidate_y)} candidate slots")

    jumps_made = 0
    for m in overlapping_macros:
        if _overlap_count(m) == 0:
            continue  # already resolved by a previous jump

        cur_x, cur_y = m.leader.x, m.leader.y
        # Per-macro keepout — slots where the macro's bbox pokes into
        # the keepout zone are rejected (would cause boundary_clamp to
        # push it back into overlap).
        ke = per_macro_keepout(m) if per_macro_keepout else 0.0

        # Search candidate slots in order of increasing distance from
        # current position. Bail out at the first clear slot.
        slots = []
        for sx in candidate_x:
            for sy in candidate_y:
                d2 = (sx - cur_x) ** 2 + (sy - cur_y) ** 2
                slots.append((d2, sx, sy))
        slots.sort(key=lambda t: t[0])

        # Limit how many slots we evaluate per macro — the top-K nearest
        # are usually enough; if none of them is clear, the macro is
        # genuinely stuck (board too dense).
        top_k = min(len(slots), 80)
        for d2, sx, sy in slots[:top_k]:
            if not _overlaps_at(m, sx, sy, macros, keepout=ke):
                # Jump to this slot. Use translate (which respects bounds)
                # so we don't accidentally place the macro OOB.
                dx = sx - cur_x
                dy = sy - cur_y
                if m.translate(dx, dy, bounds=bounds):
                    jumps_made += 1
                    break

    if verbose and jumps_made:
        print(f"  Tetris: jumped {jumps_made} macros to clear slots")

    return _count_residual_overlaps(macros)


def _count_residual_overlaps(macros: list["Macro"]) -> int:
    """Count pairwise macro overlaps (helper)."""
    n = 0
    for i in range(len(macros)):
        for j in range(i + 1, len(macros)):
            if macros[i].overlaps(macros[j]):
                n += 1
    return n


def boundary_clamp(
    macros: list["Macro"],
    bounds: tuple[float, float, float, float],
    per_macro_keepout: "callable[[Macro], float] | None" = None,
) -> int:
    """Clamp each macro inside bounds. Returns count of macros that couldn't fit.

    A macro that can't fit at any position (e.g., macro bbox larger
    than bounds) is left at its current pose; the caller must handle.

    If ``per_macro_keepout`` is provided, it is called with each macro
    and must return an extra keepout (mm) that shrinks the bounds for
    THAT macro only. Used to keep ICs/MCUs further from the board edge
    than passives (DFM rule): the macro pipeline passes a callback that
    returns 5mm for ICs and 0mm for resistors/caps, so an IC's bbox is
    clamped to (bounds + 5mm) while a resistor's bbox is clamped to
    (bounds + 0mm).

    Fixed macros (``is_fixed=True``) are skipped — their positions are
    intentional (e.g. connectors placed on the perimeter with overhang)
    and clamping them would corrupt the placement.
    """
    x_min, y_min, x_max, y_max = bounds
    failed = 0
    for m in macros:
        if m.is_fixed:
            continue  # connectors stay where place_connectors_perimeter put them
        # Per-macro extra keepout (ICs/MCUs get pushed further from edge)
        ke = per_macro_keepout(m) if per_macro_keepout else 0.0
        bx_min = x_min + ke
        by_min = y_min + ke
        bx_max = x_max - ke
        by_max = y_max - ke

        bx1, by1, bx2, by2 = m.bbox
        dx_left = bx_min - bx1
        dx_right = bx_max - bx2
        dy_top = by_min - by1
        dy_bottom = by_max - by2

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

        # Use the per-macro shrunk bounds for the translate() validity
        # check so we don't accept a move that leaves the macro inside
        # the global bounds but outside the keepout-shrunk bounds.
        macro_bounds = (bx_min, by_min, bx_max, by_max) if ke > 0 else bounds
        if not m.translate(dx, dy, bounds=macro_bounds):
            # Try only x or only y
            if abs(dx) > 1e-9 and m.translate(dx, 0.0, bounds=macro_bounds):
                continue
            if abs(dy) > 1e-9 and m.translate(0.0, dy, bounds=macro_bounds):
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
    expand_to_fit: bool = True,
    extra_padding: float = 1.0,
    target_density: float | None = None,
    per_macro_keepout: "callable[[Macro], float] | None" = None,
) -> dict[str, int]:
    """Iterative legalization: snap → (push apart → clamp) repeated.

    Each round repairs the overlaps created by the previous clamp, then
    clamps the resulting positions back inside bounds. Stops when a
    round makes no progress or after ``max_rounds`` iterations.

    If ``expand_to_fit`` is True (default) and the macro packing density
    in the bounds exceeds the shared ``target_pack_density`` (default
    0.55; override via ``target_density`` arg or config.json's
    ``placement.target_pack_density``), the bounds are uniformly scaled
    up about their center so density drops to the target. This prevents
    the OOB-rejection deadlock on over-dense boards (test4: 62 macros in
    a 49×48mm interior at 65% density). The expanded bounds are reported
    in the returned dict.

    If ``per_macro_keepout`` is provided, it is called with each macro
    and must return an extra edge keepout (mm) for that macro. Used to
    push ICs/MCUs further from the board edge than passives — DFM rule
    a human designer always applies.

    Returns a dict with overlap count, boundary-failure count, cap-IC
    distance violation count, and the (possibly expanded) bounds.
    """
    if target_density is None:
        from utils.density import target_pack_density
        target_density = target_pack_density()
    if expand_to_fit:
        expanded = expand_bounds_to_fit(
            macros, bounds,
            extra_padding=extra_padding,
            target_density=target_density,
        )
        if expanded != bounds:
            if verbose:
                old_w = bounds[2] - bounds[0]
                old_h = bounds[3] - bounds[1]
                new_w = expanded[2] - expanded[0]
                new_h = expanded[3] - expanded[1]
                print(
                    f"  Legalize: bounds expanded {old_w:.1f}×{old_h:.1f} -> "
                    f"{new_w:.1f}×{new_h:.1f}mm to fit {len(macros)} macros "
                    f"(density target {target_density:.0%})"
                )
            bounds = expanded

    grid_snap(macros, grid_mm, bounds)

    residual = push_apart_overlapping(macros, bounds, max_passes=max_push_passes)
    failed = boundary_clamp(macros, bounds, per_macro_keepout=per_macro_keepout)

    # Iterate: clamp creates overlaps, push-apart fixes them but may push
    # something back OOB, clamp fixes that, etc. Continue until stable.
    for round_idx in range(max_rounds):
        new_residual = push_apart_overlapping(macros, bounds, max_passes=max_push_passes)
        new_failed = boundary_clamp(macros, bounds, per_macro_keepout=per_macro_keepout)
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

    # ─── Force-directed spread pass (root-cause fix for greedy local minima) ──
    # The greedy push-apart above gets stuck in local minima: a macro
    # surrounded by 3-4 overlapping neighbors can't move in any direction
    # without creating a new overlap, even when there's plenty of empty
    # space elsewhere in the bounds. This is why test6 (39% density) and
    # th_sensor (38% density) had 16+ residual overlaps despite having
    # plenty of room.
    #
    # The force-directed spread pass takes a global view: for each
    # overlapping macro, compute the NET repulsion vector from ALL
    # overlapping neighbors and move along it. The move is accepted if
    # it reduces the macro's overlap count — not requiring zero new
    # overlaps, just net improvement. This escapes local minima.
    #
    # Gate: only run if (a) there are residual overlaps AND (b) the
    # density is below the target (room exists). On over-dense boards
    # (density > target) the bounds expansion above already ran and the
    # force-spread won't help — the issue is genuine lack of room, not
    # a local minimum.
    if residual > 0:
        bounds_area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
        if bounds_area > 0:
            total_macro_area = sum(
                max(0.0, m.bbox[2] - m.bbox[0]) * max(0.0, m.bbox[3] - m.bbox[1])
                for m in macros if not m.is_fixed
            )
            current_density = total_macro_area / bounds_area
            if current_density < target_density:
                if verbose:
                    print(
                        f"  Force-spread: {residual} residual overlaps at "
                        f"{current_density:.0%} density (room available) — "
                        f"running force-directed spread"
                    )
                spread_residual = force_spread_overlapping(
                    macros, bounds, max_passes=50, step_size=2.0,
                )
                if verbose and spread_residual < residual:
                    print(
                        f"  Force-spread: {residual} -> {spread_residual} "
                        f"overlaps (−{residual - spread_residual})"
                    )
                residual = spread_residual
                # Re-clamp after force-spread (it may have pushed macros OOB)
                failed = boundary_clamp(macros, bounds, per_macro_keepout=per_macro_keepout)
                # One more push-apart to clean up any overlaps the
                # boundary_clamp re-introduced.
                residual = push_apart_overlapping(macros, bounds, max_passes=max_push_passes)

    # Final per-macro keepout enforcement: push ICs/MCUs that are still
    # inside the keepout zone (but inside the global bounds) further
    # toward the center. The main loop's boundary_clamp only catches
    # macros that poke PAST the keepout boundary; it doesn't pull
    # compliant-but-close macros further in. This pass iteratively
    # nudges keepout-bounded macros toward the center until they clear
    # the keepout zone OR can't move without creating new overlaps.
    if per_macro_keepout:
        bx_min, by_min, bx_max, by_max = bounds  # bounds may have been expanded
        for _ in range(20):  # bounded iterations
            moved_any = False
            for m in macros:
                # Fixed macros (connectors) are never pushed by keepout.
                # They sit on the perimeter by design.
                if m.is_fixed:
                    continue
                ke = per_macro_keepout(m)
                if ke <= 0:
                    continue
                bx1, by1, bx2, by2 = m.bbox
                # Clearance to each keepout-shrunk boundary.
                # Positive = macro is OUTSIDE the keepout zone (good).
                # Negative = macro is INSIDE the keepout zone (needs push).
                d_left = (bx1 - bx_min) - ke       # clearance from left keepout boundary
                d_right = (bx_max - bx2) - ke      # clearance from right keepout boundary
                d_top = (by1 - by_min) - ke        # clearance from top keepout boundary
                d_bottom = (by_max - by2) - ke     # clearance from bottom keepout boundary

                # Collect violated directions with the push direction (toward center)
                # If d_left < 0: macro too close to LEFT edge → push RIGHT (+x)
                # If d_right < 0: macro too close to RIGHT edge → push LEFT (-x)
                # If d_top < 0: macro too close to TOP edge → push DOWN (+y)
                # If d_bottom < 0: macro too close to BOTTOM edge → push UP (-y)
                violated = []
                if d_left < -0.01:
                    violated.append(("x", +1.0, d_left))
                if d_right < -0.01:
                    violated.append(("x", -1.0, d_right))
                if d_top < -0.01:
                    violated.append(("y", +1.0, d_top))
                if d_bottom < -0.01:
                    violated.append(("y", -1.0, d_bottom))

                if not violated:
                    continue
                # Push toward the most-violated direction first
                violated.sort(key=lambda t: t[2])  # most negative first
                ax, sign, d = violated[0]
                push = -d + 0.5  # push past the boundary + small margin
                # Try the push, then check it doesn't create overlaps
                snap = m._snapshot()
                if ax == "x":
                    ok = m.translate(sign * push, 0.0, bounds=bounds)
                else:
                    ok = m.translate(0.0, sign * push, bounds=bounds)
                if ok:
                    # Check no new overlaps created
                    new_overlap = any(m.overlaps(o) for o in macros
                                      if o is not m)
                    if new_overlap:
                        m._restore(snap)
                    else:
                        moved_any = True
                else:
                    m._restore(snap)
            if not moved_any:
                break
        # Recount residual overlaps AND boundary failures after keepout
        # enforcement. The post-loop above can push macros OOB (a push
        # that lands inside the global bounds but outside the keepout-
        # shrunk bounds for THAT macro), and it can leave previously-
        # clamped macros in a violated state if a push was reverted.
        # Without this recount, `failed` retains its pre-loop value and
        # the legalizer silently underreports boundary violations.
        residual = 0
        for i in range(len(macros)):
            for j in range(i + 1, len(macros)):
                if macros[i].overlaps(macros[j]):
                    residual += 1
        # Recompute failed: any non-fixed macro whose keepout-shrunk
        # bounds are violated (or whose bbox pokes outside the global
        # bounds). Fixed macros (connectors) are intentionally on the
        # perimeter and may overhang — they're not counted as failures.
        failed = 0
        bx_min, by_min, bx_max, by_max = bounds
        for m in macros:
            if m.is_fixed:
                continue
            ke = per_macro_keepout(m) if per_macro_keepout else 0.0
            kx_min, ky_min = bx_min + ke, by_min + ke
            kx_max, ky_max = bx_max - ke, by_max - ke
            bx1, by1, bx2, by2 = m.bbox
            if (bx1 < kx_min - 1e-6 or bx2 > kx_max + 1e-6 or
                by1 < ky_min - 1e-6 or by2 > ky_max + 1e-6):
                failed += 1

    # ─── Tetris-style "displace to nearest empty slot" cleanup (Finding 3 fix) ──
    # The greedy push-apart + force-spread above are LOCAL heuristics.
    # They get stuck in local minima: a macro surrounded by overlapping
    # neighbors can't move in any direction without creating a new
    # overlap, even when there's plenty of empty space elsewhere in the
    # bounds. The user's evaluation report called this out:
    #
    #   "Greedy pairwise push-apart legalizer has no global recovery for
    #    chained overlaps... even after my fix to #2, test4 still has 7
    #    unresolved overlaps (all small, ≤3.6 mm², mostly test points
    #    and passives)."
    #
    # This pass takes a GLOBAL view: for each overlapping macro, search
    # a grid of candidate slots across the bounds, find the nearest one
    # that's clear of all other macros, and jump there. This is the
    # "Tetris" cleanup the user's plan mentioned — not a full row-based
    # legalizer like abacus, but a global escape from local minima that
    # the greedy passes can't reach.
    #
    # Only runs if there are still residual overlaps after force-spread
    # and the keepout enforcement pass. Skips macros whose overlaps are
    # exclusively with fixed obstacles (no clear slot would help).
    if residual > 0:
        new_residual = displace_to_clear_slots(
            macros, bounds, grid_mm=grid_mm, verbose=verbose,
            per_macro_keepout=per_macro_keepout,
        )
        if new_residual < residual:
            if verbose:
                print(
                    f"  Tetris cleanup: {residual} -> {new_residual} "
                    f"overlaps (−{residual - new_residual})"
                )
            residual = new_residual
        # Re-clamp to bounds (Tetris jumps stay in-bounds by construction,
        # but the slot search uses a sparse grid — a macro might end up
        # at a position where its bbox pokes slightly past bounds).
        # DO NOT re-run push_apart here — it would undo the Tetris jumps
        # (push_apart is a LOCAL heuristic that can push a carefully-
        # placed macro back into an overlap the Tetris cleanup just
        # resolved). Just clamp and recount.
        failed = boundary_clamp(macros, bounds, per_macro_keepout=per_macro_keepout)
        # CRITICAL: boundary_clamp can create NEW overlaps (it pushes
        # macros inside bounds, which might overlap others). Recount
        # the residual so the legalizer's self-report matches the
        # actual state. Without this recount, the legalizer silently
        # underreports overlaps — exactly the kind of discrepancy
        # the user's evaluation flagged as "Finding 1/2 effect".
        residual = _count_residual_overlaps(macros)

    # Cap-IC invariant: a follower must never overlap its own leader.
    # Macros are rigid bodies, so this geometry is fixed at construction
    # (find_cap_offset places every cap gap-spaced off the leader body)
    # and cannot be disturbed by SA or this legalizer — they move the
    # whole macro as one unit. This count is therefore an assertion that
    # construction was correct, not a distance check. (An earlier version
    # counted center-to-center distances > 8mm as "violations", which
    # fired on every cap of any IC larger than ~9mm — a false positive
    # that obscured the real bug; see models/macro.py.)
    cap_ic_overlaps = 0
    for m in macros:
        if not m.followers:
            continue
        for f in m.followers:
            if f.overlaps(m.leader):
                cap_ic_overlaps += 1

    if verbose:
        print(
            f"  Legalize: {residual} residual overlaps, "
            f"{failed} boundary failures, {cap_ic_overlaps} cap-IC overlaps"
        )

    return {
        "residual_overlaps": residual,
        "boundary_failures": failed,
        "cap_ic_overlaps": cap_ic_overlaps,
        "expanded_bounds": bounds,
    }
