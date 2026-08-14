"""Multi-pass greedy legalizer.

NOTE: earlier revisions of this module described this as a "single
pass" legalizer with "no spread pass ... no brute-force safety net".
That was true of the first version and is no longer true — real
boards broke it, and passes were added incrementally to chase down
residual overlaps. Documenting the actual pipeline honestly:

1. ``expand_bounds_to_fit`` — if macro packing density exceeds the
   target, grow the placement bounds so there's room to legalize at
   all (see the ``legalize()`` docstring).
2. ``grid_snap`` — snap macro leaders to ``grid_mm`` (followers stay
   at their fixed offsets relative to the leader).
3. ``push_apart_overlapping`` — greedy pairwise macro-macro overlap
   repair, iterated with ``boundary_clamp`` for up to ``max_rounds``
   rounds (clamp can reintroduce overlaps push-apart just fixed, and
   vice versa).
4. ``force_spread_overlapping`` — a global net-repulsion pass, run
   only when push-apart plateaus with room still available in the
   bounds (push-apart is a local heuristic and gets stuck when a
   macro is boxed in by 3-4 neighbors even with free space elsewhere).
5. Per-macro keepout enforcement — nudges ICs/MCUs that are inside
   their extra edge keepout zone (but inside global bounds) further
   toward center.
6. ``displace_to_clear_slots`` ("Tetris" pass) — a global candidate-
   slot search for macros still overlapping after all of the above.

This legalizer is a stack of greedy heuristics, not a legalizer with
a correctness guarantee (unlike, e.g., a row-based DP legalizer such
as Abacus). It CAN and DOES leave residual overlaps on dense/irregular
boards — ``legalize()`` reports ``residual_overlaps`` /
``boundary_failures`` explicitly for this reason, and callers should
treat a non-zero count as a placement that needs manual cleanup, not
a soft warning to ignore. See ``gridghost.py`` for how the CLI now
surfaces this as a hard warning + non-zero exit code.

The <8mm cap-IC distance (edge-to-edge gap; see ``models/macro.py``)
IS guaranteed regardless of legalizer outcome, because it's enforced
at Macro construction time and macros only ever move as rigid bodies
— no pass in this file can break that invariant, only the ICs'/caps'
position relative to *other* macros.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from models.macro import Macro

if TYPE_CHECKING:
    from models.board_model import BoardModel, BoardOutline


from place.abacus_bridge import abacus_legalize_macros
from place.sa_polish import sa_polish_legalize


def _snap_to_grid(value: float, grid_mm: float) -> float:
    return round(value / grid_mm) * grid_mm


class _MacroSpatialGrid:
    """Uniform-grid spatial index over Macro bboxes.

    Adapted from ``legalization/spatial_grid.py:SpatialGrid`` (which
    indexes Components) to operate on Macro union bboxes. Used by the
    4 O(N²) hot loops in this module — ``push_apart_overlapping``,
    ``_count_residual_overlaps``, ``displace_to_clear_slots``, and
    ``force_spread_overlapping`` — to skip the inner-loop scan over
    every macro when looking for overlaps. For a 70-macro board the
    win is small; for 200+ macros it's the difference between O(N²)
    and ~O(N) per pass.

    The grid is a CHEAP pre-filter: it returns CANDIDATE macro indices
    whose AABB shares a cell with the query macro. The actual overlap
    test (which honours the mounting-hole / mechanical exemption via
    ``Macro.overlaps``) still runs on each candidate — the grid never
    changes overlap semantics, only the set of pairs we test.

    See IMPROVEMENTS §2.3.
    """

    __slots__ = ("_x_min", "_y_min", "cell_size", "cells")

    def __init__(self, bounds: tuple[float, float, float, float], cell_size: float):
        self._x_min = bounds[0]
        self._y_min = bounds[1]
        self.cell_size = cell_size
        self.cells: dict[tuple[int, int], list[int]] = {}

    @classmethod
    def from_macros(
        cls,
        macros: list["Macro"],
        bounds: tuple[float, float, float, float],
    ) -> "_MacroSpatialGrid":
        """Build a grid sized to the largest macro bbox dimension.

        Cell size = max(max_macro_dim * 1.5, 1.0) so each macro overlaps
        at most ~4 neighbour cells (1 + halo) — keeps candidate lists
        short without making the grid so fine that empty cells dominate.
        """
        max_dim = 0.0
        for m in macros:
            bx1, by1, bx2, by2 = m.bbox
            max_dim = max(max_dim, bx2 - bx1, by2 - by1)
        cell_size = max(max_dim * 1.5, 1.0)
        grid = cls(bounds, cell_size)
        grid.build(macros)
        return grid

    def _cells_for_bbox(
        self, bbox: tuple[float, float, float, float]
    ) -> list[tuple[int, int]]:
        x1, y1, x2, y2 = bbox
        # Clamp lower bounds to 0 — a macro slightly OOB on the negative
        # side still needs to be indexed (we don't want to lose overlaps
        # with macros at the corner of bounds).
        col_min = max(int((x1 - self._x_min) / self.cell_size), 0)
        row_min = max(int((y1 - self._y_min) / self.cell_size), 0)
        col_max = int((x2 - self._x_min) / self.cell_size)
        row_max = int((y2 - self._y_min) / self.cell_size)
        if col_max < col_min:
            col_max = col_min
        if row_max < row_min:
            row_max = row_min
        return [
            (c, r)
            for c in range(col_min, col_max + 1)
            for r in range(row_min, row_max + 1)
        ]

    def build(self, macros: list["Macro"]) -> None:
        self.cells.clear()
        for idx in range(len(macros)):
            for cell in self._cells_for_bbox(macros[idx].bbox):
                if cell not in self.cells:
                    self.cells[cell] = []
                self.cells[cell].append(idx)

    def query_candidates(self, idx: int, macros: list["Macro"]) -> list[int]:
        """Return macro indices whose bbox MIGHT overlap ``macros[idx]``.

        Always a superset of the true overlap set — caller must run the
        real ``Macro.overlaps`` test on each candidate.
        """
        m = macros[idx]
        result: set[int] = set()
        for cell in self._cells_for_bbox(m.bbox):
            if cell in self.cells:
                result.update(self.cells[cell])
        result.discard(idx)
        return list(result)

    def query_candidates_for_bbox(
        self, bbox: tuple[float, float, float, float],
        exclude_idx: int | None = None,
    ) -> list[int]:
        """Variant for ``displace_to_clear_slots`` which queries against
        a hypothetical bbox (a candidate slot) rather than an existing
        macro's current bbox.
        """
        result: set[int] = set()
        for cell in self._cells_for_bbox(bbox):
            if cell in self.cells:
                result.update(self.cells[cell])
        if exclude_idx is not None:
            result.discard(exclude_idx)
        return list(result)


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

    Uses ``_MacroSpatialGrid`` to skip the inner-loop scan over every
    macro when looking for overlaps. The grid is rebuilt per pass
    (macros move between passes). See IMPROVEMENTS §2.3.
    """
    for _ in range(max_passes):
        any_overlap = False
        any_resolved = False

        # Rebuild the spatial grid once per pass — macros moved last
        # pass, so bboxes have changed. Building per-pass (not per-
        # macro) keeps the O(N) build cost amortised across the inner
        # loop.
        grid = _MacroSpatialGrid.from_macros(macros, bounds)
        seen_pairs: set[tuple[int, int]] = set()

        for i in range(len(macros)):
            a = macros[i]
            for j in grid.query_candidates(i, macros):
                # Deduplicate pairs — the grid returns each pair twice
                # (once from i, once from j).
                pair = (i, j) if i < j else (j, i)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
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

    return _count_residual_overlaps(macros)


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

    Uses ``_MacroSpatialGrid`` to skip the inner-loop scan over every
    macro when computing each macro's overlap count and repulsion
    vector. Grid is rebuilt per pass. See IMPROVEMENTS §2.3.
    """
    bounds_cx = (bounds[0] + bounds[2]) / 2
    bounds_cy = (bounds[1] + bounds[3]) / 2

    def _count_overlaps_for(m: "Macro", grid: _MacroSpatialGrid,
                              m_idx: int) -> int:
        return sum(1 for j in grid.query_candidates(m_idx, macros)
                    if macros[j] is not m and m.overlaps(macros[j]))

    for _ in range(max_passes):
        any_moved = False
        # Rebuild grid per pass — macros moved last pass.
        grid = _MacroSpatialGrid.from_macros(macros, bounds)
        # Snapshot current overlap counts per macro (by index, not id —
        # id() works but index is faster and stable within a pass).
        overlap_counts = {
            idx: _count_overlaps_for(m, grid, idx)
            for idx, m in enumerate(macros)
        }

        for m_idx, m in enumerate(macros):
            if m.is_fixed:
                continue
            if overlap_counts[m_idx] == 0:
                continue  # no overlaps — skip

            # Compute net repulsion from all overlapping neighbors
            mx1, my1, mx2, my2 = m.bbox
            mcx = (mx1 + mx2) / 2
            mcy = (my1 + my2) / 2
            fx, fy = 0.0, 0.0
            for j in grid.query_candidates(m_idx, macros):
                o = macros[j]
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
                    # Macros exactly co-located — push in a deterministic
                    # direction based on the leader refs (avoids PYTHONHASHSEED
                    # dependence of built-in hash() on strings).
                    pair_key = f"{m.leader.ref}|{o.leader.ref}"
                    dx, dy = 1.0 if sum(ord(c) for c in pair_key) % 2 else -1.0, 1.0
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

            new_count = _count_overlaps_for(m, grid, m_idx)
            if new_count < overlap_counts[m_idx]:
                # Net improvement — keep the move
                any_moved = True
                # Update overlap counts for the next iteration
                overlap_counts[m_idx] = new_count
            else:
                # No improvement — revert
                m._restore(snap)

        if not any_moved:
            break

    return _count_residual_overlaps(macros)


def displace_to_clear_slots(
    macros: list["Macro"],
    bounds: tuple[float, float, float, float],
    *,
    grid_mm: float = 1.0,
    max_candidate_slots: int = 400,
    verbose: bool = False,
    per_macro_keepout: "callable[[Macro], float] | None" = None,
    board: "BoardOutline | None" = None,
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
    is_poly = board is not None and board.is_polygon

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

    # Build the spatial grid ONCE for the whole Tetris pass. Macros move
    # during jumps, but the candidate-slot search tests against every
    # other macro's CURRENT position — rebuilding per slot would be more
    # expensive than the scan it saves. After each accepted jump we
    # rebuild so subsequent slot searches see the new layout. See
    # IMPROVEMENTS §2.3.
    grid = _MacroSpatialGrid.from_macros(macros, bounds)
    # Map macro object → index for grid queries.
    macro_idx = {id(m): i for i, m in enumerate(macros)}

    def _overlaps_at(m: "Macro", new_x: float, new_y: float,
                     others: list["Macro"], keepout: float = 0.0,
                     grid_ref: _MacroSpatialGrid | None = None) -> bool:
        """Would macro ``m`` overlap any other macro if its leader were at (new_x, new_y)?

        Also checks that the macro's bbox fits inside the keepout-shrunk
        bounds (bounds inset by ``keepout`` on each side). If it doesn't
        fit, returns True (treats it as an "overlap" so the slot is
        rejected).

        Computes the macro's bbox at the proposed new leader position
        (preserving current rotation) WITHOUT modifying the macro, then
        checks pairwise overlap with every other macro. When
        ``grid_ref`` is provided, only the macros whose current bbox
        shares a cell with ``m``'s hypothetical new bbox are tested —
        the rest can't possibly overlap. The grid is built from CURRENT
        positions, so it's only valid for queries against the current
        layout (the caller rebuilds it after each accepted jump).
        """
        snap = m._snapshot()
        dx = new_x - m.leader.x
        dy = new_y - m.leader.y
        m.leader.x += dx
        m.leader.y += dy
        m.apply_offsets()
        try:
            # "Off the board" check: reject the slot if the macro would be
            # physically off the board. For a rectangular outline that is
            # the per-member AABB check below; for a polygon outline
            # (notches/cutouts/holes) the macro's union bbox must be fully
            # inside the TRUE outline — a slot inside the AABB but in a
            # notch is just as invalid as one past the outer edge. Keepout
            # violations (in-bounds but within the keepout zone) are NOT
            # rejected here: a keepout violation is a soft DFM concern, not
            # a placement invalidity — better to place the macro in-bounds
            # with a keepout violation than to leave it OOB (which IS a hard
            # invalidity). (See _fit_macro_to_polygon / boundary_clamp for
            # where keepout is enforced as a margin.) The previous version
            # rejected keepout-violating slots, which caused Tetris to fail
            # on dense boards where no keepout-clear slot existed.
            if is_poly:
                if not board.contains_bbox(m.bbox):
                    return True  # off the true board (notch/hole/edge) → reject
            else:
                for c in m.members:
                    cx1, cy1, cx2, cy2 = c.bbox
                    if (cx1 < x_min - 1e-6 or cy1 < y_min - 1e-6 or
                        cx2 > x_max + 1e-6 or cy2 > y_max + 1e-6):
                        return True  # past global bounds → reject slot
            if grid_ref is not None:
                # Grid-accelerated: only test macros whose current bbox
                # shares a cell with m's hypothetical new bbox. Use the
                # exclude_idx variant so m doesn't get tested against
                # itself (its own grid entry is for its CURRENT bbox,
                # which doesn't apply to the hypothetical position).
                m_index = macro_idx.get(id(m))
                for j in grid_ref.query_candidates_for_bbox(m.bbox, exclude_idx=m_index):
                    o = others[j]
                    if o is m:
                        continue
                    if m.overlaps(o):
                        return True
            else:
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
    # Grid-accelerated: query each macro's candidate set instead of
    # scanning every other macro.
    def _has_movable_overlap(m: "Macro", m_idx: int,
                               grid_ref: _MacroSpatialGrid) -> bool:
        for j in grid_ref.query_candidates(m_idx, macros):
            o = macros[j]
            if o is m:
                continue
            if not m.overlaps(o):
                continue
            if not o.is_fixed:
                return True
        return False

    # Also identify macros that are out-of-bounds (OOB). The overlap-aware
    # boundary clamp (in sa_polish.py) refuses to push a macro into an
    # overlap, leaving it OOB instead. Tetris is the right tool to jump
    # it to a clear in-bounds slot — without this, OOB macros stay OOB
    # forever (push_apart can't fix OOB, only overlaps).
    def _is_oob(m: "Macro") -> bool:
        if m.is_fixed:
            return False
        if is_poly:
            # OOB on a polygon board = not fully inside the outline (in a
            # notch/cutout/hole, or past the outer edge). contains_bbox is
            # the polygon authority here, not the AABB.
            return not board.contains_bbox(m.bbox)
        bx1, by1, bx2, by2 = m.bbox
        return (bx1 < x_min - 1e-6 or by1 < y_min - 1e-6 or
                bx2 > x_max + 1e-6 or by2 > y_max + 1e-6)

    overlapping_macros: list["Macro"] = []
    for idx, m in enumerate(macros):
        if not m.is_fixed and _has_movable_overlap(m, idx, grid):
            overlapping_macros.append(m)
    oob_macros = [m for m in macros if _is_oob(m)]
    # Deduplicate: a macro can be both overlapping and OOB.
    seen = set(id(m) for m in overlapping_macros)
    for m in oob_macros:
        if id(m) not in seen:
            overlapping_macros.append(m)
            seen.add(id(m))

    if not overlapping_macros:
        return _count_residual_overlaps(macros)

    # Sort by overlap count (most-overlapping first — biggest win per jump).
    # Grid-accelerated: same candidate-set query as _has_movable_overlap.
    def _overlap_count(m: "Macro", m_idx: int,
                        grid_ref: _MacroSpatialGrid) -> int:
        return sum(1 for j in grid_ref.query_candidates(m_idx, macros)
                    if macros[j] is not m and m.overlaps(macros[j]))
    overlapping_macros.sort(
        key=lambda m: -_overlap_count(m, macro_idx[id(m)], grid)
    )

    if verbose:
        print(f"  Tetris: {len(overlapping_macros)} macros with overlaps, "
              f"searching {len(candidate_x) * len(candidate_y)} candidate slots")

    jumps_made = 0
    for m in overlapping_macros:
        # Skip macros that are already in-bounds AND have no overlaps
        # (a macro can be in this list because it was OOB earlier but
        # a previous jump already fixed it, or because it was overlapping
        # but a neighbor's jump resolved the overlap).
        if _overlap_count(m, macro_idx[id(m)], grid) == 0 and not _is_oob(m):
            continue

        cur_x, cur_y = m.leader.x, m.leader.y
        # Per-macro keepout — no longer used to reject slots (see the
        # _overlaps_at docstring above), but kept for API compatibility.
        ke = per_macro_keepout(m) if per_macro_keepout else 0.0

        # Search candidate slots in order of increasing distance from
        # current position. Bail out at the first clear slot.
        slots = []
        for sx in candidate_x:
            for sy in candidate_y:
                d2 = (sx - cur_x) ** 2 + (sy - cur_y) ** 2
                slots.append((d2, sx, sy))
        slots.sort(key=lambda t: t[0])

        # Evaluate ALL candidate slots (not just top-K nearest). The
        # previous top_k=80 limit caused Tetris to give up on macros
        # whose 80 nearest slots were all blocked — even when clear
        # slots existed further away. On a 19×19 grid (361 slots),
        # evaluating all of them is cheap (O(slots × candidates_per_cell)
        # ≈ 27k / 10 = 2.7k overlap checks per macro with the grid).
        # For very large boards the grid is capped at
        # max_candidate_slots=400 anyway.
        for d2, sx, sy in slots:
            if not _overlaps_at(m, sx, sy, macros, keepout=ke, grid_ref=grid):
                # Jump to this slot. Use translate (which respects bounds)
                # so we don't accidentally place the macro OOB.
                dx = sx - cur_x
                dy = sy - cur_y
                if m.translate(dx, dy, bounds=bounds):
                    jumps_made += 1
                    # Rebuild the grid so the next macro's slot search
                    # sees this macro at its new position. Cheap (O(N))
                    # and only fires on accepted jumps.
                    grid = _MacroSpatialGrid.from_macros(macros, bounds)
                    break

    if verbose and jumps_made:
        print(f"  Tetris: jumped {jumps_made} macros to clear slots")

    return _count_residual_overlaps(macros)


def _count_residual_overlaps(macros: list["Macro"]) -> int:
    """Count pairwise macro overlaps (helper).

    Uses ``_MacroSpatialGrid`` when the macro count is large enough to
    benefit (>15 macros — below that the grid build cost exceeds the
    scan cost). Below the threshold the plain O(N²) scan runs — that's
    still the right answer for small boards where the grid overhead
    would dominate. See IMPROVEMENTS §2.3.
    """
    n = len(macros)
    if n < 15:
        # Plain O(N²) — grid build cost would exceed the scan cost.
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if macros[i].overlaps(macros[j]):
                    count += 1
        return count
    # Grid-accelerated: build once, query per macro.
    bounds = _macros_bounds(macros)
    grid = _MacroSpatialGrid.from_macros(macros, bounds)
    seen: set[tuple[int, int]] = set()
    count = 0
    for i in range(n):
        for j in grid.query_candidates(i, macros):
            pair = (i, j) if i < j else (j, i)
            if pair in seen:
                continue
            seen.add(pair)
            if macros[i].overlaps(macros[j]):
                count += 1
    return count


def _macros_bounds(macros: list["Macro"]) -> tuple[float, float, float, float]:
    """Compute the AABB of every macro. Used as the spatial grid extent."""
    x_min = y_min = float("inf")
    x_max = y_max = float("-inf")
    for m in macros:
        bx1, by1, bx2, by2 = m.bbox
        if bx1 < x_min:
            x_min = bx1
        if by1 < y_min:
            y_min = by1
        if bx2 > x_max:
            x_max = bx2
        if by2 > y_max:
            y_max = by2
    if x_min == float("inf"):
        return (0.0, 0.0, 1.0, 1.0)
    return (x_min, y_min, x_max, y_max)


def _macro_in_keepout(m: "Macro", keepouts: list) -> tuple[float, float, float] | None:
    """Return ``(push_x, push_y, depth)`` to evict ``m`` from the deepest keepout it overlaps.

    Returns None when the macro is clear of every keepout (or when there
    are no keepouts). The push vector points from the center of the
    nearest violated keepout toward the macro's center — applying it
    moves the macro OUT of that keepout along the cheaper axis (the one
    with less overlap depth), matching the strategy used by
    ``push_apart_overlapping``.

    Used by ``_keepout_clamp`` to evict macros from internal cutouts
    (mounting slots, milled pockets, non-plated through-holes). Without
    this, SA's cost function (``total_keepout_overlap``) gets the
    gradient right but the legalizer would have to fight the SA gradient
    instead of reinforcing it. See IMPROVEMENTS §2.1.
    """
    if not keepouts:
        return None
    if m.is_fixed:
        # Fixed macros (connectors) intentionally overhang — skip.
        return None
    mx1, my1, mx2, my2 = m.bbox
    mcx = (mx1 + mx2) / 2.0
    mcy = (my1 + my2) / 2.0
    best: tuple[float, float, float] | None = None
    best_depth = -1.0
    for k in keepouts:
        ox1 = max(mx1, k.x_min)
        oy1 = max(my1, k.y_min)
        ox2 = min(mx2, k.x_max)
        oy2 = min(my2, k.y_max)
        if ox2 <= ox1 or oy2 <= oy1:
            continue
        ox_depth = ox2 - ox1
        oy_depth = oy2 - oy1
        kcx = (k.x_min + k.x_max) / 2.0
        kcy = (k.y_min + k.y_max) / 2.0
        # Direction from keepout center to macro center (push outward).
        dx = mcx - kcx
        dy = mcy - kcy
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            # Macro exactly centred on the keepout — pick X by default.
            dx, dy = 1.0, 0.0
            dist = 1.0
        # Cheaper axis: smaller of ox_depth, oy_depth.
        if ox_depth <= oy_depth:
            push = ox_depth + 0.01
            px = (1.0 if dx >= 0 else -1.0) * push
            py = 0.0
            depth = ox_depth
        else:
            push = oy_depth + 0.01
            px = 0.0
            py = (1.0 if dy >= 0 else -1.0) * push
            depth = oy_depth
        if depth > best_depth:
            best_depth = depth
            best = (px, py, depth)
    return best


def _keepout_clamp(
    macros: list["Macro"],
    bounds: tuple[float, float, float, float],
    keepouts: list,
) -> int:
    """Evict any non-fixed macro that overlaps an internal keepout zone.

    Iterates up to 5 passes: each pass picks the deepest keepout overlap
    for each macro and pushes it out along the cheaper axis (mirroring
    ``push_apart_overlapping``'s strategy). Stops when no macro moved.

    Returns the count of macros still inside a keepout after the loop
    (residual violations — these get reported to the caller and surface
    as a keepout_overlap penalty in the SA cost function so the next SA
    round can try to fix them with a different placement).

    Fixed macros (connectors placed on the perimeter with intentional
    overhang) are skipped — see ``_macro_in_keepout``.
    """
    if not keepouts:
        return 0
    x_min, y_min, x_max, y_max = bounds
    failed = 0
    for _ in range(5):
        any_moved = False
        for m in macros:
            push = _macro_in_keepout(m, keepouts)
            if push is None:
                continue
            px, py, _depth = push
            snap = m._snapshot()
            ok = m.translate(px, py, bounds=bounds)
            if not ok:
                # Try only the non-zero axis, then the perpendicular
                # as a fallback (one of them may be blocked by bounds).
                if abs(px) > 1e-9:
                    ok = m.translate(px, 0.0, bounds=bounds)
                elif abs(py) > 1e-9:
                    ok = m.translate(0.0, py, bounds=bounds)
            if ok:
                # Reject moves that create a new macro-macro overlap.
                new_overlap = any(m.overlaps(o) for o in macros if o is not m)
                if new_overlap:
                    m._restore(snap)
                else:
                    any_moved = True
            else:
                m._restore(snap)
        if not any_moved:
            break
    # Count residual macros still inside a keepout (for reporting).
    failed = 0
    for m in macros:
        if _macro_in_keepout(m, keepouts) is not None:
            failed += 1
    return failed


def _fit_macro_to_polygon(
    m: "Macro",
    board: "BoardOutline",
    *,
    others: list["Macro"] | None = None,
    keepout: float = 0.0,
    bounds: tuple[float, float, float, float] | None = None,
) -> bool:
    """Fit macro ``m`` fully inside a non-rectangular board outline.

    Pulls the macro out of any concavity it has strayed into (a connector
    notch, mouse-bite, mounting cutout, or hole) by translating it to the
    nearest position — found via ``board.fit_bbox_inside`` — where its
    union bbox is fully on the TRUE board, honoring ``keepout`` as a DFM
    edge margin.

    Rigid bodies make this cheap: translating the leader applies the same
    delta to every follower, so the union bbox's half-extents are preserved
    by the move. Fitting the union half-extents to the polygon therefore
    guarantees every follower fits it too — the cap-IC rigid-body invariant
    cannot be broken by a pure translation. The translation is validated
    against the board's AABB (a superset of the polygon), so a polygon-valid
    target always passes the rectangle bounds check ``Macro.translate``
    performs; ``board.contains_bbox`` is the final polygon authority,
    re-checked after the move.

    Returns True if the macro ended the call fully inside the outline,
    False otherwise. When ``others`` is given, a fit that would create a
    NEW macro-macro overlap is rolled back and returns False — the caller
    leaves the macro where it was for the next pass / Tetris to handle,
    the same recovery contract as the overlap-aware rectangle clamp.

    No-op (returns True) for fixed macros or a rectangular outline —
    callers gate polygon behavior on ``board.is_polygon``, and these
    guards keep the helper safe if they don't.
    """
    if m.is_fixed or not board.is_polygon:
        return True
    bx1, by1, bx2, by2 = m.bbox
    cur_cx = (bx1 + bx2) / 2.0
    cur_cy = (by1 + by2) / 2.0
    half_w = (bx2 - bx1) / 2.0
    half_h = (by2 - by1) / 2.0

    tx, ty = board.fit_bbox_inside(cur_cx, cur_cy, half_w, half_h, margin=keepout)
    if abs(tx - cur_cx) < 1e-9 and abs(ty - cur_cy) < 1e-9:
        # fit_bbox_inside returned the current center — accept iff it is
        # genuinely contained (it converges immediately when already in,
        # but may give up after max_iterations still partly off-board).
        return board.contains_bbox(m.bbox)

    # The board AABB is a superset of the polygon, so a polygon-valid
    # target passes the rectangle bounds check Macro.translate performs.
    rect_bounds = bounds if bounds is not None else (
        board.x_min, board.y_min, board.x_max, board.y_max)
    snap = m._snapshot()
    if not m.translate(tx - cur_cx, ty - cur_cy, bounds=rect_bounds):
        return board.contains_bbox(m.bbox)
    if not board.contains_bbox(m.bbox):
        m._restore(snap)
        return False
    if others is not None and any(m.overlaps(o) for o in others if o is not m):
        m._restore(snap)
        return False
    return True


def boundary_clamp(
    macros: list["Macro"],
    bounds: tuple[float, float, float, float],
    per_macro_keepout: "callable[[Macro], float] | None" = None,
    board: "BoardOutline | None" = None,
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

    Polygon outlines: when ``board`` is a non-rectangular outline, each
    macro is fit to the TRUE board geometry (notches/cutouts/holes
    excluded) via ``_fit_macro_to_polygon`` instead of the four-sided
    rectangle clamp — the rectangle clamp would otherwise push a
    component into a notch it shares an AABB with. The rectangular fast
    path is byte-for-byte unchanged when ``board`` is None or
    rectangular, so existing rectangle-outline behavior (and its tuning)
    is preserved exactly.

    Fixed macros (``is_fixed=True``) are skipped — their positions are
    intentional (e.g. connectors placed on the perimeter with overhang)
    and clamping them would corrupt the placement.
    """
    x_min, y_min, x_max, y_max = bounds
    failed = 0
    is_poly = board is not None and board.is_polygon
    for m in macros:
        if m.is_fixed:
            continue  # connectors stay where place_connectors_perimeter put them
        # Per-macro extra keepout (ICs/MCUs get pushed further from edge)
        ke = per_macro_keepout(m) if per_macro_keepout else 0.0
        if is_poly:
            # Polygon path: fit the union bbox inside the true outline
            # (clear of notches/cutouts/holes), honoring the per-macro
            # keepout as a margin. The rectangle path below is bypassed
            # entirely for polygon boards.
            if not _fit_macro_to_polygon(m, board, keepout=ke, bounds=bounds):
                failed += 1
            continue
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
    use_abacus: bool = False,
    use_sa_polish: bool = False,
    seed: int = 42,
) -> dict[str, int]:
    """Iterative legalization: snap → (push apart → clamp) repeated.

    Each round repairs the overlaps created by the previous clamp, then
    clamps the resulting positions back inside bounds. Stops when a
    round makes no progress or after ``max_rounds`` iterations.

    Three overlap-resolution strategies, mutually exclusive
    (``use_abacus`` is checked first, then ``use_sa_polish``, else the
    original greedy heuristic stack runs):

    - Default (both False): push-apart / force-spread — pure greedy
      descent. Gets stuck in local minima (a macro boxed in by 3-4
      overlapping neighbors can't move anywhere without making its OWN
      overlap count temporarily worse, even when that's the only path
      to a better configuration one move later).
    - ``use_abacus=True``: bridges macro-v2 onto the legacy row-based
      Abacus DP legalizer (``place/abacus_bridge.py``). MEASURED
      WORSE than the default on every bundled test board — see that
      module's docstring. Kept for comparison, not because it wins.
    - ``use_sa_polish=True``: runs a staged, overlap-weighted
      simulated-annealing pass (``place/sa_polish.py``, beta ramp +
      overlap-biased move selection) instead of greedy descent. SA's
      accept-temporarily-worse-moves criterion is the local-minima
      escape greedy descent lacks. MEASURED (seed=42, post overlap-aware-
      clamp fix): overlap-free on every bundled board — the overlap-aware
      clamp now shared by all three strategies closed the test6 residual-
      overlap gap this was built to fix. A net HPWL win over the heuristic
      on 4/6 boards (test4, cbb, cbbwO, th_sensor), narrowly behind on
      2/6 (test5, test6). See ``place/sa_polish.py`` docstring.

    Whichever strategy runs, keepout enforcement and the Tetris
    "displace to clear slot" fallback further down still run — none of
    these three has a hard zero-overlap guarantee on an arbitrarily
    dense or irregular board.

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
    # The board outline — None for a plain rectangle. Threaded into every
    # clamp / Tetris call so polygon outlines (notches/cutouts/holes) are
    # honored on the default path; rectangular boards take the unchanged
    # fast path everywhere (board is None-or-rectangular ⇒ is_poly False).
    board = getattr(model, "board", None)
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

    # Import the overlap-aware clamp for use in all three strategy branches.
    # Defined in sa_polish to avoid a circular import (sa_polish imports
    # from legalizer at function-call time, not module-load time).
    from place.sa_polish import _boundary_clamp_overlap_aware

    if use_abacus:
        abacus_legalize_macros(model, macros, bounds, grid_mm=grid_mm, verbose=verbose)
        failed = boundary_clamp(macros, bounds, per_macro_keepout=per_macro_keepout,
                                board=board)
        residual = _count_residual_overlaps(macros)
        if residual > 0 and verbose:
            print(f"  Abacus: {residual} residual overlap(s) after row DP + boundary clamp")
    elif use_sa_polish:
        polish_stats = sa_polish_legalize(
            model, macros, bounds, grid_mm=grid_mm, verbose=verbose,
            per_macro_keepout=per_macro_keepout,
            seed=seed,
        )
        # sa_polish_legalize already ran an overlap-aware boundary clamp
        # internally and returned residual_overlaps/boundary_failures.
        # Running the plain boundary_clamp here AGAIN would yank macros
        # inward with no overlap check, re-introducing the overlaps SA
        # just resolved (measured on test4 seed=0: SA reached 0, the
        # plain clamp here created 4, then the keepout loop + Tetris
        # below added 1 more → 5 final overlaps vs 0 from SA alone).
        # Trust sa_polish's clamp; only re-clamp if it reported
        # boundary failures (a macro it couldn't place in-bounds).
        failed = polish_stats.get("boundary_failures", 0)
        residual = polish_stats.get("residual_overlaps", 0)
        if failed > 0:
            # sa_polish left some macros OOB — try one more overlap-aware
            # clamp to pull them in.
            failed = _boundary_clamp_overlap_aware(
                macros, bounds, per_macro_keepout=per_macro_keepout,
                board=board,
            )
            residual = _count_residual_overlaps(macros)
    else:
        residual = push_apart_overlapping(macros, bounds, max_passes=max_push_passes)
        # Use overlap-aware clamp: the plain boundary_clamp translates each
        # macro independently with NO overlap check, which can yank an IC
        # inward by up to 5mm (its keepout) and slam it into a previously
        # overlap-free cap. The overlap-aware variant rolls back a macro's
        # translation if it would create a new overlap, preferring to
        # leave the macro slightly OOB (which the next push_apart round
        # can fix) over creating a new overlap (which push_apart might
        # not be able to fix without pushing the macro back OOB).
        failed = _boundary_clamp_overlap_aware(
            macros, bounds, per_macro_keepout=per_macro_keepout,
            board=board,
        )

        # Iterate: clamp creates overlaps, push-apart fixes them but may push
        # something back OOB, clamp fixes that, etc. Continue until stable.
        for round_idx in range(max_rounds):
            new_residual = push_apart_overlapping(macros, bounds, max_passes=max_push_passes)
            new_failed = _boundary_clamp_overlap_aware(
                macros, bounds, per_macro_keepout=per_macro_keepout,
                board=board,
            )
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
            # Re-clamp with overlap-aware variant after the cleanup push.
            failed = _boundary_clamp_overlap_aware(
                macros, bounds, per_macro_keepout=per_macro_keepout,
                board=board,
            )

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
                            f"overlaps (-{residual - spread_residual})"
                        )
                    residual = spread_residual
                    # Re-clamp after force-spread (it may have pushed
                    # macros OOB). Use overlap-aware clamp so this
                    # doesn't yank macros into each other.
                    failed = _boundary_clamp_overlap_aware(
                        macros, bounds, per_macro_keepout=per_macro_keepout,
                        board=board,
                    )
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
        # Recompute failed: any non-fixed macro whose bbox pokes outside
        # the GLOBAL bounds (hard OOB — placement is physically invalid).
        # Keepout violations (macro inside global bounds but within its
        # keepout zone) are NOT counted as failures — they're a DFM
        # concern (parts too close to board edge), not a placement
        # invalidity. The previous version counted keepout violations
        # as `failed`, which made the CLI exit 1 on boards where ICs
        # are legitimately near the edge (e.g. test4: 4 large ICs in a
        # 54×52mm board can't all clear a 2mm keepout without overlap).
        # Overlap is a hard invalid (two parts in the same space);
        # keepout violation is a soft warning (parts manufacturable but
        # close to edge). The CLI distinguishes these now.
        failed = 0
        is_poly_failed = board is not None and board.is_polygon
        bx_min, by_min, bx_max, by_max = bounds
        for m in macros:
            if m.is_fixed:
                continue
            if is_poly_failed:
                # Off the TRUE board (notch/cutout/hole or past the outer
                # edge) counts as a hard failure — contains_bbox is the
                # polygon authority, not the AABB.
                if not board.contains_bbox(m.bbox):
                    failed += 1
            else:
                bx1, by1, bx2, by2 = m.bbox
                if (bx1 < bx_min - 1e-6 or bx2 > bx_max + 1e-6 or
                    by1 < by_min - 1e-6 or by2 > by_max + 1e-6):
                    failed += 1

    # ─── Internal-keepout clamp (mounting slots, milled pockets, NPTH holes) ──
    # The cost function's `total_keepout_overlap` gives SA a gradient
    # AWAY from internal cutouts; this clamp reinforces that with a
    # hard legalizer pass that evicts any macro still sitting on a
    # keepout after the boundary clamp + push-apart rounds above.
    # See IMPROVEMENTS §2.1.
    keepouts = getattr(model, "keepouts", None) or []
    keepout_failures = 0
    if keepouts:
        keepout_failures = _keepout_clamp(macros, bounds, keepouts)
        if keepout_failures > 0 and verbose:
            print(
                f"  Keepout clamp: {keepout_failures} macro(s) still inside "
                f"an internal cutout (cost-function penalty will apply)"
            )
        # Push-apart may be needed if the keepout eviction displaced
        # macros into each other.
        residual = push_apart_overlapping(macros, bounds, max_passes=max_push_passes)
        # Re-run the overlap-aware boundary clamp (keepout eviction may
        # have pushed something OOB).
        failed = _boundary_clamp_overlap_aware(
            macros, bounds, per_macro_keepout=per_macro_keepout,
            board=board,
        )
        residual = _count_residual_overlaps(macros)

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
    # Only runs if there are still residual overlaps OR OOB macros after
    # force-spread and the keepout enforcement pass. Skips macros whose
    # overlaps are exclusively with fixed obstacles (no clear slot would
    # help). Also catches OOB macros left by the overlap-aware boundary
    # clamp — those need a global slot search, not local push_apart.
    if residual > 0 or failed > 0:
        new_residual = displace_to_clear_slots(
            macros, bounds, grid_mm=grid_mm, verbose=verbose,
            per_macro_keepout=per_macro_keepout,
            board=board,
        )
        if new_residual < residual:
            if verbose:
                print(
                    f"  Tetris cleanup: {residual} -> {new_residual} "
                    f"overlaps (-{residual - new_residual})"
                )
            residual = new_residual
        # Re-clamp to bounds (Tetris jumps stay in-bounds by construction,
        # but the slot search uses a sparse grid — a macro might end up
        # at a position where its bbox pokes slightly past bounds).
        # DO NOT re-run push_apart here — it would undo the Tetris jumps
        # (push_apart is a LOCAL heuristic that can push a carefully-
        # placed macro back into an overlap the Tetris cleanup just
        # resolved). Use overlap-aware clamp so we don't yank macros
        # into each other while pulling them in-bounds.
        failed = _boundary_clamp_overlap_aware(
            macros, bounds, per_macro_keepout=per_macro_keepout,
            board=board,
        )
        # CRITICAL: even the overlap-aware clamp can leave a macro OOB
        # (when no in-bounds position is overlap-free). Recount the
        # residual so the legalizer's self-report matches the actual
        # state. Without this recount, the legalizer silently
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
            + (f", {keepout_failures} keepout violations" if keepout_failures else "")
        )

    return {
        "residual_overlaps": residual,
        "boundary_failures": failed,
        "cap_ic_overlaps": cap_ic_overlaps,
        "keepout_failures": keepout_failures,
        "expanded_bounds": bounds,
    }
