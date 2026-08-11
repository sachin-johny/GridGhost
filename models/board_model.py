"""Core data model for PCB placement.

Defines the intermediate representation used between KiCad extraction
and the optimization engine. All coordinates are in millimeters.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, asdict
from typing import Optional


# Pre-compiled regexes for Component.is_edge_connector — previously
# re.search() ran on every property access in tight loops.
_RE_VERTICAL_THT = re.compile(r'Vertical|THT', re.IGNORECASE)
_RE_HORIZONTAL = re.compile(r'Horizontal|Angled|Side', re.IGNORECASE)

# Component types whose bboxes may legitimately overlap (pad stacks around
# drills, fiducial marker stacks, test coupons). When BOTH components in a
# pair are mechanical features, overlaps are exempt — see
# Component.overlaps and Macro.overlaps (models/macro.py) for the full
# rationale. Must stay in sync with models/macro._MECHANICAL_COMPONENT_TYPES.
_MECHANICAL_COMPONENT_TYPES = frozenset({
    "mounting_hole",
    "fiducial",
    "test_coupon",
})


# ---------------------------------------------------------------------------
# Polygon geometry helpers for non-rectangular board outlines.
#
# Real boards are frequently NOT axis-aligned rectangles: USB/HDMI connector
# notches, mouse-bite tabs, castellated edges, and mounting-hole cutouts are
# routine. These free functions implement the small set of 2D primitives
# BoardOutline needs (point-in-polygon, segment distance, segment
# intersection) so that boundary containment/clamping is correct for any
# simple polygon, not just a rectangle. Kept dependency-free (no numpy/
# shapely) to match the rest of the codebase.
# ---------------------------------------------------------------------------

Point = tuple[float, float]


def _point_on_segment(px: float, py: float, a: Point, b: Point, eps: float = 1e-9) -> bool:
    """True if (px, py) lies on segment a-b, within a small tolerance."""
    ax, ay = a
    bx, by = b
    seg_len = math.hypot(bx - ax, by - ay)
    if seg_len < eps:
        return math.hypot(px - ax, py - ay) < eps
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    if abs(cross) > eps * max(1.0, seg_len):
        return False
    dot = (px - ax) * (bx - ax) + (py - ay) * (by - ay)
    if dot < -eps or dot > seg_len * seg_len + eps:
        return False
    return True


def _point_in_polygon(x: float, y: float, poly: list[Point]) -> bool:
    """Ray-casting point-in-polygon test for a simple (possibly concave)
    polygon. Points exactly on an edge count as inside/contained."""
    n = len(poly)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if _point_on_segment(x, y, (xi, yi), (xj, yj)):
            return True
        if (yi > y) != (yj > y):
            x_intersect = xi + (y - yi) * (xj - xi) / (yj - yi)
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def _closest_point_on_segment(px: float, py: float, a: Point, b: Point) -> Point:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-18:
        return (ax, ay)
    t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    return (ax + t * dx, ay + t * dy)


def _point_segment_distance(px: float, py: float, a: Point, b: Point) -> float:
    cx, cy = _closest_point_on_segment(px, py, a, b)
    return math.hypot(px - cx, py - cy)


def _orient(a: Point, b: Point, c: Point) -> int:
    val = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if val > 1e-9:
        return 1
    if val < -1e-9:
        return -1
    return 0


def _on_seg_bbox(a: Point, b: Point, c: Point, eps: float = 1e-9) -> bool:
    return (min(a[0], b[0]) - eps <= c[0] <= max(a[0], b[0]) + eps and
            min(a[1], b[1]) - eps <= c[1] <= max(a[1], b[1]) + eps)


def _segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """Standard O(1) segment-segment intersection test, including the
    collinear-overlap edge case."""
    o1, o2 = _orient(p1, p2, p3), _orient(p1, p2, p4)
    o3, o4 = _orient(p3, p4, p1), _orient(p3, p4, p2)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_seg_bbox(p1, p2, p3):
        return True
    if o2 == 0 and _on_seg_bbox(p1, p2, p4):
        return True
    if o3 == 0 and _on_seg_bbox(p3, p4, p1):
        return True
    if o4 == 0 and _on_seg_bbox(p3, p4, p2):
        return True
    return False


def _polygon_area(poly: list[Point]) -> float:
    """Shoelace area (unsigned)."""
    area = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _is_component_overlap_exempt(a: "Component", b: "Component") -> bool:
    """Return True if the (a, b) component pair is exempt from overlap checks.

    Mirrors ``models.macro._is_overlap_exempt`` at the Component level so
    the test harness's independent overlap scan
    (``tests/run_all.py:_scan_overlaps``) agrees with the legalizer's
    Macro-level overlap count. Without this, the harness would report
    mounting-hole pad-stack overlaps that the legalizer correctly ignores
    — making it look like the legalizer underreports overlaps.
    """
    a_type = getattr(a, "component_type", "") or ""
    b_type = getattr(b, "component_type", "") or ""
    return (a_type in _MECHANICAL_COMPONENT_TYPES
            and b_type in _MECHANICAL_COMPONENT_TYPES)


def rotated_bbox_offset(
    bbox_offset_x: float,
    bbox_offset_y: float,
    rotation: float,
) -> tuple[float, float]:
    """World-space (dx, dy) from a component's KiCad origin to its bbox center.

    Matches ``Component.bbox`` / ``bbox_at`` exactly: KiCad uses clockwise-
    positive rotation, so ``sin_a = -sin(rad)``.  This is the single source
    of truth for "where is the visible body relative to the origin" — every
    call site that needs to position a component by its body (not its pin-1
    origin) should go through here or through ``Component.set_bbox_center``
    rather than reimplementing the rotation math.

    Returns the offset you add to the origin to get the bbox center, or
    subtract from a target bbox-center to get the origin.
    """
    rad = math.radians(rotation)
    cos_a = math.cos(rad)
    sin_a = -math.sin(rad)  # KiCad clockwise-positive
    dx = bbox_offset_x * cos_a - bbox_offset_y * sin_a
    dy = bbox_offset_x * sin_a + bbox_offset_y * cos_a
    return (dx, dy)


@dataclass
class Pad:
    """A single pad on a component footprint."""
    pad_name: str
    x: float  # mm, relative to component origin
    y: float  # mm, relative to component origin
    net: Optional[str] = None  # net name this pad belongs to

    def absolute_pos(self, comp_x: float, comp_y: float, rotation: float = 0.0) -> tuple[float, float]:
        """Return absolute pad position given component position and rotation."""
        import math
        rad = math.radians(rotation)
        cos_r, sin_r = math.cos(rad), -math.sin(rad)  # KiCad uses clockwise-positive rotation
        abs_x = comp_x + self.x * cos_r - self.y * sin_r
        abs_y = comp_y + self.x * sin_r + self.y * cos_r
        return abs_x, abs_y


@dataclass
class Component:
    """A placed or unplaced component on the board.

    Cached bbox / effective_width / effective_height. Position and rotation
    changes mark the cache dirty so the next read recomputes.
    """
    ref: str
    footprint: str = ""
    value: str = ""
    x: float = 0.0
    y: float = 0.0
    rotation: float = 0.0
    layer: str = "top"
    width: float = 0.0
    height: float = 0.0
    courtyard_margin: float = 0.25
    bbox_offset_x: float = 0.0
    bbox_offset_y: float = 0.0
    pads: list[Pad] = field(default_factory=list)
    nets: list[str] = field(default_factory=list)
    is_fixed: bool = False
    component_type: str = "generic"
    # Hierarchical schematic sheet this component came from (KiCad 7+).
    # Populated from the `sheetname` field in the footprint expression —
    # e.g. "/MCU/", "/POWER/".  Empty string for flat schematics or
    # root-sheet components (sheetname="/").  Used by Phase 3.1 sheet-
    # aware clustering: components sharing a sheet get a strong prior
    # edge in the clustering hypergraph so the auto-placer respects the
    # designer's own functional grouping.  See AUDIT_PHASE0.md §3.1.
    sheet: str = ""

    _cos_a: float = field(default=1.0, repr=False, compare=False)
    _sin_a: float = field(default=0.0, repr=False, compare=False)

    # Cached computed values (invalidated when x/y/rotation change)
    _dirty: bool = field(default=True, repr=False, compare=False)
    _cached_bbox: tuple = field(default=None, repr=False, compare=False)
    _cached_eff_w: float = field(default=0.0, repr=False, compare=False)
    _cached_eff_h: float = field(default=0.0, repr=False, compare=False)
    _cached_rot90: bool = field(default=False, repr=False, compare=False)
    _courtyard_w: float = field(default=0.0, repr=False, compare=False)
    _courtyard_h: float = field(default=0.0, repr=False, compare=False)
    _courtyard_rot_w: float = field(default=0.0, repr=False, compare=False)
    _courtyard_rot_h: float = field(default=0.0, repr=False, compare=False)

    # Cached at construction — footprint/value/component_type never change
    # after parse time, so this avoids re-running regex on every tight-loop
    # access (legalizer/SA hit this many times per iteration).
    is_edge_connector: bool = field(default=False, repr=False, compare=False, init=False)

    _POSITION_FIELDS = frozenset({'x', 'y', 'rotation', 'bbox_offset_x', 'bbox_offset_y'})

    def __post_init__(self):
        self._update_courtyard_sizes()
        self._dirty = True
        self.is_edge_connector = self._compute_is_edge_connector()
        # Re-initialise _cos_a/_sin_a from self.rotation. The dataclass
        # field defaults for _cos_a (1.0) and _sin_a (0.0) are applied
        # by __init__ AFTER the rotation assignment, overwriting the
        # values __setattr__ computed. Without this re-init, any
        # Component constructed with a non-zero rotation has its bbox
        # computed as if rotation=0 (because bbox uses _cos_a/_sin_a
        # to rotate bbox_offset_x/y onto the component's local frame).
        # This bug is invisible for components SA later moves (set_pose
        # calls set_rotation which fixes _cos_a/_sin_a) but corrupts
        # the bbox of every rotated component that stays at its parse
        # position (e.g. fixed mounting holes, edge connectors) and
        # every component in a freshly re-parsed output file.
        rad = math.radians(self.rotation)
        super().__setattr__('_cos_a', math.cos(rad))
        super().__setattr__('_sin_a', -math.sin(rad))

    def __setattr__(self, name: str, value) -> None:
        super().__setattr__(name, value)
        if name in Component._POSITION_FIELDS:
            super().__setattr__('_dirty', True)
            if name == 'rotation':
                rad = math.radians(value)
                super().__setattr__('_cos_a', math.cos(rad))
                super().__setattr__('_sin_a', -math.sin(rad))

    def _update_courtyard_sizes(self):
        self._courtyard_w = self.width + 2 * self.courtyard_margin
        self._courtyard_h = self.height + 2 * self.courtyard_margin
        self._courtyard_rot_w = self.height + 2 * self.courtyard_margin
        self._courtyard_rot_h = self.width + 2 * self.courtyard_margin

    def _recompute_cache(self):
        rot90 = int(self.rotation) % 180 == 90
        self._cached_rot90 = rot90
        if rot90:
            self._cached_eff_w = self._courtyard_rot_w
            self._cached_eff_h = self._courtyard_rot_h
        else:
            self._cached_eff_w = self._courtyard_w
            self._cached_eff_h = self._courtyard_h
        half_w = self._cached_eff_w / 2.0
        half_h = self._cached_eff_h / 2.0
        cx = self.x + self.bbox_offset_x * self._cos_a - self.bbox_offset_y * self._sin_a
        cy = self.y + self.bbox_offset_x * self._sin_a + self.bbox_offset_y * self._cos_a
        self._cached_bbox = (cx - half_w, cy - half_h, cx + half_w, cy + half_h)
        self._dirty = False

    def set_rotation(self, angle: float):
        self.rotation = angle % 360.0

    def bbox_at(self, x: float, y: float, rotation: float) -> tuple[float, float, float, float]:
        """Compute the bbox the component WOULD have at an arbitrary pose.

        Returns the same value ``self.bbox`` would return if the component
        were moved to (x, y, rotation) — including courtyard margin and the
        rotated bbox_offset — WITHOUT mutating the component's state or
        invalidating its cache.

        Used by ``CostState.old_bboxes_from_states`` to reconstruct pre-move
        bboxes for the SA snapshot/restore spatial index.  The previous
        implementation used raw ``width``/``height`` (missing courtyard
        margin) and ignored ``bbox_offset_x``/``bbox_offset_y``, which
        silently corrupted the xmin index over many moves and caused
        incremental cost to drift from from-scratch recompute — see
        ``test_snapshot_restore_n_moves_property``.
        """
        rad = math.radians(rotation)
        cos_a = math.cos(rad)
        sin_a = -math.sin(rad)  # KiCad uses clockwise-positive rotation
        rot90 = int(rotation) % 180 == 90
        if rot90:
            eff_w = self._courtyard_rot_w
            eff_h = self._courtyard_rot_h
        else:
            eff_w = self._courtyard_w
            eff_h = self._courtyard_h
        half_w = eff_w / 2.0
        half_h = eff_h / 2.0
        cx = x + self.bbox_offset_x * cos_a - self.bbox_offset_y * sin_a
        cy = y + self.bbox_offset_x * sin_a + self.bbox_offset_y * cos_a
        return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)

    def set_bbox_center(self, cx: float, cy: float, rotation: float) -> None:
        """Position this component so its bbox center — not its KiCad origin —
        lands at ``(cx, cy)`` at the given rotation.

        This is the bbox-space placement primitive geometry code should use
        whenever the intent is "put this component's visible body here."
        Setting ``.x``/``.y`` directly only does the right thing when the
        origin happens to coincide with the bbox center (``bbox_offset`` ≈ 0),
        which is false for most connectors (pin headers, edge-mount SMA, …).

        Rotation is applied first (the origin→bbox-center offset depends on
        it), then the origin is placed at ``(cx, cy) − rotated_offset`` so
        the bbox center — courtyard margin included — is exactly ``(cx, cy)``.
        """
        self.set_rotation(rotation)  # offset depends on rotation; set first
        dx, dy = rotated_bbox_offset(self.bbox_offset_x, self.bbox_offset_y, rotation)
        self.x = cx - dx
        self.y = cy - dy

    @property
    def _is_rotated_90(self) -> bool:
        if self._dirty:
            self._recompute_cache()
        return self._cached_rot90

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        if self._dirty:
            self._recompute_cache()
        return self._cached_bbox

    @property
    def effective_width(self) -> float:
        if self._dirty:
            self._recompute_cache()
        return self._cached_eff_w

    @property
    def effective_height(self) -> float:
        if self._dirty:
            self._recompute_cache()
        return self._cached_eff_h

    def overlaps(self, other: Component) -> bool:
        # Mounting-hole / mechanical-feature exemption (same logic as
        # Macro.overlaps in models/macro.py — see that method's docstring
        # for the full rationale). When BOTH components are non-electrical
        # mechanical features (mounting holes, fiducials, test coupons),
        # their bboxes may legitimately overlap (pad stacks around drills)
        # and reporting these as placement overlaps is a false positive.
        if _is_component_overlap_exempt(self, other):
            return False
        if self._dirty:
            self._recompute_cache()
        if other._dirty:
            other._recompute_cache()
        ax1, ay1, ax2, ay2 = self._cached_bbox
        bx1, by1, bx2, by2 = other._cached_bbox
        return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)

    def overlap_area(self, other: Component) -> float:
        if _is_component_overlap_exempt(self, other):
            return 0.0
        if self._dirty:
            self._recompute_cache()
        if other._dirty:
            other._recompute_cache()
        ax1, ay1, ax2, ay2 = self._cached_bbox
        bx1, by1, bx2, by2 = other._cached_bbox
        ox1 = max(ax1, bx1)
        oy1 = max(ay1, by1)
        ox2 = min(ax2, bx2)
        oy2 = min(ay2, by2)
        if ox2 <= ox1 or oy2 <= oy1:
            return 0.0
        return (ox2 - ox1) * (oy2 - oy1)

    def _compute_is_edge_connector(self) -> bool:
        """Compute the is_edge_connector flag once at construction.

        True for edge-mount connectors placed on the board perimeter.
        These intentionally overhang the board edge (pads inside, body
        extends outward) and should be excluded from OOB counts.
        Vertical/THT connectors are interior components, not edge connectors.
        """
        if getattr(self, 'component_type', '') != "connector":
            return False
        fp = getattr(self, 'footprint', '') or ''
        val = getattr(self, 'value', '') or ''
        name = fp + ' ' + val
        if _RE_VERTICAL_THT.search(name):
            return False
        if _RE_HORIZONTAL.search(name):
            return True
        return not _RE_VERTICAL_THT.search(name)

    # Restore identity-based hashing so Component instances can be stored in
    # sets/dicts (used by the legalizer's SpatialGrid broad-phase index).
    # dataclass with eq=True sets __hash__ to None; we override with identity
    # hashing since we never compare distinct instances for field equality.
    __hash__ = object.__hash__


@dataclass
class BoardOutline:
    """Board outline — an axis-aligned rectangle by default, or an arbitrary
    simple polygon (with optional interior holes) when ``polygon`` is set.

    Real boards routinely aren't bare rectangles: USB/HDMI connector
    notches, mouse-bite tabs, castellated edges, and non-rectangular
    mounting cutouts are all common. When ``polygon`` is provided,
    x_min/y_min/x_max/y_max become DERIVED values — the polygon's axis-
    aligned bounding box — rather than authoritative. They stay populated
    because plenty of code (spatial-grid sizing, coarse candidate search,
    area-based heuristics) only ever needs a fast outer bound and that
    remains a correct superset of the real shape. Anything that needs to
    know whether a specific point or component actually fits ON the board
    must go through contains() / contains_bbox() / clamp() /
    fit_bbox_inside(), which are polygon-aware.

    ``holes`` are interior cutouts fully enclosed by the outline (as
    opposed to notches, which are concavities in the outer polygon
    itself) — e.g. a non-rectangular mounting slot. Simple rectangular
    interior cutouts can continue to use ``BoardModel.keepouts`` instead;
    ``holes`` is for cutouts whose own shape isn't a rectangle.
    """
    x_min: float = 0.0
    y_min: float = 0.0
    x_max: float = 100.0
    y_max: float = 100.0
    polygon: Optional[list[tuple[float, float]]] = None
    holes: Optional[list[list[tuple[float, float]]]] = None

    def __post_init__(self) -> None:
        if self.polygon and len(self.polygon) >= 3:
            xs = [p[0] for p in self.polygon]
            ys = [p[1] for p in self.polygon]
            self.x_min, self.x_max = min(xs), max(xs)
            self.y_min, self.y_max = min(ys), max(ys)
        else:
            self.polygon = None

    @property
    def is_polygon(self) -> bool:
        """True if this outline has real (non-rectangular-capable) polygon
        geometry rather than just the four rectangle bounds."""
        return bool(self.polygon) and len(self.polygon) >= 3

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)

    def _loops(self) -> list[list[tuple[float, float]]]:
        """All boundary loops (outer polygon + holes) as edge sources."""
        loops = [self.polygon] if self.polygon else []
        loops.extend(self.holes or [])
        return loops

    def contains(self, x: float, y: float) -> bool:
        """Return True if (x, y) is on or inside the outline (and not
        inside any hole)."""
        if not self.is_polygon:
            return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max
        if not _point_in_polygon(x, y, self.polygon):
            return False
        for hole in (self.holes or []):
            if len(hole) >= 3 and _point_in_polygon(x, y, hole):
                # On the hole's boundary still counts as "on the board".
                n = len(hole)
                on_edge = any(
                    _point_on_segment(x, y, hole[i], hole[(i + 1) % n])
                    for i in range(n)
                )
                if not on_edge:
                    return False
        return True

    def contains_bbox(self, bbox: tuple[float, float, float, float]) -> bool:
        """Return True if the entire axis-aligned bbox is inside the board
        outline (and clear of every hole).

        Rectangle outline: simple bounds check (unchanged fast path).
        Polygon outline: a rectangle is fully contained in a simple
        polygon iff all four corners are inside it AND no polygon (or
        hole) edge crosses a rectangle edge — the standard test for
        convex-shape-in-simple-polygon containment.
        """
        x_min, y_min, x_max, y_max = bbox
        if not self.is_polygon:
            return (x_min >= self.x_min and x_max <= self.x_max and
                    y_min >= self.y_min and y_max <= self.y_max)

        corners = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        if not all(self.contains(cx, cy) for cx, cy in corners):
            return False
        # Shrink the crossing test very slightly toward the rectangle's
        # center so a bbox edge that exactly TOUCHES an outline/hole edge
        # (a component sitting flush against a notch wall — a legal,
        # common placement) doesn't register as an "intersection". Only
        # genuine crossings into forbidden territory are rejected.
        eps = 1e-6
        rcx, rcy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
        shrunk = [(cx + (rcx - cx) * eps, cy + (rcy - cy) * eps) for cx, cy in corners]
        rect_edges = [(shrunk[i], shrunk[(i + 1) % 4]) for i in range(4)]
        for loop in self._loops():
            n = len(loop)
            for i in range(n):
                a, b = loop[i], loop[(i + 1) % n]
                for (p, q) in rect_edges:
                    if _segments_intersect(a, b, p, q):
                        return False
        # Corners-inside + no-edge-crossing is sufficient to prove the
        # rectangle doesn't cross the OUTER polygon boundary (Jordan-curve
        # argument: with no crossing and all corners inside, the whole
        # rectangle is inside). But a hole entirely nested INSIDE the
        # rectangle — fully swallowed, touching none of its edges — is a
        # distinct failure mode the corner/edge tests can't see (none of
        # the rectangle's corners land inside a hole that's smaller than
        # the rectangle and centered within it). Since we've already
        # established no edge of the hole crosses the rectangle, one
        # hole vertex inside the rectangle's bounds means the entire hole
        # is swallowed by it.
        for hole in (self.holes or []):
            for (hx, hy) in hole:
                if x_min <= hx <= x_max and y_min <= hy <= y_max:
                    return False
        return True

    def distance_to_boundary(self, x: float, y: float) -> float:
        """Non-negative distance from (x, y) to the nearest outline/hole
        edge (regardless of whether the point is inside or outside)."""
        if not self.is_polygon:
            return min(x - self.x_min, self.x_max - x, y - self.y_min, self.y_max - y)
        best = float("inf")
        for loop in self._loops():
            n = len(loop)
            for i in range(n):
                d = _point_segment_distance(x, y, loop[i], loop[(i + 1) % n])
                if d < best:
                    best = d
        return best if best != float("inf") else 0.0

    def nearest_boundary_point(self, x: float, y: float) -> tuple[float, float]:
        """The closest point on the outline/hole boundary to (x, y)."""
        if not self.is_polygon:
            return self.clamp(x, y)
        best_pt = (x, y)
        best_d = float("inf")
        for loop in self._loops():
            n = len(loop)
            for i in range(n):
                a, b = loop[i], loop[(i + 1) % n]
                cx, cy = _closest_point_on_segment(x, y, a, b)
                d = math.hypot(x - cx, y - cy)
                if d < best_d:
                    best_d = d
                    best_pt = (cx, cy)
        return best_pt

    def clamp(self, x: float, y: float) -> tuple[float, float]:
        """Return the nearest point to (x, y) that is on the board."""
        if not self.is_polygon:
            return (
                max(self.x_min, min(x, self.x_max)),
                max(self.y_min, min(y, self.y_max)),
            )
        if self.contains(x, y):
            return (x, y)
        return self.nearest_boundary_point(x, y)

    def bbox_overflow(self, bbox: tuple[float, float, float, float]) -> float:
        """Non-negative, smooth penalty proportional to how far ``bbox``
        extends outside the outline (0.0 when fully contained). This is
        the boundary-cost primitive: linear ramp on overflow distance, so
        SA gets a usable gradient back toward the board.

        Rectangle outline: identical to the original four-sided overflow
        sum (left+right+top+bottom), preserved exactly for backward
        compatibility with existing cost tuning.
        Polygon outline: sum, over each bbox corner that has strayed off
        the board (outside the outline or inside a hole), of that
        corner's distance back to the nearest boundary edge.
        """
        x_min, y_min, x_max, y_max = bbox
        if not self.is_polygon:
            return (max(0.0, self.x_min - x_min) + max(0.0, x_max - self.x_max) +
                    max(0.0, self.y_min - y_min) + max(0.0, y_max - self.y_max))
        total = 0.0
        for (cx, cy) in ((x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)):
            if not self.contains(cx, cy):
                total += self.distance_to_boundary(cx, cy)
        return total

    def fit_bbox_inside(
        self, cx: float, cy: float, half_w: float, half_h: float,
        max_iterations: int = 8, margin: float = 0.0,
    ) -> tuple[float, float]:
        """Return a center point near (cx, cy) such that the axis-aligned
        bbox of half-extents (half_w, half_h) centered there fits fully on
        the board (fully inside the outline, clear of all holes), with at
        least ``margin`` mm of clearance from the boundary (e.g. a DFM
        edge-keepout requirement for an IC/MCU).

        The margin is enforced without computing a true polygon offset
        (an exact Minkowski erosion): a bbox has >= margin clearance from
        the outline boundary along both axes iff its margin-inflated bbox
        (half-extents padded by ``margin`` on every side) is itself fully
        contained by the *un*-inflated outline — so this just re-uses
        contains_bbox()/the same fitting search on the padded size, then
        returns the fitted center for the real (unpadded) component. This
        is an axis-aligned approximation of true Euclidean clearance (the
        same kind of approximation the rectangle-outline margin math
        elsewhere in this codebase already makes), not exact CAD-grade
        offsetting, but it means a DFM margin is honored for polygon
        outlines too rather than silently dropped.

        Rectangle outline: exact single clamp (unchanged behavior when
        margin=0; a plain inward shrink by margin otherwise — identical
        to the pre-existing extra_keepout_mm handling for rectangles).
        Polygon outline: an iterative nudge — corners of the (padded)
        bbox that have wandered off the board (into a notch or a hole)
        get pulled back toward the nearest boundary point, chasing the
        worst offender each pass. This mirrors the keepout-eviction
        heuristic already used elsewhere in the legalizer: a conservative,
        cheap correction rather than an exact placement solve, with any
        residual excursion caught by the legalizer's overlap-resolution
        loop and penalized by ``bbox_overflow``/the edge-keepout term in
        the cost function.
        """
        eff_hw = half_w + max(0.0, margin)
        eff_hh = half_h + max(0.0, margin)
        # Guard against the margin collapsing the fit window entirely on
        # a small board/tight polygon (mirrors the same collapse guard
        # the rectangle-outline boundary clamp uses elsewhere) — fall
        # back to no margin rather than leaving the component unclamped.
        if eff_hw * 2 > self.width or eff_hh * 2 > self.height:
            eff_hw, eff_hh = half_w, half_h

        x_lo, x_hi = self.x_min + eff_hw, self.x_max - eff_hw
        y_lo, y_hi = self.y_min + eff_hh, self.y_max - eff_hh
        x = cx if x_lo > x_hi else max(x_lo, min(cx, x_hi))
        y = cy if y_lo > y_hi else max(y_lo, min(cy, y_hi))

        if not self.is_polygon:
            return (x, y)

        for _ in range(max_iterations):
            bbox = (x - eff_hw, y - eff_hh, x + eff_hw, y + eff_hh)
            if self.contains_bbox(bbox):
                return (x, y)
            corners = [(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])]
            # Push by the vector needed to move the WORST-offending corner
            # (the one furthest from the boundary) exactly onto the
            # boundary. Averaging all four corners' pushes would let
            # opposite-signed corrections cancel out; chasing the worst
            # corner first and re-checking converges instead.
            worst = None
            worst_d = -1.0
            for (px, py) in corners:
                if not self.contains(px, py):
                    bx, by = self.nearest_boundary_point(px, py)
                    d = math.hypot(bx - px, by - py)
                    if d > worst_d:
                        worst_d = d
                        worst = (bx - px, by - py)
            if worst is None:
                # All four corners are on the board, but a notch/hole edge
                # still slices through the bbox interior (the cutout is
                # narrower than the component). Nudge toward the outline
                # centroid as a last-resort escape direction.
                gx = sum(p[0] for p in self.polygon) / len(self.polygon)
                gy = sum(p[1] for p in self.polygon) / len(self.polygon)
                x += (gx - x) * 0.25
                y += (gy - y) * 0.25
            else:
                x += worst[0]
                y += worst[1]
            if x_lo <= x_hi:
                x = max(x_lo, min(x, x_hi))
            if y_lo <= y_hi:
                y = max(y_lo, min(y, y_hi))

        return (x, y)


@dataclass
class Net:
    """A net connecting multiple component pads."""
    name: str
    pins: list[tuple[str, str]] = field(default_factory=list)  # List of (ref, pad_name)
    net_class: str = "Default"

    @property
    def component_refs(self) -> set[str]:
        """Set of unique component references on this net."""
        return {ref for ref, _ in self.pins}

    @property
    def pin_count(self) -> int:
        return len(self.pins)


@dataclass
class BoardModel:
    """Complete board model — the intermediate representation."""
    board: BoardOutline = field(default_factory=BoardOutline)
    components: list[Component] = field(default_factory=list)
    nets: list[Net] = field(default_factory=list)
    source_file: str = ""
    user_defined_outline: bool = False
    # Internal cutouts / mounting-hole zones parsed from Edge.Cuts.
    # Components must not be placed inside any keepout — the legalizer
    # treats keepouts as obstacles and the cost function charges a
    # boundary-style penalty for components overlapping a keepout.
    # See AUDIT_PHASE0.md "Board outline with internal cutouts" for the
    # bug this fixes (mounting holes were silently flattened into the
    # outer outline bbox, letting components be placed inside holes).
    keepouts: list[BoardOutline] = field(default_factory=list)

    # ---- Lookup helpers ----

    def get_component(self, ref: str) -> Optional[Component]:
        """Return the component with the given ref, or None.

        Uses a lazily-built dict index for O(1) lookup.  The index is
        rebuilt automatically if len(self.components) changes.  If you
        replace a component in-place, call rebuild_ref_index() to refresh.
        """
        idx = getattr(self, '_comp_ref_map', None)
        if idx is None or len(idx) != len(self.components):
            idx = {c.ref: c for c in self.components}
            self._comp_ref_map = idx
        return idx.get(ref)

    def rebuild_ref_index(self) -> None:
        """Force a rebuild of the ref -> Component index."""
        self._comp_ref_map = {c.ref: c for c in self.components}

    def get_net(self, name: str) -> Optional[Net]:
        for n in self.nets:
            if n.name == name:
                return n
        return None

    def nets_for_component(self, ref: str) -> list[Net]:
        """Return all nets connected to a given component."""
        return [n for n in self.nets if ref in n.component_refs]

    def components_on_net(self, net_name: str) -> list[Component]:
        """Return all components on a given net."""
        net = self.get_net(net_name)
        if not net:
            return []
        return [c for c in self.components if c.ref in net.component_refs]

    # ---- Keepout helpers ----

    def component_in_keepout(self, comp: Component) -> bool:
        """Return True if the component's bbox overlaps ANY keepout zone.

        Used by the legalizer (treat keepouts as obstacles during overlap
        resolution) and the cost function (boundary-style penalty for
        components overlapping a keepout).  Edge connectors are exempt
        — a connector's body may legitimately overhang a mounting hole
        near the board edge.
        """
        if not self.keepouts:
            return False
        if getattr(comp, 'is_edge_connector', False):
            return False
        bx1, by1, bx2, by2 = comp.bbox
        for k in self.keepouts:
            # AABB overlap test
            if bx2 <= k.x_min or k.x_max <= bx1 or by2 <= k.y_min or k.y_max <= by1:
                continue
            return True
        return False

    # ---- Serialization ----

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {
            "board": asdict(self.board),
            "components": [
                {
                    "ref": c.ref,
                    "footprint": c.footprint,
                    "value": c.value,
                    "x": c.x,
                    "y": c.y,
                    "rotation": c.rotation,
                    "layer": c.layer,
                    "width": c.width,
                    "height": c.height,
                    "courtyard_margin": c.courtyard_margin,
                    "bbox_offset_x": c.bbox_offset_x,
                    "bbox_offset_y": c.bbox_offset_y,
                    "pads": [asdict(p) for p in c.pads],
                    "nets": c.nets,
                    "is_fixed": c.is_fixed,
                    "component_type": c.component_type,
                    "sheet": c.sheet,
                }
                for c in self.components
            ],
            "nets": [
                {
                    "name": n.name,
                    "pins": n.pins,
                    "net_class": n.net_class,
                }
                for n in self.nets
            ],
            "source_file": self.source_file,
            "user_defined_outline": self.user_defined_outline,
            "keepouts": [asdict(k) for k in self.keepouts],
        }

    def to_json(self, path: str) -> None:
        """Write to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> BoardModel:
        """Deserialize from a dictionary."""
        board_data = data.get("board", {})
        board_polygon = board_data.get("polygon")
        board_holes = board_data.get("holes")
        board = BoardOutline(
            x_min=board_data.get("x_min", 0.0),
            y_min=board_data.get("y_min", 0.0),
            x_max=board_data.get("x_max", 100.0),
            y_max=board_data.get("y_max", 100.0),
            polygon=[tuple(p) for p in board_polygon] if board_polygon else None,
            holes=[[tuple(p) for p in hole] for hole in board_holes] if board_holes else None,
        )
        components = []
        for cd in data.get("components", []):
            pads = [
                Pad(pad_name=p.get("pad_name", ""), x=p.get("x", 0.0), y=p.get("y", 0.0), net=p.get("net"))
                for p in cd.get("pads", [])
            ]
            comp = Component(
                ref=cd["ref"],
                footprint=cd.get("footprint", ""),
                value=cd.get("value", ""),
                x=cd.get("x", 0.0),
                y=cd.get("y", 0.0),
                rotation=cd.get("rotation", 0.0),
                layer=cd.get("layer", "top"),
                width=cd.get("width", 0.0),
                height=cd.get("height", 0.0),
                courtyard_margin=cd.get("courtyard_margin", 0.25),
                bbox_offset_x=cd.get("bbox_offset_x", 0.0),
                bbox_offset_y=cd.get("bbox_offset_y", 0.0),
                pads=pads,
                nets=cd.get("nets", []),
                is_fixed=cd.get("is_fixed", False),
                component_type=cd.get("component_type", "generic"),
                sheet=cd.get("sheet", ""),
            )
            components.append(comp)

        nets = []
        for nd in data.get("nets", []):
            net = Net(
                name=nd["name"],
                pins=[tuple(p) for p in nd.get("pins", [])],
                net_class=nd.get("net_class", "Default"),
            )
            nets.append(net)

        keepouts: list[BoardOutline] = []
        for kd in data.get("keepouts", []):
            k_polygon = kd.get("polygon")
            keepouts.append(BoardOutline(
                x_min=kd.get("x_min", 0.0),
                y_min=kd.get("y_min", 0.0),
                x_max=kd.get("x_max", 0.0),
                y_max=kd.get("y_max", 0.0),
                polygon=[tuple(p) for p in k_polygon] if k_polygon else None,
            ))

        return cls(
            board=board,
            components=components,
            nets=nets,
            source_file=data.get("source_file", ""),
            user_defined_outline=data.get("user_defined_outline", False),
            keepouts=keepouts,
        )

    @classmethod
    def from_json(cls, path: str) -> BoardModel:
        """Read from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        model = cls.from_dict(data)
        model.source_file = path
        return model

    # ---- Statistics ----

    def stats(self) -> dict:
        """Return summary statistics of the board model."""
        movable = [c for c in self.components if not c.is_fixed]
        top_comps = [c for c in self.components if c.layer == "top"]
        bottom_comps = [c for c in self.components if c.layer == "bottom"]

        # Count overlaps
        overlap_count = 0
        overlap_area_total = 0.0
        for i, c1 in enumerate(self.components):
            for c2 in self.components[i + 1:]:
                area = c1.overlap_area(c2)
                if area > 0:
                    overlap_count += 1
                    overlap_area_total += area

        # Out-of-bounds — check full bounding box, not just center point.
        # Edge connectors are excluded: they intentionally overhang the board
        # edge with pads inside and body outside.
        oob_count = sum(
            1 for c in self.components
            if not c.is_edge_connector
            and not self.board.contains_bbox(c.bbox)
        )

        return {
            "total_components": len(self.components),
            "movable_components": len(movable),
            "fixed_components": len(self.components) - len(movable),
            "top_components": len(top_comps),
            "bottom_components": len(bottom_comps),
            "total_nets": len(self.nets),
            "overlap_count": overlap_count,
            "overlap_area_total": round(overlap_area_total, 4),
            "out_of_bounds": oob_count,
            "board_size": f"{self.board.width:.1f} x {self.board.height:.1f} mm",
        }
