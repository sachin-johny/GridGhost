"""Macro: a leader component plus rigidly-attached followers.

A Macro is the atomic unit of placement. The SA moves a Macro as one
rigid body (translation + rotation); the legalizer treats a Macro as
a single rectangle for overlap checks. Followers (typically decoupling
caps) have a fixed offset in leader-local coords; the leader's current
rotation is applied to those offsets to compute follower positions.

HARD RULE: a follower's gap to its leader body is fixed at macro
construction (the chosen fan-slot ``spacing``). Because followers move
rigidly with the leader, that gap can NEVER change at runtime. The
construction-time fan search bounds the edge-to-edge gap by
``MAX_CAP_IC_GAP_MM`` (the largest entry in ``_FAN_SPACINGS``); the
legalizer asserts the invariant (no follower overlaps its leader).

Why edge-gap, not center-to-center distance: decoupling effectiveness is
governed by trace length from the cap to the IC pin, which is ~the
edge-to-edge gap — NOT the center-to-center distance. An earlier version
capped the center-to-center distance at 8mm, which is dimensionally wrong
for large ICs: for any IC whose effective half-extent + cap half-extent
exceeds 8mm (e.g. a 12.6mm ESP32: 6.3 + 1.7 = 8.0), every fan slot was
rejected and find_cap_offset collapsed every cap onto the last-resort
slot *inside* the leader, producing a complete pairwise-overlap cluster
(the test4 U30 cluster: 28 overlaps from 7 caps).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from models.board_model import Component


# Max edge-to-edge gap (mm) between a decoupling cap body and its IC body.
# _FAN_SPACINGS are the candidate gaps; its largest entry is the ceiling.
# See module docstring for why this is a gap, not a center-to-center distance.
MAX_CAP_IC_GAP_MM = 4.0

_FAN_SPACINGS = (0.5, 1.0, 2.0, 3.0, 4.0)
_FAN_DIRS = (
    (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
    (1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0),
)

# Component types whose macros are non-electrical mechanical features.
# When BOTH macros in a pair are mechanical, their bboxes may legitimately
# overlap (KiCad places M3 mounting-hole pad+via stacks on 8mm grid spacing;
# their 6.5mm pads with bbox_margin=0.8 produce 8.1mm bboxes → 0.2mm false
# overlap that the legalizer cannot resolve because they're fixed). Reporting
# these as placement overlaps would be a false positive that obscures the
# real residual-overlap signal — see `Macro.overlaps` for the full rationale.
_MECHANICAL_COMPONENT_TYPES = frozenset({
    "mounting_hole",
    "fiducial",
    "test_coupon",
})


def _is_overlap_exempt(a: "Macro", b: "Macro") -> bool:
    """Return True if the (a, b) macro pair is exempt from overlap checks.

    A pair is exempt when BOTH macros' leaders are non-electrical
    mechanical features (mounting holes, fiducials, test coupons).
    Such pairs may legitimately have overlapping bboxes (pad stacks
    around drills) and are never resolvable by the legalizer (they're
    fixed) — counting them as placement overlaps is a false positive.
    """
    a_type = getattr(a.leader, "component_type", "") or ""
    b_type = getattr(b.leader, "component_type", "") or ""
    return (a_type in _MECHANICAL_COMPONENT_TYPES
            and b_type in _MECHANICAL_COMPONENT_TYPES)


def _rotate(dx: float, dy: float, rotation_deg: float) -> tuple[float, float]:
    """Apply KiCad CW-positive rotation to a (dx, dy) offset."""
    rad = math.radians(rotation_deg)
    cos_r = math.cos(rad)
    sin_r = -math.sin(rad)
    return (dx * cos_r - dy * sin_r, dx * sin_r + dy * cos_r)


def find_cap_offset(
    leader: "Component",
    cap: "Component",
    others: Iterable["Component"],
) -> tuple[float, float]:
    """Find an offset for ``cap`` around ``leader``.

    Returns an offset (dx, dy) in leader-local coords at leader
    rotation=0. The fan search tries 8 directions at increasing
    edge-to-edge spacings (``_FAN_SPACINGS``); the first slot whose gap
    to the leader is within ``MAX_CAP_IC_GAP_MM`` and that doesn't
    overlap the leader or any component in ``others`` is returned.

    The slot's offset is ``leader_half + cap_half + spacing`` along each
    axis, so the cap's near edge sits exactly ``spacing`` mm from the
    leader's near edge (cardinal slots) — or farther (diagonals push the
    cap out on both axes). For any ``spacing >= 0`` this can NEVER
    overlap the leader; the ``cap.overlaps(leader)`` check below is a
    defensive guard, not the primary constraint.

    The constraint is on this edge-to-edge GAP, not on the center-to-
    center distance. The old center-distance cap (8mm) was
    dimensionally wrong for large ICs: it rejected every slot once
    ``leader_half + cap_half`` exceeded 8mm, collapsing all caps onto a
    last-resort slot inside the leader. See the module docstring.

    ``others`` is typically the macro's already-placed followers — we
    want the new cap to avoid overlapping its sibling caps. We do NOT
    check against other components in the model because at macro-
    construction time everything is still at parse positions and the
    "overlap-free" check would almost always fail, producing a large-
    gap fallback offset. The legalizer's push-apart will resolve macro-
    vs-macro overlaps after SA.

    If no slot is free of siblings, falls back to the first leader-clear
    slot found (smallest gap). This keeps the cap near its IC at the cost
    of a sibling overlap — but only in the pathological case of more
    caps than directions×spacings (40+ caps on one IC).
    """
    leader_w = leader.effective_width
    leader_h = leader.effective_height
    cap_w = cap.effective_width
    cap_h = cap.effective_height

    others_list = [o for o in others if o is not cap and o is not leader]
    rad = math.radians(leader.rotation)
    cos_r = math.cos(rad)
    sin_r = -math.sin(rad)

    saved_x, saved_y = cap.x, cap.y
    fallback: tuple[float, float] | None = None

    try:
        for spacing in _FAN_SPACINGS:
            # Spacings are ascending; once we pass the gap ceiling, no
            # later (larger) spacing can qualify either.
            if spacing > MAX_CAP_IC_GAP_MM:
                break
            for dx_dir, dy_dir in _FAN_DIRS:
                offset_x = dx_dir * (leader_w / 2 + cap_w / 2 + spacing)
                offset_y = dy_dir * (leader_h / 2 + cap_h / 2 + spacing)

                cap_x = leader.x + offset_x * cos_r - offset_y * sin_r
                cap_y = leader.y + offset_x * sin_r + offset_y * cos_r
                cap.x = cap_x
                cap.y = cap_y

                # Defensive: a cardinal/diagonal slot at spacing >= 0 can
                # never overlap the leader by construction. Kept to guard
                # against future _FAN_DIRS edits that break that property.
                if cap.overlaps(leader):
                    continue

                # Track a fallback in case no slot is sibling-free.
                if fallback is None:
                    fallback = (offset_x, offset_y)

                if not any(cap.overlaps(o) for o in others_list):
                    return (offset_x, offset_y)
    finally:
        cap.x = saved_x
        cap.y = saved_y

    if fallback is not None:
        return fallback

    # Absolute last resort (only if every fan slot overlaps a sibling):
    # a cardinal slot at the max gap. By construction (offset >=
    # leader_half + cap_half) this never overlaps the leader. Unlike the
    # old ``(MAX/sqrt(2), MAX/sqrt(2))`` fallback, this scales with the
    # actual leader size instead of landing inside a large IC.
    return (leader_w / 2 + cap_w / 2 + _FAN_SPACINGS[-1], 0.0)


@dataclass
class Macro:
    """Leader + rigidly-attached followers.

    Followers' positions are derived: ``follower_offsets`` are in
    leader-local coords at leader rotation=0, and never change. Call
    ``apply_offsets()`` after any leader pose change to refresh
    follower board positions.

    ``is_fixed`` marks the macro as immovable — the legalizer and SA
    skip fixed macros when choosing a mover. Used for connector macros
    that have already been placed on the perimeter: the legalizer
    treats them as fixed obstacles so interior macros get pushed out
    of the connector zone instead of overlapping them.
    """

    leader: "Component"
    followers: list["Component"] = field(default_factory=list)
    follower_offsets: list[tuple[float, float]] = field(default_factory=list)
    is_fixed: bool = False

    @classmethod
    def alone(cls, leader: "Component") -> "Macro":
        return cls(leader=leader)

    @classmethod
    def with_caps(
        cls,
        leader: "Component",
        caps: list["Component"],
        others: Iterable["Component"] = (),
    ) -> "Macro":
        """Build a macro, placing each cap in an overlap-free fan slot.

        After each cap's offset is chosen, ``apply_offsets()`` is called
        so the cap's board position reflects the chosen offset. This
        keeps subsequent cap searches (which check overlap against the
        already-placed followers) accurate.
        """
        macro = cls(leader=leader)
        occupied = list(others)
        for cap in caps:
            offset = find_cap_offset(leader, cap, occupied)
            macro.followers.append(cap)
            macro.follower_offsets.append(offset)
            macro.apply_offsets()
            occupied.append(cap)
        return macro

    def apply_offsets(self) -> None:
        """Recompute follower positions from leader pose + offsets."""
        for cap, (dx, dy) in zip(self.followers, self.follower_offsets):
            rdx, rdy = _rotate(dx, dy, self.leader.rotation)
            cap.x = self.leader.x + rdx
            cap.y = self.leader.y + rdy
            # Cap rotation is independent — decoupling caps don't rotate with IC.

    @property
    def members(self) -> list["Component"]:
        return [self.leader, *self.followers]

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        x_min = y_min = math.inf
        x_max = y_max = -math.inf
        for c in self.members:
            x1, y1, x2, y2 = c.bbox
            if x1 < x_min:
                x_min = x1
            if y1 < y_min:
                y_min = y1
            if x2 > x_max:
                x_max = x2
            if y2 > y_max:
                y_max = y2
        return (x_min, y_min, x_max, y_max)

    def _within_bounds(
        self,
        bounds: tuple[float, float, float, float],
    ) -> bool:
        x_min, y_min, x_max, y_max = bounds
        for c in self.members:
            cx1, cy1, cx2, cy2 = c.bbox
            if cx1 < x_min or cy1 < y_min or cx2 > x_max or cy2 > y_max:
                return False
        return True

    def _snapshot(self) -> list[tuple[float, float, float]]:
        return [(c.x, c.y, c.rotation) for c in self.members]

    def _restore(self, snap: list[tuple[float, float, float]]) -> None:
        for c, (x, y, r) in zip(self.members, snap):
            c.x = x
            c.y = y
            c.set_rotation(r)

    def translate(
        self,
        dx: float,
        dy: float,
        bounds: Optional[tuple[float, float, float, float]] = None,
    ) -> bool:
        """Rigidly translate leader + followers by (dx, dy).

        Returns True if applied. If any member's bbox would cross
        ``bounds``, the move is reverted and False is returned.

        Fixed macros refuse to move — returns False immediately. This
        is a defensive guard: callers (SA, legalizer) should already
        exclude fixed macros from the candidate pool, but a fixed
        macro slipping through (e.g. SA picking one at random when
        fixed obstacles are in the macros list for cost-evaluation
        purposes) shouldn't silently corrupt placement.
        """
        if self.is_fixed:
            return False
        snap = self._snapshot()
        self.leader.x += dx
        self.leader.y += dy
        self.apply_offsets()
        if bounds is not None and not self._within_bounds(bounds):
            self._restore(snap)
            return False
        return True

    def set_pose(
        self,
        x: float,
        y: float,
        rotation: float,
        bounds: Optional[tuple[float, float, float, float]] = None,
    ) -> bool:
        """Set leader pose; followers rotate around leader center.

        Fixed macros refuse to move — returns False immediately. Same
        defensive guard as ``translate``.
        """
        if self.is_fixed:
            return False
        snap = self._snapshot()
        self.leader.x = x
        self.leader.y = y
        self.leader.set_rotation(rotation)
        self.apply_offsets()
        if bounds is not None and not self._within_bounds(bounds):
            self._restore(snap)
            return False
        return True

    def overlaps(self, other: "Macro") -> bool:
        """Macro-macro bbox overlap test.

        Mounting-hole / mechanical-feature exemption: when BOTH macros
        are non-electrical mechanical features (mounting holes, fiducials,
        test coupons), their bboxes may legitimately overlap because
        KiCad places these as pad stacks with via rings around a drill —
        adjacent M3 mounting holes on 8mm spacing have 6.5mm pads whose
        bboxes (with bbox_margin=0.8) are 8.1mm wide, producing a 0.2mm
        false overlap that NO legalizer pass can resolve (they're fixed).
        Reporting these as placement overlaps is a false positive that
        obscures the real residual-overlap signal. See the discussion of
        ``component_type == "mounting_hole"`` in ``parsers/kicad_parser.py``.
        """
        if _is_overlap_exempt(self, other):
            return False
        ax1, ay1, ax2, ay2 = self.bbox
        bx1, by1, bx2, by2 = other.bbox
        return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)

    def overlap_area(self, other: "Macro") -> float:
        """Intersection area of two macros' bboxes.

        Applies the same mechanical-feature exemption as ``overlaps``:
        exempt pairs contribute 0 to the overlap area (so SA's overlap
        cost isn't polluted by mounting-hole pad stacks).
        """
        if _is_overlap_exempt(self, other):
            return 0.0
        ax1, ay1, ax2, ay2 = self.bbox
        bx1, by1, bx2, by2 = other.bbox
        ox1 = max(ax1, bx1)
        oy1 = max(ay1, by1)
        ox2 = min(ax2, bx2)
        oy2 = min(ay2, by2)
        if ox2 <= ox1 or oy2 <= oy1:
            return 0.0
        return (ox2 - ox1) * (oy2 - oy1)

    def contains(self, comp: "Component") -> bool:
        return comp is self.leader or comp in self.followers
