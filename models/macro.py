"""Macro: a leader component plus rigidly-attached followers.

A Macro is the atomic unit of placement. The SA moves a Macro as one
rigid body (translation + rotation); the legalizer treats a Macro as
a single rectangle for overlap checks. Followers (typically decoupling
caps) have a fixed offset in leader-local coords; the leader's current
rotation is applied to those offsets to compute follower positions.

HARD RULE: a follower's center-to-leader-center distance is fixed at
macro construction (computed from the chosen fan-slot offset). Because
followers move rigidly with the leader, that distance can NEVER grow
at runtime. The MAX_CAP_IC_DISTANCE_MM constant is the upper bound
the construction-time fan search respects; the legalizer's alternate-
slot search (Commit 2) also respects it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from models.board_model import Component


MAX_CAP_IC_DISTANCE_MM = 8.0

_FAN_SPACINGS = (0.5, 1.0, 2.0, 3.0, 4.0)
_FAN_DIRS = (
    (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
    (1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0),
)


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
    spacings; the first slot within MAX_CAP_IC_DISTANCE_MM that
    doesn't overlap the leader or any component in ``others`` is
    returned.

    ``others`` is typically the macro's already-placed followers — we
    want the new cap to avoid overlapping its sibling caps. We do NOT
    check against other components in the model because at macro-
    construction time everything is still at parse positions and the
    "overlap-free" check would almost always fail, producing a >8mm
    fallback offset. The legalizer's push-apart will resolve macro-
    vs-macro overlaps after SA.

    If no slot within MAX_CAP_IC_DISTANCE_MM is free of siblings,
    falls back to the largest diagonal spacing within the limit; this
    keeps the cap near its IC even at the cost of a sibling overlap.
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
            for dx_dir, dy_dir in _FAN_DIRS:
                offset_x = dx_dir * (leader_w / 2 + cap_w / 2 + spacing)
                offset_y = dy_dir * (leader_h / 2 + cap_h / 2 + spacing)

                # Reject offsets exceeding the hard cap-IC distance.
                dist = math.hypot(offset_x, offset_y)
                if dist > MAX_CAP_IC_DISTANCE_MM:
                    continue

                cap_x = leader.x + offset_x * cos_r - offset_y * sin_r
                cap_y = leader.y + offset_x * sin_r + offset_y * cos_r
                cap.x = cap_x
                cap.y = cap_y

                # Reject slots where cap overlaps the leader's courtyard.
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

    # Absolute last resort: a diagonal slot just under the limit.
    s = MAX_CAP_IC_DISTANCE_MM / math.sqrt(2)
    return (s, s)


@dataclass
class Macro:
    """Leader + rigidly-attached followers.

    Followers' positions are derived: ``follower_offsets`` are in
    leader-local coords at leader rotation=0, and never change. Call
    ``apply_offsets()`` after any leader pose change to refresh
    follower board positions.
    """

    leader: "Component"
    followers: list["Component"] = field(default_factory=list)
    follower_offsets: list[tuple[float, float]] = field(default_factory=list)

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
        """
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
        """Set leader pose; followers rotate around leader center."""
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
        """Macro-macro bbox overlap test."""
        ax1, ay1, ax2, ay2 = self.bbox
        bx1, by1, bx2, by2 = other.bbox
        return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)

    def overlap_area(self, other: "Macro") -> float:
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
