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
        if self._dirty:
            self._recompute_cache()
        if other._dirty:
            other._recompute_cache()
        ax1, ay1, ax2, ay2 = self._cached_bbox
        bx1, by1, bx2, by2 = other._cached_bbox
        return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)

    def overlap_area(self, other: Component) -> float:
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
    """Rectangular board outline."""
    x_min: float = 0.0
    y_min: float = 0.0
    x_max: float = 100.0
    y_max: float = 100.0

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def contains_bbox(self, bbox: tuple[float, float, float, float]) -> bool:
        """Return True if the entire bounding box is inside the board outline."""
        x_min, y_min, x_max, y_max = bbox
        return (x_min >= self.x_min and x_max <= self.x_max and
                y_min >= self.y_min and y_max <= self.y_max)

    def clamp(self, x: float, y: float) -> tuple[float, float]:
        """Clamp coordinates to be within the board outline."""
        return (
            max(self.x_min, min(x, self.x_max)),
            max(self.y_min, min(y, self.y_max)),
        )


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
        board = BoardOutline(
            x_min=board_data.get("x_min", 0.0),
            y_min=board_data.get("y_min", 0.0),
            x_max=board_data.get("x_max", 100.0),
            y_max=board_data.get("y_max", 100.0),
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
            keepouts.append(BoardOutline(
                x_min=kd.get("x_min", 0.0),
                y_min=kd.get("y_min", 0.0),
                x_max=kd.get("x_max", 0.0),
                y_max=kd.get("y_max", 0.0),
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
