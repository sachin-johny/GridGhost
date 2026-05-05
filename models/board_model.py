"""Core data model for PCB placement.

Defines the intermediate representation used between KiCad extraction
and the optimization engine. All coordinates are in millimeters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


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
        cos_r, sin_r = math.cos(rad), math.sin(rad)
        abs_x = comp_x + self.x * cos_r - self.y * sin_r
        abs_y = comp_y + self.x * sin_r + self.y * cos_r
        return abs_x, abs_y


@dataclass
class Component:
    """A placed or unplaced component on the board."""
    ref: str                    # Reference designator (e.g. "U1", "R3")
    footprint: str = ""        # Footprint library ID (e.g. "Package_QFP:LQFP-48")
    value: str = ""            # Component value (e.g. "STM32F103C8T6")
    x: float = 0.0            # Position X in mm
    y: float = 0.0            # Position Y in mm
    rotation: float = 0.0     # Rotation in degrees (discrete: 0, 90, 180, 270)
    layer: str = "top"        # "top" or "bottom"
    width: float = 0.0        # Bounding box width in mm
    height: float = 0.0       # Bounding box height in mm
    courtyard_margin: float = 0.25  # Courtyard margin in mm (default 0.25mm per KiCad convention)
    bbox_offset_x: float = 0.0  # Offset from footprint origin to bbox center (mm)
    bbox_offset_y: float = 0.0  # Offset from footprint origin to bbox center (mm)
    pads: list[Pad] = field(default_factory=list)
    nets: list[str] = field(default_factory=list)  # Net names connected to this component
    is_fixed: bool = False    # If True, position should not be changed by optimizer
    component_type: str = "generic"  # "ic", "capacitor", "resistor", "connector", "crystal", "generic"

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Return bounding box (x_min, y_min, x_max, y_max) including courtyard.

        Uses bbox_offset to account for footprints whose origin is not
        at the center of their geometry (common for connectors, displays, etc.).
        """
        cx = self.x + self.bbox_offset_x
        cy = self.y + self.bbox_offset_y
        half_w = (self.width / 2.0) + self.courtyard_margin
        half_h = (self.height / 2.0) + self.courtyard_margin
        return (
            cx - half_w,
            cy - half_h,
            cx + half_w,
            cy + half_h,
        )

    @property
    def effective_width(self) -> float:
        """Width including courtyard margin."""
        return self.width + 2 * self.courtyard_margin

    @property
    def effective_height(self) -> float:
        """Height including courtyard margin."""
        return self.height + 2 * self.courtyard_margin

    def overlaps(self, other: Component) -> bool:
        """Check if this component's courtyard overlaps with another."""
        ax1, ay1, ax2, ay2 = self.bbox
        bx1, by1, bx2, by2 = other.bbox
        return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)

    def overlap_area(self, other: Component) -> float:
        """Compute the overlap area between this component and another."""
        ax1, ay1, ax2, ay2 = self.bbox
        bx1, by1, bx2, by2 = other.bbox
        ox1 = max(ax1, bx1)
        oy1 = max(ay1, by1)
        ox2 = min(ax2, bx2)
        oy2 = min(ay2, by2)
        if ox2 <= ox1 or oy2 <= oy1:
            return 0.0
        return (ox2 - ox1) * (oy2 - oy1)


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

    # ---- Lookup helpers ----

    def get_component(self, ref: str) -> Optional[Component]:
        for c in self.components:
            if c.ref == ref:
                return c
        return None

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

        return cls(
            board=board,
            components=components,
            nets=nets,
            source_file=data.get("source_file", ""),
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

        # Out-of-bounds
        oob_count = sum(
            1 for c in self.components
            if not self.board.contains(c.x, c.y)
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
