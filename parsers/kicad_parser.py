"""KiCad .kicad_pcb s-expression parser.

Parses the KiCad PCB file format (s-expressions) to extract:
- Board outline (gr_rect or gr_poly on Edge.Cuts)
- Footprints with positions, rotations, bounding boxes
- Pads and their net assignments
- Net definitions

Reference: https://dev-docs.kicad.org/en/file-formats/sexpr-pcb/
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional

from models.board_model import BoardModel, BoardOutline, Component, Net, Pad


# ---------------------------------------------------------------------------
# S-expression tokenizer / parser
# ---------------------------------------------------------------------------

def tokenize_sexp(text: str) -> list:
    """Tokenize an s-expression string into a nested list structure.

    Example:
        "(module (at 10 20) (fp_text reference U1))"
        → ["module", ["at", "10", "20"], ["fp_text", "reference", "U1"]]
    """
    tokens = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c in ' \t\n\r':
            i += 1
        elif c == '(':
            tokens.append('(')
            i += 1
        elif c == ')':
            tokens.append(')')
            i += 1
        elif c == '"':
            # Quoted string
            j = i + 1
            while j < n and text[j] != '"':
                if text[j] == '\\':
                    j += 1  # skip escaped char
                j += 1
            tokens.append(text[i + 1:j])
            i = j + 1
        else:
            # Unquoted token
            j = i
            while j < n and text[j] not in ' \t\n\r()':
                j += 1
            tokens.append(text[i:j])
            i = j
    return tokens


def parse_sexp(tokens: list) -> list:
    """Convert flat token list into nested list structure."""
    result = []
    stack = [result]
    for token in tokens:
        if token == '(':
            new_list = []
            stack[-1].append(new_list)
            stack.append(new_list)
        elif token == ')':
            stack.pop()
        else:
            stack[-1].append(token)
    return result


# ---------------------------------------------------------------------------
# Helper extractors
# ---------------------------------------------------------------------------

def find_all(sexp: list, key: str) -> list[list]:
    """Find all sub-expressions starting with `key`."""
    results = []
    for item in sexp:
        if isinstance(item, list) and len(item) > 0 and item[0] == key:
            results.append(item)
    return results


def find_first(sexp: list, key: str) -> Optional[list]:
    """Find first sub-expression starting with `key`."""
    for item in sexp:
        if isinstance(item, list) and len(item) > 0 and item[0] == key:
            return item
    return None


def find_deep(sexp: list, key: str) -> list[list]:
    """Find all sub-expressions at any depth starting with `key`."""
    results = []
    for item in sexp:
        if isinstance(item, list):
            if len(item) > 0 and item[0] == key:
                results.append(item)
            results.extend(find_deep(item, key))
    return results


def try_float(val, default=0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Bounding box estimation from fp_line / fp_rect / fp_circle / pad geometry
# ---------------------------------------------------------------------------

def _extract_fp_geometry(sexp: list, margin_mm: float = 0.5) -> tuple[float, float, float, float]:
    """Extract component bounding box from footprint geometry.

    Collects the outermost extent across ALL relevant KiCad layers
    (CrtYd, Fab, SilkS, Paste, Cu, Mask) plus actual pad sizes.

    Returns: (width, height, bbox_offset_x, bbox_offset_y)
        - width/height: total extent including margin
        - bbox_offset_x/y: offset from footprint origin to geometry center
    """
    RELEVANT = ("CrtYd", "Fab", "Silk", "Paste", ".Cu", "Mask")
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    # Collect geometry from fp_line, fp_rect, fp_circle, fp_arc, fp_poly
    geom_items = (find_deep(sexp, "fp_line") + find_deep(sexp, "fp_rect") +
                  find_deep(sexp, "fp_circle") + find_deep(sexp, "fp_arc") +
                  find_deep(sexp, "fp_poly"))
    for item in geom_items:
        layer = find_first(item, "layer")
        if not layer or len(layer) < 2:
            continue
        layer_str = str(layer[1])
        if not any(kw in layer_str for kw in RELEVANT):
            continue
        for pt_key in ("start", "end", "center"):
            pt = find_first(item, pt_key)
            if pt and len(pt) >= 3:
                v1, v2 = try_float(pt[1]), try_float(pt[2])
                min_x, min_y = min(min_x, v1), min(min_y, v2)
                max_x, max_y = max(max_x, v1), max(max_y, v2)

    # Collect actual pad sizes (not hardcoded 0.5mm)
    for pad_expr in find_all(sexp, "pad"):
        at_expr = find_first(pad_expr, "at")
        if not (at_expr and len(at_expr) >= 3):
            continue
        px, py = try_float(at_expr[1]), try_float(at_expr[2])
        size_expr = find_first(pad_expr, "size")
        sx = sy = 0.5
        if size_expr and len(size_expr) >= 3:
            sx, sy = try_float(size_expr[1]) / 2.0, try_float(size_expr[2]) / 2.0
        min_x, min_y = min(min_x, px - sx), min(min_y, py - sy)
        max_x, max_y = max(max_x, px + sx), max(max_y, py + sy)

    if min_x == float('inf'):
        return 2.0, 2.0, 0.0, 0.0

    w = max(max_x - min_x, 0.5)
    h = max(max_y - min_y, 0.5)
    # Offset from footprint origin to geometry center
    off_x = (min_x + max_x) / 2.0
    off_y = (min_y + max_y) / 2.0
    return w, h, off_x, off_y


# ---------------------------------------------------------------------------
# Component type inference
# ---------------------------------------------------------------------------

def _infer_component_type(ref: str, footprint: str, value: str) -> str:
    """Infer component type from reference prefix and footprint name."""
    ref_prefix = re.match(r'^([A-Z]+)', ref)
    prefix = ref_prefix.group(1) if ref_prefix else ""

    type_map = {
        "U": "ic",
        "IC": "ic",
        "J": "connector",
        "P": "connector",
        "CN": "connector",
        "Y": "crystal",
        "X": "crystal",
    }
    if prefix in type_map:
        return type_map[prefix]

    # Capacitors and resistors by reference prefix
    if prefix in ("C",):
        return "capacitor"
    if prefix in ("R",):
        return "resistor"

    # Check footprint hints
    fp_lower = footprint.lower()
    if "qfp" in fp_lower or "qfn" in fp_lower or "bga" in fp_lower or "sop" in fp_lower or "soic" in fp_lower:
        return "ic"
    if "crystal" in fp_lower or "xtal" in fp_lower:
        return "crystal"
    if "connector" in fp_lower or "hdr" in fp_lower or "usb" in fp_lower:
        return "connector"

    return "generic"


# ---------------------------------------------------------------------------
# Board outline extraction
# ---------------------------------------------------------------------------

def _extract_board_outline(sexp: list) -> BoardOutline:
    """Extract board outline from Edge.Cuts geometry."""
    points_x = []
    points_y = []

    # gr_rect on Edge.Cuts
    for gr_rect in find_all(sexp, "gr_rect"):
        layer = find_first(gr_rect, "layer")
        if layer and len(layer) > 1 and "Edge.Cuts" in str(layer[1]):
            start_pt = find_first(gr_rect, "start")
            end_pt = find_first(gr_rect, "end")
            if start_pt and len(start_pt) >= 3 and end_pt and len(end_pt) >= 3:
                points_x.extend([try_float(start_pt[1]), try_float(end_pt[1])])
                points_y.extend([try_float(start_pt[2]), try_float(end_pt[2])])

    # gr_line on Edge.Cuts
    for gr_line in find_all(sexp, "gr_line"):
        layer = find_first(gr_line, "layer")
        if layer and len(layer) > 1 and "Edge.Cuts" in str(layer[1]):
            start_pt = find_first(gr_line, "start")
            end_pt = find_first(gr_line, "end")
            if start_pt and len(start_pt) >= 3:
                points_x.append(try_float(start_pt[1]))
                points_y.append(try_float(start_pt[2]))
            if end_pt and len(end_pt) >= 3:
                points_x.append(try_float(end_pt[1]))
                points_y.append(try_float(end_pt[2]))

    # gr_poly on Edge.Cuts
    for gr_poly in find_all(sexp, "gr_poly"):
        layer = find_first(gr_poly, "layer")
        if layer and len(layer) > 1 and "Edge.Cuts" in str(layer[1]):
            pts_expr = find_first(gr_poly, "pts")
            if pts_expr:
                for xy in find_all(pts_expr, "xy"):
                    if len(xy) >= 3:
                        points_x.append(try_float(xy[1]))
                        points_y.append(try_float(xy[2]))

    # gr_circle on Edge.Cuts (approximate as bounding box)
    for gr_circle in find_all(sexp, "gr_circle"):
        layer = find_first(gr_circle, "layer")
        if layer and len(layer) > 1 and "Edge.Cuts" in str(layer[1]):
            center = find_first(gr_circle, "center")
            end = find_first(gr_circle, "end")
            if center and len(center) >= 3 and end and len(end) >= 3:
                cx, cy = try_float(center[1]), try_float(center[2])
                ex, ey = try_float(end[1]), try_float(end[2])
                r = math.sqrt((ex - cx) ** 2 + (ey - cy) ** 2)
                points_x.extend([cx - r, cx + r])
                points_y.extend([cy - r, cy + r])

    if not points_x:
        # Default board if no outline found
        return BoardOutline(x_min=0.0, y_min=0.0, x_max=100.0, y_max=100.0)

    margin = 0.5  # Small margin
    return BoardOutline(
        x_min=min(points_x) - margin,
        y_min=min(points_y) - margin,
        x_max=max(points_x) + margin,
        y_max=max(points_y) + margin,
    )


def _infer_board_from_components(components: list[Component]) -> BoardOutline:
    """Infer a board outline from placed components when Edge.Cuts is absent.

    This keeps the component cluster centered instead of forcing it into the
    top-left corner of the default 100x100 mm fallback board.
    """
    if not components:
        return BoardOutline(x_min=0.0, y_min=0.0, x_max=100.0, y_max=100.0)

    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')

    for comp in components:
        half_w = comp.effective_width / 2.0
        half_h = comp.effective_height / 2.0
        min_x = min(min_x, comp.x - half_w)
        min_y = min(min_y, comp.y - half_h)
        max_x = max(max_x, comp.x + half_w)
        max_y = max(max_y, comp.y + half_h)

    padding = max(5.0, max(max_x - min_x, max_y - min_y) * 0.15)
    return BoardOutline(
        x_min=min_x - padding,
        y_min=min_y - padding,
        x_max=max_x + padding,
        y_max=max_y + padding,
    )


def _needs_inferred_board(board_outline: BoardOutline, components: list[Component]) -> bool:
    """Return True when the parsed outline is just the default fallback board."""
    if not components:
        return False

    is_default_board = (
        math.isclose(board_outline.x_min, 0.0)
        and math.isclose(board_outline.y_min, 0.0)
        and math.isclose(board_outline.x_max, 100.0)
        and math.isclose(board_outline.y_max, 100.0)
    )
    if not is_default_board:
        return False

    for comp in components:
        bx1, by1, bx2, by2 = comp.bbox
        if bx1 < board_outline.x_min or by1 < board_outline.y_min or bx2 > board_outline.x_max or by2 > board_outline.y_max:
            return True

    return False


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

class KiCadParser:
    """Parser for .kicad_pcb files."""

    def __init__(self, filepath: str, bbox_margin: float = 0.5):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"PCB file not found: {filepath}")
        self._sexp = None
        self._net_id_to_name: dict[str, str] = {}  # Maps net ID ("1") → name ("VCC")
        self._bbox_margin = bbox_margin  # Margin around bbox in mm

    def parse(self) -> BoardModel:
        """Parse the .kicad_pcb file and return a BoardModel."""
        text = self.filepath.read_text(encoding="utf-8")
        tokens = tokenize_sexp(text)
        parsed = parse_sexp(tokens)
        # The root is the `kicad_pcb` expression
        self._sexp = parsed[0] if parsed else parsed
        sexp = self._sexp or []

        # Build net ID → name mapping from top-level net definitions
        self._build_net_id_map()

        board_outline = _extract_board_outline(sexp)
        components = self._extract_components()
        nets = self._extract_nets()

        # Link nets to components
        net_map: dict[str, Net] = {n.name: n for n in nets}
        for comp in components:
            comp_nets = set()
            for pad in comp.pads:
                if pad.net and pad.net in net_map:
                    comp_nets.add(pad.net)
            comp.nets = sorted(comp_nets)

        if _needs_inferred_board(board_outline, components):
            board_outline = _infer_board_from_components(components)

        model = BoardModel(
            board=board_outline,
            components=components,
            nets=nets,
            source_file=str(self.filepath),
        )
        return model

    def _build_net_id_map(self) -> None:
        """Build mapping from net IDs to net names from top-level (net <id> <name>) definitions."""
        sexp = self._sexp or []
        for net_expr in find_all(sexp, "net"):
            if len(net_expr) >= 3:
                net_id = net_expr[1]
                net_name = net_expr[2]
                self._net_id_to_name[net_id] = net_name

    def _extract_components(self) -> list[Component]:
        """Extract all footprints as Components."""
        components = []
        sexp = self._sexp or []

        for fp_expr in find_all(sexp, "footprint"):
            # Also handle "module" for older KiCad formats
            self._parse_footprint(fp_expr, components)

        for fp_expr in find_all(sexp, "module"):
            self._parse_footprint(fp_expr, components)

        return components

    def _parse_footprint(self, fp_expr: list, components: list[Component]) -> None:
        """Parse a single footprint/module expression into a Component."""
        # Library ID
        lib_id = fp_expr[1] if len(fp_expr) > 1 else ""

        # Position and rotation
        at_expr = find_first(fp_expr, "at")
        if at_expr and len(at_expr) >= 3:
            x = try_float(at_expr[1])
            y = try_float(at_expr[2])
            rotation = try_float(at_expr[3]) if len(at_expr) > 3 else 0.0
        else:
            x, y, rotation = 0.0, 0.0, 0.0

        # Layer
        layer_expr = find_first(fp_expr, "layer")
        layer = "bottom" if layer_expr and len(layer_expr) > 1 and "B." in str(layer_expr[1]) else "top"

        # Reference and value
        ref = ""
        value = ""
        for prop in find_all(fp_expr, "property"):
            if len(prop) > 2:
                if prop[1] == "Reference":
                    ref = prop[2]
                elif prop[1] == "Value":
                    value = prop[2]

        # Older format: fp_text reference / fp_text value
        if not ref:
            for ft in find_all(fp_expr, "fp_text"):
                if len(ft) > 3 and ft[1] == "reference":
                    ref = ft[2]
                elif len(ft) > 3 and ft[1] == "value":
                    value = ft[2]

        if not ref:
            return  # Skip unnamed components

        # Pads
        pads = []
        for pad_expr in find_all(fp_expr, "pad"):
            pad = self._parse_pad(pad_expr, x, y, rotation)
            if pad:
                pads.append(pad)

        # Bounding box from geometry (with margin for spacing)
        width, height, bbox_ox, bbox_oy = _extract_fp_geometry(fp_expr, margin_mm=self._bbox_margin)

        # Determine if fixed. Keep connectors movable so the auto-placer can
        # move them to the board perimeter during edge-aware placement.
        comp_type = _infer_component_type(ref, lib_id, value)
        is_fixed = False

        comp = Component(
            ref=ref,
            footprint=lib_id,
            value=value,
            x=x,
            y=y,
            rotation=rotation,
            layer=layer,
            width=width,
            height=height,
            courtyard_margin=self._bbox_margin,
            bbox_offset_x=bbox_ox,
            bbox_offset_y=bbox_oy,
            pads=pads,
            is_fixed=is_fixed,
            component_type=comp_type,
        )
        components.append(comp)

    def _resolve_net_name(self, net_expr: Optional[list]) -> Optional[str]:
        """Resolve a pad's net expression to a net name.

        Handles two formats:
          - (net <id> <name>)  → use name directly
          - (net <id>)         → look up name from _net_id_to_name map
        """
        if not net_expr or len(net_expr) < 2:
            return None

        net_id = net_expr[1]

        # Format: (net <id> <name>) — name is directly available
        if len(net_expr) >= 3:
            # Check if the third element looks like a net name (not a number)
            potential_name = net_expr[2]
            if potential_name and not potential_name.lstrip('-').replace('.', '').isdigit():
                return potential_name

        # Format: (net <id>) — look up by ID
        if net_id in self._net_id_to_name:
            return self._net_id_to_name[net_id]

        # Last resort: if the ID itself is a name (e.g., net names without quotes)
        if net_id and not net_id.lstrip('-').replace('.', '').isdigit():
            return net_id

        return None

    def _parse_pad(self, pad_expr: list, comp_x: float, comp_y: float, rotation: float) -> Optional[Pad]:
        """Parse a pad expression into a Pad object."""
        if len(pad_expr) < 3:
            return None

        pad_name = pad_expr[1]

        # Position (relative to component origin)
        at_expr = find_first(pad_expr, "at")
        if at_expr and len(at_expr) >= 3:
            px = try_float(at_expr[1])
            py = try_float(at_expr[2])
        else:
            px, py = 0.0, 0.0

        # Net — resolve using the ID→name map
        net_expr = find_first(pad_expr, "net")
        net_name = self._resolve_net_name(net_expr)

        return Pad(pad_name=pad_name, x=px, y=py, net=net_name)

    def _extract_nets(self) -> list[Net]:
        """Extract net definitions."""
        nets = []
        sexp = self._sexp or []
        # From (net <id> <name>) in the nets section
        for net_expr in find_all(sexp, "net"):
            if len(net_expr) >= 3 and isinstance(net_expr[1], str) and isinstance(net_expr[2], str):
                net_name = net_expr[2]
                # Collect pins from all footprints
                pins = []
                for comp_ref, pad_name, pad_net in self._collect_net_pins():
                    if pad_net == net_name:
                        pins.append((comp_ref, pad_name))
                nets.append(Net(net_name, pins))

        # If no net section found, build nets from pad net assignments
        if not nets:
            net_dict: dict[str, list[tuple[str, str]]] = {}
            for comp_ref, pad_name, pad_net in self._collect_net_pins():
                if pad_net:
                    net_dict.setdefault(pad_net, []).append((comp_ref, pad_name))
            for name, pins in net_dict.items():
                nets.append(Net(name, pins))

        return nets

    def _collect_net_pins(self) -> list[tuple[str, str, Optional[str]]]:
        """Collect all (comp_ref, pad_name, net_name) tuples from all footprints."""
        result = []
        seen_refs = set()
        sexp = self._sexp or []

        for fp_expr in find_all(sexp, "footprint") + find_all(sexp, "module"):
            # Get reference
            ref = ""
            for prop in find_all(fp_expr, "property"):
                if len(prop) > 2 and prop[1] == "Reference":
                    ref = prop[2]
            if not ref:
                for ft in find_all(fp_expr, "fp_text"):
                    if len(ft) > 3 and ft[1] == "reference":
                        ref = ft[2]

            if not ref or ref in seen_refs:
                continue
            seen_refs.add(ref)

            for pad_expr in find_all(fp_expr, "pad"):
                if len(pad_expr) < 3:
                    continue
                pad_name = pad_expr[1]
                net_expr = find_first(pad_expr, "net")
                net_name = self._resolve_net_name(net_expr)
                result.append((ref, pad_name, net_name))

        return result
