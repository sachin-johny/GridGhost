"""Placement application layer — writes optimized positions back to .kicad_pcb.

Maps optimized coordinates back to the KiCad PCB file format.
All changes are applied in one pass.

Coordinate system:
- KiCad stores coordinates in millimeters (floating-point) in the file
- Internal representation is already in millimeters
- Precision: Round to 6 decimal places (1 micrometer precision)
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Optional

from models.board_model import BoardModel, Component


def format_kicad_coord(mm: float) -> str:
    """Format a coordinate value for KiCad PCB file (millimeters as float).
    
    KiCad stores coordinates as floating-point millimeters.
    Round to 6 decimal places for micrometer precision.
    """
    # Round to 6 decimal places and convert to string
    # Remove trailing zeros after decimal point
    formatted = f"{round(mm, 6):.6f}".rstrip('0').rstrip('.')
    return formatted


def apply_placement(
    model: BoardModel,
    input_pcb_path: str,
    output_pcb_path: Optional[str] = None,
    backup: bool = True,
) -> str:
    """Apply optimized component positions to a .kicad_pcb file.

    Args:
        model: Board model with optimized positions
        input_pcb_path: Path to the original .kicad_pcb file
        output_pcb_path: Path for the output file. If None, overwrites input.
        backup: Create a .bak backup before modifying

    Returns:
        Path to the output file
    """
    input_path = Path(input_pcb_path)
    if not input_path.exists():
        raise FileNotFoundError(f"PCB file not found: {input_pcb_path}")

    output_path = Path(output_pcb_path) if output_pcb_path else input_path

    # Create backup
    if backup and output_path.exists():
        backup_path = output_path.with_suffix(".kicad_pcb.bak")
        shutil.copy2(output_path, backup_path)

    # Build reference → component mapping
    comp_map = {c.ref: c for c in model.components}

    # Read the file
    text = input_path.read_text(encoding="utf-8")

    # Apply position updates using regex replacement
    text = _apply_footprint_positions(text, comp_map)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")

    return str(output_path)


def _apply_footprint_positions(text: str, comp_map: dict[str, Component]) -> str:
    """Update footprint positions in the .kicad_pcb file text.

    Uses regex to find footprint blocks and update their (at x y rotation) fields.
    """
    # Pattern to find footprints with their reference and position
    # We'll process each footprint block individually

    result = []
    pos = 0
    text_len = len(text)

    while pos < text_len:
        # Find the start of a footprint or module
        fp_match = re.search(
            r'\(\s*(footprint|module)\s+',
            text[pos:]
        )
        if not fp_match:
            result.append(text[pos:])
            break

        # Add text before this footprint
        result.append(text[pos:pos + fp_match.start()])

        # Find the end of this footprint block
        fp_start = pos + fp_match.start()
        fp_end = _find_matching_paren(text, fp_start)

        if fp_end == -1:
            result.append(text[pos:])
            break

        fp_block = text[fp_start:fp_end + 1]

        # Extract reference from this block
        ref = _extract_reference(fp_block)

        if ref and ref in comp_map:
            comp = comp_map[ref]
            if not comp.is_fixed or True:  # Apply even to "fixed" if user wants
                fp_block = _update_position(fp_block, comp)

        result.append(fp_block)
        pos = fp_end + 1

    return "".join(result)


def _find_matching_paren(text: str, start: int) -> int:
    """Find the closing parenthesis matching the opening one at `start`."""
    depth = 0
    i = start
    in_string = False

    while i < len(text):
        c = text[i]
        if c == '"' and (i == 0 or text[i - 1] != '\\'):
            in_string = not in_string
        elif not in_string:
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return -1


def _extract_reference(fp_block: str) -> Optional[str]:
    """Extract the reference designator from a footprint block."""
    # Try property Reference first (KiCad 7+)
    prop_match = re.search(
        r'\(\s*property\s+"Reference"\s+"([^"]+)"',
        fp_block
    )
    if prop_match:
        return prop_match.group(1)

    # Try fp_text reference (older format)
    text_match = re.search(
        r'\(\s*fp_text\s+reference\s+([^\s)]+)',
        fp_block
    )
    if text_match:
        return text_match.group(1)

    return None


def _update_position(fp_block: str, comp: Component) -> str:
    """Update the (at x y [rotation]) field in a footprint block."""
    x_str = format_kicad_coord(comp.x)
    y_str = format_kicad_coord(comp.y)

    # Find and replace the (at ...) at the footprint level (not pad level)
    # The footprint-level (at ...) comes before any (pad ...) expressions

    # Strategy: find the first (at ...) that's at the top level of the footprint
    # (i.e., not nested inside a pad or fp_text)

    # For simplicity, we replace the first (at X Y [R]) in the footprint
    # header (before any pad definitions)

    # Split at first (pad to isolate header
    pad_split = re.split(r'\(\s*pad\s', fp_block, maxsplit=1)
    header = pad_split[0]

    # Build new (at ...) expression
    if comp.rotation != 0.0:
        new_at = f"(at {x_str} {y_str} {round(comp.rotation)})"
    else:
        new_at = f"(at {x_str} {y_str})"

    # Replace the (at ...) in the header
    # Match the footprint-level at expression
    header = re.sub(
        r'\(\s*at\s+[-\d.]+\s+[-\d.]+(?:\s+[-\d.]+)?\s*\)',
        new_at,
        header,
        count=1
    )

    # Reconstruct the block
    if len(pad_split) > 1:
        if comp.rotation != 0.0:
            pad_section = _add_pad_rotation(pad_split[1], round(comp.rotation))
            return header + "(pad " + pad_section
        return header + "(pad " + pad_split[1]
    return header


def _add_pad_rotation(pad_section: str, rotation: int) -> str:
    """Add explicit rotation to pad (at ...) expressions.

    When a footprint is rotated, KiCad applies the rotation to courtyard/
    silkscreen but not to pads. This function sets the rotation on each pad's
    (at X Y [R]) so copper layers match the courtyard orientation.

    Pads that already have an explicit rotation (3-arg at) get it replaced.
    Pads with 2-arg (at X Y) get the rotation appended.
    """
    def _replace_at(m: re.Match) -> str:
        x, y = m.group(1), m.group(2)
        return f"(at {x} {y} {rotation})"

    # Match (at X Y) or (at X Y R) at pad-body indentation level.
    # Pad (at ...) is always a direct child of the pad block, typically
    # indented with 3 tabs. We match any (at ...) with only numeric args —
    # nested property (at ...) would have a string arg (e.g. in fp_* primitives)
    # which this regex excludes.
    return re.sub(
        r'\(\s*at\s+([-\d.]+)\s+([-\d.]+)(?:\s+[-\d.]+)?\s*\)',
        _replace_at,
        pad_section,
    )


def export_positions_json(model: BoardModel, output_path: str) -> str:
    """Export component positions as a simple JSON for verification.

    This is useful for debugging and for integration with other tools.
    """
    import json
    positions = {}
    for comp in model.components:
        positions[comp.ref] = {
            "x": round(comp.x, 4),
            "y": round(comp.y, 4),
            "rotation": comp.rotation,
            "layer": comp.layer,
        }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(positions, indent=2), encoding="utf-8")
    return str(output)


def write_debug_bboxes(model: BoardModel, pcb_path: str) -> str:
    """Inject gr_rect bounding-box visuals on Dwgs.User layer for debugging.

    Each component gets a rectangle matching its bbox property and a small
    label with the ref designator so you can identify it in KiCad.
    """
    import uuid

    pcb = Path(pcb_path)
    text = pcb.read_text(encoding="utf-8")

    # KiCad uses mm coordinates as floats
    segments = []
    for comp in model.components:
        bx1, by1, bx2, by2 = comp.bbox
        x1 = format_kicad_coord(bx1)
        y1 = format_kicad_coord(by1)
        x2 = format_kicad_coord(bx2)
        y2 = format_kicad_coord(by2)
        rect_uuid = str(uuid.uuid4())
        segments.append(
            f'  (gr_rect (start {x1} {y1}) (end {x2} {y2}) '
            f'(stroke (width 0.1) (type solid)) (fill none) (layer "Dwgs.User") '
            f'(tstamp "{rect_uuid}"))'
        )
        # Label at bottom-left corner
        lx = format_kicad_coord(bx1)
        ly = format_kicad_coord(by1 - 0.3)
        text_uuid = str(uuid.uuid4())
        segments.append(
            f'  (gr_text "{comp.ref}" (at {lx} {ly}) '
            f'(layer "Dwgs.User") (tstamp "{text_uuid}") '
            f'(effects (font (size 0.8 0.8) (thickness 0.12))))'
        )

    debug_block = "\n".join(segments) + "\n"

    # Insert before the final ')' of the kicad_pcb top-level expression.
    # The file structure is: (kicad_pcb ... ) — we need our gr_* items
    # inside that closing paren, otherwise KiCad ignores them.
    text = text.rstrip()
    # Find the last ')' that closes the root (kicad_pcb ...) expression
    idx = text.rfind(')')
    if idx < 0:
        pcb.write_text(text + "\n" + debug_block, encoding="utf-8")
        return str(pcb)
    text = text[:idx] + debug_block + text[idx:]

    pcb.write_text(text, encoding="utf-8")
    return str(pcb)
