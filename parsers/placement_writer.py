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

    # Split at first (pad to isolate header
    pad_split = re.split(r'\(\s*pad\s', fp_block, maxsplit=1)
    header = pad_split[0]

    # Extract original footprint rotation from the header before replacing
    orig_match = re.search(
        r'\(\s*at\s+[-\d.]+\s+[-\d.]+(?:\s+([-\d.]+))?\s*\)',
        header,
    )
    orig_rotation = float(orig_match.group(1)) if orig_match and orig_match.group(1) else 0.0

    # Compute rotation delta that pads need
    new_rotation = round(comp.rotation)
    rotation_delta = (new_rotation - round(orig_rotation)) % 360

    # Build new (at ...) expression
    if comp.rotation != 0.0:
        new_at = f"(at {x_str} {y_str} {new_rotation})"
    else:
        new_at = f"(at {x_str} {y_str})"

    # Replace the (at ...) in the header
    header = re.sub(
        r'\(\s*at\s+[-\d.]+\s+[-\d.]+(?:\s+[-\d.]+)?\s*\)',
        new_at,
        header,
        count=1
    )

    # Reconstruct the block
    if len(pad_split) > 1:
        if rotation_delta != 0:
            pad_section = _add_pad_rotation(pad_split[1], rotation_delta)
            return header + "(pad " + pad_section
        return header + "(pad " + pad_split[1]
    return header


def _add_pad_rotation(pad_section: str, rotation_delta: int) -> str:
    """Add rotation delta to pad (at ...) expressions.

    When a footprint's rotation changes, KiCad applies the change to courtyard/
    silkscreen but not to pads. This function adjusts each pad's (at X Y [R])
    by adding the rotation delta so copper layers match the courtyard orientation.

    Pads that already have an explicit rotation (3-arg at) get the delta added.
    Pads with 2-arg (at X Y) get the delta appended as a new third arg.
    """
    def _replace_at(m: re.Match) -> str:
        x, y = m.group(1), m.group(2)
        existing_rot = float(m.group(3)) if m.group(3) else 0.0
        new_rot = round((existing_rot + rotation_delta) % 360)
        return f"(at {x} {y} {new_rot})"

    return re.sub(
        r'\(\s*at\s+([-\d.]+)\s+([-\d.]+)(?:\s+([-\d.]+))?\s*\)',
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


def write_debug_bboxes(
    model: BoardModel,
    pcb_path: str,
    interior_bbox: tuple[float, float, float, float] | None = None,
) -> str:
    """Inject gr_rect bounding-box visuals on Dwgs.User layer for debugging.

    Each component gets a rectangle matching its bbox property and a small
    label with the ref designator so you can identify it in KiCad.

    Additionally draws:
    - Board outline rect (dashed, wider stroke) so you can see the
      effective board boundary — especially useful when no Edge.Cuts
      geometry exists and the outline was inferred from components.
    - Interior bbox rect (if provided) showing the connector-free zone.
    """
    import uuid

    pcb = Path(pcb_path)
    text = pcb.read_text(encoding="utf-8")

    # KiCad uses mm coordinates as floats
    segments = []

    # --- Board outline rect ---
    board = model.board
    bx1 = format_kicad_coord(board.x_min)
    by1 = format_kicad_coord(board.y_min)
    bx2 = format_kicad_coord(board.x_max)
    by2 = format_kicad_coord(board.y_max)
    board_uuid = str(uuid.uuid4())
    segments.append(
        f'  (gr_rect (start {bx1} {by1}) (end {bx2} {by2}) '
        f'(stroke (width 0.3) (type dash)) (fill none) (layer "Dwgs.User") '
        f'(tstamp "{board_uuid}"))'
    )
    # Label at top-left corner of board outline
    blx = format_kicad_coord(board.x_min)
    bly = format_kicad_coord(board.y_min - 0.5)
    board_text_uuid = str(uuid.uuid4())
    segments.append(
        f'  (gr_text "BOARD_OUTLINE" (at {blx} {bly}) '
        f'(layer "Dwgs.User") (tstamp "{board_text_uuid}") '
        f'(effects (font (size 1.0 1.0) (thickness 0.15))))'
    )

    # --- Interior bbox rect (if provided) ---
    if interior_bbox is not None:
        ib_x1, ib_y1, ib_x2, ib_y2 = interior_bbox
        ix1 = format_kicad_coord(ib_x1)
        iy1 = format_kicad_coord(ib_y1)
        ix2 = format_kicad_coord(ib_x2)
        iy2 = format_kicad_coord(ib_y2)
        ib_uuid = str(uuid.uuid4())
        segments.append(
            f'  (gr_rect (start {ix1} {iy1}) (end {ix2} {iy2}) '
            f'(stroke (width 0.25) (type dot)) (fill none) (layer "Dwgs.User") '
            f'(tstamp "{ib_uuid}"))'
        )
        ilx = format_kicad_coord(ib_x1)
        ily = format_kicad_coord(ib_y1 - 0.5)
        ib_text_uuid = str(uuid.uuid4())
        segments.append(
            f'  (gr_text "INTERIOR_BBOX" (at {ilx} {ily}) '
            f'(layer "Dwgs.User") (tstamp "{ib_text_uuid}") '
            f'(effects (font (size 1.0 1.0) (thickness 0.15))))'
        )

    # --- Component bboxes ---
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
        # Label at bottom-left corner — include the sheet name (Phase 3.1)
        # so a human reviewing the placement can immediately see which
        # schematic sheet each component came from.  Empty sheet → just ref.
        sheet = getattr(comp, 'sheet', '') or ''
        if sheet and sheet != '/':
            label = f"{comp.ref} [{sheet}]"
        else:
            label = comp.ref
        lx = format_kicad_coord(bx1)
        ly = format_kicad_coord(by1 - 0.3)
        text_uuid = str(uuid.uuid4())
        segments.append(
            f'  (gr_text "{label}" (at {lx} {ly}) '
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
