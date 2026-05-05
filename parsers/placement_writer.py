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
        new_at = f"(at {x_str} {y_str} {int(comp.rotation)})"
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
        return header + "(pad " + pad_split[1]
    return header


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
