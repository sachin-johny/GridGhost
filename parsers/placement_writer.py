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

    Edge.Cuts handling (Finding 1 fix):
        When ``model.user_defined_outline`` is False (i.e. the original
        .kicad_pcb had no Edge.Cuts geometry and GridGhost inferred the
        outline from component footprints), the inferred outline is
        serialized as a real ``gr_rect`` on the ``Edge.Cuts`` layer in
        the output file. Without this, 5 of 6 test boards ship with no
        board edge at all — DRC cannot run, no fab house will accept
        the file, and no panelization is possible. The placer may have
        auto-expanded the board (legalizer bounds expansion / density
        target); we always write the FINAL outline the placer used, so
        the output file's Edge.Cuts matches the placement.
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

    # Write the inferred (possibly auto-expanded) outline to real Edge.Cuts
    # geometry when the source file had no Edge.Cuts to begin with.
    if not getattr(model, "user_defined_outline", False):
        text = _inject_inferred_edge_cuts(text, model)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")

    return str(output_path)


def _inject_inferred_edge_cuts(text: str, model: BoardModel) -> str:
    """Inject a ``gr_rect`` on ``Edge.Cuts`` reflecting the model's board outline.

    Idempotent: if the file already has any graphic geometry (``gr_rect``,
    ``gr_line``, ``gr_poly``, ``gr_circle``) on the ``Edge.Cuts`` layer,
    no new geometry is added. The check is regex-based to distinguish
    actual geometry from the layer-table definition (every .kicad_pcb
    has ``("Edge.Cuts" user)`` in its layers table; that's NOT geometry).

    The rect is inserted immediately before the closing ``)`` of the
    root ``(kicad_pcb ...)`` expression so KiCad parses it as a
    top-level graphic item.
    """
    # Idempotency guard: only skip if there's actual gr_* geometry whose
    # layer attribute is "Edge.Cuts". The textual "Edge.Cuts" substring
    # also appears in the layers table definition, which is always present.
    edge_cuts_geom_re = re.compile(
        r'\(\s*gr_(?:rect|line|poly|circle)\b[^)]*?'
        r'\(\s*layer\s+"Edge\.Cuts"\s*\)',
        re.DOTALL,
    )
    if edge_cuts_geom_re.search(text):
        return text

    board = model.board
    bx1 = format_kicad_coord(board.x_min)
    by1 = format_kicad_coord(board.y_min)
    bx2 = format_kicad_coord(board.x_max)
    by2 = format_kicad_coord(board.y_max)
    import uuid
    tstamp = str(uuid.uuid4())
    edge_cuts_rect = (
        f'  (gr_rect (start {bx1} {by1}) (end {bx2} {by2}) '
        f'(stroke (width 0.1) (type solid)) (fill none) (layer "Edge.Cuts") '
        f'(tstamp "{tstamp}"))\n'
    )

    text = text.rstrip()
    idx = text.rfind(')')
    if idx < 0:
        return text + "\n" + edge_cuts_rect
    return text[:idx] + edge_cuts_rect + text[idx:]


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
            # Apply position update to all components, including "fixed" ones
            # (the writer is called after the placer has settled final positions
            # — fixed components still need to be written back even if unmoved).
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
    """Update the (at x y [rotation]) field in a footprint block.

    KiCad applies the footprint's rotation to every pad on top of the pad's
    own local (at X Y R) — i.e. world_rot = footprint_R + pad_R. Previous
    versions of this function ALSO added the rotation delta to each pad's
    local ``at``, producing world_rot = footprint_R + pad_R + delta —
    double-rotating pads in the world frame. Latent on symmetric 2-pin
    components (caps/resistors — rotating 180° is a no-op in world frame)
    but real on asymmetric footprints (SOIC with offset pad orientation,
    QFN with corner pads). See IMPROVEMENTS §2.2.

    The fix is to update ONLY the footprint's (at ...) and leave pad
    (at ...) expressions untouched. Pads then rotate naturally with the
    footprint, matching KiCad's renderer semantics.
    """
    x_str = format_kicad_coord(comp.x)
    y_str = format_kicad_coord(comp.y)

    # Build new (at ...) expression for the FOOTPRINT only.
    new_rotation = round(comp.rotation)
    if comp.rotation != 0.0:
        new_at = f"(at {x_str} {y_str} {new_rotation})"
    else:
        new_at = f"(at {x_str} {y_str})"

    # Replace only the FIRST (at ...) — the footprint's own placement
    # attribute. Pad (at ...) attributes appear later in the block and
    # are left untouched.
    return re.sub(
        r'\(\s*at\s+[-\d.]+\s+[-\d.]+(?:\s+[-\d.]+)?\s*\)',
        new_at,
        fp_block,
        count=1
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
