"""Inline SVG visualizer for GridGhost placement results.

Renders a placed BoardModel as a self-contained SVG string suitable for
embedding in an HTML dashboard. Shows:

  - Board outline (Edge.Cuts) as a solid black rectangle.
  - Margin ring (light grey fill between outline and interior bbox).
  - Interior bbox (dashed green) — the region SA + legalizer clamp to.
  - All components colored by type (ICs red, caps blue, resistors green,
    connectors orange, mounting holes grey cross, generic light grey).
  - Component courtyard bboxes (the actual rectangles the legalizer sees).
  - Overlap pairs highlighted with a red fill + red border.
  - Decoupling cap → IC leader lines (thin grey dashed).
  - Connector pad dots (small black circles) so overhang is visible.
  - Mounting hole keepout rings (dashed grey).
  - Per-component labels for ICs and large components (ref + W×H).
  - Legend in the top-right corner.

Used by ``tests/run_all.py`` to produce per-board SVGs that get inlined
into the HTML dashboard.

Usage (standalone):
    python tests/visualizer.py <placed.kicad_pcb> --out board.svg
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.board_model import BoardModel, Component


# Per-type colors. Chosen to be color-blind friendly (avoid red/green
# only encoding) and to print acceptably on B&W printers (varied luminance).
TYPE_COLORS = {
    "ic":            "#d62728",  # red — silicon deserves attention
    "mcu":           "#d62728",  # same as IC
    "regulator":     "#9467bd",  # purple
    "capacitor":     "#1f77b4",  # blue
    "resistor":      "#2ca02c",  # green
    "connector":     "#ff7f0e",  # orange
    "crystal":       "#17becf",  # cyan
    "mounting_hole": "#7f7f7f",  # grey
    "generic":       "#bcbcbc",  # light grey
}


def _component_color(comp: "Component") -> str:
    t = getattr(comp, "component_type", "generic") or "generic"
    return TYPE_COLORS.get(t, TYPE_COLORS["generic"])


def _format_dim(mm: float) -> str:
    """Format a dimension in mm, stripping trailing zeros."""
    s = f"{mm:.2f}".rstrip("0").rstrip(".")
    return s


def render_board_svg(
    model: "BoardModel",
    *,
    target_width: float = 800.0,
    title: str | None = None,
    show_labels: bool = True,
    show_pads: bool = True,
    show_decaps: bool = True,
) -> str:
    """Render ``model`` as an inline SVG string.

    The SVG is self-contained (no external refs) and sized so its width
    matches ``target_width`` pixels. Height is derived from the board's
    aspect ratio. Coordinates use the KiCad convention (origin at left,
    y increases downward) — same as the board file, so debug overlays
    line up.

    Args:
        model: BoardModel with placed components.
        target_width: SVG width in pixels. Height auto-derived.
        title: Optional title text rendered top-left.
        show_labels: Render ref+size labels on ICs and large components.
        show_pads: Render pad dots for connectors.
        show_decaps: Render cap→IC leader lines.

    Returns:
        SVG markup as a string. Ready to inline in HTML.
    """
    board = model.board
    if board.width <= 0 or board.height <= 0:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="100">' \
               '<text x="10" y="20" fill="red">Invalid board (zero area)</text></svg>'

    # Compute scale: target_width pixels = board.width mm
    scale = target_width / board.width
    svg_w = target_width
    svg_h = board.height * scale
    # Add a small margin around the board for labels and the legend
    pad_px = 24.0
    total_w = svg_w + 2 * pad_px
    total_h = svg_h + 2 * pad_px + 30.0  # +30 for title strip

    # Coordinate transform: KiCad (x_min, y_min) → SVG (pad_px, pad_px + 30)
    def sx(x_mm: float) -> float:
        return pad_px + (x_mm - board.x_min) * scale

    def sy(y_mm: float) -> float:
        return pad_px + 30.0 + (y_mm - board.y_min) * scale

    # ─── Identify overlaps (component-level) ──────────────────────────
    overlap_pairs: list[tuple["Component", "Component", float]] = []
    overlapping_refs: set[str] = set()
    comps = model.components
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            a, b = comps[i], comps[j]
            if a.overlaps(b):
                area = a.overlap_area(b)
                overlap_pairs.append((a, b, area))
                overlapping_refs.add(a.ref)
                overlapping_refs.add(b.ref)

    # ─── Build decap map (cap → IC leader lines) ──────────────────────
    decap_lines: list[tuple[float, float, float, float]] = []  # (x1,y1,x2,y2)
    if show_decaps:
        try:
            from assign.assign_caps import assign_caps
            decap_map = assign_caps(model)
            ref_to_comp = {c.ref: c for c in model.components}
            for ic_ref, cap_refs in decap_map.items():
                ic = ref_to_comp.get(ic_ref)
                if ic is None:
                    continue
                for cap_ref in cap_refs:
                    cap = ref_to_comp.get(cap_ref)
                    if cap is None:
                        continue
                    decap_lines.append((ic.x, ic.y, cap.x, cap.y))
        except Exception:
            pass  # decap map is best-effort — skip on any error

    # ─── SVG construction ─────────────────────────────────────────────
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {total_w:.1f} {total_h:.1f}" '
        f'width="{total_w:.1f}" height="{total_h:.1f}" '
        f'font-family="ui-monospace, Menlo, Consolas, monospace" font-size="10">',
        f'<rect width="{total_w:.1f}" height="{total_h:.1f}" fill="#ffffff"/>',
    ]

    # Title strip
    if title:
        parts.append(
            f'<text x="{pad_px:.1f}" y="18" font-size="13" font-weight="bold" '
            f'fill="#222">{_esc(title)}</text>'
        )

    # ── Board background (white) ──
    parts.append(
        f'<rect x="{sx(board.x_min):.1f}" y="{sy(board.y_min):.1f}" '
        f'width="{board.width * scale:.1f}" height="{board.height * scale:.1f}" '
        f'fill="#ffffff" stroke="none"/>'
    )

    # ── Margin ring (light grey) — inset by 5mm from board outline ──
    margin = 5.0
    mx1, my1 = board.x_min + margin, board.y_min + margin
    mx2, my2 = board.x_max - margin, board.y_max - margin
    # Top strip
    parts.append(
        f'<rect x="{sx(board.x_min):.1f}" y="{sy(board.y_min):.1f}" '
        f'width="{board.width * scale:.1f}" height="{margin * scale:.1f}" fill="#eeeeee"/>'
    )
    # Bottom strip
    parts.append(
        f'<rect x="{sx(board.x_min):.1f}" y="{sy(board.y_max - margin):.1f}" '
        f'width="{board.width * scale:.1f}" height="{margin * scale:.1f}" fill="#eeeeee"/>'
    )
    # Left strip
    parts.append(
        f'<rect x="{sx(board.x_min):.1f}" y="{sy(board.y_min):.1f}" '
        f'width="{margin * scale:.1f}" height="{board.height * scale:.1f}" fill="#eeeeee"/>'
    )
    # Right strip
    parts.append(
        f'<rect x="{sx(board.x_max - margin):.1f}" y="{sy(board.y_min):.1f}" '
        f'width="{margin * scale:.1f}" height="{board.height * scale:.1f}" fill="#eeeeee"/>'
    )

    # ── Interior bbox (dashed green) ──
    if mx2 > mx1 and my2 > my1:
        parts.append(
            f'<rect x="{sx(mx1):.1f}" y="{sy(my1):.1f}" '
            f'width="{(mx2 - mx1) * scale:.1f}" height="{(my2 - my1) * scale:.1f}" '
            f'fill="none" stroke="#2ca02c" stroke-width="0.8" stroke-dasharray="4,3" opacity="0.7"/>'
        )

    # ── Decap leader lines (drawn BEFORE components so components cover them) ──
    for x1, y1, x2, y2 in decap_lines:
        parts.append(
            f'<line x1="{sx(x1):.1f}" y1="{sy(y1):.1f}" '
            f'x2="{sx(x2):.1f}" y2="{sy(y2):.1f}" '
            f'stroke="#cccccc" stroke-width="0.6" stroke-dasharray="2,2"/>'
        )

    # ── Components ──
    # Draw fixed components (mounting holes, edge connectors) LAST so
    # they sit on top of movable components — matches visual hierarchy
    # a PCB designer expects.
    movable = [c for c in model.components if not c.is_fixed]
    fixed = [c for c in model.components if c.is_fixed]
    for comp in movable + fixed:
        bx1, by1, bx2, by2 = comp.bbox
        bw = bx2 - bx1
        bh = by2 - by1
        if bw <= 0 or bh <= 0:
            continue
        color = _component_color(comp)
        is_overlap = comp.ref in overlapping_refs
        is_fixed = comp.is_fixed

        # Body fill (light tint of the type color)
        fill = _lighten(color, 0.75)
        stroke = color
        stroke_w = 0.8
        if is_overlap:
            # Overlapping components: thick red border + light red fill
            fill = "#ffe0e0"
            stroke = "#ff0000"
            stroke_w = 1.8

        parts.append(
            f'<rect x="{sx(bx1):.1f}" y="{sy(by1):.1f}" '
            f'width="{bw * scale:.1f}" height="{bh * scale:.1f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_w}" opacity="0.9"/>'
        )

        # Mounting hole: draw a cross inside
        if comp.component_type == "mounting_hole":
            cx, cy = comp.x, comp.y
            r = max(bw, bh) * 0.3
            parts.append(
                f'<line x1="{sx(cx - r):.1f}" y1="{sy(cy):.1f}" '
                f'x2="{sx(cx + r):.1f}" y2="{sy(cy):.1f}" '
                f'stroke="{color}" stroke-width="1"/>'
            )
            parts.append(
                f'<line x1="{sx(cx):.1f}" y1="{sy(cy - r):.1f}" '
                f'x2="{sx(cx):.1f}" y2="{sy(cy + r):.1f}" '
                f'stroke="{color}" stroke-width="1"/>'
            )

        # Connector pads (small black dots)
        if show_pads and comp.component_type == "connector" and comp.pads:
            for pad in comp.pads[:20]:  # cap at 20 to avoid clutter
                try:
                    abs_x, abs_y = pad.absolute_pos(comp.x, comp.y, comp.rotation)
                    parts.append(
                        f'<circle cx="{sx(abs_x):.1f}" cy="{sy(abs_y):.1f}" '
                        f'r="1.2" fill="#222"/>'
                    )
                except Exception:
                    pass

        # Labels: ICs, MCUs, regulators, crystals, and any component wider than 5mm
        if show_labels:
            body_max = max(getattr(comp, "width", 0.0), getattr(comp, "height", 0.0))
            should_label = (
                comp.component_type in {"ic", "mcu", "regulator", "crystal"}
                or body_max >= 5.0
                or is_overlap
            )
            if should_label and bw * scale > 18 and bh * scale > 8:
                label = f"{comp.ref}"
                # Background rectangle for legibility
                label_w = len(label) * 5.5 + 4
                label_h = 9
                lx = sx(bx1)
                ly = sy(by1) - label_h - 1
                parts.append(
                    f'<rect x="{lx:.1f}" y="{ly:.1f}" '
                    f'width="{label_w:.1f}" height="{label_h}" '
                    f'fill="white" opacity="0.85" stroke="none"/>'
                )
                parts.append(
                    f'<text x="{lx + 2:.1f}" y="{ly + 7:.1f}" '
                    f'font-size="7" fill="{color}">{_esc(label)}</text>'
                )

    # ── Board outline (Edge.Cuts) — drawn on top so it's always visible ──
    parts.append(
        f'<rect x="{sx(board.x_min):.1f}" y="{sy(board.y_min):.1f}" '
        f'width="{board.width * scale:.1f}" height="{board.height * scale:.1f}" '
        f'fill="none" stroke="#000000" stroke-width="1.5"/>'
    )

    # ── Legend (top-right) ──
    legend_x = total_w - 130
    legend_y = 8
    parts.append(
        f'<rect x="{legend_x:.1f}" y="{legend_y:.1f}" '
        f'width="125" height="{8 + 13 * len(TYPE_COLORS):.1f}" '
        f'fill="white" stroke="#cccccc" stroke-width="0.5" opacity="0.95"/>'
    )
    for i, (t, col) in enumerate(TYPE_COLORS.items()):
        ly = legend_y + 11 + i * 13
        parts.append(
            f'<rect x="{legend_x + 4:.1f}" y="{ly:.1f}" width="10" height="8" '
            f'fill="{_lighten(col, 0.75)}" stroke="{col}" stroke-width="0.8"/>'
        )
        parts.append(
            f'<text x="{legend_x + 18:.1f}" y="{ly + 7:.1f}" '
            f'font-size="8" fill="#333">{_esc(t)}</text>'
        )

    # ── Overlap count badge (top-left, below title) ──
    if overlap_pairs:
        badge_text = f"⚠ {len(overlap_pairs)} overlaps"
        badge_color = "#dc3545"
    else:
        badge_text = "✓ 0 overlaps"
        badge_color = "#28a745"
    parts.append(
        f'<text x="{pad_px:.1f}" y="{pad_px + 22:.1f}" font-size="10" '
        f'font-weight="bold" fill="{badge_color}">{_esc(badge_text)}</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts)


def _lighten(hex_color: str, factor: float) -> str:
    """Lighten a #rrggbb color toward white. factor=0 → original, 1 → white."""
    if not hex_color.startswith("#") or len(hex_color) != 7:
        return hex_color
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def _esc(s: str) -> str:
    """XML-escape a string for safe SVG embedding."""
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;"))


# ─── Standalone CLI entry point ──────────────────────────────────────

def _main():
    import argparse
    parser = argparse.ArgumentParser(description="Render a placed .kicad_pcb as SVG.")
    parser.add_argument("input", help="Path to a placed .kicad_pcb file")
    parser.add_argument("--out", default=None, help="Output SVG path (default: stdout)")
    parser.add_argument("--width", type=float, default=800.0, help="SVG width in pixels")
    parser.add_argument("--title", default=None, help="Title text")
    parser.add_argument("--no-labels", action="store_true", help="Hide component labels")
    parser.add_argument("--no-pads", action="store_true", help="Hide connector pad dots")
    parser.add_argument("--no-decaps", action="store_true", help="Hide cap→IC leader lines")
    args = parser.parse_args()

    # Set up sys.path for GridGhost imports
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))

    from parsers.kicad_parser import KiCadParser
    p = KiCadParser(args.input, bbox_margin=0.8)
    model = p.parse()
    svg = render_board_svg(
        model,
        target_width=args.width,
        title=args.title or Path(args.input).stem,
        show_labels=not args.no_labels,
        show_pads=not args.no_pads,
        show_decaps=not args.no_decaps,
    )

    if args.out:
        Path(args.out).write_text(svg, encoding="utf-8")
        print(f"Wrote {args.out} ({len(svg)} bytes)")
    else:
        print(svg)


if __name__ == "__main__":
    _main()
