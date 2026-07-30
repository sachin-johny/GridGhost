"""Visualize placement: SVG image + stats (HPWL, spread, cap-IC grouping).

UPDATED (patch-evaluation audit):
  * All components drawn — including edge connectors (overhang shown), so
    you can see how the body sits OUTSIDE the outline and the pads sit
    INSIDE.
  * Empty/usable interior region drawn as a dashed rect (margin inset,
    connector_reserve inset, and the original interior_bbox that SA +
    legalizer clamp to).
  * True per-component bounding boxes drawn (red hairline) — these are the
    actual rectangles the legalizer sees, not the colored body rect.
  * Board outline (solid black) and margin ring (light grey fill) shown.
  * Pad dots drawn for connectors so it's visually obvious when solder
    pads land outside the board (the bug we just fixed on test6).
  * Per-component labels (ref + W×H), HPWL/spread stats, overlaps, OOB.

Usage:
    python scripts/visualize_placement.py <input.kicad_pcb>
        [--profile auto] [--no-sa] [--out dir] [--seed 42]
"""
from __future__ import annotations

import os
import sys

import argparse
import math
import random
# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parsers.kicad_parser import KiCadParser
from engine.smart_placement import smart_grid_place, _compute_interior_bbox, _is_vertical_connector
from engine.constraint_evaluator import _build_decoupling_map
from engine.subcircuit_patterns import detect_subcircuit_patterns
from legalization.legalizer import legalize
from profiles.board_profiles import get_profile
from config import load_config
import engine.cost_state as cs
from engine.cost_function import total_hpwl

# Pull in the macro-v2 pipeline so SVG reflects the default placement path
from place.pipeline import place_v2, build_macros, _is_vertical_connector as _is_vertical_v2
from place.initial import compute_interior_bbox as compute_interior_bbox_v2


def place_board(pcb_path: str, profile_name: str = "auto", use_sa: bool = True, seed: int = 42):
    """Run the macro-v2 pipeline (default since 224482a) and return model + profile.

    Mirrors what `gridghost.py place --macro-v2` runs, but in-process so the
    visualization has the placed model directly.
    """
    random.seed(seed)
    cfg = load_config()

    parser = KiCadParser(pcb_path, bbox_margin=cfg.parser.bbox_margin)
    model = parser.parse()

    if profile_name == "auto":
        has_ic = any(getattr(c, "component_type", "") in {"ic", "mcu", "regulator"}
                     for c in model.components)
        profile_name = "mcu_peripheral" if has_ic else "generic"
    profile = get_profile(profile_name)
    model.active_rules = profile.active_rules()

    # Macro-v2 pipeline (default). Place + SA + legalize all happen here.
    place_v2(
        model,
        margin=cfg.placement.margin,
        grid_mm=cfg.legalization.grid_mm,
        sa_iterations=1500,
        sa_reheats=2,
        seed=seed,
        verbose=False,
    )
    return model, profile


def compute_stats(model, profile):
    """Compute HPWL, spread, cap-IC distances, signal-flow chain info."""
    import statistics

    comps = [c for c in model.components if not c.is_fixed]
    xs = [c.x for c in comps]
    ys = [c.y for c in comps]
    std_x = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    std_y = statistics.pstdev(ys) if len(ys) > 1 else 0.0

    hpwl = total_hpwl(model)

    dm = _build_decoupling_map(model)
    cap_ic_distances = []
    for ic_ref, caps in dm.items():
        ic = model.get_component(ic_ref)
        for cap_ref in caps:
            cap = model.get_component(cap_ref)
            d = math.hypot(ic.x - cap.x, ic.y - cap.y)
            cap_ic_distances.append((ic_ref, cap_ref, d, cap.overlaps(ic)))
    cap_ic_overlaps = sum(1 for _, _, _, o in cap_ic_distances if o)

    try:
        chains = [p for p in detect_subcircuit_patterns(model)
                  if p.motif_type == "signal_flow_chain"]
    except Exception:
        chains = []

    overlaps = 0
    for i, a in enumerate(comps):
        for b in comps[i + 1:]:
            if a.overlaps(b):
                overlaps += 1

    return {
        "hpwl": hpwl,
        "std_x": std_x,
        "std_y": std_y,
        "n_components": len(comps),
        "n_caps": sum(len(v) for v in dm.values()),
        "n_ics_with_caps": len(dm),
        "cap_ic_distances": cap_ic_distances,
        "cap_ic_overlaps": cap_ic_overlaps,
        "n_chains": len(chains),
        "chains": chains,
        "overlaps": overlaps,
    }


def _compute_interior_bbox_for_viz(model, margin: float, mating_margin: float = 5.0):
    """Re-compute the interior_bbox the macro-v2 pipeline uses (so we can draw it).

    Mirrors `place/pipeline.py::place_v2` Phase 2 logic.
    """
    interior_macros, connector_macros, _fixed = build_macros(model)
    if connector_macros:
        connector_reserve = min(
            mating_margin,
            min(model.board.width, model.board.height) * 0.2,
        )
    else:
        connector_reserve = 0.0
    ib = compute_interior_bbox_v2(
        interior_macros, model.board, margin, connector_reserve=connector_reserve,
    )
    return ib, connector_reserve, len(connector_macros)


def render_svg(model, stats, out_path: str, margin: float = 5.0, mating_margin: float = 5.0):
    """Render board + components as SVG.

    Draws (in this z-order, bottom → top):
      1. Margin ring (light grey fill between board and margin-inset rect).
      2. Board outline (solid black).
      3. Empty/usable interior bbox (dashed green) — where SA + legalizer clamp.
      4. Cap-IC grouping lines (light grey dashed).
      5. Signal-flow chain lines (blue).
      6. Per-component bounding box (red hairline) — TRUE rectangle.
      7. Per-component body fill (color by type, semi-transparent).
      8. Connector pads (small black dots) — visually shows pad-inside vs pad-outside.
      9. Component label (ref + W×H).
     10. Legend + title with HPWL/spread/overlaps stats.
    """
    b = model.board
    pad_margin = 8  # extra SVG margin so overhanging connectors don't get clipped
    width = b.width + 2 * pad_margin
    height = b.height + 2 * pad_margin
    scale = 8  # pixels per mm

    # Origin offset: subtract so SVG coords are board-relative.
    ox = b.x_min - pad_margin
    oy = b.y_min - pad_margin

    def sx(x: float) -> float:
        return (x - ox) * scale

    def sy(y: float) -> float:
        return (y - oy) * scale

    palette = {
        "ic": "#e6194b",
        "mcu": "#dcbeff",
        "regulator": "#fabed4",
        "capacitor": "#42d4f4",
        "resistor": "#911636",
        "crystal": "#ffe119",
        "connector": "#469990",
        "generic": "#a9a9a9",
    }

    def color_for(c):
        t = getattr(c, "component_type", "generic") or "generic"
        return palette.get(t, "#a9a9a9")

    # Build decap map for IC→cap grouping lines
    dm = _build_decoupling_map(model)
    ref_to_comp = {c.ref: c for c in model.components}

    # Build chain pairs
    chain_pairs = set()
    for ch in stats["chains"]:
        members = [r for r in ch.all_refs if r in ref_to_comp]
        for i, a in enumerate(members):
            for other in members[i + 1:]:
                chain_pairs.add((a, other))

    interior_bbox, connector_reserve, n_conn = _compute_interior_bbox_for_viz(
        model, margin, mating_margin,
    )

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width*scale} {height*scale}" '
        f'width="{width*scale}" height="{height*scale}" '
        f'font-family="monospace" font-size="10">',
        f'<rect width="{width*scale}" height="{height*scale}" fill="white"/>',
    ]

    # ── 1. Margin ring (the area between board outline and margin-inset) ──
    margin_inset = (
        b.x_min + margin, b.y_min + margin,
        b.x_max - margin, b.y_max - margin,
    )
    # Outer (board) rect
    lines.append(
        f'<rect x="{sx(b.x_min):.1f}" y="{sy(b.y_min):.1f}" '
        f'width="{b.width*scale:.1f}" height="{b.height*scale:.1f}" '
        f'fill="#f0f0f0" stroke="none"/>'
    )
    # Margin inset (light grey fill = margin ring)
    mx1, my1, mx2, my2 = margin_inset
    # Top strip
    lines.append(
        f'<rect x="{sx(b.x_min):.1f}" y="{sy(b.y_min):.1f}" '
        f'width="{b.width*scale:.1f}" height="{(margin)*scale:.1f}" fill="#d9d9d9"/>'
    )
    # Bottom strip
    lines.append(
        f'<rect x="{sx(b.x_min):.1f}" y="{sy(b.y_max-margin):.1f}" '
        f'width="{b.width*scale:.1f}" height="{margin*scale:.1f}" fill="#d9d9d9"/>'
    )
    # Left strip
    lines.append(
        f'<rect x="{sx(b.x_min):.1f}" y="{sy(b.y_min):.1f}" '
        f'width="{margin*scale:.1f}" height="{b.height*scale:.1f}" fill="#d9d9d9"/>'
    )
    # Right strip
    lines.append(
        f'<rect x="{sx(b.x_max-margin):.1f}" y="{sy(b.y_min):.1f}" '
        f'width="{margin*scale:.1f}" height="{b.height*scale:.1f}" fill="#d9d9d9"/>'
    )

    # ── 2. Board outline (solid black) ──
    lines.append(
        f'<rect x="{sx(b.x_min):.1f}" y="{sy(b.y_min):.1f}" '
        f'width="{b.width*scale:.1f}" height="{b.height*scale:.1f}" '
        f'fill="none" stroke="black" stroke-width="2"/>'
    )

    # ── 3. Empty/usable interior bbox (dashed green) ──
    ibx1, iby1, ibx2, iby2 = interior_bbox
    ib_w = ibx2 - ibx1
    ib_h = iby2 - iby1
    lines.append(
        f'<rect x="{sx(ibx1):.1f}" y="{sy(iby1):.1f}" '
        f'width="{ib_w*scale:.1f}" height="{ib_h*scale:.1f}" '
        f'fill="none" stroke="#2ca02c" stroke-width="1" stroke-dasharray="4,3"/>'
    )
    # Label interior bbox dimensions
    lines.append(
        f'<text x="{sx(ibx1):.1f}" y="{sy(iby1)-3:.1f}" font-size="7" fill="#2ca02c">'
        f'interior {ib_w:.1f}×{ib_h:.1f}mm (reserve={connector_reserve:.1f}, '
        f'margin={margin:.1f})</text>'
    )

    # ── 4. Cap-IC grouping lines (thin grey dashed) ──
    for ic_ref, caps_list in dm.items():
        ic = ref_to_comp.get(ic_ref)
        if not ic:
            continue
        for cap_ref in caps_list:
            cap = ref_to_comp.get(cap_ref)
            if not cap:
                continue
            x1, y1 = sx(ic.x), sy(ic.y)
            x2, y2 = sx(cap.x), sy(cap.y)
            lines.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="#cccccc" stroke-width="0.5" stroke-dasharray="2,2"/>'
            )

    # ── 5. Signal-flow chain lines (blue) ──
    for a, b2 in chain_pairs:
        ca = ref_to_comp.get(a)
        cb = ref_to_comp.get(b2)
        if not ca or not cb:
            continue
        x1, y1 = sx(ca.x), sy(ca.y)
        x2, y2 = sx(cb.x), sy(cb.y)
        lines.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="#0066cc" stroke-width="1" opacity="0.5"/>'
        )

    # ── 6/7/8/9. Components: bbox → body fill → pads → label ──
    # Sort: connectors LAST so they're on top (overhanging past board edge)
    sorted_comps = sorted(
        model.components,
        key=lambda c: (0 if getattr(c, "component_type", "") != "connector" else 1,
                       c.ref),
    )
    for c in sorted_comps:
        # IMPORTANT: use the BBOX CENTER (not c.x/c.y, which is the KiCad
        # origin). For components with non-zero bbox_offset (connectors,
        # edge-mount SMAs, some resistors) the origin and bbox center are
        # different points — drawing the body fill at (c.x, c.y) makes it
        # misaligned with the red bbox hairline by the bbox_offset amount.
        bx1, by1, bx2, by2 = c.bbox
        bcx = (bx1 + bx2) / 2
        bcy = (by1 + by2) / 2
        cx = sx(bcx)
        cy = sy(bcy)
        ew = (bx2 - bx1) * scale
        eh = (by2 - by1) * scale
        col = color_for(c)
        # True bbox — red hairline (visible even where body fill is transparent)
        lines.append(
            f'<rect x="{sx(bx1):.1f}" y="{sy(by1):.1f}" '
            f'width="{(bx2-bx1)*scale:.1f}" height="{(by2-by1)*scale:.1f}" '
            f'fill="none" stroke="#cc0000" stroke-width="0.4" opacity="0.9"/>'
        )
        # Body fill (semi-transparent so overhang past the board edge is visible).
        # Same rect as the bbox hairline — just filled with the type color.
        lines.append(
            f'<rect x="{sx(bx1):.1f}" y="{sy(by1):.1f}" '
            f'width="{(bx2-bx1)*scale:.1f}" height="{(by2-by1)*scale:.1f}" '
            f'fill="{col}" stroke="black" stroke-width="0.5" opacity="0.55"/>'
        )
        # Pads (small black dots) — for connectors this makes pad-inside-vs-outside obvious
        for p in c.pads:
            px, py = p.absolute_pos(c.x, c.y, c.rotation)
            lines.append(
                f'<circle cx="{sx(px):.1f}" cy="{sy(py):.1f}" r="0.8" '
                f'fill="#000000"/>'
            )
        # Label: ref on top, W×H on bottom — centered on bbox center
        lines.append(
            f'<text x="{cx:.1f}" y="{cy-1:.1f}" font-size="6" text-anchor="middle" '
            f'dominant-baseline="middle" fill="black">{c.ref}</text>'
        )
        lines.append(
            f'<text x="{cx:.1f}" y="{cy+5:.1f}" font-size="5" text-anchor="middle" '
            f'dominant-baseline="middle" fill="#333333">'
            f'{c.effective_width:.1f}×{c.effective_height:.1f}</text>'
        )

    # ── 10. Legend + title ──
    legend_x = 5
    legend_y = height * scale - 110
    lines.append(f'<rect x="{legend_x-3}" y="{legend_y-3}" '
                 f'width="{width*scale - 2*(legend_x-3)}" height="105" '
                 f'fill="white" stroke="#cccccc" stroke-width="0.5" opacity="0.95"/>')
    lines.append(f'<text x="{legend_x}" y="{legend_y}" font-size="9" font-weight="bold">Legend:</text>')
    items_per_row = 4
    for i, (k, v) in enumerate(palette.items()):
        col_x = legend_x + (i % items_per_row) * 80
        col_y = legend_y + 15 + (i // items_per_row) * 15
        lines.append(f'<rect x="{col_x}" y="{col_y-7}" width="8" height="8" fill="{v}" '
                     f'stroke="black"/>')
        lines.append(f'<text x="{col_x+12}" y="{col_y}" font-size="8">{k}</text>')
    # Lines / rects
    lines.append(
        f'<line x1="{legend_x}" y1="{legend_y+45}" x2="{legend_x+15}" y2="{legend_y+45}" '
        f'stroke="#cccccc" stroke-width="1" stroke-dasharray="2,2"/>'
        f'<text x="{legend_x+20}" y="{legend_y+48}" font-size="8">cap-IC grouping</text>'
    )
    lines.append(
        f'<line x1="{legend_x+120}" y1="{legend_y+45}" x2="{legend_x+135}" y2="{legend_y+45}" '
        f'stroke="#0066cc" stroke-width="1"/>'
        f'<text x="{legend_x+140}" y="{legend_y+48}" font-size="8">signal-flow chain</text>'
    )
    lines.append(
        f'<rect x="{legend_x}" y="{legend_y+57}" width="10" height="6" fill="none" '
        f'stroke="#cc0000" stroke-width="0.5"/>'
        f'<text x="{legend_x+15}" y="{legend_y+63}" font-size="8">true bbox (courtyard incl.)</text>'
    )
    lines.append(
        f'<rect x="{legend_x+150}" y="{legend_y+57}" width="10" height="6" fill="none" '
        f'stroke="#2ca02c" stroke-width="1" stroke-dasharray="3,2"/>'
        f'<text x="{legend_x+165}" y="{legend_y+63}" font-size="8">interior bbox (margin+reserve)</text>'
    )
    lines.append(
        f'<rect x="{legend_x}" y="{legend_y+72}" width="10" height="6" fill="#d9d9d9"/>'
        f'<text x="{legend_x+15}" y="{legend_y+78}" font-size="8">edge margin ring ({margin:.0f}mm)</text>'
    )
    lines.append(
        f'<circle cx="{legend_x+150+5}" cy="{legend_y+75}" r="1" fill="black"/>'
        f'<text x="{legend_x+165}" y="{legend_y+78}" font-size="8">pad position</text>'
    )

    # Title with stats
    title = (
        f"HPWL={stats['hpwl']:.1f}  "
        f"std_x={stats['std_x']:.1f} std_y={stats['std_y']:.1f}  "
        f"overlaps={stats['overlaps']}  "
        f"cap-IC overlaps={stats['cap_ic_overlaps']}  "
        f"chains={stats['n_chains']}  "
        f"conn={n_conn} reserve={connector_reserve:.1f}"
    )
    lines.append(f'<text x="5" y="15" font-size="11" font-weight="bold">{title}</text>')

    lines.append('</svg>')

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_stats(stats, out_path, board_name):
    lines = [
        f"Board: {board_name}",
        f"Components: {stats['n_components']}",
        f"HPWL: {stats['hpwl']:.2f}",
        f"Spread: std_x={stats['std_x']:.2f}  std_y={stats['std_y']:.2f}",
        f"ICs with caps: {stats['n_ics_with_caps']}",
        f"Total caps: {stats['n_caps']}",
        f"Cap-IC overlaps: {stats['cap_ic_overlaps']}",
        f"Total overlaps: {stats['overlaps']}",
        f"Signal-flow chains: {stats['n_chains']}",
        "",
        "Cap-IC distances:",
    ]
    for ic_ref, cap_ref, d, o in stats["cap_ic_distances"]:
        flag = " OVERLAP!" if o else ""
        lines.append(f"  {cap_ref} → {ic_ref}: {d:.2f}mm{flag}")

    if stats["chains"]:
        lines.append("")
        lines.append("Chains:")
        for ch in stats["chains"]:
            end = ch.metadata.get("end_connector", "?")
            ic_count = ch.metadata.get("ic_count", 0)
            lines.append(f"  {ch.anchor_ref} → {end}  ({ic_count} ICs, {len(ch.all_refs)} members)")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pcb")
    ap.add_argument("--profile", default="auto")
    ap.add_argument("--no-sa", action="store_true")
    ap.add_argument("--out", default="visualizations")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    board_name = os.path.splitext(os.path.basename(args.pcb))[0]

    print(f"=== Placing {board_name} (profile={args.profile}, sa={not args.no_sa}) ===")
    model, profile = place_board(args.pcb, args.profile, use_sa=not args.no_sa, seed=args.seed)
    stats = compute_stats(model, profile)

    svg_path = os.path.join(args.out, f"{board_name}_placement.svg")
    stats_path = os.path.join(args.out, f"{board_name}_stats.txt")
    render_svg(model, stats, svg_path)
    write_stats(stats, stats_path, board_name)

    print(f"\nStats:")
    print(f"  HPWL:           {stats['hpwl']:.2f}")
    print(f"  std_x, std_y:   {stats['std_x']:.2f}, {stats['std_y']:.2f}")
    print(f"  Overlaps:       {stats['overlaps']}")
    print(f"  Cap-IC overlaps:{stats['cap_ic_overlaps']}")
    print(f"  Caps/ICs:       {stats['n_caps']}/{stats['n_ics_with_caps']}")
    print(f"  Chains:         {stats['n_chains']}")
    print(f"\nWrote: {svg_path}")
    print(f"Wrote: {stats_path}")


if __name__ == "__main__":
    main()
