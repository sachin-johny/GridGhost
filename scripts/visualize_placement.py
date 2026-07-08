"""Visualize placement: SVG image + stats (HPWL, spread, cap-IC grouping).

Usage:
    python scripts/visualize_placement.py <input.kicad_pcb> [--profile auto] [--no-sa] [--out dir]

Generates:
    <out>/<board_name>_placement.svg   — component layout
    <out>/<board_name>_stats.txt       — HPWL, spread, cap-IC distances
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
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


def place_board(pcb_path: str, profile_name: str = "auto", use_sa: bool = True, seed: int = 42):
    random.seed(seed)
    cfg = load_config()
    cs.OVERLAP_WEIGHT = 25.0
    cs.BOUNDARY_WEIGHT = 8.0

    parser = KiCadParser(pcb_path, bbox_margin=cfg.parser.bbox_margin)
    model = parser.parse()

    if profile_name == "auto":
        has_ic = any(getattr(c, "component_type", "") in {"ic", "mcu", "regulator"} for c in model.components)
        profile_name = "mcu_peripheral" if has_ic else "generic"
    profile = get_profile(profile_name)

    smart_grid_place(model, margin=cfg.placement.margin,
                     spacing_factor=cfg.placement.spacing_factor, rules=profile.rules)
    model.active_rules = profile.active_rules()

    # Compute interior_bbox AFTER placement — pre-placement ib captures the
    # original tight cluster and would clamp the legalizer back into it,
    # causing massive overlaps on dense boards (e.g. test4: 50→9 overlaps).
    interior = [c for c in model.components
                if not c.is_fixed
                and (getattr(c, "component_type", "") != "connector" or _is_vertical_connector(c))]
    ib = _compute_interior_bbox(interior, model.board, 5.0) if interior else None

    legalize(model, grid_mm=cfg.legalization.grid_mm, max_iterations=800,
             push_strength=cfg.legalization.push_strength, verbose=False,
             use_abacus=True, interior_bbox=ib)

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

    # cap-IC distances
    dm = _build_decoupling_map(model)
    cap_ic_distances = []
    cap_ic_overlaps = 0
    for ic_ref, caps in dm.items():
        ic = model.get_component(ic_ref)
        for cap_ref in caps:
            cap = model.get_component(cap_ref)
            d = math.hypot(ic.x - cap.x, ic.y - cap.y)
            cap_ic_distances.append((ic_ref, cap_ref, d, cap.overlaps(ic)))

    cap_ic_overlaps = sum(1 for _, _, _, o in cap_ic_distances if o)

    # Signal-flow chain info
    try:
        chains = [p for p in detect_subcircuit_patterns(model) if p.motif_type == "signal_flow_chain"]
    except Exception:
        chains = []

    # overlaps
    overlaps = 0
    for i, a in enumerate(comps):
        for b in comps[i+1:]:
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


def render_svg(model, stats, out_path: str):
    """Render board + components as SVG.

    Boards with inferred outlines have non-zero x_min/y_min (the cluster's
    origin), so all component coords are shifted by (b.x_min, b.y_min) into
    a board-relative frame before scaling. Otherwise components render far
    outside the board rect.
    """
    b = model.board
    margin = 5
    width = b.width + 2 * margin
    height = b.height + 2 * margin
    scale = 8  # pixels per mm

    # Origin offset: subtract so SVG coords are board-relative.
    ox = b.x_min - margin
    oy = b.y_min - margin

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
            for other in members[i+1:]:
                chain_pairs.add((a, other))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width*scale} {height*scale}" '
        f'width="{width*scale}" height="{height*scale}" font-family="monospace" font-size="10">',
        f'<rect width="{width*scale}" height="{height*scale}" fill="white"/>',
        # Board outline
        f'<rect x="{margin*scale}" y="{margin*scale}" width="{b.width*scale}" '
        f'height="{b.height*scale}" fill="#f8f8f8" stroke="black" stroke-width="2"/>',
    ]

    # Draw cap-IC grouping lines (thin grey)
    for ic_ref, caps in dm.items():
        ic = ref_to_comp.get(ic_ref)
        if not ic:
            continue
        for cap_ref in caps:
            cap = ref_to_comp.get(cap_ref)
            if not cap:
                continue
            x1 = sx(ic.x); y1 = sy(ic.y)
            x2 = sx(cap.x); y2 = sy(cap.y)
            lines.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="#cccccc" stroke-width="0.5" stroke-dasharray="2,2"/>'
            )

    # Draw signal-flow chain lines (thicker blue)
    for a, b2 in chain_pairs:
        ca = ref_to_comp.get(a)
        cb = ref_to_comp.get(b2)
        if not ca or not cb:
            continue
        x1 = sx(ca.x); y1 = sy(ca.y)
        x2 = sx(cb.x); y2 = sy(cb.y)
        lines.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="#0066cc" stroke-width="1" opacity="0.5"/>'
        )

    # Draw components
    for c in model.components:
        cx = sx(c.x)
        cy = sy(c.y)
        w = c.effective_width * scale
        h = c.effective_height * scale
        col = color_for(c)
        lines.append(
            f'<rect x="{cx-w/2:.1f}" y="{cy-h/2:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'fill="{col}" stroke="black" stroke-width="0.5" opacity="0.85"/>'
        )
        # Label
        lines.append(
            f'<text x="{cx:.1f}" y="{cy:.1f}" font-size="6" text-anchor="middle" '
            f'dominant-baseline="middle" fill="black">{c.ref}</text>'
        )

    # Legend
    legend_x = 5
    legend_y = height * scale - 80
    lines.append(f'<text x="{legend_x}" y="{legend_y}" font-size="9" font-weight="bold">Legend:</text>')
    items_per_row = 4
    for i, (k, v) in enumerate(palette.items()):
        col_x = legend_x + (i % items_per_row) * 80
        col_y = legend_y + 15 + (i // items_per_row) * 15
        lines.append(f'<rect x="{col_x}" y="{col_y-7}" width="8" height="8" fill="{v}" stroke="black"/>')
        lines.append(f'<text x="{col_x+12}" y="{col_y}" font-size="8">{k}</text>')
    # Grey/blue lines
    lines.append(
        f'<line x1="{legend_x}" y1="{legend_y+30}" x2="{legend_x+15}" y2="{legend_y+30}" '
        f'stroke="#cccccc" stroke-width="1" stroke-dasharray="2,2"/>'
        f'<text x="{legend_x+20}" y="{legend_y+33}" font-size="8">cap-IC grouping</text>'
    )
    lines.append(
        f'<line x1="{legend_x+120}" y1="{legend_y+30}" x2="{legend_x+135}" y2="{legend_y+30}" '
        f'stroke="#0066cc" stroke-width="1"/>'
        f'<text x="{legend_x+140}" y="{legend_y+33}" font-size="8">signal-flow chain</text>'
    )

    # Title with stats
    title = (
        f"HPWL={stats['hpwl']:.1f}  "
        f"std_x={stats['std_x']:.1f} std_y={stats['std_y']:.1f}  "
        f"overlaps={stats['overlaps']}  "
        f"cap-IC overlaps={stats['cap_ic_overlaps']}  "
        f"chains={stats['n_chains']}"
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
