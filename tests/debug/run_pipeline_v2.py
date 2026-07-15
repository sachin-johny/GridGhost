"""End-to-end test of the macro-first pipeline on cbb.kicad_pcb."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from parsers.kicad_parser import KiCadParser
from place.pipeline import place_v2
from assign.assign_caps import assign_caps


def main():
    pcb = str(ROOT / "tests" / "test_pcbs" / "cbb.kicad_pcb")
    parser = KiCadParser(pcb, bbox_margin=0.8)
    model = parser.parse()
    print(f"Board: {model.board.width:.1f} x {model.board.height:.1f}mm, {len(model.components)} components")

    result = place_v2(
        model,
        margin=5.0,
        grid_mm=1.0,
        sa_iterations=800,
        sa_reheats=2,
        seed=42,
        verbose=True,
    )

    print()
    print("=" * 60)
    print("  Cap-IC distance check")
    print("=" * 60)
    decap_map = assign_caps(model)
    n_within_5 = 0
    n_within_8 = 0
    n_total = 0
    worst = []
    for ic_ref, cap_refs in decap_map.items():
        ic = model.get_component(ic_ref)
        if ic is None:
            continue
        for cap_ref in cap_refs:
            cap = model.get_component(cap_ref)
            if cap is None:
                continue
            d = math.hypot(cap.x - ic.x, cap.y - ic.y)
            n_total += 1
            if d <= 5.0:
                n_within_5 += 1
            if d <= 8.0:
                n_within_8 += 1
            worst.append((d, ic_ref, cap_ref))
    worst.sort(reverse=True)
    print(f"  Caps within 5mm of assigned IC: {n_within_5}/{n_total}")
    print(f"  Caps within 8mm of assigned IC: {n_within_8}/{n_total}")
    print(f"  Worst 5 cap-IC distances:")
    for d, ic_ref, cap_ref in worst[:5]:
        print(f"    {cap_ref} -> {ic_ref}: {d:.2f}mm")

    print()
    print("=" * 60)
    print("  Overlap check")
    print("=" * 60)
    overlaps = 0
    for i, a in enumerate(model.components):
        for b in model.components[i + 1:]:
            if a.overlaps(b):
                overlaps += 1
    print(f"  Overlapping pairs: {overlaps}")

    print()
    print("=" * 60)
    print("  Out-of-bounds check")
    print("=" * 60)
    oob = 0
    for c in model.components:
        if getattr(c, "is_edge_connector", False):
            continue
        x1, y1, x2, y2 = c.bbox
        if (x1 < model.board.x_min or y1 < model.board.y_min or
                x2 > model.board.x_max or y2 > model.board.y_max):
            oob += 1
    print(f"  Out-of-bounds components: {oob}")


if __name__ == "__main__":
    main()
