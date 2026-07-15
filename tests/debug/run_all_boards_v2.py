"""Run new pipeline across all test boards and report quality."""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from parsers.kicad_parser import KiCadParser
from place.pipeline import place_v2
from assign.assign_caps import assign_caps


def run_board(pcb_path):
    parser = KiCadParser(str(pcb_path), bbox_margin=0.8)
    model = parser.parse()
    n_comps = len(model.components)

    try:
        result = place_v2(
            model,
            sa_iterations=800,
            sa_reheats=2,
            seed=42,
            verbose=False,
        )
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"

    # Cap-IC distances
    decap_map = assign_caps(model)
    within_5 = 0
    within_8 = 0
    worst = 0.0
    n_caps = 0
    for ic_ref, cap_refs in decap_map.items():
        ic = model.get_component(ic_ref)
        for cap_ref in cap_refs:
            cap = model.get_component(cap_ref)
            if cap is None:
                continue
            d = math.hypot(cap.x - ic.x, cap.y - ic.y)
            n_caps += 1
            if d <= 5.0:
                within_5 += 1
            if d <= 8.0:
                within_8 += 1
            worst = max(worst, d)

    # Overlaps
    overlaps = 0
    for i, a in enumerate(model.components):
        for b in model.components[i + 1:]:
            if a.overlaps(b):
                overlaps += 1

    # Out of bounds
    oob = 0
    for c in model.components:
        if getattr(c, "is_edge_connector", False):
            continue
        x1, y1, x2, y2 = c.bbox
        if (x1 < model.board.x_min or y1 < model.board.y_min or
                x2 > model.board.x_max or y2 > model.board.y_max):
            oob += 1

    return (
        f"  comps={n_comps:3d}  caps<5mm={within_5}/{n_caps}  "
        f"caps<8mm={within_8}/{n_caps}  worst={worst:.1f}mm  "
        f"overlaps={overlaps:3d}  oob={oob}"
    )


def main():
    boards = sorted((ROOT / "tests" / "test_pcbs").glob("*.kicad_pcb"))
    print(f"{'Board':<30} {'Result'}")
    print("-" * 110)
    for pcb in boards:
        result = run_board(pcb)
        print(f"{pcb.stem:<30} {result}")


if __name__ == "__main__":
    main()
