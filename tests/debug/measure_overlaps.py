"""Measure placement overlaps for a board under the macro-v2 pipeline.

Reports total pairwise overlaps, total overlap area, and a breakdown of
intra-macro (cap<->its leader, cap<->sibling cap) vs inter-macro pairs.
Run before AND after a fix to quantify the effect.

Usage: python tests/debug/measure_overlaps.py [board]
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from parsers.kicad_parser import KiCadParser
from place.pipeline import place_v2, build_macros


def measure(board: str, seed: int = 42):
    pcb_path = ROOT / "tests" / "test_pcbs" / f"{board}.kicad_pcb"
    parser = KiCadParser(str(pcb_path), bbox_margin=0.8)
    model = parser.parse()

    place_v2(model, seed=seed, verbose=False)

    # Rebuild macros to know which components share a macro (intra-macro).
    interior, connectors, _fixed = build_macros(model)
    comp_to_macro_id = {}
    for mid, macro in enumerate(interior + connectors):
        for c in macro.members:
            comp_to_macro_id[id(c)] = mid

    comps = model.components
    total_pairs = 0
    overlap_pairs = 0
    total_area = 0.0
    intra_pairs = 0
    inter_pairs = 0
    intra_area = 0.0
    inter_area = 0.0
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            a, b = comps[i], comps[j]
            total_pairs += 1
            area = a.overlap_area(b)
            if area > 1e-9:
                overlap_pairs += 1
                total_area += area
                same_macro = (
                    id(a) in comp_to_macro_id
                    and comp_to_macro_id.get(id(a)) == comp_to_macro_id.get(id(b))
                )
                if same_macro:
                    intra_pairs += 1
                    intra_area += area
                else:
                    inter_pairs += 1
                    inter_area += area

    return {
        "board": board,
        "overlap_pairs": overlap_pairs,
        "total_overlap_area": round(total_area, 2),
        "intra_macro_pairs": intra_pairs,
        "intra_macro_area": round(intra_area, 2),
        "inter_macro_pairs": inter_pairs,
        "inter_macro_area": round(inter_area, 2),
    }


def main():
    boards = sys.argv[1:] or ["cbb", "cbbwO", "test4", "test5", "test6", "th_sensor"]
    results = []
    for board in boards:
        try:
            r = measure(board)
            results.append(r)
            print(
                f"{board:<10} overlaps={r['overlap_pairs']:>3} "
                f"area={r['total_overlap_area']:>8.2f}mm²  "
                f"[intra: {r['intra_macro_pairs']} pairs / {r['intra_macro_area']:.2f}mm², "
                f"inter: {r['inter_macro_pairs']} pairs / {r['inter_macro_area']:.2f}mm²]"
            )
        except Exception as e:
            print(f"{board:<10} ERROR: {type(e).__name__}: {e}")
            results.append({"board": board, "error": str(e)})
    print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
