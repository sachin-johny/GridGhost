"""Measure the two PLACEMENT_FIX_PLAN.md regression metrics across boards.

Issue 1 (spread): interior-footprint-bbox area / board area.  The plan
caught the center-collapse bug at 39%; the fix targets >~60%.

Issue 2 (edge centering): for each board edge, the along-edge span
midpoint of the connectors placed on that edge vs the true edge
midpoint.  The plan caught 6.4mm on cbb's left edge; the fix targets
offset ≈ 0 (within ~0.5mm).

Run:  python tests/debug/diagnose_fix_plan.py [board_stem ...]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from parsers.kicad_parser import KiCadParser
from place.pipeline import place_v2


def _classify_edge(comp, board) -> str | None:
    """Nearest board edge to a connector's bbox center (bottom/top/left/right)."""
    bx1, by1, bx2, by2 = comp.bbox
    cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
    d = {
        "bottom": board.y_max - cy,
        "top": cy - board.y_min,
        "right": board.x_max - cx,
        "left": cx - board.x_min,
    }
    return min(d, key=d.get)


def issue1_spread(model) -> float:
    """Interior-footprint-bbox area / board area (%)."""
    interior = [
        c for c in model.components
        if not c.is_fixed and not getattr(c, "is_edge_connector", False)
    ]
    if not interior:
        return 0.0
    xmin = min(c.bbox[0] for c in interior)
    ymin = min(c.bbox[1] for c in interior)
    xmax = max(c.bbox[2] for c in interior)
    ymax = max(c.bbox[3] for c in interior)
    fp_area = max(0.0, (xmax - xmin)) * max(0.0, (ymax - ymin))
    board_area = model.board.width * model.board.height
    if board_area <= 0:
        return 0.0
    return 100.0 * fp_area / board_area


def issue2_edge_offsets(model) -> dict[str, float]:
    """Per-edge |group span midpoint − true edge midpoint| in mm."""
    board = model.board
    conns = [
        c for c in model.components
        if getattr(c, "is_edge_connector", False)
    ]
    by_edge: dict[str, list] = {}
    for c in conns:
        e = _classify_edge(c, board)
        by_edge.setdefault(e, []).append(c)

    mid_x = (board.x_min + board.x_max) / 2
    mid_y = (board.y_min + board.y_max) / 2
    true_mid = {"bottom": mid_x, "top": mid_x, "left": mid_y, "right": mid_y}

    out: dict[str, float] = {}
    for e, comps in by_edge.items():
        if not comps:
            continue
        if e in ("bottom", "top"):
            lo = min(c.bbox[0] for c in comps)
            hi = max(c.bbox[2] for c in comps)
        else:
            lo = min(c.bbox[1] for c in comps)
            hi = max(c.bbox[3] for c in comps)
        group_mid = (lo + hi) / 2
        out[e] = abs(group_mid - true_mid[e])
    return out


def overlaps_and_oob(model) -> tuple[int, int]:
    overlaps = 0
    comps = model.components
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            if comps[i].overlaps(comps[j]):
                overlaps += 1
    oob = 0
    for c in comps:
        if getattr(c, "is_edge_connector", False):
            continue
        x1, y1, x2, y2 = c.bbox
        if (x1 < model.board.x_min or y1 < model.board.y_min or
                x2 > model.board.x_max or y2 > model.board.y_max):
            oob += 1
    return overlaps, oob


def run_board(pcb_path):
    parser = KiCadParser(str(pcb_path), bbox_margin=0.8)
    model = parser.parse()
    try:
        place_v2(model, sa_iterations=800, sa_reheats=2, seed=42, verbose=False)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"

    spread = issue1_spread(model)
    offs = issue2_edge_offsets(model)
    overlaps, oob = overlaps_and_oob(model)
    max_off = max(offs.values()) if offs else 0.0
    edge_str = " ".join(f"{e[0]}={v:.1f}" for e, v in sorted(offs.items()))

    flag = "  <-- SPREAD<60" if spread < 60.0 else ""
    return (
        f"spread={spread:5.1f}%  maxEdgeOff={max_off:4.1f}mm  "
        f"[{edge_str}]  overlaps={overlaps:3d}  oob={oob}{flag}"
    )


def main():
    boards_dir = ROOT / "tests" / "test_pcbs"
    stems = sys.argv[1:]
    boards = sorted(boards_dir.glob("*.kicad_pcb"))
    if stems:
        boards = [b for b in boards if b.stem in stems]

    print(f"{'Board':<14} {'Result'}")
    print("-" * 100)
    for pcb in boards:
        print(f"{pcb.stem:<14} {run_board(pcb)}")


if __name__ == "__main__":
    main()
