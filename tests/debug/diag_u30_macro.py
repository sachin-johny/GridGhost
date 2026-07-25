"""Inspect the actual U30 macro: offsets assigned to each cap + pairwise overlaps.

Reveals the true collapse mechanism (not the README's framing of it).
"""

from __future__ import annotations

import sys
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from parsers.kicad_parser import KiCadParser
from assign.assign_caps import assign_caps, IC_TYPES
from models.macro import Macro, MAX_CAP_IC_GAP_MM


def main():
    pcb = sys.argv[1] if len(sys.argv) > 1 else "test4"
    target_ic = sys.argv[2] if len(sys.argv) > 2 else "U30"
    pcb_path = ROOT / "tests" / "test_pcbs" / f"{pcb}.kicad_pcb"
    parser = KiCadParser(str(pcb_path), bbox_margin=0.8)
    model = parser.parse()

    decap_map = assign_caps(model)
    cap_refs = decap_map.get(target_ic, [])
    leader = model.get_component(target_ic)
    print(f"=== {target_ic} on {pcb} ===")
    print(f"Leader eff size: {leader.effective_width:.2f} x {leader.effective_height:.2f} mm")
    print(f"Leader body (w/h): {leader.width:.2f} x {leader.height:.2f} mm")
    print(f"Assigned caps: {len(cap_refs)} -> {sorted(cap_refs)}")
    print(f"MAX_CAP_IC_GAP_MM = {MAX_CAP_IC_GAP_MM}")

    cap_comps = [model.get_component(r) for r in cap_refs]
    cap_comps = [c for c in cap_comps if c is not None]
    macro = Macro.with_caps(leader, cap_comps)

    print(f"\n--- Assigned offsets (leader-local, rot=0) ---")
    print(f"{'cap':<6} {'value':<6} {'effsize':<10} {'offset(mm)':<18} {'dist(mm)':<8} {'overlaps_leader':<16} {'n_sibling_ovl':<12}")
    for cap, (dx, dy) in zip(macro.followers, macro.follower_offsets):
        dist = math.hypot(dx, dy)
        ov_leader = cap.overlaps(leader)
        n_sib = sum(1 for o in cap_comps if o is not cap and cap.overlaps(o))
        sz = f"{cap.effective_width:.1f}x{cap.effective_height:.1f}"
        print(f"{cap.ref:<6} {getattr(cap,'value',''):<6} {sz:<10} ({dx:+.2f},{dy:+.2f})   {dist:<8.2f} {str(ov_leader):<16} {n_sib}")

    # Pairwise overlap matrix
    print(f"\n--- Pairwise cap-cap overlap areas (mm^2) ---")
    members = [leader] + cap_comps
    total_pairs = 0
    overlap_pairs = 0
    total_area = 0.0
    for i, a in enumerate(members):
        for b in members[i + 1:]:
            total_pairs += 1
            area = a.overlap_area(b)
            if area > 0:
                overlap_pairs += 1
                total_area += area
                print(f"  {a.ref} <-> {b.ref}: {area:.3f} mm^2")
    print(f"\nTotal pairs: {total_pairs}, overlapping: {overlap_pairs}, total area: {total_area:.2f} mm^2")

    # How many DISTINCT slots did caps land on?
    slots = {}
    for cap, (dx, dy) in zip(macro.followers, macro.follower_offsets):
        key = (round(dx, 2), round(dy, 2))
        slots.setdefault(key, []).append(cap.ref)
    print(f"\n--- Distinct landing slots ---")
    for slot, refs in slots.items():
        marker = "  <-- COLLISION" if len(refs) > 1 else ""
        print(f"  offset {slot}: {refs}{marker}")


if __name__ == "__main__":
    main()
