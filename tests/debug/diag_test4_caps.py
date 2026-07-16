"""Inspect cap population on a board: values, footprints, nets, IC sharing."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from parsers.kicad_parser import KiCadParser
from assign.assign_caps import (
    classify_caps, parse_cap_value, DECAP_MAX_VALUE_F,
    is_power_net, is_ground_net, IC_TYPES,
)


def main():
    pcb = sys.argv[1] if len(sys.argv) > 1 else "test4"
    pcb_path = ROOT / "tests" / "test_pcbs" / f"{pcb}.kicad_pcb"
    parser = KiCadParser(str(pcb_path), bbox_margin=0.8)
    model = parser.parse()

    print(f"\n=== {pcb} ===")
    print(f"Board: {model.board.width:.1f} x {model.board.height:.1f}mm, {len(model.components)} comps")

    # Component type breakdown
    by_type = defaultdict(int)
    for c in model.components:
        by_type[c.component_type] += 1
    print(f"By type: {dict(by_type)}")

    ics = [c for c in model.components if c.component_type in IC_TYPES]
    caps = [c for c in model.components if c.component_type == "capacitor"]
    print(f"ICs: {len(ics)}, Caps: {len(caps)}")

    # All cap values + footprints
    print("\n--- All caps (ref, value, footprint, dimensions) ---")
    for c in sorted(caps, key=lambda x: (getattr(x, "value", ""), x.ref)):
        v = getattr(c, "value", "") or ""
        fp = getattr(c, "footprint", "") or ""
        print(f"  {c.ref:<8}  val={v:<12}  fp={fp:<35}  {c.effective_width:.1f}x{c.effective_height:.1f}mm")

    # Group caps by value
    by_value = defaultdict(list)
    for c in caps:
        v = (getattr(c, "value", "") or "").strip()
        by_value[v].append(c.ref)
    print("\n--- Caps grouped by value ---")
    for v, refs in sorted(by_value.items(), key=lambda kv: -len(kv[1])):
        print(f"  '{v}': {len(refs)} caps -> {', '.join(sorted(refs)[:8])}{'...' if len(refs) > 8 else ''}")

    # Per-cap nets — see whether each cap is on a power net shared with an IC
    print("\n--- Per-cap nets & IC power sharing ---")
    ic_refs = {c.ref for c in ics}
    for cap in sorted(caps, key=lambda x: x.ref):
        cap_nets = model.nets_for_component(cap.ref)  # list of Net objects
        names = [n.name for n in cap_nets]
        power_nets = [n for n in names if is_power_net(n)]
        gnd_nets = [n for n in names if is_ground_net(n)]
        signal_nets = [n for n in names if not is_power_net(n) and not is_ground_net(n)]
        # For each power net, see if any IC shares it
        shared_with = []
        for pn in power_nets:
            net = model.get_net(pn)
            if net:
                shared_ics = set(net.component_refs) & ic_refs
                if shared_with is not None:
                    shared_with.append((pn, sorted(shared_ics)))
        sig_str = ",".join(signal_nets[:3]) if signal_nets else "-"
        if len(signal_nets) > 3:
            sig_str += f" +{len(signal_nets)-3} more"
        shares_ic = bool(shared_with)
        print(f"  {cap.ref:<8}  P={power_nets}  G={gnd_nets}  S={sig_str}  IC_shared={shared_with if shared_with else '-'}")

    # What classify_caps returned
    decap_map, bulk_refs, coupling_refs = classify_caps(model)
    print(f"\n--- classify_caps result ---")
    print(f"  Decoupling (followers): {sum(len(v) for v in decap_map.values())} caps across {len(decap_map)} ICs")
    print(f"  Bulk (standalone):      {len(bulk_refs)} caps")
    print(f"  Coupling (standalone):  {len(coupling_refs)} caps")
    print(f"  Total:                  {sum(len(v) for v in decap_map.values()) + len(bulk_refs) + len(coupling_refs)}")

    print(f"\n--- Decoupling assignment per IC ---")
    assigned_caps = set()
    for ic_ref, cap_refs in sorted(decap_map.items()):
        assigned_caps.update(cap_refs)
        print(f"  {ic_ref}: {len(cap_refs)} caps -> {sorted(cap_refs)}")

    print(f"\n--- Bulk caps (standalone) ---")
    for ref in sorted(bulk_refs):
        comp = model.get_component(ref)
        v = getattr(comp, "value", "") if comp else "?"
        val_f = parse_cap_value(v or "")
        size = f"{val_f*1e6:.2f}uF" if val_f else "?"
        print(f"  {ref}: value={v!r} ({size})")

    print(f"\n--- Coupling caps (standalone) ---")
    for ref in sorted(coupling_refs):
        comp = model.get_component(ref)
        v = getattr(comp, "value", "") if comp else "?"
        print(f"  {ref}: value={v!r}")

    unassigned = [c.ref for c in caps if c.ref not in assigned_caps]
    print(f"\nAll caps accounted for: {len(unassigned) == len(bulk_refs) + len(coupling_refs)} (unassigned={len(unassigned)}, bulk+coupling={len(bulk_refs)+len(coupling_refs)})")


if __name__ == "__main__":
    main()
