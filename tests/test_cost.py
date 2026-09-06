"""Unit tests for cost function: HPWL + overlap + boundary."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.board_model import BoardModel, BoardOutline, Component, Net
from models.macro import Macro
from cost.cost import (
    hpwl_net,
    total_hpwl,
    total_macro_overlap,
    total_boundary,
    evaluate,
    cap_attraction_penalty,
    clearance_pair_charge,
    clearance_deficit,
)

passed = 0
failed = 0


def run(name, func):
    global passed, failed
    try:
        func()
        print(f"  PASS  {name}")
        passed += 1
    except AssertionError as e:
        print(f"  FAIL  {name}: {e}")
        failed += 1
    except Exception as e:
        print(f"  ERROR {name}: {type(e).__name__}: {e}")
        failed += 1


def _comp(ref, x, y, w=2.0, h=2.0, ctype="generic"):
    return Component(
        ref=ref, x=x, y=y, width=w, height=h,
        courtyard_margin=0.0, component_type=ctype,
    )


def test_hpwl_two_pin_horizontal():
    a = _comp("R1", 0.0, 0.0)
    b = _comp("R2", 10.0, 0.0)
    ref_map = {"R1": a, "R2": b}
    net = Net("N1", [("R1", "1"), ("R2", "1")])
    # HPWL = (max_x - min_x) + (max_y - min_y) = 10 + 0 = 10
    assert hpwl_net(net, ref_map) == 10.0


def test_hpwl_two_pin_2d():
    a = _comp("R1", 0.0, 0.0)
    b = _comp("R2", 3.0, 4.0)
    ref_map = {"R1": a, "R2": b}
    net = Net("N1", [("R1", "1"), ("R2", "1")])
    assert hpwl_net(net, ref_map) == 7.0  # 3 + 4


def test_hpwl_three_pin():
    a = _comp("R1", 0.0, 0.0)
    b = _comp("R2", 6.0, 0.0)
    c = _comp("R3", 3.0, 4.0)
    ref_map = {"R1": a, "R2": b, "R3": c}
    net = Net("N1", [("R1", "1"), ("R2", "1"), ("R3", "1")])
    # Bbox: x=[0,6], y=[0,4]. HPWL = 6 + 4 = 10
    assert hpwl_net(net, ref_map) == 10.0


def test_hpwl_single_pin_zero():
    a = _comp("R1", 0.0, 0.0)
    net = Net("N1", [("R1", "1")])
    assert hpwl_net(net, {"R1": a}) == 0.0


def test_total_hpwl_includes_power_by_default():
    a = _comp("U1", 0.0, 0.0, ctype="ic")
    b = _comp("C1", 5.0, 0.0, ctype="capacitor")
    vcc = Net("+3V3", [("U1", "1"), ("C1", "1")])
    model = BoardModel(
        board=BoardOutline(0, 0, 100, 100),
        components=[a, b],
        nets=[vcc],
    )
    # Default include_power=True
    assert total_hpwl(model) == 5.0


def test_total_hpwl_can_exclude_power():
    a = _comp("U1", 0.0, 0.0, ctype="ic")
    b = _comp("C1", 5.0, 0.0, ctype="capacitor")
    vcc = Net("+3V3", [("U1", "1"), ("C1", "1")])
    model = BoardModel(
        board=BoardOutline(0, 0, 100, 100),
        components=[a, b],
        nets=[vcc],
    )
    # When power is excluded via exclude_nets, HPWL drops to 0.
    assert total_hpwl(model, exclude_nets={"+3V3"}) == 0.0


def test_evaluate_exclude_nets_reduces_hpwl():
    """exclude_nets threaded through evaluate() should reduce HPWL term."""
    a = _comp("U1", 0.0, 0.0, ctype="ic")
    b = _comp("C1", 5.0, 0.0, ctype="capacitor")
    vcc = Net("+3V3", [("U1", "1"), ("C1", "1")])
    sig = Net("SIG", [("U1", "2"), ("C1", "2")])
    model = BoardModel(
        board=BoardOutline(0, 0, 100, 100),
        components=[a, b],
        nets=[vcc, sig],
    )
    macros = [Macro.alone(a), Macro.alone(b)]
    with_exclude = evaluate(model, macros, exclude_nets={"+3V3"})
    without_exclude = evaluate(model, macros, exclude_nets=None)
    # Excluding +3V3 should reduce HPWL (the SIG net's HPWL remains)
    assert with_exclude["hpwl"] < without_exclude["hpwl"]
    # Overlap/boundary are unaffected
    assert with_exclude["overlap"] == without_exclude["overlap"]
    assert with_exclude["boundary"] == without_exclude["boundary"]


def test_macro_overlap_no_overlap():
    a = Macro.alone(_comp("U1", 0, 0, w=10, h=10))
    b = Macro.alone(_comp("U2", 100, 100, w=10, h=10))
    assert total_macro_overlap([a, b]) == 0.0


def test_macro_overlap_partial():
    a = Macro.alone(_comp("U1", 0, 0, w=10, h=10))   # bbox -5..5
    b = Macro.alone(_comp("U2", 5, 0, w=10, h=10))   # bbox 0..10
    # Overlap in x: max(-5,0)=0, min(5,10)=5 → width 5
    # Overlap in y: max(-5,-5)=-5, min(5,5)=5 → height 10
    # Area = 5 * 10 = 50
    assert total_macro_overlap([a, b]) == 50.0


def test_total_boundary_zero_when_in_bounds():
    a = _comp("R1", 50, 50)
    model = BoardModel(
        board=BoardOutline(0, 0, 100, 100),
        components=[a],
        nets=[],
    )
    assert total_boundary(model) == 0.0


def test_total_boundary_counts_overshoot():
    # bbox 95..105 — pokes 5mm past x_max=100
    a = _comp("R1", 100, 50)
    model = BoardModel(
        board=BoardOutline(0, 0, 100, 100),
        components=[a],
        nets=[],
    )
    # Component is 2x2 centered at (100,50): bbox (99,49)-(101,51).
    # x2=101 > 100 by 1.
    assert abs(total_boundary(model) - 1.0) < 1e-9


def test_boundary_excludes_edge_connectors():
    """is_edge_connector components don't get penalized for overhang."""
    c = Component(
        ref="J1", x=100, y=50, width=4, height=4,
        courtyard_margin=0.0, component_type="connector",
        footprint="Connector Horizontal",  # triggers edge-connector flag
    )
    model = BoardModel(
        board=BoardOutline(0, 0, 100, 100),
        components=[c],
        nets=[],
    )
    assert total_boundary(model) == 0.0


def test_evaluate_returns_all_components():
    a = _comp("R1", 50, 50)
    b = _comp("R2", 60, 50)
    net = Net("N1", [("R1", "1"), ("R2", "1")])
    model = BoardModel(
        board=BoardOutline(0, 0, 100, 100),
        components=[a, b],
        nets=[net],
    )
    ma = Macro.alone(a)
    mb = Macro.alone(b)
    costs = evaluate(model, [ma, mb])
    assert "hpwl" in costs
    assert "overlap" in costs
    assert "boundary" in costs
    assert "total" in costs
    assert costs["hpwl"] == 10.0
    assert costs["overlap"] == 0.0
    assert costs["boundary"] == 0.0


def test_cap_attraction_penalty_deadband():
    # Cap at 4mm from its IC (inside the 5mm deadband) -> zero charge.
    ic = _comp("U1", 50, 50, ctype="ic")
    cap = _comp("C1", 54, 50, w=1.0, h=0.5, ctype="capacitor")
    model = BoardModel(
        board=BoardOutline(0, 0, 100, 100),
        components=[ic, cap],
        nets=[],
    )
    assert cap_attraction_penalty(model, {"C1": "U1"}) == 0.0

    # Cap at 12mm -> charge = 12 - 5 = 7 (deadband-linear).
    cap.x = 62
    assert cap_attraction_penalty(model, {"C1": "U1"}) == 7.0


def test_cap_attraction_penalty_sums_pairs():
    ic = _comp("U1", 50, 50, ctype="ic")
    c1 = _comp("C1", 70, 50, w=1.0, h=0.5, ctype="capacitor")     # 20mm -> 15
    c2 = _comp("C2", 50, 58, w=1.0, h=0.5, ctype="capacitor")     # 8mm -> 3
    model = BoardModel(
        board=BoardOutline(0, 0, 100, 100),
        components=[ic, c1, c2],
        nets=[],
    )
    assert cap_attraction_penalty(model, {"C1": "U1", "C2": "U1"}) == 18.0


def test_evaluate_cap_attraction_weighted():
    ic = _comp("U1", 50, 50, ctype="ic")
    cap = _comp("C1", 70, 50, w=1.0, h=0.5, ctype="capacitor")  # 20mm -> 15
    model = BoardModel(
        board=BoardOutline(0, 0, 100, 100),
        components=[ic, cap],
        nets=[],
    )
    mi = Macro.alone(ic)
    mc = Macro.alone(cap)
    base = evaluate(model, [mi, mc])
    with_attr = evaluate(model, [mi, mc],
                         cap_attraction_weight=2.0, cap_pairs={"C1": "U1"})
    assert with_attr["cap_attraction"] == 15.0
    assert with_attr["total"] == base["total"] + 2.0 * 15.0
    # Weight 0 (default) -> term absent, total unchanged.
    assert evaluate(model, [mi, mc], cap_pairs={"C1": "U1"})["total"] == base["total"]


def test_clearance_pair_charge_gaps():
    a = (0.0, 0.0, 10.0, 10.0)
    # 0.5mm gap in x, aligned in y -> deficit 0.5 (target 1.0).
    assert clearance_pair_charge(a, (10.5, 0.0, 20.0, 10.0)) == 0.5
    # Full target gap -> zero charge.
    assert clearance_pair_charge(a, (11.0, 0.0, 20.0, 10.0)) == 0.0
    # Far apart -> zero charge.
    assert clearance_pair_charge(a, (50.0, 50.0, 60.0, 60.0)) == 0.0
    # Touching-but-not-intersecting (gap 0, diagonal) -> full target.
    assert clearance_pair_charge(a, (10.0, 10.0, 20.0, 20.0)) == 1.0


def test_clearance_pair_charge_intersect_caps_at_target():
    a = (0.0, 0.0, 10.0, 10.0)
    # Bboxes intersect -> charge is exactly the target (never deeper):
    # β·overlap owns depth-charging, so the halo term must not
    # double-charge once bboxes actually intersect.
    assert clearance_pair_charge(a, (5.0, 5.0, 15.0, 15.0)) == 1.0
    # A deep 90% intersection charges the same 1.0, not more.
    assert clearance_pair_charge(a, (9.0, 9.0, 19.0, 19.0)) == 1.0


def test_clearance_deficit_sums_and_respects_target():
    # Use ctype="ic" so the per-class target table assigns the historical
    # uniform 1.0mm target (ic class). Verifies the high-level dispatch +
    # math primitive together. See _COMPONENT_CLEARANCE_TARGETS_MM.
    a = Macro.alone(_comp("U1", 50, 50, w=4, h=4, ctype="ic"))   # bbox 48..52
    b = Macro.alone(_comp("U2", 55, 50, w=4, h=4, ctype="ic"))   # bbox 53..57
    c = Macro.alone(_comp("U3", 90, 90, w=4, h=4, ctype="ic"))
    # U1-U2 gap = 1.0mm (53-52) -> satisfied at target 1.0, zero charge.
    assert clearance_deficit([a, b, c]) == 0.0
    # U2 slides to gap 0.6 -> deficit 0.4.
    b.leader.x = 54.6
    assert abs(clearance_deficit([a, b, c]) - 0.4) < 1e-9
    # Custom target widens the charged band (explicit override applies
    # because ic class target = 1.0 and explicit 2.0 is larger — but our
    # implementation uses per-class table strictly, so we expect 0.4 still
    # from the ic target. To verify explicit override math, call the
    # primitive directly).
    assert abs(clearance_pair_charge(a.bbox, b.bbox, 2.0) - 1.4) < 1e-9


def test_clearance_deficit_skips_mechanical_pairs():
    # Mounting-hole macros with overlapping pad-stack bboxes are exempt
    # (same exemption as the overlap term — fixed mechanical features).
    m1 = Macro.alone(_comp("H1", 50, 50, w=8, h=8, ctype="mounting_hole"))
    m2 = Macro.alone(_comp("H2", 54, 54, w=8, h=8, ctype="mounting_hole"))
    # Bboxes intersect (46..54 vs 50..58) — exempt pair charges 0.
    assert clearance_deficit([m1, m2]) == 0.0
    # An electrical pair in the same geometry (IC class, 1.0mm target)
    # charges the full target since bboxes intersect.
    e1 = Macro.alone(_comp("U1", 50, 50, w=8, h=8, ctype="ic"))
    e2 = Macro.alone(_comp("U2", 54, 54, w=8, h=8, ctype="ic"))
    assert clearance_deficit([e1, e2]) == 1.0


def test_clearance_deficit_per_class_targets():
    """Per-component-class targets: only TestPoints and mechanical-mechanical
    pairs are exempt; all other types fall back to the caller's target_mm.
    Verifies the TestPoint exemption fix for test4's TP16 touching U1's pin."""
    # Cap-cap pair at 0.5mm gap with default target 1.0mm -> deficit 0.5.
    cap1 = Macro.alone(_comp("C1", 50, 50, w=1.0, h=0.5, ctype="capacitor"))
    cap2 = Macro.alone(_comp("C2", 51.5, 50, w=1.0, h=0.5, ctype="capacitor"))
    # cap1 bbox 49.5..50.5, cap2 bbox 51.0..52.0 -> gap = 0.5mm
    # All component types now use the default 1.0mm target (only
    # mechanical-mechanical pairs and pairs containing a TestPoint
    # are exempt).
    assert abs(clearance_deficit([cap1, cap2]) - 0.5) < 1e-9  # 1.0 - 0.5 = 0.5

    # Cap next to IC at 0.5mm gap: same deficit 0.5 (MIN(1.0, 1.0) = 1.0).
    ic = Macro.alone(_comp("U1", 50, 50, w=4, h=4, ctype="ic"))
    cap = Macro.alone(_comp("C1", 53.0, 50, w=1.0, h=0.5, ctype="capacitor"))
    # ic bbox 48..52, cap bbox 52.5..53.5 -> gap = 0.5mm
    assert abs(clearance_deficit([ic, cap]) - 0.5) < 1e-9


def test_clearance_deficit_testpoint_exempt():
    """TestPoint pairs are exempt from the clearance term (intentional tight
    placement for probe access — e.g. test4's TP16 touching U1's pin)."""
    # Build a fake component with a TestPoint footprint (component_type
    # stays "generic" — the parser doesn't special-case test points).
    tp = Component(
        ref="TP1", x=50, y=50, width=1.0, height=1.0,
        courtyard_margin=0.0, component_type="generic",
        footprint="TestPoint:TestPoint_Pad_D1.0mm",
    )
    ic = Macro.alone(_comp("U1", 50, 50, w=4, h=4, ctype="ic"))
    tp_macro = Macro.alone(tp)
    # Bboxes overlap (TP inside IC body) — exempt pair, zero charge.
    assert clearance_deficit([ic, tp_macro]) == 0.0


def test_evaluate_clearance_weighted():
    # IC class for 1.0mm per-class target (matches the historical uniform
    # target, so the deficit math stays the same).
    a = _comp("U1", 50, 50, w=4, h=4, ctype="ic")   # bbox 48..52
    b = _comp("U2", 54.6, 50, w=4, h=4, ctype="ic")  # bbox 52.6..56.6 -> gap 0.6 -> deficit 0.4
    model = BoardModel(
        board=BoardOutline(0, 0, 100, 100),
        components=[a, b],
        nets=[],
    )
    ma, mb = Macro.alone(a), Macro.alone(b)
    base = evaluate(model, [ma, mb])
    with_clr = evaluate(model, [ma, mb], clearance_weight=5.0)
    assert abs(with_clr["clearance"] - 0.4) < 1e-9
    assert abs(with_clr["total"] - (base["total"] + 5.0 * 0.4)) < 1e-6
    # Weight 0 (default) -> term absent from the total.
    assert evaluate(model, [ma, mb])["total"] == base["total"]


def main():
    print("=" * 60)
    print("  Cost function tests")
    print("=" * 60)
    run("HPWL 2-pin horizontal", test_hpwl_two_pin_horizontal)
    run("HPWL 2-pin 2D", test_hpwl_two_pin_2d)
    run("HPWL 3-pin", test_hpwl_three_pin)
    run("HPWL single pin = 0", test_hpwl_single_pin_zero)
    run("total_hpwl includes power by default", test_total_hpwl_includes_power_by_default)
    run("total_hpwl can exclude power", test_total_hpwl_can_exclude_power)
    run("macro overlap: no overlap", test_macro_overlap_no_overlap)
    run("macro overlap: partial", test_macro_overlap_partial)
    run("boundary: zero in-bounds", test_total_boundary_zero_when_in_bounds)
    run("boundary: counts overshoot", test_total_boundary_counts_overshoot)
    run("boundary: excludes edge connectors", test_boundary_excludes_edge_connectors)
    run("evaluate returns all components", test_evaluate_returns_all_components)
    run("cap attraction: deadband", test_cap_attraction_penalty_deadband)
    run("cap attraction: sums pairs", test_cap_attraction_penalty_sums_pairs)
    run("cap attraction: weighted in evaluate", test_evaluate_cap_attraction_weighted)
    run("clearance pair charge: gap cases", test_clearance_pair_charge_gaps)
    run("clearance pair charge: intersect caps at target", test_clearance_pair_charge_intersect_caps_at_target)
    run("clearance deficit: sums and respects target", test_clearance_deficit_sums_and_respects_target)
    run("clearance deficit: skips mechanical pairs", test_clearance_deficit_skips_mechanical_pairs)
    run("clearance: weighted in evaluate", test_evaluate_clearance_weighted)
    print("=" * 60)
    print(f"  {passed} passed, {failed} failed")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
