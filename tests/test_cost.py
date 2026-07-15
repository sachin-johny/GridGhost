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
    print("=" * 60)
    print(f"  {passed} passed, {failed} failed")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
