"""Unit tests for cap-IC assignment."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.board_model import BoardModel, BoardOutline, Component, Net
from assign.assign_caps import assign_caps, classify_caps, is_power_net, is_ground_net

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


def _ic(ref, nets):
    return Component(
        ref=ref, x=0, y=0, width=10, height=10,
        courtyard_margin=0.0, component_type="ic", nets=nets,
    )


def _cap(ref, nets):
    return Component(
        ref=ref, x=0, y=0, width=2, height=1,
        courtyard_margin=0.0, component_type="capacitor", nets=nets,
        value="100n",
    )


def _model_with(components, nets):
    return BoardModel(
        board=BoardOutline(0, 0, 100, 100),
        components=components,
        nets=nets,
    )


def test_power_net_detection():
    assert is_power_net("+3V3")
    assert is_power_net("/+12V")
    assert is_power_net("VCC")
    assert is_power_net("VBAT")
    assert not is_power_net("Net-(U1-Out)"), "Signal net misclassified as power"
    assert not is_power_net("/SDA"), "Signal net misclassified as power"


def test_ground_net_detection():
    assert is_ground_net("GND")
    assert is_ground_net("AGND")
    assert is_ground_net("/DGND")
    assert not is_ground_net("+3V3")


def test_single_ic_single_cap_assignment():
    u1 = _ic("U1", ["+3V3", "GND"])
    c1 = _cap("C1", ["+3V3", "GND"])
    net = Net("+3V3", [("U1", "1"), ("C1", "1")])
    gnd = Net("GND", [("U1", "2"), ("C1", "2")])
    model = _model_with([u1, c1], [net, gnd])

    m = assign_caps(model)
    assert m == {"U1": ["C1"]}, f"Expected U1->[C1], got {m}"


def test_cap_assigned_to_exactly_one_ic():
    u1 = _ic("U1", ["+3V3", "GND"])
    u2 = _ic("U2", ["+3V3", "GND"])
    c1 = _cap("C1", ["+3V3", "GND"])
    vcc = Net("+3V3", [("U1", "1"), ("U2", "1"), ("C1", "1")])
    gnd = Net("GND", [("U1", "2"), ("U2", "2"), ("C1", "2")])
    model = _model_with([u1, u2, c1], [vcc, gnd])

    m = assign_caps(model)
    # C1 must appear in exactly one IC's list.
    caps_assigned = sum(1 for refs in m.values() if "C1" in refs)
    assert caps_assigned == 1, f"C1 assigned to {caps_assigned} ICs (expected 1)"


def test_caps_distribute_round_robin_across_ics():
    """When 2 ICs share +3V3 and there are 2 caps, each IC should get one."""
    u1 = _ic("U1", ["+3V3"])
    u2 = _ic("U2", ["+3V3"])
    c1 = _cap("C1", ["+3V3"])
    c2 = _cap("C2", ["+3V3"])
    vcc = Net("+3V3", [("U1", "1"), ("U2", "1"), ("C1", "1"), ("C2", "1")])
    model = _model_with([u1, u2, c1, c2], [vcc])

    m = assign_caps(model)
    assert len(m.get("U1", [])) == 1, f"U1 got {m.get('U1')} (expected 1 cap)"
    assert len(m.get("U2", [])) == 1, f"U2 got {m.get('U2')} (expected 1 cap)"


def test_unassigned_cap_not_in_map():
    """A cap not on any power net sharing with an IC is left out."""
    u1 = _ic("U1", ["+3V3"])
    c1 = _cap("C1", ["Net-(U1-Out)"])  # signal net, not power
    vcc = Net("+3V3", [("U1", "1")])
    sig = Net("Net-(U1-Out)", [("U1", "2"), ("C1", "1")])
    model = _model_with([u1, c1], [vcc, sig])

    m = assign_caps(model)
    # U1 has no caps assigned because C1 doesn't share a power net.
    assert m == {}, f"Expected empty assignment, got {m}"


def test_cap_on_multiple_rails_picks_least_loaded_ic():
    """Cap shares +3V3 with U1, +12V with U2. Either is a valid candidate;
    assignment picks deterministically via round-robin."""
    u1 = _ic("U1", ["+3V3"])  # 0 caps so far
    u2 = _ic("U2", ["+12V"])  # 0 caps so far
    c1 = _cap("C1", ["+3V3", "+12V"])  # shares with both
    vcc33 = Net("+3V3", [("U1", "1"), ("C1", "1")])
    vcc12 = Net("+12V", [("U2", "1"), ("C1", "2")])
    model = _model_with([u1, u2, c1], [vcc33, vcc12])

    m = assign_caps(model)
    # Tie-break: sorted(["U1", "U2"]) picks U1.
    assert "U1" in m and "C1" in m["U1"], f"Expected C1 -> U1, got {m}"


def test_deterministic_across_runs():
    """Same model → same assignment (sorted iteration)."""
    u1 = _ic("U1", ["+3V3"])
    u2 = _ic("U2", ["+3V3"])
    caps = [_cap(f"C{i}", ["+3V3"]) for i in range(4)]
    pins = [("U1", "1"), ("U2", "1")] + [(c.ref, "1") for c in caps]
    vcc = Net("+3V3", pins)
    model = _model_with([u1, u2] + caps, [vcc])

    m1 = assign_caps(model)
    m2 = assign_caps(model)
    assert m1 == m2, "Non-deterministic assignment"


def test_max_decaps_per_ic_limits_rigid_followers():
    """classify_caps returns at most max_decaps_per_ic rigid followers per IC.
    Excess caps become rail-adjacent (standalone)."""
    u1 = _ic("U1", ["+3V3"])
    u2 = _ic("U2", ["+3V3"])
    caps = [_cap(f"C{i}", ["+3V3"]) for i in range(6)]
    pins = [("U1", "1"), ("U2", "1")] + [(c.ref, "1") for c in caps]
    vcc = Net("+3V3", pins)
    model = _model_with([u1, u2] + caps, [vcc])

    # With default max_decaps_per_ic=2, each IC should get at most 2 rigid caps
    decap_map, rail_adj, bulk, coupling = classify_caps(model)
    for ic_ref, cap_refs in decap_map.items():
        assert len(cap_refs) <= 2, f"{ic_ref} has {len(cap_refs)} rigid followers (max 2)"
    # 6 caps total, 4 rigid (2 per IC), 2 rail-adjacent
    total_rigid = sum(len(v) for v in decap_map.values())
    assert total_rigid == 4, f"Expected 4 rigid caps, got {total_rigid}"
    assert len(rail_adj) == 2, f"Expected 2 rail-adjacent caps, got {len(rail_adj)}"


def test_rail_adjacent_caps_are_standalone():
    """Rail-adjacent caps are NOT in assign_caps() result → standalone macros."""
    u1 = _ic("U1", ["+3V3"])
    caps = [_cap(f"C{i}", ["+3V3"]) for i in range(5)]
    pins = [("U1", "1")] + [(c.ref, "1") for c in caps]
    vcc = Net("+3V3", pins)
    model = _model_with([u1] + caps, [vcc])

    decap_map = assign_caps(model)
    # Only 2 rigid followers (max_decaps_per_ic=2)
    assert len(decap_map.get("U1", [])) == 2, f"Expected 2 rigid, got {decap_map}"

    # Full classification shows 3 rail-adjacent
    _, rail_adj, _, _ = classify_caps(model)
    assert len(rail_adj) == 3, f"Expected 3 rail-adjacent, got {len(rail_adj)}"


def main():
    print("=" * 60)
    print("  Cap-IC assignment tests")
    print("=" * 60)
    run("power net detection", test_power_net_detection)
    run("ground net detection", test_ground_net_detection)
    run("single IC + single cap", test_single_ic_single_cap_assignment)
    run("cap assigned to exactly one IC", test_cap_assigned_to_exactly_one_ic)
    run("round-robin distribution", test_caps_distribute_round_robin_across_ics)
    run("unassigned cap not in map", test_unassigned_cap_not_in_map)
    run("multi-rail cap picks least-loaded IC", test_cap_on_multiple_rails_picks_least_loaded_ic)
    run("deterministic across runs", test_deterministic_across_runs)
    run("max_decaps_per_ic limits rigid followers", test_max_decaps_per_ic_limits_rigid_followers)
    run("rail-adjacent caps are standalone", test_rail_adjacent_caps_are_standalone)
    print("=" * 60)
    print(f"  {passed} passed, {failed} failed")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
