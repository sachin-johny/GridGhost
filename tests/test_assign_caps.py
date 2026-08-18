"""Unit tests for cap-IC assignment."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.board_model import BoardModel, BoardOutline, Component, Net, Pad
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


def _ic(ref, nets, pads=None):
    """``pads``: optional list of (pad_name, x, y, net) tuples. Omitting
    it means "no pad-geometry data" — rigidity then defaults to tight
    (see ``_rail_pad_spread``'s docstring)."""
    return Component(
        ref=ref, x=0, y=0, width=10, height=10,
        courtyard_margin=0.0, component_type="ic", nets=nets,
        pads=[Pad(pad_name=n, x=x, y=y, net=net) for n, x, y, net in (pads or [])],
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


def test_power_net_detection_hidden_rails():
    """Regulator VIN/VOUT and local battery rails are real power nets
    even when KiCad auto-names them (no global net label was placed in
    the schematic). Found on test5/test6: Net-(U1-VIN) and
    /Battery Protect/P+ were silently falling through to 'coupling',
    dropping real decoupling caps from IC-proximity handling entirely."""
    assert is_power_net("Net-(U1-VIN)"), "Regulator VIN pin should read as power"
    assert is_power_net("Net-(U1-VOUT)"), "Regulator VOUT pin should read as power"
    assert is_power_net("/Buck/VIN"), "Hierarchical local VIN label should read as power"
    assert is_power_net("/Battery Protect/P+"), "Local battery rail should read as power"
    # Control/feedback/switching pins on the same IC are NOT power rails,
    # even though they're electrically adjacent — must stay excluded.
    assert not is_power_net("Net-(U1-BOOT)"), "Bootstrap cap net is not a power rail"
    assert not is_power_net("Net-(U1-COMP)"), "Compensation net is not a power rail"
    assert not is_power_net("Net-(U1-EN)"), "Enable pin is not a power rail"
    assert not is_power_net("Net-(U2-VG)"), "Gate-drive net is not a power rail"
    assert not is_power_net("/ByPass/OUTBP"), "Non-power local label stays excluded"


def test_ground_net_detection():
    assert is_ground_net("GND")
    assert is_ground_net("AGND")
    assert is_ground_net("/DGND")
    assert not is_ground_net("+3V3")


def test_ground_net_detection_hidden_rails():
    """Same hidden-rail blind spot as power nets, mirrored for ground:
    an unlabeled GND pin or a local/battery-negative rail must still
    read as ground even when KiCad auto-names it."""
    assert is_ground_net("Net-(U1-GND)"), "Unlabeled GND pin should read as ground"
    assert is_ground_net("/Analog/AGND"), "Hierarchical local AGND label should read as ground"
    assert is_ground_net("/Battery Protect/P-"), "Battery pack negative is the ground return"
    assert not is_power_net("/Battery Protect/P-"), (
        "Battery pack negative must NOT double as a power rail"
    )
    assert is_power_net("/Battery Protect/P+"), "Battery pack positive stays a power rail"
    assert not is_ground_net("Net-(U1-VIN)"), "Regulator input is power, not ground"


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


def test_hidden_rail_caps_get_decoupling_treatment_end_to_end():
    """Regression test for the test5/test6 finding: a regulator's
    unlabeled VIN net and a hierarchical local battery rail must flow
    all the way through classify_caps into decap_map/rail-adjacent, not
    silently land in 'coupling' just because the net wasn't given a
    global net label."""
    u1 = _ic("U1", ["Net-(U1-VIN)"])  # regulator, auto-named input net
    vin_caps = [_cap(f"C{i}", ["Net-(U1-VIN)"]) for i in range(3)]
    u2 = _ic("U2", ["/Battery Protect/P+"])
    u3 = _ic("U3", ["/Battery Protect/P+"])
    batt_caps = [_cap(f"D{i}", ["/Battery Protect/P+"]) for i in range(2)]

    vin_net = Net("Net-(U1-VIN)", [("U1", "1")] + [(c.ref, "1") for c in vin_caps])
    batt_net = Net(
        "/Battery Protect/P+",
        [("U2", "1"), ("U3", "1")] + [(c.ref, "1") for c in batt_caps],
    )
    model = _model_with([u1, u2, u3] + vin_caps + batt_caps, [vin_net, batt_net])

    decap_map, rail_adj, bulk, coupling = classify_caps(model)
    all_decoupling = set(rail_adj) | {c for caps in decap_map.values() for c in caps}
    assert {c.ref for c in vin_caps} <= all_decoupling, (
        f"VIN caps missing from decoupling classification: {coupling}"
    )
    assert {c.ref for c in batt_caps} <= all_decoupling, (
        f"Battery rail caps missing from decoupling classification: {coupling}"
    )
    assert not (set(coupling) & {c.ref for c in vin_caps + batt_caps}), (
        "Real power-rail caps incorrectly fell through to 'coupling'"
    )



def test_all_real_decaps_stay_rigid_when_pin_group_is_tight():
    """No artificial per-IC cap: when a rail's pads on the IC are (or are
    unknown, so assumed) tight, ALL of that rail's real decoupling caps
    stay rigid — however many the netlist actually put there."""
    u1 = _ic("U1", ["+3V3"])  # no pad data -> unknown geometry -> assumed tight
    u2 = _ic("U2", ["+3V3"])
    caps = [_cap(f"C{i}", ["+3V3"]) for i in range(6)]
    pins = [("U1", "1"), ("U2", "1")] + [(c.ref, "1") for c in caps]
    vcc = Net("+3V3", pins)
    model = _model_with([u1, u2] + caps, [vcc])

    decap_map, rail_adj, bulk, coupling = classify_caps(model)
    # 6 caps round-robin -> 3 per IC, all rigid (tight/unknown geometry).
    total_rigid = sum(len(v) for v in decap_map.values())
    assert total_rigid == 6, f"Expected all 6 caps rigid, got {total_rigid}"
    assert rail_adj == [], f"Expected no rail-adjacent caps, got {rail_adj}"
    for ic_ref, cap_refs in decap_map.items():
        assert len(cap_refs) == 3, f"{ic_ref} has {len(cap_refs)} rigid followers (expected 3)"


def test_scattered_pin_group_goes_rail_adjacent():
    """When a rail's pads on the IC are physically scattered beyond the
    tightness threshold, its real decaps go rail-adjacent — regardless
    of how many there are."""
    # Two +3V3 pads 10mm apart: well beyond RAIL_PAD_SPREAD_THRESHOLD_MM.
    u1 = _ic("U1", ["+3V3"], pads=[
        ("1", -5.0, 0.0, "+3V3"), ("2", 5.0, 0.0, "+3V3"),
    ])
    caps = [_cap(f"C{i}", ["+3V3"]) for i in range(5)]
    pins = [("U1", "1")] + [(c.ref, "1") for c in caps]
    vcc = Net("+3V3", pins)
    model = _model_with([u1] + caps, [vcc])

    decap_map, rail_adj, _, _ = classify_caps(model)
    assert decap_map.get("U1", []) == [], f"Expected no rigid caps, got {decap_map}"
    assert len(rail_adj) == 5, f"Expected all 5 caps rail-adjacent, got {rail_adj}"


def test_tight_pin_group_stays_rigid_even_with_pad_data():
    """A rail whose pads ARE physically close (not just unknown) also
    keeps all its real decaps rigid — confirms geometry, not count,
    drives the split."""
    # Two +3V3 pads 1mm apart: well within the tightness threshold.
    u1 = _ic("U1", ["+3V3"], pads=[
        ("1", -0.5, 0.0, "+3V3"), ("2", 0.5, 0.0, "+3V3"),
    ])
    caps = [_cap(f"C{i}", ["+3V3"]) for i in range(5)]
    pins = [("U1", "1")] + [(c.ref, "1") for c in caps]
    vcc = Net("+3V3", pins)
    model = _model_with([u1] + caps, [vcc])

    decap_map, rail_adj, _, _ = classify_caps(model)
    assert len(decap_map.get("U1", [])) == 5, f"Expected all 5 rigid, got {decap_map}"
    assert rail_adj == [], f"Expected no rail-adjacent caps, got {rail_adj}"


def test_rail_adjacent_to_ic_maps_excess_to_assigned_ic():
    """rail_adjacent_to_ic pairs each freed cap with its round-robin IC.

    With one IC the pairing is trivial; with two ICs the round-robin
    assignment alternates, so the excess caps must map back to the IC
    that actually received them (not just any rail peer).
    """
    from assign.assign_caps import rail_adjacent_to_ic

    u1 = _ic("U1", ["+3V3"])
    u2 = _ic("U2", ["+3V3"])
    caps = [_cap(f"C{i}", ["+3V3"]) for i in range(6)]
    pins = [("U1", "1"), ("U2", "1")] + [(c.ref, "1") for c in caps]
    vcc = Net("+3V3", pins)
    model = _model_with([u1, u2] + caps, [vcc])

    decap_map, rail_adj, _, _ = classify_caps(model)
    cap_to_ic = rail_adjacent_to_ic(model)

    # Same cap set, and every freed cap maps to an IC from decap_map.
    assert set(cap_to_ic.keys()) == set(rail_adj)
    assert set(cap_to_ic.values()) <= set(decap_map.keys())

    # The freed caps per IC are exactly that IC's assignment tail: the
    # round-robin gave each IC 3 caps, of which the first 2 are rigid —
    # so each IC's freed set has exactly 1 cap, and freed + rigid per IC
    # is the full assignment (3 per IC).
    from collections import Counter
    freed_per_ic = Counter(cap_to_ic.values())
    for ic_ref, rigid in decap_map.items():
        assert freed_per_ic[ic_ref] == 3 - len(rigid), (
            f"{ic_ref}: freed={freed_per_ic[ic_ref]}, rigid={len(rigid)}"
        )


def test_seed_rail_adjacent_caps_places_caps_near_assigned_ic():
    """Phase A½ seeder puts freed caps in a ring around their assigned IC."""
    from models.macro import Macro
    from place.initial import seed_rail_adjacent_caps

    # Two independent power domains on each IC, like a real multi-rail
    # BGA: "+3V3R" is a single tight pin (stays rigid), "+3V3F" is two
    # pads scattered across the footprint (goes rail-adjacent). Mirrors
    # the real test4/U30 finding (VCCPLL single-pad vs +1V2 4-corner).
    u1 = _ic("U1", ["+3V3R", "+3V3F"], pads=[
        ("R", 0.0, 0.0, "+3V3R"),
        ("F1", -5.0, 0.0, "+3V3F"), ("F2", 5.0, 0.0, "+3V3F"),
    ])
    u2 = _ic("U2", ["+3V3R", "+3V3F"], pads=[
        ("R", 0.0, 0.0, "+3V3R"),
        ("F1", -5.0, 0.0, "+3V3F"), ("F2", 5.0, 0.0, "+3V3F"),
    ])
    rigid_caps = [_cap(f"C{i}", ["+3V3R"]) for i in range(4)]
    freed_caps = [_cap(f"C{i}", ["+3V3F"]) for i in range(4, 10)]
    caps = rigid_caps + freed_caps
    vcc_r = Net("+3V3R", [("U1", "1"), ("U2", "1")] + [(c.ref, "1") for c in rigid_caps])
    vcc_f = Net("+3V3F", [("U1", "2"), ("U2", "2")] + [(c.ref, "1") for c in freed_caps])
    model = _model_with([u1, u2] + caps, [vcc_r, vcc_f])

    # Rigid macro for each IC with its 2 tight-net caps; the 6 scattered-
    # net caps are standalone macros elsewhere on the board (far away).
    u1.x, u1.y = 20.0, 20.0
    u2.x, u2.y = 80.0, 80.0
    m1 = Macro.with_caps(u1, [rigid_caps[0], rigid_caps[1]])
    m2 = Macro.with_caps(u2, [rigid_caps[2], rigid_caps[3]])
    standalone = [Macro.alone(c) for c in freed_caps]
    for i, m in enumerate(standalone):  # park them far from both ICs
        m.set_pose(50.0, 50.0 + i * 3.0, 0.0)
    macros = [m1, m2] + standalone
    bbox = (5.0, 5.0, 95.0, 95.0)

    n = seed_rail_adjacent_caps(model, macros, bbox)
    assert n == 6, f"Expected 6 freed caps re-seeded, got {n}"

    ref = {c.ref: c for c in model.components}
    from assign.assign_caps import rail_adjacent_to_ic
    for cap, ic in rail_adjacent_to_ic(model).items():
        d = ((ref[cap].x - ref[ic].x) ** 2 + (ref[cap].y - ref[ic].y) ** 2) ** 0.5
        # Ring radius is host half-extent (~6mm incl. rigid caps) + gap +
        # pitch/2 — well under 20mm. Before the fix they sat ≥30mm away.
        assert d < 20.0, f"{cap} is {d:.1f}mm from assigned IC {ic}"

    # Freed caps stay inside the interior bbox.
    for m in standalone:
        bx1, by1, bx2, by2 = m.bbox
        assert bx1 >= bbox[0] and by1 >= bbox[1] and bx2 <= bbox[2] and by2 <= bbox[3], (
            f"{m.leader.ref} bbox {[round(v,1) for v in m.bbox]} outside {bbox}"
        )


def test_sa_cap_attraction_holds_shared_rail_caps():
    """SA with cap_attraction_weight > 0 keeps freed caps near their
    assigned IC on a SHARED rail — the exact scenario where rail-bbox
    HPWL is flat (2 ICs at opposite corners span the whole board) and
    SA otherwise has no gradient signal at all.
    """
    from models.macro import Macro
    from place.sa import run_macro_sa
    from assign.assign_caps import rail_adjacent_to_ic

    # Same two-domain setup as the seeding test: one tight pin (rigid),
    # one scattered pin pair (rail-adjacent, held only by attraction).
    u1 = _ic("U1", ["+3V3R", "+3V3F"], pads=[
        ("R", 0.0, 0.0, "+3V3R"),
        ("F1", -5.0, 0.0, "+3V3F"), ("F2", 5.0, 0.0, "+3V3F"),
    ])
    u2 = _ic("U2", ["+3V3R", "+3V3F"], pads=[
        ("R", 0.0, 0.0, "+3V3R"),
        ("F1", -5.0, 0.0, "+3V3F"), ("F2", 5.0, 0.0, "+3V3F"),
    ])
    rigid_caps = [_cap(f"C{i}", ["+3V3R"]) for i in range(4)]
    freed_caps = [_cap(f"C{i}", ["+3V3F"]) for i in range(4, 10)]
    caps = rigid_caps + freed_caps
    vcc_r = Net("+3V3R", [("U1", "1"), ("U2", "1")] + [(c.ref, "1") for c in rigid_caps])
    vcc_f = Net("+3V3F", [("U1", "2"), ("U2", "2")] + [(c.ref, "1") for c in freed_caps])
    model = _model_with([u1, u2] + caps, [vcc_r, vcc_f])
    pairs = rail_adjacent_to_ic(model)
    assert len(pairs) == 6  # the 6 scattered-net caps; tight-net caps stay rigid

    # Shared-rail corner ICs: the +3V3F bbox spans the entire board, so
    # HPWL is flat w.r.t. any freed cap's position inside it.
    u1.x, u1.y = 20.0, 20.0
    u2.x, u2.y = 80.0, 80.0
    macros = [
        Macro.with_caps(u1, [rigid_caps[0], rigid_caps[1]]),
        Macro.with_caps(u2, [rigid_caps[2], rigid_caps[3]]),
    ]
    for i, c in enumerate(freed_caps):  # freed caps start far from both ICs
        c.x, c.y = 50.0, 50.0 + i * 3.0
        macros.append(Macro.alone(c))
    for m in macros:
        m.apply_offsets()

    run_macro_sa(
        model, macros, (5.0, 5.0, 95.0, 95.0),
        iterations=1500, reheats=1, seed=42,
        cap_attraction_weight=1.0, cap_pairs=pairs,
    )

    ref = {c.ref: c for c in model.components}
    for cap, ic in pairs.items():
        d = ((ref[cap].x - ref[ic].x) ** 2 + (ref[cap].y - ref[ic].y) ** 2) ** 0.5
        assert d < 20.0, f"{cap} drifted {d:.1f}mm from assigned IC {ic}"


def main():
    print("=" * 60)
    print("  Cap-IC assignment tests")
    print("=" * 60)
    run("power net detection", test_power_net_detection)
    run("power net detection: hidden rails", test_power_net_detection_hidden_rails)
    run("hidden rail caps end-to-end", test_hidden_rail_caps_get_decoupling_treatment_end_to_end)
    run("ground net detection", test_ground_net_detection)
    run("ground net detection: hidden rails", test_ground_net_detection_hidden_rails)
    run("single IC + single cap", test_single_ic_single_cap_assignment)
    run("cap assigned to exactly one IC", test_cap_assigned_to_exactly_one_ic)
    run("round-robin distribution", test_caps_distribute_round_robin_across_ics)
    run("unassigned cap not in map", test_unassigned_cap_not_in_map)
    run("multi-rail cap picks least-loaded IC", test_cap_on_multiple_rails_picks_least_loaded_ic)
    run("deterministic across runs", test_deterministic_across_runs)
    run("all real decaps rigid when tight", test_all_real_decaps_stay_rigid_when_pin_group_is_tight)
    run("scattered pin group goes rail-adjacent", test_scattered_pin_group_goes_rail_adjacent)
    run("tight pin group stays rigid with pad data", test_tight_pin_group_stays_rigid_even_with_pad_data)
    run("rail-adjacent→IC mapping", test_rail_adjacent_to_ic_maps_excess_to_assigned_ic)
    run("seed freed caps near assigned IC", test_seed_rail_adjacent_caps_places_caps_near_assigned_ic)
    run("SA cap attraction holds shared-rail caps", test_sa_cap_attraction_holds_shared_rail_caps)
    print("=" * 60)
    print(f"  {passed} passed, {failed} failed")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
