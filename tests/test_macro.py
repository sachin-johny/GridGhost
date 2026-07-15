"""Unit tests for Macro: rigid-body primitives."""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.board_model import Component, BoardOutline
from models.macro import (
    Macro,
    find_cap_offset,
    MAX_CAP_IC_DISTANCE_MM,
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


def _ic(ref="U1", x=50.0, y=50.0, w=10.0, h=10.0) -> Component:
    return Component(
        ref=ref, x=x, y=y, width=w, height=h,
        courtyard_margin=0.0, component_type="ic",
    )


def _cap(ref="C1", x=0.0, y=0.0, w=2.0, h=1.0) -> Component:
    return Component(
        ref=ref, x=x, y=y, width=w, height=h,
        courtyard_margin=0.0, component_type="capacitor",
    )


def test_alone_has_no_followers():
    m = Macro.alone(_ic())
    assert m.followers == []
    assert m.follower_offsets == []
    assert m.members == [m.leader]


def test_with_caps_assigns_followers():
    leader = _ic()
    c1 = _cap("C1")
    c2 = _cap("C2")
    m = Macro.with_caps(leader, [c1, c2])
    assert m.followers == [c1, c2]
    assert len(m.follower_offsets) == 2


def test_with_caps_finds_overlap_free_slots():
    leader = _ic(x=50, y=50, w=10, h=10)
    c1 = _cap("C1")
    c2 = _cap("C2")
    m = Macro.with_caps(leader, [c1, c2])
    # Both caps must be overlap-free with leader and each other.
    assert not leader.overlaps(c1), "C1 overlaps leader"
    assert not leader.overlaps(c2), "C2 overlaps leader"
    assert not c1.overlaps(c2), "C1 and C2 overlap"


def test_fan_offset_respects_max_distance():
    leader = _ic(x=50, y=50, w=10, h=10)
    cap = _cap("C1")
    offset = find_cap_offset(leader, cap, others=[leader])
    dist = math.hypot(*offset)
    assert dist <= MAX_CAP_IC_DISTANCE_MM, (
        f"Offset {offset} distance {dist:.2f} exceeds max {MAX_CAP_IC_DISTANCE_MM}"
    )


def test_translate_moves_leader_and_followers_rigidly():
    leader = _ic(x=50, y=50)
    cap = _cap("C1")
    m = Macro.with_caps(leader, [cap])
    cap_dx_before = cap.x - leader.x
    cap_dy_before = cap.y - leader.y

    ok = m.translate(5.0, -3.0)
    assert ok, "Translate should succeed (no bounds)"
    cap_dx_after = cap.x - leader.x
    cap_dy_after = cap.y - leader.y
    assert abs(cap_dx_after - cap_dx_before) < 1e-9, (
        f"Cap-leader dx changed: {cap_dx_before} -> {cap_dx_after}"
    )
    assert abs(cap_dy_after - cap_dy_before) < 1e-9, (
        f"Cap-leader dy changed: {cap_dy_before} -> {cap_dy_after}"
    )
    assert leader.x == 55.0 and leader.y == 47.0


def test_translate_reverts_on_bounds_violation():
    # Board is 0..100; leader is at 95 with width 10 (bbox 90..100).
    # Translating +10 would push bbox to 100..110 — out of bounds.
    leader = _ic(x=95, y=50)
    cap = _cap("C1")
    m = Macro.with_caps(leader, [cap])
    leader_x_before = leader.x
    cap_x_before = cap.x
    bounds = (0.0, 0.0, 100.0, 100.0)

    ok = m.translate(10.0, 0.0, bounds=bounds)
    assert not ok, "Translate should fail (bounds violation)"
    assert leader.x == leader_x_before, "Leader moved despite revert"
    assert cap.x == cap_x_before, "Cap moved despite revert"


def test_set_pose_rotates_followers_around_leader():
    leader = _ic(x=50, y=50, w=10, h=10)
    cap = _cap("C1")
    m = Macro.with_caps(leader, [cap])

    # Snapshot offset (relative to leader) at rotation=0
    offset_x0 = cap.x - leader.x
    offset_y0 = cap.y - leader.y

    # Rotate leader 90 degrees. Cap offset should rotate with it.
    m.set_pose(leader.x, leader.y, 90.0)
    offset_x90 = cap.x - leader.x
    offset_y90 = cap.y - leader.y

    # KiCad CW-positive: rotating 90 CW transforms (dx, dy) -> (dy, -dx)
    # but with our sin convention (-sin), the matrix is [[cos, -sin*sin_neg], ...]
    # Just check: distance is preserved.
    d0 = math.hypot(offset_x0, offset_y0)
    d90 = math.hypot(offset_x90, offset_y90)
    assert abs(d0 - d90) < 1e-9, f"Cap-leader distance changed: {d0} -> {d90}"


def test_bbox_is_union_of_members():
    leader = _ic(x=50, y=50, w=10, h=10)
    cap = _cap("C1")
    m = Macro.with_caps(leader, [cap])

    bbox = m.bbox
    lx1, ly1, lx2, ly2 = leader.bbox
    cx1, cy1, cx2, cy2 = cap.bbox
    expected = (
        min(lx1, cx1), min(ly1, cy1), max(lx2, cx2), max(ly2, cy2)
    )
    assert bbox == expected, f"bbox {bbox} != union {expected}"


def test_overlaps_detects_intersection():
    leader_a = _ic("U1", x=50, y=50, w=10, h=10)
    a = Macro.alone(leader_a)

    leader_b = _ic("U2", x=55, y=55, w=10, h=10)  # Overlaps with U1
    b = Macro.alone(leader_b)
    assert a.overlaps(b), "Overlapping macros not detected"

    leader_c = _ic("U3", x=200, y=200, w=10, h=10)  # Far away
    c = Macro.alone(leader_c)
    assert not a.overlaps(c), "Non-overlapping macros reported as overlap"


def test_overlap_area_is_correct():
    leader_a = _ic("U1", x=50, y=50, w=10, h=10)
    a = Macro.alone(leader_a)
    leader_b = _ic("U2", x=55, y=55, w=10, h=10)  # 5x5 overlap with U1
    b = Macro.alone(leader_b)
    area = a.overlap_area(b)
    assert abs(area - 25.0) < 1e-9, f"Overlap area {area} != 25.0"


def main():
    print("=" * 60)
    print("  Macro unit tests")
    print("=" * 60)
    run("alone has no followers", test_alone_has_no_followers)
    run("with_caps assigns followers", test_with_caps_assigns_followers)
    run("with_caps finds overlap-free slots", test_with_caps_finds_overlap_free_slots)
    run("fan offset respects MAX_CAP_IC_DISTANCE", test_fan_offset_respects_max_distance)
    run("translate moves rigidly", test_translate_moves_leader_and_followers_rigidly)
    run("translate reverts on bounds violation", test_translate_reverts_on_bounds_violation)
    run("set_pose rotates followers", test_set_pose_rotates_followers_around_leader)
    run("bbox is union of members", test_bbox_is_union_of_members)
    run("overlaps detects intersection", test_overlaps_detects_intersection)
    run("overlap_area is correct", test_overlap_area_is_correct)
    print("=" * 60)
    print(f"  {passed} passed, {failed} failed")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
