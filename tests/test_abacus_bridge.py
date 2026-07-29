"""Validate the Abacus/macro-v2 bridge in isolation, before it's wired
into place/legalizer.py's legalize().
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.board_model import BoardModel, BoardOutline, Component, Net
from models.macro import Macro
from place.abacus_bridge import abacus_legalize_macros


def _make_scattered_board(n_ics=10, n_caps=2, seed=0, spread=90.0):
    rng = random.Random(seed)
    board = BoardOutline(0.0, 0.0, 100.0, 100.0)
    model = BoardModel(board=board)
    components = []
    macros = []
    for i in range(n_ics):
        cx, cy = rng.uniform(10, spread), rng.uniform(10, spread)
        ic = Component(
            ref=f"U{i}", footprint="IC", value="IC", x=cx, y=cy,
            rotation=0.0, width=6.0, height=6.0, component_type="ic",
        )
        caps = []
        for j in range(n_caps):
            cap = Component(
                ref=f"C{i}_{j}", footprint="Cap", value="100nF",
                x=cx, y=cy, rotation=0.0, width=1.0, height=0.5,
                component_type="capacitor",
            )
            caps.append(cap)
        m = Macro.with_caps(ic, caps)
        macros.append(m)
        components.append(ic)
        components.extend(caps)
    model.components = components
    nets = []
    for i in range(n_ics - 1):
        nets.append(Net(name=f"NET{i}", pins=[(f"U{i}", "1"), (f"U{i+1}", "2")]))
    gnd_pins = [(c.ref, "GND") for c in components]
    nets.append(Net(name="GND", pins=gnd_pins))
    model.nets = nets
    return model, macros


def test_bridge_does_not_crash_and_reduces_overlaps():
    model, macros = _make_scattered_board(n_ics=12, spread=40.0, seed=1)  # dense, forces overlaps
    bounds = (0.0, 0.0, 100.0, 100.0)

    before = 0
    for i in range(len(macros)):
        for j in range(i + 1, len(macros)):
            if macros[i].overlaps(macros[j]):
                before += 1
    assert before > 0, "test setup should start with overlaps"

    stats = abacus_legalize_macros(model, macros, bounds, grid_mm=0.5, verbose=False)
    assert stats["residual_overlaps"] <= before


def test_bridge_preserves_rigid_cap_offsets():
    """After the bridge moves a macro, each follower must still sit at
    exactly its original offset from the leader (rigid body invariant)
    — the bridge must never distort cap-IC geometry.
    """
    model, macros = _make_scattered_board(n_ics=6, n_caps=3, spread=70.0, seed=2)
    bounds = (0.0, 0.0, 100.0, 100.0)

    pre_offsets = []
    for m in macros:
        offs = []
        for cap, (dx, dy) in zip(m.followers, m.follower_offsets):
            offs.append((dx, dy))
        pre_offsets.append(offs)

    abacus_legalize_macros(model, macros, bounds, grid_mm=0.5, verbose=False)

    for m, offs in zip(macros, pre_offsets):
        for (dx, dy) in offs:
            pass  # offsets themselves are immutable by construction
        assert m.follower_offsets == offs, "bridge must not mutate follower_offsets"
        # And the actual cap positions must match leader pose + offset.
        for cap, (dx, dy) in zip(m.followers, m.follower_offsets):
            # leader.rotation is always 0 in this test, so no rotation needed
            expected_x = m.leader.x + dx
            expected_y = m.leader.y + dy
            assert abs(cap.x - expected_x) < 1e-6
            assert abs(cap.y - expected_y) < 1e-6


def test_bridge_respects_fixed_macros_as_obstacles():
    model, macros = _make_scattered_board(n_ics=8, spread=50.0, seed=3)
    bounds = (0.0, 0.0, 100.0, 100.0)
    # Mark one macro fixed (simulating an already-placed connector) and
    # park it in the middle of the pack.
    fixed = macros[0]
    fixed.is_fixed = True
    fixed_pos_before = (fixed.leader.x, fixed.leader.y)

    abacus_legalize_macros(model, macros, bounds, grid_mm=0.5, verbose=False)

    assert (fixed.leader.x, fixed.leader.y) == fixed_pos_before, \
        "fixed macros must never be moved by the bridge"


def test_bridge_handles_empty_macro_list():
    model, _ = _make_scattered_board(n_ics=1)
    stats = abacus_legalize_macros(model, [], (0.0, 0.0, 100.0, 100.0), grid_mm=0.5)
    assert stats == {"residual_overlaps": 0, "boundary_failures": 0}
