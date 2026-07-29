"""Correctness checks for the SA-polish legalization phase, mirroring
tests/test_abacus_bridge.py's structure.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.board_model import BoardModel, BoardOutline, Component, Net
from models.macro import Macro
from place.sa_polish import sa_polish_legalize


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


def test_skips_work_when_already_clean():
    model, macros = _make_scattered_board(n_ics=5, spread=90.0, seed=5)
    stats = sa_polish_legalize(model, macros, (0.0, 0.0, 100.0, 100.0), grid_mm=0.5)
    assert stats == {"residual_overlaps": 0, "boundary_failures": 0}


def test_reduces_overlaps_on_dense_board():
    model, macros = _make_scattered_board(n_ics=14, spread=35.0, seed=1)  # forces overlaps
    bounds = (0.0, 0.0, 100.0, 100.0)
    before = 0
    for i in range(len(macros)):
        for j in range(i + 1, len(macros)):
            if macros[i].overlaps(macros[j]):
                before += 1
    assert before > 0

    stats = sa_polish_legalize(model, macros, bounds, grid_mm=0.5, iterations=1500)
    assert stats["residual_overlaps"] <= before


def test_preserves_rigid_cap_offsets():
    from models.macro import _rotate

    model, macros = _make_scattered_board(n_ics=10, n_caps=3, spread=35.0, seed=2)
    bounds = (0.0, 0.0, 100.0, 100.0)
    sa_polish_legalize(model, macros, bounds, grid_mm=0.5, iterations=800)
    for m in macros:
        for cap, (dx, dy) in zip(m.followers, m.follower_offsets):
            rdx, rdy = _rotate(dx, dy, m.leader.rotation)
            expected_x = m.leader.x + rdx
            expected_y = m.leader.y + rdy
            assert abs(cap.x - expected_x) < 1e-6
            assert abs(cap.y - expected_y) < 1e-6


def test_respects_fixed_macros():
    model, macros = _make_scattered_board(n_ics=10, spread=35.0, seed=3)
    bounds = (0.0, 0.0, 100.0, 100.0)
    fixed = macros[0]
    fixed.is_fixed = True
    pos_before = (fixed.leader.x, fixed.leader.y)
    sa_polish_legalize(model, macros, bounds, grid_mm=0.5, iterations=800)
    assert (fixed.leader.x, fixed.leader.y) == pos_before


def test_never_produces_out_of_bounds_macros():
    model, macros = _make_scattered_board(n_ics=12, spread=35.0, seed=4)
    bounds = (0.0, 0.0, 100.0, 100.0)
    stats = sa_polish_legalize(model, macros, bounds, grid_mm=0.5, iterations=1500)
    assert stats["boundary_failures"] == 0
    for m in macros:
        x1, y1, x2, y2 = m.bbox
        assert x1 >= bounds[0] - 1e-6 and y1 >= bounds[1] - 1e-6
        assert x2 <= bounds[2] + 1e-6 and y2 <= bounds[3] + 1e-6
