"""Correctness check: IncrementalCostTracker must always agree with a
from-scratch cost.evaluate() recompute, for every move type SA uses
(translate, rotate, swap, displace/translate-of-a-neighbor) and
regardless of whether the move is committed or discarded.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cost.cost import evaluate
from cost.incremental import IncrementalCostTracker
from models.board_model import BoardModel, BoardOutline, Component, Net
from models.macro import Macro


def _make_board(n_ics=6, n_caps=2, seed=0):
    rng = random.Random(seed)
    board = BoardOutline(0.0, 0.0, 80.0, 80.0)
    model = BoardModel(board=board)
    components = []
    for i in range(n_ics):
        ic = Component(
            ref=f"U{i}", footprint="IC", value="IC", x=rng.uniform(5, 75),
            y=rng.uniform(5, 75), rotation=0.0, width=5.0, height=5.0,
            component_type="ic",
        )
        components.append(ic)
        for j in range(n_caps):
            cap = Component(
                ref=f"C{i}_{j}", footprint="Cap", value="100nF",
                x=ic.x, y=ic.y, rotation=0.0, width=1.0, height=0.5,
                component_type="capacitor",
            )
            components.append(cap)
    model.components = components

    # A few signal nets between adjacent ICs + a shared "GND" net
    # touching everything (mirrors real boards).
    nets = []
    for i in range(n_ics - 1):
        nets.append(Net(name=f"NET{i}", pins=[(f"U{i}", "1"), (f"U{i+1}", "2")]))
    gnd_pins = [(c.ref, "GND") for c in components]
    nets.append(Net(name="GND", pins=gnd_pins))
    model.nets = nets
    return model


def _build_macros(model):
    macros = []
    ics = [c for c in model.components if c.component_type == "ic"]
    caps_by_ic = {}
    for c in model.components:
        if c.component_type == "capacitor":
            ic_ref = c.ref.split("_")[0]
            caps_by_ic.setdefault(ic_ref, []).append(c)
    for ic in ics:
        caps = caps_by_ic.get(ic.ref, [])
        m = Macro.with_caps(ic, caps)
        macros.append(m)
    return macros


def test_incremental_matches_full_recompute_translate():
    model = _make_board()
    macros = _build_macros(model)
    bounds = (0.0, 0.0, 80.0, 80.0)
    tracker = IncrementalCostTracker(model, macros, alpha=1.0, beta=25.0, gamma=8.0)

    baseline = evaluate(model, macros, alpha=1.0, beta=25.0, gamma=8.0)
    assert abs(tracker.total()["total"] - baseline["total"]) < 1e-6

    rng = random.Random(1)
    for _ in range(200):
        idx = rng.randrange(len(macros))
        m = macros[idx]
        snap = m._snapshot()
        dx, dy = rng.uniform(-5, 5), rng.uniform(-5, 5)
        ok = m.translate(dx, dy, bounds=bounds)
        if not ok:
            continue

        proposed = tracker.propose([idx])
        full = evaluate(model, macros, alpha=1.0, beta=25.0, gamma=8.0)
        assert abs(proposed["hpwl"] - full["hpwl"]) < 1e-6
        assert abs(proposed["overlap"] - full["overlap"]) < 1e-6
        assert abs(proposed["boundary"] - full["boundary"]) < 1e-6
        assert abs(proposed["total"] - full["total"]) < 1e-6

        if rng.random() < 0.5:
            tracker.commit()
        else:
            m._restore(snap)
            tracker.discard()
            still_full = evaluate(model, macros, alpha=1.0, beta=25.0, gamma=8.0)
            assert abs(tracker.total()["total"] - still_full["total"]) < 1e-6


def test_incremental_matches_full_recompute_rotate_and_swap():
    model = _make_board(n_ics=8)
    macros = _build_macros(model)
    bounds = (0.0, 0.0, 80.0, 80.0)
    tracker = IncrementalCostTracker(model, macros, alpha=1.0, beta=25.0, gamma=8.0)

    rng = random.Random(2)
    for _ in range(150):
        move = rng.choice(["rotate", "swap", "translate"])
        if move == "rotate":
            idx = rng.randrange(len(macros))
            m = macros[idx]
            snap = m._snapshot()
            new_rot = (m.leader.rotation + rng.choice([90, 180, 270])) % 360
            ok = m.set_pose(m.leader.x, m.leader.y, new_rot, bounds=bounds)
            touched = [idx]
            restore = lambda: m._restore(snap)
        elif move == "swap":
            i, j = rng.sample(range(len(macros)), 2)
            a, b = macros[i], macros[j]
            snap_a, snap_b = a._snapshot(), b._snapshot()
            ax, ay = a.leader.x, a.leader.y
            bx, by = b.leader.x, b.leader.y
            ok1 = a.set_pose(bx, by, a.leader.rotation, bounds=bounds)
            ok2 = b.set_pose(ax, ay, b.leader.rotation, bounds=bounds)
            ok = ok1 and ok2
            touched = [i, j]
            restore = lambda: (a._restore(snap_a), b._restore(snap_b))
        else:
            idx = rng.randrange(len(macros))
            m = macros[idx]
            snap = m._snapshot()
            ok = m.translate(rng.uniform(-4, 4), rng.uniform(-4, 4), bounds=bounds)
            touched = [idx]
            restore = lambda: m._restore(snap)

        if not ok:
            continue

        proposed = tracker.propose(touched)
        full = evaluate(model, macros, alpha=1.0, beta=25.0, gamma=8.0)
        assert abs(proposed["total"] - full["total"]) < 1e-6, move

        if rng.random() < 0.5:
            tracker.commit()
        else:
            restore()
            tracker.discard()


def test_bias_overlapping_does_not_crash_and_reduces_overlap():
    """bias_overlapping is opt-in and touches the shared run_macro_sa
    move-selection code path — verify it runs cleanly and actually
    biases toward overlapping macros (sanity, not a strict guarantee).
    """
    from place.sa import run_macro_sa

    model = _make_board(n_ics=10, seed=7)
    macros = _build_macros(model)
    bounds = (0.0, 0.0, 80.0, 80.0)

    # Force some overlaps by clustering all leaders together first.
    rng = random.Random(9)
    for m in macros:
        m.set_pose(40.0 + rng.uniform(-3, 3), 40.0 + rng.uniform(-3, 3),
                   0.0, bounds=bounds)

    before = sum(
        1 for i in range(len(macros)) for j in range(i + 1, len(macros))
        if macros[i].overlaps(macros[j])
    )
    assert before > 0

    run_macro_sa(
        model, macros, bounds, iterations=500, reheats=0,
        alpha=0.2, beta=400.0, gamma=40.0,
        initial_window_mm=3.0, final_window_mm=0.2,
        bias_overlapping=True, bias_overlap_prob=0.7,
        seed=1, verbose=False,
    )

    after = sum(
        1 for i in range(len(macros)) for j in range(i + 1, len(macros))
        if macros[i].overlaps(macros[j])
    )
    assert after <= before
