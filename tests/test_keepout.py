"""Tests for internal-keepout (cutout) support — IMPROVEMENTS §2.1.

Three layers, mirroring the §2.1 patch:

  1. ``cost.total_keepout_overlap`` — the from-scratch penalty SA sees.
  2. ``cost.evaluate`` — keepout threaded into the total (defaults to γ).
  3. ``IncrementalCostTracker`` — the per-move cache SA actually uses in
     its inner loop must agree with the from-scratch evaluator, including
     the keepout term, after every propose/commit/discard.
  4. ``place.legalizer.legalize`` — the ``_keepout_clamp`` must evict a
     macro that starts dead-centre on an internal cutout.
  5. End-to-end parse → legalize on ``cutout_test.kicad_pcb``: the parser
     must turn the internal Edge.Cuts slot into a keepout and the
     legalizer must clear it.
"""
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cost.cost import evaluate, total_keepout_overlap
from cost.incremental import IncrementalCostTracker
from models.board_model import BoardModel, BoardOutline, Component, Net
from models.macro import Macro
from parsers.kicad_parser import KiCadParser

ROOT = Path(__file__).resolve().parents[1]
CUTOUT_PCB = ROOT / "tests" / "test_pcbs" / "cutout_test.kicad_pcb"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# 80×60 board with a 10×10 internal milled slot dead centre — the cutout
# board ships as cutout_test.kicad_pcb; we rebuild the same geometry inline
# for the unit tests so they don't depend on file parsing.
SLOT = BoardOutline(35.0, 25.0, 45.0, 35.0)


def _ic(ref, x, y, w=5.0, h=4.5):
    return Component(ref=ref, footprint="IC", value="MCU", x=x, y=y,
                     width=w, height=h, component_type="ic")


def _res(ref, x, y):
    return Component(ref=ref, footprint="R0603", value="10k", x=x, y=y,
                     width=1.6, height=0.8, component_type="resistor")


def _model_with_slot(components, nets=None):
    board = BoardOutline(0.0, 0.0, 80.0, 60.0)
    m = BoardModel(board=board, components=components, nets=nets or [])
    m.keepouts = [SLOT]
    return m


# ---------------------------------------------------------------------------
# 1. total_keepout_overlap
# ---------------------------------------------------------------------------

def test_keepout_overlap_positive_when_component_on_slot():
    model = _model_with_slot([_ic("U1", 40, 30)])  # dead centre on the slot
    assert total_keepout_overlap(model) > 0.0


def test_keepout_overlap_zero_when_component_clear():
    model = _model_with_slot([_ic("U1", 10, 10)])  # well clear of the slot
    assert total_keepout_overlap(model) == 0.0


def test_keepout_overlap_zero_when_no_keepouts():
    # The common case — 5 of 6 bundled boards have no Edge.Cuts cutouts.
    board = BoardOutline(0.0, 0.0, 80.0, 60.0)
    model = BoardModel(board=board, components=[_ic("U1", 40, 30)])
    assert total_keepout_overlap(model) == 0.0


def test_keepout_overlap_area_is_intersection_not_union():
    # A 5×4.5 IC centred on the 10×10 slot is entirely INSIDE the slot,
    # so the intersection equals the IC's own (courtyard-inflated) bbox
    # area — proving the penalty is the intersection, not the 100 mm²
    # union and not zero.
    c = _ic("U1", 40, 30, w=5.0, h=4.5)
    model = _model_with_slot([c])
    bx1, by1, bx2, by2 = c.bbox
    expected = (bx2 - bx1) * (by2 - by1)
    overlap = total_keepout_overlap(model)
    assert abs(overlap - expected) < 1e-9, f"{overlap} != bbox area {expected}"
    assert overlap < 100.0, "penalty is the union area, not the intersection"


def test_edge_connector_is_exempt():
    """A connector intentionally overhanging a mounting-hole zone near the
    edge must not be charged a keepout penalty — same exemption the outer
    boundary penalty applies."""
    conn = Component(ref="J1", footprint="HDR", value="CONN", x=40, y=30,
                     width=5.0, height=4.5, component_type="connector")
    conn.is_edge_connector = True
    model = _model_with_slot([conn])
    assert total_keepout_overlap(model) == 0.0


# ---------------------------------------------------------------------------
# 2. evaluate() includes the keepout term (default weight = γ)
# ---------------------------------------------------------------------------

def test_evaluate_keepout_defaults_to_gamma():
    model = _model_with_slot([_ic("U1", 40, 30)])
    macros = [Macro.alone(model.components[0])]
    res = evaluate(model, macros, alpha=1.0, beta=25.0, gamma=8.0)
    # keepout term present and equal to the raw overlap area.
    assert res["keepout"] == total_keepout_overlap(model)
    # Default keepout_weight == γ: zeroing it must drop exactly γ·keepout
    # from the total (and nothing else — every other term is unchanged).
    res2 = evaluate(model, macros, alpha=1.0, beta=25.0, gamma=8.0, keepout_weight=0.0)
    assert abs((res["total"] - res2["total"]) - 8.0 * res["keepout"]) < 1e-9
    # A custom keepout_weight is honored too.
    res3 = evaluate(model, macros, alpha=1.0, beta=25.0, gamma=8.0, keepout_weight=3.0)
    assert abs((res3["total"] - res2["total"]) - 3.0 * res["keepout"]) < 1e-9


# ---------------------------------------------------------------------------
# 3. IncrementalCostTracker agrees with evaluate() including keepout
# ---------------------------------------------------------------------------

def test_tracker_total_matches_evaluate_with_keepout():
    model = _model_with_slot([_ic("U1", 40, 30)])
    macros = [Macro.alone(model.components[0])]
    tracker = IncrementalCostTracker(model, macros, alpha=1.0, beta=25.0, gamma=8.0)
    full = evaluate(model, macros, alpha=1.0, beta=25.0, gamma=8.0)
    assert abs(tracker.total()["keepout"] - full["keepout"]) < 1e-9
    assert abs(tracker.total()["total"] - full["total"]) < 1e-9


def test_tracker_propose_matches_evaluate_after_move_off_keepout():
    """Move a macro from on-slot to off-slot: the incremental propose()
    must agree with a from-scratch evaluate() on both the keepout term
    and the total, whether the move is committed or discarded."""
    model = _model_with_slot([_ic("U1", 40, 30)])
    macros = [Macro.alone(model.components[0])]
    bounds = (0.0, 0.0, 80.0, 60.0)
    tracker = IncrementalCostTracker(model, macros, alpha=1.0, beta=25.0, gamma=8.0)

    # ── Commit path: push the macro clear of the slot (to 10,10). ──
    snap = macros[0]._snapshot()          # pose at (40,30), on the slot
    macros[0].set_pose(10.0, 10.0, 0.0, bounds=bounds)

    proposed = tracker.propose([0])
    full = evaluate(model, macros, alpha=1.0, beta=25.0, gamma=8.0)
    assert abs(proposed["keepout"] - full["keepout"]) < 1e-9
    assert abs(proposed["total"] - full["total"]) < 1e-9
    # Moving off the slot must have reduced the keepout penalty.
    assert proposed["keepout"] < tracker.total()["keepout"]

    tracker.commit()
    assert abs(tracker.total()["total"] - full["total"]) < 1e-9

    # ── Discard path: snapshot the committed pose, move back onto the
    #    slot, propose, then revert + discard — tracker must return to the
    #    committed (off-slot) state and agree with evaluate() there. ──
    snap2 = macros[0]._snapshot()         # pose at (10,10), off the slot
    macros[0].set_pose(40.0, 30.0, 0.0, bounds=bounds)   # back onto slot
    proposed2 = tracker.propose([0])
    on_slot = evaluate(model, macros, alpha=1.0, beta=25.0, gamma=8.0)
    assert abs(proposed2["total"] - on_slot["total"]) < 1e-9
    macros[0]._restore(snap2)             # revert to (10,10)
    tracker.discard()
    off_slot = evaluate(model, macros, alpha=1.0, beta=25.0, gamma=8.0)
    assert abs(tracker.total()["total"] - off_slot["total"]) < 1e-9
    assert tracker.total()["keepout"] == 0.0


def test_tracker_keepout_agrees_across_random_sweep():
    """Random translates on a board with a slot: the incremental tracker
    must match the from-scratch evaluator after every propose and after
    every commit/discard — the same invariant test_incremental_cost
     checks for hpwl/overlap/boundary, now for keepout."""
    rng = random.Random(7)
    ics = [_ic("U0", 40, 30)] + [_ic(f"U{i}", rng.uniform(5, 75), rng.uniform(5, 55))
                                  for i in range(1, 5)]
    model = _model_with_slot(ics)
    macros = [Macro.alone(c) for c in model.components]
    bounds = (0.0, 0.0, 80.0, 60.0)
    tracker = IncrementalCostTracker(model, macros, alpha=1.0, beta=25.0, gamma=8.0)

    for _ in range(150):
        idx = rng.randrange(len(macros))
        snap = macros[idx]._snapshot()
        ok = macros[idx].translate(rng.uniform(-8, 8), rng.uniform(-8, 8), bounds=bounds)
        if not ok:
            continue
        proposed = tracker.propose([idx])
        full = evaluate(model, macros, alpha=1.0, beta=25.0, gamma=8.0)
        assert abs(proposed["keepout"] - full["keepout"]) < 1e-9, "keepout diverged"
        assert abs(proposed["total"] - full["total"]) < 1e-9, "total diverged"
        if rng.random() < 0.5:
            tracker.commit()
        else:
            macros[idx]._restore(snap)
            tracker.discard()
        assert abs(tracker.total()["total"]
                   - evaluate(model, macros, alpha=1.0, beta=25.0, gamma=8.0)["total"]) < 1e-9


# ---------------------------------------------------------------------------
# 4. Legalizer evicts a macro from an internal cutout
# ---------------------------------------------------------------------------

def test_legalize_evicts_macro_from_keepout():
    """U1 starts dead-centre on the slot. After legalize(), the keepout
    clamp must have pushed it clear: keepout_failures == 0 and the model
    has zero keepout overlap."""
    from place.legalizer import legalize

    u1 = _ic("U1", 40, 30)
    r1, r2 = _res("R1", 10, 10), _res("R2", 70, 50)
    model = _model_with_slot([u1, r1, r2])
    macros = [Macro.alone(c) for c in model.components]
    bounds = (0.0, 0.0, 80.0, 60.0)

    assert total_keepout_overlap(model) > 0.0, "precondition: U1 starts on the slot"

    result = legalize(model, macros, bounds, grid_mm=0.5, seed=42)

    assert result["keepout_failures"] == 0, f"residual keepout failures: {result}"
    assert total_keepout_overlap(model) == 0.0, "U1 not evicted from the slot"
    # And U1 must still be on the board (eviction must not punt it OOB).
    board = model.board
    assert board.contains_bbox(u1.bbox), f"U1 evicted off-board: {u1.bbox}"


def test_legalize_no_keepout_failures_when_nothing_on_slot():
    from place.legalizer import legalize

    model = _model_with_slot([_ic("U1", 10, 10), _res("R1", 70, 50)])
    macros = [Macro.alone(c) for c in model.components]
    bounds = (0.0, 0.0, 80.0, 60.0)
    result = legalize(model, macros, bounds, grid_mm=0.5, seed=42)
    assert result["keepout_failures"] == 0


# ---------------------------------------------------------------------------
# 5. End-to-end: parse cutout_test.kicad_pcb → keepout extracted → legalized
# ---------------------------------------------------------------------------

def test_parser_extracts_internal_slot_as_keepout():
    """The internal 10×10 Edge.Cuts rect on cutout_test.kicad_pcb must be
    parsed as a keepout (area < 95% of the 80×60 outer outline), not
    folded into the board outline."""
    model = KiCadParser(str(CUTOUT_PCB), bbox_margin=0.8).parse()
    assert len(model.keepouts) == 1, f"expected 1 keepout, got {len(model.keepouts)}"
    k = model.keepouts[0]
    assert (k.x_min, k.y_min, k.x_max, k.y_max) == (35.0, 25.0, 45.0, 35.0)


def test_cutout_board_legalizes_clean():
    """Parse the cutout board (U1 starts on the slot), build macros, run
    the legalizer: every component must end up off the slot and on the
    board — the §2.1 keepout path exercised on a real parsed board."""
    from place.legalizer import legalize

    model = KiCadParser(str(CUTOUT_PCB), bbox_margin=0.8).parse()
    assert len(model.keepouts) == 1

    u1 = model.get_component("U1")
    assert total_keepout_overlap(model) > 0.0, "precondition: U1 starts on the slot"

    macros = [Macro.alone(c) for c in model.components]
    b = model.board
    bounds = (b.x_min, b.y_min, b.x_max, b.y_max)
    result = legalize(model, macros, bounds, grid_mm=0.5, seed=42)

    assert result["keepout_failures"] == 0, f"residual keepout failures: {result}"
    assert total_keepout_overlap(model) == 0.0
    for c in model.components:
        assert model.board.contains_bbox(c.bbox), f"{c.ref} off-board: {c.bbox}"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
