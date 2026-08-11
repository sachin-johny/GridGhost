"""Regression tests for non-rectangular board outlines (notches, mouse-bites,
mounting cutouts, holes).

Covers three layers:
  - BoardOutline geometry primitives (contains / contains_bbox / clamp /
    fit_bbox_inside / bbox_overflow) for both plain rectangles (backward
    compatibility) and polygon outlines with concavities and holes.
  - KiCad Edge.Cuts parsing: tracing a real polygon outline out of
    gr_line/gr_rect/gr_arc/gr_poly primitives, with a safe fallback to the
    old AABB-rectangle behavior.
  - The legalizer end-to-end: components must never be legalized into a
    notch, mouse-bite, or hole even under adversarial starting conditions.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.board_model import BoardModel, BoardOutline, Component, Net
from legalization.legalizer import legalize
from parsers.kicad_parser import _extract_board_outline, _trace_outer_board_polygon


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# 100x80 board with a 40x40 notch cut out of the top-right corner (e.g. a
# USB/board-edge connector notch), forming an L-shape.
NOTCH_POLYGON = [(0, 0), (100, 0), (100, 40), (60, 40), (60, 80), (0, 80)]


def _comp(ref, x, y, w=4.0, h=2.0):
    return Component(ref=ref, footprint="cap", value="100nF", x=x, y=y,
                      width=w, height=h, component_type="capacitor")


# ---------------------------------------------------------------------------
# BoardOutline geometry — rectangle backward compatibility
# ---------------------------------------------------------------------------

def test_rectangle_contains_unchanged():
    r = BoardOutline(x_min=0, y_min=0, x_max=100, y_max=80)
    assert not r.is_polygon
    assert r.contains(50, 40)
    assert not r.contains(150, 40)


def test_rectangle_contains_bbox_unchanged():
    r = BoardOutline(x_min=0, y_min=0, x_max=100, y_max=80)
    assert r.contains_bbox((10, 10, 90, 70))
    assert not r.contains_bbox((10, 10, 110, 70))


def test_rectangle_clamp_unchanged():
    r = BoardOutline(x_min=0, y_min=0, x_max=100, y_max=80)
    assert r.clamp(150, 40) == (100, 40)


def test_rectangle_bbox_overflow_matches_original_formula():
    r = BoardOutline(x_min=0, y_min=0, x_max=100, y_max=80)
    assert r.bbox_overflow((10, 10, 90, 70)) == 0.0
    # 5mm past x_min: originally left_overflow=5, everything else 0.
    assert r.bbox_overflow((-5, 10, 90, 70)) == 5.0


def test_rectangle_fit_bbox_inside_is_exact_clamp():
    r = BoardOutline(x_min=0, y_min=0, x_max=100, y_max=80)
    assert r.fit_bbox_inside(150, 40, 5, 5) == (95, 40)


# ---------------------------------------------------------------------------
# BoardOutline geometry — polygon outlines (notches)
# ---------------------------------------------------------------------------

def test_polygon_bbox_is_derived():
    p = BoardOutline(polygon=NOTCH_POLYGON)
    assert p.is_polygon
    assert (p.x_min, p.y_min, p.x_max, p.y_max) == (0, 0, 100, 80)


def test_polygon_contains_excludes_notch():
    p = BoardOutline(polygon=NOTCH_POLYGON)
    assert p.contains(30, 30)          # in the solid part of the board
    assert p.contains(80, 20)          # in the solid part, right of the notch
    assert not p.contains(80, 60)      # inside the removed notch region


def test_polygon_contains_bbox_rejects_notch_straddle():
    p = BoardOutline(polygon=NOTCH_POLYGON)
    assert p.contains_bbox((10, 10, 50, 70))
    assert not p.contains_bbox((70, 50, 90, 70))   # fully inside the notch
    assert not p.contains_bbox((50, 10, 90, 70))   # straddles the notch edge


def test_polygon_contains_bbox_rejects_nested_hole():
    """A hole entirely swallowed by a bbox — none of the bbox corners land
    inside the hole, and no edges cross — is a distinct failure mode from
    a bbox straddling the outer boundary; regression-tests the extra
    nested-hole vertex check."""
    outer = [(0, 0), (100, 0), (100, 80), (0, 80)]
    hole = [(40, 30), (60, 30), (60, 50), (40, 50)]
    h = BoardOutline(polygon=outer, holes=[hole])
    assert h.contains_bbox((0, 0, 30, 80))
    assert not h.contains_bbox((35, 25, 65, 55))


def test_polygon_fit_bbox_inside_escapes_notch():
    p = BoardOutline(polygon=NOTCH_POLYGON)
    for (tx, ty) in [(80, 60), (65, 45), (95, 75), (61, 41), (99, 79)]:
        cx, cy = p.fit_bbox_inside(tx, ty, 5, 5)
        assert p.contains_bbox((cx - 5, cy - 5, cx + 5, cy + 5)), (tx, ty, cx, cy)


def test_polygon_fit_bbox_inside_escapes_hole():
    outer = [(0, 0), (100, 0), (100, 80), (0, 80)]
    hole = [(40, 30), (60, 30), (60, 50), (40, 50)]
    h = BoardOutline(polygon=outer, holes=[hole])
    cx, cy = h.fit_bbox_inside(50, 40, 3, 3)
    assert h.contains_bbox((cx - 3, cy - 3, cx + 3, cy + 3))


def test_polygon_bbox_overflow_zero_when_contained():
    p = BoardOutline(polygon=NOTCH_POLYGON)
    assert p.bbox_overflow((10, 10, 50, 70)) == 0.0
    assert p.bbox_overflow((70, 50, 90, 70)) > 0.0


def test_polygon_touching_edge_is_not_a_false_intersection():
    """A component sitting flush against a notch wall is a legal, common
    placement and must not be rejected by the segment-crossing test."""
    p = BoardOutline(polygon=NOTCH_POLYGON)
    # bbox exactly touching the notch's horizontal edge (y=40) from below.
    assert p.contains_bbox((70, 30, 90, 40))


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------

def test_board_model_json_roundtrip_preserves_polygon():
    import json
    board = BoardOutline(polygon=NOTCH_POLYGON, holes=[[(10, 10), (20, 10), (20, 20), (10, 20)]])
    model = BoardModel(board=board, components=[_comp("C1", 30, 30)], nets=[])
    restored = BoardModel.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored.board.is_polygon
    assert restored.board.polygon == NOTCH_POLYGON
    assert not restored.board.contains(80, 60)


# ---------------------------------------------------------------------------
# KiCad Edge.Cuts parsing
# ---------------------------------------------------------------------------

def _gr_line(x1, y1, x2, y2):
    return ["gr_line", ["start", str(x1), str(y1)], ["end", str(x2), str(y2)], ["layer", "Edge.Cuts"]]


def test_parser_traces_notch_polygon():
    edges = [(0, 0, 100, 0), (100, 0, 100, 40), (100, 40, 60, 40),
             (60, 40, 60, 80), (60, 80, 0, 80), (0, 80, 0, 0)]
    sexp = ["kicad_pcb"] + [_gr_line(*e) for e in edges]
    outline = _extract_board_outline(sexp)
    assert outline.is_polygon
    assert not outline.contains(80, 60)
    assert outline.contains(30, 30)


def test_parser_falls_back_to_rectangle_for_plain_rect():
    edges = [(0, 0, 100, 0), (100, 0, 100, 80), (100, 80, 0, 80), (0, 80, 0, 0)]
    sexp = ["kicad_pcb"] + [_gr_line(*e) for e in edges]
    outline = _extract_board_outline(sexp)
    # Plain rectangle stays on the original code path (with its margin),
    # unchanged from pre-polygon-support behavior.
    assert not outline.is_polygon
    assert outline.x_min == -2.0 and outline.x_max == 102.0


def test_parser_falls_back_when_outline_does_not_close():
    # An open chain (missing the last edge) can't be traced into a loop.
    edges = [(0, 0, 100, 0), (100, 0, 100, 80), (100, 80, 0, 80)]
    sexp = ["kicad_pcb"] + [_gr_line(*e) for e in edges]
    outline = _extract_board_outline(sexp)
    assert not outline.is_polygon  # safe AABB fallback, no crash


def test_parser_gr_arc_included_in_polygon_trace():
    sexp = ["kicad_pcb",
            _gr_line(5, 0, 100, 0),
            _gr_line(100, 0, 100, 80),
            _gr_line(100, 80, 5, 80),
            _gr_line(5, 80, 0, 75),
            _gr_line(0, 75, 0, 5),
            ["gr_arc", ["start", "0", "5"], ["mid", "1.46", "1.46"], ["end", "5", "0"],
             ["layer", "Edge.Cuts"]]]
    poly = _trace_outer_board_polygon(sexp)
    assert poly is not None
    assert len(poly) > 4  # the arc contributed intermediate vertices


# ---------------------------------------------------------------------------
# End-to-end legalizer: components must never end up in a notch/hole
# ---------------------------------------------------------------------------

def test_legalize_pulls_components_out_of_notch():
    board = BoardOutline(polygon=NOTCH_POLYGON)
    comps = [_comp(f"C{i}", 80, 60) for i in range(3)]  # all start in the notch
    model = BoardModel(board=board, components=comps, nets=[], user_defined_outline=True)
    model = legalize(model)
    for c in model.components:
        assert board.contains_bbox(c.bbox), f"{c.ref} left in the notch: {c.bbox}"


def test_legalize_notched_board_adversarial_random_sweep():
    """20 seeds x 20 randomly-placed/sized components on a notched board —
    the legalizer must always converge to zero overlaps and zero
    out-of-bounds, exercising the full pipeline (boundary enforcement,
    push-apart, greedy resolve, and post-legalization refine all touch
    component positions and must each respect the true outline)."""
    board = BoardOutline(polygon=NOTCH_POLYGON)
    for seed in range(20):
        random.seed(seed)
        comps = [
            _comp(f"C{i}", random.uniform(0, 100), random.uniform(0, 80),
                  w=random.uniform(2, 6), h=random.uniform(2, 6))
            for i in range(20)
        ]
        model = BoardModel(board=board, components=comps, nets=[], user_defined_outline=True)
        model = legalize(model)
        for c in model.components:
            assert board.contains_bbox(c.bbox), f"seed {seed}: {c.ref} off-board: {c.bbox}"
        overlaps = sum(1 for i, c1 in enumerate(model.components)
                        for c2 in model.components[i + 1:] if c1.overlaps(c2))
        assert overlaps == 0, f"seed {seed}: {overlaps} residual overlaps"


def test_legalize_rectangular_board_unaffected():
    """Plain-rectangle boards must legalize exactly as before — this is a
    guard against the polygon-aware code paths changing behavior when
    there's no polygon."""
    board = BoardOutline(x_min=0, y_min=0, x_max=100, y_max=80)
    random.seed(7)
    comps = [_comp(f"C{i}", random.uniform(0, 100), random.uniform(0, 80)) for i in range(10)]
    model = BoardModel(board=board, components=comps, nets=[], user_defined_outline=True)
    model = legalize(model)
    for c in model.components:
        assert board.contains_bbox(c.bbox)


# ---------------------------------------------------------------------------
# DFM edge-keepout margin on polygon outlines (approximate, not skipped)
# ---------------------------------------------------------------------------

def test_fit_bbox_inside_margin_increases_clearance():
    p = BoardOutline(polygon=NOTCH_POLYGON)
    cx0, cy0 = p.fit_bbox_inside(80, 60, 5, 5, margin=0.0)
    cx2, cy2 = p.fit_bbox_inside(80, 60, 5, 5, margin=2.0)
    # The padded (component + margin) bbox must itself fit the true outline.
    assert p.contains_bbox((cx2 - 7, cy2 - 7, cx2 + 7, cy2 + 7))
    corners = [(cx2 - 5, cy2 - 5), (cx2 + 5, cy2 - 5), (cx2 + 5, cy2 + 5), (cx2 - 5, cy2 + 5)]
    assert min(p.distance_to_boundary(x, y) for x, y in corners) >= 2.0 - 1e-9


def test_fit_bbox_inside_margin_falls_back_when_it_would_collapse_the_board():
    # A margin larger than half the board itself can't be honored — must
    # fall back to no margin rather than leaving the component unclamped.
    r = BoardOutline(x_min=0, y_min=0, x_max=10, y_max=10)
    x, y = r.fit_bbox_inside(5, 5, 4, 4, margin=10.0)
    assert r.contains_bbox((x - 4, y - 4, x + 4, y + 4))


def test_legalize_honors_dfm_margin_on_notched_board():
    """An IC/MCU-class component (which carries a nonzero DFM edge
    keepout) must end up with at least that much clearance from a
    polygon board's boundary — not just somewhere fully on the board."""
    from engine.cost_state import edge_keepout_extra_for

    board = BoardOutline(polygon=NOTCH_POLYGON)
    ic = Component(ref="U1", footprint="ic", value="MCU", x=80, y=60,
                    width=10, height=10, component_type="mcu")
    extra = edge_keepout_extra_for(ic)
    assert extra > 0.0

    model = BoardModel(board=board, components=[ic], nets=[], user_defined_outline=True)
    model = legalize(model)
    c = model.components[0]
    assert board.contains_bbox(c.bbox)
    x1, y1, x2, y2 = c.bbox
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    clearances = [board.distance_to_boundary(x, y) for x, y in corners]
    assert min(clearances) >= extra - 1e-6


# ---------------------------------------------------------------------------
# Default macro-v2 pipeline: rigid macros + SA + legalizer must all honor
# the true outline (the layer the original patch left unwired).
# ---------------------------------------------------------------------------

def test_fit_macro_to_polygon_pulls_rigid_macro_out_of_notch():
    """Unit test for the macro-level polygon fit: a rigid IC+caps macro
    whose union bbox sits in the notch must be translated — as one rigid
    body — onto the true board, with the cap-IC invariant preserved
    (followers move with the leader, so the union half-extents are what
    gets fit)."""
    from models.macro import Macro
    from place.legalizer import _fit_macro_to_polygon

    board = BoardOutline(polygon=NOTCH_POLYGON)
    leader = Component(ref="U1", footprint="ic", value="MCU", x=80, y=60,
                       width=6, height=6, component_type="ic")
    caps = [
        Component(ref=f"C{i}", footprint="cap", value="100nF", x=80, y=60,
                  width=2, height=1, component_type="capacitor")
        for i in range(2)
    ]
    m = Macro.with_caps(leader, caps)

    # Sanity: the rigid union bbox starts off the true board (in the notch).
    assert not board.contains_bbox(m.bbox), "macro should start in the notch"

    aabb = (board.x_min, board.y_min, board.x_max, board.y_max)
    ok = _fit_macro_to_polygon(m, board, bounds=aabb)

    assert ok, "fit should succeed (plenty of room outside the notch)"
    assert board.contains_bbox(m.bbox), f"macro union bbox still off-board: {m.bbox}"
    # Rigid-body invariant: every member is on the true board after the move.
    for c in m.members:
        assert board.contains_bbox(c.bbox), f"{c.ref} left off-board: {c.bbox}"


def test_place_v2_keeps_components_off_notch():
    """End-to-end default macro-v2 pipeline on a notched board. After
    net-aware initial placement → SA → legalization, every non-fixed
    component must be fully on the TRUE board — nothing parked in the
    notch. This exercises the polygon-aware cost gradient (cost.bbox_overflow
    in both the from-scratch evaluator and the incremental SA tracker),
    the polygon-aware overlap-aware boundary clamp, and the polygon-aware
    Tetris pass together — the default-path coverage the foundation patch
    had left as dead code."""
    from place.pipeline import place_v2

    board = BoardOutline(polygon=NOTCH_POLYGON)
    comps = []
    # 4 ICs, each with 2 co-located decoupling caps so build_macros forms
    # rigid IC+cap macros. Several start inside the notch (adversarial).
    ic_positions = [(80, 60), (85, 70), (30, 30), (20, 60)]
    for i, (x, y) in enumerate(ic_positions):
        comps.append(Component(ref=f"U{i}", footprint="ic", value="IC", x=x, y=y,
                               width=6, height=6, component_type="ic"))
        for j in range(2):
            comps.append(Component(ref=f"C{i}_{j}", footprint="cap", value="100nF",
                                   x=x, y=y, width=2, height=1,
                                   component_type="capacitor"))
    # A couple of standalone resistors (one in the notch, one in the clear).
    comps.append(Component(ref="R0", footprint="res", value="10k", x=85, y=65,
                           width=2, height=1, component_type="resistor"))
    comps.append(Component(ref="R1", footprint="res", value="10k", x=15, y=15,
                           width=2, height=1, component_type="resistor"))

    nets = [Net(name=f"SIG{i}", pins=[(f"U{i}", "1"), (f"U{(i + 1) % 4}", "2")])
            for i in range(4)]
    model = BoardModel(board=board, components=comps, nets=nets,
                       user_defined_outline=True)
    model.rebuild_ref_index()

    place_v2(model, sa_iterations=800, sa_reheats=2, seed=42, verbose=False)

    # user_defined_outline=True ⇒ the pipeline must NOT have replaced the
    # polygon with a grown rectangle; re-read the (true) outline from the model.
    outline = model.board
    assert outline.is_polygon, "pipeline overwrote the user-defined polygon outline"
    for c in model.components:
        if c.is_fixed or getattr(c, "is_edge_connector", False):
            continue
        assert outline.contains_bbox(c.bbox), \
            f"{c.ref} ended off the true board (in the notch?): {c.bbox}"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
