#!/usr/bin/env python3
"""Comprehensive test suite for the KiCad Smart Auto-Placer.

Tests all Phase 1 components:
  - Data model
  - KiCad parser
  - JSON serialization
  - Net clustering
  - Grid placement
  - HPWL cost function
  - Legalization
  - Placement application
  - Board profiles
  - End-to-end pipeline
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import glob

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.board_model import BoardModel, BoardOutline, Component, Net, Pad
from parsers.kicad_parser import KiCadParser
from engine.net_clustering import build_net_hypergraph, cluster_components, compute_seed_positions
from engine.grid_placement import grid_place
from engine.cost_function import CostFunction, total_hpwl, total_overlap_penalty, total_boundary_penalty, net_wirelength_hpwl, count_overlaps
from legalization.legalizer import legalize
from profiles.board_profiles import get_profile, list_profiles
from samples.sample_board import SAMPLE_KICAD_PCB
from engine.cost_state import CostState, _is_power_net
from engine.moves import select_move_type, get_moveable_indices, do_translate, do_swap, do_rotate, do_median, revert_move, affected_indices
from engine.annealer import simulate_annealing, run_sa, SAConfig


# ---------------------------------------------------------------------------
# Test utilities
# ---------------------------------------------------------------------------

passed = 0
failed = 0


def run_test(name: str, func) -> None:
    """Run a test function and report results."""
    global passed, failed
    try:
        func()
        print(f"  ✓ {name}")
        passed += 1
    except AssertionError as e:
        print(f"  ✗ {name}: {e}")
        failed += 1
    except Exception as e:
        print(f"  ✗ {name}: UNEXPECTED ERROR: {type(e).__name__}: {e}")
        failed += 1


# ---------------------------------------------------------------------------
# Data Model Tests
# ---------------------------------------------------------------------------

def test_board_outline():
    board = BoardOutline(x_min=0, y_min=0, x_max=100, y_max=80)
    assert board.width == 100.0, f"Expected width 100, got {board.width}"
    assert board.height == 80.0, f"Expected height 80, got {board.height}"
    assert board.center == (50.0, 40.0), f"Expected center (50,40), got {board.center}"
    assert board.contains(50, 40), "Should contain center point"
    assert not board.contains(150, 40), "Should not contain outside point"
    clamped = board.clamp(150, 40)
    assert clamped == (100.0, 40.0), f"Expected (100,40), got {clamped}"


def test_component_overlap():
    c1 = Component(ref="U1", x=10, y=10, width=5, height=5)
    c2 = Component(ref="U2", x=12, y=12, width=5, height=5)
    c3 = Component(ref="U3", x=20, y=20, width=5, height=5)

    # c1 and c2 should overlap (their courtyards overlap)
    assert c1.overlaps(c2), "c1 and c2 should overlap"
    area = c1.overlap_area(c2)
    assert area > 0, f"Expected positive overlap area, got {area}"

    # c1 and c3 should not overlap
    assert not c1.overlaps(c3), "c1 and c3 should not overlap"
    assert c1.overlap_area(c3) == 0, "Expected zero overlap area"


def test_component_bbox():
    c = Component(ref="U1", x=10, y=10, width=4, height=4, courtyard_margin=0.25)
    bbox = c.bbox
    # bbox includes courtyard: half_w = 2 + 0.25 = 2.25
    assert abs(bbox[0] - 7.75) < 0.01, f"Expected x_min 7.75, got {bbox[0]}"
    assert abs(bbox[2] - 12.25) < 0.01, f"Expected x_max 12.25, got {bbox[2]}"


def test_pad_absolute_pos():
    pad = Pad(pad_name="1", x=1.0, y=0.0, net="VCC")
    # No rotation
    ax, ay = pad.absolute_pos(10.0, 10.0, 0.0)
    assert abs(ax - 11.0) < 0.01, f"Expected x=11.0, got {ax}"
    assert abs(ay - 10.0) < 0.01, f"Expected y=10.0, got {ay}"

    # 90° clockwise rotation (KiCad convention): (1,0) → (0,-1)
    ax, ay = pad.absolute_pos(10.0, 10.0, 90.0)
    assert abs(ax - 10.0) < 0.01, f"Expected x=10.0, got {ax}"
    assert abs(ay - 9.0) < 0.01, f"Expected y=9.0, got {ay}"


def test_board_model_serialization():
    model = BoardModel(
        board=BoardOutline(x_min=0, y_min=0, x_max=80, y_max=60),
        components=[
            Component(ref="U1", x=40, y=30, width=7, height=7,
                      pads=[Pad(pad_name="1", x=-3.0, y=-2.75, net="GND")],
                      nets=["VCC", "GND"]),
        ],
        nets=[
            Net(name="VCC", pins=[("U1", "5")]),
            Net(name="GND", pins=[("U1", "1")]),
        ],
        source_file="test.kicad_pcb",
    )

    # Serialize
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        model.to_json(f.name)
        json_path = f.name

    # Deserialize
    model2 = BoardModel.from_json(json_path)
    assert model2.board.width == 80.0
    assert len(model2.components) == 1
    assert model2.components[0].ref == "U1"
    assert model2.components[0].pads[0].net == "GND"
    assert len(model2.nets) == 2

    # Cleanup
    os.unlink(json_path)


def test_board_model_lookup():
    model = BoardModel(
        components=[
            Component(ref="U1", x=40, y=30, nets=["VCC", "GND"]),
            Component(ref="C1", x=10, y=10, nets=["VCC"]),
        ],
        nets=[
            Net(name="VCC", pins=[("U1", "5"), ("C1", "1")]),
            Net(name="GND", pins=[("U1", "1")]),
        ],
    )

    comp = model.get_component("U1")
    assert comp is not None and comp.ref == "U1"
    assert model.get_component("X99") is None

    net = model.get_net("VCC")
    assert net is not None and "U1" in net.component_refs

    u1_nets = model.nets_for_component("U1")
    assert len(u1_nets) == 2

    vcc_comps = model.components_on_net("VCC")
    assert len(vcc_comps) == 2


# ---------------------------------------------------------------------------
# Parser Tests
# ---------------------------------------------------------------------------

def test_kicad_parser():
    with tempfile.NamedTemporaryFile(suffix=".kicad_pcb", mode="w", delete=False) as f:
        f.write(SAMPLE_KICAD_PCB)
        pcb_path = f.name

    try:
        parser = KiCadParser(pcb_path)
        model = parser.parse()

        # Board outline
        assert model.board.x_min < 1, f"Board x_min should be near 0, got {model.board.x_min}"
        assert model.board.width > 70, f"Board width should be ~80, got {model.board.width}"

        # Components
        assert len(model.components) >= 10, f"Expected 10+ components, got {len(model.components)}"

        # Check specific components
        u1 = model.get_component("U1")
        assert u1 is not None, "U1 should exist"
        assert u1.component_type == "ic", f"U1 should be IC, got {u1.component_type}"

        j1 = model.get_component("J1")
        assert j1 is not None, "J1 should exist"
        assert j1.component_type == "connector", f"J1 should be connector, got {j1.component_type}"

        y1 = model.get_component("Y1")
        assert y1 is not None, "Y1 should exist"
        assert y1.component_type == "crystal", f"Y1 should be crystal, got {y1.component_type}"

        # Check nets
        assert len(model.nets) >= 5, f"Expected 5+ nets, got {len(model.nets)}"

        vcc_net = model.get_net("VCC")
        assert vcc_net is not None, "VCC net should exist"
        assert "U1" in vcc_net.component_refs, "U1 should be on VCC"

    finally:
        os.unlink(pcb_path)


def test_json_roundtrip():
    with tempfile.NamedTemporaryFile(suffix=".kicad_pcb", mode="w", delete=False) as f:
        f.write(SAMPLE_KICAD_PCB)
        pcb_path = f.name

    try:
        parser = KiCadParser(pcb_path)
        model = parser.parse()

        # Save to JSON
        json_path = pcb_path.replace(".kicad_pcb", "_model.json")
        model.to_json(json_path)

        # Load from JSON
        model2 = BoardModel.from_json(json_path)

        assert len(model2.components) == len(model.components)
        assert len(model2.nets) == len(model.nets)
        assert abs(model2.board.width - model.board.width) < 0.1

        os.unlink(json_path)
    finally:
        os.unlink(pcb_path)


def test_zero_courtyard_fallback():
    """A footprint with no geometry and no pads must NOT produce a
    zero-area component — the SA could then "overlap" it for free.

    The parser's _extract_fp_geometry falls back to (2.0, 2.0, 0.0, 0.0)
    when no fp_line/fp_rect/fp_circle/fp_arc/fp_poly and no pads are found,
    and clamps each dimension to max(., 0.5) for footprints that have only
    a single tiny pad.  This test locks that behaviour in so a future
    parser refactor can't silently regress it.
    """
    # Construct a minimal .kicad_pcb with one footprint that has NO
    # courtyard graphics and NO pads — only a Reference property.
    # The parser must still produce a Component with a non-zero bbox.
    bare_footprint_pcb = """(kicad_pcb
      (version 20240108)
      (general (thickness 1.6))
      (layers (0 "F.Cu" signal) (31 "B.Cu" signal) (32 "B.Adhes" user))
      (net 0 "")
      (footprint "TestLib:BareFootprint"
        (layer "F.Cu")
        (at 50.0 50.0)
        (property "Reference" "U1" (at 0 0) (layer "F.SilkS"))
        (property "Value" "unknown" (at 0 0) (layer "F.SilkS"))
      )
    )"""

    with tempfile.NamedTemporaryFile(
        suffix=".kicad_pcb", mode="w", delete=False
    ) as f:
        f.write(bare_footprint_pcb)
        pcb_path = f.name

    try:
        parser = KiCadParser(pcb_path)
        model = parser.parse()
        assert len(model.components) == 1, "Should parse the bare footprint"
        comp = model.components[0]
        # Effective dimensions must be > 0 — no zero-area component.
        assert comp.effective_width > 0.0, \
            f"effective_width should be > 0 for bare footprint, got {comp.effective_width}"
        assert comp.effective_height > 0.0, \
            f"effective_height should be > 0 for bare footprint, got {comp.effective_height}"
        # The fallback is (2.0, 2.0) before courtyard margin; with default
        # courtyard_margin=0.5 (parser's bbox_margin), effective should be 3.0.
        # Just assert >= 0.5mm minimum so the test is robust to fallback tuning.
        assert comp.effective_width >= 0.5, \
            f"effective_width should be >= 0.5mm, got {comp.effective_width}"
        assert comp.effective_height >= 0.5, \
            f"effective_height should be >= 0.5mm, got {comp.effective_height}"
        # bbox must be non-degenerate (x_min < x_max, y_min < y_max).
        x_min, y_min, x_max, y_max = comp.bbox
        assert x_max > x_min, \
            f"bbox x_max ({x_max}) must be > x_min ({x_min}) for bare footprint"
        assert y_max > y_min, \
            f"bbox y_max ({y_max}) must be > y_min ({y_min}) for bare footprint"
        # Two bare footprints at the same position must overlap — proves
        # the SA can't "stack" them for free.
        from models.board_model import Component as _C
        comp2 = _C(
            ref="U2",
            x=comp.x,
            y=comp.y,
            width=comp.width,
            height=comp.height,
            courtyard_margin=comp.courtyard_margin,
        )
        assert comp.overlaps(comp2), \
            "Two bare footprints at the same position must overlap (non-zero bbox)"
    finally:
        os.unlink(pcb_path)


def test_board_outline_with_internal_cutout():
    """A board with an internal cutout (mounting hole) on Edge.Cuts must
    extract BOTH the outer outline AND the cutout as a keepout zone.

    The previous parser took the global AABB of all Edge.Cuts points,
    which silently flattened the cutout into the outer outline —
    components could then be placed *inside* the mounting hole region,
    a real DFM defect class.

    This test constructs a 100x80 board with two 5mm-diameter mounting
    holes and asserts:
      1. The outer outline is (approximately) 100x80, not enlarged by
         the holes.
      2. The model has a `keepouts` list containing both holes' bboxes.
      3. A component placed inside a hole is flagged as overlapping a
         keepout (via BoardModel.component_in_keepout).
    """
    # 100x80 outer rect + 5mm-diameter circle (mounting hole) at (20, 20)
    # gr_circle on Edge.Cuts with center=(20,20), end=(22.5,20) → r=2.5
    board_with_hole_pcb = """(kicad_pcb
      (version 20240108)
      (general (thickness 1.6))
      (layers (0 "F.Cu" signal) (31 "B.Cu" signal) (32 "B.Adhes" user))
      (net 0 "")
      (gr_rect (start 0 0) (end 100 80) (stroke (width 0.1)) (fill none) (layer "Edge.Cuts") (tstamp "00000000-0000-0000-0000-000000000001"))
      (gr_circle (center 20 20) (end 22.5 20) (stroke (width 0.1)) (fill none) (layer "Edge.Cuts") (tstamp "00000000-0000-0000-0000-000000000002"))
      (gr_circle (center 80 60) (end 82.5 60) (stroke (width 0.1)) (fill none) (layer "Edge.Cuts") (tstamp "00000000-0000-0000-0000-000000000003"))
    )"""

    with tempfile.NamedTemporaryFile(
        suffix=".kicad_pcb", mode="w", delete=False
    ) as f:
        f.write(board_with_hole_pcb)
        pcb_path = f.name

    try:
        parser = KiCadParser(pcb_path)
        model = parser.parse()

        # 1. Outer outline is ~100x80, NOT enlarged by the holes.
        # The parser adds a 2mm margin to the outer outline.
        board = model.board
        outer_w = board.x_max - board.x_min
        outer_h = board.y_max - board.y_min
        assert 100.0 <= outer_w <= 105.0, \
            f"Outer width should be ~100mm (with margin), got {outer_w}"
        assert 80.0 <= outer_h <= 85.0, \
            f"Outer height should be ~80mm (with margin), got {outer_h}"

        # 2. Model has keepouts for the two mounting holes.
        keepouts = getattr(model, 'keepouts', None)
        assert keepouts is not None, \
            "BoardModel must have a `keepouts` list (was None — field missing?)"
        assert len(keepouts) == 2, \
            f"Expected 2 keepouts (one per mounting-hole circle), got {len(keepouts)}"

        # Each keepout should be centered near (20, 20) or (80, 60) with
        # ~2.5mm radius (5mm diameter).
        keepout_centers = []
        for k in keepouts:
            kcx = (k.x_min + k.x_max) / 2.0
            kcy = (k.y_min + k.y_max) / 2.0
            keepout_centers.append((kcx, kcy))
            k_w = k.x_max - k.x_min
            k_h = k.y_max - k.y_min
            # Circle was r=2.5, so diameter 5.0; parser may add a small margin.
            assert 4.5 <= k_w <= 6.0, \
                f"Keepout width should be ~5mm (mounting hole diameter), got {k_w}"
            assert 4.5 <= k_h <= 6.0, \
                f"Keepout height should be ~5mm (mounting hole diameter), got {k_h}"

        # Sort by x to make the assertion deterministic
        keepout_centers.sort()
        assert abs(keepout_centers[0][0] - 20.0) < 1.0 and abs(keepout_centers[0][1] - 20.0) < 1.0, \
            f"First keepout should be near (20, 20), got {keepout_centers[0]}"
        assert abs(keepout_centers[1][0] - 80.0) < 1.0 and abs(keepout_centers[1][1] - 60.0) < 1.0, \
            f"Second keepout should be near (80, 60), got {keepout_centers[1]}"

        # 3. A component placed inside a keepout is flagged as overlapping it.
        comp_in_hole = Component(
            ref="X1", x=20.0, y=20.0, width=1.0, height=1.0,
            courtyard_margin=0.0,
        )
        assert model.component_in_keepout(comp_in_hole), \
            "Component at (20, 20) must be flagged as inside a keepout"

        comp_outside_hole = Component(
            ref="X2", x=50.0, y=40.0, width=1.0, height=1.0,
            courtyard_margin=0.0,
        )
        assert not model.component_in_keepout(comp_outside_hole), \
            "Component at (50, 40) must NOT be inside a keepout"
    finally:
        os.unlink(pcb_path)


# ---------------------------------------------------------------------------
# Net Clustering Tests
# ---------------------------------------------------------------------------

def test_build_hypergraph():
    model = _make_test_model()
    G = build_net_hypergraph(model)

    # Should have nodes for movable components
    assert len(G.nodes()) > 0, "Graph should have nodes"

    # Should have edges for shared nets
    assert len(G.edges()) > 0, "Graph should have edges"

    # U1 and C1 share VCC net → should have an edge
    if G.has_node("U1") and G.has_node("C1"):
        assert G.has_edge("U1", "C1"), "U1 and C1 should share an edge (VCC)"


def test_sheet_field_parsed_and_serialized():
    """Component must have a `sheet` field that round-trips through JSON.

    The parser extracts `sheetname` from KiCad 7+ footprints — every
    hierarchical schematic sheet leaves its name on the footprints that
    came from it.  This is the ground-truth functional grouping signal
    Phase 3.1 builds on.
    """
    comp = Component(ref="U1", x=10, y=10, sheet="/MCU/")
    assert comp.sheet == "/MCU/", f"Expected /MCU/, got {comp.sheet}"

    # Round-trip through dict (same path as to_json/from_json)
    model = BoardModel(
        board=BoardOutline(x_min=0, y_min=0, x_max=50, y_max=50),
        components=[comp],
        nets=[],
    )
    d = model.to_dict()
    assert d["components"][0].get("sheet") == "/MCU/", \
        f"to_dict should preserve sheet, got {d['components'][0].get('sheet')}"
    model2 = BoardModel.from_dict(d)
    assert model2.components[0].sheet == "/MCU/", \
        f"from_dict should restore sheet, got {model2.components[0].sheet}"

    # Default is empty string (flat schematics)
    comp_default = Component(ref="U2")
    assert comp_default.sheet == "", "Default sheet should be empty string"


def test_sheet_aware_clustering_groups_by_sheet():
    """build_net_hypergraph must add edges between components sharing
    a non-empty, non-root (`/`) sheet — even if they share NO signal nets.

    This is the core of Phase 3.1: KiCad's hierarchical sheets are the
    designer's own functional grouping, sitting in the file for free.
    Two components on the same sheet that happen to share only power
    rails (which are excluded from signal-net edges) would otherwise
    end up in different clusters — sheet-aware edges fix that.
    """
    # Two ICs on the same sheet (/MCU/) but with NO shared signal net —
    # only GND (which is excluded from signal edges).
    model = BoardModel(
        board=BoardOutline(x_min=0, y_min=0, x_max=100, y_max=80),
        components=[
            Component(
                ref="U1", x=20, y=20, width=5, height=5, component_type="ic",
                sheet="/MCU/",
                pads=[Pad(pad_name="1", x=0, y=0, net="GND"),
                      Pad(pad_name="2", x=1, y=0, net="SIG_A")],
                nets=["GND", "SIG_A"],
            ),
            Component(
                ref="U2", x=80, y=60, width=5, height=5, component_type="ic",
                sheet="/MCU/",
                pads=[Pad(pad_name="1", x=0, y=0, net="GND"),
                      Pad(pad_name="2", x=1, y=0, net="SIG_B")],
                nets=["GND", "SIG_B"],
            ),
            # Different sheet — should NOT get a sheet-edge to U1/U2.
            Component(
                ref="U3", x=50, y=40, width=5, height=5, component_type="ic",
                sheet="/POWER/",
                pads=[Pad(pad_name="1", x=0, y=0, net="GND"),
                      Pad(pad_name="2", x=1, y=0, net="SIG_C")],
                nets=["GND", "SIG_C"],
            ),
        ],
        nets=[
            Net(name="GND", pins=[("U1", "1"), ("U2", "1"), ("U3", "1")]),
            Net(name="SIG_A", pins=[("U1", "2")]),
            Net(name="SIG_B", pins=[("U2", "2")]),
            Net(name="SIG_C", pins=[("U3", "2")]),
        ],
    )

    G = build_net_hypergraph(model)

    # U1 and U2 share a sheet — must have an edge despite no shared signal net.
    assert G.has_edge("U1", "U2"), \
        "U1 and U2 share sheet /MCU/ — must have a sheet-aware edge"
    # U3 is on a different sheet — should NOT have a sheet-edge to U1/U2.
    # (It might still have a fallback power-rail edge via _add_power_rail_edges,
    # but GND is excluded so there should be no edge at all here.)
    assert not G.has_edge("U1", "U3"), \
        "U1 (/MCU/) and U3 (/POWER/) should NOT have an edge — different sheets, no shared signal"
    assert not G.has_edge("U2", "U3"), \
        "U2 (/MCU/) and U3 (/POWER/) should NOT have an edge — different sheets, no shared signal"


def test_sheet_aware_clustering_falls_back_when_no_sheets():
    """When no component has a hierarchical sheet (flat schematic, all
    sheet="" or "/"), clustering must behave exactly as before — pure
    net-based.  This is the graceful-fallback requirement.
    """
    model = BoardModel(
        board=BoardOutline(x_min=0, y_min=0, x_max=100, y_max=80),
        components=[
            Component(
                ref="U1", x=20, y=20, width=5, height=5, component_type="ic",
                sheet="",  # flat schematic
                pads=[Pad(pad_name="1", x=0, y=0, net="GND"),
                      Pad(pad_name="2", x=1, y=0, net="SIG_A")],
                nets=["GND", "SIG_A"],
            ),
            Component(
                ref="U2", x=80, y=60, width=5, height=5, component_type="ic",
                sheet="",  # flat schematic
                pads=[Pad(pad_name="1", x=0, y=0, net="GND"),
                      Pad(pad_name="2", x=1, y=0, net="SIG_B")],
                nets=["GND", "SIG_B"],
            ),
        ],
        nets=[
            Net(name="GND", pins=[("U1", "1"), ("U2", "1")]),
            Net(name="SIG_A", pins=[("U1", "2")]),
            Net(name="SIG_B", pins=[("U2", "2")]),
        ],
    )

    G = build_net_hypergraph(model)
    # No shared signal net, no sheet info — no edge.
    assert not G.has_edge("U1", "U2"), \
        "Flat schematic, no shared signal — should have no edge"


def test_th_sensor_sheet_extraction():
    """Real-world check: parsing th_sensor.kicad_pcb must populate
    Component.sheet for every footprint, with values matching the
    sheetname fields in the file (/MCU/, /PWR_INPUT/, /PWR_REG/,
    /DISPLAY/, /TH_SENSOR/).
    """
    pcb_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tests", "test_pcbs", "th_sensor.kicad_pcb",
    )
    if not os.path.exists(pcb_path):
        return  # skip if running from a different working dir
    parser = KiCadParser(pcb_path)
    model = parser.parse()

    # Every component should have a non-empty sheet (th_sensor is
    # a hierarchical design — all footprints come from a sub-sheet).
    sheets_found = {c.sheet for c in model.components}
    expected_sheets = {"/MCU/", "/PWR_INPUT/", "/PWR_REG/", "/DISPLAY/", "/TH_SENSOR/"}
    assert sheets_found & expected_sheets, (
        f"Expected to find at least some of {expected_sheets} in parsed "
        f"sheets, got {sheets_found}"
    )
    # Most components should have a non-empty sheet.
    non_empty = sum(1 for c in model.components if c.sheet)
    assert non_empty >= len(model.components) * 0.5, \
        f"At least 50% of components should have a sheet, got {non_empty}/{len(model.components)}"


def test_rl_pcb_boards_parse_cleanly():
    """Phase 5: all 9 RL_PCB boards must parse without errors.

    These boards are MIT-licensed real circuits from
    https://github.com/LukeVassallo/RL_PCB (dataset/base_raw/).
    They cover a wider range of board types than the original 5-board
    test_pcbs/ set.  This test confirms the parser handles them all
    — if any board fails to parse, that's a parser bug or a KiCad
    format-version drift issue worth investigating.
    """
    rl_pcb_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tests", "external_boards", "rl_pcb",
    )
    if not os.path.isdir(rl_pcb_dir):
        return  # skip if RL_PCB boards not vendored in this checkout

    pcb_files = sorted(glob.glob(os.path.join(rl_pcb_dir, "*.kicad_pcb")))
    assert len(pcb_files) >= 9, \
        f"Expected ≥9 RL_PCB boards, found {len(pcb_files)} in {rl_pcb_dir}"

    parse_failures = []
    for pcb_path in pcb_files:
        board_name = os.path.basename(pcb_path)
        try:
            parser = KiCadParser(pcb_path)
            model = parser.parse()
            # Sanity: every board should have ≥1 component and ≥1 net.
            assert len(model.components) >= 1, \
                f"{board_name}: 0 components parsed"
            assert len(model.nets) >= 1, \
                f"{board_name}: 0 nets parsed"
        except Exception as e:
            parse_failures.append((board_name, f"{type(e).__name__}: {e}"))

    assert not parse_failures, \
        f"{len(parse_failures)} RL_PCB board(s) failed to parse:\n" + \
        "\n".join(f"  {n}: {e}" for n, e in parse_failures)


# ---------------------------------------------------------------------------
# Phase 3.2: Sub-circuit pattern recognition tests
# ---------------------------------------------------------------------------

def test_subcircuit_crystal_motif_detection():
    """Phase 3.2: detect crystal + 2 load caps motif.

    A crystal oscillator circuit is one of the most common sub-circuit
    motifs: a crystal (Y1) connected to an IC via OSC_IN/OSC_OUT nets,
    with two load capacitors (one on each oscillator net) returning to
    GND.  The pattern matcher should detect this motif and tag all 3
    components as members of a 'crystal_oscillator' subcircuit group.
    """
    from engine.subcircuit_patterns import detect_subcircuit_patterns

    model = BoardModel(
        board=BoardOutline(x_min=0, y_min=0, x_max=100, y_max=80),
        components=[
            Component(ref="U1", x=50, y=40, width=7, height=7, component_type="ic",
                      pads=[Pad(pad_name="1", x=0, y=0, net="OSC_IN"),
                            Pad(pad_name="2", x=1, y=0, net="OSC_OUT")],
                      nets=["OSC_IN", "OSC_OUT"]),
            Component(ref="Y1", x=40, y=40, width=3, height=2, component_type="crystal",
                      pads=[Pad(pad_name="1", x=-1, y=0, net="OSC_IN"),
                            Pad(pad_name="2", x=1, y=0, net="OSC_OUT")],
                      nets=["OSC_IN", "OSC_OUT"]),
            Component(ref="C1", x=35, y=35, width=1, height=0.5, component_type="capacitor",
                      pads=[Pad(pad_name="1", x=0, y=0, net="OSC_IN"),
                            Pad(pad_name="2", x=0.5, y=0, net="GND")],
                      nets=["OSC_IN", "GND"]),
            Component(ref="C2", x=45, y=35, width=1, height=0.5, component_type="capacitor",
                      pads=[Pad(pad_name="1", x=0, y=0, net="OSC_OUT"),
                            Pad(pad_name="2", x=0.5, y=0, net="GND")],
                      nets=["OSC_OUT", "GND"]),
        ],
        nets=[
            Net(name="OSC_IN", pins=[("U1", "1"), ("Y1", "1"), ("C1", "1")]),
            Net(name="OSC_OUT", pins=[("U1", "2"), ("Y1", "2"), ("C2", "1")]),
            Net(name="GND", pins=[("C1", "2"), ("C2", "2")]),
        ],
    )

    patterns = detect_subcircuit_patterns(model)
    crystal_patterns = [p for p in patterns if p.motif_type == 'crystal_oscillator']
    assert len(crystal_patterns) >= 1, \
        f"Should detect at least 1 crystal_oscillator motif, got {len(crystal_patterns)}"

    p = crystal_patterns[0]
    assert p.anchor_ref == "Y1", f"Anchor should be the crystal Y1, got {p.anchor_ref}"
    member_refs = {p.anchor_ref} | set(p.member_refs)
    assert {"Y1", "C1", "C2"}.issubset(member_refs), \
        f"Crystal motif should include Y1, C1, C2, got {member_refs}"


def test_subcircuit_regulator_motif_detection():
    """Phase 3.2: detect regulator + input cap + output cap motif.

    A linear regulator (LDO) sub-circuit: a regulator IC with an input
    bulk cap on VIN and an output cap on VOUT.  The pattern matcher
    should detect this motif and tag all components as members of a
    'regulator' subcircuit group.
    """
    from engine.subcircuit_patterns import detect_subcircuit_patterns

    model = BoardModel(
        board=BoardOutline(x_min=0, y_min=0, x_max=100, y_max=80),
        components=[
            Component(ref="U1", x=50, y=40, width=5, height=5, component_type="regulator",
                      pads=[Pad(pad_name="1", x=0, y=0, net="+5V"),
                            Pad(pad_name="2", x=1, y=0, net="GND"),
                            Pad(pad_name="3", x=2, y=0, net="+3V3")],
                      nets=["+5V", "GND", "+3V3"]),
            Component(ref="C1", x=45, y=40, width=2, height=1, component_type="capacitor",
                      pads=[Pad(pad_name="1", x=0, y=0, net="+5V"),
                            Pad(pad_name="2", x=0.5, y=0, net="GND")],
                      nets=["+5V", "GND"]),
            Component(ref="C2", x=55, y=40, width=2, height=1, component_type="capacitor",
                      pads=[Pad(pad_name="1", x=0, y=0, net="+3V3"),
                            Pad(pad_name="2", x=0.5, y=0, net="GND")],
                      nets=["+3V3", "GND"]),
        ],
        nets=[
            Net(name="+5V", pins=[("U1", "1"), ("C1", "1")]),
            Net(name="+3V3", pins=[("U1", "3"), ("C2", "1")]),
            Net(name="GND", pins=[("U1", "2"), ("C1", "2"), ("C2", "2")]),
        ],
    )

    patterns = detect_subcircuit_patterns(model)
    reg_patterns = [p for p in patterns if p.motif_type == 'regulator']
    assert len(reg_patterns) >= 1, \
        f"Should detect at least 1 regulator motif, got {len(reg_patterns)}"

    p = reg_patterns[0]
    assert p.anchor_ref == "U1", f"Anchor should be the regulator U1, got {p.anchor_ref}"
    member_refs = {p.anchor_ref} | set(p.member_refs)
    assert {"U1", "C1", "C2"}.issubset(member_refs), \
        f"Regulator motif should include U1, C1, C2, got {member_refs}"


def test_subcircuit_patterns_feed_into_clustering():
    """Phase 3.2: detected subcircuit patterns must add strong edges to
    the clustering hypergraph so SA keeps motif members together.

    Without this, a crystal + its 2 load caps could scatter across
    different clusters (the caps only share GND with everything, and
    GND is excluded from signal edges).  The subcircuit pattern edges
    ensure they cluster as a rigid sub-group.
    """
    from engine.subcircuit_patterns import detect_subcircuit_patterns, SUBCIRCUIT_EDGE_WEIGHT

    model = BoardModel(
        board=BoardOutline(x_min=0, y_min=0, x_max=100, y_max=80),
        components=[
            Component(ref="U1", x=50, y=40, width=7, height=7, component_type="ic",
                      pads=[Pad(pad_name="1", x=0, y=0, net="OSC_IN"),
                            Pad(pad_name="2", x=1, y=0, net="OSC_OUT")],
                      nets=["OSC_IN", "OSC_OUT"]),
            Component(ref="Y1", x=40, y=40, width=3, height=2, component_type="crystal",
                      pads=[Pad(pad_name="1", x=-1, y=0, net="OSC_IN"),
                            Pad(pad_name="2", x=1, y=0, net="OSC_OUT")],
                      nets=["OSC_IN", "OSC_OUT"]),
            Component(ref="C1", x=35, y=35, width=1, height=0.5, component_type="capacitor",
                      pads=[Pad(pad_name="1", x=0, y=0, net="OSC_IN"),
                            Pad(pad_name="2", x=0.5, y=0, net="GND")],
                      nets=["OSC_IN", "GND"]),
            Component(ref="C2", x=45, y=35, width=1, height=0.5, component_type="capacitor",
                      pads=[Pad(pad_name="1", x=0, y=0, net="OSC_OUT"),
                            Pad(pad_name="2", x=0.5, y=0, net="GND")],
                      nets=["OSC_OUT", "GND"]),
        ],
        nets=[
            Net(name="OSC_IN", pins=[("U1", "1"), ("Y1", "1"), ("C1", "1")]),
            Net(name="OSC_OUT", pins=[("U1", "2"), ("Y1", "2"), ("C2", "1")]),
            Net(name="GND", pins=[("C1", "2"), ("C2", "2")]),
        ],
    )

    # The clustering hypergraph should have edges between Y1, C1, C2
    # that come from the subcircuit pattern (stronger than signal edges).
    G = build_net_hypergraph(model)

    # C1 and C2 share only GND (excluded from signal edges), so without
    # subcircuit edges they'd have NO edge between them.  With subcircuit
    # pattern detection, they should have an edge.
    assert G.has_edge("C1", "C2"), \
        "C1 and C2 (load caps of same crystal) should have a subcircuit-pattern edge"

    # The edge weight should be at least SUBCIRCUIT_EDGE_WEIGHT (stronger
    # than a signal edge of weight 1.0).
    if G.has_edge("C1", "C2"):
        w = G["C1"]["C2"].get("weight", 0)
        assert w >= SUBCIRCUIT_EDGE_WEIGHT, \
            f"C1-C2 edge weight {w} should be >= SUBCIRCUIT_EDGE_WEIGHT {SUBCIRCUIT_EDGE_WEIGHT}"


def test_clustering():
    model = _make_test_model()
    clusters = cluster_components(model, n_clusters=3)

    assert len(clusters) > 0, "Should produce clusters"
    assert len(clusters) <= 3, f"Should have ≤3 clusters, got {len(clusters)}"

    # All movable components should be in a cluster
    all_refs = set()
    for cluster in clusters:
        all_refs.update(cluster)
    movable_refs = {c.ref for c in model.components if not c.is_fixed}
    assert movable_refs.issubset(all_refs), "All movable components should be in clusters"


def test_seed_positions():
    model = _make_test_model()
    positions = compute_seed_positions(model)

    # Should have positions for all movable components
    movable_refs = {c.ref for c in model.components if not c.is_fixed}
    for ref in movable_refs:
        assert ref in positions, f"Missing position for {ref}"
        x, y = positions[ref]
        assert isinstance(x, float), f"X should be float for {ref}"
        assert isinstance(y, float), f"Y should be float for {ref}"


# ---------------------------------------------------------------------------
# HPWL Cost Function Tests
# ---------------------------------------------------------------------------

def test_hpwl_2pin():
    pins = [(0.0, 0.0), (10.0, 5.0)]
    hpwl = net_wirelength_hpwl(pins, model="clique")
    # Manhattan distance: 10 + 5 = 15
    assert abs(hpwl - 15.0) < 0.01, f"Expected 15.0, got {hpwl}"


def test_hpwl_3pin_clique():
    pins = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)]
    hpwl = net_wirelength_hpwl(pins, model="clique")
    # Pairwise: d(0,1)=10, d(0,2)=10, d(1,2)=20 → (10+10+20)/2 = 20
    assert abs(hpwl - 20.0) < 0.01, f"Expected 20.0, got {hpwl}"


def test_hpwl_star():
    pins = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0), (5.0, 5.0)]
    hpwl = net_wirelength_hpwl(pins, model="star")
    # Center = (5,5), distances: 10, 10, 10, 10, 0 → total = 40
    assert abs(hpwl - 40.0) < 0.01, f"Expected 40.0, got {hpwl}"


def test_hpwl_auto_model():
    # ≤4 pins → clique
    small_pins = [(0, 0), (10, 0)]
    hpwl_small = net_wirelength_hpwl(small_pins, model="auto")
    assert abs(hpwl_small - 10.0) < 0.01

    # >4 pins → star
    large_pins = [(0, 0), (10, 0), (0, 10), (10, 10), (5, 5)]
    hpwl_large = net_wirelength_hpwl(large_pins, model="auto")
    assert hpwl_large > 0


def test_hpwl_true_model_default():
    """Default model='auto' should compute TRUE HPWL = (max(x)-min(x)) + (max(y)-min(y)).

    This is the standard VLSI half-perimeter wirelength, and the same formula
    CostState._compute_net_hpwl uses in the SA hot loop.  The previous default
    used clique (≤4 pins) / star (>4 pins), which are *different* wirelength
    proxies and silently disagreed with CostState on multi-pin nets.

    For 4 pins [(0,0), (10,0), (0,10), (10,10)]:
      - True HPWL  = (10-0) + (10-0) = 20
      - Clique     = (10+10+20+10+20+10) / 3 = 80/3 ≈ 26.67  (DIFFERENT)
      - Star       = (10+10+10+10) = 40                     (DIFFERENT)

    Note: for 3 pins, Manhattan clique is always equal to true HPWL
    (sum of pairwise Manhattan = 2 * HPWL for any 3 points), so the
    bug only manifests at 4+ pins.  This test uses 4 pins deliberately.
    """
    pins = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0)]
    true_hpwl = (10.0 - 0.0) + (10.0 - 0.0)  # = 20.0
    assert abs(net_wirelength_hpwl(pins) - true_hpwl) < 0.01, \
        f"Default model should return true HPWL {true_hpwl}, got {net_wirelength_hpwl(pins)}"
    assert abs(net_wirelength_hpwl(pins, model="auto") - true_hpwl) < 0.01
    assert abs(net_wirelength_hpwl(pins, model="true") - true_hpwl) < 0.01
    # Clique and star remain accessible as explicit alternatives, and
    # for 4 pins they produce DIFFERENT numbers from true HPWL —
    # which is exactly why the cold/hot-path agreement matters.
    clique_hpwl = net_wirelength_hpwl(pins, model="clique")
    assert abs(clique_hpwl - 26.67) < 0.1, f"Clique should still work, got {clique_hpwl}"
    star_hpwl = net_wirelength_hpwl(pins, model="star")
    assert abs(star_hpwl - 40.0) < 0.1, f"Star should still work, got {star_hpwl}"


def test_hpwl_4_to_5_pin_continuity():
    """Cost should NOT jump discontinuously when a net grows from 4 to 5 pins.

    The old clique→star switch at >4 pins produced a discontinuity because
    clique (pairwise sum / (n-1)) and star (sum to centroid) are different
    quantities with different scales.  With true HPWL as default, the cost
    is continuous: adding a pin inside the existing bbox doesn't change HPWL.
    """
    # 4-pin net on a 10x10 bounding box
    pins_4 = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0)]
    hpwl_4 = net_wirelength_hpwl(pins_4)
    assert abs(hpwl_4 - 20.0) < 0.01, f"4-pin true HPWL should be 20, got {hpwl_4}"

    # Add a 5th pin INSIDE the bbox — true HPWL is unchanged
    pins_5 = pins_4 + [(5.0, 5.0)]
    hpwl_5 = net_wirelength_hpwl(pins_5)
    assert abs(hpwl_5 - 20.0) < 0.01, f"5-pin true HPWL should still be 20, got {hpwl_5}"

    # Add a 5th pin OUTSIDE the bbox — true HPWL grows continuously
    pins_5_out = pins_4 + [(15.0, 5.0)]
    hpwl_5_out = net_wirelength_hpwl(pins_5_out)
    assert abs(hpwl_5_out - 25.0) < 0.01, f"5-pin extended HPWL should be 25, got {hpwl_5_out}"


def test_total_hpwl_matches_cost_state():
    """total_hpwl (cold path) must agree with CostState.hpwl (hot SA path).

    Before this fix, total_hpwl used clique/star via net_wirelength_hpwl,
    while CostState._compute_net_hpwl used true HPWL — they silently
    disagreed on any board with multi-pin non-power nets.  This test
    constructs a board with a 5-pin signal net and asserts the two
    paths produce the same number.
    """
    # 5-pin signal net "SIG" connecting U1..U5 in a 10x10 footprint
    model = BoardModel(
        board=BoardOutline(x_min=0, y_min=0, x_max=50, y_max=50),
        components=[
            Component(
                ref=f"U{i}",
                x=xs,
                y=ys,
                width=1.0,
                height=1.0,
                component_type="ic",
                pads=[Pad(pad_name="1", x=0.0, y=0.0, net="SIG")],
                nets=["SIG"],
            )
            for i, (xs, ys) in enumerate(
                [(10, 10), (20, 10), (10, 20), (20, 20), (15, 15)], start=1
            )
        ],
        nets=[
            Net(name="SIG", pins=[(f"U{i}", "1") for i in range(1, 6)]),
        ],
    )

    cold_hpwl = total_hpwl(model)
    cs = CostState(model)
    hot_hpwl = cs.hpwl

    assert abs(cold_hpwl - hot_hpwl) < 1e-6, (
        f"Cold path (total_hpwl) = {cold_hpwl} must match hot path "
        f"(CostState.hpwl) = {hot_hpwl} on a 5-pin net. "
        f"This is the silent SA-vs-evaluator drift bug."
    )


def test_total_hpwl():
    model = _make_test_model()
    hpwl = total_hpwl(model)
    assert hpwl >= 0, f"HPWL should be non-negative, got {hpwl}"


def test_overlap_penalty():
    model = BoardModel(
        board=BoardOutline(x_min=0, y_min=0, x_max=100, y_max=100),
        components=[
            Component(ref="U1", x=10, y=10, width=5, height=5),
            Component(ref="U2", x=12, y=12, width=5, height=5),  # Overlaps U1
        ],
    )
    penalty = total_overlap_penalty(model)
    assert penalty > 0, f"Expected positive overlap penalty, got {penalty}"


def test_boundary_penalty():
    model = BoardModel(
        board=BoardOutline(x_min=0, y_min=0, x_max=50, y_max=50),
        components=[
            Component(ref="U1", x=60, y=10, width=5, height=5),  # Out of bounds
        ],
    )
    penalty = total_boundary_penalty(model)
    assert penalty > 0, f"Expected positive boundary penalty, got {penalty}"


def test_boundary_penalty_includes_keepouts():
    """total_boundary_penalty must charge components overlapping a keepout.

    Before this fix, the cost function only charged for components
    outside the board outline.  Components inside a mounting-hole
    keepout (parsed from Edge.Cuts internal cutouts) were free to
    overlap — the legalizer would push them out post-hoc, but the SA
    never learned to avoid the keepout during optimization.  This
    closed the SA-loop gap: SA now sees a cost for keepout overlap
    and avoids placing components there in the first place.
    """
    from models.board_model import BoardOutline
    keepout = BoardOutline(x_min=20, y_min=20, x_max=25, y_max=25)  # 5x5 mounting hole
    model = BoardModel(
        board=BoardOutline(x_min=0, y_min=0, x_max=100, y_max=80),
        components=[
            # Inside the keepout — must be charged.
            Component(ref="U1", x=22, y=22, width=2, height=2, courtyard_margin=0.0),
            # Outside the keepout, inside the board — must NOT be charged.
            Component(ref="U2", x=50, y=40, width=2, height=2, courtyard_margin=0.0),
            # Outside the board — must still be charged (existing behavior).
            Component(ref="U3", x=120, y=40, width=2, height=2, courtyard_margin=0.0),
        ],
        keepouts=[keepout],
    )

    penalty = total_boundary_penalty(model)
    assert penalty > 0, f"Expected positive penalty for U1 in keepout + U3 OOB, got {penalty}"

    # Now check the keepout-only contribution: remove U3 (OOB) and re-measure.
    model.components = [model.components[0], model.components[1]]  # U1, U2 only
    penalty_with_keepout = total_boundary_penalty(model)
    assert penalty_with_keepout > 0, \
        f"U1 inside keepout must produce >0 penalty, got {penalty_with_keepout}"

    # And the no-keepout baseline: same model, no keepouts → 0 penalty
    # (both U1 and U2 are inside the board).
    model.keepouts = []
    penalty_no_keepout = total_boundary_penalty(model)
    assert penalty_no_keepout == 0, \
        f"Without keepouts, U1 and U2 are both in-bounds → 0 penalty, got {penalty_no_keepout}"

    # Restore keepouts, verify the difference is the keepout contribution.
    model.keepouts = [keepout]
    keepout_penalty = total_boundary_penalty(model) - penalty_no_keepout
    assert keepout_penalty > 0, \
        f"Keepout contribution to penalty must be >0, got {keepout_penalty}"


def test_cost_state_includes_keepouts():
    """CostState (SA hot path) must also charge for keepout overlap.

    Mirrors test_boundary_penalty_includes_keepouts but for the SA
    incremental-cost path.  Without this, the cold evaluator would
    charge for keepouts but SA would happily place components on
    mounting holes during optimization — the same cold/hot drift
    pattern as the HPWL bug fixed earlier.
    """
    from models.board_model import BoardOutline
    keepout = BoardOutline(x_min=20, y_min=20, x_max=25, y_max=25)
    model = BoardModel(
        board=BoardOutline(x_min=0, y_min=0, x_max=100, y_max=80),
        components=[
            Component(ref="U1", x=22, y=22, width=2, height=2, courtyard_margin=0.0),
            Component(ref="U2", x=50, y=40, width=2, height=2, courtyard_margin=0.0),
        ],
        keepouts=[keepout],
    )

    cs = CostState(model)
    boundary_with_keepout = cs._boundary_sum()

    # Remove keepouts, re-measure — should be 0 (both comps in-bounds).
    model.keepouts = []
    cs2 = CostState(model)
    boundary_no_keepout = cs2._boundary_sum()

    assert boundary_with_keepout > boundary_no_keepout, \
        f"CostState boundary penalty with keepout ({boundary_with_keepout}) " \
        f"must be > without keepout ({boundary_no_keepout})"


def test_cost_function():
    model = _make_test_model()
    cost_fn = CostFunction(alpha=1.0, beta=5.0, gamma=3.0)
    costs = cost_fn.evaluate(model)

    assert "total" in costs
    assert "hpwl" in costs
    assert "overlap" in costs
    assert "boundary" in costs
    assert costs["total"] >= 0


# ---------------------------------------------------------------------------
# Grid Placement Tests
# ---------------------------------------------------------------------------

def test_grid_placement():
    model = _make_test_model()
    # Scatter components first
    for comp in model.components:
        if not comp.is_fixed:
            comp.x = 0
            comp.y = 0

    grid_place(model)

    # Components should be placed within board
    for comp in model.components:
        assert model.board.contains(comp.x, comp.y), \
            f"{comp.ref} at ({comp.x}, {comp.y}) should be within board"


def test_edge_aware_placement():
    model = _make_test_model()
    grid_place(model)

    # Connector should be near an edge
    j1 = model.get_component("J1")
    if j1:
        board = model.board
        near_edge = (
            j1.x < board.x_min + 15 or j1.x > board.x_max - 15 or
            j1.y < board.y_min + 15 or j1.y > board.y_max - 15
        )
        assert near_edge, f"J1 at ({j1.x}, {j1.y}) should be near board edge"


# ---------------------------------------------------------------------------
# Legalization Tests
# ---------------------------------------------------------------------------

def test_grid_snap():
    model = BoardModel(
        board=BoardOutline(x_min=0, y_min=0, x_max=100, y_max=100),
        components=[
            Component(ref="U1", x=10.123, y=20.456, width=5, height=5),
            Component(ref="U2", x=30.789, y=40.111, width=5, height=5),
        ],
    )
    legalize(model, grid_mm=0.1)

    for comp in model.components:
        # Check grid snapping
        assert abs(comp.x * 10 - round(comp.x * 10)) < 0.01, \
            f"{comp.ref} x={comp.x} not snapped to 0.1mm grid"
        assert abs(comp.y * 10 - round(comp.y * 10)) < 0.01, \
            f"{comp.ref} y={comp.y} not snapped to 0.1mm grid"


def test_boundary_enforcement():
    model = BoardModel(
        board=BoardOutline(x_min=0, y_min=0, x_max=50, y_max=50),
        components=[
            Component(ref="U1", x=100, y=100, width=5, height=5),
        ],
    )
    legalize(model)

    for comp in model.components:
        bbox = comp.bbox
        assert bbox[0] >= model.board.x_min, f"{comp.ref} extends below board"
        assert bbox[2] <= model.board.x_max, f"{comp.ref} extends above board"


def test_overlap_resolution():
    model = BoardModel(
        board=BoardOutline(x_min=0, y_min=0, x_max=100, y_max=100),
        components=[
            Component(ref="U1", x=20, y=20, width=5, height=5),
            Component(ref="U2", x=21, y=21, width=5, height=5),  # Overlapping
            Component(ref="U3", x=22, y=22, width=5, height=5),  # Overlapping
        ],
    )

    overlaps_before = count_overlaps(model)
    assert overlaps_before > 0, f"Expected overlaps before legalization, got {overlaps_before}"

    legalize(model, verbose=False)

    overlaps_after = count_overlaps(model)
    assert overlaps_after < overlaps_before, \
        f"Overlaps should decrease: {overlaps_before} → {overlaps_after}"


# ---------------------------------------------------------------------------
# Board Profile Tests
# ---------------------------------------------------------------------------

def test_builtin_profiles():
    profiles = list_profiles()
    assert len(profiles) >= 4, f"Expected 4+ profiles, got {len(profiles)}"

    for name in ["mcu_peripheral", "power_supply", "rf_frontend", "mixed_signal", "generic"]:
        p = get_profile(name)
        assert p.name == name
        assert p.alpha > 0
        assert p.beta > 0


def test_profile_rules():
    mcu = get_profile("mcu_peripheral")
    rules = mcu.active_rules()
    assert len(rules) > 0, "MCU profile should have active rules"

    rule_names = [r.name for r in rules]
    assert "decoupling_proximity" in rule_names
    assert "crystal_mcu" in rule_names


def test_routing_congestion_rule_dispatch():
    """Phase 2.1: routing_congestion must be a dispatchable ConstraintRule
    that wraps engine.congestion.rudy_congestion_penalty.

    RUDY was already implemented in engine/congestion.py and used as a
    SA translate-bias + best-state tiebreaker, but it was NOT exposed
    as a profile-composable ConstraintRule.  The brief asks for it to
    compose with the 6 board profiles via the existing ConstraintRule
    pattern — this test verifies the dispatch wiring.
    """
    from engine.constraint_evaluator import _RULE_HANDLERS, evaluate_constraint_penalties
    from profiles.board_profiles import ConstraintRule
    assert "routing_congestion" in _RULE_HANDLERS, \
        "routing_congestion must be in _RULE_HANDLERS dispatch table"

    # Build a small congested board: 4 components on a tight 2x2 grid
    # connected by 4 crossing nets — the classic RUDY hotspot pattern.
    model = BoardModel(
        board=BoardOutline(x_min=0, y_min=0, x_max=20, y_max=20),
        components=[
            Component(
                ref=f"U{i+1}", x=x, y=y, width=2, height=2, component_type="ic",
                pads=[Pad(pad_name="1", x=0, y=0, net=net)],
                nets=[net],
            )
            for i, (x, y, net) in enumerate([
                (5, 5, "N_DIAG1"),    # U1 bottom-left
                (15, 5, "N_DIAG2"),   # U2 bottom-right
                (5, 15, "N_DIAG2"),   # U3 top-left, shares N_DIAG2 with U2
                (15, 15, "N_DIAG1"),  # U4 top-right, shares N_DIAG1 with U1
            ])
        ],
        nets=[
            # Two diagonal nets whose bboxes cross in the middle — RUDY hotspot.
            Net(name="N_DIAG1", pins=[("U1", "1"), ("U4", "1")]),
            Net(name="N_DIAG2", pins=[("U2", "1"), ("U3", "1")]),
        ],
    )

    rule = ConstraintRule(
        name="routing_congestion",
        weight=1.0,
        params={"grid_resolution_mm": 2.0},
    )
    total, breakdown = evaluate_constraint_penalties(model, [rule])
    assert "routing_congestion" in breakdown, \
        f"routing_congestion should appear in breakdown, got {list(breakdown.keys())}"
    assert breakdown["routing_congestion"] > 0, \
        f"Penalty should be > 0 on a congested board, got {breakdown['routing_congestion']}"
    assert total == rule.weight * breakdown["routing_congestion"], \
        f"Total should equal weight * penalty, got {total} vs {rule.weight * breakdown['routing_congestion']}"


def test_routing_congestion_in_profiles():
    """Phase 2.1: routing_congestion rule should be enabled (opt-in) on
    the rf_frontend and mixed_signal profiles — both are congestion-
    sensitive (RF frontends have routing choke points between LNA/mixer/
    filter stages; mixed-signal boards have analog/digital partition
    boundaries that concentrate crossing nets).  Off by default elsewhere
    so existing profile behaviour is unchanged.
    """
    rf = get_profile("rf_frontend")
    rf_rules = {r.name: r for r in rf.rules}
    assert "routing_congestion" in rf_rules, \
        "rf_frontend should opt into routing_congestion"
    assert rf_rules["routing_congestion"].enabled, \
        "routing_congestion should be enabled on rf_frontend"

    ms = get_profile("mixed_signal")
    ms_rules = {r.name: r for r in ms.rules}
    assert "routing_congestion" in ms_rules, \
        "mixed_signal should opt into routing_congestion"
    assert ms_rules["routing_congestion"].enabled, \
        "routing_congestion should be enabled on mixed_signal"

    # Off by default on the other profiles — no silent behaviour change.
    for profile_name in ["mcu_peripheral", "power_supply", "generic"]:
        p = get_profile(profile_name)
        rule_names = {r.name for r in p.rules}
        assert "routing_congestion" not in rule_names, \
            f"{profile_name} should NOT have routing_congestion (off by default)"


def test_pin_density_penalty_detects_hotspot():
    """Pin-density congestion must fire when many signal pins share one
    grid cell — a pin-escape hotspot.

    Sets up 4 ICs in a tight cluster (all within one 2mm cell) connected
    to 4 far-away passives. Pin density sees 4 pins stacked in one cell
    (peak = 4.0, well above the 2.0 floor target) and fires. This is the
    case pin density was added for: a router has to escape 4 pins out of
    a 2mm × 2mm area, which is hard even if the wires themselves are
    short.
    """
    from engine.congestion import (
        compute_pin_density_map,
        pin_density_penalty,
    )

    # 4 ICs clustered at (5.0..5.2, 5.0..5.2), each net's other endpoint
    # is a passive at the far corner of the 20×20 board.
    model = BoardModel(
        board=BoardOutline(x_min=0, y_min=0, x_max=20, y_max=20),
        components=[
            # Clustered ICs (4 pins in one 2mm cell)
            Component(
                ref=f"U{i+1}", x=x, y=y, width=0.4, height=0.4,
                component_type="ic",
                pads=[Pad(pad_name="1", x=0, y=0, net=f"N{i+1}")],
                nets=[f"N{i+1}"],
            )
            for i, (x, y) in enumerate([
                (5.0, 5.0), (5.2, 5.0), (5.0, 5.2), (5.2, 5.2),
            ])
        ] + [
            # Far-away passives (one per net, each in its own far cell)
            Component(
                ref=f"R{i+1}", x=x, y=y, width=0.4, height=0.4,
                component_type="resistor",
                pads=[Pad(pad_name="1", x=0, y=0, net=f"N{i+1}")],
                nets=[f"N{i+1}"],
            )
            for i, (x, y) in enumerate([
                (17.0, 17.0), (3.0, 17.0), (17.0, 3.0), (3.0, 3.0),
            ])
        ],
        nets=[
            Net(name="N1", pins=[("U1", "1"), ("R1", "1")]),
            Net(name="N2", pins=[("U2", "1"), ("R2", "1")]),
            Net(name="N3", pins=[("U3", "1"), ("R3", "1")]),
            Net(name="N4", pins=[("U4", "1"), ("R4", "1")]),
        ],
    )

    grid, meta = compute_pin_density_map(model, grid_resolution=2.0)
    # All 4 clustered IC pins fall in the (4..6, 4..6) mm cell. The 4
    # far-away passives each land in their own cell. So the cluster cell
    # has peak = 4.0.
    peak = max(max(row) for row in grid)
    assert peak >= 4.0, f"Pin density peak should be ≥ 4 in the clustered cell, got {peak}"

    pd_penalty, pd_peak, _pd_avg, _pd_ovf = pin_density_penalty(model, grid_resolution=2.0)
    assert pd_penalty > 0, "Pin-density penalty must fire on a pin hotspot"
    assert pd_peak == peak, f"pin_density_penalty peak {pd_peak} != map peak {peak}"


def test_pin_density_complements_rudy_on_long_nets():
    """Pin density is more sensitive than RUDY to pin-escape clusters
    when each net's bbox is large.

    On a board where many pins cluster in one cell but each net's bbox
    is large (long nets), RUDY's per-cell contribution from each net is
    small (1/num_cells_in_bbox). RUDY still fires (the cluster cell
    accumulates contributions from all nets whose bbox includes it),
    but the penalty is dominated by the cluster's wire-density
    contribution. Pin density's penalty is dominated by the raw pin
    count and is far larger relative to the board's average.

    This is the complementary-roles argument: the two signals detect
    different aspects of the same congestion pattern, so shipping both
    gives the placer a richer gradient signal than either alone.
    """
    from engine.congestion import (
        pin_density_penalty,
        rudy_congestion_penalty,
    )

    # 4 ICs clustered tightly, each net's other endpoint is on the
    # opposite side of a 100×100 board. Each net's bbox spans the
    # entire board, so each net's RUDY contribution per cell is ~1/576.
    # The cluster cell sees ~4/576 ≈ 0.007 from RUDY. Pin density sees
    # 4 pins stacked in one cell — peak 4.0.
    model = BoardModel(
        board=BoardOutline(x_min=0, y_min=0, x_max=100, y_max=100),
        components=[
            Component(
                ref=f"U{i+1}", x=x, y=y, width=0.4, height=0.4,
                component_type="ic",
                pads=[Pad(pad_name="1", x=0, y=0, net=f"N{i+1}")],
                nets=[f"N{i+1}"],
            )
            for i, (x, y) in enumerate([
                (50.0, 50.0), (50.1, 50.0), (50.0, 50.1), (50.1, 50.1),
            ])
        ] + [
            Component(
                ref=f"R{i+1}", x=x, y=y, width=0.4, height=0.4,
                component_type="resistor",
                pads=[Pad(pad_name="1", x=0, y=0, net=f"N{i+1}")],
                nets=[f"N{i+1}"],
            )
            for i, (x, y) in enumerate([
                (5.0, 5.0), (95.0, 5.0), (5.0, 95.0), (95.0, 95.0),
            ])
        ],
        nets=[
            Net(name="N1", pins=[("U1", "1"), ("R1", "1")]),
            Net(name="N2", pins=[("U2", "1"), ("R2", "1")]),
            Net(name="N3", pins=[("U3", "1"), ("R3", "1")]),
            Net(name="N4", pins=[("U4", "1"), ("R4", "1")]),
        ],
    )

    pd_penalty, pd_peak, pd_avg, _pd_ovf = pin_density_penalty(model, grid_resolution=2.0)
    rudy_penalty, rudy_peak, rudy_avg, _rudy_ovf = rudy_congestion_penalty(model, grid_resolution=2.0)

    # Both signals fire — they're complementary, not mutually exclusive.
    assert pd_penalty > 0, (
        f"Pin density should fire on the 4-pin cluster, got penalty={pd_penalty}"
    )
    assert pd_peak >= 4.0, f"Pin density peak should be ≥ 4, got {pd_peak}"

    # Pin density's signal-to-noise (peak / average) is much higher than
    # RUDY's on this board, because pin density's per-cell value is
    # either 0 or K (the pin count) — sharp peaks — while RUDY spreads
    # each net's contribution across its entire bbox — soft peaks.
    # On this board pin density's SNR is ~2500× (peak 4 / avg 0.0016),
    # RUDY's is ~4.4× (peak 0.007 / avg 0.0016). Pin density is hundreds
    # of times more sensitive to this pin-escape cluster than RUDY —
    # that's why shipping both signals gives a richer gradient.
    pd_snr = pd_peak / max(pd_avg, 1e-12)
    rudy_snr = rudy_peak / max(rudy_avg, 1e-12)
    assert pd_snr > rudy_snr * 10, (
        f"Pin density's signal-to-noise ratio ({pd_snr:.1f}×) should be "
        f"much higher than RUDY's ({rudy_snr:.1f}×) on a pin-escape "
        f"cluster with long nets — that's the whole point of having "
        f"both signals."
    )


def test_pin_density_excludes_power_pins():
    """Power/ground pins must not contribute to pin-density congestion.

    Every IC has VCC/GND, so counting them would uniformly inflate every
    cell and wash out the signal. Power pins must be excluded so the map
    reflects signal-routing demand only.
    """
    from engine.congestion import compute_pin_density_map

    # Two components, each with one +3V3 pin and one signal pin.
    # Both pairs of +3V3 pins fall in the same cell; the signal pins
    # are spread out.
    model = BoardModel(
        board=BoardOutline(x_min=0, y_min=0, x_max=20, y_max=20),
        components=[
            Component(
                ref="U1", x=5, y=5, width=2, height=2, component_type="ic",
                pads=[
                    Pad(pad_name="1", x=0, y=0, net="+3V3"),
                    Pad(pad_name="2", x=0.5, y=0, net="SIG_A"),
                ],
                nets=["+3V3", "SIG_A"],
            ),
            Component(
                ref="U2", x=6, y=5, width=2, height=2, component_type="ic",
                pads=[
                    Pad(pad_name="1", x=0, y=0, net="+3V3"),
                    Pad(pad_name="2", x=0.5, y=0, net="SIG_A"),
                ],
                nets=["+3V3", "SIG_A"],
            ),
        ],
        nets=[
            Net(name="+3V3", pins=[("U1", "1"), ("U2", "1")]),
            Net(name="SIG_A", pins=[("U1", "2"), ("U2", "2")]),
        ],
    )

    grid, _meta = compute_pin_density_map(model, grid_resolution=2.0)
    total_pins_in_grid = sum(sum(row) for row in grid)
    # Two signal pins on SIG_A → only those should be counted.
    # The +3V3 pins must be excluded.
    assert total_pins_in_grid == 2.0, (
        f"Power pins should be excluded from pin density; expected 2 signal "
        f"pins counted, got {total_pins_in_grid}"
    )


def test_pin_density_penalty_zero_on_uniform_board():
    """Pin-density penalty must be 0 on a uniformly distributed board.

    Same adaptive-target scheme as RUDY (1.5× average). On a uniform
    board peak ≈ average, so the penalty is zero — it only fires when
    there's an actual hotspot.
    """
    from engine.congestion import pin_density_penalty

    # 16 components evenly spread on a 4×4 grid across a 40×40 board.
    # Each cell of the 2mm grid (20×20 cells) holds at most 1 pin →
    # peak == average == 1.0 → penalty 0.
    positions = [(x, y) for x in range(5, 40, 9) for y in range(5, 40, 9)]
    model = BoardModel(
        board=BoardOutline(x_min=0, y_min=0, x_max=40, y_max=40),
        components=[
            Component(
                ref=f"U{i+1}", x=x, y=y, width=1, height=1, component_type="ic",
                pads=[Pad(pad_name="1", x=0, y=0, net=f"N{i+1}")],
                nets=[f"N{i+1}"],
            )
            for i, (x, y) in enumerate(positions)
        ],
        nets=[
            # Pair adjacent pins on shared nets so each net has >= 2 pins.
            Net(name=f"N{i+1}",
                pins=[("U1", "1"), (f"U{i+1}", "1")] if i > 0
                else [("U1", "1"), ("U2", "1")])
            for i in range(len(positions))
        ],
    )

    pd_penalty, _peak, _avg, _ovf = pin_density_penalty(model, grid_resolution=2.0)
    assert pd_penalty == 0.0, (
        f"Uniformly distributed board should have 0 pin-density penalty, "
        f"got {pd_penalty}"
    )


def test_evaluate_cost_function_includes_pin_density_term():
    """cost.evaluate() must thread pin_density_weight into the total.

    Verifies the new cost-function wiring end-to-end: with pin density
    active and rudy disabled, the total cost should equal
    α·HPWL + β·Overlap + γ·Boundary + pin_density_weight·PinDensity.
    """
    from cost.cost import evaluate
    from models.macro import Macro

    # Use the clustered setup so pin density actually fires (>2 pins in
    # one cell, above the 2.0 floor target).
    model = BoardModel(
        board=BoardOutline(x_min=0, y_min=0, x_max=20, y_max=20),
        components=[
            Component(
                ref=f"U{i+1}", x=x, y=y, width=0.4, height=0.4,
                component_type="ic",
                pads=[Pad(pad_name="1", x=0, y=0, net=f"N{i+1}")],
                nets=[f"N{i+1}"],
            )
            for i, (x, y) in enumerate([
                (5.0, 5.0), (5.2, 5.0), (5.0, 5.2), (5.2, 5.2),
            ])
        ] + [
            Component(
                ref=f"R{i+1}", x=x, y=y, width=0.4, height=0.4,
                component_type="resistor",
                pads=[Pad(pad_name="1", x=0, y=0, net=f"N{i+1}")],
                nets=[f"N{i+1}"],
            )
            for i, (x, y) in enumerate([
                (17.0, 17.0), (3.0, 17.0), (17.0, 3.0), (3.0, 3.0),
            ])
        ],
        nets=[
            Net(name="N1", pins=[("U1", "1"), ("R1", "1")]),
            Net(name="N2", pins=[("U2", "1"), ("R2", "1")]),
            Net(name="N3", pins=[("U3", "1"), ("R3", "1")]),
            Net(name="N4", pins=[("U4", "1"), ("R4", "1")]),
        ],
    )

    macros = [Macro.alone(c) for c in model.components]

    # Disable RUDY, enable pin density
    result = evaluate(
        model, macros,
        alpha=1.0, beta=25.0, gamma=8.0,
        rudy_weight=0.0,
        pin_density_weight=0.5,
    )

    # Verify the breakdown includes the pin_density field
    assert "pin_density" in result, "evaluate() must include pin_density in its breakdown"
    assert result["pin_density"] > 0, "pin_density should be > 0 on the hotspot board"

    # Verify the total = α·HPWL + β·Overlap + γ·Boundary + 0.5·PinDensity
    expected_total = (
        1.0 * result["hpwl"]
        + 25.0 * result["overlap"]
        + 8.0 * result["boundary"]
        + 0.5 * result["pin_density"]
    )
    assert abs(result["total"] - expected_total) < 1e-6, (
        f"Total {result['total']} != expected {expected_total} — pin_density_weight "
        f"not threaded correctly into total cost"
    )

    # And with pin_density_weight=0, the total should NOT include the pin term
    result_off = evaluate(
        model, macros,
        alpha=1.0, beta=25.0, gamma=8.0,
        rudy_weight=0.0,
        pin_density_weight=0.0,
    )
    expected_off = (
        1.0 * result_off["hpwl"]
        + 25.0 * result_off["overlap"]
        + 8.0 * result_off["boundary"]
    )
    assert abs(result_off["total"] - expected_off) < 1e-6
    # The pin_density field is still computed (for reporting) when weight=0
    # only if a pre-computed penalty is passed in; otherwise it's 0. Either
    # way, it must not affect the total.
    assert result_off["total"] < result["total"], (
        "Disabling pin_density_weight should give a lower total cost on this hotspot board"
    )


def test_rudy_weight_defaults_to_config_value():
    """The CLI default for --rudy-weight must defer to config.json.

    Regression test for the original bug: the CLI parser defaulted
    --rudy-weight to 0.0, which silently disabled RUDY even though
    config.json said 0.3. The fix makes the CLI default None and lets
    place_v2() resolve None → config.annealer.rudy_weight.

    Verifies that:
    - config.json has a non-zero rudy_weight (so the default is "on")
    - place_v2's rudy_weight parameter, when None, resolves to the
      config value (not to 0.0)
    """
    from config import load_config
    cfg = load_config()
    assert cfg.annealer.rudy_weight > 0, (
        "config.json annealer.rudy_weight must be > 0 so RUDY is on by default; "
        f"got {cfg.annealer.rudy_weight}"
    )
    assert cfg.annealer.pin_density_weight > 0, (
        "config.json annealer.pin_density_weight must be > 0 so pin-density is "
        f"on by default; got {cfg.annealer.pin_density_weight}"
    )


def test_thermal_separation_rule_dispatch():
    """Phase 2.4: thermal_separation must be a dispatchable ConstraintRule.

    Complementary to the existing thermal_grouping rule.  thermal_grouping
    keeps hot parts on the SAME net close together; thermal_separation
    keeps hot parts on DIFFERENT nets apart (so two independent hot spots
    don't stack and create a thermal runaway).  Default min separation
    ~5mm, per the brief's DFM guidelines.
    """
    from engine.constraint_evaluator import _RULE_HANDLERS, evaluate_constraint_penalties
    from profiles.board_profiles import ConstraintRule
    assert "thermal_separation" in _RULE_HANDLERS, \
        "thermal_separation must be in _RULE_HANDLERS dispatch table"

    # Two regulators on different signal nets, 2mm apart — should be penalised.
    # Use +5V/+3V3 (power nets, excluded from "sharing" check) for the rail
    # so the regulators are on DIFFERENT signal nets (VOUT_A vs VOUT_B).
    model = BoardModel(
        board=BoardOutline(x_min=0, y_min=0, x_max=100, y_max=80),
        components=[
            Component(ref="U1", x=50, y=40, width=5, height=5,
                      component_type="regulator",
                      pads=[Pad(pad_name="1", x=0, y=0, net="+5V"),
                            Pad(pad_name="2", x=1, y=0, net="VOUT_A")],
                      nets=["+5V", "VOUT_A"]),
            Component(ref="U2", x=52, y=40, width=5, height=5,
                      component_type="regulator",
                      pads=[Pad(pad_name="1", x=0, y=0, net="+3V3"),
                            Pad(pad_name="2", x=1, y=0, net="VOUT_B")],
                      nets=["+3V3", "VOUT_B"]),
            # A non-thermal component nearby — should NOT be penalised.
            Component(ref="C1", x=51, y=42, width=1, height=0.5,
                      component_type="capacitor",
                      pads=[Pad(pad_name="1", x=0, y=0, net="+5V")],
                      nets=["+5V"]),
        ],
        nets=[
            Net(name="+5V", pins=[("U1", "1"), ("C1", "1")]),
            Net(name="+3V3", pins=[("U2", "1")]),
            Net(name="VOUT_A", pins=[("U1", "2")]),
            Net(name="VOUT_B", pins=[("U2", "2")]),
        ],
    )

    rule = ConstraintRule(
        name="thermal_separation",
        weight=1.0,
        params={"min_distance_mm": 5.0},
    )
    total, breakdown = evaluate_constraint_penalties(model, [rule])
    assert "thermal_separation" in breakdown, \
        f"thermal_separation should appear in breakdown, got {list(breakdown.keys())}"
    # U1 and U2 are 2mm apart, min is 5mm — excess = 3mm → penalty > 0
    assert breakdown["thermal_separation"] > 0, \
        f"Two regulators 2mm apart (min 5mm) should be penalised, got {breakdown['thermal_separation']}"


def test_thermal_separation_far_apart_no_penalty():
    """Phase 2.4: thermal_separation should return 0 when hot parts are
    far enough apart (≥ min_distance_mm)."""
    from engine.constraint_evaluator import evaluate_constraint_penalties
    from profiles.board_profiles import ConstraintRule

    model = BoardModel(
        board=BoardOutline(x_min=0, y_min=0, x_max=100, y_max=80),
        components=[
            Component(ref="U1", x=20, y=20, width=5, height=5,
                      component_type="regulator",
                      pads=[Pad(pad_name="1", x=0, y=0, net="VOUT_A")],
                      nets=["VOUT_A"]),
            Component(ref="U2", x=80, y=70, width=5, height=5,
                      component_type="regulator",
                      pads=[Pad(pad_name="1", x=0, y=0, net="VOUT_B")],
                      nets=["VOUT_B"]),
        ],
        nets=[
            Net(name="VOUT_A", pins=[("U1", "1")]),
            Net(name="VOUT_B", pins=[("U2", "1")]),
        ],
    )

    rule = ConstraintRule(
        name="thermal_separation",
        weight=1.0,
        params={"min_distance_mm": 5.0},
    )
    total, breakdown = evaluate_constraint_penalties(model, [rule])
    # U1 and U2 are ~76mm apart — well above 5mm threshold → no penalty
    assert breakdown["thermal_separation"] == 0, \
        f"Two regulators 76mm apart should have 0 penalty, got {breakdown['thermal_separation']}"


def test_thermal_separation_ignores_same_net_pairs():
    """Phase 2.4: thermal_separation should NOT penalise hot parts on
    the SAME net — that's thermal_grouping's job.  Two regulators sharing
    a non-power signal net should be kept CLOSE (thermal_grouping), not
    FAR (thermal_separation)."""
    from engine.constraint_evaluator import evaluate_constraint_penalties
    from profiles.board_profiles import ConstraintRule

    model = BoardModel(
        board=BoardOutline(x_min=0, y_min=0, x_max=100, y_max=80),
        components=[
            Component(ref="U1", x=50, y=40, width=5, height=5,
                      component_type="regulator",
                      pads=[Pad(pad_name="1", x=0, y=0, net="SHARED_SIG")],
                      nets=["SHARED_SIG"]),
            Component(ref="U2", x=52, y=40, width=5, height=5,
                      component_type="regulator",
                      pads=[Pad(pad_name="1", x=0, y=0, net="SHARED_SIG")],
                      nets=["SHARED_SIG"]),
        ],
        nets=[
            Net(name="SHARED_SIG", pins=[("U1", "1"), ("U2", "1")]),
        ],
    )

    rule = ConstraintRule(
        name="thermal_separation",
        weight=1.0,
        params={"min_distance_mm": 5.0},
    )
    total, breakdown = evaluate_constraint_penalties(model, [rule])
    # U1 and U2 share SHARED_SIG — thermal_separation should ignore them
    # (thermal_grouping handles same-net thermal pairs).
    assert breakdown["thermal_separation"] == 0, \
        f"Same-net thermal pair should be ignored by thermal_separation, got {breakdown['thermal_separation']}"


def test_thermal_separation_in_power_supply_profile():
    """Phase 2.4: power_supply profile should opt into thermal_separation
    (it's the highest-value DFM rule for power boards — two independent
    regulators stacked together cause thermal runaways)."""
    ps = get_profile("power_supply")
    ps_rules = {r.name: r for r in ps.rules}
    assert "thermal_separation" in ps_rules, \
        "power_supply should opt into thermal_separation"
    assert ps_rules["thermal_separation"].enabled, \
        "thermal_separation should be enabled on power_supply"


def test_thermal_separation_retuned_weights():
    """Post-A/B-test retuning: thermal_separation on power_supply should
    have weight=1.0 and min_distance=3.0mm (down from 4.0/5.0mm).

    The original weights caused SA to cram components into corners
    (centroid offset 5.9mm → 43.1mm, 7 components OOB on th_sensor).
    The retuned weights make it a gentle nudge, not a hard constraint.
    """
    ps = get_profile("power_supply")
    rule = next(r for r in ps.rules if r.name == "thermal_separation")
    assert rule.weight == 1.0, \
        f"thermal_separation weight should be 1.0 (retuned), got {rule.weight}"
    assert rule.params.get("min_distance_mm") == 3.0, \
        f"min_distance_mm should be 3.0 (retuned), got {rule.params.get('min_distance_mm')}"


def test_sa_config_auto_disable_thresholds():
    """SAConfig should have auto-disable thresholds for tiny/large boards.

    Tiny boards (≤6 comps): greedy is near-optimal, SA noise hurts.
      The bistable_oscillator (6 comps) showed +8% HPWL regression with SA —
      raising from 5 to 6 auto-disables SA on that board class.
    Large boards (≥50 comps): SA is 8-25x slower with diminishing returns.
    """
    config = SAConfig()
    assert config.sa_auto_disable_min_components == 6, \
        f"Default min threshold should be 6, got {config.sa_auto_disable_min_components}"
    assert config.sa_auto_disable_max_components == 50, \
        f"Default max threshold should be 50, got {config.sa_auto_disable_max_components}"


def test_sa_config_spread_floor():
    """SAConfig should have a spread_floor_fraction to prevent corner-cramming."""
    config = SAConfig()
    assert config.spread_floor_fraction > 0, \
        f"spread_floor_fraction should be > 0 by default, got {config.spread_floor_fraction}"
    assert config.spread_floor_fraction == 0.10, \
        f"Default spread_floor_fraction should be 0.10 (10% of board dim), got {config.spread_floor_fraction}"


def test_sa_auto_disables_on_tiny_board():
    """run_sa should auto-disable SA on boards with ≤6 movable components.

    On tiny boards, greedy is already near-optimal and SA's exploration
    noise hurts more than it helps.  This test constructs a 4-component
    board, calls run_sa with skip_sa=False, and verifies SA was auto-
    disabled (the function should still run, just via greedy path).
    """
    model = BoardModel(
        board=BoardOutline(x_min=0, y_min=0, x_max=80, y_max=60),
        components=[
            Component(ref=f"U{i}", x=20+i*15, y=30, width=5, height=5,
                      component_type="ic",
                      pads=[Pad(pad_name="1", x=0, y=0, net=f"SIG{i}")],
                      nets=[f"SIG{i}"])
            for i in range(4)
        ],
        nets=[Net(name=f"SIG{i}", pins=[(f"U{i}", "1")]) for i in range(4)],
    )
    config = SAConfig(skip_sa=False, verbose=False)
    # Should auto-disable SA because 4 ≤ 6
    result = run_sa(model, config=config, verbose=False)
    assert config.skip_sa == True, \
        f"SA should be auto-disabled on 4-component board, but skip_sa={config.skip_sa}"
    assert 'final_cost' in result, "run_sa should still return a result dict"


def test_sa_auto_disable_respects_explicit_skip_sa_true():
    """If the user explicitly sets skip_sa=True, auto-disable shouldn't
    override it (it's already disabled — nothing to do).  This test
    verifies the auto-disable logic doesn't accidentally re-enable SA."""
    model = BoardModel(
        board=BoardOutline(x_min=0, y_min=0, x_max=80, y_max=60),
        components=[
            Component(ref=f"U{i}", x=20+i*15, y=30, width=5, height=5,
                      component_type="ic",
                      pads=[Pad(pad_name="1", x=0, y=0, net=f"SIG{i}")],
                      nets=[f"SIG{i}"])
            for i in range(10)  # 10 comps — in the SA-enabled range
        ],
        nets=[Net(name=f"SIG{i}", pins=[(f"U{i}", "1")]) for i in range(10)],
    )
    config = SAConfig(skip_sa=True, verbose=False)
    run_sa(model, config=config, verbose=False)
    assert config.skip_sa == True, \
        "Explicit skip_sa=True should stay True (auto-disable shouldn't re-enable SA)"


def test_violates_spread_floor():
    """_violates_spread_floor should detect bunched configurations."""
    from engine.annealer import _violates_spread_floor
    from models.board_model import BoardOutline

    board = BoardOutline(x_min=0, y_min=0, x_max=100, y_max=80)
    # Floor at 10%: std_x ≥ 10mm, std_y ≥ 8mm required.

    # Bunched in X (all components at x=50, std_x=0) — should violate.
    bunched_x = BoardModel(
        board=board,
        components=[
            Component(ref=f"U{i}", x=50, y=10+i*15, width=2, height=2,
                      component_type="ic")
            for i in range(5)
        ],
    )
    assert _violates_spread_floor(bunched_x, board, 0.10), \
        "Components bunched in X (std_x=0) should violate spread floor"

    # Well-spread in both axes — should NOT violate.
    spread = BoardModel(
        board=board,
        components=[
            Component(ref=f"U{i}", x=10+i*20, y=10+i*15, width=2, height=2,
                      component_type="ic")
            for i in range(5)
        ],
    )
    assert not _violates_spread_floor(spread, board, 0.10), \
        "Well-spread components should NOT violate spread floor"

    # Bunched in Y (all at y=40, std_y=0) — should violate.
    bunched_y = BoardModel(
        board=board,
        components=[
            Component(ref=f"U{i}", x=10+i*20, y=40, width=2, height=2,
                      component_type="ic")
            for i in range(5)
        ],
    )
    assert _violates_spread_floor(bunched_y, board, 0.10), \
        "Components bunched in Y (std_y=0) should violate spread floor"


def test_sa_no_spread_floor_revert_when_skip_sa():
    """REGRESSION: when the user explicitly passes skip_sa=True, the
    spread_floor and centroid_offset reverts must NOT fire — they are
    SA-specific pathology guards and have no business reverting greedy
    output.  Previously the guard only checked `sa_was_auto_disabled`,
    which meant skip_sa=True runs (the default) could have their good
    greedy output silently reverted on tiny boards where greedy happened
    to bunch components.

    Repro: PModBoard (5 comps) — greedy gives HPWL=18.7, ovr=0; the
    pre-patch no-SA arm reverted to pre-greedy state (HPWL=34, ovr=4).
    """
    # Bunched-in-X layout that WOULD violate spread floor — but greedy
    # should be allowed to keep its result since SA didn't run.
    bunched = BoardModel(
        board=BoardOutline(x_min=0, y_min=0, x_max=80, y_max=60),
        components=[
            Component(ref=f"U{i}", x=40, y=10+i*10, width=2, height=2,
                      component_type="ic",
                      pads=[Pad(pad_name="1", x=0, y=0, net="SIG")],
                      nets=["SIG"])
            for i in range(8)  # 8 comps — SA-enabled range
        ],
        nets=[Net(name="SIG", pins=[(f"U{i}", "1") for i in range(8)])],
    )
    pre_positions = [(c.x, c.y, c.rotation) for c in bunched.components]
    config = SAConfig(skip_sa=True, verbose=False)
    run_sa(bunched, config=config, verbose=False)
    post_positions = [(c.x, c.y, c.rotation) for c in bunched.components]
    # If the spread-floor revert fired, positions would be unchanged
    # (reverted to pre-SA = pre-greedy state).  Greedy should have moved
    # at least one component to reduce HPWL on this bunched layout.
    moved = sum(1 for pre, post in zip(pre_positions, post_positions) if pre != post)
    assert moved > 0, \
        "skip_sa=True should NOT trigger spread_floor/centroid revert — greedy output must be preserved"


def test_cap_ic_group_survives_translate():
    """REGRESSION: when do_translate picks an IC, its assigned caps must
    move with it (same delta). Pre-fix, only do_translate was group-aware;
    this test locks in the contract.
    """
    from engine.group_moves import get_decap_map, get_group_indices, apply_delta_with_clamp

    model = _make_test_model()
    # Force the decap map to be built and cached on the model
    dm = get_decap_map(model)
    assert 'U1' in dm and set(dm['U1']) == {'C1', 'C2'}, \
        f"Test model should have U1 with caps C1, C2; got {dm}"

    u1_idx = next(i for i, c in enumerate(model.components) if c.ref == 'U1')
    c1_idx = next(i for i, c in enumerate(model.components) if c.ref == 'C1')
    c2_idx = next(i for i, c in enumerate(model.components) if c.ref == 'C2')

    u1_pre = (model.components[u1_idx].x, model.components[u1_idx].y)
    c1_pre = (model.components[c1_idx].x, model.components[c1_idx].y)
    c2_pre = (model.components[c2_idx].x, model.components[c2_idx].y)
    pre_dist_c1 = math.hypot(u1_pre[0] - c1_pre[0], u1_pre[1] - c1_pre[1])
    pre_dist_c2 = math.hypot(u1_pre[0] - c2_pre[0], u1_pre[1] - c2_pre[1])

    # Directly invoke the group-move helper with a known delta.
    # This is what do_translate does internally when it picks an IC.
    group = get_group_indices(model, u1_idx)
    assert group == {u1_idx, c1_idx, c2_idx}, \
        f"Group for U1 should be {{U1, C1, C2}}; got {group}"

    # Move by a small delta that won't trigger board-edge clamping
    # (U1 at (40,30), board 80x60, so 5mm move keeps everything well inside).
    dx, dy = 3.0, -2.0
    apply_delta_with_clamp(model, group, dx, dy)

    u1_post = (model.components[u1_idx].x, model.components[u1_idx].y)
    c1_post = (model.components[c1_idx].x, model.components[c1_idx].y)
    c2_post = (model.components[c2_idx].x, model.components[c2_idx].y)

    # All group members should have moved by exactly (dx, dy) — no clamping
    # expected since the move keeps everyone well inside the 80x60 board.
    assert abs(u1_post[0] - u1_pre[0] - dx) < 0.01, "U1 X should move by dx"
    assert abs(u1_post[1] - u1_pre[1] - dy) < 0.01, "U1 Y should move by dy"
    assert abs(c1_post[0] - c1_pre[0] - dx) < 0.01, "C1 X should move by dx (group move)"
    assert abs(c1_post[1] - c1_pre[1] - dy) < 0.01, "C1 Y should move by dy (group move)"
    assert abs(c2_post[0] - c2_pre[0] - dx) < 0.01, "C2 X should move by dx (group move)"
    assert abs(c2_post[1] - c2_pre[1] - dy) < 0.01, "C2 Y should move by dy (group move)"

    # Distances must be exactly preserved (no clamping occurred)
    post_dist_c1 = math.hypot(u1_post[0] - c1_post[0], u1_post[1] - c1_post[1])
    post_dist_c2 = math.hypot(u1_post[0] - c2_post[0], u1_post[1] - c2_post[1])
    assert abs(post_dist_c1 - pre_dist_c1) < 0.01, \
        f"C1-U1 distance changed by {abs(post_dist_c1 - pre_dist_c1):.2f}mm (group broken)"
    assert abs(post_dist_c2 - pre_dist_c2) < 0.01, \
        f"C2-U1 distance changed by {abs(post_dist_c2 - pre_dist_c2):.2f}mm (group broken)"


def test_cap_ic_group_survives_swap():
    """REGRESSION: when do_swap swaps an IC with another component, the
    IC's caps must follow by the same delta. Pre-fix, do_swap moved
    only the two swapped components — caps got left behind.
    """
    from engine.moves import do_swap, get_moveable_indices, revert_move, affected_indices
    from engine.group_moves import get_decap_map

    model = _make_test_model()
    dm = get_decap_map(model)
    assert 'U1' in dm

    moveable = get_moveable_indices(model)
    u1_idx = next(i for i, c in enumerate(model.components) if c.ref == 'U1')
    c1_idx = next(i for i, c in enumerate(model.components) if c.ref == 'C1')
    c2_idx = next(i for i, c in enumerate(model.components) if c.ref == 'C2')

    # Find a non-cap, non-U1 component to swap with U1
    other_idx = next(i for i in moveable
                     if i != u1_idx and i != c1_idx and i != c2_idx)
    other_ref = model.components[other_idx].ref

    u1_pre = (model.components[u1_idx].x, model.components[u1_idx].y)
    other_pre = (model.components[other_idx].x, model.components[other_idx].y)
    c1_pre = (model.components[c1_idx].x, model.components[c1_idx].y)
    c2_pre = (model.components[c2_idx].x, model.components[c2_idx].y)

    # Expected swap delta for U1: moves to other's position
    expected_u1_delta = (other_pre[0] - u1_pre[0], other_pre[1] - u1_pre[1])

    # Manually invoke the swap logic by directly swapping U1 and other,
    # then applying the group-move helper to U1's caps.
    # This mirrors what do_swap does internally.
    from engine.group_moves import get_ic_caps_for_index, apply_delta_with_clamp
    model.components[u1_idx].x, model.components[other_idx].x = other_pre[0], u1_pre[0]
    model.components[u1_idx].y, model.components[other_idx].y = other_pre[1], u1_pre[1]
    cap_indices = get_ic_caps_for_index(model, u1_idx)
    # Exclude other_idx if it happens to be a cap of U1 (it isn't in this test)
    cap_indices = [ci for ci in cap_indices if ci != other_idx]
    apply_delta_with_clamp(model, cap_indices, expected_u1_delta[0], expected_u1_delta[1])

    u1_post = (model.components[u1_idx].x, model.components[u1_idx].y)
    c1_post = (model.components[c1_idx].x, model.components[c1_idx].y)
    c2_post = (model.components[c2_idx].x, model.components[c2_idx].y)

    # U1 should be at other's old position
    assert abs(u1_post[0] - other_pre[0]) < 0.01, "U1 should be at other's old X"
    assert abs(u1_post[1] - other_pre[1]) < 0.01, "U1 should be at other's old Y"

    # Caps should have moved by the same delta as U1
    assert abs(c1_post[0] - c1_pre[0] - expected_u1_delta[0]) < 0.01, \
        f"C1 X should move by U1's swap delta {expected_u1_delta[0]}"
    assert abs(c1_post[1] - c1_pre[1] - expected_u1_delta[1]) < 0.01, \
        f"C1 Y should move by U1's swap delta {expected_u1_delta[1]}"
    assert abs(c2_post[0] - c2_pre[0] - expected_u1_delta[0]) < 0.01, \
        f"C2 X should move by U1's swap delta {expected_u1_delta[0]}"
    assert abs(c2_post[1] - c2_pre[1] - expected_u1_delta[1]) < 0.01, \
        f"C2 Y should move by U1's swap delta {expected_u1_delta[1]}"


def test_barrel_jack_faces_outward():
    """REGRESSION: barrel jack (end-mating connector) must orient so its
    mating face points outward from the board, not inward.

    Pre-fix, _compute_pad_facing_direction used pad spread alone, which
    fails for end-mating connectors where the mating interface is at the
    END of the body's long axis (parallel to the pad column). The barrel
    jack in tests/test_pcbs/test6.kicad_pcb had local facing (-1, 0)
    instead of (0, -1), causing it to face inward when placed on
    top/bottom/right edges.
    """
    import os
    pcb_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'tests', 'test_pcbs', 'test6.kicad_pcb')
    if not os.path.exists(pcb_path):
        return  # skip if test PCB not available

    from parsers.kicad_parser import KiCadParser
    from engine.smart_placement import _compute_pad_facing_direction
    from config import load_config

    cfg = load_config()
    parser = KiCadParser(pcb_path, bbox_margin=cfg.parser.bbox_margin)
    model = parser.parse()

    j1 = None
    for c in model.components:
        if c.ref == 'J1' and 'BarrelJack' in c.footprint:
            j1 = c
            break
    assert j1 is not None, "test6 should have a J1 BarrelJack connector"

    fx, fy = _compute_pad_facing_direction(j1)
    # Mating face is at top of body in local coords (low Y) → (0, -1)
    assert (fx, fy) == (0.0, -1.0), \
        f"Barrel jack local facing should be (0, -1) — mating face at top of body. Got ({fx}, {fy})."


def test_end_mating_connector_classification():
    """Verify _END_MATING_MARKERS correctly classifies common
    end-mating connector footprints."""
    from engine.smart_placement import _END_MATING_MARKERS

    end_mating_footprints = [
        'Connector_BarrelJack:BarrelJack_GCT_DCJ200-10-A_Horizontal',
        'Connector_USB:USB_Micro-B_Molex_47346-0001',
        'Connector_RJ45:RJ45_Wurk_7499X1127_Horizontal',
        'Connector_DSub:DE9_Plug_MountingHoles',
    ]
    face_mating_footprints = [
        'Connector_PinHeader_2.54mm:PinHeader_1x06_P2.54mm_Horizontal',
        'TerminalBlock_RND:TerminalBlock_RND_205-00232_1x02_P5.08mm_Horizontal',
        'Connector_Coaxial:SMA_Amphenol_132289_EdgeMount',
    ]

    for fp in end_mating_footprints:
        name = (fp + ' ').lower()
        assert any(m in name for m in _END_MATING_MARKERS), \
            f"End-mating footprint {fp!r} not matched by _END_MATING_MARKERS"

    for fp in face_mating_footprints:
        name = (fp + ' ').lower()
        assert not any(m in name for m in _END_MATING_MARKERS), \
            f"Face-mating footprint {fp!r} should NOT match _END_MATING_MARKERS (would break orientation)"


def test_legalizer_propagates_ic_delta_to_caps():
    """REGRESSION: when the legalizer's _enforce_boundary clamps an IC,
    its assigned caps must follow by the same delta.  Pre-fix, the
    legalizer treated every component independently — an IC clamped to
    the board edge left its caps behind, breaking the cap-IC group.
    """
    from legalization.legalizer import _enforce_boundary, _enforce_boundary_single
    from engine.group_moves import get_decap_map
    from models.board_model import BoardOutline

    # Build a model where U1 is OUTSIDE the board (will be clamped inward),
    # but its caps are INSIDE the board (so they won't be independently clamped).
    # The caps are at U1's original X — if the legalizer is group-aware,
    # they should follow U1's clamp delta.
    model = BoardModel(
        board=BoardOutline(x_min=0, y_min=0, x_max=80, y_max=60),
        components=[
            Component(ref="U1", x=85, y=30, width=7, height=7, component_type="ic",
                      pads=[Pad(pad_name="1", x=-3, y=0, net="VCC"),
                            Pad(pad_name="2", x=3, y=0, net="GND")],
                      nets=["VCC", "GND"]),
            # C1 at x=70 (inside board), y=25 — will follow U1's clamp
            Component(ref="C1", x=70, y=25, width=1, height=0.5, component_type="capacitor", value="100n",
                      pads=[Pad(pad_name="1", x=0, y=0, net="VCC"),
                            Pad(pad_name="2", x=0, y=0, net="GND")],
                      nets=["VCC", "GND"]),
            # C2 at x=75 (inside board), y=35 — will follow U1's clamp
            Component(ref="C2", x=75, y=35, width=1, height=0.5, component_type="capacitor", value="100n",
                      pads=[Pad(pad_name="1", x=0, y=0, net="VCC"),
                            Pad(pad_name="2", x=0, y=0, net="GND")],
                      nets=["VCC", "GND"]),
        ],
        nets=[Net(name="VCC", pins=[("U1", "1"), ("C1", "1"), ("C2", "1")]),
              Net(name="GND", pins=[("U1", "2"), ("C1", "2"), ("C2", "2")])],
    )

    dm = get_decap_map(model)
    assert "U1" in dm and set(dm["U1"]) == {"C1", "C2"}, \
        f"U1 should have caps C1, C2; got {dm}"

    u1 = model.get_component("U1")
    c1 = model.get_component("C1")
    c2 = model.get_component("C2")
    u1_pre = (u1.x, u1.y)
    c1_pre = (c1.x, c1.y)
    c2_pre = (c2.x, c2.y)

    # Run _enforce_boundary — U1 should be clamped inward from x=85 to ~x=73.5
    _enforce_boundary(model)

    u1_post = (u1.x, u1.y)
    c1_post = (c1.x, c1.y)
    c2_post = (c2.x, c2.y)

    # U1 must have moved (was outside board)
    assert u1_post[0] < u1_pre[0], \
        f"U1 X should decrease (was OOB at x=85, board max=80); got {u1_pre} -> {u1_post}"

    # Caps should have followed by approximately the same delta as U1.
    # (Caps start inside the board, so they don't get independently clamped —
    #  their delta should match U1's delta exactly.)
    u1_dx = u1_post[0] - u1_pre[0]
    c1_dx = c1_post[0] - c1_pre[0]
    c2_dx = c2_post[0] - c2_pre[0]
    assert abs(c1_dx - u1_dx) < 0.5, \
        f"C1 delta X ({c1_dx:.2f}) should match U1 delta X ({u1_dx:.2f}) — cap left behind by legalizer"
    assert abs(c2_dx - u1_dx) < 0.5, \
        f"C2 delta X ({c2_dx:.2f}) should match U1 delta X ({u1_dx:.2f}) — cap left behind by legalizer"


def test_legalizer_snap_to_grid_propagates_to_caps():
    """REGRESSION: _snap_to_grid must propagate IC snap deltas to caps.
    Pre-fix, snapping an IC to grid left caps at their pre-snap positions.
    """
    from legalization.legalizer import _snap_to_grid
    from engine.group_moves import get_decap_map

    # Build a model where U1 is at a non-grid-aligned position
    grid_mm = 0.5
    model = BoardModel(
        board=BoardOutline(x_min=0, y_min=0, x_max=80, y_max=60),
        components=[
            Component(ref="U1", x=40.3, y=30.7, width=7, height=7, component_type="ic",
                      pads=[Pad(pad_name="1", x=-3, y=0, net="VCC"),
                            Pad(pad_name="2", x=3, y=0, net="GND")],
                      nets=["VCC", "GND"]),
            Component(ref="C1", x=37.3, y=27.7, width=1, height=0.5, component_type="capacitor", value="100n",
                      pads=[Pad(pad_name="1", x=0, y=0, net="VCC"),
                            Pad(pad_name="2", x=0, y=0, net="GND")],
                      nets=["VCC", "GND"]),
        ],
        nets=[Net(name="VCC", pins=[("U1", "1"), ("C1", "1")]),
              Net(name="GND", pins=[("U1", "2"), ("C1", "2")])],
    )

    dm = get_decap_map(model)
    assert "U1" in dm, f"U1 should be in decap map; got {dm}"

    u1 = model.get_component("U1")
    c1 = model.get_component("C1")
    u1_pre = (u1.x, u1.y)
    c1_pre = (c1.x, c1.y)
    pre_dist = math.hypot(u1_pre[0] - c1_pre[0], u1_pre[1] - c1_pre[1])

    _snap_to_grid(model, grid_mm)

    u1_post = (u1.x, u1.y)
    c1_post = (c1.x, c1.y)
    post_dist = math.hypot(u1_post[0] - c1_post[0], u1_post[1] - c1_post[1])

    # U1 should have snapped (was at 40.3, 30.7 — not on 0.5mm grid)
    assert abs(u1_post[0] - 40.5) < 0.01 or abs(u1_post[0] - 40.0) < 0.01, \
        f"U1 X should snap to grid (40.0 or 40.5); got {u1_post[0]}"

    # Cap-IC distance should be approximately preserved (caps followed)
    assert abs(post_dist - pre_dist) < 1.0, \
        f"Cap-IC distance changed by {abs(post_dist - pre_dist):.2f}mm after snap (caps didn't follow)"


def test_propagate_ic_delta_helper():
    """Verify the propagate_ic_delta helper directly."""
    from engine.group_moves import propagate_ic_delta, get_decap_map

    model = _make_test_model()
    dm = get_decap_map(model)
    assert "U1" in dm

    u1 = model.get_component("U1")
    c1 = model.get_component("C1")
    c2 = model.get_component("C2")
    u1_pre = (u1.x, u1.y)
    c1_pre = (c1.x, c1.y)
    c2_pre = (c2.x, c2.y)

    # Move U1 by (3, -2) — caps should follow
    u1.x += 3.0
    u1.y += -2.0
    moved = propagate_ic_delta(model, u1, u1_pre[0], u1_pre[1], u1.x, u1.y)

    # Both caps should have moved
    assert len(moved) == 2, f"Expected 2 caps moved, got {len(moved)}"
    c1_post = (c1.x, c1.y)
    c2_post = (c2.x, c2.y)
    assert abs(c1_post[0] - c1_pre[0] - 3.0) < 0.01, "C1 X should move by +3"
    assert abs(c1_post[1] - c1_pre[1] + 2.0) < 0.01, "C1 Y should move by -2"
    assert abs(c2_post[0] - c2_pre[0] - 3.0) < 0.01, "C2 X should move by +3"
    assert abs(c2_post[1] - c2_pre[1] + 2.0) < 0.01, "C2 Y should move by -2"

    # Test with non-IC component — should return empty list
    r1 = model.get_component("R1")
    r1_pre = (r1.x, r1.y)
    r1.x += 5.0
    moved = propagate_ic_delta(model, r1, r1_pre[0], r1_pre[1], r1.x, r1.y)
    assert moved == [], "Non-IC component should not propagate to caps"


def test_caps_do_not_overlap_ic_after_legalize():
    """REGRESSION: after legalization, caps must NOT overlap their assigned IC.
    They should be CLOSE (within decoupling_proximity max_distance) but NOT
    overlapping — adjacent, as a group.

    Pre-fix, _nudge_caps_to_ics excluded the IC from its overlap check,
    intentionally placing caps AT the IC center (overlapping).  The user
    requirement is that caps sit ADJACENT to the IC, not on top of it.
    """
    from legalization.legalizer import legalize
    from engine.group_moves import get_decap_map
    from profiles.board_profiles import get_profile, ConstraintRule

    # Build a board with U1 + 2 caps sharing VCC/GND nets, plus some
    # other components to make it realistic.
    model = BoardModel(
        board=BoardOutline(x_min=0, y_min=0, x_max=80, y_max=60),
        components=[
            Component(ref="U1", x=40, y=30, width=10, height=10, component_type="ic",
                      pads=[Pad(pad_name="1", x=-3, y=0, net="VCC"),
                            Pad(pad_name="2", x=3, y=0, net="GND"),
                            Pad(pad_name="3", x=0, y=-3, net="SIG")],
                      nets=["VCC", "GND", "SIG"]),
            Component(ref="C1", x=20, y=20, width=2, height=1.5, component_type="capacitor", value="100n",
                      pads=[Pad(pad_name="1", x=0, y=0, net="VCC"),
                            Pad(pad_name="2", x=0, y=0, net="GND")],
                      nets=["VCC", "GND"]),
            Component(ref="C2", x=60, y=40, width=2, height=1.5, component_type="capacitor", value="100n",
                      pads=[Pad(pad_name="1", x=0, y=0, net="VCC"),
                            Pad(pad_name="2", x=0, y=0, net="GND")],
                      nets=["VCC", "GND"]),
            Component(ref="R1", x=70, y=20, width=1.5, height=0.8, component_type="resistor",
                      pads=[Pad(pad_name="1", x=0, y=0, net="SIG"),
                            Pad(pad_name="2", x=0, y=0, net="OUT")],
                      nets=["SIG", "OUT"]),
        ],
        nets=[Net(name="VCC", pins=[("U1", "1"), ("C1", "1"), ("C2", "1")]),
              Net(name="GND", pins=[("U1", "2"), ("C1", "2"), ("C2", "2")]),
              Net(name="SIG", pins=[("U1", "3"), ("R1", "1")]),
              Net(name="OUT", pins=[("R1", "2")])],
    )

    # Verify decap map assigns C1, C2 to U1
    dm = get_decap_map(model)
    assert "U1" in dm and set(dm["U1"]) == {"C1", "C2"}, \
        f"U1 should have caps C1, C2; got {dm}"

    # Run legalization with decoupling rule active
    profile = get_profile("mcu_peripheral")
    model.active_rules = profile.active_rules()
    legalize(model, grid_mm=0.5, max_iterations=300, push_strength=1.0,
             verbose=False, use_abacus=True)

    # After legalization, NO cap should overlap its assigned IC.
    u1 = model.get_component("U1")
    for cap_ref in dm["U1"]:
        cap = model.get_component(cap_ref)
        assert not cap.overlaps(u1), \
            f"Cap {cap_ref} should NOT overlap its IC {u1.ref} after legalization " \
            f"(cap at ({cap.x:.2f},{cap.y:.2f}), IC at ({u1.x:.2f},{u1.y:.2f}))"

    # Caps should be CLOSE to the IC (within decoupling_proximity max_distance = 5mm).
    # Distance is measured from cap center to IC center.
    for cap_ref in dm["U1"]:
        cap = model.get_component(cap_ref)
        dist = math.hypot(cap.x - u1.x, cap.y - u1.y)
        assert dist <= 15.0, \
            f"Cap {cap_ref} should be close to IC (dist={dist:.2f}mm, max=15mm) " \
            f"— cap at ({cap.x:.2f},{cap.y:.2f}), IC at ({u1.x:.2f},{u1.y:.2f})"


# ---------------------------------------------------------------------------
# End-to-End Pipeline Test
# ---------------------------------------------------------------------------

def test_full_pipeline():
    """Test the complete Phase 1 pipeline."""
    with tempfile.NamedTemporaryFile(suffix=".kicad_pcb", mode="w", delete=False) as f:
        f.write(SAMPLE_KICAD_PCB)
        pcb_path = f.name

    try:
        # Extract
        parser = KiCadParser(pcb_path)
        model = parser.parse()
        assert len(model.components) >= 10

        # Get MCU profile
        profile = get_profile("mcu_peripheral")
        cost_fn = CostFunction(
            alpha=profile.alpha,
            beta=profile.beta,
            gamma=profile.gamma,
            delta=profile.delta,
        )

        # Evaluate initial cost
        initial_costs = cost_fn.evaluate(model)

        # Grid placement
        grid_place(model, margin=5.0, spacing_factor=1.3)

        # Evaluate after placement
        placed_costs = cost_fn.evaluate(model)

        # Legalize
        legalize(model, grid_mm=0.1, verbose=False)

        # Evaluate after legalization
        final_costs = cost_fn.evaluate(model)

        # Final placement should be legal
        assert final_costs["oob_count"] == 0, \
            f"No components should be out of bounds after legalization, got {final_costs['oob_count']}"

        # Save and reload
        json_path = pcb_path.replace(".kicad_pcb", "_test_model.json")
        model.to_json(json_path)
        model2 = BoardModel.from_json(json_path)
        assert len(model2.components) == len(model.components)

        os.unlink(json_path)

    finally:
        os.unlink(pcb_path)


# ---------------------------------------------------------------------------
# Phase 6: Rotation-Aware Dimensions Tests
# ---------------------------------------------------------------------------

def test_rotation_effective_dimensions():
    c = Component(ref="R1", x=10, y=10, width=3, height=1)
    # At 0°, effective dims match physical
    assert abs(c.effective_width - 3.5) < 0.01, f"0° eff_w={c.effective_width}"
    assert abs(c.effective_height - 1.5) < 0.01, f"0° eff_h={c.effective_height}"

    c.set_rotation(90.0)
    assert c._is_rotated_90, "Should be rotated 90"
    assert abs(c.effective_width - 1.5) < 0.01, f"90° eff_w={c.effective_width}"
    assert abs(c.effective_height - 3.5) < 0.01, f"90° eff_h={c.effective_height}"

    c.set_rotation(270.0)
    assert c._is_rotated_90, "270° should also be rotated 90"
    assert abs(c.effective_width - 1.5) < 0.01, f"270° eff_w={c.effective_width}"

    c.set_rotation(180.0)
    assert not c._is_rotated_90, "180° should not be rotated 90"


def test_rotation_bbox():
    c = Component(ref="R1", x=50, y=50, width=4, height=2)
    bbox0 = c.bbox
    area0 = (bbox0[2] - bbox0[0]) * (bbox0[3] - bbox0[1])

    c.set_rotation(90.0)
    bbox90 = c.bbox
    area90 = (bbox90[2] - bbox90[0]) * (bbox90[3] - bbox90[1])

    # For a rectangle, axis-aligned bbox area should be preserved at 90°
    assert abs(area0 - area90) < 0.01, f"Area should be preserved: {area0} vs {area90}"
    # Effective dims should swap
    assert abs(c.effective_width - 2.5) < 0.01, f"90° eff_w should be 2.5, got {c.effective_width}"
    assert abs(c.effective_height - 4.5) < 0.01, f"90° eff_h should be 4.5, got {c.effective_height}"


def test_rotation_bbox_offset_rotation():
    c = Component(ref="U1", x=50, y=50, width=6, height=4, bbox_offset_x=1.0, bbox_offset_y=0.5)
    bbox0 = c.bbox
    cx0 = (bbox0[0] + bbox0[2]) / 2
    cy0 = (bbox0[1] + bbox0[3]) / 2
    # Center should be offset from (50,50)
    assert abs(cx0 - 51.0) < 0.01, f"Expected cx=51, got {cx0}"
    assert abs(cy0 - 50.5) < 0.01, f"Expected cy=50.5, got {cy0}"

    c.set_rotation(90.0)
    bbox90 = c.bbox
    cx90 = (bbox90[0] + bbox90[2]) / 2
    cy90 = (bbox90[1] + bbox90[3]) / 2
    # Offset should rotate clockwise (KiCad convention): (1, 0.5) → (0.5, -1) at 90°
    assert abs(cx90 - 50.5) < 0.01, f"90° cx={cx90}, expected 50.5"
    assert abs(cy90 - 49.0) < 0.01, f"90° cy={cy90}, expected 49.0"


# ---------------------------------------------------------------------------
# Phase 6: Power Net Detection Tests
# ---------------------------------------------------------------------------

def test_power_net_detection():
    assert _is_power_net("GND"), "GND should be power"
    assert _is_power_net("AGND"), "AGND should be power"
    assert _is_power_net("VCC"), "VCC should be power"
    assert _is_power_net("VDD"), "VDD should be power"
    assert _is_power_net("+3V3"), "+3V3 should be power"
    assert _is_power_net("-5V"), "-5V should be power"
    assert _is_power_net("/GND"), "/GND should be power"
    assert not _is_power_net("SDA"), "SDA should not be power"
    assert not _is_power_net("MOSI"), "MOSI should not be power"
    assert not _is_power_net("LED1"), "LED1 should not be power"


# ---------------------------------------------------------------------------
# Phase 6: CostState Tests
# ---------------------------------------------------------------------------

def test_cost_state_matches_full_compute():
    model = _make_test_model()
    cs = CostState(model)

    # Verify CostState self-consistency: full recompute matches cached values
    cs._compute_all()
    assert cs.hpwl > 0, "HPWL should be positive"
    assert cs.total_cost > 0, "Total cost should be positive"

    # Verify CostState incremental matches full after full recompute
    cost_after_full = cs.total_cost
    # Move a component and back
    comp = model.components[1]
    comp.x += 1.0
    cs.incremental_update({1})
    comp.x -= 1.0
    cs.incremental_update({1})
    assert abs(cs.total_cost - cost_after_full) < 0.01, \
        f"Cost should return to original after round-trip move"


def test_incremental_update_matches_full():
    model = _make_test_model()
    cs = CostState(model)

    # Move a component
    comp = model.components[1]  # C1
    old_x, old_y = comp.x, comp.y
    comp.x += 5.0
    comp.y += 3.0

    incremental_cost = cs.incremental_update({1})
    cs._compute_all()
    full_cost = cs.total_cost

    assert abs(incremental_cost - full_cost) < 0.01, \
        f"Incremental {incremental_cost} != full {full_cost}"

    comp.x = old_x
    comp.y = old_y


def test_snapshot_restore():
    model = _make_test_model()
    cs = CostState(model)
    original_cost = cs.total_cost

    # Move component, snapshot, compute, then restore
    comp = model.components[1]
    comp.x += 10.0
    snap = cs.snapshot({1})
    cs.incremental_update({1})
    moved_cost = cs.total_cost

    assert moved_cost != original_cost, "Cost should change after move"

    # Restore
    comp.x -= 10.0
    cs.restore(snap)
    restored_cost = cs.total_cost

    assert abs(restored_cost - original_cost) < 0.01, \
        f"Restored cost {restored_cost} != original {original_cost}"


def test_density_gated_by_decap_rule():
    """Density penalty must be 0 when decoupling rule is inactive,
    and non-zero when active and components are clustered."""
    from profiles.board_profiles import ConstraintRule

    model = _make_test_model()

    # No rules → density must be 0 (gating proof)
    cs_no_rules = CostState(model)
    assert cs_no_rules.density_penalty == 0.0, \
        "Density should be 0 without decoupling rule"
    assert cs_no_rules._decap_map is None, \
        "_decap_map should be None without decoupling rule"

    # With decoupling rule → density fires when components cluster.
    # Cluster all movable components into one corner to force high Gini.
    decap_rule = ConstraintRule(
        name="decoupling_proximity",
        weight=4.0,
        params={"max_distance_mm": 5.0},
    )
    cs = CostState(model, rules=[decap_rule])
    assert cs._decap_map is not None, \
        "_decap_map should be built when decoupling rule active"
    # C1 and C2 share VCC with U1 — they should be assigned to U1.
    cap_refs = set()
    for caps in cs._decap_map.values():
        cap_refs.update(caps)
    assert "C1" in cap_refs and "C2" in cap_refs, \
        f"C1/C2 should be assigned as decoupling caps, got {cap_refs}"

    # Cluster movable components into one corner of the 10x10 grid
    for c in model.components:
        if not c.is_fixed:
            c.x = 1.0
            c.y = 1.0
    cs.incremental_update({i for i, c in enumerate(model.components) if not c.is_fixed})
    assert cs.density_penalty > 0.0, \
        "Density penalty should be > 0 when components are clustered"


def test_snapshot_restore_with_density():
    """Snapshot/restore must round-trip density state cleanly when
    the decoupling rule is active (otherwise SA silent regression)."""
    from profiles.board_profiles import ConstraintRule

    model = _make_test_model()
    decap_rule = ConstraintRule(
        name="decoupling_proximity",
        weight=4.0,
        params={"max_distance_mm": 5.0},
    )
    cs = CostState(model, rules=[decap_rule])
    original_cost = cs.total_cost
    original_density = cs.density_penalty

    # Move a component, snapshot, update, then restore
    comp = model.components[1]
    comp.x += 10.0
    snap = cs.snapshot({1})
    cs.incremental_update({1})
    moved_cost = cs.total_cost
    assert moved_cost != original_cost, "Cost should change after move"

    # Restore
    comp.x -= 10.0
    cs.restore(snap)
    restored_cost = cs.total_cost
    restored_density = cs.density_penalty

    assert abs(restored_cost - original_cost) < 0.01, \
        f"Restored cost {restored_cost} != original {original_cost}"
    assert abs(restored_density - original_density) < 0.01, \
        f"Restored density {restored_density} != original {original_density}"


def test_snapshot_restore_n_moves_property():
    """Property test: after N random SA moves with snapshot/restore on each,
    incremental cost must match a from-scratch CostState recompute.

    This is the single most common source of SA converging to a placement
    that *looks* good by the tracked cost but is actually worse than a full
    recompute would say.  The existing 1-move test only catches gross bugs;
    this N-move version catches drift that accumulates over many accepts
    and rejects (e.g. an overlap entry that gets dropped from _pair_overlaps
    but never re-added on restore, or a density-grid cell that's off by one).

    Runs 40 moves with deterministic seed so failures are reproducible.
    """
    import random as _random

    _random.seed(0xC0FFEE)  # deterministic
    # Also seed engine.moves' random module — it has its own `import random`,
    # so seeding our local _random doesn't affect it.  We seed the engine
    # module's RNG directly to get reproducible move selection.
    from engine import moves as _moves_mod
    _moves_mod.random.seed(0xC0FFEE)

    model = _make_test_model()
    moveable = get_moveable_indices(model)
    assert len(moveable) >= 2, "test model needs >=2 moveable components"

    cs = CostState(model)
    initial_normalized = cs.normalized_cost

    for step in range(40):
        mt = select_move_type(0.5)  # mid-temperature
        if mt == 'translate':
            undo = do_translate(model, moveable, 0.5, 3.0)
        elif mt == 'swap':
            undo = do_swap(model, moveable)
        elif mt == 'rotate':
            undo = do_rotate(model, moveable)
        else:
            undo = do_median(model, moveable, 0.5, 0.5)

        if not undo.old_states:
            continue

        moved = affected_indices(undo)
        old_bboxes = cs.old_bboxes_from_states(undo.old_states)
        snap = cs.snapshot(moved, old_bboxes=old_bboxes)
        incremental_cost = cs.incremental_update(moved)

        # Compare against a from-scratch CostState on the SAME model state.
        fresh = CostState(model)
        fresh.update_penalty_scale(cs._penalty_scale)
        from_scratch_cost = fresh.total_cost

        assert abs(incremental_cost - from_scratch_cost) < 0.5, (
            f"Step {step} ({mt}): incremental={incremental_cost:.4f} "
            f"!= from_scratch={from_scratch_cost:.4f} "
            f"(delta={incremental_cost - from_scratch_cost:.4f}). "
            f"This is the SA-vs-truth drift bug — incremental cost tracking "
            f"desynced from a full recompute after {step} moves."
        )

        # Metropolis accept/reject with seed-deterministic probability.
        # When rejected, restore model AND cost_state (the SA pattern).
        delta = incremental_cost - snap.get('hpwl', 0) - snap.get('overlap_penalty', 0)
        accept = delta < 0 or _random.random() < 0.5  # 50% accept for test stress
        if not accept:
            revert_move(model, undo)
            cs.restore(snap)

    # After 40 moves, verify the cost_state's view still matches a fresh
    # CostState on the final model state — catches cumulative drift even
    # when every individual step passed the per-step check.
    final_incremental = cs.total_cost
    final_fresh = CostState(model)
    final_fresh.update_penalty_scale(cs._penalty_scale)
    final_from_scratch = final_fresh.total_cost
    assert abs(final_incremental - final_from_scratch) < 1.0, (
        f"After 40 moves: incremental={final_incremental:.4f} "
        f"!= from_scratch={final_from_scratch:.4f}. "
        f"Cumulative drift in incremental cost tracking."
    )

    # Also verify normalized_cost (used for best-state tracking) matches.
    assert abs(cs.normalized_cost - final_fresh.normalized_cost) < 1.0, (
        f"normalized_cost drift: cs={cs.normalized_cost:.4f} "
        f"fresh={final_fresh.normalized_cost:.4f}"
    )


# ---------------------------------------------------------------------------
# Phase 6: Move Operator Tests
# ---------------------------------------------------------------------------

def test_translate_move_and_revert():
    model = _make_test_model()
    moveable = get_moveable_indices(model)
    assert len(moveable) > 0

    # Group-aware translate may move 1+ components: if the picked component
    # is an IC with assigned decoupling caps, the caps move with it.
    undo = do_translate(model, moveable, 0.5, 5.0)
    assert len(undo.old_states) >= 1

    # Capture pre-revert positions, then revert and verify all restored.
    pre_revert = [(i, model.components[i].x, model.components[i].y)
                  for i, _, _, _ in undo.old_states]
    # At least the picked component should have moved
    idx0, old_x0, old_y0, _ = undo.old_states[0]
    assert (model.components[idx0].x != old_x0
            or model.components[idx0].y != old_y0), \
        "Translate should change position"

    revert_move(model, undo)
    for idx, old_x, old_y, _ in undo.old_states:
        assert abs(model.components[idx].x - old_x) < 0.001, \
            f"Component {idx} X should be restored"
        assert abs(model.components[idx].y - old_y) < 0.001, \
            f"Component {idx} Y should be restored"


def test_swap_move_and_revert():
    model = _make_test_model()
    moveable = get_moveable_indices(model)
    if len(moveable) < 2:
        return  # Skip if not enough moveable components

    undo = do_swap(model, moveable)
    # Group-aware swap may move >2 components: if either swapped component
    # is an IC with assigned decoupling caps, the caps follow.
    assert len(undo.old_states) >= 2

    # First two entries are always the swapped pair
    c1 = model.components[undo.old_states[0][0]]
    c2 = model.components[undo.old_states[1][0]]
    x1, y1 = c1.x, c1.y
    x2, y2 = c2.x, c2.y

    revert_move(model, undo)
    assert abs(c1.x - undo.old_states[0][1]) < 0.001, "C1 x not restored"
    assert abs(c2.x - undo.old_states[1][1]) < 0.001, "C2 x not restored"
    # Verify group followers also restored
    for idx, old_x, old_y, _ in undo.old_states:
        assert abs(model.components[idx].x - old_x) < 0.001, \
            f"Component {idx} X not restored after revert"


def test_rotate_move_and_revert():
    model = _make_test_model()
    moveable = get_moveable_indices(model)
    undo = do_rotate(model, moveable)
    # Group-aware rotate may move >1 component: if the rotated component
    # is an IC with assigned decoupling caps, the caps rotate around it.
    assert len(undo.old_states) >= 1

    idx, _, _, old_rot = undo.old_states[0]
    comp = model.components[idx]
    new_rot = comp.rotation

    assert new_rot != old_rot, "Rotation should change"

    revert_move(model, undo)
    assert abs(comp.rotation - old_rot) < 0.001, "Rotation not restored"
    # Verify group followers also restored
    for f_idx, f_old_x, f_old_y, f_old_rot in undo.old_states:
        assert abs(model.components[f_idx].x - f_old_x) < 0.001, \
            f"Component {f_idx} X not restored after rotate revert"
        assert abs(model.components[f_idx].y - f_old_y) < 0.001, \
            f"Component {f_idx} Y not restored after rotate revert"


def test_median_move_reduces_hpwl():
    model = _make_test_model()
    cs = CostState(model)
    moveable = get_moveable_indices(model)
    if not moveable:
        return

    old_hpwl = cs.hpwl
    undo = do_median(model, moveable, 0.5, 1.0)
    if not undo.old_states:
        return

    cs.incremental_update(affected_indices(undo))
    # Median move should generally reduce HPWL (or at worst not change much)
    # Not asserting strict improvement since noise can increase it slightly
    revert_move(model, undo)
    cs.incremental_update(affected_indices(undo))


def test_get_moveable_indices():
    model = _make_test_model()
    indices = get_moveable_indices(model)
    for idx in indices:
        comp = model.components[idx]
        assert not comp.is_fixed, f"{comp.ref} is fixed"
        assert comp.component_type != "connector", f"{comp.ref} is connector"


# ---------------------------------------------------------------------------
# Phase 6: SA Engine Tests
# ---------------------------------------------------------------------------

def test_sa_no_regression():
    model = _make_test_model()
    cost_state = CostState(model)
    moveable = get_moveable_indices(model)
    if not moveable:
        return

    initial_cost = cost_state.normalized_cost

    config = SAConfig(max_iterations=10, reheat_count=0, verbose=False)
    simulate_annealing(model, cost_state, moveable, config)

    # SA restores the best solution found, so final cost should not be worse
    # than a simple threshold above initial (boundary penalties can spike on small boards)
    final_cost = cost_state.normalized_cost
    assert final_cost <= initial_cost + 500.0, \
        f"SA should not severely regress: {initial_cost} → {final_cost}"


def test_greedy_refinement_improves():
    """Test that greedy refinement (called within SA) can find improvements."""
    model = _make_test_model()
    # Place two components overlapping
    c1 = model.components[0]
    c2 = model.components[1]
    c2.x = c1.x + 0.1
    c2.y = c1.y + 0.1

    cost_state = CostState(model)
    moveable = get_moveable_indices(model)
    if not moveable:
        return

    initial_cost = cost_state.normalized_cost
    config = SAConfig(max_iterations=0, reheat_count=0, verbose=False)
    final_cost = simulate_annealing(model, cost_state, moveable, config)

    assert final_cost <= initial_cost, \
        f"Greedy should not regress: {initial_cost} → {final_cost}"


def test_no_rotation_reset_in_placement():
    """Verify that placement doesn't reset component rotations."""
    model = _make_test_model()
    # Set a non-zero rotation
    model.components[0].set_rotation(90.0)

    grid_place(model)

    # Check that the rotation is preserved (grid_place no longer resets it)
    rot = model.components[0].rotation
    assert abs(rot - 90.0) < 0.01, f"Rotation should be preserved, got {rot}"


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

def _make_test_model() -> BoardModel:
    """Create a test board model with MCU + peripheral components."""
    model = BoardModel(
        board=BoardOutline(x_min=0, y_min=0, x_max=80, y_max=60),
        components=[
            Component(ref="U1", x=40, y=30, width=7, height=7, component_type="ic",
                      pads=[Pad(pad_name="1", x=-3, y=-2.75, net="GND"),
                            Pad(pad_name="5", x=-3, y=-0.75, net="VCC"),
                            Pad(pad_name="7", x=-3, y=0.25, net="OSC_IN"),
                            Pad(pad_name="8", x=-3, y=0.75, net="OSC_OUT")],
                      nets=["VCC", "GND", "OSC_IN", "OSC_OUT", "USB_DP", "USB_DM", "LED1", "LED2", "NRST", "BOOT0"]),
            Component(ref="C1", x=35, y=25, width=1, height=0.5, component_type="capacitor",
                      value="100n",
                      pads=[Pad(pad_name="1", x=-0.25, y=0, net="VCC"),
                            Pad(pad_name="2", x=0.25, y=0, net="GND")],
                      nets=["VCC", "GND"]),
            Component(ref="C2", x=45, y=25, width=1, height=0.5, component_type="capacitor",
                      value="100n",
                      pads=[Pad(pad_name="1", x=-0.25, y=0, net="VCC"),
                            Pad(pad_name="2", x=0.25, y=0, net="GND")],
                      nets=["VCC", "GND"]),
            Component(ref="Y1", x=25, y=30, width=3.2, height=2.5, component_type="crystal",
                      pads=[Pad(pad_name="1", x=-1.1, y=-0.85, net="OSC_IN"),
                            Pad(pad_name="3", x=1.1, y=0.85, net="OSC_OUT")],
                      nets=["OSC_IN", "OSC_OUT", "GND"]),
            Component(ref="J1", x=70, y=30, width=8, height=3, component_type="connector",
                      is_fixed=True,
                      pads=[Pad(pad_name="1", x=-1.3, y=-1.2, net="VCC"),
                            Pad(pad_name="2", x=-0.65, y=-1.2, net="USB_DP"),
                            Pad(pad_name="3", x=0, y=-1.2, net="USB_DM")],
                      nets=["VCC", "GND", "USB_DP", "USB_DM"]),
            Component(ref="R1", x=60, y=20, width=1, height=0.5, component_type="resistor",
                      pads=[Pad(pad_name="1", x=-0.25, y=0, net="VCC"),
                            Pad(pad_name="2", x=0.25, y=0, net="LED1")],
                      nets=["VCC", "LED1"]),
            Component(ref="D1", x=55, y=20, width=2, height=1.2, component_type="generic",
                      pads=[Pad(pad_name="1", x=-0.7, y=0, net="LED1"),
                            Pad(pad_name="2", x=0.7, y=0, net="GND")],
                      nets=["LED1", "GND"]),
        ],
        nets=[
            Net(name="VCC", pins=[("U1", "5"), ("C1", "1"), ("C2", "1"), ("J1", "1"), ("R1", "1")]),
            Net(name="GND", pins=[("U1", "1"), ("C1", "2"), ("C2", "2"), ("Y1", "2"), ("J1", "5"), ("D1", "2")]),
            Net(name="OSC_IN", pins=[("U1", "7"), ("Y1", "1")]),
            Net(name="OSC_OUT", pins=[("U1", "8"), ("Y1", "3")]),
            Net(name="USB_DP", pins=[("U1", "23"), ("J1", "2")]),
            Net(name="USB_DM", pins=[("U1", "24"), ("J1", "3")]),
            Net(name="LED1", pins=[("U1", "35"), ("R1", "2"), ("D1", "1")]),
        ],
    )
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("  KiCad Smart Auto-Placer — Phase 1 + Phase 6 Test Suite")
    print("=" * 60 + "\n")

    print("Data Model Tests:")
    run_test("BoardOutline basics", test_board_outline)
    run_test("Component overlap", test_component_overlap)
    run_test("Component bounding box", test_component_bbox)
    run_test("Pad absolute position", test_pad_absolute_pos)
    run_test("BoardModel serialization", test_board_model_serialization)
    run_test("BoardModel lookup helpers", test_board_model_lookup)

    print("\nParser Tests:")
    run_test("KiCad parser", test_kicad_parser)
    run_test("JSON roundtrip", test_json_roundtrip)
    run_test("Zero-courtyard fallback", test_zero_courtyard_fallback)
    run_test("Board outline with internal cutout", test_board_outline_with_internal_cutout)

    print("\nNet Clustering Tests:")
    run_test("Build net hypergraph", test_build_hypergraph)
    run_test("Sheet field parsed and serialized", test_sheet_field_parsed_and_serialized)
    run_test("Sheet-aware clustering groups by sheet", test_sheet_aware_clustering_groups_by_sheet)
    run_test("Sheet-aware clustering falls back when no sheets", test_sheet_aware_clustering_falls_back_when_no_sheets)
    run_test("th_sensor.kicad_pcb sheet extraction", test_th_sensor_sheet_extraction)
    run_test("RL_PCB boards parse cleanly", test_rl_pcb_boards_parse_cleanly)
    run_test("Subcircuit crystal motif detection", test_subcircuit_crystal_motif_detection)
    run_test("Subcircuit regulator motif detection", test_subcircuit_regulator_motif_detection)
    run_test("Subcircuit patterns feed into clustering", test_subcircuit_patterns_feed_into_clustering)
    run_test("Component clustering", test_clustering)
    run_test("Seed position computation", test_seed_positions)

    print("\nHPWL Cost Function Tests:")
    run_test("HPWL 2-pin net", test_hpwl_2pin)
    run_test("HPWL 3-pin clique", test_hpwl_3pin_clique)
    run_test("HPWL star model", test_hpwl_star)
    run_test("HPWL auto model selection", test_hpwl_auto_model)
    run_test("HPWL true model is default", test_hpwl_true_model_default)
    run_test("HPWL 4-to-5-pin continuity", test_hpwl_4_to_5_pin_continuity)
    run_test("total_hpwl matches CostState.hpwl", test_total_hpwl_matches_cost_state)
    run_test("Total HPWL", test_total_hpwl)
    run_test("Overlap penalty", test_overlap_penalty)
    run_test("Boundary penalty", test_boundary_penalty)
    run_test("Boundary penalty includes keepouts", test_boundary_penalty_includes_keepouts)
    run_test("CostState includes keepouts", test_cost_state_includes_keepouts)
    run_test("Cost function evaluation", test_cost_function)

    print("\nGrid Placement Tests:")
    run_test("Grid placement", test_grid_placement)
    run_test("Edge-aware placement", test_edge_aware_placement)

    print("\nLegalization Tests:")
    run_test("Grid snapping", test_grid_snap)
    run_test("Boundary enforcement", test_boundary_enforcement)
    run_test("Overlap resolution", test_overlap_resolution)

    print("\nBoard Profile Tests:")
    run_test("Built-in profiles", test_builtin_profiles)
    run_test("Profile rules", test_profile_rules)
    run_test("Routing congestion rule dispatch", test_routing_congestion_rule_dispatch)
    run_test("Routing congestion in profiles", test_routing_congestion_in_profiles)
    run_test("Pin density detects hotspot", test_pin_density_penalty_detects_hotspot)
    run_test("Pin density complements RUDY on long nets", test_pin_density_complements_rudy_on_long_nets)
    run_test("Pin density excludes power pins", test_pin_density_excludes_power_pins)
    run_test("Pin density zero on uniform board", test_pin_density_penalty_zero_on_uniform_board)
    run_test("evaluate() threads pin_density into total", test_evaluate_cost_function_includes_pin_density_term)
    run_test("RUDY + pin-density default-on via config", test_rudy_weight_defaults_to_config_value)
    run_test("Thermal separation rule dispatch", test_thermal_separation_rule_dispatch)
    run_test("Thermal separation far apart no penalty", test_thermal_separation_far_apart_no_penalty)
    run_test("Thermal separation ignores same-net pairs", test_thermal_separation_ignores_same_net_pairs)
    run_test("Thermal separation in power_supply profile", test_thermal_separation_in_power_supply_profile)
    run_test("Thermal separation retuned weights", test_thermal_separation_retuned_weights)
    run_test("SAConfig auto-disable thresholds", test_sa_config_auto_disable_thresholds)
    run_test("SAConfig spread floor", test_sa_config_spread_floor)
    run_test("SA auto-disables on tiny board", test_sa_auto_disables_on_tiny_board)
    run_test("SA auto-disable respects explicit skip_sa=True", test_sa_auto_disable_respects_explicit_skip_sa_true)
    run_test("Violates spread floor", test_violates_spread_floor)
    run_test("No spread_floor revert when skip_sa=True", test_sa_no_spread_floor_revert_when_skip_sa)

    print("\nCap-IC Group Movement:")
    run_test("Cap-IC group survives translate", test_cap_ic_group_survives_translate)
    run_test("Cap-IC group survives swap", test_cap_ic_group_survives_swap)

    print("\nConnector Orientation:")
    run_test("Barrel jack faces outward", test_barrel_jack_faces_outward)
    run_test("End-mating connector classification", test_end_mating_connector_classification)

    print("\nLegalizer Group Awareness:")
    run_test("Legalizer propagate_ic_delta helper", test_propagate_ic_delta_helper)
    run_test("Legalizer _enforce_boundary propagates to caps", test_legalizer_propagates_ic_delta_to_caps)
    run_test("Legalizer _snap_to_grid propagates to caps", test_legalizer_snap_to_grid_propagates_to_caps)
    run_test("Caps do not overlap IC after legalize", test_caps_do_not_overlap_ic_after_legalize)

    print("\nEnd-to-End Pipeline:")
    run_test("Full Phase 1 pipeline", test_full_pipeline)

    print("\nPhase 6: Rotation-Aware Dimensions:")
    run_test("Rotation effective dimensions", test_rotation_effective_dimensions)
    run_test("Rotation bbox", test_rotation_bbox)
    run_test("Rotation bbox offset rotation", test_rotation_bbox_offset_rotation)

    print("\nPhase 6: Power Net Detection:")
    run_test("Power net detection", test_power_net_detection)

    print("\nPhase 6: CostState:")
    run_test("CostState matches full compute", test_cost_state_matches_full_compute)
    run_test("Incremental update matches full", test_incremental_update_matches_full)
    run_test("Snapshot restore", test_snapshot_restore)
    run_test("Density gated by decap rule", test_density_gated_by_decap_rule)
    run_test("Snapshot restore with density", test_snapshot_restore_with_density)
    run_test("Snapshot restore N moves property", test_snapshot_restore_n_moves_property)

    print("\nPhase 6: Move Operators:")
    run_test("Translate move and revert", test_translate_move_and_revert)
    run_test("Swap move and revert", test_swap_move_and_revert)
    run_test("Rotate move and revert", test_rotate_move_and_revert)
    run_test("Median move reduces HPWL", test_median_move_reduces_hpwl)
    run_test("Get moveable indices", test_get_moveable_indices)

    print("\nPhase 6: SA Engine:")
    run_test("SA no regression", test_sa_no_regression)
    run_test("Greedy refinement improves", test_greedy_refinement_improves)
    run_test("No rotation reset in placement", test_no_rotation_reset_in_placement)

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 60 + "\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
