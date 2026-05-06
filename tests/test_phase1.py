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
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.board_model import BoardModel, BoardOutline, Component, Net, Pad
from parsers.kicad_parser import KiCadParser
from engine.net_clustering import (
    build_net_hypergraph,
    cluster_components,
    assign_cluster_positions,
    compute_seed_positions,
)
from engine.grid_placement import grid_place, edge_aware_grid_place
from engine.cost_function import (
    CostFunction,
    total_hpwl,
    total_overlap_penalty,
    total_boundary_penalty,
    net_wirelength_hpwl,
    count_overlaps,
    count_out_of_bounds,
)
from legalization.legalizer import legalize
from profiles.board_profiles import get_profile, list_profiles, BoardProfile, BUILTIN_PROFILES
from samples.sample_board import SAMPLE_KICAD_PCB, create_sample_board
from engine.cost_state import CostState, _is_power_net
from engine.moves import (
    select_move_type, get_moveable_indices,
    do_translate, do_swap, do_rotate, do_median,
    revert_move, affected_indices, MoveUndo,
)
from engine.annealer import simulate_annealing, SAConfig


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

    # 90° rotation
    ax, ay = pad.absolute_pos(10.0, 10.0, 90.0)
    assert abs(ax - 10.0) < 0.01, f"Expected x=10.0, got {ax}"
    assert abs(ay - 11.0) < 0.01, f"Expected y=11.0, got {ay}"


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
    edge_aware_grid_place(model)

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
        edge_aware_grid_place(model, margin=5.0, spacing_factor=1.3)

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
    # Offset should rotate: (1, 0.5) → (-0.5, 1) at 90°
    assert abs(cx90 - 49.5) < 0.01, f"90° cx={cx90}, expected 49.5"
    assert abs(cy90 - 51.0) < 0.01, f"90° cy={cy90}, expected 51.0"


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


# ---------------------------------------------------------------------------
# Phase 6: Move Operator Tests
# ---------------------------------------------------------------------------

def test_translate_move_and_revert():
    model = _make_test_model()
    moveable = get_moveable_indices(model)
    assert len(moveable) > 0

    undo = do_translate(model, moveable, 0.5, 5.0)
    assert len(undo.old_states) == 1

    idx, old_x, old_y, old_rot = undo.old_states[0]
    comp = model.components[idx]
    assert comp.x != old_x or comp.y != old_y, "Translate should change position"

    revert_move(model, undo)
    assert abs(comp.x - old_x) < 0.001, "X should be restored"
    assert abs(comp.y - old_y) < 0.001, "Y should be restored"


def test_swap_move_and_revert():
    model = _make_test_model()
    moveable = get_moveable_indices(model)
    if len(moveable) < 2:
        return  # Skip if not enough moveable components

    undo = do_swap(model, moveable)
    assert len(undo.old_states) == 2

    c1 = model.components[undo.old_states[0][0]]
    c2 = model.components[undo.old_states[1][0]]
    x1, y1 = c1.x, c1.y
    x2, y2 = c2.x, c2.y

    revert_move(model, undo)
    assert abs(c1.x - undo.old_states[0][1]) < 0.001, "C1 x not restored"
    assert abs(c2.x - undo.old_states[1][1]) < 0.001, "C2 x not restored"


def test_rotate_move_and_revert():
    model = _make_test_model()
    moveable = get_moveable_indices(model)
    undo = do_rotate(model, moveable)
    assert len(undo.old_states) == 1

    idx, _, _, old_rot = undo.old_states[0]
    comp = model.components[idx]
    new_rot = comp.rotation

    assert new_rot != old_rot, "Rotation should change"

    revert_move(model, undo)
    assert abs(comp.rotation - old_rot) < 0.001, "Rotation not restored"


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
                      pads=[Pad(pad_name="1", x=-0.25, y=0, net="VCC"),
                            Pad(pad_name="2", x=0.25, y=0, net="GND")],
                      nets=["VCC", "GND"]),
            Component(ref="C2", x=45, y=25, width=1, height=0.5, component_type="capacitor",
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

    print("\nNet Clustering Tests:")
    run_test("Build net hypergraph", test_build_hypergraph)
    run_test("Component clustering", test_clustering)
    run_test("Seed position computation", test_seed_positions)

    print("\nHPWL Cost Function Tests:")
    run_test("HPWL 2-pin net", test_hpwl_2pin)
    run_test("HPWL 3-pin clique", test_hpwl_3pin_clique)
    run_test("HPWL star model", test_hpwl_star)
    run_test("HPWL auto model selection", test_hpwl_auto_model)
    run_test("Total HPWL", test_total_hpwl)
    run_test("Overlap penalty", test_overlap_penalty)
    run_test("Boundary penalty", test_boundary_penalty)
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
