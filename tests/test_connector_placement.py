"""Issue 2 regression tests — connector edge centering via set_bbox_center.

Covers the two bugs in PLACEMENT_FIX_PLAN.md §"Issue 2":
  1. ``set_bbox_center`` lands the bbox center exactly on target at every
     rotation (the unit test that would have caught the origin-vs-bbox-center
     conflation directly).
  2. On real boards, each edge's connector group is centered on the true
     edge midpoint (along-edge) and flush on the board edge (perpendicular).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.board_model import Component, BoardOutline, rotated_bbox_offset
from parsers.kicad_parser import KiCadParser
from place.connectors import place_connectors_perimeter


# ──────────────────────────────────────────────────────────────────────────
# 1. set_bbox_center unit test (all rotations, asymmetric bbox_offset)
# ──────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("rotation", [0.0, 90.0, 180.0, 270.0, 45.0, 33.0])
def test_set_bbox_center_places_bbox_center_on_target(rotation):
    """After set_bbox_center(cx, cy, rot), the bbox center is exactly (cx, cy)."""
    comp = Component(
        ref="J1", width=10.0, height=6.0, courtyard_margin=0.25,
        bbox_offset_x=4.38, bbox_offset_y=6.35,  # asymmetric, PinHeader-like
    )
    target = (137.0, 91.5)
    comp.set_bbox_center(target[0], target[1], rotation)
    bx1, by1, bx2, by2 = comp.bbox
    cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
    assert math.hypot(cx - target[0], cy - target[1]) < 1e-9


def test_rotated_bbox_offset_matches_plan_cbb_pinheader():
    """rotated_bbox_offset(4.38, 6.35, 180) == (-4.38, -6.35) per plan analysis."""
    dx, dy = rotated_bbox_offset(4.38, 6.35, 180.0)
    assert (dx, dy) == pytest.approx((-4.38, -6.35), abs=1e-9)


def test_set_bbox_center_matches_bbox_at_round_trip():
    """set_bbox_center then bbox == bbox_at at the same pose (single source of truth)."""
    comp = Component(
        ref="J2", width=8.0, height=8.0, courtyard_margin=0.3,
        bbox_offset_x=5.38, bbox_offset_y=0.0,  # SMA-like, offset only in local X
    )
    for rot in (0.0, 90.0, 180.0, 270.0):
        comp.set_bbox_center(100.0, 80.0, rot)
        # bbox_at at the resulting origin/rotation must equal the live bbox
        assert comp.bbox_at(comp.x, comp.y, comp.rotation) == pytest.approx(
            comp.bbox, abs=1e-9
        )


# ──────────────────────────────────────────────────────────────────────────
# 2. Board-level edge centering + flush placement
# ──────────────────────────────────────────────────────────────────────────

PCBS_WITH_EDGE_CONNECTORS = ["cbb", "cbbwO"]


def _load(stem: str):
    pcb = ROOT / "tests" / "test_pcbs" / f"{stem}.kicad_pcb"
    parser = KiCadParser(str(pcb), bbox_margin=0.8)
    return parser.parse()


def _classify_edge(comp, board) -> str:
    bx1, by1, bx2, by2 = comp.bbox
    cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
    d = {
        "bottom": board.y_max - cy, "top": cy - board.y_min,
        "right": board.x_max - cx, "left": cx - board.x_min,
    }
    return min(d, key=d.get)


@pytest.mark.parametrize("stem", PCBS_WITH_EDGE_CONNECTORS)
def test_edge_groups_centered_and_flush(stem):
    """Every edge's connector group is centered (≤0.5mm) and flush (≤0.5mm)."""
    model = _load(stem)
    board = model.board
    connectors = [
        c for c in model.components
        if getattr(c, "component_type", "") == "connector"
        and not getattr(c, "is_fixed", False)
        and getattr(c, "is_edge_connector", False)
    ]
    if not connectors:
        pytest.skip(f"{stem} has no edge connectors")

    place_connectors_perimeter(model, connectors, board, margin=5.0, mating_margin=5.0)

    by_edge: dict[str, list[Component]] = {}
    for c in connectors:
        by_edge.setdefault(_classify_edge(c, board), []).append(c)

    mid_x = (board.x_min + board.x_max) / 2
    mid_y = (board.y_min + board.y_max) / 2
    mating_margin = 5.0

    for edge, comps in by_edge.items():
        # Along-edge centering: group span midpoint vs true edge midpoint.
        if edge in ("bottom", "top"):
            lo = min(c.bbox[0] for c in comps)
            hi = max(c.bbox[2] for c in comps)
            group_mid = (lo + hi) / 2
            true_mid = mid_x
        else:
            lo = min(c.bbox[1] for c in comps)
            hi = max(c.bbox[3] for c in comps)
            group_mid = (lo + hi) / 2
            true_mid = mid_y
        assert abs(group_mid - true_mid) < 0.5, (
            f"{stem} edge={edge} group off-center by {abs(group_mid - true_mid):.2f}mm"
        )

        # Perpendicular: edge connectors OVERHANG — the inward bbox face sits
        # ``mating_margin`` inside the board (pads on the board), body extends
        # outward past the outline. Assert the inward face position.
        for c in comps:
            bx1, by1, bx2, by2 = c.bbox
            if edge == "bottom":
                inward, expected = by1, board.y_max - mating_margin
            elif edge == "top":
                inward, expected = by2, board.y_min + mating_margin
            elif edge == "right":
                inward, expected = bx1, board.x_max - mating_margin
            else:  # left
                inward, expected = bx2, board.x_min + mating_margin
            assert abs(inward - expected) < 0.5, (
                f"{stem} {c.ref} edge={edge} inward face {inward:.2f} "
                f"!= {expected:.2f} (diff {abs(inward - expected):.2f}mm)"
            )
