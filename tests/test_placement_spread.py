"""Issue 1 regression tests — per-net-weighted nudges + interior spread.

Two independent guards for the center-collapse bug in
PLACEMENT_FIX_PLAN.md §"Issue 1":

  1. ``weighted_attractor_target`` unit test (synthetic nets) — a 20-pin GND
     net must contribute the same order of weight as a 2-pin signal net, so a
     cluster wired to one dedicated connector (plus shared GND) is pulled
     toward ITS connector, not the GND centroid ≈ board center.
  2. Board-level spread floor — cbb/cbbwO's interior footprint must stay well
     above the collapsed ~35% level after the full pipeline.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.board_model import BoardModel, BoardOutline, Component, Net
from place.cluster import weighted_attractor_target
from parsers.kicad_parser import KiCadParser
from place.pipeline import place_v2


# ──────────────────────────────────────────────────────────────────────────
# 1. Per-net weighting — the dilution fix (synthetic, no Macros needed)
# ──────────────────────────────────────────────────────────────────────────

def _signal_net(name: str, refs: list[str]) -> Net:
    return Net(name=name, pins=[(r, "1") for r in refs])


def test_high_fanout_gnd_does_not_dominate_signal_net():
    """A 20-pin GND net weighted the same order as a 2-pin signal net.

    Setup: 4 connectors at the 4 board corners. A cluster wired to ONE
    dedicated connector (J0 at corner 0) via a 2-pin signal net, PLUS a
    20-pin GND net spanning every connector. Without per-net weighting the
    attractor centroid is the mean of all 4 corners (board center); with
    clique weighting the dedicated 2-pin signal dominates and the target
    lands near J0.
    """
    corners = [(10.0, 10.0), (90.0, 10.0), (10.0, 90.0), (90.0, 90.0)]
    attractor_positions = {
        "GND": corners,  # 4 connector points on GND
        "/sig0": [corners[0]],  # J0 only
        "/sig1": [corners[1]],
        "/sig2": [corners[2]],
        "/sig3": [corners[3]],
    }
    # GND touches all 4 connectors + 16 interior comps = 20 pins.
    gnd = _signal_net("GND", [f"J{i}" for i in range(4)] + [f"U{i}" for i in range(16)])
    sig0 = _signal_net("/sig0", ["J0", "Ucluster"])
    sig1 = _signal_net("/sig1", ["J1", "Ux"])
    sig2 = _signal_net("/sig2", ["J2", "Uy"])
    sig3 = _signal_net("/sig3", ["J3", "Uz"])

    # Cluster wired to J0 (via sig0) + shares GND.
    nets_for_cluster0 = [gnd, sig0]
    target = weighted_attractor_target(nets_for_cluster0, attractor_positions)
    assert target is not None
    tx, ty = target

    board_center = (50.0, 50.0)
    dist_to_j0 = math.hypot(tx - corners[0][0], ty - corners[0][1])
    dist_to_center = math.hypot(tx - board_center[0], ty - board_center[1])
    # The weighted target must be much closer to the dedicated connector J0
    # than to the board center (which is where the unweighted GND centroid
    # collapses to). This is the direct assertion that GND is damped.
    assert dist_to_j0 < dist_to_center / 3, (
        f"target ({tx:.1f},{ty:.1f}) closer to center than J0 — GND not damped"
    )


def test_per_net_centroid_one_vote_per_net():
    """A high-fan-out net's many attractor points collapse to ONE centroid,
    AND the clique model damps it by fan-out — so 5 points at (80,80) on a
    5-pin net do NOT drag the target the way the old unweighted pool did.

    Old behavior (pool every pin): mean of [80,80,80,80,80,20] = 70 → target
    dragged toward (80,80). New behavior: the 5-pin net contributes its single
    centroid (80,80) at clique weight 1/(5-1)=0.25; the 2-pin net contributes
    (20,20) at weight 1 → (0.25·80 + 1·20)/1.25 = 32.
    """
    many_pts = _signal_net("/many", ["J0", "J1", "J2", "J3", "J4"])
    one_pt = _signal_net("/one", ["J5", "J6"])
    attractor_positions = {
        "/many": [(80.0, 80.0)] * 5,
        "/one": [(20.0, 20.0)],
    }
    target = weighted_attractor_target([many_pts, one_pt], attractor_positions)
    assert target is not None
    tx, ty = target
    assert (tx, ty) == pytest.approx((32.0, 32.0), abs=1e-9)

    # And critically, NOT the old unweighted-pool value where 5 pins at 80
    # out-vote the single pin at 20 → mean 70. That drag-to-(80,80) is the
    # exact collapse mechanism this replaces.
    old_pooled = (5 * 80.0 + 20.0) / 6
    assert abs(tx - old_pooled) > 20, "high-fan-out net still dominating like the old pool"


def test_no_attractor_returns_none():
    """A cluster whose nets have no fixed/connector attractor gets no nudge."""
    net = _signal_net("/sig", ["U1", "U2"])  # no attractor positions
    assert weighted_attractor_target([net], {}) is None


# ──────────────────────────────────────────────────────────────────────────
# 2. Board-level spread floor (collapse guard)
# ──────────────────────────────────────────────────────────────────────────

def _interior_spread(model: BoardModel) -> float:
    interior = [
        c for c in model.components
        if not c.is_fixed and not getattr(c, "is_edge_connector", False)
    ]
    if not interior:
        return 0.0
    xmin = min(c.bbox[0] for c in interior)
    ymin = min(c.bbox[1] for c in interior)
    xmax = max(c.bbox[2] for c in interior)
    ymax = max(c.bbox[3] for c in interior)
    fp = max(0.0, xmax - xmin) * max(0.0, ymax - ymin)
    board_area = model.board.width * model.board.height
    return 100.0 * fp / board_area if board_area > 0 else 0.0


@pytest.mark.parametrize("stem", ["cbb", "cbbwO"])
def test_multi_cluster_boards_do_not_collapse(stem):
    """cbb/cbbwO interior footprint stays well above the collapsed ~35%.

    The bug this catches: every cluster's attractor centroid collapsing to
    the board center, packing 8 clusters into a 12×14mm box (measured 33-39%
    spread).  Post-fix, Phase A fills the interior and the spread clears 55%.
    Floor is set generously below the observed ~64-69% so this is a collapse
    guard, not a tight metric.
    """
    pcb = ROOT / "tests" / "test_pcbs" / f"{stem}.kicad_pcb"
    model = KiCadParser(str(pcb), bbox_margin=0.8).parse()
    place_v2(model, sa_iterations=800, sa_reheats=2, seed=42, verbose=False)
    spread = _interior_spread(model)
    assert spread > 55.0, f"{stem} interior spread collapsed to {spread:.1f}% (<55%)"


@pytest.mark.parametrize("stem", ["cbb", "cbbwO", "test4", "test6"])
def test_no_out_of_bounds_after_pipeline(stem):
    """Interior components stay on the board after the full pipeline."""
    pcb = ROOT / "tests" / "test_pcbs" / f"{stem}.kicad_pcb"
    model = KiCadParser(str(pcb), bbox_margin=0.8).parse()
    place_v2(model, sa_iterations=800, sa_reheats=2, seed=42, verbose=False)
    oob = 0
    for c in model.components:
        if getattr(c, "is_edge_connector", False):
            continue
        x1, y1, x2, y2 = c.bbox
        if (x1 < model.board.x_min or y1 < model.board.y_min or
                x2 > model.board.x_max or y2 > model.board.y_max):
            oob += 1
    assert oob == 0, f"{stem}: {oob} interior components out of bounds"
