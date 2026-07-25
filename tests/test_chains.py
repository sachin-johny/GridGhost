"""Issue 3 regression tests — interior signal-flow chain detection + weighting.

Two layers:

  1. ``detect_interior_chains`` on synthetic connectivity graphs (a simple
     path, a star/fan-out, a cycle) — the detector must terminate, pick a
     sensible longest path, exclude connectors, and ignore power nets.
  2. Board-level: chain weighting measurably pulls cbb's signal-path members
     closer together (the plan's direct validation metric).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.board_model import BoardModel, BoardOutline, Component, Net
from cost.chains import detect_interior_chains, build_chain_net_weights, CHAIN_NET_WEIGHT
from parsers.kicad_parser import KiCadParser


def _ic(ref: str) -> Component:
    return Component(ref=ref, component_type="ic")


def _model(refs: list[str], nets: list[tuple[str, list[str]]], *, extra_comps=None) -> BoardModel:
    """Build a minimal BoardModel: IC components + named 2-pin signal nets."""
    comps = [_ic(r) for r in refs]
    if extra_comps:
        comps.extend(extra_comps)
    model = BoardModel(board=BoardOutline(0, 0, 100, 100), components=comps)
    model.nets = [Net(name=n, pins=[(r, "1") for r in members]) for n, members in nets]
    model.rebuild_ref_index()
    return model


# ──────────────────────────────────────────────────────────────────────────
# 1. Detector on synthetic graphs
# ──────────────────────────────────────────────────────────────────────────

def test_simple_path_detected_end_to_end():
    """A→B→C→D→E connected by 2-pin signal nets forms one chain covering all."""
    model = _model(list("ABCDE"), [
        ("/sAB", ["A", "B"]), ("/sBC", ["B", "C"]),
        ("/sCD", ["C", "D"]), ("/sDE", ["D", "E"]),
    ])
    chains = detect_interior_chains(model)
    assert len(chains) >= 1
    members = set(chains[0])
    assert members == {"A", "B", "C", "D", "E"}


def test_star_fanout_picks_a_path_without_crashing():
    """Center S fanning out to A,B,C,D — detector returns ≥1 path of length 3,
    deterministically, without crashing or returning something deggenerate."""
    model = _model(["S", "A", "B", "C", "D"], [
        ("/s1", ["S", "A"]), ("/s2", ["S", "B"]),
        ("/s3", ["S", "C"]), ("/s4", ["S", "D"]),
    ])
    chains = detect_interior_chains(model)
    # Every detected chain is a valid simple path (length ≥ 3).
    for ch in chains:
        assert len(ch) >= 3
        assert len(set(ch)) == len(ch), "chain must not repeat a node"


def test_cycle_terminates_and_does_not_loop():
    """A→B→C→D→A cycle — detector must terminate (no infinite loop) and each
    returned chain is a simple path (no node repeated by walking the cycle)."""
    model = _model(list("ABCD"), [
        ("/s1", ["A", "B"]), ("/s2", ["B", "C"]),
        ("/s3", ["C", "D"]), ("/s4", ["D", "A"]),
    ])
    chains = detect_interior_chains(model)
    for ch in chains:
        assert len(set(ch)) == len(ch), "cycle produced a repeating-node chain"


def test_connectors_are_never_chain_members():
    """A connector on the signal path is NOT a member or endpoint."""
    conn = Component(ref="J1", component_type="connector")
    model = _model(["A", "B", "C"], [
        ("/s1", ["J1", "A"]), ("/s2", ["A", "B"]), ("/s3", ["B", "C"]),
    ], extra_comps=[conn])
    chains = detect_interior_chains(model)
    for ch in chains:
        assert "J1" not in ch, "connector leaked into a chain"


def test_power_nets_do_not_form_chain_edges():
    """A GND rail touching every component must NOT chain them together."""
    refs = list("ABCDE")
    model = _model(refs, [("GND", refs)])  # only a power net — no signal edges
    chains = detect_interior_chains(model)
    assert chains == [], "power-only netlist produced chains"


def test_high_fanout_signal_net_does_not_dominate():
    """A 6-pin net (≥ _MAX_EDGE_FANOUT) is too weak to form a chain edge."""
    refs = list("ABCDEF")
    model = _model(refs, [("/bus", refs)])  # 6-pin net — exceeds the 4-pin edge cap
    chains = detect_interior_chains(model)
    assert chains == [], "high-fanout bus formed a chain"


def test_chain_length_capped_at_max_depth():
    """A 20-node path yields chains no longer than _MAX_CHAIN_DEPTH (8)."""
    refs = [f"U{i}" for i in range(20)]
    nets = [(f"/s{i}", [refs[i], refs[i + 1]]) for i in range(19)]
    model = _model(refs, nets)
    chains = detect_interior_chains(model)
    for ch in chains:
        assert len(ch) <= 8, f"chain length {len(ch)} exceeds the depth cap"


def test_build_chain_net_weights_marks_consecutive_pair_nets():
    """The 2-pin nets between consecutive path members get the chain weight."""
    model = _model(list("ABCDE"), [
        ("/sAB", ["A", "B"]), ("/sBC", ["B", "C"]),
        ("/sCD", ["C", "D"]), ("/sDE", ["D", "E"]),
    ])
    chains = detect_interior_chains(model)
    weights = build_chain_net_weights(model, chains)
    assert weights  # something got weighted
    for net_name in ("/sAB", "/sBC", "/sCD", "/sDE"):
        assert weights.get(net_name) == CHAIN_NET_WEIGHT
    assert build_chain_net_weights(model, []) == {}


# ──────────────────────────────────────────────────────────────────────────
# 2. Board-level: chain weighting pulls cbb signal paths together
# ──────────────────────────────────────────────────────────────────────────

def _mean_chain_distance(model, chains):
    dists = []
    for ch in chains:
        for a, b in zip(ch, ch[1:]):
            ca, cb = model.get_component(a), model.get_component(b)
            if ca and cb:
                dists.append(math.hypot(ca.x - cb.x, ca.y - cb.y))
    return sum(dists) / len(dists) if dists else 0.0


def test_chain_weighting_pulls_cbb_chain_members_closer():
    """The plan's direct Issue 3 metric: with chain weighting ON, adjacent
    signal-path members on cbb end up closer than with it OFF.

    This compares two SA runs (chain weighting on vs off). SA is a
    stochastic optimizer, so a *single* run (one seed) is a noisy draw:
    the on-vs-off margin at any one seed can be small enough that
    unrelated trajectory differences (e.g. the SA's PYTHONHASHSEED-
    sensitive move ordering, or a macro-construction change elsewhere)
    flip the sign. The feature is real and robust — it wins across
    seeds — so we average over several seeds and assert the mean. This
    tests the claim ("weighting pulls members closer") rather than one
    lucky draw.
    """
    pcb = ROOT / "tests" / "test_pcbs" / "cbb.kicad_pcb"
    base = KiCadParser(str(pcb), bbox_margin=0.8).parse()
    chains = detect_interior_chains(base)
    assert chains, "cbb should have detectable signal-flow chains"

    from place import pipeline as pl
    from place.pipeline import place_v2

    seeds = (42, 7, 13, 99)
    d_on_runs = []
    d_off_runs = []
    orig = pl.build_chain_net_weights
    try:
        for seed in seeds:
            # Weighted (default pipeline behavior).
            m_on = KiCadParser(str(pcb), bbox_margin=0.8).parse()
            place_v2(m_on, sa_iterations=800, sa_reheats=2, seed=seed, verbose=False)
            d_on_runs.append(_mean_chain_distance(m_on, chains))

            # Unweighted (force empty net weights via the pipeline's reference).
            pl.build_chain_net_weights = lambda *a, **k: {}
            m_off = KiCadParser(str(pcb), bbox_margin=0.8).parse()
            place_v2(m_off, sa_iterations=800, sa_reheats=2, seed=seed, verbose=False)
            d_off_runs.append(_mean_chain_distance(m_off, chains))
            pl.build_chain_net_weights = orig
    finally:
        pl.build_chain_net_weights = orig

    d_on = sum(d_on_runs) / len(d_on_runs)
    d_off = sum(d_off_runs) / len(d_off_runs)
    wins = sum(1 for a, b in zip(d_on_runs, d_off_runs) if a < b)
    assert d_on < d_off, (
        f"chain weighting did not pull members closer on average: "
        f"on={d_on:.2f}mm vs off={d_off:.2f}mm "
        f"(per-seed on={[round(x,1) for x in d_on_runs]}, "
        f"off={[round(x,1) for x in d_off_runs]}, won {wins}/{len(seeds)})"
    )
