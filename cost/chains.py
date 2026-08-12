"""Interior-only signal-flow chain detection for the macro-v2 cost function.

Issue 3 of PLACEMENT_FIX_PLAN.md: bias placement toward routability for
detected linear signal paths (sensor → amp → filter → ADC) by upweighting
the HPWL of nets INTERNAL to a chain.  This is a cost-function nudge (Zhu
et al., ICCAD 2020 — signal flow as a soft penalty, optimized alongside
wirelength/overlap), NOT a rigid geometric constraint — so a bad detection
only slightly mis-weights a few nets instead of producing a permanently
bad seed that SA can't recover from.

Unlike ``engine/subcircuit_patterns._detect_signal_flow_chains`` (legacy),
connectors are NEVER members or endpoints here.  This catches internal
filter stages and fan-out dominant paths that don't touch the board I/O
directly, and avoids re-pulling endpoints toward the perimeter (which the
connector placement already handles).

Chain edges come only from strong, low-fanout signal nets (clique weight
≥ 1/3, i.e. ≤ 4 pins) — a shared bus or rail can't masquerade as a signal
path, the same per-net clique reasoning that fixes Issue 1's collapse.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import TYPE_CHECKING

from engine.cost_state import _is_power_net

if TYPE_CHECKING:
    from models.board_model import BoardModel


# Cap chain depth so a densely-connected board doesn't produce one giant
# "chain" that's really the whole netlist (matches the legacy detector).
_MAX_CHAIN_DEPTH = 8
# A chain needs ≥3 interior members (A→B→C, two edges) to be a "path".
_MIN_CHAIN_LEN = 3
# Only ≤ this many pins form a chain edge — strong, point-to-point-ish signal
# nets.  Buses/rails (high fan-out) are weak and never chain.
_MAX_EDGE_FANOUT = 4
# HPWL multiplier for chain-internal nets.  Legacy pipeline used 3-4×, but on
# macro-v2's cost scale (alpha=1.0) 3.0 over-pulls chain members into courtyard
# overlaps on dense boards (cbb: 4→10 overlaps). Empirical sweep on cbb:
#   2.0× → 8% closer chains, no overlap regression (the knee)
#   3.0× → 30% closer chains, +6 overlaps
# 2.0 keeps the routability benefit (chain members measurably closer) without
# the overlap cost — re-tunable.
CHAIN_NET_WEIGHT = 2.0


def _interior_refs(model: "BoardModel") -> set[str]:
    return {
        c.ref for c in model.components
        if not c.is_fixed and c.component_type != "connector"
    }


def _net_info(model: "BoardModel", interior: set[str]) -> dict[str, tuple[set[str], float]]:
    """Map net name → (interior member set, clique weight 1/(k-1)).

    Only non-power signal nets with 2..``_MAX_EDGE_FANOUT`` pins and ≥2
    interior members are included — these are the only nets strong enough
    to form a chain edge.
    """
    info: dict[str, tuple[set[str], float]] = {}
    for net in model.nets:
        if _is_power_net(net.name):
            continue
        k = len(net.pins)
        if k < 2 or k > _MAX_EDGE_FANOUT:
            continue
        members = {r for r in net.component_refs if r in interior}
        if len(members) < 2:
            continue
        info[net.name] = (members, 1.0 / (k - 1))
    return info


def _adjacency(net_info: dict[str, tuple[set[str], float]]) -> dict[str, dict[str, tuple[float, str]]]:
    """node → {neighbor: (strongest clique weight, connecting net name)}."""
    adj: dict[str, dict[str, tuple[float, str]]] = defaultdict(dict)
    for net_name, (members, w) in net_info.items():
        for a, b in combinations(sorted(members), 2):
            if w > adj[a].get(b, (0.0, ""))[0]:
                adj[a][b] = (w, net_name)
                adj[b][a] = (w, net_name)
    return adj


def detect_interior_chains(
    model: "BoardModel",
    *,
    max_depth: int = _MAX_CHAIN_DEPTH,
    min_len: int = _MIN_CHAIN_LEN,
) -> list[list[str]]:
    """Detect linear signal-flow chains among interior components.

    For each interior component with signal-net adjacency, greedily extend a
    path in both directions along the strongest edges (capped at
    ``max_depth``).  Returns the longest, greedily edge-non-overlapping set of
    chains of length ≥ ``min_len``.  Connectors and fixed components are never
    members or endpoints.
    """
    interior = _interior_refs(model)
    if len(interior) < min_len:
        return []

    net_info = _net_info(model, interior)
    if not net_info:
        return []
    adj = _adjacency(net_info)

    def extend_ray(start: str, forbidden: set[str]) -> list[str]:
        path = [start]
        seen = set(forbidden)
        seen.add(start)
        cur = start
        while len(path) < max_depth:
            cands = [
                (w, n) for n, (w, _net) in adj.get(cur, {}).items()
                if n not in seen
            ]
            if not cands:
                break
            cands.sort(reverse=True)
            cur = cands[0][1]
            path.append(cur)
            seen.add(cur)
        return path

    # One bidirectional chain per seed: forward ray from s, then a backward
    # ray from s avoiding the forward path, concatenated through s.
    candidates: list[list[str]] = []
    # sorted() so chain construction is deterministic across PYTHONHASHSEED
    # values (interior is a set — its iteration order is hash-randomized).
    for seed in sorted(interior):
        if seed not in adj:
            continue
        forward = extend_ray(seed, set())
        backward = extend_ray(seed, set(forward[1:]))
        chain = list(reversed(backward[1:])) + forward
        # Cap total chain length at max_depth (both rays can each reach it,
        # so the concatenation can exceed it). Keep a window centered on the
        # seed so neither direction dominates the truncation.
        if len(chain) > max_depth:
            s_idx = len(backward) - 1  # seed position within `chain`
            half = max_depth // 2
            lo = max(0, s_idx - half)
            chain = chain[lo:lo + max_depth]
        if len(chain) >= min_len:
            candidates.append(chain)

    # Greedy select: longest first, drop chains whose edges are all already
    # covered (near-duplicate shifted chains from adjacent seeds).
    candidates.sort(key=len, reverse=True)
    covered: set[frozenset] = set()
    selected: list[list[str]] = []
    for chain in candidates:
        edges = {frozenset((chain[i], chain[i + 1])) for i in range(len(chain) - 1)}
        if edges <= covered:
            continue
        selected.append(chain)
        covered |= edges
    return selected


def build_chain_net_weights(
    model: "BoardModel",
    chains: list[list[str]],
    *,
    weight: float = CHAIN_NET_WEIGHT,
) -> dict[str, float]:
    """Map each chain-internal net → ``weight`` (default ``CHAIN_NET_WEIGHT``).

    A net is chain-internal if it's the strongest connection between two
    consecutive members of a detected chain.  Nets not in any chain are absent
    from the dict (callers apply a default weight of 1.0).
    """
    if not chains:
        return {}
    interior = _interior_refs(model)
    net_info = _net_info(model, interior)

    weights: dict[str, float] = {}
    for chain in chains:
        for i in range(len(chain) - 1):
            a, b = chain[i], chain[i + 1]
            best_net: str | None = None
            best_w = 0.0
            for net_name, (members, w) in net_info.items():
                if a in members and b in members and w > best_w:
                    best_w = w
                    best_net = net_name
            if best_net is not None:
                weights[best_net] = weight
    return weights
