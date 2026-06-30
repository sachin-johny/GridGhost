"""Pure-Python undirected graph + Louvain community detection.

This is a minimal drop-in replacement for the subset of the
``networkx`` API that ``engine/net_clustering.py`` actually uses.

Supported API (mirrors networkx):
    - ``Graph()`` constructor
    - ``G.add_node(n, **attrs)`` / ``G.add_edge(u, v, weight=...)``
    - ``G.has_node(n)`` / ``G.has_edge(u, v)``
    - ``G.nodes()`` / ``G.edges()``
    - ``G.degree(n)`` / ``G.degree(n, weight="weight")``
    - ``G[u]`` (adjacency dict) and ``G[u][v]`` (edge-attr dict)
    - ``louvain_communities(G, weight="weight")`` -> list[set[node]]

The graph stores edge attributes in a single shared dict per edge so
that ``G[u][v]`` and ``G[v][u]`` return the *same* dict — updating one
side is immediately visible on the other, matching networkx semantics.

The Louvain implementation is the standard two-phase algorithm:
  1. Local moving: each node tries moving to a neighbor's community
     to maximize modularity gain.
  2. Aggregation: communities collapse into super-nodes (with
     self-loops for intra-community edges), and we repeat phase 1.

The implementation is deterministic given a fixed ``seed``.
"""

from __future__ import annotations

import random
from typing import Any, Iterator


class Graph:
    """Minimal undirected weighted graph.

    Nodes can be any hashable.  Edge attribute dicts are shared
    between the two endpoints so ``G[u][v] is G[v][u]``.
    """

    __slots__ = ("_nodes", "_adj")

    def __init__(self) -> None:
        # node -> attribute dict (currently unused for lookups but stored
        # to mirror networkx behavior where nodes can carry attributes).
        self._nodes: dict[Any, dict] = {}
        # node -> {neighbor: edge_attr_dict}
        self._adj: dict[Any, dict] = {}

    # ------------------------------------------------------------------
    # Node ops
    # ------------------------------------------------------------------

    def add_node(self, n: Any, **attrs: Any) -> None:
        if n not in self._nodes:
            self._nodes[n] = {}
            self._adj[n] = {}
        if attrs:
            self._nodes[n].update(attrs)

    def has_node(self, n: Any) -> bool:
        return n in self._nodes

    def nodes(self):
        """Return a list of nodes (supports len() and iteration)."""
        return list(self._nodes)

    def __contains__(self, n: Any) -> bool:
        return n in self._nodes

    def __iter__(self) -> Iterator:
        return iter(self._nodes)

    def __len__(self) -> int:
        return len(self._nodes)

    # ------------------------------------------------------------------
    # Edge ops
    # ------------------------------------------------------------------

    def add_edge(self, u: Any, v: Any, weight: float = 1.0, **attrs: Any) -> None:
        self.add_node(u)
        self.add_node(v)
        attrs.setdefault("weight", weight)
        # Shared attr dict so G[u][v] is G[v][u]
        self._adj[u][v] = attrs
        self._adj[v][u] = attrs

    def has_edge(self, u: Any, v: Any) -> bool:
        adj_u = self._adj.get(u)
        if adj_u is None:
            return False
        return v in adj_u

    def edges(self):
        """Return a list of (u, v) edge tuples (supports len() and iteration)."""
        seen: set = set()
        out: list[tuple[Any, Any]] = []
        for u, neighbors in self._adj.items():
            for v in neighbors:
                key = (u, v) if u <= v else (v, u)
                if key in seen:
                    continue
                seen.add(key)
                out.append(key)
        return out

    def degree(self, n: Any, weight: str | None = None) -> float:
        adj_n = self._adj.get(n)
        if adj_n is None:
            return 0
        if weight is None:
            return len(adj_n)
        return sum(attrs.get(weight, 1.0) for attrs in adj_n.values())

    # ------------------------------------------------------------------
    # Adjacency access — G[u] returns the neighbor dict, G[u][v] returns
    # the shared edge-attr dict.  This mirrors networkx's dict-of-dicts.
    # ------------------------------------------------------------------

    def __getitem__(self, n: Any) -> dict:
        return self._adj[n]


# ---------------------------------------------------------------------------
# Louvain community detection
# ---------------------------------------------------------------------------

def louvain_communities(
    G: Graph,
    weight: str = "weight",
    seed: int = 42,
) -> list[set]:
    """Detect communities using the Louvain algorithm.

    Returns a list of sets, each containing the original node identifiers
    that belong to that community.
    """
    rng = random.Random(seed)

    nodes = list(G.nodes())
    if not nodes:
        return []
    if len(nodes) == 1:
        return [{nodes[0]}]

    # Build a normalized adjacency representation: adj[u] = {v: w}
    # with the SAME weight in both directions, no self-loops.
    adj: dict[Any, dict[Any, float]] = {u: {} for u in nodes}
    for u in nodes:
        for v in G[u]:
            if v == u:
                continue
            w = G[u][v].get(weight, 1.0)
            adj[u][v] = w
            # Make sure v's side has it too (in case the edge dict wasn't
            # perfectly symmetric — shouldn't happen with our Graph, but
            # be defensive).
            if u not in adj[v]:
                adj[v][u] = w

    # Self-loop weight per node (initially 0 for the raw graph).
    self_loop: dict[Any, float] = {u: 0.0 for u in nodes}

    # Track which original nodes each "current node" represents.
    members: dict[Any, set] = {u: {u} for u in nodes}

    # m = total edge weight (each undirected edge counted once).
    m = sum(w for nbrs in adj.values() for w in nbrs.values()) / 2.0
    if m <= 0:
        return [{u} for u in nodes]

    # Iteratively run local-move then aggregate until no improvement.
    while True:
        comm = _louvain_one_level(adj, self_loop, m, rng)
        if comm is None:
            break
        adj, self_loop, members = _louvain_aggregate(adj, self_loop, members, comm)

    return list(members.values())


def _louvain_one_level(
    adj: dict,
    self_loop: dict,
    m: float,
    rng: random.Random,
) -> dict | None:
    """Try to improve modularity by moving nodes between communities.

    Returns: ``{node: community_id}`` if any move was made, else ``None``.
    """
    nodes = list(adj.keys())
    rng.shuffle(nodes)

    comm = {u: u for u in nodes}
    # k_i = degree of node = sum of edge weights + 2 * self_loop (self-loop counts twice)
    degree = {u: sum(adj[u].values()) + 2.0 * self_loop[u] for u in nodes}
    sigma_tot = dict(degree)  # sum of degrees in each community (initially each node is own comm)

    any_moved = False
    improved = True
    # The gain formula (with u removed from its current community):
    #   ΔQ_move_u_to_C = (k_i_in_C / m) - (sigma_tot_C * k_i) / (2 * m^2)
    # where k_i_in_C = sum of edge weights from u to nodes in C.
    # The "stay" gain (C = current community, after removing u) is the baseline.
    while improved:
        improved = False
        for u in nodes:
            cu = comm[u]
            k_i = degree[u]

            # Sum of edge weights from u into each neighboring community.
            k_i_in: dict = {}
            for v, w in adj[u].items():
                cv = comm[v]
                k_i_in[cv] = k_i_in.get(cv, 0.0) + w

            # Tentatively remove u from its current community.
            sigma_tot[cu] -= k_i

            # Baseline: stay in cu (after removal).
            best_gain = k_i_in.get(cu, 0.0) / m - (sigma_tot[cu] * k_i) / (2.0 * m * m)
            best_comm = cu

            for c, k_i_in_c in k_i_in.items():
                if c == cu:
                    continue
                gain = k_i_in_c / m - (sigma_tot[c] * k_i) / (2.0 * m * m)
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_comm = c

            # Put u back into the chosen community.
            sigma_tot[best_comm] += k_i
            if best_comm != cu:
                comm[u] = best_comm
                improved = True
                any_moved = True

    return comm if any_moved else None


def _louvain_aggregate(
    adj: dict,
    self_loop: dict,
    members: dict,
    comm: dict,
) -> tuple[dict, dict, dict]:
    """Collapse nodes in the same community into a single super-node."""
    unique_comms = list(set(comm.values()))
    # Use the community id (which is some node id from the previous level)
    # as the super-node id directly — keeps node labels meaningful.
    # However, multiple communities may share the same id-type (e.g. ints)
    # and collide. To be safe, relabel to 0..N-1.
    comm_to_sn = {c: i for i, c in enumerate(unique_comms)}
    n_sn = len(unique_comms)

    new_adj: dict = {i: {} for i in range(n_sn)}
    new_self_loop: dict = {i: 0.0 for i in range(n_sn)}
    new_members: dict = {i: set() for i in range(n_sn)}

    # Aggregate self-loops and members.
    for u, c in comm.items():
        sn = comm_to_sn[c]
        new_self_loop[sn] += self_loop[u]
        new_members[sn].update(members[u])

    # Aggregate edges — visit each undirected edge only once.
    seen: set = set()
    for u in adj:
        cu = comm_to_sn[comm[u]]
        for v, w in adj[u].items():
            edge_key = (u, v) if u <= v else (v, u)
            if edge_key in seen:
                continue
            seen.add(edge_key)
            cv = comm_to_sn[comm[v]]
            if cu == cv:
                new_self_loop[cu] += w
            else:
                if cv in new_adj[cu]:
                    new_adj[cu][cv] += w
                else:
                    new_adj[cu][cv] = w
                if cu in new_adj[cv]:
                    new_adj[cv][cu] += w
                else:
                    new_adj[cv][cu] = w

    return new_adj, new_self_loop, new_members
