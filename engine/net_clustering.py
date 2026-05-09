"""Net-based clustering for seeded initial placement.

Implements Phase 0 seeding strategy from the plan:
  1. Build net hypergraph
  2. Cluster by shared nets (greedy)
  3. Assign cluster centroids to board regions
  4. Return as initial positions for optimizer
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

import networkx as nx

from models.board_model import BoardModel, Component, BoardOutline


from engine.cost_state import _is_power_net


def build_net_hypergraph(model: BoardModel) -> nx.Graph:
    """Build a weighted graph where nodes are components and edges represent shared nets.

    Edge weight = number of shared nets between two components.
    This captures connectivity intensity — components sharing many nets
    should be placed close together.

    Power/ground nets are excluded because they connect nearly everything,
    destroying cluster structure (would merge 60/68 components into one cluster).
    """
    G = nx.Graph()

    # Add all movable components as nodes
    for comp in model.components:
        if not comp.is_fixed:
            G.add_node(comp.ref, component=comp)

    # Add edges for shared nets (skip power nets)
    for net in model.nets:
        # Skip power nets — they connect everything and destroy cluster structure
        if _is_power_net(net.name):
            continue
        refs = list(net.component_refs)
        # Only consider nets with 2+ movable components
        movable_refs = [r for r in refs if G.has_node(r)]
        if len(movable_refs) < 2:
            continue
        # Add edges between all pairs on this net (clique model for small nets)
        for i, r1 in enumerate(movable_refs):
            for r2 in movable_refs[i + 1:]:
                if G.has_edge(r1, r2):
                    G[r1][r2]["weight"] += 1
                else:
                    G.add_edge(r1, r2, weight=1)

    return G


def cluster_components(
    model: BoardModel,
    n_clusters: Optional[int] = None,
    method: str = "greedy",
) -> list[list[str]]:
    """Cluster components by net connectivity.

    Args:
        model: Board model with components and nets
        n_clusters: Target number of clusters. If None, auto-calculate
                    based on component count.
        method: Clustering method ("greedy" or "louvain")

    Returns:
        List of clusters, each cluster is a list of component refs.
    """
    G = build_net_hypergraph(model)
    movable_refs = [c.ref for c in model.components if not c.is_fixed]

    if not movable_refs:
        return []

    if len(movable_refs) == 1:
        return [movable_refs]

    # Auto-calculate cluster count
    if n_clusters is None:
        # Rough heuristic: sqrt of component count, min 2, max 10
        n_clusters = max(2, min(10, int(math.sqrt(len(movable_refs)))))

    # Handle isolated nodes (components with no shared nets)
    isolated = [n for n in G.nodes() if G.degree(n) == 0]

    if method == "louvain" and len(G.edges()) > 0:
        clusters = _louvain_clustering(G, n_clusters)
    else:
        clusters = _greedy_clustering(G, n_clusters)

    # Add isolated components to the smallest cluster
    for ref in isolated:
        if clusters:
            smallest = min(clusters, key=len)
            smallest.append(ref)
        else:
            clusters.append([ref])

    return clusters


def _greedy_clustering(G: nx.Graph, n_clusters: int) -> list[list[str]]:
    """Greedy clustering: seed clusters from highest-degree nodes,
    then assign each remaining node to the cluster it shares the most edges with."""
    nodes = list(G.nodes())
    if not nodes:
        return []

    n_clusters = min(n_clusters, len(nodes))

    # Seed: pick highest-degree nodes
    degree_sorted = sorted(nodes, key=lambda n: G.degree(n, weight="weight"), reverse=True)
    seeds = degree_sorted[:n_clusters]

    clusters: list[list[str]] = [[seed] for seed in seeds]
    assigned = set(seeds)

    # Assign remaining nodes greedily by edge weight
    remaining = [n for n in nodes if n not in assigned]
    # Sort by degree descending so well-connected nodes get better placements
    remaining.sort(key=lambda n: G.degree(n, weight="weight"), reverse=True)

    for node in remaining:
        best_cluster = 0
        best_weight = -1
        for i, cluster in enumerate(clusters):
            weight = sum(G[node][c].get("weight", 1) for c in cluster if G.has_edge(node, c))
            if weight > best_weight:
                best_weight = weight
                best_cluster = i
        clusters[best_cluster].append(node)

    return clusters


def _louvain_clustering(G: nx.Graph, n_clusters: int) -> list[list[str]]:
    """Community detection using Louvain method via networkx."""
    try:
        communities = nx.community.louvain_communities(G, weight="weight")
    except AttributeError:
        # Fallback for older networkx
        communities = _greedy_clustering(G, n_clusters)
        return communities

    # If too many communities, merge smallest
    while len(communities) > n_clusters:
        # Merge two smallest
        communities.sort(key=len)
        merged = communities[0] | communities[1]
        communities = [merged] + communities[2:]

    return [list(c) for c in communities]


def assign_cluster_positions(
    model: BoardModel,
    clusters: list[list[str]],
) -> dict[str, tuple[float, float]]:
    """Assign each cluster a region on the board, then distribute components within.

    Returns a dict of {component_ref: (x, y)} seed positions.
    """
    board = model.board
    positions: dict[str, tuple[float, float]] = {}

    # Place fixed components at their current positions
    for comp in model.components:
        if comp.is_fixed:
            positions[comp.ref] = (comp.x, comp.y)

    # Distribute cluster centroids evenly across the board
    n_clusters = len(clusters)
    if n_clusters == 0:
        return positions

    # Calculate grid layout for clusters
    cols = max(1, int(math.ceil(math.sqrt(n_clusters))))
    rows = max(1, int(math.ceil(n_clusters / cols)))

    # Usable area with margins
    margin = 5.0  # mm margin from edges
    usable_x_min = board.x_min + margin
    usable_y_min = board.y_min + margin
    usable_x_max = board.x_max - margin
    usable_y_max = board.y_max - margin

    region_w = (usable_x_max - usable_x_min) / cols
    region_h = (usable_y_max - usable_y_min) / rows

    for idx, cluster in enumerate(clusters):
        col = idx % cols
        row = idx // cols

        # Cluster centroid — center of this grid region
        cx = usable_x_min + (col + 0.5) * region_w
        cy = usable_y_min + (row + 0.5) * region_h

        # Distribute components within the cluster around the centroid
        positions.update(
            _distribute_in_region(model, cluster, cx, cy, region_w, region_h)
        )

    return positions


def _distribute_in_region(
    model: BoardModel,
    cluster_refs: list[str],
    cx: float,
    cy: float,
    region_w: float,
    region_h: float,
) -> dict[str, tuple[float, float]]:
    """Distribute components of a cluster within a rectangular region
    around the centroid, using a simple grid arrangement."""
    positions = {}
    n = len(cluster_refs)
    if n == 0:
        return positions

    comp_map = {c.ref: c for c in model.components}

    # Simple grid within region
    cols = max(1, int(math.ceil(math.sqrt(n))))
    rows = max(1, int(math.ceil(n / cols)))

    # Spacing based on average component size
    avg_w = 3.0  # default spacing
    avg_h = 3.0
    if cluster_refs:
        widths = [comp_map[r].effective_width for r in cluster_refs if r in comp_map]
        heights = [comp_map[r].effective_height for r in cluster_refs if r in comp_map]
        if widths:
            avg_w = max(widths) * 1.2  # 20% spacing
            avg_h = max(heights) * 1.2

    spacing_x = max(avg_w, 2.0)
    spacing_y = max(avg_h, 2.0)

    # Start from top-left of region, centered on centroid
    start_x = cx - (cols - 1) * spacing_x / 2.0
    start_y = cy - (rows - 1) * spacing_y / 2.0

    for i, ref in enumerate(cluster_refs):
        col = i % cols
        row = i // cols
        x = start_x + col * spacing_x
        y = start_y + row * spacing_y
        positions[ref] = (x, y)

    return positions


def compute_seed_positions(model: BoardModel) -> dict[str, tuple[float, float]]:
    """Full pipeline: cluster + assign positions.

    This is the main entry point for Phase 1 seeded placement.
    """
    clusters = cluster_components(model)
    positions = assign_cluster_positions(model, clusters)
    return positions
