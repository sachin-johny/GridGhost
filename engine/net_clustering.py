"""Net-based clustering for seeded initial placement.

Implements Phase 0 seeding strategy from the plan:
  1. Build net hypergraph
  2. Cluster by shared nets (greedy + power-rail edges)
  3. Assign cluster centroids to board regions
  4. Return as initial positions for optimizer
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

from engine._pure_graph import Graph, louvain_communities

from models.board_model import BoardModel, Component, BoardOutline


from engine.cost_state import _is_power_net


def build_net_hypergraph(model: BoardModel) -> Graph:
    """Build a weighted graph where nodes are components and edges represent shared nets.

    Edge weight = number of shared nets between two components.
    This captures connectivity intensity — components sharing many nets
    should be placed close together.

    Power/ground nets are excluded from signal edges because they connect
    nearly everything, destroying cluster structure.  However, power-rail
    edges are added separately (with lower weight) so that decoupling caps
    cluster with the ICs that share their power rail — this is the key
    insight from the user's suggestion: use +3V3, +12V labels to link
    caps to their ICs.

    Phase 3.1 — Sheet-aware edges: components sharing a non-empty, non-
    root (`/`) `sheet` field (extracted from KiCad 7+ `sheetname`) get
    a synthetic edge with weight SHEET_EDGE_WEIGHT.  This is the designer's
    own functional grouping, sitting in the file for free — two components
    on the same sheet that happen to share only power rails would
    otherwise end up in different clusters.  Falls back gracefully to
    pure net-based clustering when no component has a hierarchical sheet.
    See AUDIT_PHASE0.md §3.1.
    """
    G = Graph()

    # Add all movable components as nodes
    comp_type_map: dict[str, str] = {}
    for comp in model.components:
        if not comp.is_fixed:
            G.add_node(comp.ref, component=comp)
            comp_type_map[comp.ref] = getattr(comp, 'component_type', 'generic')

    # Add edges for shared signal nets (skip power nets)
    for net in model.nets:
        if _is_power_net(net.name):
            continue
        refs = list(net.component_refs)
        movable_refs = [r for r in refs if G.has_node(r)]
        if len(movable_refs) < 2:
            continue
        for i, r1 in enumerate(movable_refs):
            for r2 in movable_refs[i + 1:]:
                if G.has_edge(r1, r2):
                    G[r1][r2]["weight"] += 1
                else:
                    G.add_edge(r1, r2, weight=1)

    # Add power-rail edges: link decoupling caps to ICs that share
    # the same non-GND power rail.  This prevents caps from becoming
    # isolated nodes (since they typically only connect to VCC/GND
    # and have no signal-net edges).
    _add_power_rail_edges(G, model, comp_type_map)

    # Phase 3.1: Add sheet-aware edges between components on the same
    # hierarchical schematic sheet.  This is the core of the brief's
    # "human-like functional grouping" ask.
    _add_sheet_edges(G, model)

    # Phase 3.2: Add subcircuit-pattern edges (crystal + load caps,
    # regulator + input/output caps, etc.).  These are RIGID sub-groups
    # — a human PCB designer would never scatter the members.  Stronger
    # than sheet edges (2.0) and signal edges (1.0).
    _add_subcircuit_edges(G, model)

    return G


def _add_subcircuit_edges(G: Graph, model: BoardModel) -> None:
    """Add clique edges between components in detected subcircuit motifs.

    Delegates to engine.subcircuit_patterns.add_subcircuit_edges.  Kept
    as a thin wrapper here so build_net_hypergraph stays self-contained
    and the import is lazy (avoids circular import with cost_state).
    """
    from engine.subcircuit_patterns import add_subcircuit_edges as _add
    _add(G, model)


# Weight for sheet-aware clustering edges.  Stronger than a single signal-
# net edge (weight=1) so sheet membership dominates when components share
# only power rails, but weaker than a 3+ shared-signal connection so
# genuine high-fanout signal groups still cluster tightly.  Tunable.
SHEET_EDGE_WEIGHT = 2.0


def _add_sheet_edges(G: Graph, model: BoardModel) -> None:
    """Add synthetic edges between components sharing a non-empty, non-root
    hierarchical sheet.

    The KiCad `sheetname` field records which schematic sheet each
    footprint came from — e.g. "/MCU/", "/POWER/", "/Display/".  This
    is the designer's own functional grouping: components on the same
    sheet are usually a coherent sub-circuit (a regulator + its caps +
    its feedback divider, an MCU + its crystal + its decoupling caps)
    that a human PCB designer would keep together.  Adding a strong
    prior edge means the clustering naturally respects schematic
    organisation while still letting genuine cross-sheet net
    connectivity pull related sheets near each other.

    Components with sheet="" or sheet="/" (root sheet of a flat
    schematic) are skipped — they carry no grouping information.
    """
    # Group movable components by sheet
    sheet_groups: dict[str, list[str]] = defaultdict(list)
    for comp in model.components:
        if comp.is_fixed:
            continue
        sheet = getattr(comp, 'sheet', '') or ''
        # Skip empty and root-sheet ("/") entries — they carry no info.
        if not sheet or sheet == '/':
            continue
        if not G.has_node(comp.ref):
            continue
        sheet_groups[sheet].append(comp.ref)

    # Add edges within each sheet group, but only between pairs that don't
    # already share a signal edge.  Adding a sheet boost on top of an existing
    # signal edge double-counts the connection and collapses tight sub-circuits
    # (e.g. an IC + its decoupling caps) into a single Louvain community that
    # gets placed as a block — observable on th_sensor as 3x HPWL regression
    # and U2 pushed off-board.  Skipping those pairs preserves the intent
    # (help weakly-connected sheet-mates cluster) without disrupting tight ones.
    for sheet in sorted(sheet_groups.keys()):
        refs = sorted(sheet_groups[sheet])
        if len(refs) < 2:
            continue
        for i, r1 in enumerate(refs):
            for r2 in refs[i + 1:]:
                if G.has_edge(r1, r2):
                    continue
                G.add_edge(r1, r2, weight=SHEET_EDGE_WEIGHT)


def _add_power_rail_edges(
    G: Graph,
    model: BoardModel,
    comp_type_map: dict[str, str],
) -> None:
    """Add edges between decoupling caps and ICs sharing the same power rail.

    Non-GND power rails (e.g. +3V3, +12V, VCC) are discriminative: only
    specific ICs and their decoupling caps share each rail.  Adding edges
    with weight < 1.0 gives these connections influence without letting
    them dominate the signal-net structure.

    Fallback: caps that share NO power rail with any IC (but share signal
    nets) also get edges to those ICs.  This handles cases like U3 where
    the cap is connected via signal nets only, or where the power rail
    naming doesn't match the expected patterns.
    """
    ic_types = {'ic', 'mcu', 'regulator'}

    # Track which caps got at least one power-rail edge
    caps_with_power_edge: set[str] = set()

    for net in model.nets:
        if not _is_power_net(net.name):
            continue
        # Exclude GND — every component shares it, no discriminative power
        clean = net.name.lstrip('/').upper()
        if any(clean.startswith(p) for p in ('GND', 'AGND', 'DGND', 'PGND', 'SGND', 'VSS')):
            continue

        refs = list(net.component_refs)
        movable_refs = [r for r in refs if G.has_node(r)]

        ic_refs = [r for r in movable_refs if comp_type_map.get(r) in ic_types]
        cap_refs = [r for r in movable_refs if comp_type_map.get(r) == 'capacitor']

        if not ic_refs or not cap_refs:
            continue

        # Add edges between each cap and each IC on this rail.
        # Weight < 1.0 so power-rail edges are weaker than signal edges,
        # but strong enough to pull caps toward their IC's cluster.
        power_edge_weight = 0.5
        for cap_ref in cap_refs:
            for ic_ref in ic_refs:
                if G.has_edge(cap_ref, ic_ref):
                    G[cap_ref][ic_ref]["weight"] += power_edge_weight
                else:
                    G.add_edge(cap_ref, ic_ref, weight=power_edge_weight)
                caps_with_power_edge.add(cap_ref)

    # Fallback: for caps that got NO power-rail edge, create edges to
    # ICs they share ANY net with (including signal nets).  This handles
    # the U3 case where its decoupling cap isn't on the same named power
    # rail or the cap only connects via GND + a signal net.
    cap_refs_all = [r for r in G.nodes() if comp_type_map.get(r) == 'capacitor']
    ic_refs_all = [r for r in G.nodes() if comp_type_map.get(r) in ic_types]

    # Build net→refs mapping for fallback
    net_members: dict[str, set[str]] = defaultdict(set)
    for net in model.nets:
        for ref in net.component_refs:
            if G.has_node(ref):
                net_members[net.name].add(ref)

    fallback_weight = 0.3  # weaker than power-rail edges
    for cap_ref in cap_refs_all:
        if cap_ref in caps_with_power_edge:
            continue  # already has a power-rail edge, skip

        # Find ICs that share any net with this cap
        for net_name, members in net_members.items():
            if cap_ref not in members:
                continue
            for ic_ref in ic_refs_all:
                if ic_ref not in members:
                    continue
                # Cap and IC share this net — add edge if not already present
                if G.has_edge(cap_ref, ic_ref):
                    G[cap_ref][ic_ref]["weight"] += fallback_weight
                else:
                    G.add_edge(cap_ref, ic_ref, weight=fallback_weight)


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
        n_clusters = max(2, min(10, int(math.sqrt(len(movable_refs)))))

    # Handle isolated nodes (components with no shared nets)
    isolated = [n for n in G.nodes() if G.degree(n) == 0]

    if method == "louvain" and len(G.edges()) > 0:
        clusters = _louvain_clustering(G, n_clusters)
    else:
        clusters = _greedy_clustering(G, n_clusters)

    # Add isolated components to the smallest cluster — but only if they
    # aren't already assigned.  Isolated nodes (degree 0) ARE in G.nodes()
    # so they get assigned by greedy/louvain first; without this check they
    # get added a second time, causing duplicate placements → overlaps.
    already_assigned: set[str] = set()
    for cluster in clusters:
        already_assigned.update(cluster)

    for ref in isolated:
        if ref in already_assigned:
            continue
        if clusters:
            smallest = min(clusters, key=len)
            smallest.append(ref)
        else:
            clusters.append([ref])

    # Post-processing: ensure every cluster with an IC has at least one cap.
    # This handles cases like U3 where the IC's decoupling caps ended up
    # in other clusters due to weaker graph connectivity.
    _ensure_ic_has_caps(clusters, model)

    return clusters


def _ensure_ic_has_caps(
    clusters: list[list[str]],
    model: BoardModel,
) -> None:
    """Post-processing: move one cap to each IC cluster that has no caps.

    For each cluster containing an IC but no capacitor, find the nearest
    cap-rich cluster (one with multiple caps) and transfer one cap.
    This ensures every IC has at least one decoupling cap nearby.
    """
    ic_types = {'ic', 'mcu', 'regulator'}
    comp_type_map = {c.ref: getattr(c, 'component_type', 'generic') for c in model.components}

    for i, cluster in enumerate(clusters):
        has_ic = any(comp_type_map.get(r) in ic_types for r in cluster)
        has_cap = any(comp_type_map.get(r) == 'capacitor' for r in cluster)

        if has_ic and not has_cap:
            # Find which IC this cluster has
            ic_ref = next(r for r in cluster if comp_type_map.get(r) in ic_types)

            # Find the best cap to steal from another cluster:
            # prefer the cluster with the most caps (can afford to lose one)
            best_donor = None
            best_cap_ref = None
            best_cap_count = 0

            for j, other in enumerate(clusters):
                if j == i:
                    continue
                other_caps = [r for r in other if comp_type_map.get(r) == 'capacitor']
                if len(other_caps) > best_cap_count and len(other_caps) > 1:
                    # Check if this cap shares any net with our IC
                    ic_nets = set()
                    for net in model.nets:
                        if ic_ref in net.component_refs:
                            ic_nets.add(net.name)

                    for cap_ref in other_caps:
                        cap_nets = set()
                        for net in model.nets:
                            if cap_ref in net.component_refs:
                                cap_nets.add(net.name)
                        # Prefer caps that share at least one net with the IC
                        if ic_nets & cap_nets:
                            best_donor = j
                            best_cap_ref = cap_ref
                            best_cap_count = len(other_caps)
                            break

                    # If no shared-net cap found, take any cap from the richest cluster
                    if best_cap_ref is None and len(other_caps) > 1:
                        best_donor = j
                        best_cap_ref = other_caps[0]
                        best_cap_count = len(other_caps)

            if best_donor is not None and best_cap_ref is not None:
                clusters[best_donor].remove(best_cap_ref)
                cluster.append(best_cap_ref)


def _greedy_clustering(G: Graph, n_clusters: int) -> list[list[str]]:
    """Balanced greedy clustering with soft size constraint.

    Seed clusters from highest-degree nodes, then assign each remaining
    node to the cluster it shares the most edges with.  A soft size
    penalty prevents any cluster from growing much larger than the
    target size (total_nodes / n_clusters), which fixes the Cluster 0
    bloat problem where the highest-degree seed accumulates everything.
    """
    nodes = list(G.nodes())
    if not nodes:
        return []

    n_clusters = min(n_clusters, len(nodes))
    target_size = len(nodes) / n_clusters

    # Seed: pick highest-degree nodes
    degree_sorted = sorted(nodes, key=lambda n: G.degree(n, weight="weight"), reverse=True)
    seeds = degree_sorted[:n_clusters]

    clusters: list[list[str]] = [[seed] for seed in seeds]
    assigned = set(seeds)
    cluster_sizes = [1] * n_clusters

    # Assign remaining nodes greedily by edge weight + size balance
    remaining = [n for n in nodes if n not in assigned]
    remaining.sort(key=lambda n: G.degree(n, weight="weight"), reverse=True)

    for node in remaining:
        best_cluster = 0
        best_score = -1
        for i, cluster in enumerate(clusters):
            weight = sum(G[node][c].get("weight", 1) for c in cluster if G.has_edge(node, c))
            # Soft size penalty: reduce attractiveness of oversized clusters.
            # At 2x target size, score is reduced by 50%.  This prevents
            # Cluster 0 from absorbing everything just because it has the
            # highest-degree seed (typically the MCU).
            size_penalty = max(0, (cluster_sizes[i] - target_size) / max(target_size, 1))
            score = weight * (1.0 - 0.5 * size_penalty)
            if score > best_score:
                best_score = score
                best_cluster = i
        clusters[best_cluster].append(node)
        cluster_sizes[best_cluster] += 1

    return clusters


def _louvain_clustering(G: Graph, n_clusters: int) -> list[list[str]]:
    """Community detection using the Louvain method (pure-Python implementation)."""
    try:
        communities = louvain_communities(G, weight="weight")
    except Exception:
        communities = _greedy_clustering(G, n_clusters)
        return communities

    # If too many communities, merge smallest
    while len(communities) > n_clusters:
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
