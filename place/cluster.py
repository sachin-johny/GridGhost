"""Cluster ordering + bounded connectivity nudges for interior placement.

Issue 1 of PLACEMENT_FIX_PLAN.md splits interior seeding into two
independently-testable phases:

  * **Phase A** (space-filling, connectivity-*blind* positioning) needs
    macros grouped by net cluster (so connected macros stay together) and
    the *clusters* ordered by descending max-member height — shelf-pack
    area efficiency depends on height-sorted ordering.
  * **Phase B** (bounded connectivity nudge) needs a per-net-weighted pull
    toward genuinely informative attractors, capped so a degenerate
    attractor centroid can only pull a cluster part-way off its Phase-A
    home — never collapse the board.

Both ``order_macros_by_connectivity`` and ``compute_attractor_nudges`` are
pure functions so the regression that motivated this (GND's 18 pins
dominating the unweighted average → every cluster's attractor centroid
collapses to the board center) can be asserted directly with synthetic
nets instead of rendering a board.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from engine.cost_state import _is_power_net
from engine.net_clustering import cluster_components

if TYPE_CHECKING:
    from models.board_model import BoardModel, Net
    from models.macro import Macro


# Clique net-model weight floor for power/ground nets.  Applied ON TOP of
# the 1/(k-1) fan-out weighting as belt-and-braces: a rail that happens to
# be low-fan-out on a given board is still not placement-relevant, so it
# shouldn't dominate the nudge even when k is small.  See plan §Issue 1.
POWER_NET_DISCOUNT = 0.25


def _macro_height(m: "Macro") -> float:
    return m.bbox[3] - m.bbox[1]


def group_macros_by_cluster(
    model: "BoardModel",
    macros: list["Macro"],
) -> list[list["Macro"]]:
    """Group macros by net cluster (``cluster_components``), preserving identity.

    Returns a list of clusters; each cluster is a list of Macros.  Cap
    followers are part of their leader's macro and never appear standalone,
    so they don't fragment clusters.  Any macro that didn't surface in a
    cluster is appended as a final orphan cluster.
    """
    if not macros:
        return []

    ref_to_macro: dict[str, "Macro"] = {m.leader.ref: m for m in macros}

    macro_clusters: list[list["Macro"]] = []
    seen: set[str] = set()
    for cluster_refs in cluster_components(model):
        group: list["Macro"] = []
        for ref in cluster_refs:
            m = ref_to_macro.get(ref)
            if m is None or m.leader.ref in seen:
                continue
            group.append(m)
            seen.add(m.leader.ref)
        if group:
            macro_clusters.append(group)

    orphans = [m for m in macros if m.leader.ref not in seen]
    if orphans:
        macro_clusters.append(orphans)
    return macro_clusters


def order_macros_by_connectivity(
    model: "BoardModel",
    macros: list["Macro"],
) -> list[list["Macro"]]:
    """Cluster by signal-net connectivity, order clusters by descending height.

    Grouping respects connectivity (connected macros share a cluster so they
    shelf-pack as a block); ordering is by **height**, not connectivity, so
    the shelf-pack in Phase A is area-efficient.  This is the ordering input
    Phase A consumes — it carries no positional/connectivity-pull information,
    which is the whole point of the Phase A/B split.
    """
    clusters = group_macros_by_cluster(model, macros)
    clusters.sort(key=lambda grp: -max((_macro_height(m) for m in grp), default=0.0))
    return clusters


def collect_attractor_positions(
    model: "BoardModel",
) -> dict[str, list[tuple[float, float]]]:
    """Map net name → list of (x, y) attractor positions for that net.

    Attractors are components that are either fixed (pre-placed ICs, mounting
    holes) OR already-placed connectors.  Read from the model's current state
    so connector positions (placed earlier in the pipeline) count.
    """
    out: dict[str, list[tuple[float, float]]] = {}
    for net in model.nets:
        for ref in net.component_refs:
            comp = model.get_component(ref)
            if comp is None:
                continue
            if getattr(comp, "is_fixed", False) or comp.component_type == "connector":
                out.setdefault(net.name, []).append((comp.x, comp.y))
    return out


def weighted_attractor_target(
    nets: list["Net"],
    attractor_positions: dict[str, list[tuple[float, float]]],
    *,
    power_discount: float = POWER_NET_DISCOUNT,
) -> tuple[float, float] | None:
    """Per-net-centroid, clique-net-weighted attractor target for a set of nets.

    Pure function — the regression-friendly core of Phase B.  For each net
    that has attractor points, contributes ONE point (the net's own centroid)
    weighted by the clique net model ``1/(k-1)`` (k = pin count), so a high-
    fan-out net can't dominate the average by raw pin count.  Power nets get
    an extra ``power_discount`` floor.  Returns the weighted mean, or ``None``
    if no net contributed (no informative attractor).

    This is what makes a 20-pin GND net contribute the same order of weight
    as a 2-pin signal net — the exact dilution the old unweighted pool caused
    every cluster's attractor to collapse to the board center.
    """
    wx_sum = 0.0
    wy_sum = 0.0
    w_sum = 0.0
    for net in nets:
        pts = attractor_positions.get(net.name)
        if not pts:
            continue
        ncx = sum(p[0] for p in pts) / len(pts)
        ncy = sum(p[1] for p in pts) / len(pts)
        k = len(net.pins)
        weight = 1.0 / (k - 1) if k > 1 else 1.0
        if _is_power_net(net.name):
            weight *= power_discount
        wx_sum += ncx * weight
        wy_sum += ncy * weight
        w_sum += weight
    if w_sum <= 0.0:
        return None
    return (wx_sum / w_sum, wy_sum / w_sum)


def compute_attractor_nudges(
    model: "BoardModel",
    clusters: list[list["Macro"]],
    attractor_positions: dict[str, list[tuple[float, float]]],
    *,
    nudge_fraction: float = 0.25,
    max_nudge_mm: float | None = None,
    power_discount: float = POWER_NET_DISCOUNT,
) -> dict[int, tuple[float, float]]:
    """Per-net-centroid, fan-out-weighted, capped connectivity nudge per cluster.

    For each cluster, the target attractor is built as:

      1. **One vote per net.**  Each net touching the cluster contributes its
         OWN centroid (mean of its attractor points) — one point per net,
         regardless of fan-out.  This is what stops GND's 18 perimeter pins
         from out-voting a 2-pin signal net 18-to-1.
      2. **Clique net-model weighting.**  A k-pin net contributes with weight
         ``1/(k-1)`` (Alpert & Kahng), so a high-fan-out net is damped by its
         own fan-out.  GND (k=18) → 1/17, a 2-pin signal → 1/1.  Power nets
         get an extra ``power_discount`` floor on top.
      3. **Capped magnitude.**  The nudge toward the weighted target is at
         most ``nudge_fraction`` of the distance from the cluster's current
         centroid (and at most ``max_nudge_mm``).  Bounded by construction:
         even a degenerate target can only pull a cluster part-way off its
         Phase-A home, never collapse the board.

    Returns ``{cluster_index: (dx, dy)}``.  Clusters with no informative
    attractor get no entry (no nudge — they keep their Phase-A position).
    """
    nudges: dict[int, tuple[float, float]] = {}

    for idx, cluster_macros in enumerate(clusters):
        # Cluster member refs (leaders + cap followers) for net membership.
        cluster_refs: set[str] = {m.leader.ref for m in cluster_macros}
        for m in cluster_macros:
            for f in m.followers:
                cluster_refs.add(f.ref)

        touching = [
            net for net in model.nets
            if any(r in cluster_refs for r in net.component_refs)
        ]
        target = weighted_attractor_target(
            touching, attractor_positions, power_discount=power_discount,
        )
        if target is None:
            continue

        target_x, target_y = target

        # Cluster's current centroid (macro bbox centers, post Phase A).
        cx = sum((m.bbox[0] + m.bbox[2]) / 2 for m in cluster_macros) / len(cluster_macros)
        cy = sum((m.bbox[1] + m.bbox[3]) / 2 for m in cluster_macros) / len(cluster_macros)

        dx = target_x - cx
        dy = target_y - cy
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            continue

        # Cap: nudge_fraction of the distance, and an absolute ceiling.
        mag = dist * nudge_fraction
        if max_nudge_mm is not None:
            mag = min(mag, max_nudge_mm)
        if mag < 1e-9:
            continue

        nudges[idx] = (dx / dist * mag, dy / dist * mag)

    return nudges
