"""Macro-first placement pipeline.

Single-entry orchestrator: classify → assign caps → build macros →
initial place → SA → legalize. Returns the placed model.

This replaces the 9-phase pipeline in engine/smart_placement.py with
5 stages, all sharing one abstraction (the Macro) and one cost
function (cost.cost.evaluate).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from assign.assign_caps import assign_caps, IC_TYPES
from cost.cost import evaluate
from cost.chains import detect_interior_chains, build_chain_net_weights, CHAIN_NET_WEIGHT
from models.macro import Macro
from place.connectors import place_connectors_perimeter
from place.initial import (
    place_interior_phase_a,
    apply_connectivity_nudges,
    compute_interior_bbox,
)
from place.legalizer import legalize as legalize_macros
from place.sa import run_macro_sa

if TYPE_CHECKING:
    from models.board_model import BoardModel, Component


def _is_vertical_connector(comp: "Component") -> bool:
    """True for THT/vertical connectors that should be treated as interior."""
    fp = (getattr(comp, "footprint", "") or "").lower()
    val = (getattr(comp, "value", "") or "").lower()
    name = fp + " " + val
    if "vertical" in name or "tht" in name:
        return True
    if "horizontal" in name or "angled" in name or "side" in name:
        return False
    return False  # default to edge-connector for ambiguous cases


def build_macros(model: "BoardModel") -> tuple[list[Macro], list[Macro]]:
    """Build interior macros and connector macros.

    Returns ``(interior_macros, connector_macros)``. Interior macros
    include IC+caps groups and standalone components. Connector macros
    are empty-shell macros (leader only, no followers) so the legalizer
    treats them as rectangles.

    A cap assigned to an IC is ONLY in that IC's macro — never as a
    standalone macro. This is critical: if a cap were in two macros,
    push_apart could move the standalone cap away from its IC.
    """
    decap_map = assign_caps(model)
    cap_ref_to_ic: dict[str, str] = {
        cap: ic for ic, caps in decap_map.items() for cap in caps
    }

    interior_components: list["Component"] = []
    connectors: list["Component"] = []

    for c in model.components:
        if c.is_fixed:
            continue  # Fixed components stay put
        if c.component_type == "connector" and not _is_vertical_connector(c):
            connectors.append(c)
        else:
            interior_components.append(c)

    # Interior components that are NOT assigned-as-followers become leaders.
    # Caps assigned to an IC are followers only — never their own macro.
    leaders = [c for c in interior_components if c.ref not in cap_ref_to_ic]
    leader_refs = {c.ref for c in leaders}

    interior_macros: list[Macro] = []
    for leader in leaders:
        if leader.component_type in IC_TYPES and leader.ref in decap_map:
            cap_refs = decap_map[leader.ref]
            cap_comps = [model.get_component(r) for r in cap_refs]
            cap_comps = [c for c in cap_comps if c is not None]
            # Don't pass other leaders — they're at parse positions and would
            # force fallback offsets. find_cap_offset will still avoid
            # overlapping sibling caps within this macro.
            macro = Macro.with_caps(leader, cap_comps)
        else:
            macro = Macro.alone(leader)
        interior_macros.append(macro)

    # Connectors become standalone macros for legalizer bbox purposes
    connector_macros = [Macro.alone(c) for c in connectors]
    return interior_macros, connector_macros


def place_v2(
    model: "BoardModel",
    *,
    margin: float = 5.0,
    grid_mm: float = 1.0,
    sa_iterations: int = 1500,
    sa_reheats: int = 2,
    alpha: float = 1.0,
    beta: float = 25.0,
    gamma: float = 8.0,
    connector_mating_margin: float = 5.0,
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, object]:
    """Run the macro-first placement pipeline.

    Returns a dict with cost breakdown before/after SA and legalizer stats.
    """
    # ─── Phase 1: classify → assign → build macros ───────────────────
    interior_macros, connector_macros = build_macros(model)
    if verbose:
        n_caps = sum(len(m.followers) for m in interior_macros)
        n_alone = sum(1 for m in interior_macros if not m.followers)
        print(
            f"  Built {len(interior_macros)} interior macros "
            f"({sum(1 for m in interior_macros if m.followers)} with caps, "
            f"{n_alone} standalone, {n_caps} caps total) + "
            f"{len(connector_macros)} connector macros"
        )

    # ─── Phase 2: connector placement on perimeter ───────────────────
    # Connectors go first so interior placement can be net-aware about
    # their final positions (connector centroid acts as attractor).
    if connector_macros:
        # Edge connectors OVERHANG the outline (body outside, pads inside),
        # so their interior footprint is just the pad depth (mating_margin),
        # not the full body. Reserving more than that needlessly shrinks the
        # interior — on cbb the old along-edge-extent estimate carved out
        # ~12.5mm/edge and left only ~57% of the board usable.
        connector_reserve = min(
            connector_mating_margin,
            min(model.board.width, model.board.height) * 0.2,
        )
        connectors = [m.leader for m in connector_macros]
        place_connectors_perimeter(
            model, connectors, model.board, margin,
            mating_margin=connector_mating_margin,
        )
        for m in connector_macros:
            m.apply_offsets()
    else:
        connector_reserve = 0.0

    # ─── Phase 3: interior placement (Phase A space-fill + Phase B nudge) ──
    interior_bbox = compute_interior_bbox(
        interior_macros, model.board, margin, connector_reserve=connector_reserve,
    )
    # Phase A: height-ordered shelf-pack of net-clusters to fill the interior
    # (connectivity-blind positioning — uses the board, doesn't collapse).
    clusters = place_interior_phase_a(model, interior_macros, interior_bbox)
    if verbose:
        print(f"  Phase A (space-filling shelf-pack): {len(clusters)} clusters seeded")

    # Phase B: bounded, per-net-weighted connectivity nudge on top of Phase A.
    apply_connectivity_nudges(model, clusters, interior_bbox, verbose=verbose)

    # Issue 3: detect interior signal-flow chains and upweight their internal
    # nets' HPWL so SA pulls chain members into a line.  Connectivity-based, so
    # it's computed once here and threaded through SA + evaluate.  No change to
    # the shelf-packer — this is purely a cost-function nudge (Zhu et al. 2020).
    chains = detect_interior_chains(model)
    chain_net_weights = build_chain_net_weights(model, chains)
    if verbose and chains:
        n_chain_nets = len(chain_net_weights)
        print(f"  Issue 3 (signal-flow chains): {len(chains)} chain(s), "
              f"{n_chain_nets} net(s) upweighted x{CHAIN_NET_WEIGHT:.1f}")

    if verbose:
        cost_init = evaluate(model, interior_macros + connector_macros,
                              alpha=alpha, beta=beta, gamma=gamma,
                              net_weights=chain_net_weights)
        print(f"  After initial: total={cost_init['total']:.2f} "
              f"(hpwl={cost_init['hpwl']:.1f}, overlap={cost_init['overlap']:.1f}, "
              f"boundary={cost_init['boundary']:.1f})")

    # ─── Phase 4: SA on interior macros (connectors stay put) ────────
    # Bounds = interior_bbox so SA doesn't push into connector zone
    sa_bounds = interior_bbox
    sa_result = run_macro_sa(
        model, interior_macros, sa_bounds,
        iterations=sa_iterations, reheats=sa_reheats,
        alpha=alpha, beta=beta, gamma=gamma,
        seed=seed, verbose=verbose,
        net_weights=chain_net_weights,
    )

    # ─── Phase 5: legalize all macros together ───────────────────────
    all_macros = interior_macros + connector_macros
    # Use full board bounds for legalize (connectors can overhang)
    board_bounds = (
        model.board.x_min, model.board.y_min,
        model.board.x_max, model.board.y_max,
    )
    legal_stats = legalize_macros(
        model, interior_macros, sa_bounds,
        grid_mm=grid_mm, verbose=verbose,
    )

    # Final cost (all macros including connectors)
    final_cost = evaluate(model, all_macros, alpha=alpha, beta=beta, gamma=gamma,
                          net_weights=chain_net_weights)
    if verbose:
        print(
            f"  Final: total={final_cost['total']:.2f} "
            f"(hpwl={final_cost['hpwl']:.1f}, overlap={final_cost['overlap']:.1f}, "
            f"boundary={final_cost['boundary']:.1f})"
        )

    return {
        "sa_result": sa_result,
        "legal_stats": legal_stats,
        "final_cost": final_cost,
        "n_interior_macros": len(interior_macros),
        "n_connector_macros": len(connector_macros),
    }
