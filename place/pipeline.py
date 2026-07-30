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
from place.legalizer import legalize as legalize_macros, expand_bounds_to_fit
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


def build_macros(model: "BoardModel") -> tuple[list[Macro], list[Macro], list[Macro]]:
    """Build interior macros, connector macros, and fixed-obstacle macros.

    Returns ``(interior_macros, connector_macros, fixed_macros)``.

    - ``interior_macros``: IC+caps groups and standalone movable
      components. SA operates on these.
    - ``connector_macros``: edge-connector empty-shell macros. Marked
      ``is_fixed`` by the caller after perimeter placement so the
      legalizer treats them as immovable obstacles.
    - ``fixed_macros`` (Finding 4 fix): macros for components already
      marked ``is_fixed`` at parse time — currently mounting holes.
      These are NOT in the interior or connector lists. They're passed
      to the legalizer as additional immovable obstacles so the
      push-apart pass pushes movable macros away from them, and to SA's
      cost function so SA sees the obstacle penalty (SA itself doesn't
      move them — see ``run_macro_sa``'s fixed-macro filtering).

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
    fixed_components: list["Component"] = []

    for c in model.components:
        if c.is_fixed:
            fixed_components.append(c)  # mounting holes, pre-placed hardware
            continue
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

    # Fixed components (mounting holes, pre-placed hardware) become
    # is_fixed=True macros — the legalizer treats them as immovable
    # obstacles that movable macros get pushed away from.
    fixed_macros = [Macro.alone(c) for c in fixed_components]
    for m in fixed_macros:
        m.is_fixed = True

    return interior_macros, connector_macros, fixed_macros


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
    rudy_weight: float = 0.0,
    use_abacus: bool = False,
    use_sa_polish: bool = False,
) -> dict[str, object]:
    """Run the macro-first placement pipeline.

    Returns a dict with cost breakdown before/after SA and legalizer stats.

    Finding 7 fix: ``rudy_weight`` adds RUDY congestion to the SA cost
    function (default 0 = disabled). When > 0, SA gets gradient signal
    to spread macros away from routing choke points. The verbose report
    always shows RUDY (initial + final) so the user can see congestion
    regardless of whether SA is using it as a cost term.
    """
    # ─── Phase 1: classify → assign → build macros ───────────────────
    interior_macros, connector_macros, fixed_macros = build_macros(model)
    if verbose:
        n_caps = sum(len(m.followers) for m in interior_macros)
        n_alone = sum(1 for m in interior_macros if not m.followers)
        n_fixed = len(fixed_macros)
        print(
            f"  Built {len(interior_macros)} interior macros "
            f"({sum(1 for m in interior_macros if m.followers)} with caps, "
            f"{n_alone} standalone, {n_caps} caps total) + "
            f"{len(connector_macros)} connector macros"
            + (f" + {n_fixed} fixed obstacle macros" if n_fixed else "")
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
        # Mark connector macros as FIXED — the legalizer treats them as
        # immovable obstacles so interior macros get pushed out of the
        # connector zone instead of overlapping them. This is the fix
        # for the test6 macro-connector collision regression.
        for m in connector_macros:
            m.is_fixed = True
    else:
        connector_reserve = 0.0

    # ─── Phase 3: interior placement (Phase A space-fill + Phase B nudge) ──
    interior_bbox = compute_interior_bbox(
        interior_macros, model.board, margin, connector_reserve=connector_reserve,
    )

    # If macros are too dense for the interior, expand the bounds BEFORE
    # Phase A so the shelf-packer distributes them across the full expanded
    # area (not crammed into the tight original bbox). This is the same
    # expansion the legalizer would do later, but doing it here means SA
    # starts from a well-spread pose instead of a tight cluster.
    expanded_interior = expand_bounds_to_fit(
        interior_macros, interior_bbox,
        extra_padding=1.0, target_density=0.55,
    )
    if expanded_interior != interior_bbox:
        if verbose:
            iw = interior_bbox[2] - interior_bbox[0]
            ih = interior_bbox[3] - interior_bbox[1]
            ew = expanded_interior[2] - expanded_interior[0]
            eh = expanded_interior[3] - expanded_interior[1]
            print(
                f"  Pre-Phase-A bounds expansion: {iw:.1f}×{ih:.1f} -> "
                f"{ew:.1f}×{eh:.1f}mm (density target 55%)"
            )
        interior_bbox = expanded_interior

        # ROOT-CAUSE FIX (test4 OOB regression): grow the auto-inferred
        # board outline to contain the expanded interior.
        #
        # _ensure_board_capacity (Step 1.5) sizes model.board by *board-level
        # effective-area* density (target_pack_density = 0.55), but the
        # expansion just above sizes the *interior* by *courtyard-aware macro
        # area* at the same 0.55. For dense small boards the courtyard-aware
        # interior overshoots the effective-area board, so the legalizer
        # (which clamps to sa_bounds = expanded interior) can leave components
        # beyond model.board. Because the output Edge.Cuts is generated from
        # model.board (see _inject_inferred_edge_cuts), those components then
        # overhang the board edge in the written file.
        #
        # The missing piece is the feedback loop: once the interior has been
        # forced bigger to fit the macros, the inferred board must follow.
        # Growing it here makes the written outline always cover the
        # placement — and incidentally makes the legalizer's keepout inset
        # math (Phase 5, min_inset) well-defined again. User-drawn outlines
        # (model.user_defined_outline, set when the source had Edge.Cuts) are
        # respected and never grown.
        if not getattr(model, "user_defined_outline", False):
            b = model.board
            new_xmin = min(b.x_min, expanded_interior[0])
            new_ymin = min(b.y_min, expanded_interior[1])
            new_xmax = max(b.x_max, expanded_interior[2])
            new_ymax = max(b.y_max, expanded_interior[3])
            if (new_xmin, new_ymin, new_xmax, new_ymax) != (
                b.x_min, b.y_min, b.x_max, b.y_max,
            ):
                from models.board_model import BoardOutline
                old_w, old_h = b.width, b.height
                model.board = BoardOutline(new_xmin, new_ymin, new_xmax, new_ymax)
                if verbose:
                    print(
                        f"  Board grown to contain expanded interior: "
                        f"{old_w:.1f}×{old_h:.1f} -> "
                        f"{model.board.width:.1f}×{model.board.height:.1f}mm"
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
        cost_init = evaluate(model, interior_macros + connector_macros + fixed_macros,
                              alpha=alpha, beta=beta, gamma=gamma,
                              net_weights=chain_net_weights)
        # Finding 7 fix: wire RUDY congestion into the macro-v2 pipeline's
        # verbose report (was only in the legacy smart_placement path).
        # RUDY (Rectangular Uniform wire DensitY) estimates routing
        # congestion by distributing each net's bbox uniformly across the
        # grid cells it covers. The penalty is non-zero only when peak
        # congestion exceeds 1.5× the average — i.e. when there's a
        # routing choke point HPWL alone misses.
        # See engine/congestion.py for the full implementation.
        from engine.congestion import rudy_congestion_penalty
        rudy_penalty_init, rudy_peak_init, rudy_avg_init, _ = rudy_congestion_penalty(model)
        print(f"  After initial: total={cost_init['total']:.2f} "
              f"(hpwl={cost_init['hpwl']:.1f}, overlap={cost_init['overlap']:.1f}, "
              f"boundary={cost_init['boundary']:.1f})")
        if rudy_penalty_init > 0:
            print(f"  RUDY: penalty={rudy_penalty_init:.3f} peak={rudy_peak_init:.3f} "
                  f"avg={rudy_avg_init:.3f} (routing congestion hotspot detected)")

    # ─── Phase 4: SA on interior macros (connectors + fixed stay put) ──
    # Bounds = interior_bbox (possibly already expanded in Phase 3) so SA
    # doesn't push into the connector zone. SA distributes macros across
    # the full expanded area.
    #
    # Fixed macros (mounting holes) are included in the macros list so
    # the cost function sees their overlap penalty — SA gets gradient
    # signal to keep movable macros away from them. Macro.translate /
    # set_pose refuse to move is_fixed macros, so SA's random picks of
    # fixed macros no-op (a few iterations wasted, no correctness issue).
    sa_bounds = interior_bbox
    sa_result = run_macro_sa(
        model, interior_macros + fixed_macros, sa_bounds,
        iterations=sa_iterations, reheats=sa_reheats,
        alpha=alpha, beta=beta, gamma=gamma,
        seed=seed, verbose=verbose,
        net_weights=chain_net_weights,
        rudy_weight=rudy_weight,
    )

    # ─── Phase 5: legalize all macros together ───────────────────────
    all_macros = interior_macros + connector_macros + fixed_macros
    # Use full board bounds for legalize (connectors can overhang)
    board_bounds = (
        model.board.x_min, model.board.y_min,
        model.board.x_max, model.board.y_max,
    )
    # Per-macro edge keepout: ICs/MCUs/regulators get extra edge clearance
    # from the BOARD OUTLINE — DFM rule a human designer always applies,
    # keeps silicon away from the board edge where it's harder to route,
    # harder to rework, and more exposed to mechanical stress.
    #
    # ROOT-CAUSE FIX (B1): the legalizer runs on `sa_bounds` (the interior
    # bbox), which is already inset from the board outline by `margin +
    # connector_reserve`. Applying the full keepout on top of `sa_bounds`
    # double-counts the inset and pushes ICs 15-20mm from the true board
    # edge on dense boards, causing overlap regressions.
    #
    # The fix has two parts:
    #   1. Subtract the interior-bbox inset from the keepout in the
    #      callback. If the keepout is smaller than the inset, the macro
    #      is already far enough from the board edge — effective keepout
    #      is 0.
    #   2. Density-scale the keepout using the INTERIOR-BBOX density
    #      (the actual packing pressure the legalizer faces), not the
    #      board-level density that `edge_keepout_extra_for` uses. On
    #      dense boards (interior density > 40%) the keepout is scaled
    #      down proportionally — the DFM rule is a nice-to-have, not a
    #      hard constraint, and on packed boards it's better to have a
    #      routable placement with ICs slightly close to the edge than
    #      an un-routable one with ICs in the "right" place.
    board = model.board
    inset_left   = sa_bounds[0] - board.x_min
    inset_right  = board.x_max - sa_bounds[2]
    inset_top    = sa_bounds[1] - board.y_min
    inset_bottom = board.y_max - sa_bounds[3]
    min_inset = min(inset_left, inset_right, inset_top, inset_bottom)

    # Compute interior-bbox density for keepout scaling. This is the
    # density the legalizer actually faces — board-level density
    # (component_area / board_area) understates the packing pressure
    # because it includes the margin ring and connector reserve.
    sa_bounds_area = (sa_bounds[2] - sa_bounds[0]) * (sa_bounds[3] - sa_bounds[1])
    interior_macro_area = sum(
        max(0.0, m.bbox[2] - m.bbox[0]) * max(0.0, m.bbox[3] - m.bbox[1])
        for m in interior_macros
    )
    interior_density = interior_macro_area / sa_bounds_area if sa_bounds_area > 0 else 0.0

    # Finding 5 fix: ONE density-adaptive keepout scale, here in the
    # legalizer callback. Previously this callback applied its own
    # 1.0→0.2 scaling AND `engine.cost_state.edge_keepout_extra_for`
    # applied another 1.0→0.3 scaling — the two multiplied, giving
    # 0.06× at 0.55 density (almost certainly not what either author
    # intended). Now `edge_keepout_extra_for` returns the BASE value
    # (no scaling) and ALL density scaling happens here, using the
    # INTERIOR-BBOX density (the actual packing pressure the legalizer
    # faces), not the board-level density.
    #
    # Finding 6 fix: the density thresholds (0.35 / 0.55) match the
    # shared target_pack_density (0.55), so the keepout is at full
    # strength whenever the board is below target density and scales
    # down to 0.2× only when the board is at-or-above target density
    # (where there's no room for full keepout without creating overlaps).
    from utils.density import target_pack_density
    target_density = target_pack_density()
    # Below the low-density threshold (target_density - 0.20), keepout is
    # at full strength. Above target_density, keepout scales to 0.2×
    # (still a small margin, never fully zero). Linear in between.
    low_density_threshold = max(0.10, target_density - 0.20)
    if interior_density <= low_density_threshold:
        keepout_scale = 1.0
    elif interior_density >= target_density:
        keepout_scale = 0.2
    else:
        # Linear from 1.0 at low_density_threshold to 0.2 at target_density.
        span = max(target_density - low_density_threshold, 1e-6)
        keepout_scale = 1.0 - 0.8 * (interior_density - low_density_threshold) / span

    try:
        from engine.cost_state import edge_keepout_extra_for

        def _keepout_cb(macro: "Macro") -> float:
            # Finding 5 fix: edge_keepout_extra_for now returns the BASE
            # value (no density scaling). All density scaling happens
            # via keepout_scale above.
            raw = edge_keepout_extra_for(macro.leader, model=model)
            # Apply interior-density scaling (root-cause fix for dense boards
            # where the full keepout causes overlap regressions).
            scaled = raw * keepout_scale
            # Subtract the interior inset so the keepout is measured
            # from the BOARD OUTLINE, not the interior bbox. Clamp at 0
            # — if the inset already exceeds the keepout, no extra
            # push is needed.
            return max(0.0, scaled - min_inset)
    except Exception:
        _keepout_cb = None

    # Pass sa_bounds (already expanded if needed) so legalizer doesn't
    # re-expand; the bounds are already at target density. The per-macro
    # keepout callback handles the board-edge clearance correctly (see
    # comment above) by subtracting the interior-bbox inset.
    #
    # Pass interior_macros + connector_macros + fixed_macros to the legalizer
    # so it can resolve macro-connector overlaps AND push interior macros
    # away from fixed mounting holes. Connector macros and fixed macros are
    # both marked is_fixed=True so the legalizer treats them as immovable
    # obstacles — interior macros get pushed out of their zones instead of
    # overlapping them. This is the fix for the test6 macro-connector /
    # macro-mounting-hole collision regression (Finding 4).
    legal_stats = legalize_macros(
        model, interior_macros + connector_macros + fixed_macros, sa_bounds,
        grid_mm=grid_mm, verbose=verbose,
        expand_to_fit=False,  # already expanded above
        per_macro_keepout=_keepout_cb,
        use_abacus=use_abacus,
        use_sa_polish=use_sa_polish,
        seed=seed,
    )

    # Final cost (all macros including connectors)
    final_cost = evaluate(model, all_macros, alpha=alpha, beta=beta, gamma=gamma,
                          net_weights=chain_net_weights)
    if verbose:
        # Finding 7: report final RUDY so the user can see whether SA +
        # legalization created or resolved routing choke points.
        from engine.congestion import rudy_congestion_penalty
        rudy_penalty_final, rudy_peak_final, rudy_avg_final, _ = rudy_congestion_penalty(model)
        print(
            f"  Final: total={final_cost['total']:.2f} "
            f"(hpwl={final_cost['hpwl']:.1f}, overlap={final_cost['overlap']:.1f}, "
            f"boundary={final_cost['boundary']:.1f})"
        )
        if rudy_penalty_final > 0:
            print(f"  RUDY: penalty={rudy_penalty_final:.3f} peak={rudy_peak_final:.3f} "
                  f"avg={rudy_avg_final:.3f}")
        elif rudy_penalty_init > 0:
            print(f"  RUDY: no hotspots remaining (was {rudy_penalty_init:.3f} before SA)")

    return {
        "sa_result": sa_result,
        "legal_stats": legal_stats,
        "final_cost": final_cost,
        "n_interior_macros": len(interior_macros),
        "n_connector_macros": len(connector_macros),
    }
