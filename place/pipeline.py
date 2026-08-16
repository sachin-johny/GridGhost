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

from assign.assign_caps import classify_caps, IC_TYPES
from cost.cost import evaluate
from cost.chains import detect_interior_chains, build_chain_net_weights, CHAIN_NET_WEIGHT
from models.macro import Macro
from place.connectors import place_connectors_perimeter
from place.initial import (
    place_interior_phase_a,
    apply_connectivity_nudges,
    compute_interior_bbox,
    seed_rail_adjacent_caps,
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

    A cap assigned to an IC as a rigid follower is ONLY in that IC's
    macro — never as a standalone macro. Rail-adjacent caps (excess
    decoupling caps beyond ``max_decaps_per_ic`` per IC) become
    standalone macros so SA can move them freely toward the power rail.
    This avoids rigidly locking a large fan of caps to one IC on dense
    shared-rail boards, which starves SA of degrees of freedom and
    produces a macro block that is hard to legalize.
    """
    # classify_caps splits decoupling caps into rigid followers (≤ N per
    # IC → Macro.with_caps) and rail-adjacent (the rest → standalone).
    # Only rigid followers are in decap_map / cap_ref_to_ic, so the
    # rail-adjacent caps naturally become Macro.alone() below.
    decap_map, _rail_adjacent_refs, _bulk_refs, _coupling_refs = classify_caps(model)
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
    rudy_weight: float | None = None,
    pin_density_weight: float | None = None,
    use_abacus: bool = False,
    use_sa_polish: bool = False,
    delta: float = 0.0,
    rules: list | None = None,
    cap_attraction_weight: float | None = None,
    clearance_weight: float | None = None,
    exclude_nets: set[str] | None = None,
) -> dict[str, object]:
    """Run the macro-first placement pipeline.

    Returns a dict with cost breakdown before/after SA and legalizer stats.

    Routability signals: ``rudy_weight`` and ``pin_density_weight`` add
    two complementary congestion terms to the SA cost function. Both
    default to ``None`` — resolved against ``config.json``
    (``annealer.rudy_weight`` / ``annealer.pin_density_weight``), which
    ships with both ON so the placer produces a routable result out of
    the box. Pass ``0`` to explicitly disable either signal. The verbose
    report always shows both signals' initial + final state regardless
    of whether they're active in the cost, so the user can see
    congestion even when running with the signals off.

    Constraint penalties (opt-in): ``delta`` (default 0) and ``rules``
    (a list of ``ConstraintRule`` from a board profile) wire the legacy
    ``engine/constraint_evaluator.py`` evaluator into the macro-v2 SA
    cost. When ``delta > 0`` and ``rules`` is non-empty, SA gets
    gradient signal for decoupling proximity, crystal-MCU, thermal
    grouping/separation, analog/digital separation, etc. — same rules
    the legacy path applies. Default 0 = current macro-v2 behavior
    (no constraint penalties). See IMPROVEMENTS §2.4.

    Cap attraction (``cap_attraction_weight``, default None → config
    ``annealer.cap_attraction_weight``): weak cap→IC attraction for
    rail-adjacent (freed) caps. Root-cause fix for shared-rail cap
    drift — see ``cost.cost.cap_attraction_penalty``. The cap→IC
    assignment is derived here from ``rail_adjacent_to_ic`` (deterministic,
    same classification the macro builder used).

    Clearance (``clearance_weight``, default None → config
    ``annealer.clearance_weight``): routing-halo term charging
    ``max(0, target − edge_gap)`` per macro pair — the missing "close
    enough" floor between HPWL's monotone pull and the overlap term's
    cliff at touching. Scales down automatically on dense boards (where
    the target is unachievable) so it degrades to a tie-breaker instead
    of fighting the legalizer. Pass ``0`` to disable. See
    ``cost.cost.clearance_deficit``.

    """
    # Resolve routability weights against config when the caller didn't
    # explicitly pass one. The CLI passes ``None`` when the user didn't
    # set the flag, so the config default (0.3 / 0.2 in config.json)
    # wins — fixing the old bug where the CLI default of 0.0 silently
    # disabled RUDY even though config.json said 0.3.
    if (rudy_weight is None or pin_density_weight is None
            or cap_attraction_weight is None or clearance_weight is None):
        from config import load_config
        _cfg = load_config()
        if rudy_weight is None:
            rudy_weight = _cfg.annealer.rudy_weight
        if pin_density_weight is None:
            pin_density_weight = getattr(_cfg.annealer, "pin_density_weight", 0.0)
        if cap_attraction_weight is None:
            cap_attraction_weight = getattr(_cfg.annealer, "cap_attraction_weight", 0.0)
        if clearance_weight is None:
            clearance_weight = getattr(_cfg.annealer, "clearance_weight", 0.0)

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

    # Phase A½: re-seed rail-adjacent caps around their assigned IC.
    # After the shelf-pack, freed caps sit wherever their cluster landed —
    # often a tall cap-only column far from their assigned IC. Pulling
    # each cap into a ring around its ASSIGNED IC (round-robin-balanced)
    # spreads the excess across the rail's ICs and removes the pile-up
    # the clamp pass would otherwise create. See seed_rail_adjacent_caps.
    n_reseeded = seed_rail_adjacent_caps(model, interior_macros, interior_bbox)
    if verbose and n_reseeded:
        print(f"  Phase A½ (cap re-seed): {n_reseeded} rail-adjacent caps "
              f"seeded around their assigned ICs")

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
        # Wire RUDY + pin-density congestion into the macro-v2 pipeline's
        # verbose report (was only in the legacy smart_placement path).
        # RUDY (Rectangular Uniform wire DensitY) estimates routing
        # congestion by distributing each net's bbox uniformly across the
        # grid cells it covers. Pin density counts signal pins per cell,
        # catching pin-escape congestion RUDY misses. Both penalties are
        # non-zero only when peak congestion exceeds 1.5× the average —
        # i.e. when there's a routing choke point HPWL alone misses.
        # See engine/congestion.py for the full implementation.
        from engine.congestion import (
            rudy_congestion_penalty,
            pin_density_penalty as _pin_density_penalty,
        )
        rudy_penalty_init, rudy_peak_init, rudy_avg_init, _ = rudy_congestion_penalty(model)
        pd_penalty_init, pd_peak_init, pd_avg_init, _ = _pin_density_penalty(model)
        print(f"  After initial: total={cost_init['total']:.2f} "
              f"(hpwl={cost_init['hpwl']:.1f}, overlap={cost_init['overlap']:.1f}, "
              f"boundary={cost_init['boundary']:.1f})")
        if rudy_penalty_init > 0:
            print(f"  RUDY: penalty={rudy_penalty_init:.3f} peak={rudy_peak_init:.3f} "
                  f"avg={rudy_avg_init:.3f} (wire-density hotspot detected)")
        if pd_penalty_init > 0:
            print(f"  Pin-density: penalty={pd_penalty_init:.3f} peak={pd_peak_init:.3f} "
                  f"avg={pd_avg_init:.3f} (pin-escape hotspot detected)")
        if rudy_penalty_init == 0 and pd_penalty_init == 0:
            print("  Routability: no congestion hotspots detected (RUDY + pin-density clean)")

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
    #
    # Density-adaptive RUDY weight (routability-first tuning):
    # The config.json rudy_weight is the BASELINE; we scale it by a
    # factor that depends on the board's interior density AND its raw
    # RUDY peak (the actual congestion pressure). On sparse, low-
    # congestion boards (cbb, cbbwO: density<0.30, peak<0.1) RUDY is
    # just noise that competes with HPWL — we scale it DOWN. On dense,
    # high-congestion boards (test4, test6, th_sensor: density>0.40
    # and/or peak>0.5) routability matters most — we scale it UP.
    #
    # The scale is multiplicative on top of the configured baseline, so
    # users who explicitly set --rudy-weight still get their value
    # scaled by the same factor (the scale is a board-shape signal, not
    # a user-intent signal).
    sa_bounds = interior_bbox

    # Compute interior density now (we re-derive it for the keepout
    # callback below anyway; doing it here lets us also use it for
    # the RUDY-weight scaling).
    sa_bounds_area = (sa_bounds[2] - sa_bounds[0]) * (sa_bounds[3] - sa_bounds[1])
    interior_macro_area = sum(
        max(0.0, m.bbox[2] - m.bbox[0]) * max(0.0, m.bbox[3] - m.bbox[1])
        for m in interior_macros
    )
    interior_density = interior_macro_area / sa_bounds_area if sa_bounds_area > 0 else 0.0

    # Raw RUDY peak on the CURRENT (pre-SA) placement — a direct signal
    # of how congested the board is. Cheap to compute (one RUDY map pass).
    raw_rudy_peak = 0.0
    try:
        from engine.congestion import rudy_congestion_penalty as _rcp
        _pen, raw_rudy_peak, _avg, _ovf = _rcp(model)
    except Exception:
        raw_rudy_peak = 0.0

    # Density-adaptive scale (v2 — calibrated on 15 boards: 6 test_pcbs +
    # 9 external_boards/rl_pcb):
    #   density < 0.30  AND  peak < 0.10  -> 0.3× (sparse, low congestion)
    #   density > 0.45                   -> 1.2× (dense alone)
    #   peak   > 0.50                    -> 1.2× (very high peak alone)
    #   otherwise                           1.0× (the configured baseline)
    #
    # v2 change: the v1 rule `density > 0.45 OR peak > 0.40` fired too
    # broadly on medium-density boards with one moderately-high peak cell.
    # test6 (density 0.36, peak 0.46) is the canonical bad case: at 1.2×
    # scale, SA chased overflow at the cost of BOTH HPWL and the RUDY
    # peak itself (peak rose from 0.38 → 0.61 — an actual routability
    # regression at the hotspot). Raising the peak threshold from 0.40
    # to 0.50 lets test6 fall to baseline 1.0× while keeping th_sensor
    # (peak 0.52), tc_logger_silabs (peak 1.12), and v_d_afe (peak 0.62)
    # at 1.2× — those boards responded well to 1.2× amplification.
    #
    # The peak-rising failure mode (SA trading a higher peak for lower
    # overflow) is addressed at the THRESHOLD level here, not via a cost
    # formula change. A quadratic peak term `0.5*peak*peak` was tried in
    # rudy_congestion_penalty — it fixed test6 but regressed th_sensor,
    # test5, and tc_logger_silabs (both HPWL and peak worsened under the
    # altered cost landscape), so it was reverted. See engine/congestion.py.
    #
    # Calibration across 15 boards (ON vs OFF HPWL %), per TUNING_SUMMARY.md:
    #   v1: average -2.2%; test6 +17% (the bad case above)
    #   v2: average -3.2%; 13 wins, 1 neutral, 1 near-neutral (test6 +1.4%,
    #       was +17%), 1 structural loss (PModBoard +15.2% but peak drops —
    #       a routability win). Zero residual overlaps on any board.
    #
    # The dense/congested scale stays at 1.2× (not 1.5×) — 1.5× was too
    # aggressive on test6: at effective weight 1.5, SA thrashed and HPWL
    # regressed by +42mm with high variance.
    if interior_density < 0.30 and raw_rudy_peak < 0.10:
        rudy_scale = 0.3
    elif interior_density > 0.45:
        rudy_scale = 1.2
    elif raw_rudy_peak > 0.50:
        rudy_scale = 1.2
    else:
        rudy_scale = 1.0
    effective_rudy_weight = rudy_weight * rudy_scale
    if verbose:
        print(f"  Density-adaptive RUDY: density={interior_density:.2f} "
              f"raw_peak={raw_rudy_peak:.3f} -> scale={rudy_scale:.1f}× "
              f"(effective weight={effective_rudy_weight:.2f}, config={rudy_weight:.2f})")

    # Cap→IC pairs for the attraction term — the same deterministic
    # rail-adjacent assignment build_macros classified. Only freed caps
    # whose assigned IC actually HAS a macro participate (fixed/absent
    # ICs give the term nothing to pull toward).
    cap_pairs: dict[str, str] = {}
    if cap_attraction_weight > 0:
        from assign.assign_caps import rail_adjacent_to_ic
        ra_map = rail_adjacent_to_ic(model)
        movable_leader_refs = {
            m.leader.ref for m in interior_macros if not m.is_fixed
        }
        cap_pairs = {
            cap: ic for cap, ic in ra_map.items() if ic in movable_leader_refs
        }
        if verbose:
            print(f"  Cap attraction: weight={cap_attraction_weight:.2f}, "
                  f"{len(cap_pairs)} rail-adjacent cap->IC pair(s)")

    # Clearance (routing halo) density scaling. On sparse boards the 1mm
    # target between macro bboxes is cheap to satisfy — full weight gives
    # every pair a trace lane. As density rises the target eventually
    # becomes unachievable (components simply don't fit with 1mm lanes),
    # so the term's weight tapers to a weak tie-breaker. Anchors are
    # conservative because macro bboxes already carry courtyard margins
    # (0.5-1.5mm per side, utils/courtyard.py): raw fill density
    # OVERSTATES physical occupancy, and the calibration boards held a
    # full halo well past 0.45 (cbb: density 0.30, halo achieved with
    # HPWL actually improving; test4: density 0.51, deficit cleared on
    # most pairs). Unlike RUDY's scale, this only DECREASES from the
    # configured weight — a dense board never amplifies the halo term.
    if clearance_weight > 0:
        if interior_density <= 0.45:
            clearance_scale = 1.0
        elif interior_density >= 0.65:
            clearance_scale = 0.25
        else:
            # Linear taper between the two anchors.
            clearance_scale = 1.0 - 0.75 * (interior_density - 0.45) / 0.20
    else:
        clearance_scale = 0.0
    effective_clearance_weight = clearance_weight * clearance_scale
    if verbose and clearance_weight > 0:
        print(f"  Clearance halo: density={interior_density:.2f} "
              f"-> scale={clearance_scale:.2f}x "
              f"(effective weight={effective_clearance_weight:.2f}, "
              f"config={clearance_weight:.2f})")

    sa_result = run_macro_sa(
        model, interior_macros + fixed_macros, sa_bounds,
        iterations=sa_iterations, reheats=sa_reheats,
        alpha=alpha, beta=beta, gamma=gamma,
        seed=seed, verbose=verbose,
        net_weights=chain_net_weights,
        rudy_weight=effective_rudy_weight,
        pin_density_weight=pin_density_weight,
        delta=delta,
        rules=rules,
        cap_attraction_weight=cap_attraction_weight,
        cap_pairs=cap_pairs if cap_attraction_weight > 0 else None,
        clearance_weight=effective_clearance_weight,
        exclude_nets=exclude_nets,
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
    # (Re-using the sa_bounds_area/interior_macro_area/interior_density
    # computed above for the RUDY-weight scaling — they're the same
    # quantities.)

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
        # Report final RUDY + pin-density so the user can see whether SA +
        # legalization created or resolved routing choke points.
        from engine.congestion import (
            rudy_congestion_penalty,
            pin_density_penalty as _pin_density_penalty,
        )
        rudy_penalty_final, rudy_peak_final, rudy_avg_final, _ = rudy_congestion_penalty(model)
        pd_penalty_final, pd_peak_final, pd_avg_final, _ = _pin_density_penalty(model)
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
        if pd_penalty_final > 0:
            print(f"  Pin-density: penalty={pd_penalty_final:.3f} peak={pd_peak_final:.3f} "
                  f"avg={pd_avg_final:.3f}")
        elif pd_penalty_init > 0:
            print(f"  Pin-density: no hotspots remaining (was {pd_penalty_init:.3f} before SA)")
        if clearance_weight > 0:
            # Routing-halo state after legalization (which can shrink the
            # gaps SA opened — the legalizer only resolves true overlaps).
            from cost.cost import clearance_deficit as _clr_deficit
            clr_final = _clr_deficit(all_macros)
            print(f"  Clearance: residual deficit={clr_final:.2f}mm "
                  f"(halo target 1.0mm edge-to-edge)")

    return {
        "sa_result": sa_result,
        "legal_stats": legal_stats,
        "final_cost": final_cost,
        "n_interior_macros": len(interior_macros),
        "n_connector_macros": len(connector_macros),
    }
