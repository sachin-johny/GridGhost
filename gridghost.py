#!/usr/bin/env python3
"""GridGhost — CLI Interface.

Usage:
    python gridghost.py place <input.kicad_pcb> [options]
    python gridghost.py extract <input.kicad_pcb> [-o output.json]
    python gridghost.py profiles
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.board_model import BoardModel
from parsers.kicad_parser import KiCadParser
from parsers.placement_writer import apply_placement, export_positions_json
from profiles.board_profiles import get_profile, list_profiles, BoardProfile
from utils.display import (
    print_board_summary,
    print_cost_breakdown,
    print_component_table,
    print_cluster_info,
)
from config import load_config, Config

# Legacy-only modules are lazy-imported inside cmd_place()'s legacy branch to
# avoid pulling in the ~8000-line legacy stack (engine/annealer.py,
# engine/smart_placement.py, legalization/legalizer.py) on every macro-v2 run.
# `write_debug_bboxes` is also legacy-only — imported lazily where used.

ALGORITHMS = ("force-directed", "grid")


def _ensure_board_capacity(model: BoardModel, target_density: float | None = None) -> None:
    """Expand board if component density exceeds target.

    Finding 6 fix: ``target_density`` defaults to the shared
    ``utils.density.target_pack_density()`` value (0.55) so this
    routine agrees with the outline-inference and legalizer routines.
    """
    if target_density is None:
        from utils.density import target_pack_density
        target_density = target_pack_density()
    if getattr(model, 'user_defined_outline', False):
        # Respect user-drawn Edge.Cuts — warn but don't expand
        board = model.board
        board_area = board.width * board.height
        if board_area > 0:
            comp_area = sum(c.effective_width * c.effective_height for c in model.components)
            density = comp_area / board_area
            if density > target_density:
                print(f"  Note: density {density:.2f} > {target_density:.2f} "
                      f"(user Edge.Cuts respected, no expansion)")
        return

    import math
    board = model.board
    board_area = board.width * board.height
    if board_area <= 0:
        return

    comp_area = sum(c.effective_width * c.effective_height for c in model.components)
    current_density = comp_area / board_area

    if current_density <= target_density:
        return

    scale = math.sqrt(current_density / target_density)
    new_w = board.width * scale
    new_h = board.height * scale

    cx = (board.x_min + board.x_max) / 2.0
    cy = (board.y_min + board.y_max) / 2.0

    from models.board_model import BoardOutline
    model.board = BoardOutline(
        x_min=cx - new_w / 2.0,
        y_min=cy - new_h / 2.0,
        x_max=cx + new_w / 2.0,
        y_max=cy + new_h / 2.0,
    )

    print(f"  Board expanded for density: {board.width:.1f}x{board.height:.1f} -> "
          f"{new_w:.1f}x{new_h:.1f}mm "
          f"(density {current_density:.2f} -> {target_density:.2f})")


def cmd_extract(args) -> None:
    """Extract board data from .kicad_pcb to JSON."""
    cfg = load_config(args.config)
    print(f"Loading: {args.input}")
    parser = KiCadParser(args.input, bbox_margin=cfg.parser.bbox_margin)
    model = parser.parse()

    print_board_summary(model, "Extraction Results")
    print_component_table(model)

    output = args.output or args.input.replace(".kicad_pcb", "_model.json")
    model.to_json(output)
    print(f"Intermediate model saved to: {output}")


def cmd_place(args) -> None:
    """Run the full placement pipeline."""
    cfg = load_config(args.config)

    # Seed PRNG for deterministic placement. SA uses Python's random
    # module; without a seed, each run produces a different placement.
    # Default seed=42 (matches scripts/visualize_placement.py). Use
    # --seed 0 to disable seeding (truly random).
    import random as _random
    seed = getattr(args, 'seed', 42)
    if seed != 0:
        _random.seed(seed)
        if 'numpy' in sys.modules:
            try:
                import numpy as np
                np.random.seed(seed)
            except Exception:
                pass
        print(f"  Random seed: {seed} (deterministic mode)")

    print(f"\n{'#' * 60}")
    print(f"  GridGhost")
    print(f"{'#' * 60}\n")

    # Step 1: Extract
    print("Step 1: Extracting board data...")
    parser = KiCadParser(args.input, bbox_margin=cfg.parser.bbox_margin)
    model = parser.parse()
    print_board_summary(model, "After Extraction")
    print_component_table(model)

    # Step 1.5: Ensure board is large enough for component density
    # (Finding 6: target_density defaults to shared 0.55 inside _ensure_board_capacity)
    _ensure_board_capacity(model)

    # Step 2: Select board profile
    print("Step 2: Selecting board profile...")
    profile_name = args.profile
    if profile_name == "auto":
        ic_types = {'ic', 'mcu', 'regulator'}
        has_ics = any(getattr(c, 'component_type', '') in ic_types for c in model.components)
        profile_name = "mcu_peripheral" if has_ics else "generic"
        print(f"  Auto-detected: {'MCU/peripheral (ICs found)' if has_ics else 'generic (no ICs)'}")
    profile = get_profile(profile_name)
    print(f"  Profile: {profile.display_name}")
    print(f"  Weights: alpha={profile.alpha}, beta={profile.beta}, gamma={profile.gamma}, delta={profile.delta}")
    if profile.active_rules():
        print("  Active rules:")
        for rule in profile.active_rules():
            print(f"    - {rule.name} (weight={rule.weight})")
    print()

    # Step 3: Interactive tuning (if requested — legacy-only effect).
    if args.interactive:
        _interactive_tuning(profile)

    # Step 4: Placement algorithm
    print("Step 3: Running placement algorithm...")
    algorithm = args.algorithm
    pcfg = cfg.placement

    if getattr(args, "macro_v2", True):
        # Macro-first pipeline: cap-IC as a true rigid macro through every
        # stage. Default since 2026-07. Use --no-macro-v2 to fall back to
        # the legacy grid pipeline (engine/smart_placement.py).
        from place.pipeline import place_v2
        print("  Algorithm: macro-v2 (rigid cap-IC macros)")
        margin = args.margin if args.margin is not None else pcfg.margin

        # SA iteration budget — Finding 2 fix.
        # The previous code did `cfg.annealer.max_iterations // 2`, which
        # silently ran SA at 10% of place_v2's own default (1500) and as
        # a flat constant regardless of board size. The macro-v2 SA cost
        # is O(iterations × macros); on dense boards the budget must
        # scale with macro count or SA plateaus before convergence and
        # leaves residual overlaps (test6: 9→1, test5: 3→1 just from
        # fixing this).
        #
        # Override with --sa-iterations N if you want manual control.
        if args.sa_iterations is not None:
            sa_iters = args.sa_iterations
        else:
            n_macros = sum(1 for c in model.components if not c.is_fixed)
            sa_iters = max(1500, 25 * n_macros)
        print(f"  SA iterations: {sa_iters} (macro_count={sum(1 for c in model.components if not c.is_fixed)})")

        grid_mm = args.grid_mm if args.grid_mm is not None else cfg.legalization.grid_mm
        place_v2_result = place_v2(
            model,
            margin=margin,
            grid_mm=grid_mm,
            sa_iterations=sa_iters,
            sa_reheats=cfg.annealer.reheat_count,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            connector_mating_margin=args.connector_mating_margin,
            seed=seed,
            verbose=getattr(args, "verbose", False),
            rudy_weight=getattr(args, "rudy_weight", None),
            pin_density_weight=getattr(args, "pin_density_weight", None),
            use_abacus=(getattr(args, "legalizer", "heuristic") == "abacus"),
            use_sa_polish=(getattr(args, "legalizer", "heuristic") == "sa_polish"),
            delta=getattr(args, "delta", 0.0),
            rules=profile.rules if getattr(args, "delta", 0.0) > 0 else None,
        )
    elif algorithm == "force-directed":
        # Legacy-only: --algorithm force-directed
        from engine.grid_placement import force_directed_place
        print("  Algorithm: force-directed (attractive + repulsive forces)")
        force_directed_place(
            model,
            margin=args.margin if args.margin is not None else pcfg.margin,
            iterations=pcfg.force_iterations,
            k_attract=pcfg.force_k_attract,
            k_repel=pcfg.force_k_repel,
            min_spacing=pcfg.force_min_spacing,
            dt=pcfg.force_dt,
        )
    else:
        # Legacy-only: --algorithm grid
        from engine.net_clustering import cluster_components
        from engine.smart_placement import smart_grid_place
        print("  Algorithm: grid (cluster-based seed placement)")
        # Net clustering is only consumed by the legacy grid pipeline —
        # macro-v2 has its own clustering via place/cluster.py.
        print("  Computing net clusters...")
        clusters = cluster_components(model)
        print_cluster_info(clusters)
        smart_grid_place(
            model,
            margin=args.margin if args.margin is not None else pcfg.margin,
            spacing_factor=pcfg.spacing_factor,
            rules=profile.rules,
        )

    print_board_summary(model, "After Placement")

    if getattr(args, "macro_v2", False):
        # Macro-v2 pipeline does its own SA + legalize. Skip the legacy
        # post-processing (Steps 4.5–7.7) and go straight to saving.
        legal_stats = place_v2_result.get("legal_stats", {})
        residual_overlaps = legal_stats.get("residual_overlaps", 0)
        boundary_failures = legal_stats.get("boundary_failures", 0)
        placement_invalid = residual_overlaps > 0 or boundary_failures > 0
        if placement_invalid:
            print(
                f"\n  WARNING: legalizer could not produce a valid placement — "
                f"{residual_overlaps} overlapping footprint pair(s), "
                f"{boundary_failures} out-of-bounds footprint(s) remain.\n"
                f"  The written board is NOT physically valid as placed. Try "
                f"--sa-iterations with a larger value, a lower --beta board "
                f"density, or manually resolve the flagged components in "
                f"KiCad before fabrication."
            )

        print("\nStep 8: Saving results (macro-v2 pipeline — skipping legacy SA/legalize)")
        model_json = args.input.replace(".kicad_pcb", "_placed_model.json")
        model.to_json(model_json)
        print(f"  Board model JSON: {model_json}")

        pos_json = args.input.replace(".kicad_pcb", "_positions.json")
        export_positions_json(model, pos_json)
        print(f"  Positions JSON: {pos_json}")

        if not args.dry_run:
            output_pcb = args.output or args.input.replace(".kicad_pcb", "_placed.kicad_pcb")
            apply_placement(model, args.input, output_pcb)
            print(f"  Placed PCB: {output_pcb}")
        else:
            print("  [DRY RUN] Not writing PCB file")

        print(f"\n{'#' * 60}")
        if placement_invalid:
            print(f"  Placement FINISHED WITH ERRORS (macro-v2)")
        else:
            print(f"  Placement Complete! (macro-v2)")
        print(f"  Components placed: {len([c for c in model.components if not c.is_fixed])}")
        print(f"{'#' * 60}\n")
        if placement_invalid:
            sys.exit(1)
        return

    # Step 4.5: Pre-place decoupling caps adjacent to their ICs.
    # Gives SA a good starting configuration so the group-aware density
    # grid has correct grouping from t=0. Gated on decoupling_proximity
    # rule (only mcu_peripheral enables it today).
    if any(r.name == 'decoupling_proximity' and r.enabled for r in profile.rules):
        from engine.smart_placement import _is_vertical_connector, _compute_interior_bbox
        from engine.placement_prepass import preplace_caps_near_ics
        interior_comps_pp = [
            c for c in model.components
            if not c.is_fixed and (
                getattr(c, 'component_type', '') != 'connector'
                or _is_vertical_connector(c)
            )
        ]
        interior_bbox_pp = (
            _compute_interior_bbox(interior_comps_pp, model.board, 5.0)
            if interior_comps_pp else None
        )
        n_moved = preplace_caps_near_ics(
            model, profile.rules, interior_bbox=interior_bbox_pp, verbose=False
        )
        if n_moved:
            print(f"  Pre-placed {n_moved} decoupling cap(s) near their ICs")

    # Step 5.5: Post-placement optimization
    # SA is ON by default; use --no-sa to disable and run greedy+swap only.
    # SA auto-disables on tiny (≤6 comps) and large (≥50 comps) boards.
    import engine.cost_state as cs
    from engine.annealer import run_sa, SAConfig
    from engine.cost_function import CostFunction, count_overlaps
    from engine.smart_placement import _is_vertical_connector, _compute_interior_bbox
    from legalization.legalizer import legalize
    cs.OVERLAP_WEIGHT = max(profile.beta, 25.0)   # strong — SA should avoid overlaps
    cs.BOUNDARY_WEIGHT = max(profile.gamma, 8.0)   # strong — prevent OOB during SA
    cs.CONSTRAINT_WEIGHT = profile.delta

    # Build SAConfig from config.json (single source of truth), with CLI overrides.
    # CLI args default to None → fall back to config.json values.
    sa_overrides = {"verbose": True, "skip_sa": args.no_sa}
    if args.sa_iterations is not None:
        sa_overrides["max_iterations"] = args.sa_iterations
    if args.sa_reheat is not None:
        sa_overrides["reheat_count"] = args.sa_reheat
    sa_config = SAConfig.from_config(cfg.annealer, **sa_overrides)

    if args.no_sa:
        print("Step 5: Running enhanced greedy+swap optimization (--no-sa)...")
    else:
        print("Step 5: Running SA + enhanced greedy optimization...")

    sa_result = run_sa(model, config=sa_config, verbose=True, rules=profile.rules)
    mode_label = "Greedy" if args.no_sa else "SA"
    print(f"  {mode_label}: cost {sa_result['initial_cost']:.1f} -> {sa_result['final_cost']:.1f} "
          f"(delta={sa_result['improvement']:.1f})")
    print(f"  {mode_label}: HPWL {sa_result['initial_hpwl']:.1f} -> {sa_result['final_hpwl']:.1f}")
    print(f"  {mode_label}: overlaps={sa_result['overlap_count']}")
    print_board_summary(model, "After Optimization")

    # Step 6: Cost evaluation
    print("Step 6: Evaluating placement cost...")
    cost_fn = CostFunction(
        alpha=profile.alpha,
        beta=profile.beta,
        gamma=profile.gamma,
        delta=profile.delta,
        rules=profile.rules,
    )
    costs = cost_fn.evaluate(model)
    print_cost_breakdown(costs, "Placement Cost (Pre-Legalization)")

    # Step 7: Legalization
    print("Step 7: Running legalization...")

    # Attach active rules to the model so the legalizer's cap-nudge pass
    # can use the decoupling_proximity constraint parameters.
    model.active_rules = profile.active_rules()

    lcfg = cfg.legalization
    n_total = len(model.components)
    adaptive_max_iter = max(100, min(800, n_total * 5))

    # Compute interior bbox for legalizer so it clamps interior components
    # away from edge connectors. Must be computed AFTER placement —
    # pre-placement ib captures the original tight cluster and would
    # clamp the legalizer back into it, causing massive overlaps on
    # dense boards (e.g. test4: 50→9 overlaps).
    interior_comps = [
        c for c in model.components
        if not c.is_fixed and (
            getattr(c, 'component_type', '') != 'connector'
            or _is_vertical_connector(c)
        )
    ]
    interior_bbox = _compute_interior_bbox(interior_comps, model.board, 5.0) if interior_comps else None

    legalize(model, grid_mm=lcfg.grid_mm, max_iterations=adaptive_max_iter,
             push_strength=lcfg.push_strength, verbose=True,
             use_abacus=True,
             interior_bbox=interior_bbox,
             max_bbox_expansions=lcfg.max_bbox_expansions,
             push_apart_hard_cap=lcfg.push_apart_hard_cap,
             spread_pass_enabled=lcfg.spread_pass_enabled,
             bbox_expansion_factor=lcfg.bbox_expansion_factor,
             bbox_expansion_density_threshold=lcfg.bbox_expansion_density_threshold,
             gradient_plateau_threshold=lcfg.gradient_plateau_threshold,
             gradient_history_window=lcfg.gradient_history_window,
             gradient_split=lcfg.gradient_split,
             density_push_min=lcfg.density_push_min,
             density_push_max=lcfg.density_push_max)
    print_board_summary(model, "After Legalization")

    # Step 7.5: Post-legalization overlap resolution
    # Use the legalizer's greedy resolver instead of SA's overlap resolver.
    # The SA resolver doesn't check if resolving one overlap creates new ones.
    # The legalizer's greedy resolver checks all neighbors before accepting moves.
    #
    # Cap-IC overlaps introduced by the cap-IC nudge are intentional (decoupling
    # caps placed adjacent to their ICs may share courtyard space).  Count
    # overlaps excluding cap↔IC pairs so the cleanup doesn't undo the nudge.
    from engine.constraint_evaluator import _build_decoupling_map
    final_decap_map = (_build_decoupling_map(model)
                       if getattr(model, 'active_rules', None) else None)
    cap_ic_pairs: set[tuple[str, str]] = set()
    if final_decap_map:
        for ic_ref, cap_refs in final_decap_map.items():
            for cap_ref in cap_refs:
                cap_ic_pairs.add((min(ic_ref, cap_ref), max(ic_ref, cap_ref)))

    def _overlaps_excluding_cap_ic(m: BoardModel) -> int:
        if not cap_ic_pairs:
            return count_overlaps(m)
        n = 0
        comps = m.components
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                a, b = comps[i], comps[j]
                if not a.overlaps(b):
                    continue
                key = (min(a.ref, b.ref), max(a.ref, b.ref))
                if key in cap_ic_pairs:
                    continue
                n += 1
        return n

    post_legal_overlaps = _overlaps_excluding_cap_ic(model)
    if post_legal_overlaps > 0:
        print(f"  Post-legalization overlap resolution ({post_legal_overlaps} "
              f"non-cap-IC overlaps)...")
        from legalization.legalizer import (
            _greedy_resolve, _cleanup_cap_ic_overlaps,
        )
        _greedy_resolve(model, lcfg.grid_mm, True, interior_bbox)
        # With the macro approach, caps follow their IC through greedy
        # resolve via propagate_ic_move. No displace/nudge needed.
        if final_decap_map and getattr(model, 'active_rules', None):
            _cleanup_cap_ic_overlaps(model, final_decap_map, interior_bbox,
                                     verbose=True)
        print_board_summary(model, "After Post-Legalization Overlap Resolution")

        # The greedy cleanup moves components to clear overlaps without much
        # HPWL care.  Run cell_slide + pair_swap now that the board is clean
        # to recover HPWL lost in the cleanup.
        from legalization.post_legalize import post_legalization_refine
        post_legalization_refine(
            model, lcfg.grid_mm, interior_bbox, True, final_decap_map)

        # Safety net: post_legalization_refine can re-introduce overlaps
        # (its cell_slide and pair_swap are HPWL-driven, not overlap-aware).
        # If it did, run one more greedy + any-cap-IC + IC-IC cleanup pass.
        from legalization.legalizer import (
            _greedy_resolve as _cli_greedy_resolve,
            _cleanup_any_cap_ic_overlaps, _resolve_ic_ic_overlaps,
            _enforce_boundary, _compute_overlap_stats,
        )
        from legalization.spatial_grid import SpatialGrid
        _grid = SpatialGrid.from_components(list(model.components), model.board)
        _grid.build(list(model.components))
        _post_refine_overlaps, _ = _compute_overlap_stats(model, _grid)
        if _post_refine_overlaps > 0:
            print(f"  Post-refine safety net: {_post_refine_overlaps} overlaps "
                  f"re-introduced by HPWL recovery, running final cleanup")
            _cli_greedy_resolve(model, lcfg.grid_mm, True, interior_bbox)
            _enforce_boundary(model, interior_bbox)
            _cleanup_any_cap_ic_overlaps(model, interior_bbox, verbose=True)
            _grid.build(list(model.components))
            _resolve_ic_ic_overlaps(model, interior_bbox, _grid, verbose=True)
            _enforce_boundary(model, interior_bbox)

    # Step 7.6: FINAL brute-force overlap check.
    # With the macro approach, caps follow their IC — no displacement needed.
    from legalization.legalizer import (
        _greedy_resolve as _final_greedy,
        _enforce_boundary as _final_boundary,
        _cleanup_cap_ic_overlaps as _final_cleanup,
    )
    _bf_overlaps = 0
    _comps = list(model.components)
    for _i in range(len(_comps)):
        for _j in range(_i + 1, len(_comps)):
            if _comps[_i].overlaps(_comps[_j]):
                _bf_overlaps += 1
    print(f"  Step 7.6 brute-force check: {_bf_overlaps} overlaps found")
    if _bf_overlaps > 0:
        print(f"  Final brute-force check: {_bf_overlaps} overlaps found, resolving")
        _final_greedy(model, lcfg.grid_mm, True, interior_bbox)
        _final_boundary(model, interior_bbox)
        if final_decap_map:
            _final_cleanup(model, final_decap_map, interior_bbox, verbose=True)
        _bf_after = 0
        _comps = list(model.components)
        for _i in range(len(_comps)):
            for _j in range(_i + 1, len(_comps)):
                if _comps[_i].overlaps(_comps[_j]):
                    _bf_after += 1
        if _bf_after > 0:
            print(f"  WARNING: {_bf_after} overlaps could not be resolved (board may be too dense)")

    # Step 7.7: FINAL FINAL brute-force overlap check (right before save).
    # Verify no overlaps exist. If any slipped through, resolve them now.
    _bf_final = 0
    _comps = list(model.components)
    for _i in range(len(_comps)):
        for _j in range(_i + 1, len(_comps)):
            if _comps[_i].overlaps(_comps[_j]):
                _bf_final += 1
    if _bf_final > 0:
        print(f"  Pre-save overlap check: {_bf_final} overlaps found, final resolution")
        from legalization.legalizer import (
            _greedy_resolve as _pre_save_greedy,
            _enforce_boundary as _pre_save_boundary,
            _cleanup_cap_ic_overlaps as _pre_save_cleanup,
        )
        _pre_save_greedy(model, lcfg.grid_mm, True, interior_bbox)
        _pre_save_boundary(model, interior_bbox)
        if final_decap_map:
            _pre_save_cleanup(model, final_decap_map, interior_bbox, verbose=True)
        # Last resort: if still overlapping, nudge the overlapping cap away
        _bf_check = 0
        _comps = list(model.components)
        for _i in range(len(_comps)):
            for _j in range(_i + 1, len(_comps)):
                if _comps[_i].overlaps(_comps[_j]):
                    _bf_check += 1
        if _bf_check > 0:
            print(f"  WARNING: {_bf_check} overlaps remain after all resolution passes")

    # Step 8: Final cost evaluation
    costs_after = cost_fn.evaluate(model)
    print_cost_breakdown(costs_after, "Placement Cost (Post-Legalization)")
    improvement = costs["total"] - costs_after["total"]
    print(f"  Cost change from legalization: {improvement:+.2f}")

    # Step 9: Save outputs
    print("\nStep 8: Saving results...")

    model_json = args.input.replace(".kicad_pcb", "_placed_model.json")
    model.to_json(model_json)
    print(f"  Board model JSON: {model_json}")

    pos_json = args.input.replace(".kicad_pcb", "_positions.json")
    export_positions_json(model, pos_json)
    print(f"  Positions JSON: {pos_json}")

    if not args.dry_run:
        output_pcb = args.output or args.input.replace(".kicad_pcb", "_placed.kicad_pcb")
        apply_placement(model, args.input, output_pcb)
        print(f"  Placed PCB: {output_pcb}")

        if getattr(args, 'debug_bbox', False):
            from parsers.placement_writer import write_debug_bboxes
            write_debug_bboxes(model, output_pcb, interior_bbox=interior_bbox)
            print(f"  Debug bboxes (board outline + interior bbox + component bboxes) written to Dwgs.User layer")
    else:
        print("  [DRY RUN] Not writing PCB file")
        if getattr(args, 'debug_bbox', False):
            print("  [DRY RUN] Skipping debug bboxes (need PCB file to write to)")

    # Final summary
    print(f"\n{'#' * 60}")
    print(f"  Placement Complete!")
    print(f"  Components placed: {len([c for c in model.components if not c.is_fixed])}")
    print(f"  Final HPWL: {costs_after['hpwl']:.2f}")
    print(f"  Remaining overlaps: {costs_after['overlap_count']}")
    print(f"  Out-of-bounds: {costs_after['oob_count']}")
    print(f"{'#' * 60}\n")


def cmd_profiles(args) -> None:
    """List available board profiles."""
    print(f"\n{'=' * 60}")
    print("  Available Board Profiles")
    print(f"{'=' * 60}\n")
    for profile in list_profiles():
        print(f"  {profile.name}")
        print(f"    Display: {profile.display_name}")
        print(f"    Desc:    {profile.description}")
        print(f"    Weights: alpha={profile.alpha}, beta={profile.beta}, gamma={profile.gamma}, delta={profile.delta}")
        if profile.rules:
            print(f"    Rules:")
            for rule in profile.rules:
                status = "ON" if rule.enabled else "OFF"
                print(f"      [{status}] {rule.name} (weight={rule.weight})")
        print()


def _interactive_tuning(profile: BoardProfile) -> None:
    """Interactive CLI for tuning profile weights and rule priorities."""
    print("\n  Interactive Profile Tuning")
    print("  (Press Enter to keep current value)\n")

    for attr, label in [("alpha", "HPWL (alpha)"), ("beta", "Overlap (beta)"), ("gamma", "Boundary (gamma)"), ("delta", "Constraint (delta)")]:
        current = getattr(profile, attr)
        val = input(f"  {label} weight [{current}]: ").strip()
        if val:
            try:
                setattr(profile, attr, float(val))
            except ValueError:
                print(f"    Invalid value, keeping {current}")

    if profile.rules:
        print("\n  Constraint Rule Weights:")
        for rule in profile.rules:
            val = input(f"  {rule.name} weight [{rule.weight}] (desc: {rule.description}): ").strip()
            if val:
                try:
                    rule.weight = float(val)
                except ValueError:
                    print(f"    Invalid value, keeping {rule.weight}")

    print()


def main():
    parser = argparse.ArgumentParser(
        prog="gridghost",
        description="GridGhost - Constraint-aware PCB auto-placement for KiCad",
    )
    parser.add_argument("--config", default=None, help="Path to config.json (default: config.json in script dir)")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # extract
    p_extract = subparsers.add_parser("extract", help="Extract board data to JSON")
    p_extract.add_argument("input", help="Path to .kicad_pcb file")
    p_extract.add_argument("-o", "--output", help="Output JSON path")

    # place
    p_place = subparsers.add_parser("place", help="Auto-place components on a PCB")
    p_place.add_argument("input", help="Path to .kicad_pcb file")
    p_place.add_argument("-o", "--output", help="Output .kicad_pcb path")
    p_place.add_argument(
        "-p", "--profile", default="auto",
        choices=["auto", "mcu_peripheral", "power_supply", "rf_frontend", "mixed_signal", "generic", "small_board"],
        help="Board profile (default: auto — selects mcu_peripheral if ICs detected, otherwise generic)",
    )
    p_place.add_argument("-a", "--algorithm", default="grid", choices=ALGORITHMS,
                         help="Placement algorithm (default: grid)")
    p_place.add_argument("-m", "--margin", type=float, default=None,
                         help="Board edge margin in mm (default: from config, else 5.0)")
    p_place.add_argument("--dry-run", action="store_true", help="Don't write PCB output file")
    p_place.add_argument("--interactive", action="store_true", help="Interactive profile weight tuning")
    p_place.add_argument("--no-sa", action="store_true",
                         help="Disable global SA optimization (default: SA enabled; auto-disables on tiny/large boards)")
    p_place.add_argument("--sa-iterations", type=int, default=None,
                         help="Max SA temperature steps. Macro-v2 default: "
                              "max(1500, 25*N) where N=moving macro count. "
                              "Legacy default: from config.json "
                              "(annealer.max_iterations).")
    p_place.add_argument("--sa-reheat", type=int, default=None,
                         help="Number of SA reheat rounds. Legacy path uses "
                              "this flag directly; macro-v2 always uses "
                              "config.json annealer.reheat_count (currently 3) "
                              "and ignores this flag.")
    p_place.add_argument("--debug-bbox", action="store_true", help="Draw component bounding boxes on Dwgs.User layer for visual debugging")
    p_place.add_argument("--macro-v2", action=argparse.BooleanOptionalAction, default=True,
                         help="Use the macro-first placement pipeline (rigid cap-IC macros). Default: on. Use --no-macro-v2 for the legacy grid pipeline.")
    p_place.add_argument("--seed", type=int, default=42,
                         help="Random seed for SA/placement determinism (default: 42; use --seed 0 for non-deterministic)")
    p_place.add_argument("--rudy-weight", type=float, default=None,
                         help="RUDY wire-density congestion penalty weight in SA cost function "
                              "(default: from config.json, currently 1.0 = enabled). "
                              "When > 0, SA gets gradient signal to spread macros away from "
                              "routing choke points (areas where many nets' bounding boxes "
                              "overlap). Pass 0 to disable. The verbose report always shows "
                              "RUDY regardless of this setting.")
    p_place.add_argument("--pin-density-weight", type=float, default=None,
                         help="Pin-density congestion penalty weight in SA cost function "
                              "(default: from config.json, currently 0.2 = enabled). "
                              "Complementary to --rudy-weight: catches pin-escape congestion "
                              "(dense clusters of signal pins) that RUDY's wire-density model "
                              "misses. Pass 0 to disable. The verbose report always shows "
                              "pin density regardless of this setting.")
    # --- macro-v2-only knobs (previously documented in README but never
    # registered as CLI args; place_v2() always accepted them). ---
    p_place.add_argument("--grid-mm", type=float, default=None,
                         help="Legalization grid pitch in mm for the macro-v2 "
                              "pipeline (default: from config.json "
                              "legalization.grid_mm, currently 0.5)")
    p_place.add_argument("--alpha", type=float, default=1.0,
                         help="HPWL weight in the macro-v2 cost function (default: 1.0)")
    p_place.add_argument("--beta", type=float, default=25.0,
                         help="Overlap penalty weight in the macro-v2 cost function (default: 25.0)")
    p_place.add_argument("--gamma", type=float, default=8.0,
                         help="Boundary penalty weight in the macro-v2 cost function (default: 8.0)")
    p_place.add_argument("--delta", type=float, default=0.0,
                         help="Constraint penalty weight in the macro-v2 cost function "
                              "(default: 0.0 = disabled). When > 0, the active profile's "
                              "constraint rules (decoupling proximity, crystal-MCU, thermal "
                              "grouping/separation, analog/digital separation, etc.) are "
                              "evaluated via engine/constraint_evaluator.py and added to the "
                              "SA cost. Opt-in — the default 0 preserves the current "
                              "macro-v2 behavior. See IMPROVEMENTS §2.4.")
    p_place.add_argument("--connector-mating-margin", type=float, default=5.0,
                         help="Edge offset (mm) for perimeter connectors in the macro-v2 pipeline (default: 5.0)")
    p_place.add_argument("-v", "--verbose", action="store_true",
                         help="Print per-stage cost breakdown (macro-v2 pipeline). "
                              "Legacy pipeline always prints verbose output regardless of this flag.")
    p_place.add_argument("--legalizer", choices=["abacus", "heuristic", "sa_polish"], default="heuristic",
                         help="Macro-v2 overlap-resolution strategy (default: heuristic). "
                              "'abacus' bridges macro-v2's rigid Macro objects into the "
                              "legacy row-based Abacus DP legalizer (place/abacus_bridge.py). "
                              "Measured on the bundled test boards it is WORSE than the "
                              "default on every board (more residual overlaps on 2/6, "
                              "10-30%% worse HPWL on all 6) — its row-binning model assumes "
                              "roughly uniform component heights, which doesn't hold once "
                              "small passives and large IC+cap macros share one row grid, "
                              "and it discards a large fraction of SA's Y-optimization doing "
                              "so (see place/abacus_bridge.py docstring). Kept available for "
                              "comparison/further work, not because it currently wins. "
                              "'sa_polish' runs a staged overlap-weighted simulated-"
                              "annealing pass (beta ramp + overlap-biased move selection, "
                              "place/sa_polish.py) instead of greedy push-apart/force-"
                              "spread. Better than the first two attempts: wins outright "
                              "on 3/6 boards (better HPWL, still overlap-free), near-ties "
                              "on 1/6, but still doesn't beat the heuristic on the board "
                              "this was built to fix (test6: same overlap count, worse "
                              "area). See place/sa_polish.py docstring for the numbers.")

    # profiles
    subparsers.add_parser("profiles", help="List available board profiles")

    args = parser.parse_args()

    if args.command == "extract":
        cmd_extract(args)
    elif args.command == "place":
        cmd_place(args)
    elif args.command == "profiles":
        cmd_profiles(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
