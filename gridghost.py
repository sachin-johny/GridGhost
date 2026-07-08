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
from parsers.placement_writer import apply_placement, export_positions_json, write_debug_bboxes
from engine.net_clustering import cluster_components, compute_seed_positions
from engine.grid_placement import grid_place, force_directed_place
from engine.smart_placement import smart_grid_place, _is_vertical_connector, _compute_interior_bbox
from engine.cost_function import CostFunction, total_hpwl, count_overlaps, count_out_of_bounds
from engine.annealer import run_sa, SAConfig
from engine.placement_prepass import preplace_caps_near_ics
from legalization.legalizer import legalize
from profiles.board_profiles import get_profile, list_profiles, BoardProfile
from utils.display import (
    print_board_summary,
    print_cost_breakdown,
    print_component_table,
    print_cluster_info,
)
from config import load_config, Config

ALGORITHMS = ("force-directed", "grid")


def _ensure_board_capacity(model: BoardModel, target_density: float = 0.35) -> None:
    """Expand board if component density exceeds target."""
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


def _apply_config_to_globals(cfg: Config) -> None:
    """Push config values into module-level constants used by cost_state."""
    import engine.cost_state as cs
    cs.OVERLAP_WEIGHT = cfg.cost.overlap_weight
    cs.BOUNDARY_WEIGHT = cfg.cost.boundary_weight
    # CONSTRAINT_WEIGHT is set by the profile's delta weight — not from config


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
    _apply_config_to_globals(cfg)

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
    _ensure_board_capacity(model, target_density=0.35)

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

    # Step 3: Interactive tuning (if requested)
    if args.interactive:
        _interactive_tuning(profile)

    # Step 4: Net clustering
    print("Step 3: Computing net clusters...")
    clusters = cluster_components(model)
    print_cluster_info(clusters)

    # Step 5: Placement algorithm
    print("Step 4: Running placement algorithm...")
    algorithm = args.algorithm
    pcfg = cfg.placement

    if algorithm == "force-directed":
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
        print("  Algorithm: grid (cluster-based seed placement)")
        smart_grid_place(
            model,
            margin=args.margin if args.margin is not None else pcfg.margin,
            spacing_factor=pcfg.spacing_factor,
            rules=profile.rules,
        )

    print_board_summary(model, "After Placement")

    # Step 4.5: Pre-place decoupling caps adjacent to their ICs.
    # Gives SA a good starting configuration so the group-aware density
    # grid has correct grouping from t=0. Gated on decoupling_proximity
    # rule (only mcu_peripheral enables it today).
    if any(r.name == 'decoupling_proximity' and r.enabled for r in profile.rules):
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
             interior_bbox=interior_bbox)
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
            _greedy_resolve, _nudge_caps_to_ics, _cleanup_cap_ic_overlaps,
        )
        _greedy_resolve(model, lcfg.grid_mm, True, interior_bbox)
        # The greedy cleanup treats every overlap as bad, so it can push caps
        # away from their ICs while clearing other overlaps.  Re-run the
        # cap-IC nudge to restore decoupling adjacency before HPWL recovery.
        # _nudge_caps_to_ics intentionally lets a cap overlap its own IC (its
        # overlap check excludes the IC), so the cleanup companion must run
        # afterward to relocate such caps to a free adjacent slot — otherwise
        # the nudge leaves illegal cap-IC overlaps behind.
        if final_decap_map and getattr(model, 'active_rules', None):
            _nudge_caps_to_ics(model, model.active_rules, interior_bbox,
                               verbose=True, cached_decap_map=final_decap_map)
            _cleanup_cap_ic_overlaps(model, final_decap_map, interior_bbox,
                                     verbose=True)
        print_board_summary(model, "After Post-Legalization Overlap Resolution")

        # The greedy cleanup moves components to clear overlaps without much
        # HPWL care.  Run cell_slide + pair_swap now that the board is clean
        # to recover HPWL lost in the cleanup.
        from legalization.post_legalize import post_legalization_refine
        post_legalization_refine(
            model, lcfg.grid_mm, interior_bbox, True, final_decap_map)

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
                         help="Max SA temperature steps (default: from config.json)")
    p_place.add_argument("--sa-reheat", type=int, default=None,
                         help="Number of SA reheat rounds (default: from config.json)")
    p_place.add_argument("--debug-bbox", action="store_true", help="Draw component bounding boxes on Dwgs.User layer for visual debugging")

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
