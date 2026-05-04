#!/usr/bin/env python3
"""KiCad Smart Auto-Placer — CLI Interface.

Usage:
    python -m auto_placer place <input.kicad_pcb> [options]
    python -m auto_placer extract <input.kicad_pcb> [options]
    python -m auto_placer profiles
    python -m auto_placer cost <board_model.json> [options]

Phase 1 implements:
  - Board data extraction from .kicad_pcb → JSON intermediate model
  - Net-based clustering + grid placement
  - HPWL cost evaluation
  - Legalization pass
  - Placement application back to .kicad_pcb
"""

from __future__ import annotations

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.board_model import BoardModel
from parsers.kicad_parser import KiCadParser
from parsers.placement_writer import apply_placement, export_positions_json
from engine.net_clustering import cluster_components, compute_seed_positions
from engine.grid_placement import grid_place, edge_aware_grid_place
from engine.cost_function import CostFunction, total_hpwl, count_overlaps, count_out_of_bounds
from legalization.legalizer import legalize
from profiles.board_profiles import get_profile, list_profiles, BoardProfile
from utils.display import (
    print_board_summary,
    print_cost_breakdown,
    print_component_table,
    print_cluster_info,
)


def cmd_extract(args) -> None:
    """Extract board data from .kicad_pcb to JSON."""
    print(f"Loading: {args.input}")
    parser = KiCadParser(args.input)
    model = parser.parse()

    print_board_summary(model, "Extraction Results")
    print_component_table(model)

    # Save intermediate model
    output = args.output or args.input.replace(".kicad_pcb", "_model.json")
    model.to_json(output)
    print(f"Intermediate model saved to: {output}")


def cmd_place(args) -> None:
    """Run the full placement pipeline."""
    print(f"\n{'#' * 60}")
    print(f"  KiCad Smart Auto-Placer — Phase 1")
    print(f"{'#' * 60}\n")

    # Step 1: Extract
    print("Step 1: Extracting board data...")
    parser = KiCadParser(args.input)
    model = parser.parse()
    print_board_summary(model, "After Extraction")
    print_component_table(model)

    # Step 2: Select board profile
    print("Step 2: Selecting board profile...")
    profile = get_profile(args.profile)
    print(f"  Profile: {profile.display_name}")
    print(f"  Description: {profile.description}")
    print(f"  Cost weights: α={profile.alpha}, β={profile.beta}, γ={profile.gamma}, δ={profile.delta}")
    if profile.active_rules():
        print("  Active rules:")
        for rule in profile.active_rules():
            print(f"    • {rule.name} (weight={rule.weight}) — {rule.description}")
    print()

    # Step 3: Interactive tuning (if requested)
    if args.interactive:
        _interactive_tuning(profile)

    # Step 4: Net clustering
    print("Step 3: Computing net clusters...")
    clusters = cluster_components(model)
    print_cluster_info(clusters)

    # Step 5: Grid placement
    print("Step 4: Running grid placement...")
    if args.edge_aware:
        edge_aware_grid_place(model, margin=args.margin, spacing_factor=args.spacing)
        print("  Using edge-aware placement (connectors near edges)")
    else:
        grid_place(model, margin=args.margin, spacing_factor=args.spacing)
    print_board_summary(model, "After Grid Placement")

    # Step 6: Cost evaluation
    print("Step 5: Evaluating placement cost...")
    cost_fn = CostFunction(
        alpha=profile.alpha,
        beta=profile.beta,
        gamma=profile.gamma,
        delta=profile.delta,
    )
    costs = cost_fn.evaluate(model)
    print_cost_breakdown(costs, "Placement Cost (Pre-Legalization)")

    # Optional Phase 2 optimization
    if args.optimize:
        print("Step 6: Running Phase 2 optimizer (greedy swap)...")
        from engine.simple_optimizer import greedy_swap_optimize
        model = greedy_swap_optimize(model, cost_fn, max_iters=10)
        costs_opt = cost_fn.evaluate(model)
        print_cost_breakdown(costs_opt, "Placement Cost (Post-Optimization)")

    # Step 7: Legalization
    print("Step 7: Running legalization pass...")
    legalize(model, grid_mm=args.grid, verbose=True)
    print_board_summary(model, "After Legalization")

    # Step 8: Final cost evaluation
    costs_after = cost_fn.evaluate(model)
    print_cost_breakdown(costs_after, "Placement Cost (Post-Legalization)")

    # Show improvement
    improvement = costs["total"] - costs_after["total"]
    print(f"  Cost change from legalization: {improvement:+.2f}")

    # Step 9: Save outputs
    print("\nStep 7: Saving results...")

    # Save intermediate model
    model_json = args.output_json or args.input.replace(".kicad_pcb", "_placed_model.json")
    model.to_json(model_json)
    print(f"  Board model JSON: {model_json}")

    # Save position map
    pos_json = args.input.replace(".kicad_pcb", "_positions.json")
    export_positions_json(model, pos_json)
    print(f"  Positions JSON: {pos_json}")

    # Apply to PCB file
    if not args.dry_run:
        output_pcb = args.output or args.input.replace(".kicad_pcb", "_placed.kicad_pcb")
        apply_placement(model, args.input, output_pcb)
        print(f"  Placed PCB: {output_pcb}")
    else:
        print("  [DRY RUN] Not writing PCB file")

    # Final summary
    print(f"\n{'#' * 60}")
    print(f"  Placement Complete!")
    print(f"  Components placed: {len([c for c in model.components if not c.is_fixed])}")
    print(f"  Final HPWL: {costs_after['hpwl']:.2f}")
    print(f"  Remaining overlaps: {costs_after['overlap_count']}")
    print(f"  Out-of-bounds: {costs_after['oob_count']}")
    print(f"{'#' * 60}\n")


def cmd_cost(args) -> None:
    """Evaluate cost of an existing board model."""
    print(f"Loading: {args.input}")
    model = BoardModel.from_json(args.input)

    profile = get_profile(args.profile)
    cost_fn = CostFunction(
        alpha=profile.alpha,
        beta=profile.beta,
        gamma=profile.gamma,
        delta=profile.delta,
    )
    costs = cost_fn.evaluate(model)
    print_cost_breakdown(costs)
    print_component_table(model)


def cmd_profiles(args) -> None:
    """List available board profiles."""
    print(f"\n{'=' * 60}")
    print("  Available Board Profiles")
    print(f"{'=' * 60}\n")
    for profile in list_profiles():
        print(f"  {profile.name}")
        print(f"    Display: {profile.display_name}")
        print(f"    Desc:    {profile.description}")
        print(f"    Weights: α={profile.alpha}, β={profile.beta}, γ={profile.gamma}, δ={profile.delta}")
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

    # Tune cost weights
    for attr, label in [("alpha", "HPWL (α)"), ("beta", "Overlap (β)"), ("gamma", "Boundary (γ)"), ("delta", "Constraint (δ)")]:
        current = getattr(profile, attr)
        val = input(f"  {label} weight [{current}]: ").strip()
        if val:
            try:
                setattr(profile, attr, float(val))
            except ValueError:
                print(f"    Invalid value, keeping {current}")

    # Tune rule weights
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
        prog="auto_placer",
        description="KiCad Smart Auto-Placer — Constraint-aware PCB auto-placement",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ---- extract ----
    p_extract = subparsers.add_parser("extract", help="Extract board data to JSON")
    p_extract.add_argument("input", help="Path to .kicad_pcb file")
    p_extract.add_argument("-o", "--output", help="Output JSON path")

    # ---- place ----
    p_place = subparsers.add_parser("place", help="Run placement pipeline")
    p_place.add_argument("input", help="Path to .kicad_pcb file")
    p_place.add_argument("-o", "--output", help="Output .kicad_pcb path")
    p_place.add_argument("--output-json", help="Output board model JSON path")
    p_place.add_argument(
        "-p", "--profile", default="generic",
        choices=["mcu_peripheral", "power_supply", "rf_frontend", "mixed_signal", "generic"],
        help="Board profile (default: generic)",
    )
    p_place.add_argument("-m", "--margin", type=float, default=5.0, help="Board edge margin in mm (default: 5.0)")
    p_place.add_argument("-s", "--spacing", type=float, default=1.3, help="Component spacing factor (default: 1.3)")
    p_place.add_argument("-g", "--grid", type=float, default=0.1, help="Legalization grid in mm (default: 0.1)")
    p_place.add_argument("--edge-aware", action="store_true", help="Use edge-aware placement for connectors")
    p_place.add_argument("--interactive", action="store_true", help="Interactive profile tuning")
    p_place.add_argument("--optimize", action="store_true", help="Run Phase 2 optimizer after grid placement")
    p_place.add_argument("--dry-run", action="store_true", help="Don't write PCB output file")

    # ---- cost ----
    p_cost = subparsers.add_parser("cost", help="Evaluate placement cost")
    p_cost.add_argument("input", help="Path to board model JSON")
    p_cost.add_argument("-p", "--profile", default="generic", help="Board profile")

    # ---- profiles ----
    subparsers.add_parser("profiles", help="List available board profiles")

    args = parser.parse_args()

    if args.command == "extract":
        cmd_extract(args)
    elif args.command == "place":
        cmd_place(args)
    elif args.command == "cost":
        cmd_cost(args)
    elif args.command == "profiles":
        cmd_profiles(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
