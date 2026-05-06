#!/usr/bin/env python3
"""KiCad Smart Auto-Placer — CLI Interface.

Usage:
    python -m auto_placer place <input.kicad_pcb> [options]
    python -m auto_placer extract <input.kicad_pcb> [-o output.json]
    python -m auto_placer profiles
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.board_model import BoardModel
from parsers.kicad_parser import KiCadParser
from parsers.placement_writer import apply_placement, export_positions_json
from engine.net_clustering import cluster_components, compute_seed_positions
from engine.grid_placement import grid_place, force_directed_place
from engine.cost_function import CostFunction, total_hpwl, count_overlaps, count_out_of_bounds
from legalization.legalizer import legalize
from profiles.board_profiles import get_profile, list_profiles, BoardProfile
from utils.display import (
    print_board_summary,
    print_cost_breakdown,
    print_component_table,
    print_cluster_info,
)

ALGORITHMS = ("force-directed", "grid")


def cmd_extract(args) -> None:
    """Extract board data from .kicad_pcb to JSON."""
    print(f"Loading: {args.input}")
    parser = KiCadParser(args.input)
    model = parser.parse()

    print_board_summary(model, "Extraction Results")
    print_component_table(model)

    output = args.output or args.input.replace(".kicad_pcb", "_model.json")
    model.to_json(output)
    print(f"Intermediate model saved to: {output}")


def cmd_place(args) -> None:
    """Run the full placement pipeline."""
    print(f"\n{'#' * 60}")
    print(f"  KiCad Smart Auto-Placer")
    print(f"{'#' * 60}\n")

    # Step 1: Extract
    print("Step 1: Extracting board data...")
    parser = KiCadParser(args.input, bbox_margin=0.5)
    model = parser.parse()
    print_board_summary(model, "After Extraction")
    print_component_table(model)

    # Step 2: Select board profile
    print("Step 2: Selecting board profile...")
    profile = get_profile(args.profile)
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

    if algorithm == "force-directed":
        print("  Algorithm: force-directed (attractive + repulsive forces)")
        force_directed_place(model, margin=args.margin)
    else:
        print("  Algorithm: grid (cluster-based seed placement)")
        grid_place(model, margin=args.margin)

    print_board_summary(model, "After Placement")

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

    # Step 7: Legalization
    print("Step 6: Running legalization...")
    legalize(model, grid_mm=0.1, verbose=True)
    print_board_summary(model, "After Legalization")

    # Step 8: Final cost evaluation
    costs_after = cost_fn.evaluate(model)
    print_cost_breakdown(costs_after, "Placement Cost (Post-Legalization)")
    improvement = costs["total"] - costs_after["total"]
    print(f"  Cost change from legalization: {improvement:+.2f}")

    # Step 9: Save outputs
    print("\nStep 7: Saving results...")

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
        prog="auto_placer",
        description="KiCad Smart Auto-Placer - Constraint-aware PCB auto-placement",
    )
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
        "-p", "--profile", default="generic",
        choices=["mcu_peripheral", "power_supply", "rf_frontend", "mixed_signal", "generic", "small_board"],
        help="Board profile (default: generic)",
    )
    p_place.add_argument("-a", "--algorithm", default="force-directed", choices=ALGORITHMS,
                         help="Placement algorithm (default: force-directed)")
    p_place.add_argument("-m", "--margin", type=float, default=5.0,
                         help="Board edge margin in mm (default: 5.0)")
    p_place.add_argument("--dry-run", action="store_true", help="Don't write PCB output file")
    p_place.add_argument("--interactive", action="store_true", help="Interactive profile weight tuning")

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
