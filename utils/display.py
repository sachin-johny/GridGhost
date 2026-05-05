"""Utility functions for the auto-placer."""

from __future__ import annotations

import math
from models.board_model import BoardModel, Component


def print_board_summary(model: BoardModel, title: str = "Board Summary") -> None:
    """Print a formatted summary of the board model."""
    stats = model.stats()
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"  Board size:       {stats['board_size']}")
    print(f"  Total components: {stats['total_components']}")
    print(f"    Movable:        {stats['movable_components']}")
    print(f"    Fixed:          {stats['fixed_components']}")
    print(f"    Top layer:      {stats['top_components']}")
    print(f"    Bottom layer:   {stats['bottom_components']}")
    print(f"  Total nets:       {stats['total_nets']}")
    print(f"  Overlaps:         {stats['overlap_count']}")
    print(f"  Overlap area:     {stats['overlap_area_total']:.2f} mm²")
    print(f"  Out-of-bounds:    {stats['out_of_bounds']}")
    print(f"{'=' * 60}\n")


def print_cost_breakdown(costs: dict, title: str = "Cost Breakdown") -> None:
    """Print a formatted cost breakdown."""
    print(f"\n{'-' * 50}")
    print(f"  {title}")
    print(f"{'-' * 50}")
    print(f"  HPWL:              {costs['hpwl']:.2f}")
    print(f"  Overlap penalty:   {costs['overlap']:.2f}")
    print(f"  Boundary penalty:  {costs['boundary']:.2f}")
    print(f"  Constraint penalty:{costs['constraint']:.2f}")
    print(f"  {'-' * 29}")
    print(f"  TOTAL COST:        {costs['total']:.2f}")
    print(f"  Overlaps:          {costs['overlap_count']}")
    print(f"  Out-of-bounds:     {costs['oob_count']}")
    print(f"{'-' * 50}\n")


def print_component_table(model: BoardModel, show_pads: bool = False) -> None:
    """Print a table of all components with their positions."""
    print(f"\n{'Ref':<8} {'Type':<12} {'Layer':<7} {'X':>8} {'Y':>8} {'Rot':>5} {'W':>6} {'H':>6} {'Fixed':>6} {'Nets':>5}")
    print("-" * 85)
    for c in sorted(model.components, key=lambda c: c.ref):
        print(
            f"{c.ref:<8} {c.component_type:<12} {c.layer:<7} "
            f"{c.x:>8.2f} {c.y:>8.2f} {c.rotation:>5.0f} "
            f"{c.width:>6.1f} {c.height:>6.1f} "
            f"{'Yes' if c.is_fixed else 'No':>6} {len(c.nets):>5}"
        )
    print()


def print_cluster_info(clusters: list[list[str]]) -> None:
    """Print clustering results."""
    print(f"\n{'-' * 50}")
    print(f"  Net Clustering Results ({len(clusters)} clusters)")
    print(f"{'-' * 50}")
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {i}: [{', '.join(cluster)}]")
    print(f"{'-' * 50}\n")
