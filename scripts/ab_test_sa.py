#!/usr/bin/env python3
"""Controlled A/B test: does SA help on the PATCHED codebase?

Runs the full placement pipeline twice on each board with the same seed:
  A) skip_sa=True  (greedy + swap + greedy refinement only — the default)
  B) skip_sa=False (full SA + greedy + swap + greedy)

Compares HPWL, overlap count, oob count, RUDY, and wall-clock runtime.

This re-tests the 'SA didn't help' conclusion from the pre-patch era.
The patches changed SA's cost signal materially (HPWL consistency,
snapshot/restore correctness, keepout costs, RUDY as acceptance cost on
rf_frontend/mixed_signal, thermal separation on power_supply) — so the
old conclusion may not hold.

Usage:
    python scripts/ab_test_sa.py [--boards b1,b2,..] [--profiles p1,p2,..|auto]
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parsers.kicad_parser import KiCadParser
from engine.smart_placement import smart_grid_place, _is_vertical_connector, _compute_interior_bbox
from engine.cost_function import CostFunction, total_hpwl, count_overlaps, count_out_of_bounds
from engine.annealer import run_sa, SAConfig
from engine.placement_prepass import preplace_caps_near_ics
from engine.congestion import rudy_congestion_penalty
from legalization.legalizer import legalize
from profiles.board_profiles import get_profile
from config import load_config
import engine.cost_state as cs


def _expand_board(model):
    if not getattr(model, 'user_defined_outline', False):
        board = model.board
        board_area = board.width * board.height
        if board_area > 0:
            comp_area = sum(c.effective_width * c.effective_height for c in model.components)
            density = comp_area / board_area
            target = 0.35
            if density > target:
                scale = math.sqrt(density / target)
                from models.board_model import BoardOutline
                model.board = BoardOutline(
                    x_min=(board.x_min + board.x_max) / 2.0 - board.width * scale / 2.0,
                    y_min=(board.y_min + board.y_max) / 2.0 - board.height * scale / 2.0,
                    x_max=(board.x_min + board.x_max) / 2.0 + board.width * scale / 2.0,
                    y_max=(board.y_min + board.y_max) / 2.0 + board.height * scale / 2.0,
                )


def run_pipeline(pcb_path: str, profile_name: str, skip_sa: bool, seed: int = 42):
    """Run the full pipeline and return metrics dict."""
    random.seed(seed)

    cfg = load_config()
    cs.OVERLAP_WEIGHT = 25.0
    cs.BOUNDARY_WEIGHT = 8.0

    t_start = time.perf_counter()

    parser = KiCadParser(pcb_path, bbox_margin=cfg.parser.bbox_margin)
    model = parser.parse()
    _expand_board(model)

    # Resolve profile
    if profile_name == "auto":
        ic_types = {'ic', 'mcu', 'regulator'}
        has_ics = any(getattr(c, 'component_type', '') in ic_types for c in model.components)
        profile = get_profile("mcu_peripheral" if has_ics else "generic")
    else:
        profile = get_profile(profile_name)

    # Placement
    smart_grid_place(model, margin=cfg.placement.margin,
                    spacing_factor=cfg.placement.spacing_factor,
                    rules=profile.rules)

    # Pre-place caps
    if any(r.name == 'decoupling_proximity' and r.enabled for r in profile.rules):
        interior = [
            c for c in model.components
            if not c.is_fixed and (
                getattr(c, 'component_type', '') != 'connector'
                or _is_vertical_connector(c)
            )
        ]
        ib = _compute_interior_bbox(interior, model.board, 5.0) if interior else None
        preplace_caps_near_ics(model, profile.rules, interior_bbox=ib, verbose=False)

    # SA / greedy — THIS IS THE A/B VARIABLE
    cs.OVERLAP_WEIGHT = max(profile.beta, 25.0)
    cs.BOUNDARY_WEIGHT = max(profile.gamma, 8.0)
    cs.CONSTRAINT_WEIGHT = profile.delta

    sa_config = SAConfig(verbose=False, skip_sa=skip_sa)
    run_sa(model, config=sa_config, verbose=False, rules=profile.rules)

    # Legalize
    model.active_rules = profile.active_rules()
    interior = [
        c for c in model.components
        if not c.is_fixed and (
            getattr(c, 'component_type', '') != 'connector'
            or _is_vertical_connector(c)
        )
    ]
    ib = _compute_interior_bbox(interior, model.board, 5.0) if interior else None

    n_total = len(model.components)
    adaptive_max_iter = max(100, min(800, n_total * 5))
    legalize(model, grid_mm=cfg.legalization.grid_mm,
             max_iterations=adaptive_max_iter,
             push_strength=cfg.legalization.push_strength, verbose=False,
             use_abacus=True, interior_bbox=ib)

    t_end = time.perf_counter()

    # === Metrics ===
    # HPWL, overlaps, oob, RUDY (existing)
    hpwl = total_hpwl(model)
    overlaps = count_overlaps(model)
    oob = count_out_of_bounds(model)
    rudy_penalty, rudy_peak, _, _ = rudy_congestion_penalty(model, 2.0)

    # Constraint violations
    from engine.constraint_evaluator import evaluate_constraint_penalties
    _, cb = evaluate_constraint_penalties(model, profile.active_rules())

    # === Spread / centroid / coverage metrics (NEW) ===
    # These catch the "SA crams everything into one corner to minimise HPWL"
    # failure mode that HPWL alone rewards.
    board = model.board
    interior_comps = [
        c for c in model.components
        if not c.is_fixed and getattr(c, 'component_type', '') != 'connector'
    ]
    if interior_comps:
        xs = [c.x for c in interior_comps]
        ys = [c.y for c in interior_comps]
        n = len(interior_comps)
        cx_comp = sum(xs) / n
        cy_comp = sum(ys) / n
        cx_board = (board.x_min + board.x_max) / 2
        cy_board = (board.y_min + board.y_max) / 2
        # Centroid offset: distance from component centroid to board center.
        # HIGH = components bunched in one corner/edge (bad).
        # LOW = components centered on board (good).
        centroid_offset = math.hypot(cx_comp - cx_board, cy_comp - cy_board)
        # Standard deviation of x and y — measures spread.
        # LOW = components bunched together (bad — even if centered).
        # HIGH = components spread across board (good).
        std_x = math.sqrt(sum((x - cx_comp) ** 2 for x in xs) / n)
        std_y = math.sqrt(sum((y - cy_comp) ** 2 for y in ys) / n)
        # Coverage: bbox of interior components / board area (as %).
        # LOW = components occupy a small fraction of the board (bad —
        #       either bunched in a corner OR a thin sliver).
        # ~100% = components use the full board area (good).
        comp_w = max(xs) - min(xs)
        comp_h = max(ys) - min(ys)
        board_area = board.width * board.height
        coverage = (comp_w * comp_h / board_area * 100) if board_area > 0 else 0
    else:
        centroid_offset = std_x = std_y = coverage = 0.0

    # Gini coefficient of 10x10 cell-occupancy (lower = more uniform spread).
    # HIGH = components clustered into few cells (bad).
    # LOW = components uniformly distributed across cells (good).
    gini, empty_cells = _gini_density(model.components, board)

    return {
        'hpwl': hpwl,
        'overlaps': overlaps,
        'oob': oob,
        'rudy': rudy_penalty,
        'rudy_peak': rudy_peak,
        'runtime_s': t_end - t_start,
        'n_components': len(model.components),
        'constraint_violations': {k: v for k, v in cb.items() if v > 0.01},
        'profile': profile.name,
        # Spread/centroid metrics (NEW)
        'centroid_offset': centroid_offset,
        'std_x': std_x,
        'std_y': std_y,
        'coverage_pct': coverage,
        'gini': gini,
        'empty_cells': empty_cells,
    }


def _gini_density(components, board, grid_n=10):
    """Gini coefficient of cell-occupancy on grid_n x grid_n grid.

    Returns (gini_scaled, empty_cells).  Lower gini = more uniform spread.
    """
    cell_w = (board.x_max - board.x_min) / grid_n
    cell_h = (board.y_max - board.y_min) / grid_n
    if cell_w <= 0 or cell_h <= 0:
        return 0.0, 0
    grid = [0] * (grid_n * grid_n)
    total = 0
    for c in components:
        if c.is_fixed or getattr(c, 'component_type', '') == 'connector':
            continue
        gx = int((c.x - board.x_min) / cell_w)
        gy = int((c.y - board.y_min) / cell_h)
        gx = max(0, min(grid_n - 1, gx))
        gy = max(0, min(grid_n - 1, gy))
        grid[gx + gy * grid_n] += 1
        total += 1
    if total == 0:
        return 0.0, 0
    n = grid_n * grid_n
    s = sorted(grid)
    cum = sum((i + 1) * v for i, v in enumerate(s))
    gini = (2 * cum) / (n * total) - (n + 1) / n
    empty = sum(1 for v in grid if v == 0)
    return gini * total, empty


def main():
    ap = argparse.ArgumentParser(description="A/B test: SA vs no-SA on patched code")
    ap.add_argument("--boards", type=str, default="test5,th_sensor,PModBoard,voltage_datalogger_adc2",
                    help="Comma-separated board names (without .kicad_pcb)")
    ap.add_argument("--profiles", type=str, default="auto",
                    help="Comma-separated profiles, or 'auto'")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    boards = [b.strip() for b in args.boards.split(",") if b.strip()]
    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]

    # Try both test_pcbs/ and external_boards/rl_pcb/
    pcb_dirs = [
        ROOT / 'tests' / 'test_pcbs',
        ROOT / 'tests' / 'external_boards' / 'rl_pcb',
    ]

    print(f"A/B test: SA vs no-SA on patched code (seed={args.seed})")
    print(f"Metrics: HPWL (lower=better) | ovr/oob (0=better) | RUDY (lower=better) | "
          f"centroid_off (lower=better) | std_x,std_y (higher=better) | cov% (higher=better) | "
          f"Gini (lower=better) | empty (lower=better)")
    print()
    print(f"{'Board':<30} {'Prof':<8} {'N':>3} {'Mode':<6}  "
          f"{'HPWL':>7} {'ovr':>3} {'oob':>3} {'RUDY':>6}  "
          f"{'c_off':>5} {'std_x':>5} {'std_y':>5} {'cov%':>5} {'Gini':>5} {'emp':>3}  {'t(s)':>6}")
    print("-" * 130)

    results = []
    for board_name in boards:
        pcb_path = None
        for d in pcb_dirs:
            candidate = d / f'{board_name}.kicad_pcb'
            if candidate.exists():
                pcb_path = candidate
                break
        if not pcb_path:
            print(f"{board_name}: NOT FOUND")
            continue

        for profile_name in profiles:
            # Run A (no SA)
            try:
                a = run_pipeline(str(pcb_path), profile_name, skip_sa=True, seed=args.seed)
            except Exception as e:
                print(f"{board_name} [{profile_name}] A (no-SA): ERROR {e}")
                continue
            # Run B (with SA)
            try:
                b = run_pipeline(str(pcb_path), profile_name, skip_sa=False, seed=args.seed)
            except Exception as e:
                print(f"{board_name} [{profile_name}] B (SA): ERROR {e}")
                continue

            label = f"{board_name[:28]}"
            print(f"{label:<30} {a['profile'][:7]:<8} {a['n_components']:>3} {'no-SA':<6}  "
                  f"{a['hpwl']:>7.1f} {a['overlaps']:>3d} {a['oob']:>3d} {a['rudy']:>6.2f}  "
                  f"{a['centroid_offset']:>5.1f} {a['std_x']:>5.1f} {a['std_y']:>5.1f} "
                  f"{a['coverage_pct']:>5.1f} {a['gini']:>5.1f} {a['empty_cells']:>3d}  {a['runtime_s']:>6.2f}")
            print(f"{'':<30} {b['profile'][:7]:<8} {b['n_components']:>3} {'SA':<6}  "
                  f"{b['hpwl']:>7.1f} {b['overlaps']:>3d} {b['oob']:>3d} {b['rudy']:>6.2f}  "
                  f"{b['centroid_offset']:>5.1f} {b['std_x']:>5.1f} {b['std_y']:>5.1f} "
                  f"{b['coverage_pct']:>5.1f} {b['gini']:>5.1f} {b['empty_cells']:>3d}  {b['runtime_s']:>6.2f}")

            # Delta row — sign convention: + means SA increased the metric
            # For HPWL/ovr/oob/RUDY/centroid/Gini/empty/runtime: + is WORSE
            # For std_x/std_y/coverage: + is BETTER
            hpwl_pct = ((b['hpwl'] - a['hpwl']) / a['hpwl'] * 100) if a['hpwl'] > 0 else 0
            cov_pct_delta = b['coverage_pct'] - a['coverage_pct']
            # Score: count wins/losses across MULTIPLE metrics, not just HPWL
            # "SA better" = HPWL down AND not catastrophically worse on spread
            sa_hpwl_better = b['hpwl'] < a['hpwl'] - 1
            sa_hpwl_worse = b['hpwl'] > a['hpwl'] + 1
            sa_spread_better = (b['std_x'] + b['std_y']) > (a['std_x'] + a['std_y']) + 0.5
            sa_spread_worse = (b['std_x'] + b['std_y']) < (a['std_x'] + a['std_y']) - 0.5
            sa_centroid_better = b['centroid_offset'] < a['centroid_offset'] - 0.5
            sa_centroid_worse = b['centroid_offset'] > a['centroid_offset'] + 0.5

            verdict_parts = []
            if sa_hpwl_better: verdict_parts.append("HPWL↓")
            if sa_hpwl_worse: verdict_parts.append("HPWL↑")
            if sa_spread_better: verdict_parts.append("spread↑")
            if sa_spread_worse: verdict_parts.append("spread↓")
            if sa_centroid_better: verdict_parts.append("centroid↓")
            if sa_centroid_worse: verdict_parts.append("centroid↑")
            if b['overlaps'] < a['overlaps']: verdict_parts.append("ovr↓")
            if b['overlaps'] > a['overlaps']: verdict_parts.append("ovr↑")
            if b['oob'] < a['oob']: verdict_parts.append("oob↓")
            if b['oob'] > a['oob']: verdict_parts.append("oob↑")
            verdict = " ".join(verdict_parts) if verdict_parts else "tie"

            print(f"{'':<30} {'':<8} {'':>3} {'Δ':<6}  "
                  f"{b['hpwl']-a['hpwl']:>+7.1f} ({hpwl_pct:>+5.1f}%) "
                  f"{b['overlaps']-a['overlaps']:>+3d} {b['oob']-a['oob']:>+3d} "
                  f"{b['rudy']-a['rudy']:>+6.2f}  "
                  f"{b['centroid_offset']-a['centroid_offset']:>+5.1f} "
                  f"{b['std_x']-a['std_x']:>+5.1f} {b['std_y']-a['std_y']:>+5.1f} "
                  f"{cov_pct_delta:>+5.1f} {b['gini']-a['gini']:>+5.1f} "
                  f"{b['empty_cells']-a['empty_cells']:>+3d}  "
                  f"{b['runtime_s']-a['runtime_s']:>+6.2f}  [{verdict}]")
            print()
            results.append({
                'board': board_name, 'profile': a['profile'], 'n': a['n_components'],
                'a': a, 'b': b,
                'hpwl_pct': hpwl_pct,
                'sa_hpwl_better': sa_hpwl_better, 'sa_hpwl_worse': sa_hpwl_worse,
                'sa_spread_better': sa_spread_better, 'sa_spread_worse': sa_spread_worse,
                'sa_centroid_better': sa_centroid_better, 'sa_centroid_worse': sa_centroid_worse,
                'verdict': verdict,
            })

    # Summary
    if results:
        print("\n" + "=" * 130)
        print("SUMMARY (multi-metric)")
        print("=" * 130)
        # A "clear SA win" = HPWL better AND no catastrophic spread/centroid regression
        # A "clear SA loss" = HPWL worse OR oob/overlaps worse OR spread much worse
        clear_wins = []
        clear_losses = []
        mixed = []
        for r in results:
            v = r['verdict']
            # Catastrophic = oob went up, or spread collapsed (std down >2mm)
            catastrophic = (
                r['b']['oob'] > r['a']['oob']
                or r['b']['overlaps'] > r['a']['overlaps']
                or (r['sa_spread_worse'] and (r['b']['std_x'] + r['b']['std_y']) < (r['a']['std_x'] + r['a']['std_y']) * 0.7)
            )
            if r['sa_hpwl_better'] and not catastrophic:
                clear_wins.append(r)
            elif r['sa_hpwl_worse'] or catastrophic:
                clear_losses.append(r)
            else:
                mixed.append(r)

        print(f"  Clear SA wins  (HPWL↓, no catastrophic spread/oob regression): {len(clear_wins)}/{len(results)}")
        print(f"  Clear SA losses (HPWL↑ OR oob/ovr↑ OR spread collapsed):       {len(clear_losses)}/{len(results)}")
        print(f"  Mixed / tie:                                                     {len(mixed)}/{len(results)}")
        print()

        # Per-metric breakdown
        hpwl_better = sum(1 for r in results if r['sa_hpwl_better'])
        hpwl_worse = sum(1 for r in results if r['sa_hpwl_worse'])
        spread_better = sum(1 for r in results if r['sa_spread_better'])
        spread_worse = sum(1 for r in results if r['sa_spread_worse'])
        centroid_better = sum(1 for r in results if r['sa_centroid_better'])
        centroid_worse = sum(1 for r in results if r['sa_centroid_worse'])
        oob_better = sum(1 for r in results if r['b']['oob'] < r['a']['oob'])
        oob_worse = sum(1 for r in results if r['b']['oob'] > r['a']['oob'])
        ovr_better = sum(1 for r in results if r['b']['overlaps'] < r['a']['overlaps'])
        ovr_worse = sum(1 for r in results if r['b']['overlaps'] > r['a']['overlaps'])

        print(f"  Per-metric (SA better / worse / tie):")
        print(f"    HPWL:          {hpwl_better} / {hpwl_worse} / {len(results) - hpwl_better - hpwl_worse}")
        print(f"    Spread (std):  {spread_better} / {spread_worse} / {len(results) - spread_better - spread_worse}")
        print(f"    Centroid off:  {centroid_better} / {centroid_worse} / {len(results) - centroid_better - centroid_worse}")
        print(f"    OOB count:     {oob_better} / {oob_worse} / {len(results) - oob_better - oob_worse}")
        print(f"    Overlap count: {ovr_better} / {ovr_worse} / {len(results) - ovr_better - ovr_worse}")
        print()

        avg_hpwl_pct = sum(r['hpwl_pct'] for r in results) / len(results)
        print(f"  Avg HPWL change: {avg_hpwl_pct:+.1f}%")
        print()

        # List the clear losses in detail — these are the regressions to fix
        if clear_losses:
            print("  CLEAR SA LOSSES (regressions to investigate):")
            for r in clear_losses:
                print(f"    {r['board']:<30} [{r['profile']:<16}] n={r['n']:>3}  "
                      f"HPWL {r['a']['hpwl']:.0f}→{r['b']['hpwl']:.0f} ({r['hpwl_pct']:+.0f}%)  "
                      f"oob {r['a']['oob']}→{r['b']['oob']}  "
                      f"ovr {r['a']['overlaps']}→{r['b']['overlaps']}  "
                      f"[{r['verdict']}]")


if __name__ == '__main__':
    sys.exit(main())
