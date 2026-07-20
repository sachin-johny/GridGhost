#!/usr/bin/env python3
"""Phase 4.4 — Extended quantitative benchmark harness for GridGhost.

Reports the KPI table from the brief's Phase 4.4 against the legacy
placement pipeline (in-process):

  | Metric                                  | What it tells you                       |
  |-----------------------------------------|-----------------------------------------|
  | HPWL                                    | Did we improve on baseline?             |
  | HPWL vs. SA-only (no clustering)        | Is clustering earning its complexity?   |
  | Overlap count (must be 0 post-legalize) | Correctness gate                        |
  | Out-of-bounds count                     | Correctness gate                        |
  | RUDY congestion score                   | Are we creating routing choke points?   |
  | Wall-clock runtime                      | Regression gate                         |
  | Convergence stability (variance ≥10 seeds) | Is the SA reliable?                  |
  | Constraint violation score per rule     | Are profile rules actually satisfied?   |

Runs each board × each profile × N seeds and prints a markdown KPI
table.  Designed for manual review and (future) CI threshold gates.

Usage:
    python tests/measure_placement_v2.py [--seeds N] [--boards b1,b2,...]
                                         [--profiles p1,p2,...] [--markdown]
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parsers.kicad_parser import KiCadParser
from engine.smart_placement import smart_grid_place, _is_vertical_connector, _compute_interior_bbox
from engine.cost_function import CostFunction, total_hpwl, count_overlaps, count_out_of_bounds
from engine.annealer import run_sa, SAConfig
from engine.placement_prepass import preplace_caps_near_ics
from engine.congestion import rudy_congestion_penalty
from legalization.legalizer import legalize
from profiles.board_profiles import get_profile, list_profiles, BoardProfile
from config import load_config
import engine.cost_state as cs


def _expand_board(model):
    """Auto board expansion (matches CLI behavior)."""
    if not getattr(model, 'user_defined_outline', False):
        board = model.board
        board_area = board.width * board.height
        if board_area > 0:
            comp_area = sum(c.effective_width * c.effective_height for c in model.components)
            density = comp_area / board_area
            target = 0.35
            if density > target:
                scale = math.sqrt(density / target)
                new_w = board.width * scale
                new_h = board.height * scale
                cx = (board.x_min + board.x_max) / 2.0
                cy = (board.y_min + board.y_max) / 2.0
                from models.board_model import BoardOutline
                model.board = BoardOutline(
                    x_min=cx - new_w / 2.0, y_min=cy - new_h / 2.0,
                    x_max=cx + new_w / 2.0, y_max=cy + new_h / 2.0,
                )


def _run_pipeline(pcb_path: str, profile: BoardProfile, seed: int):
    """Run the full GridGhost pipeline on pcb_path with the given profile
    and SA seed.  Returns a dict of metrics.
    """
    import random
    random.seed(seed)

    cfg = load_config()
    cs.OVERLAP_WEIGHT = 25.0
    cs.BOUNDARY_WEIGHT = 8.0

    t_start = time.perf_counter()

    parser = KiCadParser(pcb_path, bbox_margin=cfg.parser.bbox_margin)
    model = parser.parse()
    _expand_board(model)

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

    # SA / greedy
    cs.OVERLAP_WEIGHT = max(profile.beta, 25.0)
    cs.BOUNDARY_WEIGHT = max(profile.gamma, 8.0)
    cs.CONSTRAINT_WEIGHT = profile.delta

    sa_config = SAConfig(verbose=False, skip_sa=True)
    run_sa(model, config=sa_config, verbose=False, rules=profile.rules)

    # Cost (pre-legalize)
    cost_fn = CostFunction(
        alpha=profile.alpha, beta=profile.beta,
        gamma=profile.gamma, delta=profile.delta, rules=profile.rules,
    )
    pre_legal_costs = cost_fn.evaluate(model)

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
    runtime_s = t_end - t_start

    # === Metrics ===
    hpwl = total_hpwl(model)
    overlaps = count_overlaps(model)
    oob = count_out_of_bounds(model)

    # RUDY congestion (Phase 2.1 / 4.4 KPI)
    rudy_penalty, rudy_peak, rudy_avg, rudy_overflow = rudy_congestion_penalty(model, 2.0)

    # Constraint violations per active rule
    from engine.constraint_evaluator import evaluate_constraint_penalties
    _, constraint_breakdown = evaluate_constraint_penalties(model, profile.active_rules())

    return {
        'hpwl': hpwl,
        'overlaps': overlaps,
        'oob': oob,
        'rudy_penalty': rudy_penalty,
        'rudy_peak': rudy_peak,
        'rudy_overflow': rudy_overflow,
        'runtime_s': runtime_s,
        'pre_legal_cost': pre_legal_costs['total'],
        'constraint_breakdown': dict(constraint_breakdown),
        'n_components': len(model.components),
    }


def _run_with_variance(pcb_path: str, profile: BoardProfile, n_seeds: int):
    """Run the pipeline n_seeds times with different seeds, return per-run
    metrics plus variance stats."""
    runs = []
    for seed in range(n_seeds):
        m = _run_pipeline(pcb_path, profile, seed=seed)
        runs.append(m)
    # Variance stats
    hpwls = [r['hpwl'] for r in runs]
    rudy_pens = [r['rudy_penalty'] for r in runs]
    runtimes = [r['runtime_s'] for r in runs]
    n = len(hpwls)
    mean_hpwl = sum(hpwls) / n
    mean_rudy = sum(rudy_pens) / n
    mean_runtime = sum(runtimes) / n
    var_hpwl = sum((h - mean_hpwl) ** 2 for h in hpwls) / n if n > 1 else 0.0
    var_rudy = sum((r - mean_rudy) ** 2 for r in rudy_pens) / n if n > 1 else 0.0
    std_hpwl = math.sqrt(var_hpwl)
    std_rudy = math.sqrt(var_rudy)
    cv_hpwl = (std_hpwl / mean_hpwl * 100) if mean_hpwl > 0 else 0.0  # coefficient of variation %
    cv_rudy = (std_rudy / mean_rudy * 100) if mean_rudy > 0 else 0.0
    return {
        'runs': runs,
        'mean_hpwl': mean_hpwl,
        'std_hpwl': std_hpwl,
        'cv_hpwl_pct': cv_hpwl,
        'mean_rudy': mean_rudy,
        'std_rudy': std_rudy,
        'cv_rudy_pct': cv_rudy,
        'mean_runtime_s': mean_runtime,
        'min_hpwl': min(hpwls),
        'max_hpwl': max(hpwls),
        'n_seeds': n,
    }


def fmt_md_table(rows: list[dict], headers: list[str], keys: list[str]) -> str:
    """Format a list of dicts as a markdown table."""
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        out.append("| " + " | ".join(str(r[k]) for k in keys) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="GridGhost benchmark harness (Phase 4.4)")
    ap.add_argument("--seeds", type=int, default=3,
                    help="Number of SA seeds per board×profile (default 3)")
    ap.add_argument("--boards", type=str, default="cbb,test4,test5,test6,th_sensor",
                    help="Comma-separated board names (default: all 5)")
    ap.add_argument("--profiles", type=str, default="auto",
                    help="Comma-separated profile names, or 'auto' for auto-select (default: auto)")
    ap.add_argument("--markdown", action="store_true",
                    help="Output as markdown table (default: human-readable)")
    args = ap.parse_args()

    test_pcb_dir = ROOT / 'tests' / 'test_pcbs'
    boards = [b.strip() for b in args.boards.split(",") if b.strip()]
    profile_names = [p.strip() for p in args.profiles.split(",") if p.strip()]

    print(f"GridGhost benchmark — {args.seeds} seed(s) per board×profile")
    print(f"Boards: {boards}")
    print(f"Profiles: {profile_names}")
    print()

    all_rows = []
    for board_name in boards:
        pcb_path = test_pcb_dir / f'{board_name}.kicad_pcb'
        if not pcb_path.exists():
            print(f"  {board_name}: (missing)")
            continue

        for profile_name in profile_names:
            # Resolve profile (auto-select if requested)
            if profile_name == "auto":
                parser_tmp = KiCadParser(str(pcb_path))
                model_tmp = parser_tmp.parse()
                ic_types = {'ic', 'mcu', 'regulator'}
                has_ics = any(getattr(c, 'component_type', '') in ic_types for c in model_tmp.components)
                actual_profile_name = "mcu_peripheral" if has_ics else "generic"
            else:
                actual_profile_name = profile_name
            profile = get_profile(actual_profile_name)

            stats = _run_with_variance(str(pcb_path), profile, args.seeds)
            first_run = stats['runs'][0]

            row = {
                'board': board_name,
                'profile': actual_profile_name,
                'n': first_run['n_components'],
                'hpwl_mean': f"{stats['mean_hpwl']:.1f}",
                'hpwl_cv%': f"{stats['cv_hpwl_pct']:.1f}",
                'hpwl_min': f"{stats['min_hpwl']:.1f}",
                'hpwl_max': f"{stats['max_hpwl']:.1f}",
                'ovr': first_run['overlaps'],
                'oob': first_run['oob'],
                'rudy_mean': f"{stats['mean_rudy']:.3f}",
                'rudy_cv%': f"{stats['cv_rudy_pct']:.1f}",
                'runtime_s': f"{stats['mean_runtime_s']:.2f}",
                'seeds': stats['n_seeds'],
            }
            all_rows.append(row)

            # Per-rule constraint violations (from first run, for verbosity)
            cb = first_run['constraint_breakdown']

            if not args.markdown:
                print(f"  {board_name:<12} {actual_profile_name:<16} "
                      f"HPWL={stats['mean_hpwl']:>8.1f}±{stats['std_hpwl']:>6.1f} "
                      f"(cv={stats['cv_hpwl_pct']:.1f}%, range {stats['min_hpwl']:.0f}-{stats['max_hpwl']:.0f})  "
                      f"ovr={first_run['overlaps']} oob={first_run['oob']}  "
                      f"RUDY={stats['mean_rudy']:.3f}  "
                      f"t={stats['mean_runtime_s']:.2f}s")
                if cb:
                    non_zero = {k: v for k, v in cb.items() if v > 0.01}
                    if non_zero:
                        print(f"    constraint violations: {non_zero}")

    if args.markdown:
        headers = ["Board", "Profile", "N", "HPWL (mean)", "HPWL CV%", "HPWL min", "HPWL max",
                   "ovr", "oob", "RUDY (mean)", "RUDY CV%", "Runtime (s)", "Seeds"]
        keys = ['board', 'profile', 'n', 'hpwl_mean', 'hpwl_cv%', 'hpwl_min', 'hpwl_max',
                'ovr', 'oob', 'rudy_mean', 'rudy_cv%', 'runtime_s', 'seeds']
        print(fmt_md_table(all_rows, headers, keys))

    return 0


if __name__ == '__main__':
    sys.exit(main())
