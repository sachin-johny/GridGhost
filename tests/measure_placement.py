#!/usr/bin/env python3
"""Measure placement quality on the test PCBs.

Runs the full placement pipeline (parse -> place -> pre-pass -> greedy/SA ->
legalize) in-process on each board in tests/test_pcbs/, then reports a
compact row of spread/HPWL/overlap metrics per board. Used to compare
tuning changes — e.g. density weight, spread move operator, RUDY threshold.

Metrics per board:
  HPWL            half-perimeter wirelength (excludes power nets)
  ovr             overlap count (post-legalization)
  oob             out-of-bounds component count
  cov%            interior-comp bbox area / board area (100 = full coverage)
  Gini            Gini coefficient of 10x10 cell-occupancy (lower = more uniform)
  empty           empty cells out of 100 (lower = fuller outline)
  std(x,y)        standard deviation of interior-comp X/Y coords (higher = wider spread)
  centroid        distance from interior centroid to board center (lower = centered)

No changes to the auto-placer itself.
"""
from __future__ import annotations

import sys
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parsers.kicad_parser import KiCadParser
from engine.smart_placement import smart_grid_place, _is_vertical_connector, _compute_interior_bbox
from engine.cost_function import CostFunction, total_hpwl, count_overlaps, count_out_of_bounds
from engine.annealer import run_sa, SAConfig
from engine.placement_prepass import preplace_caps_near_ics
from legalization.legalizer import legalize
from profiles.board_profiles import get_profile
from config import load_config
import engine.cost_state as cs


def gini_density(components, board, grid_n=10):
    """Gini coefficient of cell-occupancy on grid_n x grid_n grid."""
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


def measure(pcb_path, label=""):
    """Place + measure. Returns dict of metrics."""
    cfg = load_config()
    cs.OVERLAP_WEIGHT = 25.0
    cs.BOUNDARY_WEIGHT = 8.0

    parser = KiCadParser(pcb_path, bbox_margin=cfg.parser.bbox_margin)
    model = parser.parse()

    # Auto board expansion (matches CLI behavior)
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

    # Auto profile (matches CLI)
    ic_types = {'ic', 'mcu', 'regulator'}
    has_ics = any(getattr(c, 'component_type', '') in ic_types for c in model.components)
    profile_name = "mcu_peripheral" if has_ics else "generic"
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

    # SA / greedy
    cs.OVERLAP_WEIGHT = max(profile.beta, 25.0)
    cs.BOUNDARY_WEIGHT = max(profile.gamma, 8.0)
    cs.CONSTRAINT_WEIGHT = profile.delta

    sa_config = SAConfig(verbose=False, skip_sa=True)
    run_sa(model, config=sa_config, verbose=False, rules=profile.rules)

    # Cost
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

    # Metrics
    hpwl = total_hpwl(model)
    overlaps = count_overlaps(model)
    oob = count_out_of_bounds(model)
    board = model.board

    # Comp bbox (interior only — exclude connectors)
    interior_comps = [
        c for c in model.components
        if not c.is_fixed and getattr(c, 'component_type', '') != 'connector'
    ]
    if interior_comps:
        xs = [c.x for c in interior_comps]
        ys = [c.y for c in interior_comps]
        comp_xmin, comp_xmax = min(xs), max(xs)
        comp_ymin, comp_ymax = min(ys), max(ys)
        comp_w = comp_xmax - comp_xmin
        comp_h = comp_ymax - comp_ymin
        # Coverage: comp bbox area / board area
        comp_bbox_area = comp_w * comp_h
        board_area = board.width * board.height
        coverage = (comp_bbox_area / board_area * 100) if board_area > 0 else 0
        # Centroid
        cx_comp = sum(xs) / len(xs)
        cy_comp = sum(ys) / len(ys)
        cx_board = (board.x_min + board.x_max) / 2
        cy_board = (board.y_min + board.y_max) / 2
        centroid_offset = math.hypot(cx_comp - cx_board, cy_comp - cy_board)
        # Std-dev
        n = len(interior_comps)
        std_x = math.sqrt(sum((x - cx_comp) ** 2 for x in xs) / n)
        std_y = math.sqrt(sum((y - cy_comp) ** 2 for y in ys) / n)
        # Mean distance from board center
        mean_dist = sum(math.hypot(x - cx_board, y - cy_board)
                        for x, y in zip(xs, ys)) / n
    else:
        coverage = std_x = std_y = mean_dist = centroid_offset = 0

    gini, empty = gini_density(model.components, board)

    return {
        'label': label,
        'hpwl': hpwl,
        'overlaps': overlaps,
        'oob': oob,
        'coverage': coverage,
        'std_x': std_x,
        'std_y': std_y,
        'mean_dist': mean_dist,
        'centroid_offset': centroid_offset,
        'gini': gini,
        'empty_cells': empty,
        'interior_n': len(interior_comps) if interior_comps else 0,
        'pre_legal_cost': pre_legal_costs['total'],
    }


def fmt(metrics):
    return (
        f"HPWL={metrics['hpwl']:7.1f}  "
        f"ovr={metrics['overlaps']:3d}  "
        f"oob={metrics['oob']:3d}  "
        f"cov={metrics['coverage']:5.1f}%  "
        f"Gini={metrics['gini']:5.2f}  "
        f"empty={metrics['empty_cells']:3d}/100  "
        f"std=({metrics['std_x']:.1f},{metrics['std_y']:.1f})  "
        f"centroid_off={metrics['centroid_offset']:.2f}"
    )


if __name__ == '__main__':
    test_pcb_dir = ROOT / 'tests' / 'test_pcbs'
    boards = ['cbb', 'test4', 'test5', 'test6', 'th_sensor']
    print(f"{'Board':<12} {'HPWL':>8} {'ovr':>4} {'oob':>4} {'cov%':>6} "
          f"{'Gini':>6} {'empty':>6} {'std(x,y)':>14} {'centroid':>9}")
    print('-' * 95)
    for b in boards:
        path = test_pcb_dir / f'{b}.kicad_pcb'
        if not path.exists():
            print(f"{b:<12} (missing)")
            continue
        m = measure(str(path), label=b)
        print(f"{b:<12} {m['hpwl']:>8.1f} {m['overlaps']:>4d} {m['oob']:>4d} "
              f"{m['coverage']:>5.1f}% {m['gini']:>6.2f} {m['empty_cells']:>3d}/100 "
              f"({m['std_x']:>4.1f},{m['std_y']:>4.1f})    {m['centroid_offset']:>6.2f}")
