"""Dump per-component placement diagnostics for a single board.

Runs the same pipeline as measure_placement.py but prints each interior
component's position, distance from board center, and OOB status. Used to
diagnose centroid shift and OOB regressions.
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


def run(board_name):
    cfg = load_config()
    cs.OVERLAP_WEIGHT = 25.0
    cs.BOUNDARY_WEIGHT = 8.0

    pcb_path = ROOT / 'tests' / 'test_pcbs' / f'{board_name}.kicad_pcb'
    parser = KiCadParser(str(pcb_path), bbox_margin=cfg.parser.bbox_margin)
    model = parser.parse()

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

    ic_types = {'ic', 'mcu', 'regulator'}
    has_ics = any(getattr(c, 'component_type', '') in ic_types for c in model.components)
    profile_name = "mcu_peripheral" if has_ics else "generic"
    profile = get_profile(profile_name)

    smart_grid_place(model, margin=cfg.placement.margin,
                    spacing_factor=cfg.placement.spacing_factor,
                    rules=profile.rules)

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

    cs.OVERLAP_WEIGHT = max(profile.beta, 25.0)
    cs.BOUNDARY_WEIGHT = max(profile.gamma, 8.0)
    cs.CONSTRAINT_WEIGHT = profile.delta

    sa_config = SAConfig(verbose=False, skip_sa=True)
    run_sa(model, config=sa_config, verbose=False, rules=profile.rules)

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

    # Dump diagnostics
    board = model.board
    cx_b = (board.x_min + board.x_max) / 2
    cy_b = (board.y_min + board.y_max) / 2

    print(f"\n=== {board_name} ===")
    print(f"board=({board.x_min:.1f},{board.y_min:.1f})-({board.x_max:.1f},{board.y_max:.1f})  "
          f"w={board.width:.1f} h={board.height:.1f}  center=({cx_b:.1f},{cy_b:.1f})")
    print(f"keepouts: {len(getattr(model, 'keepouts', []) or [])}")
    for k in (getattr(model, 'keepouts', None) or []):
        print(f"  KEEPOUT ({k.x_min:.1f},{k.y_min:.1f})-({k.x_max:.1f},{k.y_max:.1f})")

    print(f"\n{'ref':<8} {'type':<10} {'x':>7} {'y':>7} {'dist_center':>12} {'oob':>5} {'fixed':>6}")
    interior_comps = []
    for c in sorted(model.components, key=lambda c: c.ref):
        is_conn = getattr(c, 'component_type', '') == 'connector'
        if c.is_fixed or (is_conn and not _is_vertical_connector(c)):
            continue
        interior_comps.append(c)
        dist = math.hypot(c.x - cx_b, c.y - cy_b)
        x_min, y_min, x_max, y_max = c.bbox
        oob = (x_min < board.x_min or x_max > board.x_max or
               y_min < board.y_min or y_max > board.y_max)
        # keepout overlap?
        in_keepout = False
        for k in (getattr(model, 'keepouts', []) or []):
            if not c.is_edge_connector:
                ox1 = max(x_min, k.x_min); oy1 = max(y_min, k.y_min)
                ox2 = min(x_max, k.x_max); oy2 = min(y_max, k.y_max)
                if ox2 > ox1 and oy2 > oy1:
                    in_keepout = True
                    break
        flag = ""
        if oob: flag += "OOB "
        if in_keepout: flag += "KEEP "
        print(f"{c.ref:<8} {getattr(c,'component_type',''):<10} "
              f"{c.x:>7.1f} {c.y:>7.1f} {dist:>12.1f} {flag:>5} {str(c.is_fixed):>6}  "
              f"bbox=({x_min:.1f},{y_min:.1f})-({x_max:.1f},{y_max:.1f}) "
              f"w={x_max-x_min:.1f} h={y_max-y_min:.1f} "
              f"rot={c.rotation:.0f} {'edge' if c.is_edge_connector else ''}")

    if interior_comps:
        xs = [c.x for c in interior_comps]
        ys = [c.y for c in interior_comps]
        cx_comp = sum(xs) / len(xs)
        cy_comp = sum(ys) / len(ys)
        print(f"\ncentroid=({cx_comp:.2f}, {cy_comp:.2f})  "
              f"board_center=({cx_b:.2f}, {cy_b:.2f})  "
              f"offset={math.hypot(cx_comp - cx_b, cy_comp - cy_b):.2f}")
        print(f"  dx={cx_comp - cx_b:+.2f}  dy={cy_comp - cy_b:+.2f}")

    print(f"\nHPWL={total_hpwl(model):.1f} ovr={count_overlaps(model)} oob={count_out_of_bounds(model)}")


if __name__ == '__main__':
    for b in sys.argv[1:]:
        run(b)
