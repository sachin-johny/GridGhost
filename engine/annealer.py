"""Simulated Annealing engine for PCB component placement.

Ports CadMust-Neo's SA approach with adaptive cooling, reheating,
temperature-dependent move operators, and greedy refinement.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from models.board_model import BoardModel
from engine.cost_state import CostState, OVERLAP_WEIGHT, BOUNDARY_WEIGHT, CONSTRAINT_WEIGHT
from engine.moves import (
    select_move_type, get_moveable_indices,
    do_translate, do_swap, do_rotate, do_median,
    revert_move, affected_indices, MoveUndo,
)


@dataclass
class SAConfig:
    max_iterations: int = 200
    reheat_count: int = 2
    reheat_ratio: float = 0.30
    calibration_samples: int = 200
    initial_accept_rate: float = 0.95
    penalty_scale_min: float = 0.80  # high floor: don't let SA ignore penalties at hot T
    min_temperature: float = 1e-6
    freeze_threshold: float = 0.01
    greedy_nudge_distances: tuple[float, ...] = (0.05, 0.1, 0.2, 0.5, 1.0)
    greedy_rotations: tuple[float, ...] = (90.0, 180.0, 270.0)
    greedy_improve_threshold: float = 1.0
    verbose: bool = True


def _save_positions(model: BoardModel, indices: list[int]) -> dict[int, tuple[float, float, float]]:
    return {i: (model.components[i].x, model.components[i].y, model.components[i].rotation) for i in indices}


def _restore_positions(model: BoardModel, saved: dict[int, tuple[float, float, float]]):
    for i, (x, y, rot) in saved.items():
        model.components[i].x = x
        model.components[i].y = y
        model.components[i].set_rotation(rot)


def _calibrate_t0(
    model: BoardModel,
    cost_state: CostState,
    moveable_indices: list[int],
    config: SAConfig,
) -> float:
    """Auto-calibrate initial temperature by sampling random moves.

    Uses the SAME window as the first SA step (max_window) so the
    calibration deltas match actual SA move magnitudes. Previously used
    0.1*board which underestimated deltas, making T0 too low and
    accept rate at T0 only ~0.40 instead of 0.95.
    """
    board = model.board
    # Use the same window the first SA step will use (max_window)
    window = max(board.width, board.height) * 0.15
    deltas = []

    for _ in range(config.calibration_samples):
        idx = random.choice(moveable_indices)
        comp = model.components[idx]
        old_x, old_y, old_rot = comp.x, comp.y, comp.rotation
        old_cost = cost_state.total_cost

        dx = random.uniform(-window, window)
        dy = random.uniform(-window, window)
        comp.x += dx
        comp.y += dy

        new_cost = cost_state.incremental_update({idx})
        delta = new_cost - old_cost
        if delta > 0:
            deltas.append(delta)

        # Revert
        comp.x = old_x
        comp.y = old_y
        comp.set_rotation(old_rot)
        cost_state.incremental_update({idx})

    if not deltas:
        return 1.0

    median_delta = sorted(deltas)[len(deltas) // 2]
    return -median_delta / math.log(config.initial_accept_rate)


def _run_sa_pass(
    model: BoardModel,
    cost_state: CostState,
    moveable_indices: list[int],
    t0: float,
    config: SAConfig,
    max_iter_override: int | None = None,
) -> float:
    """Run one SA pass. Returns final temperature."""
    T = t0
    n_moveable = len(moveable_indices)
    moves_per_temp = max(200, 20 * n_moveable)
    board = model.board
    max_window = max(board.width, board.height) * 0.15
    min_window = max(board.width, board.height) * 0.005
    max_iter = max_iter_override or config.max_iterations

    best_positions = _save_positions(model, moveable_indices)
    best_cost = cost_state.normalized_cost

    for step in range(max_iter):
        t_ratio = math.log(T + 1.0) / math.log(t0 + 1.0) if t0 > 0 else 0.0
        t_ratio = max(0.0, min(1.0, t_ratio))

        # Adaptive penalty scale
        scale = config.penalty_scale_min + (1.0 - config.penalty_scale_min) * (1.0 - t_ratio)
        cost_state.update_penalty_scale(scale)

        window = min_window + (max_window - min_window) * t_ratio
        noise = board.width * 0.02 * t_ratio

        accepted = 0
        for _ in range(moves_per_temp):
            mt = select_move_type(t_ratio)
            if mt == 'translate':
                undo = do_translate(model, moveable_indices, t_ratio, window)
            elif mt == 'swap':
                undo = do_swap(model, moveable_indices)
            elif mt == 'rotate':
                undo = do_rotate(model, moveable_indices)
            else:
                undo = do_median(model, moveable_indices, t_ratio, noise)

            if not undo.old_states:
                continue

            moved = affected_indices(undo)
            snap = cost_state.snapshot(moved)

            new_cost = cost_state.incremental_update(moved)
            old_cost = (snap['hpwl']
                        + OVERLAP_WEIGHT * sum(snap['pair_overlaps'].values()) * scale
                        + BOUNDARY_WEIGHT * sum(snap['comp_boundary'].values()) * scale
                        + CONSTRAINT_WEIGHT * snap['constraint_total'] * scale)

            delta = new_cost - old_cost

            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                accepted += 1
                # Track best
                cur_norm = cost_state.normalized_cost
                if cur_norm < best_cost:
                    best_cost = cur_norm
                    best_positions = _save_positions(model, moveable_indices)
            else:
                cost_state.restore(snap)
                revert_move(model, undo)

        accept_rate = accepted / max(moves_per_temp, 1)

        if config.verbose and step % 20 == 0:
            print(f"    T={T:.4f} accept={accept_rate:.2f} cost={cost_state.normalized_cost:.1f} hpwl={cost_state.hpwl:.1f}")

        # Adaptive cooling — slow schedule to prevent premature freezing
        if accept_rate > 0.6:
            T *= 0.95
        elif accept_rate > 0.3:
            T *= 0.97
        else:
            T *= 0.99

        T = max(T, config.min_temperature)

        # Early stop
        if accept_rate < config.freeze_threshold and T < t0 * 0.01:
            if config.verbose:
                print(f"    Frozen at step {step}, accept_rate={accept_rate:.3f}")
            break

    # Restore best
    _restore_positions(model, best_positions)
    cost_state._compute_all()
    return T


def _greedy_refine(
    model: BoardModel,
    cost_state: CostState,
    moveable_indices: list[int],
    config: SAConfig,
) -> None:
    """Greedy refinement: try nudges and rotations, accept only if improving."""
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1),
                  (1, 1), (-1, 1), (1, -1), (-1, -1)]

    improved = True
    sweep = 0
    # Track best cost/positions across sweeps to prevent oscillation
    best_sweep_cost = cost_state.normalized_cost
    best_sweep_positions = _save_positions(model, moveable_indices)

    while improved:
        improved = False
        sweep += 1
        for idx in moveable_indices:
            comp = model.components[idx]
            # Try nudges
            for dx_dir, dy_dir in directions:
                for dist in config.greedy_nudge_distances:
                    old_x, old_y = comp.x, comp.y
                    base_cost = cost_state.normalized_cost
                    comp.x += dx_dir * dist
                    comp.y += dy_dir * dist

                    new_cost = cost_state.incremental_update({idx})

                    if new_cost < base_cost - config.greedy_improve_threshold:
                        improved = True
                    else:
                        comp.x = old_x
                        comp.y = old_y
                        cost_state.incremental_update({idx})

            # Try rotations
            base_rot = comp.rotation
            for rot in config.greedy_rotations:
                base_cost = cost_state.normalized_cost
                comp.set_rotation((base_rot + rot) % 360.0)
                new_cost = cost_state.incremental_update({idx})

                if new_cost < base_cost - config.greedy_improve_threshold:
                    improved = True
                    base_rot = comp.rotation
                else:
                    comp.set_rotation(base_rot)
                    cost_state.incremental_update({idx})

        cur_cost = cost_state.normalized_cost
        if cur_cost < best_sweep_cost:
            best_sweep_cost = cur_cost
            best_sweep_positions = _save_positions(model, moveable_indices)

        if config.verbose and improved:
            print(f"    Greedy sweep {sweep}: cost={cur_cost:.1f}")

        if sweep >= 5:
            break

    # Restore best sweep result (prevents oscillation)
    if cost_state.normalized_cost > best_sweep_cost:
        _restore_positions(model, best_sweep_positions)
        cost_state._compute_all()


def simulate_annealing(
    model: BoardModel,
    cost_state: CostState,
    moveable_indices: list[int],
    config: SAConfig | None = None,
) -> float:
    """Run SA optimization. Returns final normalized cost. Mutates model in-place."""
    if config is None:
        config = SAConfig()

    if not moveable_indices:
        return cost_state.normalized_cost

    # Calibrate T0
    t0 = _calibrate_t0(model, cost_state, moveable_indices, config)
    if config.verbose:
        print(f"  SA: T0={t0:.4f}, {len(moveable_indices)} moveable components")

    # Main SA pass
    _run_sa_pass(model, cost_state, moveable_indices, t0, config)

    # Reheating rounds
    for reheat_i in range(config.reheat_count):
        reheat_t0 = t0 * config.reheat_ratio * (0.5 ** reheat_i)
        if config.verbose:
            print(f"  Reheat {reheat_i + 1}: T0={reheat_t0:.4f}")
        _run_sa_pass(
            model, cost_state, moveable_indices,
            reheat_t0, config,
            max_iter_override=config.max_iterations // 2,
        )

    # Greedy refinement
    if config.verbose:
        print("  Greedy refinement...")
    _greedy_refine(model, cost_state, moveable_indices, config)

    final_cost = cost_state.normalized_cost
    if config.verbose:
        print(f"  SA complete: cost={final_cost:.1f} hpwl={cost_state.hpwl:.1f} overlaps={cost_state.overlap_count}")
    return final_cost


def run_sa(
    model: BoardModel,
    profile_weights: dict | None = None,
    config: SAConfig | None = None,
    verbose: bool = True,
    rules: list | None = None,
) -> dict:
    """High-level entry point: build CostState, get moveable indices, run SA.

    Args:
        rules: Optional list of ConstraintRule objects from board profile.
               When provided, constraint penalties are included in the SA
               cost function so the annealer respects placement constraints
               (decoupling proximity, connector edge, etc.).
    """
    if config is None:
        config = SAConfig(verbose=verbose)
    elif verbose and not config.verbose:
        config = SAConfig(**{**config.__dict__, 'verbose': True})

    cost_state = CostState(model, rules=rules)
    moveable_indices = get_moveable_indices(model)

    initial_cost = cost_state.normalized_cost
    initial_hpwl = cost_state.hpwl

    final_cost = simulate_annealing(model, cost_state, moveable_indices, config)

    return {
        'initial_cost': initial_cost,
        'final_cost': final_cost,
        'initial_hpwl': initial_hpwl,
        'final_hpwl': cost_state.hpwl,
        'overlap_count': cost_state.overlap_count,
        'improvement': initial_cost - final_cost,
    }
