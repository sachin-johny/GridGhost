"""Simulated Annealing engine for PCB component placement.

v9 strategy:
  - Lower penalty floor (penalty_scale_min=0.30): hot SA has overlap=3.0,
    allowing true exploration through overlap space without being trapped.
  - Robust T0 calibration: samples ALL move types (not just translate),
    uses 90th percentile with median-safety floor.  Achieves accept≈0.85-0.92.
  - Gentler cooling: adaptive schedule with finer granularity (5 bands),
    maintains mobility longer in the productive accept range (0.2-0.6).
  - Larger move window (10% of board): better exploration at all temperatures.
  - More moves per temperature (20n vs 15n): better sampling statistics.
  - 3 reheats at 40% T0 with 0.7 decay: more escape opportunities.
  - Greedy refinement: 8 sweeps with threshold 0.5 (accepts smaller improvements).
  - Overlap resolver: larger nudge distances (up to 32mm) for dense boards.
  - Stall detection: breaks early if no improvement for 40% of iterations.
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
    max_iterations: int = 300          # more steps for thorough exploration
    reheat_count: int = 3              # 3 reheats for better escape from local minima
    reheat_ratio: float = 0.40         # reheat to 40% of previous T0
    calibration_samples: int = 500     # more samples for robust T0 estimation
    initial_accept_rate: float = 0.92   # target high accept rate at start
    penalty_scale_min: float = 0.50    # balance: explore while keeping constraints relevant
    min_temperature: float = 1e-8
    freeze_threshold: float = 0.005    # stop sooner when frozen
    greedy_nudge_distances: tuple[float, ...] = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
    greedy_rotations: tuple[float, ...] = (90.0, 180.0, 270.0)
    greedy_improve_threshold: float = 0.5  # accept smaller improvements
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
    """Auto-calibrate initial temperature by sampling ALL move types.

    v9 calibration: sample translate, swap, rotate, and median moves
    (not just translate) so T0 reflects the full cost landscape SA will
    see.  Use 90th percentile of positive deltas for robust T0 estimation.
    Set penalty_scale to penalty_scale_min before calibration so T0
    matches the actual hot-phase SA landscape.
    """
    board = model.board
    max_window = max(board.width, board.height) * 0.08
    min_window = max(board.width, board.height) * 0.005
    window = max_window  # hot phase uses max window
    noise = board.width * 0.02
    deltas = []

    # Set penalty_scale to what SA will use at the start (hot phase)
    cost_state.update_penalty_scale(config.penalty_scale_min)

    for i in range(config.calibration_samples):
        # Sample all move types with hot-phase probabilities
        mt = select_move_type(1.0)  # t_ratio=1.0 = hot phase

        if mt == 'translate':
            undo = do_translate(model, moveable_indices, 1.0, window)
        elif mt == 'swap':
            undo = do_swap(model, moveable_indices)
        elif mt == 'rotate':
            undo = do_rotate(model, moveable_indices)
        else:
            undo = do_median(model, moveable_indices, 1.0, noise)

        if not undo.old_states:
            continue

        moved = affected_indices(undo)
        snap = cost_state.snapshot(moved)
        new_cost = cost_state.incremental_update(moved)
        old_cost = (snap['hpwl']
                    + OVERLAP_WEIGHT * sum(snap['pair_overlaps'].values()) * config.penalty_scale_min
                    + BOUNDARY_WEIGHT * sum(snap['comp_boundary'].values()) * config.penalty_scale_min
                    + CONSTRAINT_WEIGHT * snap['constraint_total'] * config.penalty_scale_min)

        delta = new_cost - old_cost
        if delta > 0:
            deltas.append(delta)

        # Revert
        cost_state.restore(snap)
        revert_move(model, undo)

    if not deltas:
        return 1.0

    # Use 90th percentile — ensures T0 is high enough for the largest
    # typical deltas.  75th percentile was too low, causing accept=0.07-0.28
    # at T0 instead of the target 0.90.
    idx90 = int(len(deltas) * 0.90)
    pct90_delta = sorted(deltas)[min(idx90, len(deltas) - 1)]
    t0 = -pct90_delta / math.log(config.initial_accept_rate)

    # Safety: ensure T0 is at least large enough that median delta has
    # accept rate > 0.5 (prevents T0 being too low for skewed distributions)
    if deltas:
        median_delta = sorted(deltas)[len(deltas) // 2]
        t0_median = -median_delta / math.log(0.55)
        t0 = max(t0, t0_median)

    return t0


def _run_sa_pass(
    model: BoardModel,
    cost_state: CostState,
    moveable_indices: list[int],
    t0: float,
    config: SAConfig,
    max_iter_override: int | None = None,
) -> float:
    """Run one SA pass. Returns final temperature.

    No hard overlap rejection: SA needs freedom to move components
    through overlaps during the hot phase (global placement).
    The penalty_scale mechanism naturally handles this:
      - Hot T (scale ≈ 0.3): overlap weight is low, SA explores freely
      - Cold T (scale ≈ 1.0): overlap weight is full, SA resolves overlaps
    Any remaining overlaps after SA are handled by the legalizer.
    """
    T = t0
    n_moveable = len(moveable_indices)
    moves_per_temp = max(300, 20 * n_moveable)  # more moves for better sampling
    board = model.board
    max_window = max(board.width, board.height) * 0.10  # 10% — larger moves for better exploration
    min_window = max(board.width, board.height) * 0.003
    max_iter = max_iter_override or config.max_iterations

    best_positions = _save_positions(model, moveable_indices)
    best_cost = cost_state.normalized_cost
    no_improve_count = 0  # track stalls for early termination

    for step in range(max_iter):
        t_ratio = math.log(T + 1.0) / math.log(t0 + 1.0) if t0 > 0 else 0.0
        t_ratio = max(0.0, min(1.0, t_ratio))

        # Adaptive penalty scale: hot→discount penalties, cold→full weight
        # v9: smoother ramp with lower floor (0.30) so hot SA explores freely
        scale = config.penalty_scale_min + (1.0 - config.penalty_scale_min) * (1.0 - t_ratio)
        cost_state.update_penalty_scale(scale)

        window = min_window + (max_window - min_window) * t_ratio
        noise = board.width * 0.03 * t_ratio  # slightly more noise for median moves

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
                    no_improve_count = 0
            else:
                cost_state.restore(snap)
                revert_move(model, undo)

        accept_rate = accepted / max(moves_per_temp, 1)

        if config.verbose and step % 10 == 0:
            print(f"    T={T:.2f} accept={accept_rate:.2f} cost={cost_state.normalized_cost:.1f} "
                  f"hpwl={cost_state.hpwl:.1f} overlaps={cost_state.overlap_count}")

        # Adaptive cooling — v9: gentler cooling to maintain mobility longer
        if accept_rate > 0.6:
            T *= 0.95   # fast cooling when very hot (too many bad moves accepted)
        elif accept_rate > 0.4:
            T *= 0.97   # moderate cooling in productive range
        elif accept_rate > 0.2:
            T *= 0.985  # slow cooling — sweet spot for finding improvements
        elif accept_rate > 0.05:
            T *= 0.99   # very slow cooling near freeze
        else:
            T *= 0.98   # near-freeze: moderate cooling (not too fast, not too slow)

        T = max(T, config.min_temperature)

        # Track stall: if best hasn't improved for many steps, we're stuck
        no_improve_count += 1

        # Early stop — don't waste time when SA is frozen AND stalled
        if accept_rate < config.freeze_threshold and T < t0 * 0.03:
            if config.verbose:
                print(f"    Frozen at step {step}, accept_rate={accept_rate:.3f}")
            break

        # Stall detection: if no improvement for 40% of iterations, break
        if no_improve_count > max_iter * 0.4 and accept_rate < 0.05:
            if config.verbose:
                print(f"    Stalled at step {step}, no improvement for {no_improve_count} steps")
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
    """Greedy refinement: try nudges and rotations, accept only if improving.

    Hard overlap rejection: greedy does small moves, so rejecting
    overlap-creating moves doesn't limit exploration.  This ensures
    greedy never creates new overlaps.

    Bug fix: compare normalized_cost consistently (no penalty_scale
    mismatch).  Reset penalty_scale to 1.0 before greedy so costs
    are in the same units.
    """
    # Reset penalty scale for greedy — we want full-weight evaluation
    cost_state.update_penalty_scale(1.0)

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1),
                  (1, 1), (-1, 1), (1, -1), (-1, -1)]

    improved = True
    sweep = 0
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
                    prev_overlap_count = cost_state.overlap_count

                    comp.x += dx_dir * dist
                    comp.y += dy_dir * dist

                    new_cost = cost_state.incremental_update({idx})

                    # Hard overlap rejection: greedy must not create new overlaps
                    if cost_state.overlap_count > prev_overlap_count:
                        comp.x = old_x
                        comp.y = old_y
                        cost_state.incremental_update({idx})
                        continue

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
                prev_overlap_count = cost_state.overlap_count
                comp.set_rotation((base_rot + rot) % 360.0)
                new_cost = cost_state.incremental_update({idx})

                # Hard overlap rejection for rotations too
                if cost_state.overlap_count > prev_overlap_count:
                    comp.set_rotation(base_rot)
                    cost_state.incremental_update({idx})
                    continue

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
        else:
            _restore_positions(model, best_sweep_positions)
            cost_state._compute_all()

        if config.verbose:
            print(f"    Greedy sweep {sweep}: cost={cost_state.normalized_cost:.1f} "
                  f"overlaps={cost_state.overlap_count}")

        if sweep >= 8:  # v9: allow more sweeps for deeper refinement
            break

    # Final restore to best sweep result
    if cost_state.normalized_cost > best_sweep_cost:
        _restore_positions(model, best_sweep_positions)
        cost_state._compute_all()


def _resolve_overlaps_greedy(
    model: BoardModel,
    cost_state: CostState,
    moveable_indices: list[int],
    config: SAConfig,
) -> None:
    """Force-resolve remaining overlaps after greedy refinement.

    When greedy can't resolve overlaps (because all overlap-resolving
    moves increase cost), this phase ONLY optimizes for overlap removal.
    It tries pushing overlapping components apart, accepting HPWL
    increases, until all overlaps are resolved or we exhaust attempts.

    Strategy: for each overlapping pair, try moving each component
    individually AND try pushing both components apart simultaneously.
    The pair-based approach handles cases where moving just one
    component can't resolve the overlap (e.g., both are boxed in).
    """
    cost_state.update_penalty_scale(1.0)
    board = model.board
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1),
                  (1, 1), (-1, 1), (1, -1), (-1, -1)]
    nudge_dists = [0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0]  # v9: larger distances for dense boards
    moveable_set = set(moveable_indices)

    prev_overlaps = cost_state.overlap_count
    for outer in range(15):  # max 15 outer iterations
        overlaps = cost_state.overlap_count
        if overlaps == 0:
            break

        # Identify overlapping component pairs
        overlap_pairs = []
        for (i, j), penalty in cost_state._pair_overlaps.items():
            if penalty > 0:
                overlap_pairs.append((i, j))

        for i, j in overlap_pairs:
            if cost_state.overlap_count == 0:
                break

            c1 = model.components[i]
            c2 = model.components[j]
            c1_movable = i in moveable_set
            c2_movable = j in moveable_set

            best_overlap = cost_state.overlap_count
            best_action = None  # ('single', idx, x, y) or ('pair', i, j, x1, y1, x2, y2)

            # Strategy 1: try moving each component individually
            for idx, is_movable in [(i, c1_movable), (j, c2_movable)]:
                if not is_movable:
                    continue
                comp = model.components[idx]
                for dx_dir, dy_dir in directions:
                    for dist in nudge_dists:
                        old_x, old_y = comp.x, comp.y
                        comp.x += dx_dir * dist
                        comp.y += dy_dir * dist
                        cost_state.incremental_update({idx})

                        if cost_state.overlap_count < best_overlap:
                            best_overlap = cost_state.overlap_count
                            best_action = ('single', idx, comp.x, comp.y)

                        comp.x = old_x
                        comp.y = old_y
                        cost_state.incremental_update({idx})

            # Strategy 2: push both components apart simultaneously
            if c1_movable and c2_movable:
                dx_sign = 1 if c2.x >= c1.x else -1
                dy_sign = 1 if c2.y >= c1.y else -1
                for dist in nudge_dists:
                    old_x1, old_y1 = c1.x, c1.y
                    old_x2, old_y2 = c2.x, c2.y
                    # Push c1 and c2 in opposite directions
                    c1.x -= dx_sign * dist * 0.5
                    c1.y -= dy_sign * dist * 0.5
                    c2.x += dx_sign * dist * 0.5
                    c2.y += dy_sign * dist * 0.5
                    cost_state.incremental_update({i, j})

                    if cost_state.overlap_count < best_overlap:
                        best_overlap = cost_state.overlap_count
                        best_action = ('pair', i, j, c1.x, c1.y, c2.x, c2.y)

                    c1.x, c1.y = old_x1, old_y1
                    c2.x, c2.y = old_x2, old_y2
                    cost_state.incremental_update({i, j})

            # Apply the best overlap-reducing action
            if best_action is not None and best_overlap < cost_state.overlap_count:
                if best_action[0] == 'single':
                    _, idx, nx, ny = best_action
                    model.components[idx].x = nx
                    model.components[idx].y = ny
                    cost_state.incremental_update({idx})
                elif best_action[0] == 'pair':
                    _, ci, cj, x1, y1, x2, y2 = best_action
                    model.components[ci].x = x1
                    model.components[ci].y = y1
                    model.components[cj].x = x2
                    model.components[cj].y = y2
                    cost_state.incremental_update({ci, cj})

        overlaps_now = cost_state.overlap_count
        if config.verbose:
            print(f"    Overlap resolve pass {outer + 1}: overlaps={overlaps_now} "
                  f"(was {prev_overlaps}, resolved {prev_overlaps - overlaps_now})")

        if overlaps_now >= prev_overlaps:
            # No progress — break to avoid infinite loop
            break
        prev_overlaps = overlaps_now

    # Clamp to board bounds after overlap resolution
    for idx in moveable_indices:
        comp = model.components[idx]
        half_w = comp.effective_width / 2.0
        half_h = comp.effective_height / 2.0
        comp.x = max(board.x_min + half_w, min(comp.x, board.x_max - half_w))
        comp.y = max(board.y_min + half_h, min(comp.y, board.y_max - half_h))
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

    # Reheating rounds — v9: progressively shorter but still substantial
    for reheat_i in range(config.reheat_count):
        reheat_t0 = t0 * config.reheat_ratio * (0.7 ** reheat_i)  # gentler decay
        if config.verbose:
            print(f"  Reheat {reheat_i + 1}: T0={reheat_t0:.4f}")
        _run_sa_pass(
            model, cost_state, moveable_indices,
            reheat_t0, config,
            max_iter_override=config.max_iterations // 3,
        )

    # Greedy refinement
    if config.verbose:
        print("  Greedy refinement...")
    _greedy_refine(model, cost_state, moveable_indices, config)

    # v7: Dedicated overlap resolver — guarantee zero overlaps before returning
    if cost_state.overlap_count > 0:
        if config.verbose:
            print(f"  Overlap resolution ({cost_state.overlap_count} remaining)...")
        _resolve_overlaps_greedy(model, cost_state, moveable_indices, config)

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

    # Save pre-SA positions so we can restore if SA makes things worse
    pre_sa_positions = _save_positions(model, moveable_indices)

    final_cost = simulate_annealing(model, cost_state, moveable_indices, config)

    # Safety net: if SA+greedy made cost worse, revert to pre-SA state
    if final_cost > initial_cost:
        if config.verbose:
            print(f"  SA worsened cost ({initial_cost:.1f} → {final_cost:.1f}), reverting to pre-SA positions")
        _restore_positions(model, pre_sa_positions)
        cost_state._compute_all()
        final_cost = cost_state.normalized_cost

    return {
        'initial_cost': initial_cost,
        'final_cost': final_cost,
        'initial_hpwl': initial_hpwl,
        'final_hpwl': cost_state.hpwl,
        'overlap_count': cost_state.overlap_count,
        'improvement': initial_cost - final_cost,
    }
