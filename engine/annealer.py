"""Simulated Annealing engine for PCB component placement.

v12 strategy — fixes incremental drift + safe best-state tracking:
  - CRITICAL FIX: periodic _compute_all() resync every 20 steps prevents
    incremental overlap tracking from drifting from ground truth.  Without
    this, SA thinks it has 34 overlaps when it actually has 87.
  - CRITICAL FIX: safe best-state restore — after restoring "best" positions,
    verify against _compute_all().  If the true overlap count is higher than
    the final SA state, fall back to final (which was recently resync'd).
  - CRITICAL FIX: overlap cap allows recovery — when already above cap,
    only reject moves that INCREASE overlaps, not all moves.
  - Density-aware overlap cap factor (1.5x for dense, 2.0x for sparse).
  - Board density determines SA aggressiveness:
    * Sparse boards (density < 0.25): penalty_scale_min=0.50, full exploration
    * Dense boards (density > 0.40): penalty_scale_min=0.85+, focused refinement
  - Fewer reheats for dense boards (reheats re-destroy placements)
  - Greedy refinement with overlap rejection
  - Overlap resolver with neighbor-aware moves
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
    reheat_count: int = 3              # 3 reheats for sparse boards, fewer for dense
    reheat_ratio: float = 0.40         # reheat to 40% of previous T0
    calibration_samples: int = 500     # more samples for robust T0 estimation
    initial_accept_rate: float = 0.92   # target high accept rate at start
    penalty_scale_min: float = 0.50    # will be overridden by density-adaptive logic
    min_temperature: float = 1e-8
    freeze_threshold: float = 0.005    # stop sooner when frozen
    greedy_nudge_distances: tuple[float, ...] = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
    greedy_rotations: tuple[float, ...] = (90.0, 180.0, 270.0)
    greedy_improve_threshold: float = 0.5  # accept smaller improvements
    overlap_cap_factor: float = 2.0    # reject moves that exceed this * initial overlaps
    verbose: bool = True


def _compute_density(model: BoardModel) -> float:
    """Compute board density = total component area / board area."""
    board = model.board
    board_area = board.width * board.height
    if board_area <= 0:
        return 0.0
    comp_area = sum(c.effective_width * c.effective_height for c in model.components)
    return min(1.0, comp_area / board_area)


def _density_adaptive_penalty_scale(density: float) -> float:
    """Compute penalty_scale_min based on board density.

    Sparse boards (density < 0.25): penalty_scale_min = 0.50 (let SA explore)
    Dense boards (density > 0.40): penalty_scale_min = 0.85+ (prevent overlap explosion)

    For test4 (density=0.43): penalty_scale_min ≈ 0.87
    This prevents SA from accepting too many overlap-increasing moves.
    """
    if density < 0.25:
        return 0.50
    elif density < 0.40:
        # Linear ramp from 0.50 to 0.85
        t = (density - 0.25) / 0.15
        return 0.50 + 0.35 * t
    else:
        # Dense boards: 0.85 to 0.95
        t = min(1.0, (density - 0.40) / 0.30)
        return 0.85 + 0.10 * t


def _density_adaptive_reheat_count(density: float) -> int:
    """Fewer reheats for dense boards (reheats re-destroy placements)."""
    if density < 0.25:
        return 3
    elif density < 0.40:
        return 2
    else:
        return 1


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

    v11: For dense boards, cap T0 to prevent excessive initial disruption.
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
        # v10 BUG FIX: use snap['overlap_penalty'] (full sum) not
        # sum(snap['pair_overlaps'].values()) (partial — only moved-component
        # overlaps).  The old code compared new_cost (all overlaps) with
        # old_cost (only moved-component overlaps), inflating delta by the
        # sum of all unaffected overlaps.
        old_cost = (snap['hpwl']
                    + OVERLAP_WEIGHT * snap['overlap_penalty'] * config.penalty_scale_min
                    + BOUNDARY_WEIGHT * snap['boundary_penalty'] * config.penalty_scale_min
                    + CONSTRAINT_WEIGHT * snap['constraint_total'] * config.penalty_scale_min)

        delta = new_cost - old_cost
        if delta > 0:
            deltas.append(delta)

        # Revert
        cost_state.restore(snap)
        revert_move(model, undo)

    if not deltas:
        return 1.0

    # Use 75th percentile for dense boards (lower T0 = less disruption)
    # vs 90th percentile for sparse boards (higher T0 = more exploration)
    density = _compute_density(model)
    if density > 0.35:
        pct = 0.75
    elif density > 0.25:
        pct = 0.85
    else:
        pct = 0.90

    idx_pct = int(len(deltas) * pct)
    pct_delta = sorted(deltas)[min(idx_pct, len(deltas) - 1)]
    t0 = -pct_delta / math.log(config.initial_accept_rate)

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
    initial_overlap_count: int = 0,
) -> float:
    """Run one SA pass. Returns final temperature.

    v11: Hard overlap cap — reject moves that increase overlap count
    beyond overlap_cap_factor * initial_overlap_count. This prevents
    SA from exploring states that are clearly worse (too many overlaps).
    """
    T = t0
    n_moveable = len(moveable_indices)
    moves_per_temp = max(300, 20 * n_moveable)
    board = model.board
    max_window = max(board.width, board.height) * 0.10
    min_window = max(board.width, board.height) * 0.003
    max_iter = max_iter_override or config.max_iterations

    best_positions = _save_positions(model, moveable_indices)
    best_cost = cost_state.normalized_cost
    best_overlap_count = cost_state.overlap_count  # v12: track overlaps at best
    no_improve_count = 0

    # v12: Save current positions so we can fall back if "best" is bad
    current_positions = _save_positions(model, moveable_indices)
    current_overlap_count = cost_state.overlap_count

    # Overlap cap: reject moves that create too many overlaps
    # If initial placement has 0 overlaps, set cap=0 (never accept overlap-creating moves)
    if initial_overlap_count == 0:
        overlap_cap = 0  # preserve the overlap-free state
    else:
        overlap_cap = max(initial_overlap_count * config.overlap_cap_factor, initial_overlap_count + 10)

    for step in range(max_iter):
        t_ratio = math.log(T + 1.0) / math.log(t0 + 1.0) if t0 > 0 else 0.0
        t_ratio = max(0.0, min(1.0, t_ratio))

        # Adaptive penalty scale: hot→discount penalties, cold→full weight
        scale = config.penalty_scale_min + (1.0 - config.penalty_scale_min) * (1.0 - t_ratio)
        cost_state.update_penalty_scale(scale)

        window = min_window + (max_window - min_window) * t_ratio
        noise = board.width * 0.03 * t_ratio

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

            # v12: Hard overlap cap — reject moves that create too many overlaps
            # For overlap-free initial placements, reject ANY overlap-increasing move
            # When already above cap, allow overlap-reducing moves (recovery)
            if overlap_cap == 0:
                # Strict mode: reject if overlap count increased at all
                if cost_state.overlap_count > initial_overlap_count:
                    cost_state.restore(snap)
                    revert_move(model, undo)
                    continue
            elif cost_state.overlap_count > overlap_cap:
                # Above cap: only reject if overlaps INCREASED from current state
                # Allow overlap-reducing moves even if still above cap (recovery path)
                if cost_state.overlap_count > current_overlap_count:
                    cost_state.restore(snap)
                    revert_move(model, undo)
                    continue

            # v10 BUG FIX: use snap['overlap_penalty'] (full sum) not
            # sum(snap['pair_overlaps'].values()) (partial).  See _calibrate_t0.
            old_cost = (snap['hpwl']
                        + OVERLAP_WEIGHT * snap['overlap_penalty'] * scale
                        + BOUNDARY_WEIGHT * snap['boundary_penalty'] * scale
                        + CONSTRAINT_WEIGHT * snap['constraint_total'] * scale)

            delta = new_cost - old_cost

            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                accepted += 1
                # v12: Update current overlap tracking
                current_overlap_count = cost_state.overlap_count
                # Track best
                cur_norm = cost_state.normalized_cost
                if cur_norm < best_cost:
                    best_cost = cur_norm
                    best_positions = _save_positions(model, moveable_indices)
                    best_overlap_count = cost_state.overlap_count
                    no_improve_count = 0
            else:
                cost_state.restore(snap)
                revert_move(model, undo)

        accept_rate = accepted / max(moves_per_temp, 1)

        # v12: Periodic resync — incremental update drifts from ground truth,
        # undercounting overlaps by ~30-40 on dense boards. Resync every 5
        # steps to balance accuracy vs performance.
        if step > 0 and step % 5 == 0:
            cost_state._compute_all()
            current_overlap_count = cost_state.overlap_count

        if config.verbose and step % 10 == 0:
            print(f"    T={T:.2f} accept={accept_rate:.2f} cost={cost_state.normalized_cost:.1f} "
                  f"hpwl={cost_state.hpwl:.1f} overlaps={cost_state.overlap_count}")

        # Adaptive cooling
        if accept_rate > 0.6:
            T *= 0.95
        elif accept_rate > 0.4:
            T *= 0.97
        elif accept_rate > 0.2:
            T *= 0.985
        elif accept_rate > 0.05:
            T *= 0.99
        else:
            T *= 0.98

        T = max(T, config.min_temperature)

        # Track stall
        no_improve_count += 1

        # v12: Save current state for safe fallback
        current_positions = _save_positions(model, moveable_indices)
        current_overlap_count = cost_state.overlap_count

        # Early stop when frozen AND stalled
        if accept_rate < config.freeze_threshold and T < t0 * 0.03:
            if config.verbose:
                print(f"    Frozen at step {step}, accept_rate={accept_rate:.3f}")
            break

        # Stall detection
        if no_improve_count > max_iter * 0.4 and accept_rate < 0.05:
            if config.verbose:
                print(f"    Stalled at step {step}, no improvement for {no_improve_count} steps")
            break

    # v12: Safe best-state restore
    # The incremental update can drift from ground truth, so the "best" state
    # might have more overlaps than expected after _compute_all().  If restoring
    # "best" gives worse overlaps than the current state, keep current instead.
    final_positions = _save_positions(model, moveable_indices)
    final_overlap_count = cost_state.overlap_count

    _restore_positions(model, best_positions)
    cost_state._compute_all()
    true_best_overlaps = cost_state.overlap_count

    if true_best_overlaps > final_overlap_count:
        # "Best" state has more overlaps than final — incremental drift made it
        # look better than it was.  Fall back to the final SA state which is
        # more reliable (resync'd recently).
        if config.verbose:
            print(f"    Best-state restore rejected: {true_best_overlaps} overlaps "
                  f"(incremental said {best_overlap_count}) vs final {final_overlap_count}")
        _restore_positions(model, final_positions)
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

        if sweep >= 8:
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

    v11: Added overlap-aware moves — check if resolving one overlap
    creates new overlaps with other components, and choose the move
    that minimizes total overlap count.
    """
    cost_state.update_penalty_scale(1.0)
    board = model.board
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1),
                  (1, 1), (-1, 1), (1, -1), (-1, -1)]
    nudge_dists = [0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0]
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

        # Sort by overlap area (smallest first — easier to resolve)
        pair_areas = []
        for i, j in overlap_pairs:
            c1 = model.components[i]
            c2 = model.components[j]
            area = c1.overlap_area(c2)
            pair_areas.append((area, i, j))
        pair_areas.sort()

        for area, i, j in pair_areas:
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
    # v11: Don't blindly clamp — check if clamping creates new overlaps.
    # Only clamp if it doesn't increase the overlap count.
    prev_overlaps = cost_state.overlap_count
    for idx in moveable_indices:
        comp = model.components[idx]
        half_w = comp.effective_width / 2.0
        half_h = comp.effective_height / 2.0
        new_x = max(board.x_min + half_w, min(comp.x, board.x_max - half_w))
        new_y = max(board.y_min + half_h, min(comp.y, board.y_max - half_h))
        if new_x != comp.x or new_y != comp.y:
            old_x, old_y = comp.x, comp.y
            comp.x = new_x
            comp.y = new_y
            cost_state.incremental_update({idx})
            # If clamping created new overlaps, undo
            if cost_state.overlap_count > prev_overlaps:
                comp.x = old_x
                comp.y = old_y
                cost_state.incremental_update({idx})
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

    # Record initial overlap count for hard cap
    initial_overlap_count = cost_state.overlap_count

    # Calibrate T0
    t0 = _calibrate_t0(model, cost_state, moveable_indices, config)
    if config.verbose:
        density = _compute_density(model)
        print(f"  SA: T0={t0:.4f}, {len(moveable_indices)} moveable components, "
              f"density={density:.3f}, penalty_scale_min={config.penalty_scale_min:.2f}")

    # Main SA pass
    _run_sa_pass(model, cost_state, moveable_indices, t0, config,
                 initial_overlap_count=initial_overlap_count)

    # Reheating rounds — v11: fewer for dense boards
    for reheat_i in range(config.reheat_count):
        reheat_t0 = t0 * config.reheat_ratio * (0.7 ** reheat_i)
        if config.verbose:
            print(f"  Reheat {reheat_i + 1}: T0={reheat_t0:.4f}")
        _run_sa_pass(
            model, cost_state, moveable_indices,
            reheat_t0, config,
            max_iter_override=config.max_iterations // 3,
            initial_overlap_count=initial_overlap_count,
        )

    # Greedy refinement
    if config.verbose:
        print("  Greedy refinement...")
    _greedy_refine(model, cost_state, moveable_indices, config)

    # Dedicated overlap resolver
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

    v11: Computes board density and adjusts SA parameters accordingly.
    Dense boards get higher penalty_scale_min and fewer reheats to
    prevent SA from destroying the placement.
    """
    if config is None:
        config = SAConfig(verbose=verbose)
    elif verbose and not config.verbose:
        config = SAConfig(**{**config.__dict__, 'verbose': True})

    # v11: Density-adaptive SA parameters
    density = _compute_density(model)
    adaptive_penalty_min = _density_adaptive_penalty_scale(density)
    adaptive_reheat_count = _density_adaptive_reheat_count(density)

    # Override config with density-adaptive values (only if user didn't explicitly set)
    if config.penalty_scale_min == 0.50:  # default value — user didn't override
        config.penalty_scale_min = adaptive_penalty_min
    if config.reheat_count == 3:  # default value — user didn't override
        config.reheat_count = adaptive_reheat_count
    # v12: Density-aware overlap cap — tighter for dense boards
    if config.overlap_cap_factor == 2.0:  # default value
        if density > 0.35:
            config.overlap_cap_factor = 1.5  # dense: allow 50% increase
        elif density > 0.25:
            config.overlap_cap_factor = 1.75  # medium: allow 75% increase

    if config.verbose:
        print(f"  Density: {density:.3f}, penalty_scale_min: {config.penalty_scale_min:.2f}, "
              f"reheat_count: {config.reheat_count}, overlap_cap: {config.overlap_cap_factor:.1f}x")

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
