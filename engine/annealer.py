"""Simulated Annealing engine for PCB component placement.

This module includes the SA stability fixes, density-aware tuning,
global-best tracking across reheats, and HPWL-based revert logic used
by the placement pipeline.
Enhanced greedy refinement + greedy swap refinement as default
optimization path. Global SA is now opt-in via --sa flag.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from models.board_model import BoardModel
from engine.cost_state import CostState, OVERLAP_WEIGHT, OVERLAP_COUNT_WEIGHT, BOUNDARY_WEIGHT, CONSTRAINT_WEIGHT
from engine.moves import (
    select_move_type, get_moveable_indices,
    do_translate, do_swap, do_rotate, do_median,
    revert_move, affected_indices, MoveUndo,
)
from engine.congestion import rudy_congestion_penalty, rudy_gradient_for_comp


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
    skip_sa: bool = False              # v13: if True, skip global SA, run greedy+swap+greedy only
    rudy_weight: float = 0.3           # RUDY congestion penalty weight (0 = disabled)
    rudy_grid_resolution: float = 2.0  # RUDY grid cell size in mm
    # Auto-disable SA on tiny/large boards.  Set to 0 to disable the gate.
    # Tiny boards (≤5 comps): greedy is already near-optimal, SA noise hurts.
    # Large boards (≥50 comps): SA is 8-25x slower with diminishing returns.
    sa_auto_disable_min_components: int = 5
    sa_auto_disable_max_components: int = 50
    # Spread floor: reject SA moves that collapse component spread below this
    # fraction of board dimensions.  Prevents the "SA crams everything into
    # one corner to minimise HPWL" failure mode.  0 = disabled.
    # At 0.10, a 100x80 board requires std_x ≥ 10mm AND std_y ≥ 8mm
    # (10% of each dimension) — components must span at least 2 sigma each way.
    spread_floor_fraction: float = 0.10


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

    Sparse boards (density < 0.25): penalty_scale_min = 0.80
    Dense boards (density > 0.40): penalty_scale_min = 0.95+

    Overlap penalties are stronger now (OVERLAP_WEIGHT=25, COUNT_WEIGHT=12),
    so even at 0.80 the penalties are significant.
    """
    if density < 0.25:
        return 0.80
    elif density < 0.40:
        # Linear ramp from 0.80 to 0.95
        t = (density - 0.25) / 0.15
        return 0.80 + 0.15 * t
    else:
        # Dense boards: 0.95 to 1.00
        t = min(1.0, (density - 0.40) / 0.30)
        return 0.95 + 0.05 * t


def _density_adaptive_reheat_count(density: float) -> int:
    """Fewer reheats for dense boards, but still at least 2.

    Dense boards get 2 reheats (was 1). A single reheat is
    insufficient - the best state often gets rejected due to overlap
    drift. With accurate per-step resync, 2 reheats give better results.
    """
    if density < 0.25:
        return 3
    elif density < 0.40:
        return 2
    else:
        return 2


def _violates_spread_floor(
    model: BoardModel,
    board,
    spread_floor_fraction: float,
) -> bool:
    """Return True if the current placement's spread is below the floor.

    Computes the standard deviation of movable-component x and y positions
    and compares each to ``spread_floor_fraction * board_dimension``.  If
    EITHER dimension's std-dev is below its floor, the placement is too
    bunched in that axis and the move should be rejected.

    This is a cheap O(n) check called on every low-temperature SA move,
    so it avoids building any intermediate data structures — just a
    single pass over movable components.

    Only movable, non-connector components are counted — fixed components
    and edge connectors have predetermined positions and shouldn't
    influence the spread metric.
    """
    n = 0
    sum_x = 0.0
    sum_y = 0.0
    for c in model.components:
        if c.is_fixed or getattr(c, 'component_type', '') == 'connector':
            continue
        sum_x += c.x
        sum_y += c.y
        n += 1
    if n < 2:
        return False  # can't compute meaningful std-dev with <2 components

    mean_x = sum_x / n
    mean_y = sum_y / n
    var_x = 0.0
    var_y = 0.0
    for c in model.components:
        if c.is_fixed or getattr(c, 'component_type', '') == 'connector':
            continue
        var_x += (c.x - mean_x) ** 2
        var_y += (c.y - mean_y) ** 2
    std_x = math.sqrt(var_x / n)
    std_y = math.sqrt(var_y / n)

    floor_x = spread_floor_fraction * board.width
    floor_y = spread_floor_fraction * board.height
    return std_x < floor_x or std_y < floor_y


def _centroid_offset(model: BoardModel, board) -> float:
    """Distance from the movable-component centroid to the board center.

    A high value means the component cluster is off-center — either
    bunched in a corner or pushed off-board.  Used as a final revert
    check in run_sa to catch cases where SA found a low-HPWL placement
    that's visually terrible (components scattered off-board).
    """
    n = 0
    sum_x = 0.0
    sum_y = 0.0
    for c in model.components:
        if c.is_fixed or getattr(c, 'component_type', '') == 'connector':
            continue
        sum_x += c.x
        sum_y += c.y
        n += 1
    if n == 0:
        return 0.0
    cx = sum_x / n
    cy = sum_y / n
    board_cx = (board.x_min + board.x_max) / 2.0
    board_cy = (board.y_min + board.y_max) / 2.0
    return math.hypot(cx - board_cx, cy - board_cy)


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
        old_bboxes = cost_state.old_bboxes_from_states(undo.old_states)
        snap = cost_state.snapshot(moved, old_bboxes=old_bboxes)
        new_cost = cost_state.incremental_update(moved)
        # v10 BUG FIX: use snap['overlap_penalty'] (full sum) not
        # sum(snap['pair_overlaps'].values()) (partial — only moved-component
        # overlaps).  The old code compared new_cost (all overlaps) with
        # old_cost (only moved-component overlaps), inflating delta by the
        # sum of all unaffected overlaps.
        old_cost = (snap['hpwl']
                    + (OVERLAP_WEIGHT * snap['overlap_penalty'] + OVERLAP_COUNT_WEIGHT * len(snap['pair_overlaps'])) * config.penalty_scale_min
                    + BOUNDARY_WEIGHT * snap['boundary_penalty'] * config.penalty_scale_min
                    + CONSTRAINT_WEIGHT * snap['constraint_total'] * config.penalty_scale_min)

        delta = new_cost - old_cost
        if delta > 0:
            deltas.append(delta)

        # Revert — model FIRST so spatial index is consistent
        revert_move(model, undo)
        cost_state.restore(snap)

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
    global_best_cost: float = float('inf'),
    global_best_positions: dict | None = None,
    global_best_overlap_count: int = 999999,
) -> tuple[float, float, dict, int]:
    """Run one SA pass. Returns (final_temperature, global_best_cost,
    global_best_positions, global_best_overlap_count).

    Tracks global best across all passes (main + reheats).
    Each reheat receives the global best from previous passes and
    preserves it — the old code let reheats overwrite the global best
    with a worse state, then greedy couldn't recover enough.

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
    best_overlap_count = cost_state.overlap_count

    no_improve_count = 0

    # Global best state across all passes (main + reheats)
    if global_best_positions is None:
        global_best_cost = best_cost
        global_best_positions = best_positions
        global_best_overlap_count = best_overlap_count

    # Unified adaptive overlap cap — replaces the old dual-path
    # (strict mode for initial_overlap_count==0 vs relaxed mode for >0).
    # The old strict mode (max 3-4 overlaps) caused SA to freeze when
    # resync discovered more overlaps than the cap allowed.
    # The new cap scales with density and component count, and always
    # provides headroom above the current overlap count.
    density = _compute_density(model)
    density_factor = max(1.0, density / 0.25)  # 1.0 at sparse, ~1.6 at dense
    base_overlap_cap = max(8, int(n_moveable * 0.15 * density_factor))

    # Current overlap count (incremental, updated each move)
    current_overlap_count = cost_state.overlap_count

    # RUDY congestion state — active for any non-trivial board (>=30 comps).
    # The old >=100 threshold disabled RUDY for small/mid boards, leaving SA
    # with no spreading force. Lowered to 30 so boards like cbb (68 comps)
    # get congestion-driven spreading. NOTE: in this annealer path RUDY enters
    # acceptance only as a best-state tiebreaker (lines below); the actual
    # spreading force comes from the RUDY gradient used as a translate-move
    # bias. For a true acceptance-cost spreading force, see smart_placement.py
    # _optimize_interior_sa where RUDY is added to new_cost directly.
    rudy_active = (config.rudy_weight > 0 and len(model.components) >= 30)
    rudy_penalty = 0.0
    rudy_step_counter = 0
    if rudy_active:
        try:
            rudy_penalty, _rudy_peak, _rudy_avg, _rudy_overflow = \
                rudy_congestion_penalty(model, config.rudy_grid_resolution)
        except Exception:
            rudy_active = False
            rudy_penalty = 0.0

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
            rudy_bias_dx = 0.0
            rudy_bias_dy = 0.0
            if rudy_active and mt == 'translate':
                _bias_idx = random.choice(moveable_indices)
                _bias_comp = model.components[_bias_idx]
                try:
                    rudy_bias_dx, rudy_bias_dy = rudy_gradient_for_comp(
                        _bias_comp, model, config.rudy_grid_resolution)
                    bias_scale = window * 0.05
                    rudy_bias_dx *= bias_scale
                    rudy_bias_dy *= bias_scale
                except Exception:
                    rudy_bias_dx = 0.0
                    rudy_bias_dy = 0.0
            if mt == 'translate':
                undo = do_translate(model, moveable_indices, t_ratio, window,
                                    bias_dx=rudy_bias_dx, bias_dy=rudy_bias_dy)
            elif mt == 'swap':
                undo = do_swap(model, moveable_indices)
            elif mt == 'rotate':
                undo = do_rotate(model, moveable_indices)
            else:
                undo = do_median(model, moveable_indices, t_ratio, noise)

            if not undo.old_states:
                continue

            moved = affected_indices(undo)
            old_bboxes = cost_state.old_bboxes_from_states(undo.old_states)
            snap = cost_state.snapshot(moved, old_bboxes=old_bboxes)

            new_cost = cost_state.incremental_update(moved)

            # Unified adaptive overlap cap — single path for all boards.
            # Scales with density, component count, and temperature.
            # At high T: generous cap for exploration (1.5x base).
            # At low T: tighter cap (1.0x base) for exploitation.
            # Always provides headroom above current overlap count to
            # prevent freeze after resync discovers drift.
            cap_relaxation = 1.0 + 0.5 * t_ratio  # 1.5x at hot, 1.0x at cold
            overlap_cap = int(base_overlap_cap * cap_relaxation)
            # Ensure cap provides headroom above current state (prevents freeze)
            overlap_cap = max(overlap_cap, current_overlap_count + 3)
            if cost_state.overlap_count > overlap_cap:
                # Above cap: only reject if overlaps INCREASED from current
                if cost_state.overlap_count > current_overlap_count:
                    cost_state.restore(snap)
                    revert_move(model, undo)
                    continue

            # v10 BUG FIX: use snap['overlap_penalty'] (full sum) not
            # sum(snap['pair_overlaps'].values()) (partial).  See _calibrate_t0.
            old_cost = (snap['hpwl']
                        + (OVERLAP_WEIGHT * snap['overlap_penalty'] + OVERLAP_COUNT_WEIGHT * len(snap['pair_overlaps'])) * scale
                        + BOUNDARY_WEIGHT * snap['boundary_penalty'] * scale
                        + CONSTRAINT_WEIGHT * snap['constraint_total'] * scale)

            delta = new_cost - old_cost

            accept_move = (delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)))
            # Temperature-dependent overlap rejection.
            # At HIGH temperature (t_ratio > 0.4), allow SA to explore via normal
            # Metropolis probability — most exploratory moves create temporary overlaps.
            # At LOW temperature, hard-reject overlap increases to preserve quality.
            # The old code hard-rejected at ALL temperatures, causing SA to freeze.
            if t_ratio < 0.4:
                if cost_state.overlap_count > current_overlap_count and cost_state.normalized_cost >= best_cost:
                    accept_move = False

            # Spread floor: reject moves that collapse component spread below
            # a fraction of board dimensions.  Prevents the "SA crams
            # everything into one corner to minimise HPWL" failure mode
            # identified in the multi-metric A/B test.  Only active at LOW
            # temperature (t_ratio < 0.4) — at high temperature we want SA
            # to freely explore, including temporarily compact configurations.
            # The floor is computed from the board dimensions, not the
            # current spread, so it's a stable target.
            if accept_move and t_ratio < 0.4 and config.spread_floor_fraction > 0:
                if _violates_spread_floor(model, board, config.spread_floor_fraction):
                    accept_move = False

            if accept_move:
                accepted += 1
                current_overlap_count = cost_state.overlap_count
            else:
                revert_move(model, undo)
                cost_state.restore(snap)

        accept_rate = accepted / max(moves_per_temp, 1)

        if rudy_active:
            rudy_step_counter += 1
            if rudy_step_counter >= 50:
                rudy_step_counter = 0
                try:
                    rudy_penalty, _rudy_peak, _rudy_avg, _rudy_overflow = \
                        rudy_congestion_penalty(model, config.rudy_grid_resolution)
                except Exception:
                    rudy_penalty = 0.0

        rudy_cost = rudy_penalty * config.rudy_weight if rudy_active else 0.0

        # Resync every 3 steps for accurate overlap tracking.
        # Every step is too expensive on large boards. The old_bboxes_from_states
        # fix makes incremental tracking much more reliable, so 3 steps is
        # a good balance between accuracy and performance.
        if step % 3 == 0:
            cost_state._compute_all()
            current_overlap_count = cost_state.overlap_count

            # Update best state using verified cost (after resync)
            # Use overlap count as tiebreaker: prefer fewer overlaps at same cost
            verified_cost = cost_state.normalized_cost
            if (cost_state.overlap_count < best_overlap_count or
                    (cost_state.overlap_count == best_overlap_count and verified_cost + rudy_cost < best_cost)):
                best_cost = verified_cost
                best_positions = _save_positions(model, moveable_indices)

            # Update global best across all passes (main + reheats)
            if (cost_state.overlap_count < global_best_overlap_count or
                    (cost_state.overlap_count == global_best_overlap_count and verified_cost + rudy_cost < global_best_cost)):
                global_best_cost = verified_cost
                global_best_positions = _save_positions(model, moveable_indices)
                global_best_overlap_count = cost_state.overlap_count

            best_overlap_count = cost_state.overlap_count
            no_improve_count = 0

        if config.verbose and step % 10 == 0:
            print(f"    T={T:.2f} accept={accept_rate:.2f} cost={verified_cost:.1f} "
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

    # Restore global best, not just pass-local best. This prevents
    # reheats from destroying the main pass's best state.
    _restore_positions(model, global_best_positions)
    cost_state._compute_all()

    if config.verbose:
        print(f"    Restored best state: cost={cost_state.normalized_cost:.1f} "
              f"overlaps={cost_state.overlap_count}")

    return T, global_best_cost, global_best_positions, global_best_overlap_count


def _greedy_refine(
    model: BoardModel,
    cost_state: CostState,
    moveable_indices: list[int],
    config: SAConfig,
    enhanced: bool = False,
) -> None:
    """Greedy refinement: try nudges and rotations, accept only if improving.

    Hard overlap rejection: greedy does small moves, so rejecting
    overlap-creating moves doesn't limit exploration.  This ensures
    greedy never creates new overlaps.

    Enhanced mode adds:
    - Finer nudge distances for more precise placement
    - Lower improve threshold for more aggressive acceptance
    - Adaptive sweep count based on component count
    - Position-aware ordering: process components by HPWL contribution
    """
    # Reset penalty scale for greedy — we want full-weight evaluation
    cost_state.update_penalty_scale(1.0)

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1),
                  (1, 1), (-1, 1), (1, -1), (-1, -1)]

    if enhanced:
        # Enhanced: finer nudge distances + extended range
        nudge_distances = (0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 2.5, 4.0, 6.0)
        improve_threshold = 0.1  # accept smaller improvements
        max_sweeps = min(15, 5 + len(moveable_indices) // 10)
    else:
        nudge_distances = config.greedy_nudge_distances
        improve_threshold = config.greedy_improve_threshold
        max_sweeps = 8

    improved = True
    sweep = 0
    best_sweep_cost = cost_state.normalized_cost
    best_sweep_positions = _save_positions(model, moveable_indices)

    while improved:
        improved = False
        sweep += 1

        # In enhanced mode, order components by their HPWL contribution
        # (components connected to high-HPWL nets first)
        if enhanced:
            indices_to_process = _order_by_hpwl_contribution(model, cost_state, moveable_indices)
        else:
            indices_to_process = moveable_indices

        for idx in indices_to_process:
            comp = model.components[idx]
            # Try nudges
            for dx_dir, dy_dir in directions:
                for dist in nudge_distances:
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

                    if new_cost < base_cost - improve_threshold:
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

                if new_cost < base_cost - improve_threshold:
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
                  f"hpwl={cost_state.hpwl:.1f} overlaps={cost_state.overlap_count}")

        if sweep >= max_sweeps:
            break

    # Final restore to best sweep result
    if cost_state.normalized_cost > best_sweep_cost:
        _restore_positions(model, best_sweep_positions)
        cost_state._compute_all()


def _order_by_hpwl_contribution(
    model: BoardModel,
    cost_state: CostState,
    moveable_indices: list[int],
) -> list[int]:
    """Order moveable indices by their HPWL contribution (highest first).

    Components connected to high-HPWL nets are processed first,
    so they get the most benefit from greedy nudges.
    """
    # Build per-component HPWL contribution
    comp_hpwl = {}
    for idx in moveable_indices:
        comp_nets = cost_state._comp_nets[idx]
        total = 0.0
        for net_name in comp_nets:
            if net_name in cost_state._net_hpwl:
                total += cost_state._net_hpwl[net_name]
        comp_hpwl[idx] = total

    # Sort descending by HPWL contribution
    return sorted(moveable_indices, key=lambda i: comp_hpwl.get(i, 0.0), reverse=True)


def _build_swap_candidates(
    model: BoardModel,
    cost_state: CostState,
    moveable_indices: list[int],
) -> dict[int, list[int]]:
    """Build candidate swap partners for each moveable component.

    Smart swap: Instead of brute-force O(n^2) all-pairs, filter
    candidates using three criteria:

    1. **Net-connected**: Components sharing at least one signal net
       are always candidates (swapping them directly reduces HPWL).
    2. **Size-class**: Components with similar area (within 3x ratio)
       are candidates (avoids placing a large IC in a resistor's spot).
    3. **Proximity**: Components within 3x of each other's HPWL
       contribution are candidates (prioritises high-impact swaps).

    For small boards (<60 components), falls back to all-pairs since
    the overhead of filtering exceeds the brute-force cost.
    """
    n = len(moveable_indices)

    # For small boards, all-pairs is fine
    if n < 60:
        all_set = set(moveable_indices)
        return {idx: [j for j in moveable_indices if j != idx] for idx in moveable_indices}

    # Build per-component data
    comp_area = {}
    comp_nets = {}
    comp_hpwl = {}

    for idx in moveable_indices:
        comp = model.components[idx]
        comp_area[idx] = comp.effective_width * comp.effective_height
        # Store as a sorted list so downstream iteration is deterministic
        # (set iteration varies with id() under ASLR, which made the
        # swap-candidate order — and therefore the entire greedy+swap
        # trajectory — non-deterministic across runs).
        comp_nets[idx] = sorted(cost_state._comp_nets[idx])
        # HPWL contribution
        total = 0.0
        for net_name in comp_nets[idx]:
            if net_name in cost_state._net_hpwl:
                total += cost_state._net_hpwl[net_name]
        comp_hpwl[idx] = total

    # Build net → component index map for fast lookup
    net_to_comps: dict[str, list[int]] = {}
    for idx in moveable_indices:
        for net_name in comp_nets[idx]:
            if net_name not in net_to_comps:
                net_to_comps[net_name] = []
            net_to_comps[net_name].append(idx)

    # Build size-class buckets (log-scale area bins)
    size_buckets: dict[int, list[int]] = {}
    for idx in moveable_indices:
        area = comp_area[idx]
        if area <= 0:
            bucket = 0
        else:
            bucket = int(math.log2(area + 1))
        if bucket not in size_buckets:
            size_buckets[bucket] = []
        size_buckets[bucket].append(idx)

    # Build candidates for each component
    candidates: dict[int, list[int]] = {idx: [] for idx in moveable_indices}
    seen_pairs: set[tuple[int, int]] = set()

    # Strategy 1: Net-connected candidates (highest priority)
    for net_name, comp_list in net_to_comps.items():
        if net_name in cost_state._power_nets:
            continue  # skip power nets — too many components, low value
        for i_idx in range(len(comp_list)):
            for j_idx in range(i_idx + 1, len(comp_list)):
                a, b = comp_list[i_idx], comp_list[j_idx]
                pair = (min(a, b), max(a, b))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    candidates[a].append(b)
                    candidates[b].append(a)

    # Strategy 2: Size-class candidates (adjacent buckets)
    for bucket, comps in size_buckets.items():
        adjacent = list(comps)
        for adj_bucket in [bucket - 1, bucket + 1]:
            if adj_bucket in size_buckets:
                adjacent.extend(size_buckets[adj_bucket])
        for idx in comps:
            for other in adjacent:
                if other == idx:
                    continue
                # Check area ratio
                area_a = comp_area[idx]
                area_b = comp_area[other]
                if area_a > 0 and area_b > 0:
                    ratio = max(area_a, area_b) / min(area_a, area_b)
                    if ratio <= 3.0:
                        pair = (min(idx, other), max(idx, other))
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            candidates[idx].append(other)

    # Strategy 3: Top-20% HPWL contributors can swap with each other
    # (even if not net-connected or size-similar)
    sorted_by_hpwl = sorted(moveable_indices, key=lambda i: comp_hpwl[i], reverse=True)
    top_k = max(5, n // 5)
    top_comps = sorted_by_hpwl[:top_k]
    for i_idx in range(len(top_comps)):
        for j_idx in range(i_idx + 1, len(top_comps)):
            a, b = top_comps[i_idx], top_comps[j_idx]
            pair = (min(a, b), max(a, b))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                candidates[a].append(b)
                candidates[b].append(a)

    return candidates


def _try_single_swap(
    model: BoardModel,
    cost_state: CostState,
    idx_a: int,
    idx_b: int,
    improve_threshold: float = 0.5,
) -> bool:
    """Try swapping components idx_a and idx_b. Returns True if swap accepted.

    Tries: position swap, then position+rotation swap. Accepts the best
    improving option. Reverts everything if no improvement.
    """
    c_a = model.components[idx_a]
    c_b = model.components[idx_b]

    prev_overlap_count = cost_state.overlap_count
    base_cost = cost_state.normalized_cost

    old_ax, old_ay, old_arot = c_a.x, c_a.y, c_a.rotation
    old_bx, old_by, old_brot = c_b.x, c_b.y, c_b.rotation

    # Try position swap (keep each component's own rotation)
    c_a.x, c_b.x = c_b.x, c_a.x
    c_a.y, c_b.y = c_b.y, c_a.y
    new_cost = cost_state.incremental_update({idx_a, idx_b})

    if cost_state.overlap_count > prev_overlap_count:
        # Position swap creates overlaps — revert and try with rotation exchange
        c_a.x, c_a.y = old_ax, old_ay
        c_b.x, c_b.y = old_bx, old_by
        cost_state.incremental_update({idx_a, idx_b})

        # Try swap with rotation exchange
        c_a.x, c_b.x = c_b.x, c_a.x
        c_a.y, c_b.y = c_b.y, c_a.y
        c_a.set_rotation(old_brot)
        c_b.set_rotation(old_arot)
        new_cost = cost_state.incremental_update({idx_a, idx_b})

        if cost_state.overlap_count > prev_overlap_count:
            # Both overlap — revert fully
            c_a.x, c_a.y, c_a.rotation = old_ax, old_ay, old_arot
            c_b.x, c_b.y, c_b.rotation = old_bx, old_by, old_brot
            cost_state.incremental_update({idx_a, idx_b})
            return False

        if new_cost < base_cost - improve_threshold:
            return True
        else:
            c_a.x, c_a.y, c_a.rotation = old_ax, old_ay, old_arot
            c_b.x, c_b.y, c_b.rotation = old_bx, old_by, old_brot
            cost_state.incremental_update({idx_a, idx_b})
            return False

    # Position swap didn't increase overlaps — check cost
    if new_cost < base_cost - improve_threshold:
        # Position swap is good — try rotation exchange to see if even better
        old_arot_now = c_a.rotation
        old_brot_now = c_b.rotation
        c_a.set_rotation(old_brot_now)
        c_b.set_rotation(old_arot_now)
        rot_cost = cost_state.incremental_update({idx_a, idx_b})
        if rot_cost < new_cost and cost_state.overlap_count <= prev_overlap_count:
            # Rotation exchange is even better — keep it
            return True
        else:
            # Revert rotation, keep position swap
            c_a.set_rotation(old_arot_now)
            c_b.set_rotation(old_brot_now)
            cost_state.incremental_update({idx_a, idx_b})
            return True
    else:
        # Position swap not good enough — try with rotation exchange
        c_a.x, c_a.y = old_ax, old_ay
        c_b.x, c_b.y = old_bx, old_by
        cost_state.incremental_update({idx_a, idx_b})

        c_a.x, c_b.x = c_b.x, c_a.x
        c_a.y, c_b.y = c_b.y, c_a.y
        c_a.set_rotation(old_brot)
        c_b.set_rotation(old_arot)
        new_cost = cost_state.incremental_update({idx_a, idx_b})

        if cost_state.overlap_count <= prev_overlap_count and new_cost < base_cost - improve_threshold:
            return True
        else:
            c_a.x, c_a.y, c_a.rotation = old_ax, old_ay, old_arot
            c_b.x, c_b.y, c_b.rotation = old_bx, old_by, old_brot
            cost_state.incremental_update({idx_a, idx_b})
            return False


def _greedy_swap_refine(
    model: BoardModel,
    cost_state: CostState,
    moveable_indices: list[int],
    config: SAConfig,
) -> None:
    """Greedy swap refinement: try swapping pairs of components, accept if improving.

    Smart swap: Uses candidate filtering instead of brute-force O(n^2).
    Candidates are built from:
    - Net-connected components (sharing signal nets)
    - Size-class similar components (area within 3x)
    - Top HPWL contributors

    For small boards (<60 components), falls back to all-pairs.

    Hard overlap rejection: swaps that increase overlap count are rejected.
    Also tries rotation swaps (swap positions + exchange rotations) which
    the SA's do_swap() never does.
    """
    cost_state.update_penalty_scale(1.0)
    n = len(moveable_indices)

    if n < 2:
        return

    # Build candidate swap partners (smart filtering for large boards)
    candidates = _build_swap_candidates(model, cost_state, moveable_indices)

    if config.verbose:
        total_pairs = sum(len(v) for v in candidates.values()) // 2
        max_brute = n * (n - 1) // 2
        print(f"    Swap candidates: {total_pairs} pairs (vs {max_brute} brute-force)")

    best_swap_cost = cost_state.normalized_cost
    best_positions = _save_positions(model, moveable_indices)
    total_swaps = 0

    for pass_num in range(3):  # max 3 swap passes
        any_swap_accepted = False

        for idx_a in moveable_indices:
            for idx_b in candidates.get(idx_a, []):
                if idx_b <= idx_a:
                    continue  # avoid duplicate pairs (only process a < b)

                if _try_single_swap(model, cost_state, idx_a, idx_b, improve_threshold=0.5):
                    any_swap_accepted = True
                    total_swaps += 1

        # After a full pass, check if we improved
        cur_cost = cost_state.normalized_cost
        if cur_cost < best_swap_cost:
            best_swap_cost = cur_cost
            best_positions = _save_positions(model, moveable_indices)
        else:
            _restore_positions(model, best_positions)
            cost_state._compute_all()
            break  # no improvement, stop swapping

        if config.verbose:
            print(f"    Swap pass {pass_num + 1}: cost={cost_state.normalized_cost:.1f} "
                  f"hpwl={cost_state.hpwl:.1f} overlaps={cost_state.overlap_count} "
                  f"(swaps accepted: {total_swaps})")

        if not any_swap_accepted:
            break

    # Final restore to best result
    if cost_state.normalized_cost > best_swap_cost:
        _restore_positions(model, best_positions)
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
    """Run SA optimization. Returns final normalized cost. Mutates model in-place.

    When config.skip_sa is True, runs enhanced greedy + swap + greedy
    instead of the full SA pass. This is the new default — global SA is
    opt-in via --sa flag.
    """
    if config is None:
        config = SAConfig()

    if not moveable_indices:
        return cost_state.normalized_cost

    # --- Greedy-only path (default) ---
    if config.skip_sa:
        if config.verbose:
            print(f"  Enhanced greedy optimization ({len(moveable_indices)} moveable components)...")

        # Phase 1: Enhanced greedy nudge+rotate
        if config.verbose:
            print("  Phase 1: Enhanced greedy nudge+rotate...")
        _greedy_refine(model, cost_state, moveable_indices, config, enhanced=True)

        # Phase 2: Greedy swap
        if config.verbose:
            print("  Phase 2: Greedy swap refinement...")
        _greedy_swap_refine(model, cost_state, moveable_indices, config)

        # Phase 3: Another pass of enhanced greedy (to settle after swaps)
        if config.verbose:
            print("  Phase 3: Final greedy nudge+rotate...")
        _greedy_refine(model, cost_state, moveable_indices, config, enhanced=True)

        if cost_state.overlap_count > 0 and config.verbose:
            print(f"  Greedy leaving {cost_state.overlap_count} overlaps for legalizer to resolve")

        final_cost = cost_state.normalized_cost
        if config.verbose:
            print(f"  Greedy complete: cost={final_cost:.1f} hpwl={cost_state.hpwl:.1f} overlaps={cost_state.overlap_count}")
        return final_cost

    # --- Full SA path (opt-in with --sa) ---
    # Record initial overlap count for hard cap
    initial_overlap_count = cost_state.overlap_count

    density = _compute_density(model)
    config.penalty_scale_min = max(config.penalty_scale_min, _density_adaptive_penalty_scale(density))
    config.reheat_count = min(config.reheat_count, _density_adaptive_reheat_count(density))

    # Calibrate T0
    t0 = _calibrate_t0(model, cost_state, moveable_indices, config)
    if config.verbose:
        print(f"  SA: T0={t0:.4f}, {len(moveable_indices)} moveable components, "
              f"density={density:.3f}, penalty_scale_min={config.penalty_scale_min:.2f}")

    # Compute initial cost at penalty_scale=1.0 for fair comparison.
    # The old code compared initial_cost (at penalty_scale_min, e.g. 0.80)
    # with final_cost (at penalty_scale=1.0 after greedy), causing false reverts.
    # Example: initial=1470 @scale=0.80 vs final=1681 @scale=1.0 → ratio=1.14 > 1.10
    # But at scale=1.0, initial would be ~1840 → ratio=0.91 (improvement!).
    cost_state.update_penalty_scale(1.0)
    initial_cost_at_full_scale = cost_state.normalized_cost
    initial_hpwl_for_revert = cost_state.hpwl
    cost_state.update_penalty_scale(config.penalty_scale_min)

    # Store initial cost at full scale so run_sa can use it for revert logic
    simulate_annealing._initial_cost_full_scale = initial_cost_at_full_scale

    # Main SA pass — capture global best from each pass
    _, global_best_cost, global_best_positions, global_best_overlap_count = \
        _run_sa_pass(model, cost_state, moveable_indices, t0, config,
                     initial_overlap_count=initial_overlap_count)

    # Reheating rounds — pass global best between reheats
    for reheat_i in range(config.reheat_count):
        reheat_t0 = t0 * config.reheat_ratio * (0.7 ** reheat_i)
        if config.verbose:
            print(f"  Reheat {reheat_i + 1}: T0={reheat_t0:.4f}")
        _, global_best_cost, global_best_positions, global_best_overlap_count = \
            _run_sa_pass(
                model, cost_state, moveable_indices,
                reheat_t0, config,
                max_iter_override=config.max_iterations // 3,
                initial_overlap_count=initial_overlap_count,
                global_best_cost=global_best_cost,
                global_best_positions=global_best_positions,
                global_best_overlap_count=global_best_overlap_count,
            )

    # After SA, run enhanced greedy + swap + greedy (not just basic greedy)
    if config.verbose:
        print("  Post-SA enhanced greedy refinement...")
    _greedy_refine(model, cost_state, moveable_indices, config, enhanced=True)
    _greedy_swap_refine(model, cost_state, moveable_indices, config)
    _greedy_refine(model, cost_state, moveable_indices, config, enhanced=True)

    # NO overlap resolution in SA. The legalizer handles overlaps.
    # SA overlap resolution destroys HPWL optimization for dense boards
    # by pushing components far apart to eliminate overlaps, causing
    # massive cost increase that triggers SA revert.
    # The legalizer has better context for resolving overlaps.
    if cost_state.overlap_count > 0 and config.verbose:
        print(f"  SA leaving {cost_state.overlap_count} overlaps for legalizer to resolve")

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

    When config.skip_sa is True, runs enhanced greedy + swap + greedy
    instead of global SA. This is the new default behavior.

    HPWL-based revert policy.
    Consistent penalty_scale comparison prevents false reverts.
    Density-adaptive SA parameters.
    """
    if config is None:
        config = SAConfig(verbose=verbose)
    elif verbose and not config.verbose:
        config = SAConfig(**{**config.__dict__, 'verbose': True})

    # Density-adaptive SA parameters
    density = _compute_density(model)
    adaptive_penalty_min = _density_adaptive_penalty_scale(density)
    adaptive_reheat_count = _density_adaptive_reheat_count(density)

    # Override config with density-adaptive values (only if user didn't explicitly set)
    if config.penalty_scale_min == 0.50:  # default value — user didn't override
        config.penalty_scale_min = adaptive_penalty_min
    if config.reheat_count == 3:  # default value — user didn't override
        config.reheat_count = adaptive_reheat_count
    # Density-aware overlap cap — tighter for dense boards
    if config.overlap_cap_factor == 2.0:  # default value
        if density > 0.35:
            config.overlap_cap_factor = 1.5  # dense: allow 50% increase
        elif density > 0.25:
            config.overlap_cap_factor = 1.75  # medium: allow 75% increase

    # Auto-disable SA on tiny/large boards.  Greedy is near-optimal on
    # tiny boards (≤5 comps) and SA noise can hurt; on large boards (≥50
    # comps) SA is 8-25x slower with diminishing returns.  Only auto-
    # disable if the user hasn't explicitly set skip_sa=True (they asked
    # for no-SA) or skip_sa=False (they explicitly want SA).  When the
    # user passes skip_sa=False, respect that.
    n_movable = sum(1 for c in model.components if not c.is_fixed
                    and getattr(c, 'component_type', '') != 'connector')
    sa_was_auto_disabled = False
    if not config.skip_sa:  # only auto-disable if SA was going to run
        min_n = config.sa_auto_disable_min_components
        max_n = config.sa_auto_disable_max_components
        if min_n > 0 and n_movable <= min_n:
            if config.verbose:
                print(f"  Auto-disabling SA: {n_movable} movable components "
                      f"(≤{min_n} threshold — greedy is near-optimal on tiny boards)")
            config.skip_sa = True
            sa_was_auto_disabled = True
        elif max_n > 0 and n_movable >= max_n:
            if config.verbose:
                print(f"  Auto-disabling SA: {n_movable} movable components "
                      f"(≥{max_n} threshold — SA too slow on large boards, "
                      f"greedy+legalize is the better default)")
            config.skip_sa = True
            sa_was_auto_disabled = True

    if config.verbose:
        mode = "greedy+swap" if config.skip_sa else "SA+greedy+swap"
        print(f"  Mode: {mode}, Density: {density:.3f}, penalty_scale_min: {config.penalty_scale_min:.2f}, "
              f"reheat_count: {config.reheat_count}, overlap_cap: {config.overlap_cap_factor:.1f}x")

    cost_state = CostState(model, rules=rules)
    moveable_indices = get_moveable_indices(model)
    cost_state.update_penalty_scale(config.penalty_scale_min)

    initial_cost = cost_state.normalized_cost
    initial_hpwl = cost_state.hpwl
    initial_overlaps = cost_state.overlap_count

    # Save pre-SA positions so we can restore if SA makes things worse
    pre_sa_positions = _save_positions(model, moveable_indices)

    final_cost = simulate_annealing(model, cost_state, moveable_indices, config)

    # HPWL-based revert logic
    final_overlaps = cost_state.overlap_count
    final_hpwl = cost_state.hpwl
    overlap_improvement = initial_overlaps - final_overlaps
    hpwl_ratio = final_hpwl / initial_hpwl if initial_hpwl > 0 else float('inf')

    should_revert = False
    revert_reason = ""

    if final_overlaps == 0:
        should_revert = False
        if config.verbose and final_cost > initial_cost:
            print(f"  Optimization kept: 0 overlaps achieved "
                  f"(cost {initial_cost:.1f} -> {final_cost:.1f}, "
                  f"HPWL {initial_hpwl:.1f} -> {final_hpwl:.1f})")
    elif hpwl_ratio > 2.0:
        should_revert = True
        revert_reason = f"HPWL doubled ({initial_hpwl:.1f} -> {final_hpwl:.1f})"
    elif hpwl_ratio > 1.5 and overlap_improvement <= 0:
        should_revert = True
        revert_reason = f"HPWL increased {hpwl_ratio:.0%} with no overlap improvement"
    elif overlap_improvement > 0:
        if config.verbose and final_cost > initial_cost:
            print(f"  Optimization kept: overlaps improved ({initial_overlaps} -> {final_overlaps}) "
                  f"despite cost increase ({initial_cost:.1f} -> {final_cost:.1f})")
    elif final_cost > initial_cost and config.verbose:
        print(f"  Optimization kept: cost slightly worse ({initial_cost:.1f} -> {final_cost:.1f})")

    if should_revert:
        if config.verbose:
            print(f"  Optimization reverted: {revert_reason}")
        _restore_positions(model, pre_sa_positions)
        cost_state._compute_all()
        final_cost = cost_state.normalized_cost

    # Spread-floor revert: if the final placement is bunched into a
    # corner/edge (std_x or std_y below spread_floor_fraction of board
    # dimensions), revert to pre-SA positions.  This catches the case
    # where SA+greedy found a low-HPWL but visually terrible placement
    # (everything crammed in one corner).  The per-move spread floor in
    # _run_sa_pass only guards SA moves, not the greedy refinement that
    # runs afterward — so we need this final check.
    if not should_revert and config.spread_floor_fraction > 0 and not sa_was_auto_disabled:
        if _violates_spread_floor(model, model.board, config.spread_floor_fraction):
            if config.verbose:
                print(f"  Optimization reverted: spread floor violated "
                      f"(components bunched into corner/edge)")
            _restore_positions(model, pre_sa_positions)
            cost_state._compute_all()
            final_cost = cost_state.normalized_cost
            should_revert = True  # skip centroid check if we already reverted

    # Centroid-offset revert: if the final centroid is far from board
    # center (more than 25% of the board's min dimension off-center),
    # revert.  This catches the case where SA pushed the entire component
    # cluster off-board (e.g. thermal_separation pushing hot parts to
    # opposite corners, with the cluster ending up outside the board
    # outline).  The spread floor doesn't catch this because the
    # components ARE spread — just in the wrong location.
    if not should_revert and not sa_was_auto_disabled:
        centroid_off = _centroid_offset(model, model.board)
        min_board_dim = min(model.board.width, model.board.height)
        centroid_threshold = 0.25 * min_board_dim  # 25% of min dimension
        if centroid_off > centroid_threshold:
            if config.verbose:
                print(f"  Optimization reverted: centroid {centroid_off:.1f}mm off-center "
                      f"(threshold {centroid_threshold:.1f}mm — components pushed off-board)")
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
