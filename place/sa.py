"""Macro-aware Simulated Annealing.

The atomic move unit is a Macro (leader + rigid followers). All moves
— translate, rotate, swap — operate on the macro as a single rigid
body. No independent follower clamping; moves that would push any
member out of bounds are REJECTED.

Cost function is from ``cost.cost.evaluate``: HPWL (with power rails
included) + overlap + boundary. No penalty scaling, no spread floor,
no fill-first move operator, no RUDY, no Metropolis-on-density.

This is intentionally simpler than the existing engine/annealer.py.
The complexity lived there because it was papering over the absence
of true macros.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from cost.cost import evaluate
from cost.incremental import IncrementalCostTracker

if TYPE_CHECKING:
    from models.board_model import BoardModel
    from models.macro import Macro


def _snapshot_positions(model: "BoardModel") -> dict[str, tuple[float, float, float]]:
    return {c.ref: (c.x, c.y, c.rotation) for c in model.components}


def _restore_positions(model: "BoardModel", snap: dict[str, tuple[float, float, float]]) -> None:
    for c in model.components:
        if c.ref in snap:
            x, y, r = snap[c.ref]
            c.x = x
            c.y = y
            c.set_rotation(r)


def _macro_snapshot(macros: list["Macro"]) -> list[list[tuple[float, float, float]]]:
    return [[(c.x, c.y, c.rotation) for c in m.members] for m in macros]


def _macro_restore(macros: list["Macro"], snap: list[list[tuple[float, float, float]]]) -> None:
    for m, members in zip(macros, snap):
        for c, (x, y, r) in zip(m.members, members):
            c.x = x
            c.y = y
            c.set_rotation(r)


def _calibrate_initial_temp(
    model: "BoardModel",
    macros: list["Macro"],
    bounds: tuple[float, float, float, float],
    n_samples: int = 100,
    window_mm: float = 5.0,
    *,
    alpha: float,
    beta: float,
    gamma: float,
    net_weights: dict[str, float] | None = None,
    rng: random.Random | None = None,
) -> float:
    """Estimate an initial temperature by sampling random moves.

    Returns T0 such that a move with average positive delta has ~85%
    acceptance probability: T0 = -avg_delta / ln(0.85).

    Crucially, samples that *create or increase* macro overlap are
    excluded from the average. The overlap penalty (beta=25) dominates
    HPWL by ~3 orders of magnitude; if we include those samples, T0
    ends up in the thousands and SA spends the hot phase doing nothing
    useful — every accepted move is dominated by overlap noise. By
    restricting the average to overlap-clean moves, T0 reflects the
    HPWL gradient SA actually wants to follow.
    """
    if not macros or n_samples <= 0:
        return 1.0
    if rng is None:
        rng = random

    positive_deltas: list[float] = []
    base = evaluate(model, macros, alpha=alpha, beta=beta, gamma=gamma,
                    net_weights=net_weights)

    for _ in range(n_samples):
        m = rng.choice(macros)
        dx = rng.uniform(-window_mm, window_mm)
        dy = rng.uniform(-window_mm, window_mm)
        snap = m._snapshot()
        if not m.translate(dx, dy, bounds=bounds):
            continue
        new_cost = evaluate(model, macros, alpha=alpha, beta=beta, gamma=gamma,
                            net_weights=net_weights)
        delta_overlap = new_cost["overlap"] - base["overlap"]
        m._restore(snap)
        # Skip overlap-penalty samples: they're noise for HPWL calibration.
        if delta_overlap > 1e-9:
            continue
        delta = new_cost["total"] - base["total"]
        if delta > 0:
            positive_deltas.append(delta)

    if not positive_deltas:
        return 1.0
    avg_delta = sum(positive_deltas) / len(positive_deltas)
    if avg_delta < 1e-9:
        return 1.0
    return -avg_delta / math.log(0.85)


def run_macro_sa(
    model: "BoardModel",
    macros: list["Macro"],
    bounds: tuple[float, float, float, float],
    *,
    iterations: int = 2000,
    reheats: int = 2,
    reheat_ratio: float = 0.4,
    alpha: float = 1.0,
    beta: float = 25.0,
    gamma: float = 8.0,
    initial_window_mm: float = 10.0,
    final_window_mm: float = 0.5,
    rotate_prob: float = 0.15,
    swap_prob: float = 0.10,
    displace_prob: float = 0.15,
    seed: int = 42,
    verbose: bool = False,
    net_weights: dict[str, float] | None = None,
    rudy_weight: float = 0.0,
    rudy_recompute_every: int = 50,
    bias_overlapping: bool = False,
    bias_overlap_prob: float = 0.6,
) -> dict[str, float]:
    """Run macro-aware simulated annealing.

    Returns a dict with initial/final cost breakdown.

    The macro is rigid throughout SA. Cap-leader distance is fixed at
    construction time, so the <8mm hard rule is automatically enforced.

    Move operators:
      - ``translate``: rigid (dx, dy) on one macro.
      - ``rotate``: 90/180/270 of one macro around its leader.
      - ``swap``: exchange leader positions of two macros.
      - ``displace``: pick a macro, find a neighbor it overlaps, and
        try to push the neighbor to a clear slot adjacent to the
        picked macro. Helps SA escape jammed configurations where
        pure translate can't make room.

    Finding 7 fix: ``rudy_weight`` adds RUDY congestion to the cost
    function (default 0 = disabled, matching previous behavior). When
    > 0, the RUDY penalty is recomputed every ``rudy_recompute_every``
    steps and passed to ``evaluate`` as a pre-computed penalty (avoids
    re-computing the RUDY map on every cost evaluation — that would
    dominate SA runtime). Boards with routing choke points get
    gradient signal to spread macros away from congested cells.

    ``bias_overlapping`` (default False — opt-in, doesn't affect the
    main placement pipeline unless explicitly requested): with
    probability ``bias_overlap_prob`` each iteration, pick the macro to
    move from a currently-overlapping pair (via the incremental cost
    tracker's live overlap-pair cache) instead of uniformly at random.
    On a mostly-legal board with a handful of overlapping macros,
    uniform selection spends most iterations moving macros that were
    never part of the problem; this concentrates search on the ones
    actually in conflict. Falls back to uniform selection whenever
    there's nothing currently overlapping or the tracker isn't active.
    """
    if not macros:
        return {"initial_total": 0.0, "final_total": 0.0}

    rng = random.Random(seed)

    # Finding 7: RUDY penalty cache. Recomputed every rudy_recompute_every
    # steps. rudy_weight=0 (default) disables RUDY in SA — matching
    # previous behavior — but the verbose report in place/pipeline.py
    # still shows RUDY so the user can see congestion.
    rudy_penalty_cache: float | None = None
    if rudy_weight > 0:
        try:
            from engine.congestion import rudy_congestion_penalty
            rudy_penalty_cache, _peak, _avg, _overflow = rudy_congestion_penalty(model)
        except Exception:
            rudy_penalty_cache = 0.0
        if verbose:
            print(f"  SA: RUDY enabled (weight={rudy_weight}, initial penalty={rudy_penalty_cache:.3f})")

    initial = evaluate(model, macros, alpha=alpha, beta=beta, gamma=gamma,
                       net_weights=net_weights,
                       rudy_weight=rudy_weight, rudy_penalty=rudy_penalty_cache)
    initial_total = initial["total"]
    best_total = initial_total
    best_snapshot = _snapshot_positions(model)

    T0 = _calibrate_initial_temp(
        model, macros, bounds, n_samples=80, window_mm=initial_window_mm,
        alpha=alpha, beta=beta, gamma=gamma, net_weights=net_weights, rng=rng,
    )
    T0 = max(T0, 1.0)
    T_min = max(T0 * 1e-4, 1e-6)

    if verbose:
        print(f"  SA: T0={T0:.3f}, T_min={T_min:.6f}, iterations={iterations}, reheats={reheats}")

    current_total = initial_total

    # _calibrate_initial_temp always reverts every sample move, so the
    # model/macros are back in their pre-calibration state here — safe
    # to build the incremental tracker's cache from current positions.
    #
    # rudy_weight's contribution is intentionally NOT part of the
    # tracker (it's recomputed wholesale every rudy_recompute_every
    # steps anyway, see below) — it's added on top of the tracker's
    # hpwl/overlap/boundary total for the accept/reject decision, same
    # as the non-incremental cost.evaluate() does.
    tracker = IncrementalCostTracker(
        model, macros, alpha=alpha, beta=beta, gamma=gamma, net_weights=net_weights,
    )
    # Sanity: the tracker's from-cache total must agree with the
    # from-scratch evaluate() above (excluding the rudy term, which
    # the tracker doesn't carry). Cheap to check once; catches macro/
    # model mismatches (e.g. a macro whose members aren't all in
    # model.components) immediately instead of silently drifting.
    tracker_initial = tracker.total()
    if abs(tracker_initial["total"] - (initial_total - rudy_weight * (rudy_penalty_cache or 0.0))) > 1e-3:
        # Fall back to full recompute every iteration rather than trust
        # a tracker that disagrees with ground truth — correctness over
        # speed if the two ever diverge (e.g. a future macro field the
        # tracker doesn't know about).
        tracker = None
        if verbose:
            print("  SA: incremental cost tracker disagreed with evaluate() at init — "
                  "falling back to full recompute per iteration.")

    # macro identity -> index, used to look up the index of a macro
    # object handed back by _find_overlapping_neighbor / swap partner
    # selection without an O(n) linear scan every time.
    macro_index = {id(mac): i for i, mac in enumerate(macros)}

    for reheat_round in range(reheats + 1):
        T_start = T0 * (reheat_ratio ** reheat_round)
        T_start = max(T_start, T_min)
        cooling = (T_min / T_start) ** (1.0 / max(1, iterations))
        T = T_start

        for it in range(iterations):
            m_idx = None
            if bias_overlapping and tracker is not None and rng.random() < bias_overlap_prob:
                pairs = tracker.overlapping_pairs()
                if pairs:
                    pair = pairs[rng.randrange(len(pairs))]
                    m_idx = pair[rng.randrange(2)]
            if m_idx is None:
                m_idx = rng.randrange(len(macros))
            m = macros[m_idx]

            # Pick move type. Probabilities are cumulative thresholds
            # checked in order: rotate → displace → swap → translate.
            r = rng.random()
            if r < rotate_prob:
                move = "rotate"
            elif r < rotate_prob + displace_prob and len(macros) >= 2:
                move = "displace"
            elif r < rotate_prob + displace_prob + swap_prob and len(macros) >= 2:
                move = "swap"
            else:
                move = "translate"

            # Window shrinks with temperature
            t_ratio = (T - T_min) / max(T_start - T_min, 1e-9)
            window = final_window_mm + (initial_window_mm - final_window_mm) * t_ratio

            snap = m._snapshot()
            swap_partner = None
            swap_partner_snap = None
            displaced = None
            displaced_snap = None
            # Macro indices actually moved by this iteration — feeds the
            # incremental tracker so it only recomputes what changed.
            touched_indices = [m_idx]

            if move == "translate":
                dx = rng.uniform(-window, window)
                dy = rng.uniform(-window, window)
                ok = m.translate(dx, dy, bounds=bounds)
                if not ok:
                    continue
            elif move == "rotate":
                rot = rng.choice([90.0, 180.0, 270.0])
                new_rot = (m.leader.rotation + rot) % 360.0
                ok = m.set_pose(m.leader.x, m.leader.y, new_rot, bounds=bounds)
                if not ok:
                    continue
            elif move == "displace":
                # Try to find an overlapping neighbor and shove it sideways.
                neighbor = _find_overlapping_neighbor(m, macros, rng)
                if neighbor is None:
                    # No overlap to resolve — fall through to a translate
                    # so we don't waste this iteration.
                    dx = rng.uniform(-window, window)
                    dy = rng.uniform(-window, window)
                    ok = m.translate(dx, dy, bounds=bounds)
                    if not ok:
                        continue
                    move = "translate"
                else:
                    displaced = neighbor
                    displaced_snap = neighbor._snapshot()
                    ok = _try_displace_neighbor(m, neighbor, bounds, window, rng)
                    if not ok:
                        # Displace failed (no clear slot found) — try a
                        # plain translate instead so the iteration still
                        # does something.
                        dx = rng.uniform(-window, window)
                        dy = rng.uniform(-window, window)
                        ok = m.translate(dx, dy, bounds=bounds)
                        if not ok:
                            continue
                        move = "translate"
                    else:
                        # Only the neighbor actually moved — m stayed put.
                        touched_indices = [macro_index[id(neighbor)]]
            else:  # swap
                other_idx = rng.randrange(len(macros))
                if other_idx == m_idx:
                    continue
                other = macros[other_idx]
                swap_partner = other
                swap_partner_snap = other._snapshot()
                # Swap leader positions; each keeps own rotation and offsets
                ox, oy = other.leader.x, other.leader.y
                mx, my = m.leader.x, m.leader.y
                ok1 = m.set_pose(ox, oy, m.leader.rotation, bounds=bounds)
                ok2 = other.set_pose(mx, my, other.leader.rotation, bounds=bounds)
                if not (ok1 and ok2):
                    m._restore(snap)
                    other._restore(swap_partner_snap)
                    continue
                touched_indices = [m_idx, other_idx]

            if tracker is not None:
                proposed = tracker.propose(touched_indices)
                new_total = proposed["total"] + rudy_weight * (rudy_penalty_cache or 0.0)
            else:
                new_cost = evaluate(model, macros, alpha=alpha, beta=beta, gamma=gamma,
                                    net_weights=net_weights,
                                    rudy_weight=rudy_weight, rudy_penalty=rudy_penalty_cache)
                new_total = new_cost["total"]
            delta = new_total - current_total

            if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
                # Accept
                current_total = new_total
                if tracker is not None:
                    tracker.commit()
                if new_total < best_total:
                    best_total = new_total
                    best_snapshot = _snapshot_positions(model)
            else:
                # Reject — revert all touched macros.
                if tracker is not None:
                    tracker.discard()
                m._restore(snap)
                if move == "swap" and swap_partner is not None:
                    swap_partner._restore(swap_partner_snap)
                elif move == "displace" and displaced is not None:
                    displaced._restore(displaced_snap)

            T *= cooling

            # Finding 7: periodically recompute the RUDY penalty so SA's
            # cost reflects the current routing congestion (not a stale
            # snapshot from the start of the reheat round). The recompute
            # is O(nets × cells) — doing it every step would dominate
            # SA runtime, so we batch it every rudy_recompute_every steps.
            if rudy_weight > 0 and (it + 1) % rudy_recompute_every == 0:
                try:
                    from engine.congestion import rudy_congestion_penalty
                    rudy_penalty_cache, _peak, _avg, _overflow = rudy_congestion_penalty(model)
                except Exception:
                    pass

        if verbose:
            print(f"  SA reheat {reheat_round}: T={T:.4f}, current={current_total:.2f}, best={best_total:.2f}")

    # Restore best state found
    _restore_positions(model, best_snapshot)
    # Refresh macro follower positions to match leader
    for m in macros:
        m.apply_offsets()

    final = evaluate(model, macros, alpha=alpha, beta=beta, gamma=gamma,
                     net_weights=net_weights,
                     rudy_weight=rudy_weight, rudy_penalty=rudy_penalty_cache)
    if verbose:
        print(
            f"  SA done: initial={initial_total:.2f}, final={final['total']:.2f} "
            f"(best={best_total:.2f})"
        )

    return {
        "initial_total": initial_total,
        "final_total": final["total"],
        "initial_hpwl": initial["hpwl"],
        "final_hpwl": final["hpwl"],
        "initial_overlap": initial["overlap"],
        "final_overlap": final["overlap"],
        "best_total": best_total,
    }


def _find_overlapping_neighbor(
    m: "Macro",
    macros: list["Macro"],
    rng: random.Random,
) -> "Macro | None":
    """Pick a random macro whose bbox overlaps ``m``'s bbox. None if no overlap."""
    candidates = [other for other in macros if other is not m and m.overlaps(other)]
    if not candidates:
        return None
    return rng.choice(candidates)


def _try_displace_neighbor(
    m: "Macro",
    neighbor: "Macro",
    bounds: tuple[float, float, float, float],
    window: float,
    rng: random.Random,
) -> bool:
    """Push ``neighbor`` out of ``m`` along the cheaper axis.

    Tries the four cardinal directions, picks the first that lands
    ``neighbor`` inside bounds. Distance pushed is the current overlap
    depth plus a small jitter drawn from ``window`` so the move
    explores, not just barely resolves.
    """
    mx1, my1, mx2, my2 = m.bbox
    nx1, ny1, nx2, ny2 = neighbor.bbox
    ox = min(mx2, nx2) - max(mx1, nx1)
    oy = min(my2, ny2) - max(my1, ny1)
    if ox <= 0 and oy <= 0:
        return False

    # Try cheaper axis first (less overlap depth to clear).
    if ox <= oy:
        primary = ("x", ox)
        secondary = ("y", oy)
    else:
        primary = ("y", oy)
        secondary = ("x", ox)

    snap = neighbor._snapshot()
    for axis, depth in (primary, secondary):
        # Direction: push neighbor away from m along this axis.
        if axis == "x":
            sign = -1.0 if (nx1 + nx2) / 2 < (mx1 + mx2) / 2 else 1.0
            jitter = rng.uniform(0.0, max(window - depth, 0.0))
            ok = neighbor.translate(sign * (depth + 0.5 + jitter), 0.0, bounds=bounds)
        else:
            sign = -1.0 if (ny1 + ny2) / 2 < (my1 + my2) / 2 else 1.0
            jitter = rng.uniform(0.0, max(window - depth, 0.0))
            ok = neighbor.translate(0.0, sign * (depth + 0.5 + jitter), bounds=bounds)
        if ok:
            return True
        neighbor._restore(snap)
    return False
