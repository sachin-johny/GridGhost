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
) -> float:
    """Estimate an initial temperature by sampling random moves.

    Returns T0 such that a move with average positive delta has ~85%
    acceptance probability: T0 = -avg_delta / ln(0.85).
    """
    if not macros or n_samples <= 0:
        return 1.0

    positive_deltas: list[float] = []
    base = evaluate(model, macros, alpha=alpha, beta=beta, gamma=gamma)["total"]

    for _ in range(n_samples):
        m = random.choice(macros)
        dx = random.uniform(-window_mm, window_mm)
        dy = random.uniform(-window_mm, window_mm)
        snap = m._snapshot()
        if not m.translate(dx, dy, bounds=bounds):
            continue
        new_cost = evaluate(model, macros, alpha=alpha, beta=beta, gamma=gamma)["total"]
        delta = new_cost - base
        m._restore(snap)
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
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, float]:
    """Run macro-aware simulated annealing.

    Returns a dict with initial/final cost breakdown.

    The macro is rigid throughout SA. Cap-leader distance is fixed at
    construction time, so the <8mm hard rule is automatically enforced.
    """
    if not macros:
        return {"initial_total": 0.0, "final_total": 0.0}

    rng = random.Random(seed)
    # Patch the global random too — helper functions in cost/macro use it
    # only via choice/uniform here, so the local rng is sufficient as long
    # as we route all randomness through it.

    initial = evaluate(model, macros, alpha=alpha, beta=beta, gamma=gamma)
    initial_total = initial["total"]
    best_total = initial_total
    best_snapshot = _snapshot_positions(model)

    T0 = _calibrate_initial_temp(
        model, macros, bounds, n_samples=80, window_mm=initial_window_mm,
        alpha=alpha, beta=beta, gamma=gamma,
    )
    T0 = max(T0, 1.0)
    T_min = max(T0 * 1e-4, 1e-6)

    if verbose:
        print(f"  SA: T0={T0:.3f}, T_min={T_min:.6f}, iterations={iterations}, reheats={reheats}")

    current_total = initial_total

    for reheat_round in range(reheats + 1):
        T_start = T0 * (reheat_ratio ** reheat_round)
        T_start = max(T_start, T_min)
        cooling = (T_min / T_start) ** (1.0 / max(1, iterations))
        T = T_start

        for it in range(iterations):
            m_idx = rng.randrange(len(macros))
            m = macros[m_idx]

            # Pick move type
            r = rng.random()
            if r < rotate_prob:
                move = "rotate"
            elif r < rotate_prob + swap_prob and len(macros) >= 2:
                move = "swap"
            else:
                move = "translate"

            # Window shrinks with temperature
            t_ratio = (T - T_min) / max(T_start - T_min, 1e-9)
            window = final_window_mm + (initial_window_mm - final_window_mm) * t_ratio

            snap = m._snapshot()
            swap_partner = None
            swap_partner_snap = None

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

            new_cost = evaluate(model, macros, alpha=alpha, beta=beta, gamma=gamma)
            new_total = new_cost["total"]
            delta = new_total - current_total

            if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
                # Accept
                current_total = new_total
                if new_total < best_total:
                    best_total = new_total
                    best_snapshot = _snapshot_positions(model)
            else:
                # Reject — revert
                if move == "swap" and swap_partner is not None:
                    m._restore(snap)
                    swap_partner._restore(swap_partner_snap)
                else:
                    m._restore(snap)

            T *= cooling

        if verbose:
            print(f"  SA reheat {reheat_round}: T={T:.4f}, current={current_total:.2f}, best={best_total:.2f}")

    # Restore best state found
    _restore_positions(model, best_snapshot)
    # Refresh macro follower positions to match leader
    for m in macros:
        m.apply_offsets()

    final = evaluate(model, macros, alpha=alpha, beta=beta, gamma=gamma)
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
