"""Simple local optimizer for Phase 2: greedy pairwise swaps.

This provides a deterministic, easy-to-understand improvement pass
that uses the existing CostFunction to accept swaps that reduce cost.
"""

from __future__ import annotations

from typing import Optional

from models.board_model import BoardModel, Component
from engine.cost_function import CostFunction


def greedy_swap_optimize(model: BoardModel, cost_fn: CostFunction, max_iters: int = 5) -> BoardModel:
    """Greedy pairwise swap optimizer.

    Args:
        model: BoardModel to optimize in-place.
        cost_fn: CostFunction instance to evaluate cost.
        max_iters: Number of full passes over all pairs.

    Returns:
        The (mutated) BoardModel with improved placement when possible.
    """
    movable = [c for c in model.components if not c.is_fixed]
    if len(movable) < 2:
        return model

    best_cost = cost_fn.cost(model)

    for it in range(max_iters):
        improved = False

        # Iterate deterministic order
        for i in range(len(movable)):
            for j in range(i + 1, len(movable)):
                a: Component = movable[i]
                b: Component = movable[j]

                # Save state
                ax, ay, ar = a.x, a.y, a.rotation
                bx, by, br = b.x, b.y, b.rotation

                # Swap positions and rotations
                a.x, a.y, a.rotation, b.x, b.y, b.rotation = bx, by, br, ax, ay, ar

                new_cost = cost_fn.cost(model)

                if new_cost < best_cost:
                    best_cost = new_cost
                    improved = True
                else:
                    # Revert
                    a.x, a.y, a.rotation = ax, ay, ar
                    b.x, b.y, b.rotation = bx, by, br

        if not improved:
            break

    return model


__all__ = ["greedy_swap_optimize"]
