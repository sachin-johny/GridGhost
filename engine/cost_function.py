"""HPWL (Half-Perimeter Wirelength) cost function and related penalties.

Implements the cost function from the plan:
    Total Cost = alpha·W_HPWL + beta·P_overlap + gamma·P_boundary + delta·Σ(wk·Ck)

For Phase 1, we implement:
- W_HPWL with clique/star net models
- P_overlap (component overlap penalty)
- P_boundary (out-of-bounds penalty)
- δ·Σ(wk·Ck) — constraint rule penalties (decoupling proximity, connector
  edge, thermal grouping, etc.) via constraint_evaluator module.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np

from models.board_model import BoardModel, Component, Net
from profiles.board_profiles import ConstraintRule
from engine.constraint_evaluator import evaluate_constraint_penalties


# ---------------------------------------------------------------------------
# HPWL Wirelength
# ---------------------------------------------------------------------------

def net_wirelength_hpwl(pins: list[tuple[float, float]], model: str = "auto") -> float:
    """Compute HPWL for a single net.

    Args:
        pins: List of (x, y) absolute pin positions
        model: "clique" for pairwise (≤4 pins), "star" for star model (>4 pins),
               "auto" to select automatically

    Returns:
        HPWL value for this net
    """
    if len(pins) < 2:
        return 0.0

    if model == "auto":
        model = "clique" if len(pins) <= 4 else "star"

    if model == "clique":
        # Clique model: sum of pairwise Manhattan distances, normalized
        total = 0.0
        for (x1, y1), (x2, y2) in combinations(pins, 2):
            total += abs(x1 - x2) + abs(y1 - y2)
        return total / (len(pins) - 1)

    if model == "star":
        # Star model: distances to center-of-mass auxiliary point
        center_x = sum(p[0] for p in pins) / len(pins)
        center_y = sum(p[1] for p in pins) / len(pins)
        return sum(abs(p[0] - center_x) + abs(p[1] - center_y) for p in pins)

    return 0.0


def total_hpwl(model: BoardModel) -> float:
    """Compute total HPWL across all nets.

    Uses component positions to compute absolute pad positions,
    then calculates HPWL per net with the appropriate model.
    """
    total = 0.0
    comp_map = {c.ref: c for c in model.components}

    for net in model.nets:
        pins = []
        for ref, pad_name in net.pins:
            comp = comp_map.get(ref)
            if not comp:
                continue
            # Find the pad
            for pad in comp.pads:
                if pad.pad_name == pad_name:
                    abs_x, abs_y = pad.absolute_pos(comp.x, comp.y, comp.rotation)
                    pins.append((abs_x, abs_y))
                    break
            else:
                # Pad not found — use component center as approximation
                pins.append((comp.x, comp.y))

        total += net_wirelength_hpwl(pins)

    return total


# ---------------------------------------------------------------------------
# Overlap Penalty
# ---------------------------------------------------------------------------

def total_overlap_penalty(model: BoardModel) -> float:
    """Compute total overlap area penalty across all component pairs."""
    total = 0.0
    components = model.components
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            area = components[i].overlap_area(components[j])
            total += area
    return total


def count_overlaps(model: BoardModel) -> int:
    """Count the number of overlapping component pairs."""
    count = 0
    components = model.components
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            if components[i].overlaps(components[j]):
                count += 1
    return count


# ---------------------------------------------------------------------------
# Boundary Penalty
# ---------------------------------------------------------------------------

def total_boundary_penalty(model: BoardModel) -> float:
    """Compute penalty for components outside the board outline.

    Penalty is proportional to how far outside the boundary each component is.
    """
    total = 0.0
    board = model.board

    for comp in model.components:
        x_min, y_min, x_max, y_max = comp.bbox

        # How far outside each edge
        left_overflow = max(0.0, board.x_min - x_min)
        right_overflow = max(0.0, x_max - board.x_max)
        top_overflow = max(0.0, board.y_min - y_min)
        bottom_overflow = max(0.0, y_max - board.y_max)

        # Quadratic penalty (proportional to square of distance outside)
        overflow = left_overflow + right_overflow + top_overflow + bottom_overflow
        total += overflow ** 2

    return total


def count_out_of_bounds(model: BoardModel) -> int:
    """Count components whose bounding box extends outside the board."""
    count = 0
    board = model.board
    for comp in model.components:
        x_min, y_min, x_max, y_max = comp.bbox
        if (x_min < board.x_min or x_max > board.x_max or
                y_min < board.y_min or y_max > board.y_max):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Combined Cost Function
# ---------------------------------------------------------------------------

class CostFunction:
    """Weighted cost function for placement evaluation.

    Total Cost = α·W_HPWL + β·P_overlap + γ·P_boundary + δ·Σ(wk·Ck)

    The δ·Σ(wk·Ck) term evaluates each active constraint rule from the
    board profile and sums rule.weight × rule_penalty.  Pass a list of
    ConstraintRule objects via `rules=` to activate this term; otherwise
    it degrades gracefully to 0.0 (backward compatible).
    """

    def __init__(
        self,
        alpha: float = 1.0,    # HPWL weight
        beta: float = 5.0,     # Overlap penalty weight
        gamma: float = 3.0,    # Boundary penalty weight
        delta: float = 4.0,    # Constraint penalty weight
        rules: list[ConstraintRule] | None = None,  # Board profile constraint rules
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.rules = rules or []

    def evaluate(self, model: BoardModel) -> dict[str, float]:
        """Evaluate all cost terms and return a detailed breakdown."""
        hpwl = total_hpwl(model)
        overlap = total_overlap_penalty(model)
        boundary = total_boundary_penalty(model)

        # Constraint penalties — fully implemented via constraint_evaluator
        constraint, constraint_breakdown = evaluate_constraint_penalties(
            model, self.rules
        )

        total = (
            self.alpha * hpwl +
            self.beta * overlap +
            self.gamma * boundary +
            self.delta * constraint
        )

        result = {
            "total": total,
            "hpwl": hpwl,
            "overlap": overlap,
            "boundary": boundary,
            "constraint": constraint,
            "overlap_count": count_overlaps(model),
            "oob_count": count_out_of_bounds(model),
        }
        # Merge per-rule breakdown so display code can show details
        for rule_name, rule_penalty in constraint_breakdown.items():
            result[f"constraint_{rule_name}"] = rule_penalty

        return result

    def cost(self, model: BoardModel) -> float:
        """Return the scalar total cost."""
        return self.evaluate(model)["total"]
