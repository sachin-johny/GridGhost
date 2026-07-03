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

from models.board_model import BoardModel, Component, Net
from profiles.board_profiles import ConstraintRule
from engine.constraint_evaluator import evaluate_constraint_penalties
from engine.cost_state import _is_power_net


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
        # Skip power nets — their HPWL is nearly constant regardless of placement,
        # and SA also skips them, so we need consistent reporting.
        if _is_power_net(net.name):
            continue
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


def overlap_penalty_and_count(model: BoardModel) -> tuple[float, int]:
    """Single O(n^2) pass returning (total_overlap_area, overlap_pair_count).

    Replaces the two separate O(n^2) scans previously done by
    ``total_overlap_penalty`` and ``count_overlaps``.  ``Component.overlaps``
    is essentially the same AABB test as the bounds check inside
    ``overlap_area``, so doing both in one pass halves the work.
    """
    total_area = 0.0
    count = 0
    components = model.components
    n = len(components)
    for i in range(n):
        ci = components[i]
        ax1, ay1, ax2, ay2 = ci.bbox
        for j in range(i + 1, n):
            cj = components[j]
            bx1, by1, bx2, by2 = cj.bbox
            # Quick AABB reject — avoid the function-call overhead of
            # overlap_area when there is no overlap at all.
            ox1 = ax1 if ax1 > bx1 else bx1
            oy1 = ay1 if ay1 > by1 else by1
            ox2 = ax2 if ax2 < bx2 else bx2
            oy2 = ay2 if ay2 < by2 else by2
            if ox2 <= ox1 or oy2 <= oy1:
                continue
            count += 1
            total_area += (ox2 - ox1) * (oy2 - oy1)
    return total_area, count


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

        # Linear penalty (proportional to distance outside)
        overflow = left_overflow + right_overflow + top_overflow + bottom_overflow
        total += overflow

    return total


def count_out_of_bounds(model: BoardModel) -> int:
    """Count components whose bounding box extends outside the board.

    Edge connectors are excluded: they are intentionally placed with pads
    on the board edge and body overhanging.  Their pads are inside the
    board boundary, so they are not truly out-of-bounds.
    """
    count = 0
    board = model.board
    for comp in model.components:
        if comp.is_edge_connector:
            continue
        x_min, y_min, x_max, y_max = comp.bbox
        if (x_min < board.x_min or x_max > board.x_max or
                y_min < board.y_min or y_max > board.y_max):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Density Penalty (ePlace-style Gini coefficient on a 10x10 grid)
# ---------------------------------------------------------------------------

def _gini_density_penalty(model: BoardModel, rules: list | None = None) -> float:
    """Gini coefficient of cell-occupancy on a 10x10 grid, scaled by total.

    Mirrors the density penalty in CostState so CostFunction.evaluate and
    CostState.normalized_cost agree. Returns 0 when components are uniformly
    distributed; increases as they cluster into fewer cells.

    Decoupling caps assigned to an IC are absorbed into the IC's cell
    (counted once at the IC's position) so density doesn't fight the
    decoupling_proximity constraint — same grouping logic as CostState.
    """
    from engine.cost_state import DENSITY_GRID
    from engine.constraint_evaluator import _build_decoupling_map

    board = model.board
    cell_w = (board.x_max - board.x_min) / DENSITY_GRID
    cell_h = (board.y_max - board.y_min) / DENSITY_GRID
    if cell_w <= 0 or cell_h <= 0:
        return 0.0

    # Build cap-IC grouping (only when decoupling rule is active)
    grouped_cap_refs: set[str] = set()
    if rules:
        decap_map = _build_decoupling_map(model)
        if decap_map:
            for cap_refs in decap_map.values():
                grouped_cap_refs.update(cap_refs)

    grid = [0] * (DENSITY_GRID * DENSITY_GRID)
    total = 0
    bx_min = board.x_min
    by_min = board.y_min
    for c in model.components:
        if c.is_fixed or c.is_edge_connector:
            continue
        if c.ref in grouped_cap_refs:
            continue  # absorbed into IC's cell
        gx = int((c.x - bx_min) / cell_w)
        gy = int((c.y - by_min) / cell_h)
        gx = max(0, min(DENSITY_GRID - 1, gx))
        gy = max(0, min(DENSITY_GRID - 1, gy))
        grid[gx + gy * DENSITY_GRID] += 1
        total += 1

    if total == 0:
        return 0.0
    n = DENSITY_GRID * DENSITY_GRID
    s = sorted(grid)
    cum = sum((i + 1) * v for i, v in enumerate(s))
    gini = (2 * cum) / (n * total) - (n + 1) / n
    return gini * total


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
        beta: float = 25.0,    # Overlap penalty weight (aligned more closely with SA)
        gamma: float = 4.0,    # Boundary penalty weight (aligned more closely with SA)
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
        # Single O(n^2) pass for both overlap area and overlap count.
        overlap, overlap_count = overlap_penalty_and_count(model)
        boundary = total_boundary_penalty(model)

        # Constraint penalties — fully implemented via constraint_evaluator
        constraint, constraint_breakdown = evaluate_constraint_penalties(
            model, self.rules
        )

        # Density penalty (ePlace-style Gini on a 10x10 grid). Active when
        # the decoupling_proximity rule is enabled (matches CostState gating).
        # Reported here so the post-placement cost breakdown reflects the
        # same spreading pressure the optimizer felt during SA/greedy.
        density = 0.0
        if self.rules and any(r.name == 'decoupling_proximity' and r.enabled for r in self.rules):
            density = _gini_density_penalty(model, self.rules)

        total = (
            self.alpha * hpwl +
            self.beta * overlap +
            self.gamma * boundary +
            self.delta * constraint
        )
        # Add density at the adaptive weight CostState uses, so evaluate()
        # and CostState.normalized_cost agree on the density contribution.
        from engine.cost_state import density_adaptive_weight
        board_area = model.board.width * model.board.height
        if board_area > 0:
            comp_area = sum(c.effective_width * c.effective_height for c in model.components)
            n_moveable = sum(1 for c in model.components if not c.is_fixed and not c.is_edge_connector)
            dw = density_adaptive_weight(comp_area / board_area, n_moveable)
        else:
            from engine.cost_state import DENSITY_WEIGHT
            dw = DENSITY_WEIGHT
        total += dw * density

        result = {
            "total": total,
            "hpwl": hpwl,
            "overlap": overlap,
            "boundary": boundary,
            "constraint": constraint,
            "density": density,
            "overlap_count": overlap_count,
            "oob_count": count_out_of_bounds(model),
        }
        # Merge per-rule breakdown so display code can show details
        for rule_name, rule_penalty in constraint_breakdown.items():
            result[f"constraint_{rule_name}"] = rule_penalty

        return result

    def cost(self, model: BoardModel) -> float:
        """Return the scalar total cost."""
        return self.evaluate(model)["total"]
