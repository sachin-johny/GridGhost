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

from itertools import combinations

from models.board_model import BoardModel, Component, Net
from profiles.board_profiles import ConstraintRule
from engine.constraint_evaluator import evaluate_constraint_penalties
from engine.cost_state import _is_power_net


# ---------------------------------------------------------------------------
# Net Crossing (routability proxy)
# ---------------------------------------------------------------------------

def _compute_net_crossings(model: BoardModel) -> int:
    """Count pairs of non-power nets whose bounding boxes intersect.

    This is a routability proxy: nets whose bboxes overlap are likely
    to cross each other during routing, causing congestion and vias.
    Inspired by Cypress (ISPD 2025 Best Paper).
    """
    net_bboxes: dict[str, tuple[float, float, float, float]] = {}
    comp_map = {c.ref: c for c in model.components}
    for net in model.nets:
        if _is_power_net(net.name):
            continue
        pins = []
        for ref, pad_name in net.pins:
            comp = comp_map.get(ref)
            if not comp:
                continue
            for pad in comp.pads:
                if pad.pad_name == pad_name:
                    ax, ay = pad.absolute_pos(comp.x, comp.y, comp.rotation)
                    pins.append((ax, ay))
                    break
            else:
                pins.append((comp.x, comp.y))
        if len(pins) < 2:
            continue
        xs = [p[0] for p in pins]
        ys = [p[1] for p in pins]
        net_bboxes[net.name] = (min(xs), min(ys), max(xs), max(ys))

    names = list(net_bboxes.keys())
    count = 0
    for i in range(len(names)):
        bx1, by1, bx2, by2 = net_bboxes[names[i]]
        for j in range(i + 1, len(names)):
            ax1, ay1, ax2, ay2 = net_bboxes[names[j]]
            if bx1 < ax2 and ax1 < bx2 and by1 < ay2 and ay1 < by2:
                count += 1
    return count


# ---------------------------------------------------------------------------
# HPWL Wirelength
# ---------------------------------------------------------------------------

def net_wirelength_hpwl(pins: list[tuple[float, float]], model: str = "auto") -> float:
    """Compute HPWL for a single net.

    Args:
        pins: List of (x, y) absolute pin positions
        model: "true" (default) for true half-perimeter wirelength
               `(max(x)-min(x)) + (max(y)-min(y))`,
               "clique" for the pairwise-Manhattan / (n-1) approximation,
               "star" for the sum-of-distances-to-centroid approximation,
               "auto" selects "true" (matches CostState._compute_net_hpwl
               so the cold evaluator and the SA hot loop agree).

               Historically "auto" picked clique for ≤4 pins and star for
               >4 pins, but those are *different* wirelength proxies and
               silently disagreed with CostState on multi-pin nets — see
               test_total_hpwl_matches_cost_state for the regression that
               caught it.  Clique and star remain accessible via the
               explicit model param for callers that want them.

    Returns:
        HPWL value for this net
    """
    if len(pins) < 2:
        return 0.0

    # Resolve "auto" to "true" — this is the fix for the silent
    # cold/hot-path disagreement.  CostState._compute_net_hpwl always
    # uses true HPWL, so the cold path must too.
    if model == "auto":
        model = "true"

    if model == "true":
        # Standard VLSI half-perimeter wirelength.  O(n), no pairwise work.
        xs = [p[0] for p in pins]
        ys = [p[1] for p in pins]
        return (max(xs) - min(xs)) + (max(ys) - min(ys))

    if model == "clique":
        # Clique model: sum of pairwise Manhattan distances, normalized.
        # Kept as an explicit alternative for callers that want it.
        total = 0.0
        for (x1, y1), (x2, y2) in combinations(pins, 2):
            total += abs(x1 - x2) + abs(y1 - y2)
        return total / (len(pins) - 1)

    if model == "star":
        # Star model: distances to center-of-mass auxiliary point.
        # Kept as an explicit alternative for callers that want it.
        center_x = sum(p[0] for p in pins) / len(pins)
        center_y = sum(p[1] for p in pins) / len(pins)
        return sum(abs(p[0] - center_x) + abs(p[1] - center_y) for p in pins)

    return 0.0


def total_hpwl(model: BoardModel) -> float:
    """Compute total HPWL across all nets.

    Uses component positions to compute absolute pad positions,
    then calculates HPWL per net with the appropriate model.

    Signal-flow chain internal nets are weighted 3.0 (via
    ``_build_net_weights``) to pull chain members together.  Power/ground
    nets are excluded.
    """
    # Build/load net weights (cached on model).
    from engine.cost_state import _build_net_weights
    net_weights = _build_net_weights(model)

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

        hpwl = net_wirelength_hpwl(pins)
        weight = net_weights.get(net.name, 1.0)
        total += hpwl * weight

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
    """Compute penalty for components outside the board outline AND for
    components overlapping any internal keepout zone (mounting holes, slots).

    Penalty is proportional to how far outside the boundary each component is
    (linear ramp on overflow distance).  For keepouts, the penalty is the
    overlap area between the component's bbox and the keepout — this gives SA
    a smooth gradient: a component partially overlapping a keepout is charged
    less than one fully inside it, so SA can learn to slide out gradually
    instead of jumping discontinuously.

    Without the keepout term, SA could freely place components on mounting
    holes during optimization (the legalizer would push them out post-hoc,
    but SA never learned to avoid the keepout in the first place).  This
    closes the SA-loop gap identified in the post-audit recommendations.

    Also adds a linear edge-proximity penalty for component types that need
    extra edge clearance (ICs, MCUs, regulators) — see
    ``cost_state.edge_keepout_extra_for``.  This gives SA a gradient toward
    the interior even when the component is inside the board, so ICs don't
    end up at the board edge (a common-sense DFM rule).
    """
    from engine.cost_state import edge_keepout_extra_for

    total = 0.0
    board = model.board
    keepouts = getattr(model, 'keepouts', None) or []

    for comp in model.components:
        # Edge connectors are exempt from keepout penalties — their body
        # may legitimately overhang a mounting hole near the board edge.
        if comp.is_edge_connector:
            continue
        x_min, y_min, x_max, y_max = comp.bbox

        # --- Board boundary overflow (existing behavior) ---
        left_overflow = max(0.0, board.x_min - x_min)
        right_overflow = max(0.0, x_max - board.x_max)
        top_overflow = max(0.0, board.y_min - y_min)
        bottom_overflow = max(0.0, y_max - board.y_max)

        # Linear penalty (proportional to distance outside)
        overflow = left_overflow + right_overflow + top_overflow + bottom_overflow
        total += overflow

        # --- Keepout overlap (new) ---
        # Charge the overlap AREA between component bbox and each keepout.
        # Area (not distance) gives SA a smooth gradient: a component
        # partially overlapping is charged less than one fully inside,
        # so SA can slide out gradually instead of jumping.
        if keepouts:
            for k in keepouts:
                ox1 = max(x_min, k.x_min)
                oy1 = max(y_min, k.y_min)
                ox2 = min(x_max, k.x_max)
                oy2 = min(y_max, k.y_max)
                if ox2 > ox1 and oy2 > oy1:
                    total += (ox2 - ox1) * (oy2 - oy1)

        # --- Edge-proximity penalty for ICs/MCUs/regulators ---
        # Only fires when the component is INSIDE the board (overflow == 0);
        # once it's outside, the overflow term dominates.  Linear ramp so
        # SA has a smooth gradient toward the interior.
        extra = edge_keepout_extra_for(comp, model)
        if extra > 0.0 and overflow == 0.0:
            d_left = x_min - board.x_min
            d_right = board.x_max - x_max
            d_top = y_min - board.y_min
            d_bottom = board.y_max - y_max
            total += max(0.0, extra - d_left)
            total += max(0.0, extra - d_right)
            total += max(0.0, extra - d_top)
            total += max(0.0, extra - d_bottom)

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

        # Net crossing penalty (routability proxy, à la Cypress ISPD 2025).
        # Computed on the cold path only — NOT in SA's incremental cost.
        from engine.cost_state import NET_CROSSING_WEIGHT, _build_net_weights, _is_power_net
        net_crossings = _compute_net_crossings(model)
        total += NET_CROSSING_WEIGHT * net_crossings

        result = {
            "total": total,
            "hpwl": hpwl,
            "overlap": overlap,
            "boundary": boundary,
            "constraint": constraint,
            "density": density,
            "overlap_count": overlap_count,
            "oob_count": count_out_of_bounds(model),
            "net_crossings": net_crossings,
        }
        # Merge per-rule breakdown so display code can show details
        for rule_name, rule_penalty in constraint_breakdown.items():
            result[f"constraint_{rule_name}"] = rule_penalty

        return result

    def cost(self, model: BoardModel) -> float:
        """Return the scalar total cost."""
        return self.evaluate(model)["total"]
