"""Incremental cost computation for Simulated Annealing.

Provides O(k) incremental updates instead of O(n^2) full recomputation,
essential for SA performance where thousands of moves are evaluated per second.
"""

from __future__ import annotations

import math
import re
from bisect import bisect_left, insort
from itertools import combinations

from models.board_model import BoardModel
from profiles.board_profiles import ConstraintRule
from engine.constraint_evaluator import (
    evaluate_constraint_penalties,
    _build_decoupling_map,
    _find_crystal_mcu_pairs,
)

OVERLAP_WEIGHT = 25.0   # stronger — SA should avoid creating overlaps
BOUNDARY_WEIGHT = 4.0   # stronger — discourage OOB during SA, not just legalization
CONSTRAINT_WEIGHT = 4.0  # delta — matches BoardProfile default
OVERLAP_COUNT_WEIGHT = 12.0  # extra penalty per overlapping pair


# ---------------------------------------------------------------------------
# Component-type-aware edge keepout
# ---------------------------------------------------------------------------
# Loads the per-type edge keepout extra from config.json (single source of
# truth). Returns {component_type: extra_mm}. ICs/MCUs/regulators get extra
# edge clearance so they don't end up at the board edge — a common-sense
# DFM rule a human PCB designer always applies. Falls back to the
# PlacementConfig defaults if config loading fails.
_EDGE_KEEPOUT_TABLE_CACHE: dict[str, float] | None = None


def _load_edge_keepout_table() -> dict[str, float]:
    """Load the per-type edge keepout extra (mm) from config.json.

    Cached after first call. Falls back to PlacementConfig defaults if
    config loading fails (e.g. config.json missing).
    """
    global _EDGE_KEEPOUT_TABLE_CACHE
    if _EDGE_KEEPOUT_TABLE_CACHE is not None:
        return _EDGE_KEEPOUT_TABLE_CACHE
    try:
        from config import load_config
        cfg = load_config()
        _EDGE_KEEPOUT_TABLE_CACHE = dict(cfg.placement.edge_keepout_extra)
    except Exception:
        # Fall back to the defaults hardcoded in PlacementConfig.
        _EDGE_KEEPOUT_TABLE_CACHE = {
            "ic": 5.0, "mcu": 5.0, "regulator": 5.0, "crystal": 3.0,
        }
    return _EDGE_KEEPOUT_TABLE_CACHE


def edge_keepout_extra_for(comp, model=None) -> float:
    """Return the extra edge keepout (mm) for a component's type.

    Returns 0.0 for passives/connectors/generic — they only get the base
    `margin`. ICs/MCUs/regulators get the base `margin` PLUS this extra.

    Finding 5 fix (density scaling consolidation):
    This function returns the BASE keepout extra (no density scaling).
    Density scaling is the caller's responsibility — e.g. the macro-v2
    legalizer's `_keepout_cb` in `place/pipeline.py` applies a single
    density-adaptive scale factor based on INTERIOR-BBOX density (the
    actual packing pressure the legalizer faces), not the board-level
    density this function used to use.

    Why the change: previously this function applied its own 1.0→0.3
    scaling (board-density based) AND the macro legalizer's callback
    applied another 1.0→0.2 scaling (interior-density based). The two
    multiplied, giving 0.06× at 0.55 density — almost certainly not what
    either author intended. Now there's ONE density scaling, in the
    legalizer callback, using the density metric that actually reflects
    packing pressure.

    The ``model`` argument is retained for backwards compatibility but
    no longer affects the return value. Callers that previously relied
    on the density scaling here should apply their own scaling at the
    call site (see `place/pipeline.py:_keepout_cb` for an example).
    """
    t = getattr(comp, 'component_type', '') or ''
    return _load_edge_keepout_table().get(t, 0.0)

# Density-equality penalty (Gini coefficient on a 10x10 cell-occupancy grid).
# Penalizes inequality of cell-occupancy — 0 when components are uniformly
# spread, increases as they cluster.
#
# Tuning: 0.4 was negligible, 8.0 caused 5-22% HPWL regression on 4/5 test
# boards while only marginally improving spread. 3.0 keeps density as a soft
# signal (~5-12% of HPWL). The actual filling of empty cells is done by the
# spread MOVE OPERATOR (smart_placement._pick_spread_move) which JUMPS
# components to empty cells within the outline — fill first, expand only if
# no empty cells remain.
#
# Use density_adaptive_weight(board_density, n_components) at call sites
# instead of reading this constant directly — it scales the base weight DOWN
# for small/dense boards (no room to spread) and UP for sparse boards.
#
# Cap-IC grouping: decoupling caps assigned to an IC are absorbed into the
# IC's cell (counted as ONE unit at the IC's position). This prevents the
# Gini penalty from fighting the decoupling_proximity constraint.
DENSITY_WEIGHT = 3.0
DENSITY_GRID = 10

# Net-crossing weight: penalizes pairs of nets whose bounding boxes intersect.
# This is a routability proxy inspired by Cypress (ISPD 2025 Best Paper) —
# net crossings correlate better with routing congestion than HPWL alone.
# Soft weight (not scaled by penalty_scale) — influences SA at all temperatures.
# Updated on _compute_all (every 3 SA steps via resync).
NET_CROSSING_WEIGHT = 0.5


def density_adaptive_weight(board_density: float, n_components: int = 0,
                            grid_cells: int = DENSITY_GRID * DENSITY_GRID) -> float:
    """Scale DENSITY_WEIGHT by board density AND component count per cell.

    Two factors multiply:

    1. Board density (physical room to spread):
       - Sparse (< 0.20):       1.5x — lots of empty space, push hard.
       - Medium (0.20–0.30):    1.0x — base weight.
       - Dense (0.30–0.40):     0.5x — limited room, gentle nudge.
       - Very dense (> 0.40):   0.2x — almost no room, barely nudge.

    2. Component count per grid cell (congestion potential):
       - Low (< 0.65):          1.0x — plenty of cells, spreading helps.
       - Tight (0.65–0.70):     0.7x — getting crowded.
       - Very tight (0.70–0.78):0.5x — spreading shreds HPWL.
       - Packed (> 0.78):       0.25x — barely nudge, no room to spread.

    The second factor is critical: a fixed weight that helps cbb (0.68
    comps/cell, -14.7% HPWL) catastrophically hurts test4 (0.82 comps/cell,
    +8% HPWL) and test6 (0.74 comps/cell, +7% HPWL) because those boards
    have too many components competing for too few cells — the density
    force just pulls components apart with nowhere good to put them.
    """
    # Factor 1: board density (room to spread physically)
    if board_density < 0.20:
        density_scale = 1.5
    elif board_density < 0.30:
        density_scale = 1.0
    elif board_density < 0.40:
        density_scale = 0.5
    else:
        density_scale = 0.2

    # Factor 2: component count per cell (congestion potential)
    if n_components > 0:
        comps_per_cell = n_components / grid_cells
        if comps_per_cell > 0.78:
            count_scale = 0.25  # packed — barely nudge
        elif comps_per_cell > 0.70:
            count_scale = 0.5   # very tight — spreading shreds HPWL
        elif comps_per_cell > 0.65:
            count_scale = 0.7   # tight — reduce force
        else:
            count_scale = 1.0   # plenty of room — full force
    else:
        count_scale = 1.0

    return DENSITY_WEIGHT * density_scale * count_scale

_POWER_PREFIXES = (
    'GND', 'AGND', 'DGND', 'PGND', 'SGND',
    'VSS', 'VCC', 'VDD', 'VEE', 'VBAT', 'VBUS',
)
_POWER_VOLTAGE_RE = re.compile(r'^[+\-]\d[\d.]*V', re.IGNORECASE)


def _is_power_net(name: str) -> bool:
    n = name.lstrip('/').upper()
    return any(n.startswith(p) for p in _POWER_PREFIXES) or bool(_POWER_VOLTAGE_RE.match(n))


# Weight multiplier for signal-flow chain internal nets.  3.0 means a
# chain internal net (e.g. U1→U5 op-amp output) contributes 3× the HPWL
# of a normal signal net.  This pulls chain members together against the
# connector-perimeter placement which tries to split endpoints.
SIGNAL_FLOW_NET_WEIGHT = 3.0


def _build_net_weights(model: BoardModel) -> dict[str, float]:
    """Build per-net weight dict from detected signal-flow chains.

    Nets that connect components within a signal-flow chain get weight
    SIGNAL_FLOW_NET_WEIGHT (3.0); all other nets get weight 1.0 (default,
    not stored in the dict to save memory).  The weight is applied to
    HPWL in both the SA hot path (CostState._compute_net_hpwl) and the
    cold path (cost_function.total_hpwl).

    Cached on model as ``_net_weights_cache`` so it's built once per
    placement pipeline invocation, not per SA move.
    """
    cache = getattr(model, '_net_weights_cache', None)
    if cache is not None:
        return cache
    weights: dict[str, float] = {}
    try:
        from engine.subcircuit_patterns import detect_subcircuit_patterns
        patterns = detect_subcircuit_patterns(model)
        # Build a set of all chain member refs for fast lookup.
        for p in patterns:
            if p.motif_type != 'signal_flow_chain':
                continue
            chain_refs = set(p.all_refs)
            # Find all non-power signal nets that connect ≥2 chain members.
            for net in model.nets:
                if _is_power_net(net.name):
                    continue
                net_members = set(net.component_refs)
                # If this net connects ≥2 chain members, it's a chain internal net.
                if len(net_members & chain_refs) >= 2:
                    weights[net.name] = SIGNAL_FLOW_NET_WEIGHT
    except Exception:
        pass  # best-effort; if detection fails, all nets stay weight 1.0
    model._net_weights_cache = weights
    return weights


def _pair_key(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


class CostState:
    """Incremental cost state for SA moves.

    Maintains per-net HPWL, per-pair overlap penalties, and per-component
    boundary penalties. Supports O(k) incremental updates and O(k) snapshot/restore.
    """

    def __init__(self, model: BoardModel, power_net_prefixes=None,
                 rules: list[ConstraintRule] | None = None):
        self.model = model
        self._comps = model.components
        self._n = len(self._comps)
        self._rules = rules or []

        # Build net -> set of component indices
        self._net_indices: dict[str, list[int]] = {}
        self._power_nets: set[str] = set()
        self._comp_nets: list[set[str]] = [set() for _ in range(self._n)]

        for net in model.nets:
            if _is_power_net(net.name):
                self._power_nets.add(net.name)
            indices = []
            for ref, _pad_name in net.pins:
                for idx, comp in enumerate(self._comps):
                    if comp.ref == ref:
                        indices.append(idx)
                        self._comp_nets[idx].add(net.name)
                        break
            if len(indices) >= 2:
                self._net_indices[net.name] = indices

        # Per-net HPWL cache
        self._net_hpwl: dict[str, float] = {}
        # Per-pair overlap penalty cache
        self._pair_overlaps: dict[tuple[int, int], float] = {}
        # Per-component boundary penalty
        self._comp_boundary: list[float] = [0.0] * self._n
        # Sorted xmin index for fast overlap neighbor lookup
        self._xmin_items: list[tuple[float, int]] = []

        # Constraint penalty cache (recomputed incrementally)
        self._constraint_total: float = 0.0
        self._constraint_breakdown: dict[str, float] = {}

        # Cached constraint topology — SA only moves components, it never
        # changes net connectivity.  Build once and reuse for the lifetime
        # of this CostState.  The penalty VALUE still depends on positions,
        # so we still recompute the value every move — but skip the O(nets
        # × components) topology rebuild.
        self._decap_map: dict[str, list[str]] | None = None
        self._decap_pairs: list[tuple] | None = None  # materialised (ic, cap) Component pairs
        self._chain_adjacent_pairs: list[tuple] | None = None  # materialised chain pairs
        self._crystal_pairs: list[tuple[str, str]] | None = None
        self._connectors: list | None = None  # list[Component], but annotated as list to avoid import cycle
        # Set of cap refs absorbed into an IC group for density grouping.
        # Empty when decoupling rule is inactive (density stays zero).
        self._grouped_cap_refs: set[str] = set()
        if self._rules:
            rule_names = {r.name for r in self._rules if r.enabled}
            if 'decoupling_proximity' in rule_names:
                self._decap_map = _build_decoupling_map(model)
                # Publish to model so do_translate (moves.py) can reuse the
                # same map without rebuilding — SA calls it thousands of times.
                model._decap_map_cache = self._decap_map
                for cap_refs in self._decap_map.values():
                    self._grouped_cap_refs.update(cap_refs)
                # Materialise (ic_component, cap_component) pairs once —
                # eliminates ~2.5M model.get_component lookups during SA.
                self._decap_pairs = []
                for ic_ref, cap_refs in self._decap_map.items():
                    ic = model.get_component(ic_ref)
                    if not ic:
                        continue
                    for cap_ref in cap_refs:
                        cap = model.get_component(cap_ref)
                        if cap:
                            self._decap_pairs.append((ic, cap))
                # Warm macro registry caches — legalizer/post_legalize
                # query them on every stage. Build once here so the first
                # stage doesn't pay the build cost mid-pipeline.
                from engine.group_moves import (
                    get_macro_member_refs, get_macro_leader_of,
                )
                get_macro_member_refs(model)
                get_macro_leader_of(model)
            if 'crystal_mcu' in rule_names:
                self._crystal_pairs = _find_crystal_mcu_pairs(model)
            if 'connector_edge' in rule_names:
                # Cache Component objects (not refs) — avoids O(N) get_component
                # lookups per SA move. SA mutates positions but never replaces
                # the Component instances themselves, so the cache stays valid.
                self._connectors = [
                    c for c in model.components
                    if getattr(c, 'component_type', '') == 'connector'
                    and not c.is_fixed
                ]
            # Always build chain adjacent pairs (even if signal_flow_grouping
            # rule is not active) — they're cheap and used by net weights too.
            self._chain_adjacent_pairs = []
            try:
                from engine.subcircuit_patterns import detect_subcircuit_patterns
                for p in detect_subcircuit_patterns(model):
                    if p.motif_type != 'signal_flow_chain':
                        continue
                    chain = p.metadata.get('chain', [])
                    if len(chain) < 2:
                        continue
                    for i in range(len(chain) - 1):
                        ca = model.get_component(chain[i])
                        cb = model.get_component(chain[i + 1])
                        if ca and cb:
                            self._chain_adjacent_pairs.append((ca, cb))
            except Exception:
                pass

        # Net weights: signal-flow chain internal nets get weight 3.0 to
        # pull chain members together against the connector-perimeter pull.
        # Default weight 1.0 for all other nets.  Cached on model so both
        # CostState (SA hot path) and CostFunction (cold path) share it.
        self._net_weights: dict[str, float] = _build_net_weights(model)

        # Density grid state (only meaningful when _decap_map is set, i.e.
        # the profile opts into density via the decoupling rule).
        self._density_grid: list[int] = [0] * (DENSITY_GRID * DENSITY_GRID)
        self._density_total: int = 0
        self._density_cell_w: float = 0.0
        self._density_cell_h: float = 0.0
        self._density_penalty: float = 0.0
        # Net crossing count (routability proxy) — updated on _compute_all.
        self._net_crossing_count: int = 0
        # Adaptive density weight — scaled by board density AND component
        # count per cell so dense/packed boards (no room to spread) get a
        # gentle nudge while sparse boards get a strong push.
        # See density_adaptive_weight() for the rationale.
        board_area = model.board.width * model.board.height
        if board_area > 0:
            comp_area = sum(c.effective_width * c.effective_height for c in model.components)
            n_moveable = sum(1 for c in model.components if not c.is_fixed and not c.is_edge_connector)
            self._density_weight = density_adaptive_weight(
                comp_area / board_area, n_moveable)
        else:
            self._density_weight = DENSITY_WEIGHT

        # Penalty scaling (annealer controls this)
        self._penalty_scale = 1.0

        self._compute_all()

    # ------------------------------------------------------------------
    # Full computation
    # ------------------------------------------------------------------

    def _compute_all(self):
        self._net_hpwl.clear()
        self._pair_overlaps.clear()
        self._xmin_items.clear()

        # HPWL per net (skip power nets)
        for net_name, indices in self._net_indices.items():
            if net_name in self._power_nets:
                continue
            self._net_hpwl[net_name] = self._compute_net_hpwl(net_name, indices)

        # Overlaps — build sorted xmin index + compute all pairs
        for i in range(self._n):
            bbox = self._comps[i].bbox
            self._xmin_items.append((bbox[0], i))

        self._xmin_items.sort()

        for i in range(self._n):
            bbox_i = self._comps[i].bbox
            for j in range(i + 1, self._n):
                bbox_j = self._comps[j].bbox
                penalty = self._compute_overlap_penalty(bbox_i, bbox_j)
                if penalty > 0:
                    self._pair_overlaps[(i, j)] = penalty

        # Boundary
        board = self.model.board
        keepouts = getattr(self.model, 'keepouts', None) or []
        for i in range(self._n):
            comp = self._comps[i]
            self._comp_boundary[i] = self._compute_boundary(
                comp.bbox, board, keepouts=keepouts,
                is_edge_connector=comp.is_edge_connector,
                edge_keepout_extra=edge_keepout_extra_for(comp, self.model),
            )

        # Constraint penalties
        if self._rules:
            self._constraint_total, self._constraint_breakdown = (
                evaluate_constraint_penalties(
                    self.model, self._rules,
                    decap_map=self._decap_map,
                    crystal_pairs=self._crystal_pairs,
                    connectors=self._connectors,
                    decap_pairs=self._decap_pairs,
                    chain_adjacent_pairs=self._chain_adjacent_pairs,
                )
            )
        else:
            self._constraint_total = 0.0
            self._constraint_breakdown = {}

        # Density grid (gated on decoupling rule active — other profiles
        # keep density = 0 with zero overhead).
        if self._decap_map is not None:
            self._compute_density_grid()
            self._density_penalty = self._compute_density_penalty()
        else:
            self._density_penalty = 0.0

        # Net crossing count: pairs of non-power nets with intersecting bboxes.
        # Computed on _compute_all (every 3 SA steps via resync) — approximate
        # between resyncs but corrected on next full recompute.
        self._net_crossing_count = self._compute_net_crossings()

    # ------------------------------------------------------------------
    # Density grid (Gini coefficient spreading penalty)
    # ------------------------------------------------------------------

    def _compute_density_grid(self) -> None:
        """Build the 10x10 cell-occupancy grid.

        Only counts movable, non-edge-connector components. Decoupling caps
        assigned to an IC (via _decap_map) are absorbed into the IC's group
        and NOT counted individually — the group occupies one cell at the
        IC's position. This prevents the Gini penalty from fighting the
        decoupling_proximity constraint: spreading "groups" instead of
        individual components means caps stay near their IC.
        """
        board = self.model.board
        cell_w = (board.x_max - board.x_min) / DENSITY_GRID
        cell_h = (board.y_max - board.y_min) / DENSITY_GRID
        self._density_cell_w = cell_w
        self._density_cell_h = cell_h

        grid = [0] * (DENSITY_GRID * DENSITY_GRID)
        total = 0
        bx_min = board.x_min
        by_min = board.y_min
        grouped = self._grouped_cap_refs

        for c in self._comps:
            if c.is_fixed or c.is_edge_connector:
                continue
            if c.ref in grouped:
                continue

            gx = int((c.x - bx_min) / cell_w) if cell_w > 0 else 0
            gy = int((c.y - by_min) / cell_h) if cell_h > 0 else 0
            if gx < 0:
                gx = 0
            elif gx >= DENSITY_GRID:
                gx = DENSITY_GRID - 1
            if gy < 0:
                gy = 0
            elif gy >= DENSITY_GRID:
                gy = DENSITY_GRID - 1
            grid[gx + gy * DENSITY_GRID] += 1
            total += 1

        self._density_grid = grid
        self._density_total = total

    def _compute_density_penalty(self) -> float:
        """Gini coefficient of cell-occupancy, scaled by total movable count.

        Returns 0 when components are uniformly distributed; increases as
        they cluster into fewer cells.
        """
        if self._density_total == 0:
            return 0.0
        n = DENSITY_GRID * DENSITY_GRID
        s = sorted(self._density_grid)
        cum = sum((i + 1) * v for i, v in enumerate(s))
        total = sum(s)
        if total == 0:
            return 0.0
        gini = (2 * cum) / (n * total) - (n + 1) / n
        return gini * self._density_total

    # ------------------------------------------------------------------
    # Per-item computation helpers
    # ------------------------------------------------------------------

    def _compute_net_hpwl(self, net_name: str, indices: list[int]) -> float:
        comp_map = {idx: self._comps[idx] for idx in indices}
        pins = []
        net_obj = self.model.get_net(net_name)
        if not net_obj:
            return 0.0

        for ref, pad_name in net_obj.pins:
            comp = None
            comp_idx = None
            for idx in indices:
                if self._comps[idx].ref == ref:
                    comp = self._comps[idx]
                    comp_idx = idx
                    break
            if not comp:
                continue
            for pad in comp.pads:
                if pad.pad_name == pad_name:
                    abs_x, abs_y = pad.absolute_pos(comp.x, comp.y, comp.rotation)
                    pins.append((abs_x, abs_y))
                    break
            else:
                pins.append((comp.x, comp.y))

        if len(pins) < 2:
            return 0.0

        xs = [p[0] for p in pins]
        ys = [p[1] for p in pins]
        hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys))
        # Apply signal-flow net weight (1.0 for normal nets, 3.0 for chain internal).
        weight = self._net_weights.get(net_name, 1.0)
        return hpwl * weight

    @staticmethod
    def _compute_overlap_penalty(
        bbox_a: tuple[float, float, float, float],
        bbox_b: tuple[float, float, float, float],
    ) -> float:
        ox1 = max(bbox_a[0], bbox_b[0])
        oy1 = max(bbox_a[1], bbox_b[1])
        ox2 = min(bbox_a[2], bbox_b[2])
        oy2 = min(bbox_a[3], bbox_b[3])
        if ox2 <= ox1 or oy2 <= oy1:
            return 0.0
        return (ox2 - ox1) * (oy2 - oy1)

    @staticmethod
    def _compute_boundary(
        bbox: tuple[float, float, float, float],
        board,
        keepouts: list | None = None,
        is_edge_connector: bool = False,
        edge_keepout_extra: float = 0.0,
    ) -> float:
        """Boundary penalty for a single component bbox.

        Charges linear overflow distance for the board outline (existing
        behavior) PLUS the overlap area with any internal keepout zone
        (mounting holes, slots).  The keepout term closes the SA-loop
        gap: without it, SA could freely place components on mounting
        holes during optimization (the legalizer would push them out
        post-hoc, but SA never learned to avoid the keepout).

        Area (not distance) is used for keepouts so SA sees a smooth
        gradient: a component partially overlapping a keepout is charged
        less than one fully inside, so SA can slide out gradually instead
        of jumping discontinuously.

        Edge connectors are exempt from keepout penalties — their body
        may legitimately overhang a mounting hole near the board edge.

        ``edge_keepout_extra`` adds a LINEAR edge-proximity penalty for
        component types that need extra edge clearance (ICs, MCUs,
        regulators).  Without this, SA has no gradient to keep ICs away
        from the board edge — the overflow penalty only fires once a
        component is OUTSIDE the board, so an IC sitting 0.1mm inside the
        margin gets zero penalty.  The proximity term charges
        ``max(0, extra - dist_to_edge)`` for each of the 4 edges, giving
        SA a smooth ramp that pushes ICs toward the interior.  This is
        the common-sense DFM rule a human PCB designer always applies
        (routing room, panelization clearance, assembly clearance).
        """
        left = max(0.0, board.x_min - bbox[0])
        right = max(0.0, bbox[2] - board.x_max)
        top = max(0.0, board.y_min - bbox[1])
        bottom = max(0.0, bbox[3] - board.y_max)
        overflow = left + right + top + bottom  # linear ramp

        # Keepout overlap contribution
        if keepouts and not is_edge_connector:
            x_min, y_min, x_max, y_max = bbox
            for k in keepouts:
                ox1 = max(x_min, k.x_min)
                oy1 = max(y_min, k.y_min)
                ox2 = min(x_max, k.x_max)
                oy2 = min(y_max, k.y_max)
                if ox2 > ox1 and oy2 > oy1:
                    overflow += (ox2 - ox1) * (oy2 - oy1)

        # Edge-proximity penalty for component types that need extra
        # clearance (ICs, MCUs, regulators). Linear ramp so SA has a
        # smooth gradient toward the interior. Only fires when the
        # component is INSIDE the board (overflow == 0); once it's
        # outside, the overflow term dominates.
        if edge_keepout_extra > 0.0 and overflow == 0.0 and not is_edge_connector:
            d_left = bbox[0] - board.x_min
            d_right = board.x_max - bbox[2]
            d_top = bbox[1] - board.y_min
            d_bottom = board.y_max - bbox[3]
            overflow += max(0.0, edge_keepout_extra - d_left)
            overflow += max(0.0, edge_keepout_extra - d_right)
            overflow += max(0.0, edge_keepout_extra - d_top)
            overflow += max(0.0, edge_keepout_extra - d_bottom)

        return overflow  # linear ramp (quadratic too aggressive for SA)

    # ------------------------------------------------------------------
    # Incremental update
    # ------------------------------------------------------------------

    def incremental_update(self, moved_indices: set[int]) -> float:
        """Recompute costs for moved components. Returns new total_cost."""
        # Update xmin index
        for i in moved_indices:
            bbox = self._comps[i].bbox
            # Remove old entry
            self._xmin_items = [(x, idx) for x, idx in self._xmin_items if idx != i]
            insort(self._xmin_items, (bbox[0], i))

        # Recompute HPWL for affected nets
        affected_nets: set[str] = set()
        for i in moved_indices:
            affected_nets.update(self._comp_nets[i])

        for net_name in affected_nets:
            if net_name in self._power_nets:
                continue
            indices = self._net_indices.get(net_name)
            if indices:
                self._net_hpwl[net_name] = self._compute_net_hpwl(net_name, indices)

        # Recompute overlaps involving moved components
        # Remove old overlap entries
        keys_to_remove = [k for k in self._pair_overlaps if k[0] in moved_indices or k[1] in moved_indices]
        for k in keys_to_remove:
            del self._pair_overlaps[k]

        # Scan nearby using sorted xmin
        board_xmax = self.model.board.x_max
        for i in moved_indices:
            bbox_i = self._comps[i].bbox
            # Use binary search: find all components with xmin < bbox_i.xmax
            pos = bisect_left(self._xmin_items, (bbox_i[2],))
            for k in range(pos):
                _, j = self._xmin_items[k]
                if j == i:
                    continue
                pk = _pair_key(i, j)
                if pk in self._pair_overlaps:
                    continue
                bbox_j = self._comps[j].bbox
                # Quick reject
                if bbox_j[0] >= bbox_i[2] or bbox_i[0] >= bbox_j[2]:
                    continue
                penalty = self._compute_overlap_penalty(bbox_i, bbox_j)
                if penalty > 0:
                    self._pair_overlaps[pk] = penalty

        # Boundary
        board = self.model.board
        keepouts = getattr(self.model, 'keepouts', None) or []
        for i in moved_indices:
            comp = self._comps[i]
            self._comp_boundary[i] = self._compute_boundary(
                comp.bbox, board, keepouts=keepouts,
                is_edge_connector=comp.is_edge_connector,
                edge_keepout_extra=edge_keepout_extra_for(comp, self.model),
            )

        # Constraint penalties — recompute the VALUE (depends on positions)
        # but skip the topology rebuild (decap_map / crystal_pairs /
        # connectors are cached since net connectivity never changes
        # during SA).
        if self._rules:
            self._constraint_total, self._constraint_breakdown = (
                evaluate_constraint_penalties(
                    self.model, self._rules,
                    decap_map=self._decap_map,
                    crystal_pairs=self._crystal_pairs,
                    connectors=self._connectors,
                    decap_pairs=self._decap_pairs,
                    chain_adjacent_pairs=self._chain_adjacent_pairs,
                )
            )

        # Density grid: recompute on every move. O(n) with minimal per-comp
        # overhead. Skipped entirely when decoupling rule inactive.
        if self._decap_map is not None:
            self._compute_density_grid()
            self._density_penalty = self._compute_density_penalty()

        return self.total_cost

    # ------------------------------------------------------------------
    # Snapshot / Restore
    # ------------------------------------------------------------------

    def snapshot(
        self,
        moved_indices: set[int],
        old_bboxes: dict[int, tuple[float, float, float, float]] | None = None,
    ) -> dict:
        """Save state for the given indices so we can restore on rejection.

        Callers may pass pre-move bounding boxes via ``old_bboxes`` when the
        model has already been mutated by a proposed move. This preserves the
        spatial index for rejected SA moves."""
        saved_net_hpwl = {
            net_name: self._net_hpwl[net_name]
            for net_name in self._comp_nets_moved(moved_indices)
            if net_name in self._net_hpwl
        }
        saved_pair_overlaps = {
            k: v for k, v in self._pair_overlaps.items()
            if k[0] in moved_indices or k[1] in moved_indices
        }
        saved_boundary = {i: self._comp_boundary[i] for i in moved_indices}
        saved_bbox = dict(old_bboxes) if old_bboxes is not None else {i: self._comps[i].bbox for i in moved_indices}

        return {
            'hpwl': self._hpwl_sum(),
            'overlap_penalty': self._overlap_sum(),
            'boundary_penalty': self._boundary_sum(),
            'constraint_total': self._constraint_total,
            'constraint_breakdown': dict(self._constraint_breakdown),
            'net_hpwl': saved_net_hpwl,
            'pair_overlaps': saved_pair_overlaps,
            'comp_boundary': saved_boundary,
            'saved_bbox': saved_bbox,
            'density_penalty': self._density_penalty,
            'density_grid': list(self._density_grid),
            'density_total': self._density_total,
        }

    def restore(self, snap: dict):
        """Restore from a snapshot taken before a rejected move."""
        # Restore net HPWL
        for net_name, hpwl in snap['net_hpwl'].items():
            if net_name in self._net_hpwl or net_name in self._net_indices:
                self._net_hpwl[net_name] = hpwl

        # Restore pair overlaps: remove all current involving moved, add saved
        moved_indices = set(snap['saved_bbox'].keys())
        keys_to_remove = [k for k in self._pair_overlaps if k[0] in moved_indices or k[1] in moved_indices]
        for k in keys_to_remove:
            del self._pair_overlaps[k]
        self._pair_overlaps.update(snap['pair_overlaps'])

        # Restore boundary
        for i, val in snap['comp_boundary'].items():
            self._comp_boundary[i] = val

        # Restore constraint cache
        self._constraint_total = snap.get('constraint_total', 0.0)
        self._constraint_breakdown = snap.get('constraint_breakdown', {})

        # Restore density state (only present when density is active)
        if 'density_grid' in snap:
            self._density_grid = list(snap['density_grid'])
            self._density_total = snap.get('density_total', 0)
            self._density_penalty = snap.get('density_penalty', 0.0)

        # Restore xmin index
        for i, bbox in snap['saved_bbox'].items():
            self._xmin_items = [(x, idx) for x, idx in self._xmin_items if idx != i]
            insort(self._xmin_items, (bbox[0], i))

    def old_bboxes_from_states(
        self,
        old_states: list[tuple[int, float, float, float]],
    ) -> dict[int, tuple[float, float, float, float]]:
        """Reconstruct pre-move bounding boxes from MoveUndo state.

        Uses ``Component.bbox_at`` so the reconstructed bbox matches what
        ``Component.bbox`` would actually return at the old pose — including
        courtyard margin and the rotated bbox_offset.  The previous
        implementation used raw ``width``/``height`` (missing courtyard
        margin) and ignored ``bbox_offset_x``/``bbox_offset_y``, which
        silently corrupted the xmin spatial index over many SA moves and
        caused incremental cost to drift from from-scratch recompute.

        Regression covered by ``test_snapshot_restore_n_moves_property``.
        """
        return {
            idx: self._comps[idx].bbox_at(old_x, old_y, old_rot)
            for idx, old_x, old_y, old_rot in old_states
        }

    def _comp_nets_moved(self, moved_indices: set[int]) -> set[str]:
        nets: set[str] = set()
        for i in moved_indices:
            nets.update(self._comp_nets[i])
        return nets

    # ------------------------------------------------------------------
    # Cost views
    # ------------------------------------------------------------------

    def _hpwl_sum(self) -> float:
        return sum(self._net_hpwl.values())

    def _overlap_sum(self) -> float:
        return sum(self._pair_overlaps.values())

    def _boundary_sum(self) -> float:
        return sum(self._comp_boundary)

    def _compute_net_crossings(self) -> int:
        """Count pairs of non-power nets whose bounding boxes intersect.

        This is a routability proxy: nets whose bboxes overlap are likely
        to cross each other during routing, causing congestion and vias.
        Inspired by Cypress (ISPD 2025 Best Paper) which treats net
        crossing as a first-class objective distinct from HPWL.

        O(nets²) — only called on _compute_all (every 3 SA steps via
        resync), not on every incremental_update.
        """
        # Build per-net bounding boxes from pin positions.
        net_bboxes: dict[str, tuple[float, float, float, float]] = {}
        for net_name, indices in self._net_indices.items():
            if net_name in self._power_nets:
                continue
            if not indices:
                continue
            xs: list[float] = []
            ys: list[float] = []
            net_obj = self.model.get_net(net_name)
            if not net_obj:
                continue
            comp_by_ref = {self._comps[i].ref: self._comps[i] for i in indices}
            for ref, pad_name in net_obj.pins:
                comp = comp_by_ref.get(ref)
                if not comp:
                    continue
                for pad in comp.pads:
                    if pad.pad_name == pad_name:
                        ax, ay = pad.absolute_pos(comp.x, comp.y, comp.rotation)
                        xs.append(ax)
                        ys.append(ay)
                        break
                else:
                    xs.append(comp.x)
                    ys.append(comp.y)
            if len(xs) < 2:
                continue
            net_bboxes[net_name] = (min(xs), min(ys), max(xs), max(ys))

        # Count intersecting pairs.
        names = list(net_bboxes.keys())
        count = 0
        for i in range(len(names)):
            bx1, by1, bx2, by2 = net_bboxes[names[i]]
            for j in range(i + 1, len(names)):
                ax1, ay1, ax2, ay2 = net_bboxes[names[j]]
                # Bbox intersection test: NOT (a_left >= b_right OR a_right <= b_left OR ...)
                if bx1 < ax2 and ax1 < bx2 and by1 < ay2 and ay1 < by2:
                    count += 1
        return count

    @property
    def total_cost(self) -> float:
        hpwl = self._hpwl_sum()
        overlap = (OVERLAP_WEIGHT * self._overlap_sum() + OVERLAP_COUNT_WEIGHT * self.overlap_count) * self._penalty_scale
        boundary = BOUNDARY_WEIGHT * self._boundary_sum() * self._penalty_scale
        constraint = CONSTRAINT_WEIGHT * self._constraint_total * self._penalty_scale
        # Density is intentionally NOT scaled by penalty_scale — it's a soft
        # signal that should influence SA at all temperatures.
        density = self._density_weight * self._density_penalty
        # Net crossing is NOT included in SA cost — too expensive to update
        # incrementally.  Computed on _compute_all for reporting only.
        return hpwl + overlap + boundary + constraint + density

    @property
    def normalized_cost(self) -> float:
        """Unscaled cost for best-solution tracking."""
        return (self._hpwl_sum()
                + OVERLAP_WEIGHT * self._overlap_sum()
                + OVERLAP_COUNT_WEIGHT * self.overlap_count
                + BOUNDARY_WEIGHT * self._boundary_sum()
                + CONSTRAINT_WEIGHT * self._constraint_total
                + self._density_weight * self._density_penalty)

    @property
    def net_crossing_count(self) -> int:
        """Number of net bbox intersection pairs (reporting only).

        Updated on _compute_all (every 3 SA steps via resync).  NOT
        included in total_cost — too expensive to update incrementally.
        Use CostFunction.evaluate for the final net-crossing penalty.
        """
        return self._net_crossing_count

    @property
    def hpwl(self) -> float:
        return self._hpwl_sum()

    @property
    def overlap_count(self) -> int:
        return len(self._pair_overlaps)

    def update_penalty_scale(self, scale: float):
        self._penalty_scale = scale

    @property
    def density_penalty(self) -> float:
        """Current Gini density penalty (debug/reporting)."""
        return self._density_penalty
