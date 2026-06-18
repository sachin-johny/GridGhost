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

_POWER_PREFIXES = (
    'GND', 'AGND', 'DGND', 'PGND', 'SGND',
    'VSS', 'VCC', 'VDD', 'VEE', 'VBAT', 'VBUS',
)
_POWER_VOLTAGE_RE = re.compile(r'^[+\-]\d[\d.]*V', re.IGNORECASE)


def _is_power_net(name: str) -> bool:
    n = name.lstrip('/').upper()
    return any(n.startswith(p) for p in _POWER_PREFIXES) or bool(_POWER_VOLTAGE_RE.match(n))


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
        self._crystal_pairs: list[tuple[str, str]] | None = None
        self._connectors: list | None = None  # list[Component], but annotated as list to avoid import cycle
        if self._rules:
            rule_names = {r.name for r in self._rules if r.enabled}
            if 'decoupling_proximity' in rule_names:
                self._decap_map = _build_decoupling_map(model)
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
        for i in range(self._n):
            self._comp_boundary[i] = self._compute_boundary(self._comps[i].bbox, board)

        # Constraint penalties
        if self._rules:
            self._constraint_total, self._constraint_breakdown = (
                evaluate_constraint_penalties(
                    self.model, self._rules,
                    decap_map=self._decap_map,
                    crystal_pairs=self._crystal_pairs,
                    connectors=self._connectors,
                )
            )
        else:
            self._constraint_total = 0.0
            self._constraint_breakdown = {}

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
        return (max(xs) - min(xs)) + (max(ys) - min(ys))

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
    ) -> float:
        left = max(0.0, board.x_min - bbox[0])
        right = max(0.0, bbox[2] - board.x_max)
        top = max(0.0, board.y_min - bbox[1])
        bottom = max(0.0, bbox[3] - board.y_max)
        overflow = left + right + top + bottom
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
        for i in moved_indices:
            self._comp_boundary[i] = self._compute_boundary(self._comps[i].bbox, board)

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
                )
            )

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

        # Restore xmin index
        for i, bbox in snap['saved_bbox'].items():
            self._xmin_items = [(x, idx) for x, idx in self._xmin_items if idx != i]
            insort(self._xmin_items, (bbox[0], i))

    def old_bboxes_from_states(
        self,
        old_states: list[tuple[int, float, float, float]],
    ) -> dict[int, tuple[float, float, float, float]]:
        """Reconstruct pre-move bounding boxes from MoveUndo state."""
        result: dict[int, tuple[float, float, float, float]] = {}
        for idx, old_x, old_y, old_rot in old_states:
            comp = self._comps[idx]
            old_width = comp.width if old_rot in (0.0, 180.0) else comp.height
            old_height = comp.height if old_rot in (0.0, 180.0) else comp.width
            result[idx] = (
                old_x - old_width / 2.0,
                old_y - old_height / 2.0,
                old_x + old_width / 2.0,
                old_y + old_height / 2.0,
            )
        return result

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

    @property
    def total_cost(self) -> float:
        hpwl = self._hpwl_sum()
        overlap = (OVERLAP_WEIGHT * self._overlap_sum() + OVERLAP_COUNT_WEIGHT * self.overlap_count) * self._penalty_scale
        boundary = BOUNDARY_WEIGHT * self._boundary_sum() * self._penalty_scale
        constraint = CONSTRAINT_WEIGHT * self._constraint_total * self._penalty_scale
        return hpwl + overlap + boundary + constraint

    @property
    def normalized_cost(self) -> float:
        """Unscaled cost for best-solution tracking."""
        return (self._hpwl_sum()
                + OVERLAP_WEIGHT * self._overlap_sum()
                + OVERLAP_COUNT_WEIGHT * self.overlap_count
                + BOUNDARY_WEIGHT * self._boundary_sum()
                + CONSTRAINT_WEIGHT * self._constraint_total)

    @property
    def hpwl(self) -> float:
        return self._hpwl_sum()

    @property
    def overlap_count(self) -> int:
        return len(self._pair_overlaps)

    def update_penalty_scale(self, scale: float):
        self._penalty_scale = scale
