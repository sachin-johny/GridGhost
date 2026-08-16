"""Incremental cost tracking for macro-aware SA.

``cost.evaluate()`` is O(nets) for HPWL and O(macros^2) for overlap,
recomputed FROM SCRATCH on every SA move (see ``place/sa.py``). Combined
with an SA iteration budget that itself scales with board size
(``max(1500, 25 * n_macros)``, see ``gridghost.py``), that's roughly
cubic behavior in board size — fine on the ~70-macro test boards
shipped in this repo, not fine on a 300+ component real board.

``IncrementalCostTracker`` caches the three cost terms per touched
entity and only recomputes what a move could possibly have changed:

  - HPWL: only nets with a pin on a component belonging to a touched
    macro (not every net on the board).
  - Overlap: only macro-pairs involving a touched macro (not every
    pair of macros).
  - Boundary: only components belonging to a touched macro.

This turns a single-macro move from O(macros^2 + nets) into
O(macros + local_nets) — the macros^2 term becomes macros (recompute
this macro's overlap against every other macro once, not every pair
against every other pair), and the nets term becomes "nets touching
this macro" instead of "every net on the board".

Usage (mirrors a propose/commit/discard transaction):

    tracker = IncrementalCostTracker(model, macros, alpha=..., beta=..., gamma=...)
    ...
    # caller has already applied a move to macros[idx]'s components
    proposed = tracker.propose([idx])
    if accept(proposed):
        tracker.commit()
    else:
        macros[idx]._restore(snapshot)   # caller's job
        tracker.discard()

The tracker never touches component positions itself — it only reads
them. Correctness is validated against ``cost.evaluate()`` directly in
``tests/test_incremental_cost.py`` (random move sequences, assert the
incremental total matches a from-scratch recompute after every move).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from cost.cost import hpwl_net, macro_overlap_area, cap_attraction_penalty, \
    CAP_ATTRACTION_TARGET_GAP_MM, clearance_pair_charge

if TYPE_CHECKING:
    from models.board_model import BoardModel, Component, Net
    from models.macro import Macro


class IncrementalCostTracker:
    def __init__(
        self,
        model: "BoardModel",
        macros: list["Macro"],
        *,
        alpha: float = 1.0,
        beta: float = 25.0,
        gamma: float = 8.0,
        exclude_nets: set[str] | None = None,
        net_weights: dict[str, float] | None = None,
        keepout_weight: float | None = None,
        delta: float = 0.0,
        rules: list | None = None,
        cap_attraction_weight: float = 0.0,
        cap_pairs: dict[str, str] | None = None,
        clearance_weight: float = 0.0,
        clearance_target_mm: float = 1.0,
    ) -> None:
        self.model = model
        self.macros = macros
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # Keepout weight defaults to γ — matches cost.evaluate()'s default
        # so the incremental and from-scratch costs agree. See IMPROVEMENTS §2.1.
        self.keepout_weight = gamma if keepout_weight is None else keepout_weight
        # Constraint penalty weight (default 0 = disabled). When > 0 AND
        # rules is provided, the tracker asks evaluate_constraint_penalties
        # for the current constraint penalty. This is NOT incremental —
        # the tracker stores the last-seen value and lets the SA caller
        # refresh it every N steps via ``refresh_constraint_penalty()``.
        # See IMPROVEMENTS §2.4.
        self.delta = delta
        self.rules = rules
        self._constraint_total = 0.0
        # Cap→IC attraction (deadband-linear drift penalty on freed caps).
        # Incremental by construction: each pair's charge depends only on
        # the cap and IC positions, and SA only moves whole macros — so a
        # move can only change pairs whose cap or IC member belongs to a
        # touched macro. Pair indexes are built after ``_ref_map`` below.
        self.cap_attraction_weight = cap_attraction_weight
        self._cap_pairs = cap_pairs or {}
        self._cap_attraction_total = 0.0
        self._cap_pair_keys_by_ref: dict[str, list[str]] = {}
        self._cap_pair_charge: dict[str, float] = {}
        # Clearance (routing halo) deficit — cached per macro pair, right
        # next to the overlap cache it shares a loop with. The per-pair
        # charge is max(0, target − edge_gap), clamped at touching; see
        # cost.clearance_pair_charge. Only tracked when enabled (weight>0):
        # otherwise the per-pair dict stays empty and propose/commit skip
        # it, keeping the disabled path byte-for-byte identical to before.
        self.clearance_weight = clearance_weight
        self.clearance_target_mm = clearance_target_mm
        self._clearance_total = 0.0
        if delta > 0 and rules:
            try:
                from engine.constraint_evaluator import evaluate_constraint_penalties
                self._constraint_total, _ = evaluate_constraint_penalties(model, rules)
            except Exception:
                self._constraint_total = 0.0
        self.exclude_nets = exclude_nets or set()
        self.net_weights = net_weights or {}

        self._ref_map: dict[str, "Component"] = {c.ref: c for c in model.components}

        # Cap→IC attraction caches — needs _ref_map for positions. The
        # per-pair charge is max(0, dist − target_gap); pairs are indexed
        # by BOTH member refs so a macro move (which touches whole macros)
        # only recomputes the pairs its members participate in.
        if cap_attraction_weight > 0 and self._cap_pairs:
            self._cap_attraction_total = cap_attraction_penalty(
                model, self._cap_pairs)
            for cap_ref, ic_ref in self._cap_pairs.items():
                pair_key = f"{cap_ref}::{ic_ref}"
                self._cap_pair_keys_by_ref.setdefault(cap_ref, []).append(pair_key)
                self._cap_pair_keys_by_ref.setdefault(ic_ref, []).append(pair_key)
                cap = self._ref_map.get(cap_ref)
                ic = self._ref_map.get(ic_ref)
                d = math.hypot(cap.x - ic.x, cap.y - ic.y) if (cap and ic) else 0.0
                self._cap_pair_charge[pair_key] = \
                    max(0.0, d - CAP_ATTRACTION_TARGET_GAP_MM)

        # ref -> nets that include that ref as a pin (only nets we're
        # actually tracking, i.e. not in exclude_nets).
        self._nets_by_ref: dict[str, list["Net"]] = {}
        self._active_nets: dict[str, "Net"] = {}
        for net in model.nets:
            if net.name in self.exclude_nets:
                continue
            self._active_nets[net.name] = net
            for ref, _ in net.pins:
                self._nets_by_ref.setdefault(ref, []).append(net)

        self._net_hpwl: dict[str, float] = {
            name: hpwl_net(net, self._ref_map, weight=self.net_weights.get(name, 1.0))
            for name, net in self._active_nets.items()
        }
        self._hpwl_total = sum(self._net_hpwl.values())

        self._pair_overlap: dict[tuple[int, int], float] = {}
        self._pair_clearance: dict[tuple[int, int], float] = {}
        n = len(macros)
        overlap_total = 0.0
        clearance_total = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                a = macro_overlap_area(macros[i], macros[j])
                if a:
                    self._pair_overlap[(i, j)] = a
                overlap_total += a
                if clearance_weight > 0:
                    c = clearance_pair_charge(
                        macros[i].bbox, macros[j].bbox, clearance_target_mm)
                    if c:
                        self._pair_clearance[(i, j)] = c
                    clearance_total += c
        self._overlap_total = overlap_total
        self._clearance_total = clearance_total

        self._boundary_by_ref: dict[str, float] = {
            c.ref: self._component_boundary(c) for c in model.components
        }
        self._boundary_total = sum(self._boundary_by_ref.values())

        # Keepout overlap cache — per-component, like boundary. Only built
        # when the board actually has internal cutouts (no-op otherwise).
        # See IMPROVEMENTS §2.1.
        self._keepouts = getattr(model, "keepouts", None) or []
        self._keepout_by_ref: dict[str, float] = {
            c.ref: self._component_keepout(c) for c in model.components
        }
        self._keepout_total = sum(self._keepout_by_ref.values())

        self._pending_nets: dict[str, float] | None = None
        self._pending_pairs: dict[tuple[int, int], float] | None = None
        self._pending_boundary: dict[str, float] | None = None
        self._pending_keepout: dict[str, float] | None = None
        self._pending_cap_pairs: dict[str, float] | None = None
        self._pending_clearance: dict[tuple[int, int], float] | None = None

    def _component_boundary(self, c: "Component") -> float:
        if getattr(c, "is_edge_connector", False):
            return 0.0
        # Polygon-aware — matches cost.total_boundary exactly. Rectangle
        # outline: byte-for-byte the old four-sided overflow sum (the SA
        # inner loop hits this on every proposed move, so it must stay
        # cheap and identical for the common rectangular case).
        return self.model.board.bbox_overflow(c.bbox)

    def _component_keepout(self, c: "Component") -> float:
        """Intersection area of ``c``'s bbox with every internal keepout.

        Matches ``cost.total_keepout_overlap`` per-component. 0.0 when
        the board has no internal cutouts (the common case). Edge
        connectors are exempt. See IMPROVEMENTS §2.1.
        """
        if not self._keepouts:
            return 0.0
        if getattr(c, "is_edge_connector", False):
            return 0.0
        cx1, cy1, cx2, cy2 = c.bbox
        total = 0.0
        for k in self._keepouts:
            ox1 = max(cx1, k.x_min)
            oy1 = max(cy1, k.y_min)
            ox2 = min(cx2, k.x_max)
            oy2 = min(cy2, k.y_max)
            if ox2 > ox1 and oy2 > oy1:
                total += (ox2 - ox1) * (oy2 - oy1)
        return total

    def overlapping_pairs(self) -> list[tuple[int, int]]:
        """Macro-index pairs with nonzero cached overlap right now.

        O(1) — ``_pair_overlap`` only ever holds nonzero entries (zero
        entries are popped in ``commit()``), so this is just a live
        view, not a scan. Used for overlap-biased move selection: pick
        the macros actually in conflict instead of a uniform random
        macro, most of which aren't overlapping anything.
        """
        return list(self._pair_overlap.keys())

    def total(self) -> dict[str, float]:
        return self._compose(self._hpwl_total, self._overlap_total,
                              self._boundary_total, self._keepout_total,
                              self._constraint_total,
                              self._cap_attraction_total,
                              self._clearance_total)

    def _compose(self, hpwl: float, overlap: float, boundary: float,
                  keepout: float = 0.0, constraint: float = 0.0,
                  cap_attraction: float = 0.0,
                  clearance: float = 0.0) -> dict[str, float]:
        return {
            "hpwl": hpwl,
            "overlap": overlap,
            "boundary": boundary,
            "keepout": keepout,
            "constraint": constraint,
            "cap_attraction": cap_attraction,
            "clearance": clearance,
            "total": (self.alpha * hpwl + self.beta * overlap
                       + self.gamma * boundary + self.keepout_weight * keepout
                       + self.delta * constraint
                       + self.cap_attraction_weight * cap_attraction
                       + self.clearance_weight * clearance),
        }

    def refresh_constraint_penalty(self) -> float:
        """Recompute the constraint penalty from scratch.

        SA callers should call this every N steps (e.g. every 50) when
        ``delta > 0`` so the constraint term stays current. Between
        refreshes, the tracker uses the last-seen value — the constraint
        penalty changes slowly (it depends on component positions, not
        on every micro-move), so this is a reasonable approximation.

        Returns the new constraint penalty.
        """
        if not (self.delta > 0 and self.rules):
            return 0.0
        try:
            from engine.constraint_evaluator import evaluate_constraint_penalties
            self._constraint_total, _ = evaluate_constraint_penalties(self.model, self.rules)
        except Exception:
            self._constraint_total = 0.0
        return self._constraint_total

    def propose(self, touched_macro_indices: list[int]) -> dict[str, float]:
        """Recompute cost assuming the touched macros' positions have
        ALREADY been updated by the caller. Returns the proposed total.
        Must be followed by exactly one of commit() / discard() before
        the next propose() call.
        """
        touched = set(touched_macro_indices)

        pending_nets: dict[str, float] = {}
        hpwl_delta = 0.0
        for idx in touched:
            for c in self.macros[idx].members:
                for net in self._nets_by_ref.get(c.ref, ()):
                    if net.name in pending_nets:
                        continue
                    new_v = hpwl_net(net, self._ref_map,
                                      weight=self.net_weights.get(net.name, 1.0))
                    pending_nets[net.name] = new_v
                    hpwl_delta += new_v - self._net_hpwl[net.name]

        pending_pairs: dict[tuple[int, int], float] = {}
        pending_clearance: dict[tuple[int, int], float] = {}
        overlap_delta = 0.0
        clearance_delta = 0.0
        track_clearance = self.clearance_weight > 0
        n = len(self.macros)
        for idx in touched:
            for j in range(n):
                if j == idx:
                    continue
                i, k = (idx, j) if idx < j else (j, idx)
                if (i, k) in pending_pairs:
                    continue
                new_a = macro_overlap_area(self.macros[i], self.macros[k])
                old_a = self._pair_overlap.get((i, k), 0.0)
                pending_pairs[(i, k)] = new_a
                overlap_delta += new_a - old_a
                # Clearance rides the same loop — the touched-pair set is
                # identical (a move can only change the overlap OR the
                # gap of pairs involving a touched macro).
                if track_clearance:
                    new_c = clearance_pair_charge(
                        self.macros[i].bbox, self.macros[k].bbox,
                        self.clearance_target_mm)
                    old_c = self._pair_clearance.get((i, k), 0.0)
                    pending_clearance[(i, k)] = new_c
                    clearance_delta += new_c - old_c

        pending_boundary: dict[str, float] = {}
        boundary_delta = 0.0
        for idx in touched:
            for c in self.macros[idx].members:
                if c.ref in pending_boundary:
                    continue
                new_b = self._component_boundary(c)
                pending_boundary[c.ref] = new_b
                boundary_delta += new_b - self._boundary_by_ref[c.ref]

        # Keepout delta — only non-trivial when the board has internal
        # cutouts. Matches the per-component structure of boundary so the
        # same touched-set walks both. See IMPROVEMENTS §2.1.
        pending_keepout: dict[str, float] = {}
        keepout_delta = 0.0
        if self._keepouts:
            for idx in touched:
                for c in self.macros[idx].members:
                    if c.ref in pending_keepout:
                        continue
                    new_k = self._component_keepout(c)
                    pending_keepout[c.ref] = new_k
                    keepout_delta += new_k - self._keepout_by_ref[c.ref]

        # Cap→IC attraction delta — only pairs whose cap or IC member
        # belongs to a touched macro. The per-pair charge is
        # max(0, dist − target_gap); both sides can move (SA moves whole
        # macros, and both the freed cap and its IC are macro leaders).
        pending_cap_pairs: dict[str, float] = {}
        cap_attraction_delta = 0.0
        if self.cap_attraction_weight > 0 and self._cap_pairs:
            seen: set[str] = set()
            for idx in touched:
                for c in self.macros[idx].members:
                    for pair_key in self._cap_pair_keys_by_ref.get(c.ref, ()):
                        if pair_key in seen:
                            continue
                        seen.add(pair_key)
                        cap_ref, ic_ref = pair_key.split("::")
                        cap = self._ref_map.get(cap_ref)
                        ic = self._ref_map.get(ic_ref)
                        if cap is None or ic is None:
                            continue
                        d = math.hypot(cap.x - ic.x, cap.y - ic.y)
                        new_charge = max(0.0, d - CAP_ATTRACTION_TARGET_GAP_MM)
                        pending_cap_pairs[pair_key] = new_charge
                        cap_attraction_delta += (new_charge
                                                 - self._cap_pair_charge[pair_key])

        self._pending_nets = pending_nets
        self._pending_pairs = pending_pairs
        self._pending_boundary = pending_boundary
        self._pending_keepout = pending_keepout
        self._pending_cap_pairs = pending_cap_pairs
        self._pending_clearance = pending_clearance if track_clearance else None

        # Include the cached constraint term so propose() is symmetric with
        # total() — both carry delta·constraint. The constraint penalty is
        # NOT recomputed per move (too expensive — it walks the whole model);
        # it's a cached value refreshed every N steps by the SA caller via
        # refresh_constraint_penalty(). Between refreshes it's constant, so
        # it cancels in the SA accept/reject Δ (exactly like the externally-
        # added RUDY/pin-density terms). Without this, the tracker path's
        # new_total omitted the constraint term while current_total (from the
        # full evaluate() at init) included it — so after the first accepted
        # step the constraint penalty dropped out of SA's running comparison
        # and effectively stopped influencing per-step acceptance. See
        # IMPROVEMENTS §2.4.
        return self._compose(
            self._hpwl_total + hpwl_delta,
            self._overlap_total + overlap_delta,
            self._boundary_total + boundary_delta,
            self._keepout_total + keepout_delta,
            self._constraint_total,
            self._cap_attraction_total + cap_attraction_delta,
            self._clearance_total + clearance_delta,
        )

    def commit(self) -> None:
        """Bake the last propose()'d values into the permanent cache."""
        if self._pending_nets is None:
            return
        for name, v in self._pending_nets.items():
            self._hpwl_total += v - self._net_hpwl[name]
            self._net_hpwl[name] = v
        for pair, v in self._pending_pairs.items():
            old = self._pair_overlap.get(pair, 0.0)
            self._overlap_total += v - old
            if v:
                self._pair_overlap[pair] = v
            else:
                self._pair_overlap.pop(pair, None)
        for ref, v in self._pending_boundary.items():
            self._boundary_total += v - self._boundary_by_ref[ref]
            self._boundary_by_ref[ref] = v
        if self._pending_keepout:
            for ref, v in self._pending_keepout.items():
                self._keepout_total += v - self._keepout_by_ref[ref]
                self._keepout_by_ref[ref] = v
        if self._pending_cap_pairs:
            for pair_key, v in self._pending_cap_pairs.items():
                self._cap_attraction_total += v - self._cap_pair_charge[pair_key]
                self._cap_pair_charge[pair_key] = v
        if self._pending_clearance:
            for pair, v in self._pending_clearance.items():
                old = self._pair_clearance.get(pair, 0.0)
                self._clearance_total += v - old
                if v:
                    self._pair_clearance[pair] = v
                else:
                    self._pair_clearance.pop(pair, None)
        self._pending_nets = None
        self._pending_pairs = None
        self._pending_boundary = None
        self._pending_keepout = None
        self._pending_cap_pairs = None
        self._pending_clearance = None

    def discard(self) -> None:
        """Drop the last propose()'d values (caller must also revert
        the underlying component positions itself)."""
        self._pending_nets = None
        self._pending_pairs = None
        self._pending_boundary = None
        self._pending_keepout = None
        self._pending_cap_pairs = None
        self._pending_clearance = None
