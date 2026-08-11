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

from typing import TYPE_CHECKING

from cost.cost import hpwl_net, macro_overlap_area

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
    ) -> None:
        self.model = model
        self.macros = macros
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.exclude_nets = exclude_nets or set()
        self.net_weights = net_weights or {}

        self._ref_map: dict[str, "Component"] = {c.ref: c for c in model.components}

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
        n = len(macros)
        overlap_total = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                a = macro_overlap_area(macros[i], macros[j])
                if a:
                    self._pair_overlap[(i, j)] = a
                overlap_total += a
        self._overlap_total = overlap_total

        self._boundary_by_ref: dict[str, float] = {
            c.ref: self._component_boundary(c) for c in model.components
        }
        self._boundary_total = sum(self._boundary_by_ref.values())

        self._pending_nets: dict[str, float] | None = None
        self._pending_pairs: dict[tuple[int, int], float] | None = None
        self._pending_boundary: dict[str, float] | None = None

    def _component_boundary(self, c: "Component") -> float:
        if getattr(c, "is_edge_connector", False):
            return 0.0
        # Polygon-aware — matches cost.total_boundary exactly. Rectangle
        # outline: byte-for-byte the old four-sided overflow sum (the SA
        # inner loop hits this on every proposed move, so it must stay
        # cheap and identical for the common rectangular case).
        return self.model.board.bbox_overflow(c.bbox)

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
        return self._compose(self._hpwl_total, self._overlap_total, self._boundary_total)

    def _compose(self, hpwl: float, overlap: float, boundary: float) -> dict[str, float]:
        return {
            "hpwl": hpwl,
            "overlap": overlap,
            "boundary": boundary,
            "total": self.alpha * hpwl + self.beta * overlap + self.gamma * boundary,
        }

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
        overlap_delta = 0.0
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

        pending_boundary: dict[str, float] = {}
        boundary_delta = 0.0
        for idx in touched:
            for c in self.macros[idx].members:
                if c.ref in pending_boundary:
                    continue
                new_b = self._component_boundary(c)
                pending_boundary[c.ref] = new_b
                boundary_delta += new_b - self._boundary_by_ref[c.ref]

        self._pending_nets = pending_nets
        self._pending_pairs = pending_pairs
        self._pending_boundary = pending_boundary

        return self._compose(
            self._hpwl_total + hpwl_delta,
            self._overlap_total + overlap_delta,
            self._boundary_total + boundary_delta,
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
        self._pending_nets = None
        self._pending_pairs = None
        self._pending_boundary = None

    def discard(self) -> None:
        """Drop the last propose()'d values (caller must also revert
        the underlying component positions itself)."""
        self._pending_nets = None
        self._pending_pairs = None
        self._pending_boundary = None
