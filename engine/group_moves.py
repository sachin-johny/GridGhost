"""Cap-IC atomic group helpers.

A "group" is an IC plus its assigned decoupling capacitors. The group
should move as a unit through every pipeline stage (SA moves, greedy
refinement, legalizer snap/clamp/overlap-resolve, abacus row DP, post-
legalize slide/swap) so the cap stays adjacent to its IC.

This module centralises:
  - decap_map lookup (cached on the model)
  - ref→idx map lookup (cached on the model)
  - group membership query
  - apply-delta helper (move IC + caps by the same dx,dy with board clamp)
  - apply-bounds-clamp helper (when an IC is clamped, clamp its caps with
    the same delta; caps that can't follow are flagged for later repair)

The helpers are PURE — they mutate `model.components[i].x/.y/.rotation`
in place and return the set of indices they touched. Callers are
responsible for CostState updates and MoveUndo construction.

Design note: the cap-IC map is owned by `engine/constraint_evaluator.
py:_build_decoupling_map` and published onto `model._decap_map_cache`
by `engine/cost_state.py:CostState.__init__`. If that cache is missing
(e.g. legalizer runs without SA), we rebuild lazily and re-cache.
"""
from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

from models.board_model import BoardModel

if TYPE_CHECKING:
    from models.board_model import Component


IC_TYPES = frozenset({'ic', 'mcu', 'regulator'})


def get_decap_map(model: BoardModel) -> dict[str, list[str]]:
    """Return {ic_ref: [cap_ref, ...]} — lazily built and cached on model."""
    cache = getattr(model, '_decap_map_cache', None)
    if cache is not None:
        return cache
    try:
        from engine.constraint_evaluator import _build_decoupling_map
        cache = _build_decoupling_map(model)
    except Exception:
        cache = {}
    model._decap_map_cache = cache
    return cache


def get_signal_flow_chain_map(model: BoardModel) -> dict[str, list[str]]:
    """Return {ref: [chain_member_refs]} for signal-flow chains.

    Builds a map where each component ref that is part of a signal-flow
    chain maps to ALL other member refs of that chain.  This lets
    ``get_group_indices`` return the full chain group for any member.

    Cached on model as ``_signal_flow_chain_map_cache``.
    """
    cache = getattr(model, '_signal_flow_chain_map_cache', None)
    if cache is not None:
        return cache
    chain_map: dict[str, list[str]] = {}
    try:
        from engine.subcircuit_patterns import detect_subcircuit_patterns
        for p in detect_subcircuit_patterns(model):
            if p.motif_type != 'signal_flow_chain':
                continue
            members = p.all_refs
            for ref in members:
                # Each ref maps to all OTHER chain members (not itself).
                chain_map[ref] = [m for m in members if m != ref]
    except Exception:
        pass
    model._signal_flow_chain_map_cache = chain_map
    return chain_map


def get_ref_idx_map(model: BoardModel) -> dict[str, int]:
    """Return {ref: idx} — lazily built and cached on model."""
    rim = getattr(model, '_comp_ref_idx_map', None)
    if rim is not None and len(rim) == len(model.components):
        return rim
    rim = {c.ref: i for i, c in enumerate(model.components)}
    model._comp_ref_idx_map = rim
    return rim


def get_group_indices(
    model: BoardModel,
    idx: int,
    decap_map: dict[str, list[str]] | None = None,
    ref_idx_map: dict[str, int] | None = None,
    chain_map: dict[str, list[str]] | None = None,
) -> set[int]:
    """Return {idx} ∪ {group member indices}.

    Group members are:
      1. If idx is an IC: its assigned decoupling caps (from decap_map)
      2. If idx is in a signal-flow chain: all non-connector chain members

    Fixed components and edge connectors are excluded — they should not
    be moved by group moves.
    """
    comp = model.components[idx]
    group = {idx}

    # 1. IC + decoupling caps
    if getattr(comp, 'component_type', '') in IC_TYPES:
        if decap_map is None:
            decap_map = get_decap_map(model)
        cap_refs = decap_map.get(comp.ref, [])
        if cap_refs:
            if ref_idx_map is None:
                ref_idx_map = get_ref_idx_map(model)
            for cap_ref in cap_refs:
                cap_idx = ref_idx_map.get(cap_ref)
                if cap_idx is None:
                    continue
                cap = model.components[cap_idx]
                if cap.is_fixed or getattr(cap, 'is_edge_connector', False):
                    continue
                group.add(cap_idx)

    # 2. Signal-flow chain members (non-connector, non-fixed)
    if chain_map is None:
        chain_map = get_signal_flow_chain_map(model)
    chain_refs = chain_map.get(comp.ref, [])
    if chain_refs:
        if ref_idx_map is None:
            ref_idx_map = get_ref_idx_map(model)
        for chain_ref in chain_refs:
            chain_idx = ref_idx_map.get(chain_ref)
            if chain_idx is None:
                continue
            chain_comp = model.components[chain_idx]
            # Skip connectors (they're frozen on the perimeter) and fixed comps.
            if chain_comp.is_fixed or getattr(chain_comp, 'is_edge_connector', False):
                continue
            if getattr(chain_comp, 'component_type', '') == 'connector':
                continue
            group.add(chain_idx)

    return group


def apply_delta_with_clamp(
    model: BoardModel,
    indices: Iterable[int],
    dx: float,
    dy: float,
    bounds: tuple[float, float, float, float] | None = None,
) -> None:
    """Translate every component in `indices` by (dx, dy).

    Each component is independently clamped to `bounds = (x_min, y_min,
    x_max, y_max)` so it stays inside the board (or interior bbox). If
    `bounds` is None, uses `model.board` extents.

    Note: independent clamping means a cap may end up at a different
    offset from its IC if the IC hits the boundary. This is acceptable
    for SA exploration — `_nudge_caps_to_ics` (legalizer) restores
    adjacency post-SA. The benefit of clamping is that caps never go
    fully out of bounds, so the cost function's boundary penalty stays
    finite and gradient-friendly.
    """
    if bounds is None:
        b = model.board
        x_min, y_min, x_max, y_max = b.x_min, b.y_min, b.x_max, b.y_max
    else:
        x_min, y_min, x_max, y_max = bounds

    for i in indices:
        comp = model.components[i]
        half_w = comp.effective_width / 2.0
        half_h = comp.effective_height / 2.0
        comp.x = max(x_min + half_w, min(comp.x + dx, x_max - half_w))
        comp.y = max(y_min + half_h, min(comp.y + dy, y_max - half_h))


def move_group_with_ic(
    model: BoardModel,
    ic_idx: int,
    dx: float,
    dy: float,
    bounds: tuple[float, float, float, float] | None = None,
    decap_map: dict[str, list[str]] | None = None,
    ref_idx_map: dict[str, int] | None = None,
) -> set[int]:
    """Move an IC and all its assigned caps by (dx, dy).

    Returns the set of indices that were moved (always includes ic_idx;
    includes cap indices that actually moved). Caller is responsible
    for CostState.incremental_update(moved) and MoveUndo capture.
    """
    group = get_group_indices(model, ic_idx, decap_map, ref_idx_map)
    apply_delta_with_clamp(model, group, dx, dy, bounds)
    return group


def get_ic_caps_for_index(
    model: BoardModel,
    idx: int,
    decap_map: dict[str, list[str]] | None = None,
    ref_idx_map: dict[str, int] | None = None,
) -> list[int]:
    """Return list of cap indices assigned to the IC at `idx` (empty if
    `idx` is not an IC or has no caps). Excludes fixed/edge caps.
    """
    comp = model.components[idx]
    if getattr(comp, 'component_type', '') not in IC_TYPES:
        return []
    if decap_map is None:
        decap_map = get_decap_map(model)
    cap_refs = decap_map.get(comp.ref, [])
    if not cap_refs:
        return []
    if ref_idx_map is None:
        ref_idx_map = get_ref_idx_map(model)
    out = []
    for cap_ref in cap_refs:
        cap_idx = ref_idx_map.get(cap_ref)
        if cap_idx is None:
            continue
        cap = model.components[cap_idx]
        if cap.is_fixed or getattr(cap, 'is_edge_connector', False):
            continue
        out.append(cap_idx)
    return out


def get_group_followers(
    model: BoardModel,
    idx: int,
    decap_map: dict[str, list[str]] | None = None,
    ref_idx_map: dict[str, int] | None = None,
    chain_map: dict[str, list[str]] | None = None,
) -> list[int]:
    """Return list of group member indices that should follow the component
    at `idx` when it moves (excluding `idx` itself).

    This is the union of:
      - Decoupling caps assigned to the IC at `idx` (if it's an IC)
      - Signal-flow chain members (non-connector, non-fixed)

    Use this in SA move operators to propagate deltas to all followers.
    """
    group = get_group_indices(model, idx, decap_map, ref_idx_map, chain_map)
    group.discard(idx)
    return sorted(group)


def propagate_ic_delta(
    model: BoardModel,
    ic: Component,
    old_x: float,
    old_y: float,
    new_x: float,
    new_y: float,
    bounds: tuple[float, float, float, float] | None = None,
    decap_map: dict[str, list[str]] | None = None,
) -> list[int]:
    """Propagate an IC's position delta to its assigned caps.

    Call this AFTER a legalizer/SA function has moved an IC from
    (old_x, old_y) to (new_x, new_y). Each cap is translated by the
    same (dx, dy) and independently clamped to `bounds` (or board).

    Returns the list of cap indices that were moved. If `ic` is not an
    IC or has no caps, returns [].

    This is the legalizer's group-awareness hook: any function that
    moves a component should call this with the IC's old/new position
    so caps follow. Caps that get clamped to bounds (because the IC
    moved near a board edge) will be temporarily desynchronized from
    the IC — the legalizer's `_nudge_caps_to_ics` (Step 8) repairs
    that at the end.
    """
    if getattr(ic, 'component_type', '') not in IC_TYPES:
        return []
    if abs(new_x - old_x) < 1e-9 and abs(new_y - old_y) < 1e-9:
        return []
    dx = new_x - old_x
    dy = new_y - old_y
    # Find the IC's index by ref
    if decap_map is None:
        decap_map = get_decap_map(model)
    cap_refs = decap_map.get(ic.ref, [])
    if not cap_refs:
        return []
    rim = get_ref_idx_map(model)
    moved = []
    if bounds is None:
        b = model.board
        x_min, y_min, x_max, y_max = b.x_min, b.y_min, b.x_max, b.y_max
    else:
        x_min, y_min, x_max, y_max = bounds
    for cap_ref in cap_refs:
        cap_idx = rim.get(cap_ref)
        if cap_idx is None:
            continue
        cap = model.components[cap_idx]
        if cap.is_fixed or getattr(cap, 'is_edge_connector', False):
            continue
        half_w = cap.effective_width / 2.0
        half_h = cap.effective_height / 2.0
        cap.x = max(x_min + half_w, min(cap.x + dx, x_max - half_w))
        cap.y = max(y_min + half_h, min(cap.y + dy, y_max - half_h))
        moved.append(cap_idx)
    return moved
