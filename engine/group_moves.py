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

import math
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


def get_macro_member_refs(model: BoardModel) -> frozenset[str]:
    """Cached set of all cap refs that belong to any macro.

    Use for O(1) 'is this component a macro member?' checks across all
    pipeline stages. The set is the flattened union of every cap ref in
    ``get_decap_map(model).values()``.
    """
    cached = getattr(model, '_macro_member_refs_cache', None)
    if cached is None:
        decap_map = get_decap_map(model)
        cached = frozenset(
            ref for cap_refs in decap_map.values() for ref in cap_refs
        )
        model._macro_member_refs_cache = cached
    return cached


def get_macro_leader_of(model: BoardModel, cap_ref: str | None = None) -> str | None:
    """Cached reverse map: cap_ref -> ic_ref. None if cap is independent.

    Pass ``cap_ref=None`` to just warm the cache (returns None).
    """
    cached = getattr(model, '_macro_leader_of_cache', None)
    if cached is None:
        decap_map = get_decap_map(model)
        cached = {}
        for ic_ref, cap_refs in decap_map.items():
            for r in cap_refs:
                cached[r] = ic_ref
        model._macro_leader_of_cache = cached
    if cap_ref is None:
        return None
    return cached.get(cap_ref)


def clear_macro_caches(model: BoardModel) -> None:
    """Invalidate macro-related caches on ``model``.

    Call this whenever components are added/removed (refs change). SA and
    legalizer mutate positions in place — refs don't change — so they do
    NOT need to call this.
    """
    for attr in ('_macro_member_refs_cache', '_macro_leader_of_cache',
                 '_decap_map_cache', '_signal_flow_chain_map_cache',
                 '_comp_ref_idx_map'):
        if hasattr(model, attr):
            delattr(model, attr)


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
    """Propagate an IC's TRANSLATION delta to its assigned caps.

    Call this AFTER a legalizer/SA function has moved an IC from
    (old_x, old_y) to (new_x, new_y). Each cap is translated by the
    same (dx, dy) and independently clamped to `bounds` (or board).

    Returns the list of cap indices that were moved. If `ic` is not an
    IC or has no caps, returns [].

    NOTE: This only handles translation. If the IC was also rotated,
    use `propagate_ic_move` instead (which handles both translation
    AND rotation).
    """
    if getattr(ic, 'component_type', '') not in IC_TYPES:
        return []
    if abs(new_x - old_x) < 1e-9 and abs(new_y - old_y) < 1e-9:
        return []
    dx = new_x - old_x
    dy = new_y - old_y
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


def propagate_ic_move(
    model: BoardModel,
    ic: Component,
    old_x: float,
    old_y: float,
    old_rot: float,
    new_x: float,
    new_y: float,
    new_rot: float,
    bounds: tuple[float, float, float, float] | None = None,
    decap_map: dict[str, list[str]] | None = None,
) -> list[int]:
    """Propagate an IC's TRANSLATION + ROTATION to its assigned caps.

    This is the macro-aware propagation function. When an IC moves from
    (old_x, old_y, old_rot) to (new_x, new_y, new_rot), each cap:
      1. Computes its offset from the IC's OLD center
      2. Rotates that offset by (new_rot - old_rot) around the IC center
      3. Translates by (new_x - old_x, new_y - old_y)
      4. Clamps to bounds

    This preserves the cap's relative position to the IC through both
    translation AND rotation — the cap-IC group acts as a rigid body
    (a "macro" or "composite component").

    Caps do NOT change their own rotation — only their (x,y) position
    rotates around the IC. Cap orientation is independent of IC
    orientation (a decoupling cap doesn't need to rotate when the IC
    rotates; it just needs to stay adjacent to the same power pin).

    Returns the list of cap indices that were moved.
    """
    if getattr(ic, 'component_type', '') not in IC_TYPES:
        return []
    dx = new_x - old_x
    dy = new_y - old_y
    drot = new_rot - old_rot
    # Normalize rotation delta to [-360, 360]
    drot = ((drot + 360.0) % 360.0)
    if drot > 180.0:
        drot -= 360.0

    if abs(dx) < 1e-9 and abs(dy) < 1e-9 and abs(drot) < 1e-9:
        return []

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

    # Rotation matrix for the cap offset (KiCad CW-positive convention)
    if abs(drot) > 1e-9:
        rad = math.radians(drot)
        cos_r = math.cos(rad)
        sin_r = -math.sin(rad)  # KiCad CW-positive
    else:
        cos_r, sin_r = 1.0, 0.0

    for cap_ref in cap_refs:
        cap_idx = rim.get(cap_ref)
        if cap_idx is None:
            continue
        cap = model.components[cap_idx]
        if cap.is_fixed or getattr(cap, 'is_edge_connector', False):
            continue

        # Compute cap offset from IC's OLD center, rotate, then translate
        ox = cap.x - old_x
        oy = cap.y - old_y
        if abs(drot) > 1e-9:
            new_ox = ox * cos_r - oy * sin_r
            new_oy = ox * sin_r + oy * cos_r
        else:
            new_ox, new_oy = ox, oy

        # New cap position = IC new center + rotated offset
        cap_new_x = new_x + new_ox
        cap_new_y = new_y + new_oy

        # Clamp to bounds
        half_w = cap.effective_width / 2.0
        half_h = cap.effective_height / 2.0
        cap.x = max(x_min + half_w, min(cap_new_x, x_max - half_w))
        cap.y = max(y_min + half_h, min(cap_new_y, y_max - half_h))
        moved.append(cap_idx)
    return moved


def is_macro_member(
    model: BoardModel,
    idx: int,
    decap_map: dict[str, list[str]] | None = None,
) -> bool:
    """Return True if the component at `idx` is a macro MEMBER (a cap
    assigned to an IC). Macro members should NOT be moved independently
    by legalizer/post-legalize stages — they follow their IC.

    ICs themselves are NOT macro members (they're macro LEADERS). Fixed
    components and edge connectors are not macro members.

    O(1) via ``get_macro_member_refs`` cached lookup.
    """
    comp = model.components[idx]
    if comp.is_fixed or getattr(comp, 'is_edge_connector', False):
        return False
    if getattr(comp, 'component_type', '') not in ('capacitor',):
        return False
    return comp.ref in get_macro_member_refs(model)
