"""SA move operators for Simulated Annealing.

Four move types: translate, swap, rotate, median — selected with
temperature-dependent probabilities to balance exploration vs exploitation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from models.board_model import BoardModel
from engine.group_moves import get_group_followers, apply_delta_with_clamp


@dataclass
class MoveUndo:
    move_type: str
    old_states: list[tuple[int, float, float, float]]  # (idx, old_x, old_y, old_rotation)


def select_move_type(t_ratio: float) -> str:
    """Select move type based on temperature ratio (0=cold, 1=hot).

    Cold: translate=70%, median=15%, swap=5%, rotate=10%
    Hot:  translate=55%, median=5%,  swap=20%, rotate=20%
    """
    r = random.random()
    # Interpolate probabilities
    t_pct = 0.55 + 0.15 * (1 - t_ratio)  # translate: 55% hot → 70% cold
    s_pct = 0.20 - 0.15 * (1 - t_ratio)  # swap: 20% hot → 5% cold
    r_pct = 0.20 - 0.10 * (1 - t_ratio)  # rotate: 20% hot → 10% cold
    # median gets the rest

    if r < t_pct:
        return 'translate'
    elif r < t_pct + s_pct:
        return 'swap'
    elif r < t_pct + s_pct + r_pct:
        return 'rotate'
    else:
        return 'median'


def get_moveable_indices(model: BoardModel) -> list[int]:
    """Get indices of components that SA can move (non-fixed, non-connector)."""
    return [
        i for i, c in enumerate(model.components)
        if not c.is_fixed and c.component_type != 'connector'
    ]


def do_translate(
    model: BoardModel,
    moveable_indices: list[int],
    t_ratio: float,
    window_mm: float,
    bias_dx: float = 0.0,
    bias_dy: float = 0.0,
) -> MoveUndo:
    """Pick a random component and apply a random displacement.

    Group-aware: when the selected component is an IC with assigned
    decoupling caps, the caps are translated by the same delta. Keeps
    cap-IC groups together so the density penalty (which treats the
    group as one unit) doesn't fight the decoupling_proximity constraint.

    Caps are clamped to board bounds — a temporarily broken group is
    acceptable; the next nudge pass restores adjacency.
    """
    idx = random.choice(moveable_indices)
    comp = model.components[idx]
    old = (idx, comp.x, comp.y, comp.rotation)

    dx = random.uniform(-window_mm, window_mm) + bias_dx
    dy = random.uniform(-window_mm, window_mm) + bias_dy
    comp.x += dx
    comp.y += dy

    old_states = [old]

    # If this is an IC, move its assigned caps by the same delta.
    follower_indices = get_group_followers(model, idx)
    for cap_idx in follower_indices:
        cap = model.components[cap_idx]
        old_states.append((cap_idx, cap.x, cap.y, cap.rotation))
    if follower_indices:
        apply_delta_with_clamp(model, follower_indices, dx, dy)

    return MoveUndo(move_type='translate', old_states=old_states)


def do_swap(
    model: BoardModel,
    moveable_indices: list[int],
) -> MoveUndo:
    """Swap positions of two random moveable components.

    Group-aware: when either swapped component is an IC with assigned
    caps, its caps are translated by the same delta as the IC so the
    cap-IC group survives the swap.
    """
    if len(moveable_indices) < 2:
        return MoveUndo(move_type='translate', old_states=[])

    idx1 = random.choice(moveable_indices)
    idx2 = idx1
    for _ in range(10):
        idx2 = random.choice(moveable_indices)
        if idx2 != idx1:
            break
    if idx2 == idx1:
        return MoveUndo(move_type='translate', old_states=[])

    c1 = model.components[idx1]
    c2 = model.components[idx2]
    old1 = (idx1, c1.x, c1.y, c1.rotation)
    old2 = (idx2, c2.x, c2.y, c2.rotation)

    # Compute swap deltas: c1 moves to c2's position (delta = c2 - c1)
    # and vice versa. If either is an IC, its caps follow by the same delta.
    delta1 = (c2.x - c1.x, c2.y - c1.y)
    delta2 = (c1.x - c2.x, c1.y - c2.y)

    c1.x, c2.x = c2.x, c1.x
    c1.y, c2.y = c2.y, c1.y

    # Build old_states as a dict to dedupe — if a cap is ALSO one of the
    # swapped pair (e.g. swapping an IC with its own cap), the swap-pair
    # entry wins (it was captured first) and we skip the cap-follower entry.
    old_states_map: dict[int, tuple[int, float, float, float]] = {
        idx1: old1,
        idx2: old2,
    }

    # Caps follow their IC's swap delta.
    for ic_idx, delta in ((idx1, delta1), (idx2, delta2)):
        follower_indices = get_group_followers(model, ic_idx)
        # Skip caps that are part of the swap pair itself — already recorded.
        follower_indices = [ci for ci in follower_indices if ci not in old_states_map]
        for cap_idx in follower_indices:
            cap = model.components[cap_idx]
            old_states_map[cap_idx] = (cap_idx, cap.x, cap.y, cap.rotation)
        if follower_indices:
            apply_delta_with_clamp(model, follower_indices, delta[0], delta[1])

    return MoveUndo(move_type='swap', old_states=list(old_states_map.values()))


def do_rotate(
    model: BoardModel,
    moveable_indices: list[int],
) -> MoveUndo:
    """Rotate a random component by 90, 180, or 270 degrees.

    Group-aware: when the rotated component is an IC, its assigned caps
    are translated so they keep their pre-rotation offset relative to
    the IC's center. Caps don't rotate themselves (their orientation is
    independent of the IC's), but their (x,y) position rotates around
    the IC's center to preserve the relative geometry of the group.
    """
    idx = random.choice(moveable_indices)
    comp = model.components[idx]
    old = (idx, comp.x, comp.y, comp.rotation)

    rot = random.choice([90.0, 180.0, 270.0])
    new_rotation = (comp.rotation + rot) % 360.0

    follower_indices = get_group_followers(model, idx)
    old_states = [old]

    # If this is an IC with caps, capture pre-rotation cap states and
    # rotate cap positions around the IC's center.
    if follower_indices:
        ic_cx, ic_cy = comp.x, comp.y
        rad = math.radians(rot)
        cos_r = math.cos(rad)
        sin_r = -math.sin(rad)  # KiCad CW-positive convention
        for cap_idx in follower_indices:
            cap = model.components[cap_idx]
            old_states.append((cap_idx, cap.x, cap.y, cap.rotation))
            # Rotate cap offset around IC center
            ox = cap.x - ic_cx
            oy = cap.y - ic_cy
            new_ox = ox * cos_r - oy * sin_r
            new_oy = ox * sin_r + oy * cos_r
            cap.x = ic_cx + new_ox
            cap.y = ic_cy + new_oy

    comp.set_rotation(new_rotation)

    return MoveUndo(move_type='rotate', old_states=old_states)


def do_median(
    model: BoardModel,
    moveable_indices: list[int],
    t_ratio: float,
    noise_mm: float,
) -> MoveUndo:
    """Move a component toward the centroid of its connected pads.

    v9 optimization: uses the component's pre-built nets list and pad
    lookup for O(k) instead of O(nets * components) per call.
    """
    idx = random.choice(moveable_indices)
    comp = model.components[idx]
    old = (idx, comp.x, comp.y, comp.rotation)

    # Build a quick ref→component map for lookups
    cx_sum, cy_sum = 0.0, 0.0
    count = 0

    # Use comp.nets (pre-built list) for fast net lookup
    comp_ref = comp.ref
    comp_nets = set(comp.nets) if comp.nets else set()

    if not comp_nets:
        # Fallback: small random translate
        dx = random.uniform(-1.0, 1.0)
        dy = random.uniform(-1.0, 1.0)
        comp.x += dx
        comp.y += dy
        # Group-aware: caps follow
        old_states = [old]
        follower_indices = get_group_followers(model, idx)
        for cap_idx in follower_indices:
            cap = model.components[cap_idx]
            old_states.append((cap_idx, cap.x, cap.y, cap.rotation))
        if follower_indices:
            apply_delta_with_clamp(model, follower_indices, dx, dy)
        return MoveUndo(move_type='median', old_states=old_states)

    # Build ref→component index once (cached on model for performance).
    # BoardModel.get_component maintains this index lazily and rebuilds
    # automatically if components are appended/removed.
    ref_map = None
    if hasattr(model, '_comp_ref_map') and len(model._comp_ref_map) == len(model.components):
        ref_map = model._comp_ref_map
    else:
        ref_map = {c.ref: c for c in model.components}
        model._comp_ref_map = ref_map

    for net in model.nets:
        if net.name not in comp_nets:
            continue
        for ref, pad_name in net.pins:
            if ref == comp_ref:
                continue
            other = ref_map.get(ref)
            if not other:
                continue
            for pad in other.pads:
                if pad.pad_name == pad_name:
                    ax, ay = pad.absolute_pos(other.x, other.y, other.rotation)
                    cx_sum += ax
                    cy_sum += ay
                    count += 1
                    break

    if count == 0:
        # Fallback: small random translate
        dx = random.uniform(-1.0, 1.0)
        dy = random.uniform(-1.0, 1.0)
        comp.x += dx
        comp.y += dy
        # Group-aware: caps follow
        old_states = [old]
        follower_indices = get_group_followers(model, idx)
        for cap_idx in follower_indices:
            cap = model.components[cap_idx]
            old_states.append((cap_idx, cap.x, cap.y, cap.rotation))
        if follower_indices:
            apply_delta_with_clamp(model, follower_indices, dx, dy)
        return MoveUndo(move_type='median', old_states=old_states)

    centroid_x = cx_sum / count
    centroid_y = cy_sum / count

    # Move 50-80% toward centroid (fraction varies with temperature)
    fraction = 0.5 + 0.3 * (1.0 - t_ratio)
    dx = (centroid_x - comp.x) * fraction
    dy = (centroid_y - comp.y) * fraction

    # Add noise proportional to temperature
    dx += random.uniform(-noise_mm, noise_mm)
    dy += random.uniform(-noise_mm, noise_mm)

    comp.x += dx
    comp.y += dy

    # Group-aware: caps follow the IC's median move.
    old_states = [old]
    follower_indices = get_group_followers(model, idx)
    for cap_idx in follower_indices:
        cap = model.components[cap_idx]
        old_states.append((cap_idx, cap.x, cap.y, cap.rotation))
    if follower_indices:
        apply_delta_with_clamp(model, follower_indices, dx, dy)

    return MoveUndo(move_type='median', old_states=old_states)


def revert_move(model: BoardModel, undo: MoveUndo):
    """Restore component positions/rotations from a MoveUndo."""
    for idx, old_x, old_y, old_rot in undo.old_states:
        comp = model.components[idx]
        comp.x = old_x
        comp.y = old_y
        comp.set_rotation(old_rot)


def affected_indices(undo: MoveUndo) -> set[int]:
    """Get the set of component indices affected by a move."""
    return {idx for idx, _, _, _ in undo.old_states}
