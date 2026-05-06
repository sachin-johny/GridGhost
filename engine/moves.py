"""SA move operators for Simulated Annealing.

Four move types: translate, swap, rotate, median — selected with
temperature-dependent probabilities to balance exploration vs exploitation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from models.board_model import BoardModel


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
) -> MoveUndo:
    """Pick a random component and apply a random displacement."""
    idx = random.choice(moveable_indices)
    comp = model.components[idx]
    old = (idx, comp.x, comp.y, comp.rotation)

    dx = random.uniform(-window_mm, window_mm)
    dy = random.uniform(-window_mm, window_mm)
    comp.x += dx
    comp.y += dy

    return MoveUndo(move_type='translate', old_states=[old])


def do_swap(
    model: BoardModel,
    moveable_indices: list[int],
) -> MoveUndo:
    """Swap positions of two random moveable components."""
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

    c1.x, c2.x = c2.x, c1.x
    c1.y, c2.y = c2.y, c1.y

    return MoveUndo(move_type='swap', old_states=[old1, old2])


def do_rotate(
    model: BoardModel,
    moveable_indices: list[int],
) -> MoveUndo:
    """Rotate a random component by 90, 180, or 270 degrees."""
    idx = random.choice(moveable_indices)
    comp = model.components[idx]
    old = (idx, comp.x, comp.y, comp.rotation)

    rot = random.choice([90.0, 180.0, 270.0])
    comp.set_rotation((comp.rotation + rot) % 360.0)

    return MoveUndo(move_type='rotate', old_states=[old])


def do_median(
    model: BoardModel,
    moveable_indices: list[int],
    t_ratio: float,
    noise_mm: float,
) -> MoveUndo:
    """Move a component toward the centroid of its connected pads."""
    idx = random.choice(moveable_indices)
    comp = model.components[idx]
    old = (idx, comp.x, comp.y, comp.rotation)

    # Compute centroid of connected pads (excluding own pads)
    cx_sum, cy_sum = 0.0, 0.0
    count = 0
    for net in model.nets:
        # Check if this component is on this net
        comp_on_net = False
        for ref, _ in net.pins:
            if ref == comp.ref:
                comp_on_net = True
                break
        if not comp_on_net:
            continue

        for ref, pad_name in net.pins:
            if ref == comp.ref:
                continue
            other = None
            for c in model.components:
                if c.ref == ref:
                    other = c
                    break
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
        return MoveUndo(move_type='median', old_states=[old])

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

    return MoveUndo(move_type='median', old_states=[old])


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
