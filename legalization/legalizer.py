"""Legalization pass for PCB placement.

Runs AFTER optimization to produce a legal, grid-snapped placement:
1. Grid snapping — round coordinates to KiCad's placement grid
2. Boundary enforcement — clamp out-of-bounds components
3. Overlap resolution — iteratively shift overlapping components apart

IMPORTANT: Run legalization as a single post-optimization pass,
NOT iteratively inside the optimizer loop — it would corrupt
gradient/energy signals.
"""

from __future__ import annotations

import math
from typing import Optional

from models.board_model import BoardModel, Component, BoardOutline


def legalize(
    model: BoardModel,
    grid_mm: float = 0.1,
    max_iterations: int = 300,
    push_strength: float = 1,
    verbose: bool = False,
) -> BoardModel:
    """Full legalization pipeline.

    Args:
        model: Board model with optimized but potentially illegal positions
        grid_mm: Grid size in mm (0.1mm or 0.05mm typical for KiCad)
        max_iterations: Max iterations for overlap resolution (300 for dense designs)
        push_strength: How far to push overlapping components (fraction of overlap, 0.8 is aggressive)
        verbose: Print progress information

    Returns:
        BoardModel with legalized component positions
    """
    if verbose:
        overlaps_before = _count_overlaps(model)
        oob_before = _count_oob(model)
        print(f"Legalization input: {overlaps_before} overlaps, {oob_before} out-of-bounds")

    # Step 1: Snap to grid
    _snap_to_grid(model, grid_mm)

    # Step 2: Enforce board boundary
    _enforce_boundary(model)

    # Step 3: Resolve overlaps
    _resolve_overlaps(model, max_iterations, push_strength, grid_mm, verbose)

    # Step 4: Final grid snap and boundary check
    _snap_to_grid(model, grid_mm)
    _enforce_boundary(model)

    if verbose:
        overlaps_after = _count_overlaps(model)
        oob_after = _count_oob(model)
        print(f"Legalization output: {overlaps_after} overlaps, {oob_after} out-of-bounds")

    return model


def _snap_to_grid(model: BoardModel, grid_mm: float) -> None:
    """Round all component positions to the nearest grid point.
    
    Connectors are NOT snapped - they keep their exact perimeter positions.
    """
    for comp in model.components:
        # Skip fixed components AND connectors
        if comp.is_fixed or getattr(comp, 'component_type', '') == "connector":
            continue
        comp.x = round(comp.x / grid_mm) * grid_mm
        comp.y = round(comp.y / grid_mm) * grid_mm
        # Snap rotation to nearest 90°
        comp.rotation = round(comp.rotation / 90.0) * 90.0


def _enforce_boundary(model: BoardModel) -> None:
    """Clamp component positions so their bounding boxes stay within the board.
    
    Connectors are treated as fixed - they stay where placed on perimeter.
    """
    board = model.board
    for comp in model.components:
        # Skip fixed components AND connectors (connectors stay on perimeter)
        if comp.is_fixed or getattr(comp, 'component_type', '') == "connector":
            continue
        half_w = comp.effective_width / 2.0
        half_h = comp.effective_height / 2.0

        comp.x = max(board.x_min + half_w, min(comp.x, board.x_max - half_w))
        comp.y = max(board.y_min + half_h, min(comp.y, board.y_max - half_h))


def _is_connector(comp: Component) -> bool:
    """Check if a component is a connector."""
    return getattr(comp, 'component_type', '') == "connector"


def _resolve_overlaps(
    model: BoardModel,
    max_iterations: int,
    push_strength: float,
    grid_mm: float,
    verbose: bool,
) -> None:
    """Iteratively resolve component overlaps by pushing components apart.

    Uses a force-directed push-apart strategy: for each overlapping pair,
    compute the overlap direction and push both components away from each other.
    Increases push_strength adaptively if progress stalls.
    
    Connectors are treated as fixed - they stay on perimeter and only interior
    components get pushed away from them.
    """
    prev_overlap_count = float("inf")
    stall_iterations = 0
    adaptive_strength = push_strength

    for iteration in range(max_iterations):
        overlap_count = 0
        components = list(model.components)

        # Sort by position for deterministic resolution
        components.sort(key=lambda c: (c.x, c.y))

        for i, c1 in enumerate(components):
            for c2 in components[i + 1:]:
                if not c1.overlaps(c2):
                    continue

                # Cannot resolve if both are fixed or both are connectors
                c1_fixed = c1.is_fixed or _is_connector(c1)
                c2_fixed = c2.is_fixed or _is_connector(c2)
                if c1_fixed and c2_fixed:
                    continue

                overlap_count += 1
                
                # If one is a connector, only move the non-connector
                if c1_fixed:
                    _push_apart_one(c2, c1, adaptive_strength, grid_mm)
                elif c2_fixed:
                    _push_apart_one(c1, c2, adaptive_strength, grid_mm)
                else:
                    _push_apart(c1, c2, adaptive_strength, grid_mm)

        # Keep everything inside board after each sweep.
        _enforce_boundary(model)

        # Detect stalling and increase push strength
        if overlap_count >= prev_overlap_count:
            stall_iterations += 1
            if stall_iterations >= 10:
                adaptive_strength = min(1.5, adaptive_strength * 1.2)
                stall_iterations = 0
        else:
            stall_iterations = 0

        prev_overlap_count = overlap_count

        if overlap_count == 0:
            if verbose:
                print(f"  Overlap resolution converged in {iteration + 1} iterations")
            break
    else:
        if verbose:
            print(f"  Overlap resolution: max iterations ({max_iterations}) reached, "
                  f"{_count_overlaps(model)} overlaps remaining")


def _push_apart_one(movable: Component, fixed: Component, strength: float, grid_mm: float) -> None:
    """Push the movable component away from a fixed component (connector).
    
    Args:
        movable: Component to move (interior component)
        fixed: Fixed component to move away from (connector on perimeter)
        strength: Multiplier for push distance
        grid_mm: Minimum movement (1.5x grid unit)
    """
    # Compute overlap on each axis
    ax1, ay1, ax2, ay2 = movable.bbox
    bx1, by1, bx2, by2 = fixed.bbox

    overlap_x = min(ax2, bx2) - max(ax1, bx1)
    overlap_y = min(ay2, by2) - max(ay1, by1)

    if overlap_x <= 0 or overlap_y <= 0:
        return  # No actual overlap

    # Ensure minimum push distance = 1.5x grid unit to guarantee progress
    push_mm = max(grid_mm * 1.5, 0.15)

    # Push along axis of minimum separation for minimal displacement
    if overlap_x <= overlap_y:
        # Push along X axis
        push_x = max(overlap_x * strength, push_mm)
        push_y = 0
    else:
        # Push along Y axis
        push_x = 0
        push_y = max(overlap_y * strength, push_mm)

    # Determine push direction: move movable away from fixed
    dx = 1 if movable.x >= fixed.x else -1
    dy = 1 if movable.y >= fixed.y else -1

    # Apply push to movable component only
    movable.x += dx * push_x
    movable.y += dy * push_y


def _push_apart(c1: Component, c2: Component, strength: float, grid_mm: float) -> None:
    """Push two overlapping components apart along axis of minimum separation.

    The direction is chosen to minimize displacement (push along axis
    where overlap is smallest). For severe overlaps, pushes on both axes.

    Args:
        c1, c2: Overlapping components
        strength: Multiplier for push distance
        grid_mm: Minimum movement (1.5x grid unit)
    """
    # Compute overlap on each axis
    ax1, ay1, ax2, ay2 = c1.bbox
    bx1, by1, bx2, by2 = c2.bbox

    overlap_x = min(ax2, bx2) - max(ax1, bx1)
    overlap_y = min(ay2, by2) - max(ay1, by1)

    if overlap_x <= 0 or overlap_y <= 0:
        return  # No actual overlap

    # For severe overlaps (>70% of smaller dimension), push on both axes
    c1_min = min(c1.effective_width, c1.effective_height)
    c2_min = min(c2.effective_width, c2.effective_height)
    severity_threshold = 0.7 * min(c1_min, c2_min)

    push_both = (overlap_x > severity_threshold) or (overlap_y > severity_threshold)

    # Ensure minimum push distance = 1.5x grid unit to guarantee progress
    push_mm = max(grid_mm * 1.5, 0.15)

    # Calculate push distances
    if push_both:
        # Push along diagonal (both axes) for severe overlap
        push_x = max(overlap_x * strength * 0.5, push_mm)
        push_y = max(overlap_y * strength * 0.5, push_mm)
    elif overlap_x <= overlap_y:
        # Push along X axis (smaller overlap)
        push_x = max(overlap_x * strength, push_mm)
        push_y = 0
    else:
        # Push along Y axis (smaller overlap)
        push_x = 0
        push_y = max(overlap_y * strength, push_mm)

    # Determine push direction based on relative positions
    dx = 1 if c2.x >= c1.x else -1
    dy = 1 if c2.y >= c1.y else -1

    # Apply push
    if c1.is_fixed and not c2.is_fixed:
        c2.x += dx * push_x
        c2.y += dy * push_y
    elif c2.is_fixed and not c1.is_fixed:
        c1.x -= dx * push_x
        c1.y -= dy * push_y
    else:
        # Split push between both components
        c1.x -= dx * push_x / 2.0
        c1.y -= dy * push_y / 2.0
        c2.x += dx * push_x / 2.0
        c2.y += dy * push_y / 2.0


def _count_overlaps(model: BoardModel) -> int:
    """Count overlapping component pairs."""
    count = 0
    for i, c1 in enumerate(model.components):
        for c2 in model.components[i + 1:]:
            if c1.overlaps(c2):
                count += 1
    return count


def _count_oob(model: BoardModel) -> int:
    """Count out-of-bounds components."""
    count = 0
    board = model.board
    for comp in model.components:
        x1, y1, x2, y2 = comp.bbox
        if x1 < board.x_min or x2 > board.x_max or y1 < board.y_min or y2 > board.y_max:
            count += 1
    return count
