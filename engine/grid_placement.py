"""Grid placement algorithm for Phase 1.

Implements a deterministic initial placement:
1. Use net clustering to seed component groups
2. Place groups into board grid regions
3. Arrange components within each group using grid layout

This provides a reasonable starting point that can later be refined
by force-directed or SA optimization (Phase 2+).
"""

from __future__ import annotations

import math
from typing import Optional

from models.board_model import BoardModel, Component
from engine.net_clustering import compute_seed_positions, assign_cluster_positions, cluster_components


def grid_place(
    model: BoardModel,
    margin: float = 5.0,
    spacing_factor: float = 1.3,
    sort_by: str = "connectivity",
) -> BoardModel:
    """Perform grid-based initial placement.

    Args:
        model: Board model with unplaced or scattered components
        margin: Margin from board edges in mm
        spacing_factor: Multiplier for component spacing (1.0 = touching, 1.5 = 50% gap)
        sort_by: How to sort components within clusters:
                 "connectivity" - by net connectivity degree
                 "size" - by bounding box area
                 "ref" - alphabetical by reference

    Returns:
        BoardModel with updated component positions
    """
    # Step 1: Compute seed positions from net clustering
    seed_positions = compute_seed_positions(model)

    # Step 2: Apply seed positions to movable components
    comp_map = {c.ref: c for c in model.components}
    for ref, (x, y) in seed_positions.items():
        comp = comp_map.get(ref)
        if comp and not comp.is_fixed:
            comp.x = x
            comp.y = y

    # Step 3: Refine placement with proper grid spacing within clusters
    _refine_cluster_placement(model, margin, spacing_factor, sort_by)

    return model


def _refine_cluster_placement(
    model: BoardModel,
    margin: float,
    spacing_factor: float,
    sort_by: str,
) -> None:
    """Refine component positions within each cluster for proper spacing."""
    clusters = cluster_components(model)
    board = model.board
    comp_map = {c.ref: c for c in model.components}

    # Calculate usable board area
    usable_x_min = board.x_min + margin
    usable_y_min = board.y_min + margin
    usable_x_max = board.x_max - margin
    usable_y_max = board.y_max - margin

    # Calculate grid for clusters
    n_clusters = len(clusters)
    if n_clusters == 0:
        return

    cols = max(1, int(math.ceil(math.sqrt(n_clusters))))
    rows = max(1, int(math.ceil(n_clusters / cols)))
    region_w = (usable_x_max - usable_x_min) / cols
    region_h = (usable_y_max - usable_y_min) / rows

    for idx, cluster_refs in enumerate(clusters):
        col = idx % cols
        row = idx // cols

        region_x_min = usable_x_min + col * region_w
        region_y_min = usable_y_min + row * region_h
        region_x_max = region_x_min + region_w
        region_y_max = region_y_min + region_h

        # Sort components within cluster
        components = [comp_map[r] for r in cluster_refs if r in comp_map]

        if sort_by == "connectivity":
            components.sort(key=lambda c: len(c.nets), reverse=True)
        elif sort_by == "size":
            components.sort(key=lambda c: c.width * c.height, reverse=True)
        else:
            components.sort(key=lambda c: c.ref)

        # Place in grid within region
        _place_in_region(components, region_x_min, region_y_min, region_x_max, region_y_max, spacing_factor)


def _place_in_region(
    components: list[Component],
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    spacing_factor: float,
) -> None:
    """Arrange components in a grid within a rectangular region."""
    n = len(components)
    if n == 0:
        return

    # Calculate grid dimensions
    cols = max(1, int(math.ceil(math.sqrt(n))))
    rows = max(1, int(math.ceil(n / cols)))

    # Calculate available space and required spacing
    region_w = x_max - x_min
    region_h = y_max - y_min

    # Determine max component sizes for spacing
    if components:
        max_w = max(c.effective_width for c in components)
        max_h = max(c.effective_height for c in components)
    else:
        max_w, max_h = 2.0, 2.0

    # Spacing with factor
    spacing_x = max(max_w * spacing_factor, region_w / max(cols, 1))
    spacing_y = max(max_h * spacing_factor, region_h / max(rows, 1))

    # Center the grid in the region
    total_grid_w = (cols - 1) * spacing_x
    total_grid_h = (rows - 1) * spacing_y
    start_x = x_min + (region_w - total_grid_w) / 2.0
    start_y = y_min + (region_h - total_grid_h) / 2.0

    # Place each component
    for i, comp in enumerate(components):
        col = i % cols
        row = i // cols
        comp.x = start_x + col * spacing_x
        comp.y = start_y + row * spacing_y
        # Reset rotation to 0 for initial placement
        comp.rotation = 0.0


def edge_aware_grid_place(
    model: BoardModel,
    margin: float = 5.0,
    spacing_factor: float = 1.3,
) -> BoardModel:
    """Grid placement with edge-aware connector positioning.

    Places connectors near board edges and other components
    in the interior using net clustering.
    """
    comp_map = {c.ref: c for c in model.components}

    # First, do standard grid placement
    grid_place(model, margin=margin, spacing_factor=spacing_factor)

    # Then move connectors to board edges
    board = model.board
    for comp in model.components:
        if comp.component_type == "connector" and not comp.is_fixed:
            _move_to_nearest_edge(comp, board, margin)

    return model


def _move_to_nearest_edge(comp: Component, board, margin: float) -> None:
    """Move a connector component to the nearest board edge."""
    cx, cy = board.center

    # Determine which edge is closest to current position
    dx = comp.x - cx
    dy = comp.y - cy

    # Offset from edge
    edge_offset = margin + comp.effective_width / 2.0

    if abs(dx) > abs(dy):
        # Left or right edge
        if dx > 0:
            comp.x = board.x_max - edge_offset
        else:
            comp.x = board.x_min + edge_offset
        comp.y = max(board.y_min + margin + comp.effective_height / 2.0,
                     min(comp.y, board.y_max - margin - comp.effective_height / 2.0))
    else:
        # Top or bottom edge
        if dy > 0:
            comp.y = board.y_max - edge_offset
        else:
            comp.y = board.y_min + edge_offset
        comp.x = max(board.x_min + margin + comp.effective_width / 2.0,
                     min(comp.x, board.x_max - margin - comp.effective_width / 2.0))
