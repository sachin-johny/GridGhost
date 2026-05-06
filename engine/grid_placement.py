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
from engine.cost_function import total_hpwl


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
    
    # Step 4: Force connectors to perimeter and apply strong repulsion to interior
    connectors = [c for c in model.components if c.component_type == "connector" and not c.is_fixed]
    _place_connectors_on_perimeter(connectors, model.board, margin)
    _apply_strong_repulsion(model)

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

    # Determine max component sizes for spacing and shrink the grid until each
    # cell can safely contain the largest component in the region.
    max_w = max(c.effective_width for c in components)
    max_h = max(c.effective_height for c in components)

    while cols > 1 and (region_w / cols) < (max_w * spacing_factor):
        cols -= 1
        rows = max(1, int(math.ceil(n / cols)))
    while rows > 1 and (region_h / rows) < (max_h * spacing_factor):
        rows -= 1
        cols = max(1, int(math.ceil(n / rows)))

    cell_w = region_w / cols
    cell_h = region_h / rows

    # Place each component
    for i, comp in enumerate(components):
        col = i % cols
        row = i // cols
        comp.x = x_min + (col + 0.5) * cell_w
        comp.y = y_min + (row + 0.5) * cell_h
        comp.x = max(x_min + comp.effective_width / 2.0, min(comp.x, x_max - comp.effective_width / 2.0))
        comp.y = max(y_min + comp.effective_height / 2.0, min(comp.y, y_max - comp.effective_height / 2.0))


def edge_aware_grid_place(
    model: BoardModel,
    margin: float = 5.0,
    spacing_factor: float = 1.3,
) -> BoardModel:
    """Grid placement with edge-aware connector positioning.

    Places connectors around the perimeter first, then spreads the
    remaining components across the interior in one balanced grid.
    """
    board = model.board
    connectors = [c for c in model.components if c.component_type == "connector"]
    interior_components = [c for c in model.components if c.component_type != "connector"]

    if connectors:
        _place_connectors_on_perimeter(connectors, board, margin)

    if interior_components:
        _place_interior_components(interior_components, board, margin, spacing_factor)

    return model


def _place_interior_components(
    components: list[Component],
    board,
    margin: float,
    spacing_factor: float,
) -> None:
    """Pack non-connector components into a single balanced interior grid."""
    if not components:
        return

    inner_margin = margin + 2.0
    x_min = board.x_min + inner_margin
    y_min = board.y_min + inner_margin
    x_max = board.x_max - inner_margin
    y_max = board.y_max - inner_margin

    if x_max <= x_min or y_max <= y_min:
        return

    # Spread parts according to board aspect ratio so the grid uses the full
    # interior area instead of collapsing into a single dense cluster.
    n = len(components)
    region_w = x_max - x_min
    region_h = y_max - y_min
    cols = max(1, min(n, int(round(math.sqrt(n * (region_w / max(region_h, 1e-9)))))))
    rows = max(1, int(math.ceil(n / cols)))

    # Sort larger parts first so they claim more central cells.
    components.sort(key=lambda c: (c.effective_width * c.effective_height, len(c.nets)), reverse=True)

    cell_w = region_w / cols
    cell_h = region_h / rows

    # Keep a little slack around each part. If the cell is too small for the
    # part, we still center it but do not shrink the whole layout into a line.
    for index, comp in enumerate(components):
        col = index % cols
        row = index // cols
        center_x = x_min + (col + 0.5) * cell_w
        center_y = y_min + (row + 0.5) * cell_h

        # Apply a small alternating offset to prevent long straight rows.
        wiggle = min(cell_w, cell_h) * 0.12 * spacing_factor
        if row % 2 == 1:
            center_x += wiggle if col % 2 == 0 else -wiggle
        else:
            center_y += wiggle if col % 2 == 0 else -wiggle

        comp.x = max(x_min + comp.effective_width / 2.0, min(center_x, x_max - comp.effective_width / 2.0))
        comp.y = max(y_min + comp.effective_height / 2.0, min(center_y, y_max - comp.effective_height / 2.0))


def _place_connectors_on_perimeter(
    connectors: list[Component],
    board,
    margin: float,
) -> None:
    """Place connectors evenly spaced around the board perimeter."""
    if not connectors:
        return
    
    perimeter_margin = margin + 2.0  # Extra margin for connector placement
    
    # Calculate perimeter positions (in order: top, right, bottom, left)
    perimeter_points = []
    
    # Top edge (left to right)
    top_y = board.y_min + perimeter_margin
    spacing = (board.x_max - board.x_min - 2 * perimeter_margin) / max(1, len(connectors))
    for i in range(len(connectors)):
        x = board.x_min + perimeter_margin + i * spacing
        perimeter_points.append((x, top_y, "top"))
    
    # Right edge (top to bottom)
    right_x = board.x_max - perimeter_margin
    spacing = (board.y_max - board.y_min - 2 * perimeter_margin) / max(1, len(connectors))
    for i in range(len(connectors)):
        y = board.y_min + perimeter_margin + i * spacing
        perimeter_points.append((right_x, y, "right"))
    
    # Bottom edge (right to left)
    bottom_y = board.y_max - perimeter_margin
    spacing = (board.x_max - board.x_min - 2 * perimeter_margin) / max(1, len(connectors))
    for i in range(len(connectors)):
        x = board.x_max - perimeter_margin - i * spacing
        perimeter_points.append((x, bottom_y, "bottom"))
    
    # Left edge (bottom to top)
    left_x = board.x_min + perimeter_margin
    spacing = (board.y_max - board.y_min - 2 * perimeter_margin) / max(1, len(connectors))
    for i in range(len(connectors)):
        y = board.y_max - perimeter_margin - i * spacing
        perimeter_points.append((left_x, y, "left"))
    
    # Distribute connectors around the perimeter
    for i, connector in enumerate(connectors):
        # Cycle through perimeter points if more connectors than calculated points
        perimeter_idx = i % len(perimeter_points)
        x, y, edge = perimeter_points[perimeter_idx]
        
        # Clamp to valid range
        x = max(board.x_min + connector.effective_width / 2.0,
                min(x, board.x_max - connector.effective_width / 2.0))
        y = max(board.y_min + connector.effective_height / 2.0,
                min(y, board.y_max - connector.effective_height / 2.0))
        
        connector.x = x
        connector.y = y
        if edge == "left":
            connector.set_rotation(90.0)
        elif edge == "right":
            connector.set_rotation(270.0)
        elif edge == "top":
            connector.set_rotation(180.0)
        else:
            connector.set_rotation(0.0)


def _orient_connector_outward(comp: Component, board) -> None:
    pass  # Removed (now handled in _place_connectors_on_perimeter)


def _apply_strong_repulsion(model: BoardModel) -> None:
    """Apply strong repulsive forces between interior components (R, C, U)."""
    interior = [c for c in model.components if c.component_type in ("resistor", "capacitor", "ic") and not c.is_fixed]
    if len(interior) < 2:
        return

    min_distances = {"resistor": 10.0, "capacitor": 8.0, "ic": 12.0}
    board = model.board

    for _ in range(150):
        moved = False
        for i, ca in enumerate(interior):
            for cb in interior[i+1:]:
                dx = cb.x - ca.x
                dy = cb.y - ca.y
                dist = (dx**2 + dy**2) ** 0.5
                min_d = max(min_distances.get(ca.component_type, 10), min_distances.get(cb.component_type, 10))

                if dist < min_d:
                    if dist < 0.1:
                        dx, dy = math.cos(_ * 0.3), math.sin(_ * 0.3)
                    else:
                        dx, dy = dx/dist, dy/dist

                    push = (min_d - dist) * 2.0 + 1.0  # Strong push
                    ca.x = max(board.x_min + ca.effective_width/2, min(ca.x - push*dx, board.x_max - ca.effective_width/2))
                    ca.y = max(board.y_min + ca.effective_height/2, min(ca.y - push*dy, board.y_max - ca.effective_height/2))
                    cb.x = max(board.x_min + cb.effective_width/2, min(cb.x + push*dx, board.x_max - cb.effective_width/2))
                    cb.y = max(board.y_min + cb.effective_height/2, min(cb.y + push*dy, board.y_max - cb.effective_height/2))
                    moved = True

        if not moved:
            break


def force_directed_place(
    model: BoardModel,
    margin: float = 5.0,
    iterations: int = 100,
    k_attract: float = 0.01,  # Attractive force coefficient (HPWL gradient)
    k_repel: float = 500.0,   # Repulsive force coefficient
    min_spacing: float = 2.0,   # Minimum component spacing in mm
    dt: float = 0.5,            # Time step for simulation
    verbose: bool = False,
) -> BoardModel:
    """Force-directed placement with balanced attractive and repulsive forces.

    Attractive forces: Derived from HPWL gradient - pulls connected components together
    Repulsive forces: Inverse distance - pushes all components apart
    Cooling schedule: Reduces dt over iterations to stabilize

    Args:
        model: Board model to optimize in-place
        margin: Board edge margin in mm
        iterations: Number of simulation iterations
        k_attract: Attractive force coefficient (lower = gentler)
        k_repel: Repulsive force coefficient (higher = stronger spreading)
        min_spacing: Minimum spacing between components
        dt: Time step (controls movement per iteration)
        verbose: Print progress

    Returns:
        BoardModel with force-directed placement
    """
    movable = [c for c in model.components if not c.is_fixed]
    if not movable:
        return model

    board = model.board
    comp_map = {c.ref: c for c in model.components}

    # Build adjacency list for efficient attractive force calculation
    adjacency = {}
    for net in model.nets:
        refs = list(net.component_refs)
        for i, r1 in enumerate(refs):
            for r2 in refs[i+1:]:
                if r1 in comp_map and r2 in comp_map:
                    adjacency.setdefault(r1, []).append(r2)
                    adjacency.setdefault(r2, []).append(r1)

    # Initial positions from shelf packing or current placement
    # (ensure components start well-separated)
    shelf_packing_place(model, margin, spacing=min_spacing)

    prev_hpwl = float('inf')

    for iter_num in range(iterations):
        forces = {c.ref: (0.0, 0.0) for c in movable}

        # Cooling schedule - reduce movement over time
        current_dt = dt * (1.0 - iter_num / iterations) * 0.8 + dt * 0.2

        # Calculate attractive forces (HPWL gradient)
        for comp in movable:
            fx, fy = 0.0, 0.0
            if comp.ref in adjacency:
                for neighbor_ref in adjacency[comp.ref]:
                    neighbor = comp_map.get(neighbor_ref)
                    if neighbor and neighbor.is_fixed:
                        # Attractive to fixed components
                        dx = neighbor.x - comp.x
                        dy = neighbor.y - comp.y
                        dist = math.sqrt(dx*dx + dy*dy) + 0.1
                        fx += k_attract * dx / dist
                        fy += k_attract * dy / dist
            forces[comp.ref] = (forces[comp.ref][0] + fx, forces[comp.ref][1] + fy)

        # Calculate repulsive forces (all pairs)
        for i, c1 in enumerate(movable):
            fx, fy = forces[c1.ref]
            for c2 in movable[i+1:]:
                dx = c2.x - c1.x
                dy = c2.y - c1.y
                dist_sq = dx*dx + dy*dy
                dist = math.sqrt(dist_sq)

                # Repulsion only applies if close
                if dist < min_spacing * 3:
                    if dist < 0.1:
                        dx, dy = 1.0, 0.0  # Avoid division by zero
                        dist = 1.0

                    # Inverse distance repulsion
                    force = k_repel / (dist_sq + 0.01)
                    fx -= force * dx / dist
                    fy -= force * dy / dist

                    # Apply equal and opposite force to c2
                    c2_fx, c2_fy = forces[c2.ref]
                    forces[c2.ref] = (c2_fx + force * dx / dist, c2_fy + force * dy / dist)

            forces[c1.ref] = (fx, fy)

        # Apply forces with time step
        max_move = 0.0
        for comp in movable:
            fx, fy = forces[comp.ref]
            new_x = comp.x + fx * current_dt
            new_y = comp.y + fy * current_dt

            # Clamp to board bounds
            new_x = max(board.x_min + margin + comp.effective_width/2,
                       min(new_x, board.x_max - margin - comp.effective_width/2))
            new_y = max(board.y_min + margin + comp.effective_height/2,
                       min(new_y, board.y_max - margin - comp.effective_height/2))

            move = math.sqrt((new_x - comp.x)**2 + (new_y - comp.y)**2)
            max_move = max(max_move, move)

            comp.x = new_x
            comp.y = new_y

        # Check convergence
        current_hpwl = total_hpwl(model)
        overlap_count = sum(1 for i, c1 in enumerate(movable)
                           for c2 in movable[i+1:] if c1.overlaps(c2))

        if verbose and iter_num % 10 == 0:
            print(f"  Iter {iter_num}: HPWL={current_hpwl:.1f}, overlaps={overlap_count}, max_move={max_move:.2f}")

        # Stop if converged (HPWL not improving and no significant movement)
        if iter_num > 10 and max_move < 0.05:
            if verbose:
                print(f"  Converged at iteration {iter_num}")
            break

    return model


__all__ = [
    "grid_place",
    "edge_aware_grid_place",
    "shelf_packing_place",
    "force_directed_place",
]


def shelf_packing_place(
    model: BoardModel,
    margin: float = 5.0,
    spacing: float = 0.5,
    sort_by: str = "size",
) -> BoardModel:
    """Shelf-packing algorithm for non-overlapping initial placement.

    Uses a shelf-packing strategy where components are sorted by size
    and placed in rows (shelves) across the board, ensuring no overlaps.

    Args:
        model: Board model with unplaced components
        margin: Margin from board edges in mm
        spacing: Minimum spacing between components in mm
        sort_by: How to sort components: "size", "connectivity", or "ref"

    Returns:
        BoardModel with updated component positions
    """
    board = model.board
    connectors = [c for c in model.components if c.component_type == "connector"]
    interior = [c for c in model.components if c.component_type != "connector"]

    # Place connectors on perimeter first
    if connectors:
        _place_connectors_on_perimeter(connectors, board, margin)

    # Sort interior components
    if sort_by == "size":
        interior.sort(key=lambda c: (c.effective_width * c.effective_height), reverse=True)
    elif sort_by == "connectivity":
        interior.sort(key=lambda c: len(c.nets), reverse=True)
    else:
        interior.sort(key=lambda c: c.ref)

    # Determine usable interior area (accounting for connector perimeter zone)
    connector_zone = max([c.effective_width for c in connectors], default=0) + spacing
    x_min = board.x_min + margin + connector_zone
    y_min = board.y_min + margin + connector_zone
    x_max = board.x_max - margin - connector_zone
    y_max = board.y_max - margin - connector_zone

    # Shelf packing
    current_x = x_min
    current_y = y_min
    current_shelf_height = 0

    for comp in interior:
        if comp.is_fixed:
            continue

        comp_w = comp.effective_width + spacing
        comp_h = comp.effective_height + spacing

        # Check if component fits in current shelf
        if current_x + comp_w > x_max:
            # Start new shelf
            current_x = x_min
            current_y += current_shelf_height
            current_shelf_height = 0

        # Check if new shelf fits in board
        if current_y + comp_h > y_max:
            # Reset to top and try to find space
            current_y = y_min
            current_x = x_min

        # Place component at current position
        comp.x = current_x + comp.effective_width / 2.0
        comp.y = current_y + comp.effective_height / 2.0

        # Advance cursor
        current_x += comp_w
        current_shelf_height = max(current_shelf_height, comp_h)

    return model
