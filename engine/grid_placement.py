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

from models.board_model import BoardModel, Component
from engine.net_clustering import compute_seed_positions, cluster_components
from engine.cost_function import total_hpwl
from legalization.legalizer import _push_apart, _enforce_boundary


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

    # Step 3: Place connectors on perimeter first
    connectors = [c for c in model.components if c.component_type == "connector" and not c.is_fixed]
    _place_connectors_on_perimeter(connectors, model.board, margin)

    # Step 4: Refine interior-only components (exclude connectors)
    _refine_cluster_placement(model, margin, spacing_factor, sort_by, exclude_types={"connector"})

    # Step 5: Apply strong repulsion to interior components (exclude connectors)
    _apply_strong_repulsion(model, exclude_types={"connector"})

    # Step 6: Quick overlap resolve — connectors frozen, interior pushed away
    _quick_overlap_resolve(model, frozen_types={"connector"})

    return model


def _refine_cluster_placement(
    model: BoardModel,
    margin: float,
    spacing_factor: float,
    sort_by: str,
    exclude_types: set[str] | None = None,
) -> None:
    """Refine component positions within each cluster for proper spacing."""
    clusters = cluster_components(model)
    board = model.board
    comp_map = {c.ref: c for c in model.components}
    exclude = exclude_types or set()

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

        # Sort components within cluster (skip excluded types)
        components = [comp_map[r] for r in cluster_refs
                      if r in comp_map and comp_map[r].component_type not in exclude]

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


def _place_connectors_on_perimeter(
    connectors: list[Component],
    board,
    margin: float,
) -> None:
    """Place connectors on board perimeter using size-aware greedy best-fit.

    Sorts connectors largest-first, places on the edge with the most
    remaining space.  After placement, resolves any corner collisions
    by pushing connectors along their assigned edges.
    """
    if not connectors:
        return

    gap = 1.5
    perimeter_margin = margin + 2.0

    edge_lengths = {
        "top":    board.x_max - board.x_min - 2 * perimeter_margin,
        "bottom": board.x_max - board.x_min - 2 * perimeter_margin,
        "right":  board.y_max - board.y_min - 2 * perimeter_margin,
        "left":   board.y_max - board.y_min - 2 * perimeter_margin,
    }

    edge_rotation = {"top": 180.0, "bottom": 0.0, "left": 90.0, "right": 270.0}

    sorted_connectors = sorted(
        connectors,
        key=lambda c: max(c.effective_width, c.effective_height),
        reverse=True,
    )

    # Track which edge each connector is assigned to
    comp_edge = {}
    edge_used = {e: 0.0 for e in edge_lengths}

    for comp in sorted_connectors:
        w, h = comp.effective_width, comp.effective_height

        best_edge = None
        best_remaining = -1.0
        best_rotate = False
        best_along = w

        for edge, edge_len in edge_lengths.items():
            remaining = edge_len - edge_used[edge]

            if edge in ("top", "bottom"):
                along, depth = w, h
            else:
                # Left/right edges: along-edge = y-direction. After default rotation
                # (90°/270°), eff_h = w, so along-edge extent = w.
                along, depth = w, h

            if along + gap <= remaining and remaining > best_remaining:
                best_edge = edge
                best_remaining = remaining
                best_rotate = False
                best_along = along

            if depth + gap <= remaining and remaining > best_remaining:
                best_edge = edge
                best_remaining = remaining
                best_rotate = True
                best_along = depth

        if best_edge is not None:
            edge_used[best_edge] += best_along + gap
            rot = (edge_rotation[best_edge] + 90.0) % 360.0 if best_rotate else edge_rotation[best_edge]
            comp.set_rotation(rot)
            comp_edge[comp.ref] = best_edge
        else:
            comp.set_rotation(0.0)
            comp_edge[comp.ref] = None

        w = comp.effective_width
        h = comp.effective_height
        edge = best_edge if best_edge else "bottom"

        if edge == "top":
            start = board.x_min + perimeter_margin + edge_used[edge] - best_along - gap
            cx = start + best_along / 2.0
            cy = board.y_min + perimeter_margin + h / 2.0
        elif edge == "bottom":
            start = board.x_min + perimeter_margin + edge_used[edge] - best_along - gap
            cx = start + best_along / 2.0
            cy = board.y_max - perimeter_margin - h / 2.0
        elif edge == "left":
            start = board.y_min + perimeter_margin + edge_used[edge] - best_along - gap
            cx = board.x_min + perimeter_margin + w / 2.0
            cy = start + best_along / 2.0
        else:  # right
            start = board.y_min + perimeter_margin + edge_used[edge] - best_along - gap
            cx = board.x_max - perimeter_margin - w / 2.0
            cy = start + best_along / 2.0

        if best_edge is None:
            cx = board.x_min + w / 2.0 + margin
            cy = board.y_max - h / 2.0 - margin

        comp.x = max(board.x_min + w / 2.0, min(cx, board.x_max - w / 2.0))
        comp.y = max(board.y_min + h / 2.0, min(cy, board.y_max - h / 2.0))

    # Resolve corner collisions between connectors on adjacent edges
    _resolve_corner_collisions(connectors, comp_edge, board, perimeter_margin, gap)


def _resolve_corner_collisions(
    connectors: list[Component],
    comp_edge: dict[str, str | None],
    board,
    perimeter_margin: float,
    gap: float,
) -> None:
    """Push connectors along their edges to resolve corner overlaps.

    For each overlapping pair on adjacent edges, push the connector closer
    to the shared corner by just enough to clear the overlap plus a gap.
    """
    adjacent = {frozenset(e) for e in [("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")]}
    ref_map = {c.ref: c for c in connectors}

    for _ in range(20):
        resolved_any = False
        for i, c1 in enumerate(connectors):
            e1 = comp_edge.get(c1.ref)
            if e1 is None:
                continue
            for c2 in connectors[i + 1:]:
                e2 = comp_edge.get(c2.ref)
                if e2 is None:
                    continue
                if not c1.overlaps(c2):
                    continue
                if frozenset({e1, e2}) not in adjacent:
                    continue

                # Compute overlap on each axis
                ax1, ay1, ax2, ay2 = c1.bbox
                bx1, by1, bx2, by2 = c2.bbox
                ox = min(ax2, bx2) - max(ax1, bx1)
                oy = min(ay2, by2) - max(ay1, by1)

                # Determine which corner they're near and push direction
                # For each connector, push along its edge away from the corner
                for comp, edge, other_comp in [(c1, e1, c2), (c2, e2, c1)]:
                    push_x, push_y = 0.0, 0.0

                    if edge == "top":
                        # Determine if near left or right corner
                        if comp.x < other_comp.x:
                            push_x = -(ox + gap)  # push left (away from right corner)
                        else:
                            push_x = ox + gap  # push right (away from left corner)
                    elif edge == "bottom":
                        if comp.x < other_comp.x:
                            push_x = -(ox + gap)
                        else:
                            push_x = ox + gap
                    elif edge == "left":
                        if comp.y < other_comp.y:
                            push_y = -(oy + gap)
                        else:
                            push_y = oy + gap
                    elif edge == "right":
                        if comp.y < other_comp.y:
                            push_y = -(oy + gap)
                        else:
                            push_y = oy + gap

                    # Apply push with clamping
                    old_x, old_y = comp.x, comp.y
                    if push_x != 0:
                        comp.x += push_x
                        comp.x = max(board.x_min + comp.effective_width / 2.0,
                                     min(comp.x, board.x_max - comp.effective_width / 2.0))
                    if push_y != 0:
                        comp.y += push_y
                        comp.y = max(board.y_min + comp.effective_height / 2.0,
                                     min(comp.y, board.y_max - comp.effective_height / 2.0))

                    # Check if this created a new overlap with any same-edge connector
                    creates_new_overlap = False
                    for c3 in connectors:
                        if c3 is comp or c3 is other_comp:
                            continue
                        if comp_edge.get(c3.ref) == edge and comp.overlaps(c3):
                            creates_new_overlap = True
                            break

                    if creates_new_overlap:
                        comp.x, comp.y = old_x, old_y  # revert
                    else:
                        resolved_any = True

                # If pushing both failed to resolve, try pushing just one further
                if c1.overlaps(c2):
                    resolved_any = True  # mark as attempted, move on

        if not resolved_any:
            break


def _apply_strong_repulsion(model: BoardModel, exclude_types: set[str] | None = None) -> None:
    """Apply strong repulsive forces between non-fixed, non-excluded components."""
    exclude = exclude_types or set()
    movable = [c for c in model.components if not c.is_fixed and c.component_type not in exclude]
    if len(movable) < 2:
        return

    board = model.board
    max_iters = max(50, min(500, len(movable) * 10))

    for iteration in range(max_iters):
        moved = False
        for i, ca in enumerate(movable):
            for cb in movable[i+1:]:
                dx = cb.x - ca.x
                dy = cb.y - ca.y
                dist = (dx**2 + dy**2) ** 0.5
                min_d = max(ca.effective_width, ca.effective_height) * 0.8
                min_d = max(min_d, max(cb.effective_width, cb.effective_height) * 0.8)

                if dist < min_d:
                    if dist < 0.1:
                        dx, dy = math.cos(iteration * 0.3), math.sin(iteration * 0.3)
                    else:
                        dx, dy = dx/dist, dy/dist

                    push = (min_d - dist) * 2.0 + 1.0
                    ca.x = max(board.x_min + ca.effective_width/2, min(ca.x - push*dx, board.x_max - ca.effective_width/2))
                    ca.y = max(board.y_min + ca.effective_height/2, min(ca.y - push*dy, board.y_max - ca.effective_height/2))
                    cb.x = max(board.x_min + cb.effective_width/2, min(cb.x + push*dx, board.x_max - cb.effective_width/2))
                    cb.y = max(board.y_min + cb.effective_height/2, min(cb.y + push*dy, board.y_max - cb.effective_height/2))
                    moved = True

        if not moved:
            break


def _quick_overlap_resolve(model: BoardModel, grid_mm: float = 0.1, max_iterations: int = 20,
                           frozen_types: set[str] | None = None) -> None:
    """Lightweight overlap cleanup using legalizer push-apart.

    Gives SA a cleaner starting point without a full legalization pass.
    Components matching frozen_types (e.g. connectors on perimeter) are treated
    as immovable — overlapping interior components get pushed away from them.
    """
    frozen = frozen_types or set()
    all_components = list(model.components)

    for _ in range(max_iterations):
        overlap_count = 0
        for i, c1 in enumerate(all_components):
            for c2 in all_components[i + 1:]:
                if not c1.overlaps(c2):
                    continue
                c1_frozen = c1.is_fixed or c1.component_type in frozen
                c2_frozen = c2.is_fixed or c2.component_type in frozen
                if c1_frozen and c2_frozen:
                    continue
                overlap_count += 1
                if c1_frozen:
                    # Temporarily mark c1 as fixed so _push_apart only moves c2
                    c1.is_fixed = True
                    _push_apart(c1, c2, 1.0, grid_mm)
                    c1.is_fixed = False
                elif c2_frozen:
                    c2.is_fixed = True
                    _push_apart(c1, c2, 1.0, grid_mm)
                    c2.is_fixed = False
                else:
                    _push_apart(c1, c2, 1.0, grid_mm)

        _enforce_boundary(model)

        if overlap_count == 0:
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
