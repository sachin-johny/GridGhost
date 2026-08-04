"""RUDY congestion estimation for PCB placement.

RUDY (Rectangular Uniform wire DensitY) estimates routing congestion
by computing the wire density contribution of each net across a uniform
grid. For each net, its bounding box contributes uniformly to all grid
cells it covers. The total congestion at each cell is the sum of all
net contributions.

This is used as an additional cost term during SA to discourage
placing components in congested areas, improving routability.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.board_model import BoardModel, Component


from engine.cost_state import _is_power_net


def compute_rudy_map(
    model: "BoardModel",
    grid_resolution: float = 2.0,
) -> tuple[list[list[float]], dict]:
    """Compute the RUDY congestion map.

    Divides the board into a uniform grid (cell size = grid_resolution mm).
    For each non-power net with >= 2 pins, computes the bounding box of all
    pin positions and distributes a unit of wire density uniformly across
    the covered grid cells. Short nets contribute more per cell than long
    nets (1/num_cells_covered), modelling the higher per-area routing
    difficulty of dense local interconnect.

    Args:
        model: BoardModel with components and nets.
        grid_resolution: Cell size in mm (default 2.0).

    Returns:
        (grid, metadata) where:
        - grid is a 2D list of floats [row][col], congestion values
        - metadata is a dict with x_offset, y_offset, cell_size, nx, ny
    """
    board = model.board
    cell_size = max(0.5, grid_resolution)  # safety: at least 0.5mm

    nx = max(1, int(math.ceil(board.width / cell_size)))
    ny = max(1, int(math.ceil(board.height / cell_size)))

    x_offset = board.x_min
    y_offset = board.y_min

    # Initialize grid with zeros
    grid = [[0.0] * nx for _ in range(ny)]

    # Build a quick ref -> component lookup
    ref_to_comp: dict[str, "Component"] = {}
    for comp in model.components:
        ref_to_comp[comp.ref] = comp

    # For each non-power net, compute bounding box of pin positions
    # and distribute wire density across covered cells
    for net in model.nets:
        if _is_power_net(net.name):
            continue

        # Collect absolute pin positions for this net
        pin_xs: list[float] = []
        pin_ys: list[float] = []
        for ref, pad_name in net.pins:
            comp = ref_to_comp.get(ref)
            if not comp:
                continue
            # Find the pad on this component
            found_pad = False
            for pad in comp.pads:
                if pad.pad_name == pad_name:
                    abs_x, abs_y = pad.absolute_pos(comp.x, comp.y, comp.rotation)
                    pin_xs.append(abs_x)
                    pin_ys.append(abs_y)
                    found_pad = True
                    break
            if not found_pad:
                # Fallback: use component center
                pin_xs.append(comp.x)
                pin_ys.append(comp.y)

        if len(pin_xs) < 2:
            continue

        # Bounding box of pins
        bbox_x_min = min(pin_xs)
        bbox_y_min = min(pin_ys)
        bbox_x_max = max(pin_xs)
        bbox_y_max = max(pin_ys)

        # Convert to grid cell indices
        col_min = max(0, int((bbox_x_min - x_offset) / cell_size))
        col_max = min(nx - 1, int((bbox_x_max - x_offset) / cell_size))
        row_min = max(0, int((bbox_y_min - y_offset) / cell_size))
        row_max = min(ny - 1, int((bbox_y_max - y_offset) / cell_size))

        num_cells = (col_max - col_min + 1) * (row_max - row_min + 1)
        if num_cells <= 0:
            continue

        # Uniform distribution: 1.0 / num_cells per cell
        contribution = 1.0 / num_cells

        for row in range(row_min, row_max + 1):
            grid_row = grid[row]
            for col in range(col_min, col_max + 1):
                grid_row[col] += contribution

    metadata = {
        'x_offset': x_offset,
        'y_offset': y_offset,
        'cell_size': cell_size,
        'nx': nx,
        'ny': ny,
    }

    return grid, metadata


def rudy_congestion_penalty(
    model: "BoardModel",
    grid_resolution: float = 2.0,
) -> tuple[float, float, float, float]:
    """Compute a scalar congestion penalty from the RUDY map.

    The penalty is based on how much the peak congestion exceeds an
    adaptive target, plus total overflow (sum of excess above target
    across all cells). The target is set as a fraction above the
    average congestion, so it adapts to the board's overall density.

    Args:
        model: BoardModel with components and nets.
        grid_resolution: Cell size in mm (default 2.0).

    Returns:
        (penalty, peak, average, overflow) where:
        - penalty: scalar congestion penalty for the cost function
        - peak: maximum congestion value in any cell
        - average: average congestion across all cells
        - overflow: sum of (cell_value - target) for cells above target
    """
    grid, _meta = compute_rudy_map(model, grid_resolution)
    ny = len(grid)
    if ny == 0:
        return 0.0, 0.0, 0.0, 0.0
    nx = len(grid[0])
    if nx == 0:
        return 0.0, 0.0, 0.0, 0.0

    total = 0.0
    peak = 0.0
    for row in grid:
        for val in row:
            total += val
            if val > peak:
                peak = val

    num_cells = nx * ny
    average = total / num_cells

    # Adaptive target: 1.5x the average congestion.
    # Boards with uniformly distributed congestion will have peak ~ average,
    # so the target provides headroom. Boards with hotspots will have
    # peak >> average, and the penalty kicks in to spread things out.
    target = average * 1.5

    # If target is very small (nearly empty board), skip
    if target < 1e-9:
        return 0.0, peak, average, 0.0

    overflow = 0.0
    for row in grid:
        for val in row:
            if val > target:
                overflow += val - target

    # Penalty: peak * (excess above target) + overflow
    # This penalises both high peak congestion and widespread overflow.
    excess = max(0.0, peak - target)
    penalty = peak * excess + overflow

    return penalty, peak, average, overflow


def rudy_gradient_for_comp(
    comp: "Component",
    model: "BoardModel",
    grid_resolution: float = 2.0,
) -> tuple[float, float]:
    """Compute a congestion gradient vector for a component.

    Finds the grid cells occupied by the component's bounding box
    and computes the direction toward lower congestion. Returns
    a unit vector pointing away from congested areas.

    This is used as a gentle force during SA translate moves to
    nudge components away from congested regions.

    Args:
        comp: The component to compute the gradient for.
        model: BoardModel with components and nets.
        grid_resolution: Cell size in mm (default 2.0).

    Returns:
        (dx, dy) unit vector pointing toward lower congestion.
        Returns (0.0, 0.0) if no gradient can be computed.
    """
    grid, meta = compute_rudy_map(model, grid_resolution)
    ny = len(grid)
    if ny == 0:
        return 0.0, 0.0
    nx = len(grid[0])
    if nx == 0:
        return 0.0, 0.0

    x_offset = meta['x_offset']
    y_offset = meta['y_offset']
    cell_size = meta['cell_size']

    # Get component bounding box
    bbox = comp.bbox
    cx_min, cy_min, cx_max, cy_max = bbox

    # Convert to grid cell indices
    col_min = max(0, int((cx_min - x_offset) / cell_size))
    col_max = min(nx - 1, int((cx_max - x_offset) / cell_size))
    row_min = max(0, int((cy_min - y_offset) / cell_size))
    row_max = min(ny - 1, int((cy_max - y_offset) / cell_size))

    # Compute gradient using central differences on the congestion
    # values surrounding the component's occupied cells.
    # We look one cell beyond the component's bbox in each direction.
    sum_dx = 0.0
    sum_dy = 0.0
    count = 0

    # For each occupied cell, compute the difference between the
    # cell to the left vs right (dx) and above vs below (dy).
    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            # Horizontal gradient: right - left
            if col > 0 and col < nx - 1:
                dx = grid[row][col + 1] - grid[row][col - 1]
            elif col > 0:
                dx = grid[row][col] - grid[row][col - 1]
            elif col < nx - 1:
                dx = grid[row][col + 1] - grid[row][col]
            else:
                dx = 0.0

            # Vertical gradient: down - up
            if row > 0 and row < ny - 1:
                dy = grid[row + 1][col] - grid[row - 1][col]
            elif row > 0:
                dy = grid[row][col] - grid[row - 1][col]
            elif row < ny - 1:
                dy = grid[row + 1][col] - grid[row][col]
            else:
                dy = 0.0

            sum_dx += dx
            sum_dy += dy
            count += 1

    if count == 0:
        return 0.0, 0.0

    avg_dx = sum_dx / count
    avg_dy = sum_dy / count

    # The gradient points toward higher congestion.
    # We want to move AWAY from congestion, so negate it.
    move_dx = -avg_dx
    move_dy = -avg_dy

    # Normalize to a unit vector
    magnitude = math.sqrt(move_dx * move_dx + move_dy * move_dy)
    if magnitude < 1e-12:
        return 0.0, 0.0

    return move_dx / magnitude, move_dy / magnitude
