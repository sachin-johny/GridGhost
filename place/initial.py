"""Initial interior placement — simple bin-packing on a grid.

For Commit 2 we keep this deliberately simple: macros go onto a
row-major grid sized to fit the largest macro bbox. SA + legalizer
will optimize from there. We do NOT do net-aware clustering here;
HPWL optimization is SA's job.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.board_model import BoardModel, Component, BoardOutline
    from models.macro import Macro


def place_interior_grid(
    macros: list["Macro"],
    interior_bbox: tuple[float, float, float, float],
    gap: float = 2.0,
) -> None:
    """Place macros in a row-major grid filling ``interior_bbox``.

    Each macro is centered in its own cell. Cell size is determined
    by the macro's bbox extent plus ``gap``. Macros are placed in
    sorted-ref order so the layout is deterministic.
    """
    if not macros:
        return

    x_min, y_min, x_max, y_max = interior_bbox
    interior_w = x_max - x_min
    interior_h = y_max - y_min

    # Sort macros by ref of leader for determinism
    sorted_macros = sorted(macros, key=lambda m: m.leader.ref)

    # Cell size = max macro bbox + gap
    cell_w = max(m.bbox[2] - m.bbox[0] for m in sorted_macros) + gap
    cell_h = max(m.bbox[3] - m.bbox[1] for m in sorted_macros) + gap

    cols = max(1, int(interior_w // cell_w))
    rows = max(1, math.ceil(len(sorted_macros) / cols))

    # If we can't fit, shrink cells
    while cols * cell_w > interior_w and cols > 1:
        cols -= 1
        rows = math.ceil(len(sorted_macros) / cols)
    # Recompute cell size to fit
    cell_w = max(cell_w, interior_w / cols)
    cell_h = max(cell_h, interior_h / rows)

    for i, m in enumerate(sorted_macros):
        col = i % cols
        row = i // cols
        cx = x_min + (col + 0.5) * cell_w
        cy = y_min + (row + 0.5) * cell_h
        # Position macro so its leader is at the cell center. We use
        # set_pose with rotation=0 (initial placement, no rotation).
        m.set_pose(cx, cy, 0.0)


def compute_interior_bbox(
    macros: list["Macro"],
    board: "BoardOutline",
    margin: float,
    connector_reserve: float = 0.0,
) -> tuple[float, float, float, float]:
    """Compute the usable interior region.

    Shrinks the board by ``margin`` plus ``connector_reserve`` (room
    reserved on each edge for connectors).
    """
    x_min = board.x_min + margin + connector_reserve
    y_min = board.y_min + margin + connector_reserve
    x_max = board.x_max - margin - connector_reserve
    y_max = board.y_max - margin - connector_reserve
    return (x_min, y_min, x_max, y_max)
