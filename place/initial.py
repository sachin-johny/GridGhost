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
    """Shelf-pack macros into ``interior_bbox``.

    Sort by macro-bbox height descending so the first row carries the
    tallest macros and sets that row's height. Place left-to-right;
    when the next macro would overflow the row width, wrap. Macros
    that don't fit in the interior (overpacked board) overflow past
    the bottom edge — SA + legalizer will deal with them.

    The leader is placed so the macro's bbox starts at the cursor
    (leader offset within bbox is preserved). set_pose is called
    without bounds so initial placement never rejects a position.
    """
    if not macros:
        return

    x_min, y_min, x_max, y_max = interior_bbox

    sorted_macros = sorted(macros, key=lambda m: -(m.bbox[3] - m.bbox[1]))

    cursor_x = x_min
    cursor_y = y_min
    row_height = 0.0

    for m in sorted_macros:
        bx1, by1, bx2, by2 = m.bbox
        mw = bx2 - bx1
        mh = by2 - by1

        # Wrap if this macro would extend past the right edge.
        if cursor_x + mw > x_max and cursor_x > x_min:
            cursor_y += row_height + gap
            cursor_x = x_min
            row_height = 0.0

        # Leader's offset within macro bbox is preserved through set_pose
        # (rotation=0, no followers change their relative positions).
        leader_offset_x = m.leader.x - bx1
        leader_offset_y = m.leader.y - by1
        m.set_pose(cursor_x + leader_offset_x, cursor_y + leader_offset_y, 0.0)

        cursor_x += mw + gap
        row_height = max(row_height, mh)


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
