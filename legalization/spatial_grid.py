"""Uniform grid spatial index for overlap detection acceleration.

Divides the board into uniform grid cells. Each cell stores a list of
component indices. Overlap checks query only nearby cells, reducing
average complexity from O(N^2) to O(N) for sparse boards.
"""

from __future__ import annotations

from models.board_model import BoardModel, Component, BoardOutline


class SpatialGrid:
    """Uniform grid spatial index for fast overlap candidate lookup."""

    def __init__(self, board: BoardOutline, cell_size: float):
        self._x_min = board.x_min
        self._y_min = board.y_min
        self._x_max = board.x_max
        self._y_max = board.y_max
        self.cell_size = cell_size
        self.cells: dict[tuple[int, int], list[int]] = {}
        self.comp_cells: dict[int, set[tuple[int, int]]] = {}

    @classmethod
    def from_components(
        cls, components: list[Component], board: BoardOutline
    ) -> SpatialGrid:
        max_dim = 0.0
        for comp in components:
            max_dim = max(max_dim, comp.effective_width, comp.effective_height)
        cell_size = max(max_dim * 1.5, 1.0)
        grid = cls(board, cell_size)
        grid.build(components)
        return grid

    def _cells_for_bbox(
        self, bbox: tuple[float, float, float, float]
    ) -> list[tuple[int, int]]:
        x1, y1, x2, y2 = bbox
        col_min = max(int((x1 - self._x_min) / self.cell_size), 0)
        row_min = max(int((y1 - self._y_min) / self.cell_size), 0)
        col_max = int((x2 - self._x_min) / self.cell_size)
        row_max = int((y2 - self._y_min) / self.cell_size)
        return [
            (c, r)
            for c in range(col_min, col_max + 1)
            for r in range(row_min, row_max + 1)
        ]

    def build(self, components: list[Component]) -> None:
        self.cells.clear()
        self.comp_cells.clear()
        for idx in range(len(components)):
            self._insert(idx, components[idx])

    def _insert(self, idx: int, comp: Component) -> None:
        occupied = self._cells_for_bbox(comp.bbox)
        self.comp_cells[idx] = set(occupied)
        for cell in occupied:
            if cell not in self.cells:
                self.cells[cell] = []
            self.cells[cell].append(idx)

    def query_overlaps(
        self, comp_idx: int, components: list[Component]
    ) -> list[int]:
        comp = components[comp_idx]
        cells = self._cells_for_bbox(comp.bbox)
        result: set[int] = set()
        for cell in cells:
            if cell in self.cells:
                result.update(self.cells[cell])
        result.discard(comp_idx)
        return list(result)


def compute_overlap_stats_fast(
    model: BoardModel, grid: SpatialGrid,
) -> tuple[int, float]:
    count = 0
    total_area = 0.0
    components = model.components
    seen_pairs: set[tuple[int, int]] = set()
    for i in range(len(components)):
        candidates = grid.query_overlaps(i, components)
        for j in candidates:
            pair = (i, j) if i < j else (j, i)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            if components[pair[0]].overlaps(components[pair[1]]):
                count += 1
                total_area += components[pair[0]].overlap_area(components[pair[1]])
    return count, total_area


def count_overlaps_involving_fast(
    comp: Component, components: list[Component], grid: SpatialGrid,
) -> int:
    comp_idx: int | None = None
    for i, c in enumerate(components):
        if c is comp:
            comp_idx = i
            break
    if comp_idx is None:
        return 0
    candidates = grid.query_overlaps(comp_idx, components)
    count = 0
    for j in candidates:
        if comp.overlaps(components[j]):
            count += 1
    return count


def count_pair_overlaps_involving_fast(
    c1: Component, c2: Component, components: list[Component], grid: SpatialGrid,
) -> int:
    c1_idx: int | None = None
    c2_idx: int | None = None
    for i, c in enumerate(components):
        if c is c1:
            c1_idx = i
        elif c is c2:
            c2_idx = i
    count = 0
    if c1_idx is not None:
        candidates = grid.query_overlaps(c1_idx, components)
        for j in candidates:
            if j == c2_idx:
                continue
            if c1.overlaps(components[j]):
                count += 1
    if c2_idx is not None:
        candidates = grid.query_overlaps(c2_idx, components)
        for j in candidates:
            if j == c1_idx:
                continue
            if c2.overlaps(components[j]):
                count += 1
    if c1.overlaps(c2):
        count += 1
    return count
