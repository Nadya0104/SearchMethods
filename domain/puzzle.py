"""
15-puzzle domain.

State is represented as a flat tuple of 16 integers, where 0 is the blank.
Goal state: (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0)

Layout (index → row, col):
  0  1  2  3
  4  5  6  7
  8  9 10 11
 12 13 14 15
"""

from __future__ import annotations
from typing import Iterator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIZE = 4
N_TILES = SIZE * SIZE  # 16 cells total

GOAL_STATE: tuple[int, ...] = tuple(range(1, N_TILES)) + (0,)
# (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0)

# Precompute goal position of each tile for fast heuristic lookups.
# GOAL_POS[tile] = (row, col)
GOAL_POS: dict[int, tuple[int, int]] = {
    GOAL_STATE[i]: (i // SIZE, i % SIZE)
    for i in range(N_TILES)
}

# Neighbours of each cell index (precomputed for speed).
# _NEIGHBOURS[i] = list of cell indices reachable by sliding the blank.
def _build_neighbours() -> list[list[int]]:
    nb: list[list[int]] = []
    for i in range(N_TILES):
        row, col = divmod(i, SIZE)
        adj = []
        if row > 0:          adj.append(i - SIZE)   # up
        if row < SIZE - 1:   adj.append(i + SIZE)   # down
        if col > 0:          adj.append(i - 1)       # left
        if col < SIZE - 1:   adj.append(i + 1)       # right
        nb.append(adj)
    return nb

_NEIGHBOURS: list[list[int]] = _build_neighbours()


# ---------------------------------------------------------------------------
# Core functions  (stateless — operate on plain tuples)
# ---------------------------------------------------------------------------

def is_goal(state: tuple[int, ...]) -> bool:
    return state == GOAL_STATE


def blank_index(state: tuple[int, ...]) -> int:
    """Return the index of the blank (0) tile."""
    return state.index(0)


def get_neighbors(state: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    """
    Yield all states reachable in one move from *state*.

    A move slides an adjacent tile into the blank cell.
    Each neighbour is a new tuple (no mutation).
    """
    bi = blank_index(state)
    lst = list(state)
    for nb in _NEIGHBOURS[bi]:
        lst[bi], lst[nb] = lst[nb], lst[bi]        # swap blank with neighbour
        yield tuple(lst)
        lst[bi], lst[nb] = lst[nb], lst[bi]        # undo swap (restore lst)


def get_neighbors_with_cost(
    state: tuple[int, ...]
) -> Iterator[tuple[tuple[int, ...], int]]:
    """
    Yield (neighbour_state, cost) pairs.

    For the standard 15-puzzle every move costs 1.
    """
    for nb in get_neighbors(state):
        yield nb, 1


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def state_to_str(state: tuple[int, ...]) -> str:
    """Return a readable 4×4 grid string for debugging."""
    rows = []
    for r in range(SIZE):
        row_tiles = [state[r * SIZE + c] for c in range(SIZE)]
        rows.append(" ".join(f"{t:2d}" for t in row_tiles))
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Solvability check
# ---------------------------------------------------------------------------

def is_solvable(state: tuple[int, ...]) -> bool:
    """
    A 15-puzzle state is solvable iff:
        inversions_count + blank_row_from_bottom  is even.

    (Standard result for even-sized sliding puzzles.)
    """
    tiles = [t for t in state if t != 0]
    inversions = 0
    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            if tiles[i] > tiles[j]:
                inversions += 1

    blank_row_from_bottom = SIZE - blank_index(state) // SIZE  # 1-indexed
    return (inversions + blank_row_from_bottom) % 2 == 1