"""
Linear Conflict heuristic for the 15-puzzle.

LC(state) = Manhattan Distance(state) + 2 * linear_conflicts(state)

A linear conflict occurs when two tiles t_j and t_k are both in their
goal row (or goal column), but t_j is to the RIGHT of t_k while t_j's
goal position is to the LEFT of t_k's goal position.  Each such pair
requires at least 2 extra moves (one tile must move out of the row and
back in to let the other pass).

This heuristic is:
  - Admissible  (never overestimates)
  - Consistent  (satisfies the triangle inequality)
  - Strictly dominates Manhattan Distance  (LC >= MD always)

Reference:
    Hansson, Mayer & Yung (1992)
    "Generating Admissible Heuristics by Criticizing Solutions to
     Relaxed Models"
"""

from __future__ import annotations
from domain.puzzle import SIZE, GOAL_POS
from heuristics.manhattan import manhattan_distance


# ---------------------------------------------------------------------------
# Row and column conflict counters
# ---------------------------------------------------------------------------

def _row_conflicts(state: tuple[int, ...]) -> int:
    """
    Count the minimum number of tiles that must leave their goal row
    to resolve all linear conflicts in rows.

    We use a greedy algorithm: repeatedly remove the tile involved in
    the most conflicts until no conflicts remain.
    """
    total_removed = 0

    for row in range(SIZE):
        # Collect tiles that are in this row AND whose goal is also in this row.
        # Store as list of (current_col, goal_col) pairs.
        row_tiles: list[tuple[int, int]] = []
        for col in range(SIZE):
            tile = state[row * SIZE + col]
            if tile == 0:
                continue
            goal_row, goal_col = GOAL_POS[tile]
            if goal_row == row:
                row_tiles.append((col, goal_col))

        # Build conflict counts: how many other tiles does each conflict with?
        n = len(row_tiles)
        conflicts = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                ci, gi = row_tiles[i]
                cj, gj = row_tiles[j]
                # Conflict: they are in reversed order relative to their goals
                if (ci < cj and gi > gj) or (ci > cj and gi < gj):
                    conflicts[i] += 1
                    conflicts[j] += 1

        # Greedily remove the tile with the most conflicts
        removed = [False] * n
        while True:
            max_c = max((c for c, r in zip(conflicts, removed) if not r), default=0)
            if max_c == 0:
                break
            # Remove the tile with max conflicts
            idx = next(i for i in range(n) if not removed[i] and conflicts[i] == max_c)
            removed[idx] = True
            total_removed += 1
            # Update conflict counts
            ci, gi = row_tiles[idx]
            for j in range(n):
                if removed[j]:
                    continue
                cj, gj = row_tiles[j]
                if (ci < cj and gi > gj) or (ci > cj and gi < gj):
                    conflicts[j] -= 1
                    conflicts[idx] = 0  # already removed

    return total_removed


def _col_conflicts(state: tuple[int, ...]) -> int:
    """
    Count the minimum number of tiles that must leave their goal column
    to resolve all linear conflicts in columns.
    """
    total_removed = 0

    for col in range(SIZE):
        col_tiles: list[tuple[int, int]] = []
        for row in range(SIZE):
            tile = state[row * SIZE + col]
            if tile == 0:
                continue
            goal_row, goal_col = GOAL_POS[tile]
            if goal_col == col:
                col_tiles.append((row, goal_row))

        n = len(col_tiles)
        conflicts = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                ri, gi = col_tiles[i]
                rj, gj = col_tiles[j]
                if (ri < rj and gi > gj) or (ri > rj and gi < gj):
                    conflicts[i] += 1
                    conflicts[j] += 1

        removed = [False] * n
        while True:
            max_c = max((c for c, r in zip(conflicts, removed) if not r), default=0)
            if max_c == 0:
                break
            idx = next(i for i in range(n) if not removed[i] and conflicts[i] == max_c)
            removed[idx] = True
            total_removed += 1
            ri, gi = col_tiles[idx]
            for j in range(n):
                if removed[j]:
                    continue
                rj, gj = col_tiles[j]
                if (ri < rj and gi > gj) or (ri > rj and gi < gj):
                    conflicts[j] -= 1
                    conflicts[idx] = 0

    return total_removed


# ---------------------------------------------------------------------------
# Public heuristic
# ---------------------------------------------------------------------------

def linear_conflict(state: tuple[int, ...]) -> int:
    """
    Return the Linear Conflict heuristic value for *state*.

    LC(state) = MD(state) + 2 * (row_conflicts + col_conflicts)

    Each removed tile needs at least 2 extra moves (out of row/col and back).
    """
    md = manhattan_distance(state)
    rc = _row_conflicts(state)
    cc = _col_conflicts(state)
    return md + 2 * (rc + cc)