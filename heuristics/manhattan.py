"""
Manhattan Distance heuristic for the 15-puzzle.

h(state) = sum over all tiles t != 0 of:
    |current_row(t) - goal_row(t)| + |current_col(t) - goal_col(t)|

Admissible and consistent.
"""

from domain.puzzle import SIZE, N_TILES, GOAL_POS


def manhattan_distance(state: tuple[int, ...]) -> int:
    """
    Return the Manhattan Distance heuristic value for *state*.

    This is also used as the distance-to-go estimate d(state) for EES,
    since in the standard 15-puzzle cost == number of moves.
    """
    total = 0
    for i, tile in enumerate(state):
        if tile == 0:
            continue
        gr, gc = GOAL_POS[tile]
        cr, cc = i // SIZE, i % SIZE
        total += abs(cr - gr) + abs(cc - gc)
    return total