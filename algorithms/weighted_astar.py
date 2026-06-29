"""
Weighted A* (wA*).

Evaluation function: f'(n) = g(n) + w * h(n)

w = 1  →  standard A* (optimal)
w > 1  →  suboptimal but faster; guarantees cost ≤ w * optimal
"""

from __future__ import annotations
import heapq
import time
from typing import Callable

from domain.puzzle import GOAL_STATE, get_neighbors_with_cost


def weighted_astar(
    start:     tuple[int, ...],
    h:         Callable[[tuple[int, ...]], int | float],
    weight:    float = 1.0,
    max_nodes: int   = 5_000_000,
    timeout:   float = 60.0,          # seconds
) -> dict:
    """
    Run Weighted A* from *start* to GOAL_STATE.

    Parameters
    ----------
    start      : initial puzzle state
    h          : admissible heuristic  h(state) -> non-negative number
    weight     : suboptimality bound  w  (use 1.0 for optimal A*)
    max_nodes  : hard cap on expansions (returns failure if exceeded)
    timeout    : wall-clock limit in seconds

    Returns
    -------
    dict with keys:
        solved          bool
        cost            solution cost (None if not solved)
        path            list of states from start to goal (None if not solved)
        nodes_expanded  int
        nodes_generated int
        elapsed         float  (seconds)
        weight          float
    """
    t0 = time.perf_counter()

    h_start = h(start)

    # Each heap entry: (f', g, state, parent_index)
    # We store the full path implicitly via a parent table.
    # open_heap entries: (f_prime, g, state)
    open_heap: list[tuple[float, int, tuple[int, ...]]] = []
    heapq.heappush(open_heap, (weight * h_start, 0, start))

    # g_table[state] = best g known so far
    g_table: dict[tuple[int, ...], int] = {start: 0}

    # For path reconstruction: came_from[state] = parent_state
    came_from: dict[tuple[int, ...], tuple[int, ...] | None] = {start: None}

    nodes_expanded  = 0
    nodes_generated = 1

    while open_heap:
        if time.perf_counter() - t0 > timeout:
            break
        if nodes_expanded >= max_nodes:
            break

        f_prime, g, state = heapq.heappop(open_heap)

        # Skip stale entries (lazy deletion)
        if g > g_table.get(state, float("inf")):
            continue

        nodes_expanded += 1

        if state == GOAL_STATE:
            path = _reconstruct_path(came_from, state)
            return {
                "solved":          True,
                "cost":            g,
                "path":            path,
                "nodes_expanded":  nodes_expanded,
                "nodes_generated": nodes_generated,
                "elapsed":         time.perf_counter() - t0,
                "weight":          weight,
            }

        for neighbor, cost in get_neighbors_with_cost(state):
            new_g = g + cost
            if new_g < g_table.get(neighbor, float("inf")):
                g_table[neighbor]  = new_g
                came_from[neighbor] = state
                f_prime_nb = new_g + weight * h(neighbor)
                heapq.heappush(open_heap, (f_prime_nb, new_g, neighbor))
                nodes_generated += 1

    return {
        "solved":          False,
        "cost":            None,
        "path":            None,
        "nodes_expanded":  nodes_expanded,
        "nodes_generated": nodes_generated,
        "elapsed":         time.perf_counter() - t0,
        "weight":          weight,
    }


def _reconstruct_path(
    came_from: dict[tuple[int, ...], tuple[int, ...] | None],
    goal:      tuple[int, ...],
) -> list[tuple[int, ...]]:
    path = []
    node: tuple[int, ...] | None = goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path