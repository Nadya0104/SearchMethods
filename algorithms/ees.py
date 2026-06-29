"""
Explicit Estimation Search  (EES)
Thayer & Ruml, IJCAI 2011.

Three queues:
  OPEN    — all generated nodes, ordered by f̂(n) = g(n) + ĥ(n)
              where ĥ is a potentially inadmissible cost-to-go estimate.
              bestfhat = front of OPEN.

  FOCAL   — prefix of OPEN where f̂(n) ≤ w * f̂(bestfhat),
              ordered by d̂(n) (inadmissible distance-to-go estimate).
              bestd = front of FOCAL.

  CLEANUP — all nodes from OPEN, ordered by f(n) = g(n) + h(n)
              where h is the *admissible* heuristic.
              bestf = front of CLEANUP.

Node selection  (selectNode):
  1. if f̂(bestd) ≤ w * f(bestf)   → expand bestd     (pursue nearest goal)
  2. elif f̂(bestfhat) ≤ w * f(bestf) → expand bestfhat  (improve cost estimate)
  3. else                            → expand bestf     (raise the lower bound)

Suboptimality guarantee: every expanded node n satisfies f(n) ≤ w * g(opt).

Notes on this implementation
----------------------------
* We use lazy-deletion heaps for all three queues.
* For the 15-puzzle (unit costs, h admissible):
    - h  = admissible heuristic  (Manhattan Distance or Linear Conflict)
    - ĥ  = same as h by default  (or a slightly inflated version)
    - d̂  = h by default          (distance ≈ cost in unit-cost domains)
  You can pass different callables for h_hat and d_hat.
* The paper notes EES is not the fastest on unit-cost tiles (overhead
  dominates), but it is still w-admissible and correct.
"""

from __future__ import annotations
import heapq
import time
from typing import Callable

from domain.puzzle import GOAL_STATE, get_neighbors_with_cost


def ees(
    start:     tuple[int, ...],
    h:         Callable[[tuple[int, ...]], int | float],   # admissible
    h_hat:     Callable[[tuple[int, ...]], int | float] | None = None,  # inadmissible cost est.
    d_hat:     Callable[[tuple[int, ...]], int | float] | None = None,  # inadmissible dist est.
    weight:    float = 1.0,
    max_nodes: int   = 5_000_000,
    timeout:   float = 60.0,
) -> dict:
    """
    Run Explicit Estimation Search from *start* to GOAL_STATE.

    Parameters
    ----------
    start   : initial puzzle state
    h       : ADMISSIBLE heuristic — used only for the cleanup list (bounding)
    h_hat   : inadmissible cost-to-go estimate (defaults to h)
    d_hat   : inadmissible distance-to-go estimate (defaults to h)
    weight  : suboptimality bound w
    max_nodes / timeout : resource limits

    Returns
    -------
    Same dict schema as weighted_astar.
    """
    if h_hat is None:
        h_hat = h
    if d_hat is None:
        d_hat = h

    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Initialise
    # ------------------------------------------------------------------
    h0     = h(start)
    hhat0  = h_hat(start)
    dhat0  = d_hat(start)
    g0     = 0
    f0     = g0 + h0          # admissible f  → cleanup order
    fhat0  = g0 + hhat0       # inadmissible f̂ → open order

    # Heap entries
    # open_heap    : (fhat, g, state)
    # focal_heap   : (dhat, g, state)
    # cleanup_heap : (f,    g, state)
    open_heap:    list[tuple] = [(fhat0, g0, start)]
    focal_heap:   list[tuple] = [(dhat0, g0, start)]
    cleanup_heap: list[tuple] = [(f0,    g0, start)]

    g_table:   dict[tuple[int, ...], int]                     = {start: 0}
    came_from: dict[tuple[int, ...], tuple[int, ...] | None]  = {start: None}

    nodes_expanded  = 0
    nodes_generated = 1

    def _valid(g: int, state: tuple[int, ...]) -> bool:
        """Check whether a heap entry is still current."""
        return g_table.get(state) == g

    def _peek_valid(heap: list) -> tuple | None:
        """Return front of heap, skipping stale entries. Returns None if empty."""
        while heap:
            if _valid(heap[0][1], heap[0][2]):
                return heap[0]
            heapq.heappop(heap)
        return None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    while True:
        if time.perf_counter() - t0 > timeout:
            break
        if nodes_expanded >= max_nodes:
            break

        # ── Identify the three candidates ────────────────────────────
        top_cleanup = _peek_valid(cleanup_heap)
        if top_cleanup is None:
            break                           # search exhausted

        f_bestf = top_cleanup[0]            # admissible lower bound

        top_open = _peek_valid(open_heap)
        if top_open is None:
            break

        fhat_bestfhat = top_open[0]         # f̂ of bestfhat

        # ── selectNode ───────────────────────────────────────────────
        # Rebuild focal: it should contain all open nodes with
        # f̂(n) ≤ w * f̂(bestfhat).  We use lazy addition:
        # nodes are added to focal when they are generated (if eligible),
        # and we check eligibility again when we pop from focal.
        focal_threshold = weight * fhat_bestfhat

        # Try bestd (front of focal)
        selected_state = None
        selected_g     = None
        source         = None

        # Check bestd
        top_focal = _peek_valid(focal_heap)
        if top_focal is not None:
            fhat_of_bestd = top_focal[1] + h_hat(top_focal[2])  # g + ĥ
            if fhat_of_bestd <= weight * f_bestf:
                # Use bestd
                _, g, state = heapq.heappop(focal_heap)
                if _valid(g, state):
                    selected_state = state
                    selected_g     = g
                    source         = "focal"

        # Check bestfhat
        if selected_state is None:
            top_open = _peek_valid(open_heap)
            if top_open is not None and top_open[0] <= weight * f_bestf:
                _, g, state = heapq.heappop(open_heap)
                if _valid(g, state):
                    selected_state = state
                    selected_g     = g
                    source         = "open"

        # Fall back to bestf
        if selected_state is None:
            _, g, state = heapq.heappop(cleanup_heap)
            while not _valid(g, state):
                if not cleanup_heap:
                    break
                _, g, state = heapq.heappop(cleanup_heap)
            if _valid(g, state):
                selected_state = state
                selected_g     = g
                source         = "cleanup"

        if selected_state is None:
            break

        # ── Expand selected node ──────────────────────────────────────
        nodes_expanded += 1
        state = selected_state
        g     = selected_g

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
                g_table[neighbor]   = new_g
                came_from[neighbor] = state
                nodes_generated    += 1

                h_nb    = h(neighbor)
                hhat_nb = h_hat(neighbor)
                dhat_nb = d_hat(neighbor)
                f_nb    = new_g + h_nb
                fhat_nb = new_g + hhat_nb

                heapq.heappush(cleanup_heap, (f_nb,    new_g, neighbor))
                heapq.heappush(open_heap,    (fhat_nb, new_g, neighbor))

                # Add to focal only if within threshold
                if fhat_nb <= focal_threshold:
                    heapq.heappush(focal_heap, (dhat_nb, new_g, neighbor))

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