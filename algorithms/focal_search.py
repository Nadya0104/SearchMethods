"""
Focal Search  (A*ε).

Maintains two lists:
  OPEN   — all generated nodes, ordered by f(n) = g(n) + h(n)
  FOCAL  — subset of OPEN where f(n) ≤ w * f_min, ordered by d(n)

At each step the node with minimum d in FOCAL is expanded.
Guarantee: every solution found has cost ≤ w * optimal.

Implementation notes
--------------------
We use lazy-deletion heaps for both OPEN and FOCAL.
The tricky part is keeping FOCAL in sync when f_min rises:
  - We track f_min as the smallest f seen on valid OPEN entries.
  - When we pop from FOCAL, we re-validate that the node is still
    within the current threshold (f(n) ≤ w * f_min).
  - New nodes are added to FOCAL immediately if they are within threshold.
  - When f_min rises (because the old f_min node was expanded or re-discovered),
    we do a targeted sweep of OPEN to admit newly eligible nodes into FOCAL.

The sweep scans the lazy-deletion heap and adds every valid OPEN node with
f ≤ new_threshold. Expanded and stale entries are ignored explicitly.
"""

from __future__ import annotations
import heapq
import time
from typing import Callable

from domain.puzzle import GOAL_STATE, get_neighbors_with_cost


def focal_search(
    start:     tuple[int, ...],
    h:         Callable[[tuple[int, ...]], int | float],
    d:         Callable[[tuple[int, ...]], int | float] | None = None,
    weight:    float = 1.0,
    max_nodes: int   = 5_000_000,
    timeout:   float = 60.0,
) -> dict:
    """
    Run Focal Search from *start* to GOAL_STATE.

    Parameters
    ----------
    start   : initial puzzle state
    h       : admissible heuristic for f = g + h  (bounds solution cost)
    d       : distance-to-go estimate for ordering FOCAL
              (defaults to h; can be inadmissible)
    weight  : suboptimality bound w
    max_nodes / timeout : resource limits
    """
    if d is None:
        d = h

    t0 = time.perf_counter()

    h0 = h(start)
    d0 = d(start)

    # Heap entries: (key, g, state)
    # OPEN  ordered by f = g + h
    # FOCAL ordered by d value
    open_heap:  list[tuple] = [(h0, 0, start)]   # (f, g, state)
    focal_heap: list[tuple] = [(d0, 0, start)]   # (d, g, state)

    # Best known g for each state
    g_table: dict[tuple[int, ...], int | float] = {start: 0}
    open_states: set[tuple[int, ...]] = {start}

    # Path reconstruction
    came_from: dict[tuple[int, ...], tuple[int, ...] | None] = {start: None}

    # Track the current focal threshold
    f_min:           float = float(h0)
    focal_threshold: float = weight * f_min

    # Track which states are currently in focal (to avoid duplicate pushes)
    in_focal: set[tuple[tuple[int, ...], int | float]] = {(start, 0)}

    nodes_expanded  = 0
    nodes_generated = 1

    def _update_f_min() -> float:
        """Peek at OPEN, skipping stale entries, return current f_min."""
        while open_heap:
            f, g, s = open_heap[0]
            if s in open_states and g_table.get(s) == g:
                return float(f)
            heapq.heappop(open_heap)
        return float("inf")

    def _admit_to_focal(new_threshold: float) -> None:
        """
        Scan OPEN and push any valid node with f ≤ new_threshold into FOCAL
        that isn't already there.
        Because OPEN is a min-heap on f, we can use a temporary list to
        peek at all entries up to the threshold.
        """
        # We can't break early on a lazy-deletion heap (stale entries may
        # have smaller keys than valid ones), so we scan all entries.
        # In practice OPEN is small for bounded-suboptimal search.
        for entry in open_heap:
            f, g, s = entry
            if f > new_threshold:
                # Min-heap: once we see f > threshold we could break,
                # but stale entries may be mixed in — so keep going.
                # For correctness we do a full scan; for large instances
                # this is still fast because FOCAL is small relative to OPEN.
                continue
            if (s in open_states and g_table.get(s) == g
                    and (s, g) not in in_focal):
                d_val = d(s)
                heapq.heappush(focal_heap, (d_val, g, s))
                in_focal.add((s, g))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    while open_heap:
        if time.perf_counter() - t0 > timeout:
            break
        if nodes_expanded >= max_nodes:
            break

        # ── Refresh f_min ─────────────────────────────────────────────
        new_f_min = _update_f_min()
        if new_f_min == float("inf"):
            break   # OPEN exhausted

        if new_f_min > f_min:
            # f_min rose — admit newly eligible nodes into FOCAL
            new_threshold = weight * new_f_min
            _admit_to_focal(new_threshold)
            f_min           = new_f_min
            focal_threshold = new_threshold

        # ── Pop best node from FOCAL ───────────────────────────────────
        state = None
        g     = None
        while focal_heap:
            d_val, g_entry, s = heapq.heappop(focal_heap)
            in_focal.discard((s, g_entry))

            # Skip stale entries
            if s not in open_states or g_table.get(s) != g_entry:
                continue

            # Re-check threshold (f_min may have moved while this sat in heap)
            f_s = g_entry + h(s)
            if f_s > focal_threshold + 1e-9:
                # Node no longer qualifies — re-add to FOCAL only if still eligible
                # (it may qualify again if f_min has already risen to cover it,
                # but that would have been handled in _admit_to_focal already).
                continue

            state = s
            g     = g_entry
            break

        if state is None:
            # FOCAL is empty — should not happen if bookkeeping is correct,
            # but if it does, fall back to best OPEN node (like wA*).
            # This keeps us sound (still w-admissible).
            new_f_min = _update_f_min()
            if new_f_min == float("inf"):
                break
            f_top, g_top, s_top = open_heap[0]
            while (open_heap and
                   (open_heap[0][2] not in open_states or
                    g_table.get(open_heap[0][2]) != open_heap[0][1])):
                heapq.heappop(open_heap)
            if not open_heap:
                break
            f_top, g_top, s_top = heapq.heappop(open_heap)
            state = s_top
            g     = g_top

        open_states.discard(state)

        # ── Expand ────────────────────────────────────────────────────
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
                g_table[neighbor]    = new_g
                came_from[neighbor]  = state
                nodes_generated     += 1
                open_states.add(neighbor)

                f_nb = new_g + h(neighbor)
                heapq.heappush(open_heap, (f_nb, new_g, neighbor))

                focal_key = (neighbor, new_g)
                if f_nb <= focal_threshold + 1e-9 and focal_key not in in_focal:
                    d_nb = d(neighbor)
                    heapq.heappush(focal_heap, (d_nb, new_g, neighbor))
                    in_focal.add(focal_key)

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
