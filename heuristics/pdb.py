"""
Disjoint Pattern Database (PDB) heuristic for the 15-puzzle.

A PDB stores the exact number of moves required to place a *subset* of
tiles into their goal positions, ignoring all other tiles.  This is an
admissible lower bound on the full problem.

Disjoint PDBs
-------------
When the tile subsets are disjoint (no tile appears in two groups), the
costs from each sub-problem can be SUMMED without double-counting, giving
a heuristic that strictly dominates Manhattan Distance.

We use the classic 5-5-5 partition of the 15 tiles:
    Group A:  tiles { 1,  2,  3,  4,  5}
    Group B:  tiles { 6,  7,  8,  9, 10}
    Group C:  tiles {11, 12, 13, 14, 15}

Each database is built by a *backwards BFS* from the goal using a
compact state encoding:
    (blank_pos, pos_tile_t0, pos_tile_t1, ..., pos_tile_tk)
This avoids storing the full 16-cell board and keeps memory < 50 MB total.

State space size for k=5 group tiles:
    P(16, 6) = 16 × 15 × 14 × 13 × 12 × 11 = 5,765,760 entries per DB

Build time: ~10-30 seconds per DB on a modern CPU.

Usage
-----
    from heuristics.pdb import DisjointPDB

    pdb = DisjointPDB()          # builds all three groups (or loads from cache)
    h_val = pdb.heuristic(state) # sum of the three group costs

Persistence
-----------
    pdb.save("pdb_cache.pkl")
    pdb = DisjointPDB.load("pdb_cache.pkl")
"""

from __future__ import annotations
import pickle
from collections import deque
from pathlib import Path

from domain.puzzle import SIZE, N_TILES, GOAL_STATE

# ---------------------------------------------------------------------------
# Default tile partition  (5-5-5)
# ---------------------------------------------------------------------------

DEFAULT_GROUPS: list[frozenset[int]] = [
    frozenset({1,  2,  3,  4,  5}),
    frozenset({6,  7,  8,  9, 10}),
    frozenset({11, 12, 13, 14, 15}),
]

# Precomputed neighbour table
def _build_neighbours() -> list[list[int]]:
    nb: list[list[int]] = []
    for i in range(N_TILES):
        row, col = divmod(i, SIZE)
        adj = []
        if row > 0:        adj.append(i - SIZE)
        if row < SIZE - 1: adj.append(i + SIZE)
        if col > 0:        adj.append(i - 1)
        if col < SIZE - 1: adj.append(i + 1)
        nb.append(adj)
    return nb

_NEIGHBOURS = _build_neighbours()


# ---------------------------------------------------------------------------
# BFS builder  (compact encoding)
# ---------------------------------------------------------------------------

def _build_pdb(group: frozenset[int]) -> dict[tuple[int, ...], int]:
    """
    Build a PDB for *group* via backwards BFS from the goal.

    State encoding: (blank_pos, pos_t0, pos_t1, ..., pos_tk)
    where t0 < t1 < ... tk is the sorted group order.

    Expansion: the blank can move into any neighbour.
      - Neighbour holds a group tile → that tile moves to blank's old pos.
      - Neighbour holds a wildcard   → blank moves freely (still costs 1).

    This compact representation keeps memory ~50 MB for three 5-tile groups.
    """
    sorted_group = sorted(group)
    k = len(sorted_group)

    # Map tile value → index in sorted_group (for fast lookup)
    tile_to_idx: dict[int, int] = {t: i for i, t in enumerate(sorted_group)}

    def encode(full_state: tuple[int, ...]) -> tuple[int, ...]:
        enc = [0] * (k + 1)
        enc[0] = full_state.index(0)
        for i, tile in enumerate(sorted_group):
            enc[i + 1] = full_state.index(tile)
        return tuple(enc)

    goal_enc = encode(GOAL_STATE)
    db: dict[tuple[int, ...], int] = {goal_enc: 0}
    queue: deque[tuple[tuple[int, ...], int]] = deque()
    queue.append((goal_enc, 0))

    while queue:
        enc, cost = queue.popleft()
        blank_pos = enc[0]

        # Build position → enc_index map for group tiles
        pos_to_enc_idx: dict[int, int] = {}
        for i in range(k):
            pos_to_enc_idx[enc[i + 1]] = i + 1   # enc index (1-based)

        for nb in _NEIGHBOURS[blank_pos]:
            new_enc = list(enc)
            new_enc[0] = nb                        # blank moves to nb

            if nb in pos_to_enc_idx:
                # A group tile is at nb — it moves to blank_pos
                # This counts as a move (cost += 1)
                enc_idx = pos_to_enc_idx[nb]
                new_enc[enc_idx] = blank_pos
                new_cost = cost + 1
            else:
                # Wildcard tile at nb — blank slides through for FREE
                # (disjoint PDB: only group tile moves count)
                new_cost = cost

            child = tuple(new_enc)
            if child not in db:
                db[child] = new_cost
                queue.append((child, new_cost))

    return db


# ---------------------------------------------------------------------------
# Abstract state key (for heuristic lookup)
# ---------------------------------------------------------------------------

def _make_key(
    state: tuple[int, ...],
    sorted_group: list[int],
) -> tuple[int, ...]:
    """
    Produce the compact encoding key for *state* given a sorted tile group.
    Must match the encoding used during BFS.
    """
    k = len(sorted_group)
    enc = [0] * (k + 1)
    enc[0] = state.index(0)
    for i, tile in enumerate(sorted_group):
        enc[i + 1] = state.index(tile)
    return tuple(enc)


# ---------------------------------------------------------------------------
# DisjointPDB class
# ---------------------------------------------------------------------------

class DisjointPDB:
    """
    Disjoint Pattern Database heuristic.

    Parameters
    ----------
    groups  : list of frozensets of tile numbers (must be disjoint, cover 1-15)
              Defaults to the classic 5-5-5 partition.
    verbose : print progress during build
    """

    def __init__(
        self,
        groups: list[frozenset[int]] | None = None,
        verbose: bool = True,
    ) -> None:
        self.groups = groups if groups is not None else DEFAULT_GROUPS
        self._sorted_groups: list[list[int]] = [sorted(g) for g in self.groups]
        self._dbs: list[dict[tuple[int, ...], int]] = []
        self._build(verbose)

    def _build(self, verbose: bool) -> None:
        for i, group in enumerate(self.groups):
            if verbose:
                print(f"  Building PDB group {i+1}/{len(self.groups)}: "
                      f"tiles {sorted(group)} ...", end=" ", flush=True)
            db = _build_pdb(group)
            self._dbs.append(db)
            if verbose:
                print(f"{len(db):,} states")

    def heuristic(self, state: tuple[int, ...]) -> int:
        """
        Return the disjoint PDB heuristic value for *state*.
        h(state) = sum of lookup costs across all groups.
        """
        total = 0
        for sg, db in zip(self._sorted_groups, self._dbs):
            key = _make_key(state, sg)
            total += db.get(key, 0)
        return total

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Pickle the PDB to disk."""
        with open(path, "wb") as f:
            pickle.dump({
                "groups":        self.groups,
                "sorted_groups": self._sorted_groups,
                "dbs":           self._dbs,
            }, f)
        print(f"PDB saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "DisjointPDB":
        """Load a previously saved PDB from disk (skips BFS build)."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls.__new__(cls)
        obj.groups         = data["groups"]
        obj._sorted_groups = data["sorted_groups"]
        obj._dbs           = data["dbs"]
        print(f"PDB loaded from {path}  "
              f"({sum(len(d) for d in obj._dbs):,} total entries)")
        return obj

    def __repr__(self) -> str:
        sizes = [len(d) for d in self._dbs]
        return (f"DisjointPDB(groups={[sorted(g) for g in self.groups]}, "
                f"db_sizes={sizes})")


# ---------------------------------------------------------------------------
# Convenience: build or load
# ---------------------------------------------------------------------------

def get_pdb(
    cache_path: str | Path = "pdb_cache.pkl",
    groups: list[frozenset[int]] | None = None,
    verbose: bool = True,
) -> DisjointPDB:
    """
    Return a DisjointPDB, loading from *cache_path* if it exists,
    otherwise building from scratch and saving to cache.
    """
    cache = Path(cache_path)
    if cache.exists():
        if verbose:
            print(f"Loading PDB from cache: {cache}")
        return DisjointPDB.load(cache)
    else:
        if verbose:
            print("Building PDB from scratch...")
        pdb = DisjointPDB(groups=groups, verbose=verbose)
        pdb.save(cache)
        return pdb