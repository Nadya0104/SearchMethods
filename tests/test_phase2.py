"""
Unit tests for Phase 2: Linear Conflict, PDB, and d_hat estimators.

Run with:
    cd puzzle_search
    python -m pytest tests/test_phase2.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from domain.puzzle import GOAL_STATE, SIZE
from domain.generator import generate_instance
from heuristics.manhattan      import manhattan_distance
from heuristics.linear_conflict import (
    linear_conflict, _row_conflicts, _col_conflicts,
    _minimum_vertex_cover_size,
)
from heuristics.pdb            import DisjointPDB, DEFAULT_GROUPS, _make_key, _build_pdb
from heuristics.d_hat          import inflated_d_hat, combined_d_hat, identity_d_hat
from algorithms.weighted_astar  import weighted_astar
from algorithms.focal_search    import focal_search
from algorithms.ees             import ees


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ONE_MOVE = list(GOAL_STATE)
ONE_MOVE[-1], ONE_MOVE[-2] = ONE_MOVE[-2], ONE_MOVE[-1]
ONE_MOVE_STATE = tuple(ONE_MOVE)

# State with a known linear conflict:
# Row 0 of goal: tiles 1,2,3,4 at positions 0,1,2,3
# Swap tiles 1 and 2 → conflict in row 0
CONFLICT_STATE = list(GOAL_STATE)
# place tile 2 at col 0 and tile 1 at col 1 in row 0
CONFLICT_STATE[0] = 2   # tile 2 in col 0  (goal col = 1)
CONFLICT_STATE[1] = 1   # tile 1 in col 1  (goal col = 0)
CONFLICT_STATE = tuple(CONFLICT_STATE)


# ---------------------------------------------------------------------------
# Linear Conflict tests
# ---------------------------------------------------------------------------

class TestLinearConflict:

    def test_minimum_vertex_cover_is_exact(self):
        # A path of three edges has minimum vertex cover size two.
        assert _minimum_vertex_cover_size(
            4, [(0, 1), (1, 2), (2, 3)]
        ) == 2

    def test_goal_is_zero(self):
        assert linear_conflict(GOAL_STATE) == 0

    def test_one_move_equals_md(self):
        # A single move cannot create a linear conflict
        # (swapping blank with 15 doesn't put two tiles in reversed goal order)
        lc = linear_conflict(ONE_MOVE_STATE)
        md = manhattan_distance(ONE_MOVE_STATE)
        assert lc == md

    def test_lc_dominates_md(self):
        """LC must be >= MD for all states."""
        for diff in ("easy", "medium", "hard"):
            for seed in range(10):
                s = generate_instance(diff, seed=seed)
                assert linear_conflict(s) >= manhattan_distance(s), \
                    f"LC < MD for {diff} seed={seed}"

    def test_conflict_state_greater_than_md(self):
        """A state with a known conflict should have LC > MD."""
        lc = linear_conflict(CONFLICT_STATE)
        md = manhattan_distance(CONFLICT_STATE)
        assert lc > md, f"Expected LC > MD for conflict state, got LC={lc} MD={md}"

    def test_conflict_state_penalty(self):
        """The swap of tiles 1 and 2 in row 0 adds exactly 2."""
        lc = linear_conflict(CONFLICT_STATE)
        md = manhattan_distance(CONFLICT_STATE)
        assert lc == md + 2

    def test_row_conflicts_goal(self):
        assert _row_conflicts(GOAL_STATE) == 0

    def test_col_conflicts_goal(self):
        assert _col_conflicts(GOAL_STATE) == 0

    def test_nonnegative(self):
        for seed in range(20):
            s = generate_instance("medium", seed=seed)
            assert linear_conflict(s) >= 0

    def test_admissibility_proxy(self):
        """
        Verify LC <= true optimal by solving small instances with A*
        and checking h(start) <= cost.
        """
        for seed in range(5):
            s = generate_instance("easy", seed=seed)
            result = weighted_astar(s, manhattan_distance, weight=1.0, timeout=10)
            if result["solved"]:
                lc = linear_conflict(s)
                assert lc <= result["cost"], \
                    f"LC={lc} > optimal={result['cost']} — admissibility violated!"


# ---------------------------------------------------------------------------
# PDB tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pdb():
    """Build the PDB once for all tests in this module."""
    return DisjointPDB(groups=DEFAULT_GROUPS, verbose=True)


class TestPDB:

    def test_single_tile_pdb_matches_manhattan(self):
        db = _build_pdb(frozenset({1}))
        for diff in ("easy", "medium", "hard"):
            for seed in range(5):
                state = generate_instance(diff, seed=seed)
                key = _make_key(state, [1])
                position = state.index(1)
                goal_position = GOAL_STATE.index(1)
                row, col = divmod(position, SIZE)
                goal_row, goal_col = divmod(goal_position, SIZE)
                assert db[key] == abs(row - goal_row) + abs(col - goal_col)

    def test_goal_is_zero(self, pdb):
        assert pdb.heuristic(GOAL_STATE) == 0

    def test_one_move_is_one(self, pdb):
        # The one-move state moves exactly one group tile by 1 step
        assert pdb.heuristic(ONE_MOVE_STATE) == 1

    def test_pdb_dominates_md(self, pdb):
        """PDB must be >= MD for all test states."""
        for diff in ("easy", "medium"):
            for seed in range(10):
                s = generate_instance(diff, seed=seed)
                assert pdb.heuristic(s) >= manhattan_distance(s), \
                    f"PDB < MD for {diff} seed={seed}"

    def test_pdb_beats_md_on_average(self, pdb):
        """
        PDB and LC are incomparable in general, but PDB should beat MD
        on average and be equal or better than MD on every state.
        (PDB and LC can be incomparable — neither always dominates.)
        """
        pdb_total, md_total = 0, 0
        for diff in ("easy", "medium"):
            for seed in range(10):
                s = generate_instance(diff, seed=seed)
                pdb_total += pdb.heuristic(s)
                md_total  += manhattan_distance(s)
        assert pdb_total >= md_total, "PDB should be stronger than MD on average"

    def test_nonnegative(self, pdb):
        for seed in range(10):
            s = generate_instance("medium", seed=seed)
            assert pdb.heuristic(s) >= 0

    def test_db_sizes(self, pdb):
        """Each sub-DB should have >100k entries (sanity check on BFS coverage)."""
        for db in pdb._dbs:
            assert len(db) > 100_000, f"DB too small: {len(db)}"

    def test_make_key_group(self):
        """Key should be a tuple of length k+1 with valid positions."""
        group = frozenset({1, 2, 3, 4, 5})
        key = _make_key(GOAL_STATE, sorted(group))
        assert len(key) == len(group) + 1          # blank + 5 tiles
        assert all(0 <= pos < 16 for pos in key)   # all positions valid
        assert len(set(key)) == len(key)            # no two tiles share a cell

    def test_admissibility_proxy(self, pdb):
        """PDB h(start) <= optimal cost, verified on easy instances."""
        for seed in range(5):
            s = generate_instance("easy", seed=seed)
            result = weighted_astar(s, manhattan_distance, weight=1.0, timeout=10)
            if result["solved"]:
                h_pdb = pdb.heuristic(s)
                assert h_pdb <= result["cost"], \
                    f"PDB={h_pdb} > optimal={result['cost']} — admissibility violated!"

    def test_save_load(self, tmp_path):
        """PDB save/load roundtrip should give identical results (small groups)."""
        # Use a 3-3-3 partition to stay within memory limits in CI
        small_groups = [
            frozenset({1, 2, 3}),
            frozenset({4, 5, 6}),
            frozenset({7, 8, 9}),
        ]
        small_pdb = DisjointPDB(groups=small_groups, verbose=False)
        cache = tmp_path / "small_pdb.pkl"
        small_pdb.save(cache)
        assert cache.exists()
        loaded = DisjointPDB.load(cache)
        for seed in range(3):
            s = generate_instance("easy", seed=seed)
            assert small_pdb.heuristic(s) == loaded.heuristic(s)


# ---------------------------------------------------------------------------
# d_hat tests
# ---------------------------------------------------------------------------

class TestDHat:

    def test_identity_matches_h(self):
        h = manhattan_distance
        d_hat = identity_d_hat(h)
        for seed in range(5):
            s = generate_instance("easy", seed=seed)
            assert d_hat(s) == h(s)

    def test_inflated_is_larger(self):
        h = manhattan_distance
        d_hat = inflated_d_hat(h, factor=1.5)
        for seed in range(5):
            s = generate_instance("easy", seed=seed)
            assert d_hat(s) >= h(s)

    def test_inflated_factor_1_equals_h(self):
        h = manhattan_distance
        d_hat = inflated_d_hat(h, factor=1.0)
        for seed in range(5):
            s = generate_instance("easy", seed=seed)
            assert d_hat(s) == pytest.approx(h(s))

    def test_combined_alpha_1_equals_d(self):
        h = manhattan_distance
        d = manhattan_distance
        d_hat = combined_d_hat(d, h, alpha=1.0)
        for seed in range(5):
            s = generate_instance("easy", seed=seed)
            assert d_hat(s) == pytest.approx(d(s))


# ---------------------------------------------------------------------------
# Algorithm tests with Phase 2 heuristics
# ---------------------------------------------------------------------------

class TestAlgorithmsWithLC:
    """All algorithms should still solve correctly with Linear Conflict."""

    def setup_method(self):
        self.state = generate_instance("easy", seed=42)
        self.h     = linear_conflict
        self.w     = 1.5

    def test_wA_solves(self):
        r = weighted_astar(self.state, self.h, weight=self.w, timeout=30)
        assert r["solved"]

    def test_focal_solves(self):
        r = focal_search(self.state, self.h, d=self.h, weight=self.w, timeout=30)
        assert r["solved"]

    def test_ees_solves(self):
        r = ees(self.state, self.h, weight=self.w, timeout=30)
        assert r["solved"]

    def test_lc_expands_fewer_nodes_than_md(self):
        """A stronger heuristic should expand fewer or equal nodes."""
        r_md = weighted_astar(self.state, manhattan_distance, weight=1.0, timeout=30)
        r_lc = weighted_astar(self.state, linear_conflict,   weight=1.0, timeout=30)
        if r_md["solved"] and r_lc["solved"]:
            assert r_lc["nodes_expanded"] <= r_md["nodes_expanded"]


class TestAlgorithmsWithPDB:
    """All algorithms should solve correctly with PDB heuristic."""

    @pytest.fixture(autouse=True)
    def setup(self, pdb):
        self.state = generate_instance("easy", seed=42)
        self.h     = pdb.heuristic
        self.w     = 1.5

    def test_wA_solves(self):
        r = weighted_astar(self.state, self.h, weight=self.w, timeout=30)
        assert r["solved"]

    def test_focal_solves(self):
        r = focal_search(self.state, self.h, d=self.h, weight=self.w, timeout=30)
        assert r["solved"]

    def test_ees_solves(self):
        r = ees(self.state, self.h, weight=self.w, timeout=30)
        assert r["solved"]

    def test_pdb_expands_fewer_nodes_than_md(self):
        """PDB should expand significantly fewer nodes than MD on easy instances."""
        r_md  = weighted_astar(self.state, manhattan_distance, weight=1.0, timeout=30)
        r_pdb = weighted_astar(self.state, self.h,             weight=1.0, timeout=30)
        if r_md["solved"] and r_pdb["solved"]:
            assert r_pdb["nodes_expanded"] <= r_md["nodes_expanded"]
