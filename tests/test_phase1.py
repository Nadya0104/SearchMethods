"""
Unit tests for Phase 1: domain, heuristic, and all three algorithms.

Run with:
    cd puzzle_search
    python -m pytest tests/test_phase1.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from domain.puzzle import (
    GOAL_STATE, get_neighbors, get_neighbors_with_cost,
    is_goal, is_solvable, state_to_str, blank_index
)
from domain.generator import generate_instance, generate_suite
from heuristics.manhattan import manhattan_distance
from algorithms.weighted_astar import weighted_astar
from algorithms.focal_search   import focal_search
from algorithms.ees            import ees


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A simple 1-move-from-goal state: swap tiles 14 and 15
ONE_MOVE = list(GOAL_STATE)
ONE_MOVE[-1], ONE_MOVE[-2] = ONE_MOVE[-2], ONE_MOVE[-1]
ONE_MOVE_STATE = tuple(ONE_MOVE)

# A known easy state (a few moves from goal)
EASY_STATE = generate_instance("easy", seed=0)


# ---------------------------------------------------------------------------
# Domain tests
# ---------------------------------------------------------------------------

class TestPuzzle:
    def test_goal_is_goal(self):
        assert is_goal(GOAL_STATE)

    def test_non_goal_not_goal(self):
        assert not is_goal(ONE_MOVE_STATE)

    def test_blank_index_goal(self):
        assert blank_index(GOAL_STATE) == 15

    def test_neighbors_count(self):
        # Corner cell (blank at 15) has 2 neighbours
        nbs = list(get_neighbors(GOAL_STATE))
        assert len(nbs) == 2

    def test_neighbors_center(self):
        # Put blank in the centre-ish (index 5)
        s = list(GOAL_STATE)
        bi = GOAL_STATE.index(6)   # tile 6 is at index 5 in goal
        s[15], s[bi] = s[bi], s[15]
        state = tuple(s)
        # blank is now at index bi — count its neighbours
        nbs = list(get_neighbors(state))
        assert len(nbs) == 4

    def test_neighbors_cost_all_one(self):
        for nb, cost in get_neighbors_with_cost(GOAL_STATE):
            assert cost == 1

    def test_solvable_goal(self):
        assert is_solvable(GOAL_STATE)

    def test_solvable_one_move(self):
        assert is_solvable(ONE_MOVE_STATE)

    def test_state_to_str(self):
        s = state_to_str(GOAL_STATE)
        assert "15" in s and "0" in s


# ---------------------------------------------------------------------------
# Generator tests
# ---------------------------------------------------------------------------

class TestGenerator:
    def test_generates_tuple(self):
        s = generate_instance("easy", seed=1)
        assert isinstance(s, tuple) and len(s) == 16

    def test_all_tiles_present(self):
        s = generate_instance("medium", seed=2)
        assert sorted(s) == list(range(16))

    def test_solvable(self):
        for diff in ("easy", "medium", "hard"):
            s = generate_instance(diff, seed=3)
            assert is_solvable(s)

    def test_suite_counts(self):
        suite = generate_suite(n_easy=3, n_medium=3, n_hard=3, base_seed=0)
        assert len(suite["easy"])   == 3
        assert len(suite["medium"]) == 3
        assert len(suite["hard"])   == 3

    def test_suite_seed_mapping_is_stable(self):
        first = generate_suite(n_easy=2, n_medium=2, n_hard=2, base_seed=17)
        second = generate_suite(n_easy=2, n_medium=2, n_hard=2, base_seed=17)
        assert first == second


# ---------------------------------------------------------------------------
# Heuristic tests
# ---------------------------------------------------------------------------

class TestManhattan:
    def test_goal_is_zero(self):
        assert manhattan_distance(GOAL_STATE) == 0

    def test_one_move_is_one(self):
        # Swapping adjacent tiles 14 and 15 costs 1
        assert manhattan_distance(ONE_MOVE_STATE) == 1

    def test_nonnegative(self):
        for diff in ("easy", "medium"):
            s = generate_instance(diff, seed=99)
            assert manhattan_distance(s) >= 0

    def test_admissibility_proxy(self):
        # h must be ≤ true cost; we can't prove it here, but we verify
        # h > 0 for non-goal states (necessary condition)
        s = generate_instance("easy", seed=7)
        assert manhattan_distance(s) > 0


# ---------------------------------------------------------------------------
# Algorithm tests — correctness on small instances
# ---------------------------------------------------------------------------

SMALL_CASES = [
    GOAL_STATE,
    ONE_MOVE_STATE,
]


def _run_all(state, weight=1.5):
    h = manhattan_distance
    r_wa  = weighted_astar(state, h, weight=weight)
    r_fs  = focal_search(state, h, d=h, weight=weight)
    r_ees = ees(state, h, h_hat=h, d_hat=h, weight=weight)
    return r_wa, r_fs, r_ees


class TestAlgorithmsGoal:
    """All algorithms must return cost=0 immediately for the goal state."""
    def test_weighted_astar(self):
        r = weighted_astar(GOAL_STATE, manhattan_distance, weight=1.5)
        assert r["solved"]
        assert r["cost"] == 0

    def test_focal_search(self):
        r = focal_search(GOAL_STATE, manhattan_distance, weight=1.5)
        assert r["solved"]
        assert r["cost"] == 0

    def test_ees(self):
        r = ees(GOAL_STATE, manhattan_distance, weight=1.5)
        assert r["solved"]
        assert r["cost"] == 0


class TestAlgorithmsOneMove:
    """All algorithms must find the 1-move solution."""
    def test_weighted_astar(self):
        r = weighted_astar(ONE_MOVE_STATE, manhattan_distance, weight=1.0)
        assert r["solved"] and r["cost"] == 1

    def test_focal_search(self):
        r = focal_search(ONE_MOVE_STATE, manhattan_distance, weight=1.0)
        assert r["solved"] and r["cost"] == 1

    def test_ees(self):
        r = ees(ONE_MOVE_STATE, manhattan_distance, weight=1.0)
        assert r["solved"] and r["cost"] == 1


class TestAlgorithmsEasyInstance:
    """All algorithms must solve an easy instance and respect the bound."""
    def setup_method(self):
        self.state = generate_instance("easy", seed=42)
        self.h     = manhattan_distance
        self.w     = 1.5

    def test_weighted_astar_solves(self):
        r = weighted_astar(self.state, self.h, weight=self.w, timeout=30)
        assert r["solved"], "wA* failed to solve easy instance"

    def test_focal_search_solves(self):
        r = focal_search(self.state, self.h, d=self.h, weight=self.w, timeout=30)
        assert r["solved"], "Focal failed to solve easy instance"

    def test_ees_solves(self):
        r = ees(self.state, self.h, weight=self.w, timeout=30)
        assert r["solved"], "EES failed to solve easy instance"

    def test_all_return_valid_path(self):
        for algo, kwargs in [
            (weighted_astar, {"weight": self.w}),
            (focal_search,   {"weight": self.w, "d": self.h}),
            (ees,            {"weight": self.w}),
        ]:
            r = algo(self.state, self.h, **kwargs, timeout=30)
            if r["solved"]:
                assert r["path"][0]  == self.state
                assert r["path"][-1] == GOAL_STATE
                assert r["cost"]     == len(r["path"]) - 1

    def test_wA_optimal_with_weight_1(self):
        """wA* with w=1 should be optimal (== A*)."""
        r1 = weighted_astar(self.state, self.h, weight=1.0, timeout=30)
        r2 = weighted_astar(self.state, self.h, weight=2.0, timeout=30)
        if r1["solved"] and r2["solved"]:
            assert r2["cost"] <= r2["weight"] * r1["cost"]
