"""
Tests for Phase 3: harness and analysis pipeline.

Run with:
    cd puzzle_search
    python -m pytest tests/test_phase3.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import csv
import pytest
import tempfile
from pathlib import Path

import pandas as pd

from domain.generator import generate_suite
from heuristics.manhattan       import manhattan_distance
from heuristics.linear_conflict import linear_conflict
from experiments.harness import run_experiment, make_algorithms, EPSILONS
from experiments.analyze import (
    load, mean_ci, plot_metric_vs_epsilon, plot_heuristic_comparison,
    plot_solve_rate, print_summary,
)


# ---------------------------------------------------------------------------
# Tiny smoke suite shared across tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def smoke_suite():
    return generate_suite(n_easy=3, n_medium=3, n_hard=0, base_seed=0)


@pytest.fixture(scope="module")
def smoke_heuristics():
    return {
        "manhattan":       manhattan_distance,
        "linear_conflict": linear_conflict,
    }


@pytest.fixture(scope="module")
def smoke_results(tmp_path_factory, smoke_suite, smoke_heuristics):
    """Run a tiny experiment and return (rows, csv_path)."""
    out = tmp_path_factory.mktemp("results") / "smoke.csv"
    rows = run_experiment(
        suite=smoke_suite,
        heuristics=smoke_heuristics,
        epsilons=[1.5, 2.0],           # only 2 epsilons to keep it fast
        output_path=out,
        compute_opt=True,
        opt_timeout=15.0,
        verbose=False,
    )
    return rows, out


# ---------------------------------------------------------------------------
# Harness tests
# ---------------------------------------------------------------------------

class TestHarness:

    def test_returns_list(self, smoke_results):
        rows, _ = smoke_results
        assert isinstance(rows, list) and len(rows) > 0

    def test_csv_exists(self, smoke_results):
        _, csv_path = smoke_results
        assert csv_path.exists()

    def test_csv_has_expected_columns(self, smoke_results):
        _, csv_path = smoke_results
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames
        expected = {
            "algo", "heuristic", "epsilon", "difficulty", "instance_id",
            "solved", "cost", "optimal_cost", "subopt_ratio",
            "nodes_expanded", "nodes_generated", "elapsed_ms",
        }
        assert expected.issubset(set(cols))

    def test_row_count(self, smoke_results, smoke_suite, smoke_heuristics):
        rows, _ = smoke_results
        n_algos    = 3
        n_epsilons = 2
        n_instances = sum(len(v) for v in smoke_suite.values())
        n_heuristics = len(smoke_heuristics)
        expected = n_algos * n_epsilons * n_instances * n_heuristics
        assert len(rows) == expected

    def test_all_algos_present(self, smoke_results):
        rows, _ = smoke_results
        algos = {r["algo"] for r in rows}
        assert algos == {"weighted_astar", "focal_search", "ees"}

    def test_all_heuristics_present(self, smoke_results, smoke_heuristics):
        rows, _ = smoke_results
        heuristics = {r["heuristic"] for r in rows}
        assert heuristics == set(smoke_heuristics.keys())

    def test_solved_flag_is_bool_string(self, smoke_results):
        rows, _ = smoke_results
        for r in rows:
            assert r["solved"] in (True, False)

    def test_easy_instances_all_solved(self, smoke_results):
        rows, _ = smoke_results
        easy_rows = [r for r in rows if r["difficulty"] == "easy"]
        assert all(r["solved"] for r in easy_rows), \
            "Some easy instances were not solved"

    def test_cost_nonnegative_when_solved(self, smoke_results):
        rows, _ = smoke_results
        for r in rows:
            if r["solved"]:
                assert r["cost"] >= 0

    def test_subopt_ratio_within_bound(self, smoke_results):
        """Actual ratio must be ≤ ε (suboptimality guarantee)."""
        rows, _ = smoke_results
        for r in rows:
            if r["subopt_ratio"] is not None:
                assert r["subopt_ratio"] <= r["epsilon"] + 1e-6, \
                    (f"{r['algo']} violated bound: ratio={r['subopt_ratio']} "
                     f"> ε={r['epsilon']}")

    def test_nodes_expanded_positive(self, smoke_results):
        rows, _ = smoke_results
        for r in rows:
            assert r["nodes_expanded"] >= 1

    def test_elapsed_ms_positive(self, smoke_results):
        rows, _ = smoke_results
        for r in rows:
            assert r["elapsed_ms"] >= 0

    def test_make_algorithms_returns_three(self):
        algos = make_algorithms(manhattan_distance, epsilon=1.5)
        assert set(algos.keys()) == {"weighted_astar", "focal_search", "ees"}

    def test_make_algorithms_callable(self):
        state = generate_suite(n_easy=1, n_medium=0, n_hard=0)["easy"][0]
        algos = make_algorithms(manhattan_distance, epsilon=2.0)
        for name, fn in algos.items():
            r = fn(state)
            assert "solved" in r, f"{name} missing 'solved' key"


# ---------------------------------------------------------------------------
# Analysis tests
# ---------------------------------------------------------------------------

class TestAnalysis:

    @pytest.fixture(autouse=True)
    def df(self, smoke_results):
        _, csv_path = smoke_results
        self.df = load(csv_path)

    def test_load_returns_dataframe(self):
        assert isinstance(self.df, pd.DataFrame)
        assert len(self.df) > 0

    def test_numeric_columns(self):
        for col in ["nodes_expanded", "elapsed_ms", "epsilon"]:
            assert pd.api.types.is_numeric_dtype(self.df[col]), \
                f"{col} is not numeric"

    def test_mean_ci_basic(self):
        m, ci = mean_ci(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert abs(m - 3.0) < 1e-9
        assert ci >= 0

    def test_mean_ci_empty(self):
        m, ci = mean_ci(pd.Series([], dtype=float))
        import math
        assert math.isnan(m)
        assert ci == 0.0

    def test_plot_nodes_vs_epsilon(self, tmp_path):
        out = tmp_path / "nodes.png"
        plot_metric_vs_epsilon(
            self.df, "nodes_expanded", "Nodes", "Title", out,
            difficulty="easy", log_y=True,
        )
        assert out.exists()

    def test_plot_heuristic_comparison(self, tmp_path):
        out = tmp_path / "hcomp.png"
        plot_heuristic_comparison(self.df, out, fixed_eps=1.5, difficulty="easy")
        assert out.exists()

    def test_plot_solve_rate(self, tmp_path):
        out = tmp_path / "solve.png"
        plot_solve_rate(self.df, out)
        assert out.exists()

    def test_print_summary_runs(self, capsys):
        print_summary(self.df, difficulty="easy")
        captured = capsys.readouterr()
        assert "manhattan" in captured.out.lower() or "Summary" in captured.out