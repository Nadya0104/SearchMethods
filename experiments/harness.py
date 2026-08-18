"""
Experiment harness for the 15-puzzle bounded suboptimal search project.

Runs all combinations of:
    algorithms  × heuristics × epsilon values × instances (per difficulty)

and writes results to a CSV file.

CSV columns:
    algo          algorithm name
    heuristic     heuristic name
    epsilon       suboptimality bound used
    difficulty    easy / medium / hard
    instance_id   integer index within difficulty
    solved        True / False
    cost          solution cost (None if unsolved)
    optimal_cost  known optimal (None if not pre-computed)
    subopt_ratio  cost / optimal_cost  (None if either unknown)
    nodes_expanded
    nodes_generated
    elapsed_ms    wall time in milliseconds

Usage
-----
    cd puzzle_search
    python experiments/harness.py

    # with a pre-built PDB cache:
    python experiments/harness.py --pdb-cache pdb_cache.pkl

    # quick smoke-test (tiny suite):
    python experiments/harness.py --smoke
"""

from __future__ import annotations
import argparse
import csv
import json
import sys
import os
import time
from pathlib import Path
from typing import Callable

# Make sure project root is on path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from domain.generator      import generate_suite, Difficulty
from domain.puzzle         import GOAL_STATE
from heuristics.manhattan      import manhattan_distance
from heuristics.linear_conflict import linear_conflict
from heuristics.d_hat          import identity_d_hat
from algorithms.weighted_astar  import weighted_astar
from algorithms.focal_search    import focal_search
from algorithms.ees             import ees


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EPSILONS = [1.1, 1.5, 2.0, 3.0]

# Timeout per single run (seconds)
TIMEOUT = 60.0

# Default suite size per difficulty
DEFAULT_N = 20

# Output directory
RESULTS_DIR = Path(__file__).parent / "results"


def save_suite(
    suite: dict[str, list[tuple[int, ...]]],
    path: str | Path,
) -> None:
    """Persist the exact experimental states in a portable JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {difficulty: [list(state) for state in states]
               for difficulty, states in suite.items()}
    with open(path, "w", encoding="utf-8") as suite_file:
        json.dump(payload, suite_file, indent=2)


def load_suite(path: str | Path) -> dict[str, list[tuple[int, ...]]]:
    """Load an exact suite previously written by :func:`save_suite`."""
    with open(path, encoding="utf-8") as suite_file:
        payload = json.load(suite_file)
    return {difficulty: [tuple(state) for state in states]
            for difficulty, states in payload.items()}


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

def make_algorithms(
    h: Callable,
    epsilon: float,
) -> dict[str, Callable]:
    """
    Return a dict of algo_name → zero-arg callable that runs the algorithm
    and returns a result dict.  Captures h and epsilon via closure.
    """
    d_hat = identity_d_hat(h)
    return {
        "weighted_astar": lambda state: weighted_astar(
            state, h, weight=epsilon, timeout=TIMEOUT
        ),
        "focal_search": lambda state: focal_search(
            state, h, d=d_hat, weight=epsilon, timeout=TIMEOUT
        ),
        "ees": lambda state: ees(
            state, h, h_hat=h, d_hat=d_hat, weight=epsilon, timeout=TIMEOUT
        ),
    }


# ---------------------------------------------------------------------------
# Optimal cost lookup
# ---------------------------------------------------------------------------

def compute_optimal(
    state: tuple[int, ...],
    h: Callable,
    timeout: float = 120.0,
) -> int | None:
    """
    Compute optimal cost via A* (w=1).
    Returns None if not solved within timeout.
    Only feasible for easy instances and medium with strong heuristics.
    """
    r = weighted_astar(state, h, weight=1.0, timeout=timeout)
    return r["cost"] if r["solved"] else None


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_experiment(
    suite:         dict[str, list[tuple[int, ...]]],
    heuristics:    dict[str, Callable],
    epsilons:      list[float] = EPSILONS,
    output_path:   Path | None = None,
    compute_opt:   bool = True,
    opt_timeout:   float = 30.0,
    verbose:       bool = True,
) -> list[dict]:
    """
    Run all combinations and return a list of result dicts.
    Optionally writes to CSV as results come in (safe against crashes).

    Parameters
    ----------
    suite         : {difficulty: [state, ...]}
    heuristics    : {name: callable}
    epsilons      : list of suboptimality bounds
    output_path   : CSV file to write results to
    compute_opt   : whether to pre-compute optimal costs (slow for hard)
    opt_timeout   : timeout for optimal A* per instance
    verbose       : print progress
    """
    if output_path is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_path = RESULTS_DIR / f"results_{ts}.csv"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "algo", "heuristic", "epsilon", "difficulty", "instance_id",
        "solved", "cost", "optimal_cost", "subopt_ratio",
        "nodes_expanded", "nodes_generated", "elapsed_ms",
    ]

    all_rows: list[dict] = []

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Pre-compute optimal costs per instance (once, not per algo/epsilon)
        optimal_cache: dict[tuple[int, ...], int | None] = {}
        if compute_opt:
            if verbose:
                print("\n── Pre-computing optimal costs (A* w=1) ──")
            # Use the strongest available heuristic for optimal computation
            strong_h = next(reversed(list(heuristics.values())))
            for diff, states in suite.items():
                for idx, state in enumerate(states):
                    if state not in optimal_cache:
                        opt = compute_optimal(state, strong_h, timeout=opt_timeout)
                        optimal_cache[state] = opt
                        if verbose:
                            status = str(opt) if opt is not None else "timeout"
                            print(f"  {diff}[{idx}] optimal={status}")

        # Main experiment loop
        total = (3 * len(heuristics) * len(epsilons) *
                 sum(len(v) for v in suite.values()))
        done = 0

        if verbose:
            print(f"\n── Running {total} algorithm calls ──")

        for h_name, h_fn in heuristics.items():
            for epsilon in epsilons:
                algos = make_algorithms(h_fn, epsilon)

                for algo_name, algo_fn in algos.items():
                    for diff, states in suite.items():
                        for idx, state in enumerate(states):
                            result = algo_fn(state)
                            opt    = optimal_cache.get(state)

                            subopt_ratio = None
                            if result["solved"] and opt is not None and opt > 0:
                                subopt_ratio = round(result["cost"] / opt, 4)

                            row = {
                                "algo":            algo_name,
                                "heuristic":       h_name,
                                "epsilon":         epsilon,
                                "difficulty":      diff,
                                "instance_id":     idx,
                                "solved":          result["solved"],
                                "cost":            result["cost"],
                                "optimal_cost":    opt,
                                "subopt_ratio":    subopt_ratio,
                                "nodes_expanded":  result["nodes_expanded"],
                                "nodes_generated": result["nodes_generated"],
                                "elapsed_ms":      round(result["elapsed"] * 1000, 2),
                            }
                            writer.writerow(row)
                            csvfile.flush()      # safe against crashes
                            all_rows.append(row)

                            done += 1
                            if verbose and done % 20 == 0:
                                pct = 100 * done / total
                                print(f"  [{pct:5.1f}%] {algo_name} / {h_name} "
                                      f"/ ε={epsilon} / {diff}[{idx}]  "
                                      f"solved={result['solved']}  "
                                      f"nodes={result['nodes_expanded']}")

    if verbose:
        print(f"\n✓ Results written to {output_path}")

    return all_rows


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run bounded suboptimal search experiments on the 15-puzzle"
    )
    parser.add_argument(
        "--pdb-cache", type=str, default="pdb_cache.pkl",
        help="Path to pre-built PDB cache file (built on first run)"
    )
    parser.add_argument(
        "--no-pdb", action="store_true",
        help="Skip PDB heuristic (much faster, useful for quick tests)"
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Tiny run: 3 easy + 3 medium instances, no hard, no PDB"
    )
    parser.add_argument(
        "--n-easy",   type=int, default=DEFAULT_N)
    parser.add_argument(
        "--n-medium", type=int, default=DEFAULT_N)
    parser.add_argument(
        "--n-hard",   type=int, default=DEFAULT_N)
    parser.add_argument(
        "--no-optimal", action="store_true",
        help="Skip optimal cost computation (saves time)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path (default: experiments/results/results_<timestamp>.csv)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed for instance generation"
    )
    parser.add_argument(
        "--suite-input", type=str, default=None,
        help="Load exact instances from a JSON suite instead of generating them"
    )
    parser.add_argument(
        "--suite-output", type=str, default=None,
        help="Save the exact generated/loaded instances to this JSON file"
    )
    args = parser.parse_args()

    # ── Build suite ──────────────────────────────────────────────────────
    if args.suite_input:
        suite = load_suite(args.suite_input)
        use_pdb = not args.no_pdb and not args.smoke
        compute_opt = not args.no_optimal
        opt_timeout = 20.0 if args.smoke else 30.0
        print(f"Loaded fixed instance suite from {args.suite_input}")
    elif args.smoke:
        suite = generate_suite(n_easy=3, n_medium=3, n_hard=0, base_seed=args.seed)
        suite.pop("hard", None)
        use_pdb    = False
        compute_opt = True
        opt_timeout = 20.0
        print("SMOKE TEST MODE — tiny suite, no PDB, no hard instances")
    else:
        suite = generate_suite(
            n_easy=args.n_easy,
            n_medium=args.n_medium,
            n_hard=args.n_hard,
            base_seed=args.seed,
        )
        use_pdb    = not args.no_pdb
        compute_opt = not args.no_optimal
        opt_timeout = 30.0

    if args.suite_output:
        save_suite(suite, args.suite_output)
        print(f"Saved fixed instance suite to {args.suite_output}")

    # ── Build heuristics ──────────────────────────────────────────────────
    heuristics: dict[str, Callable] = {
        "manhattan":       manhattan_distance,
        "linear_conflict": linear_conflict,
    }

    if use_pdb:
        from heuristics.pdb import get_pdb
        print(f"\nLoading/building PDB from {args.pdb_cache} ...")
        pdb = get_pdb(args.pdb_cache, verbose=True)
        heuristics["pdb"] = pdb.heuristic

    # ── Run ──────────────────────────────────────────────────────────────
    output_path = Path(args.output) if args.output else None
    run_experiment(
        suite=suite,
        heuristics=heuristics,
        epsilons=EPSILONS,
        output_path=output_path,
        compute_opt=compute_opt,
        opt_timeout=opt_timeout,
        verbose=True,
    )


if __name__ == "__main__":
    main()
