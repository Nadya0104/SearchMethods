"""
Analysis and plotting for bounded suboptimal search experiments.

Reads the CSV produced by harness.py and generates:

  1. nodes_vs_epsilon.png   — nodes expanded vs ε, grouped by algorithm,
                               one subplot per heuristic
  2. time_vs_epsilon.png    — runtime vs ε, same layout
  3. subopt_ratio.png       — suboptimality ratio vs ε (should stay ≤ ε)
  4. pareto_nodes.png       — Pareto frontier: nodes expanded vs solution quality
  5. pareto_time.png        — Pareto frontier: runtime vs solution quality
  6. heuristic_comparison.png — nodes expanded by heuristic, at fixed ε
  7. solve_rate.png         — fraction of instances solved within timeout
  8. memory_footprint.png   — Python allocations per heuristic (requires a
                               pre-built PDB cache; skipped if unavailable)

Usage
-----
    cd puzzle_search
    python experiments/analyze.py experiments/results/results_<timestamp>.csv

    # save plots to a specific directory:
    python experiments/analyze.py results.csv --out-dir my_plots/

    # point at a specific PDB cache for the memory-footprint plot:
    python experiments/analyze.py results.csv --pdb-cache pdb_cache.pkl

    # skip the memory-footprint plot entirely:
    python experiments/analyze.py results.csv --skip-memory
"""

from __future__ import annotations
import argparse
import gc
import sys
import tracemalloc
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Make sure project root is on path when run directly (needed for the
# domain.* / heuristics.pdb imports used by the memory-footprint measurement)
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (safe everywhere)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

ALGO_COLORS = {
    "weighted_astar": "#4C72B0",   # blue
    "focal_search":   "#DD8452",   # orange
    "ees":            "#55A868",   # green
}
ALGO_LABELS = {
    "weighted_astar": "Weighted A*",
    "focal_search":   "Focal Search",
    "ees":            "EES",
}
HEURISTIC_LABELS = {
    "manhattan":       "Manhattan Distance",
    "linear_conflict": "Linear Conflict",
    "pdb":             "Pattern Database (5-5-5)",
}
MARKERS = {
    "weighted_astar": "o",
    "focal_search":   "s",
    "ees":            "^",
}
HEURISTIC_COLORS = {
    "manhattan":       "#4C72B0",   # blue
    "linear_conflict": "#DD8452",   # orange
    "pdb":             "#55A868",   # green
}

DIFFICULTIES = ["easy", "medium", "hard"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Coerce numeric columns that may have been written as strings
    for col in ["cost", "optimal_cost", "subopt_ratio",
                "nodes_expanded", "nodes_generated", "elapsed_ms"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Helper: aggregate mean ± 95% CI
# ---------------------------------------------------------------------------

def mean_ci(series: pd.Series) -> tuple[float, float, float]:
    """
    Return (geometric_mean, err_low, err_high) for a 95% CI.

    Computed in log-space: nodes_expanded / elapsed_ms / subopt_ratio are
    strictly positive and right-skewed (a handful of hard instances can be
    orders of magnitude larger than the rest). A symmetric CI on raw values
    (mean ± 1.96·SE) can go negative under that skew, which is invisible on
    a linear axis but silently clips to the bottom of a log-scale axis.
    Building the CI in log-space keeps both bounds positive and reflects
    the multiplicative nature of the variability.
    """
    s = series.dropna()
    s = s[s > 0]
    n = s.count()
    if n == 0:
        return float("nan"), 0.0, 0.0
    if n == 1:
        v = float(s.iloc[0])
        return v, 0.0, 0.0

    log_s    = np.log(s)
    log_mean = log_s.mean()
    log_se   = log_s.std(ddof=1) / np.sqrt(n)
    log_half = 1.96 * log_se

    geo_mean = float(np.exp(log_mean))
    lower    = float(np.exp(log_mean - log_half))
    upper    = float(np.exp(log_mean + log_half))
    return geo_mean, geo_mean - lower, upper - geo_mean


# ---------------------------------------------------------------------------
# Plot 1 & 2: nodes / time vs epsilon, one subplot per heuristic
# ---------------------------------------------------------------------------

def plot_metric_vs_epsilon(
    df:         pd.DataFrame,
    metric:     str,            # "nodes_expanded" or "elapsed_ms"
    ylabel:     str,
    title:      str,
    out_path:   Path,
    difficulty: str = "medium",
    log_y:      bool = True,
) -> None:
    heuristics = [h for h in ["manhattan", "linear_conflict", "pdb"]
                  if h in df["heuristic"].unique()]
    n_cols = len(heuristics)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4), sharey=True)
    if n_cols == 1:
        axes = [axes]

    sub = df[df["difficulty"] == difficulty]

    for ax, h_name in zip(axes, heuristics):
        sub_h = sub[sub["heuristic"] == h_name]
        for algo in ["weighted_astar", "focal_search", "ees"]:
            sub_a = sub_h[sub_h["algo"] == algo]
            if sub_a.empty:
                continue
            epsilons = sorted(sub_a["epsilon"].unique())
            means, err_lo, err_hi = [], [], []
            for eps in epsilons:
                m, lo, hi = mean_ci(sub_a[sub_a["epsilon"] == eps][metric])
                means.append(m)
                err_lo.append(lo)
                err_hi.append(hi)

            ax.errorbar(
                epsilons, means, yerr=[err_lo, err_hi],
                label=ALGO_LABELS[algo],
                color=ALGO_COLORS[algo],
                marker=MARKERS[algo],
                linewidth=1.8,
                markersize=6,
                capsize=4,
            )

        ax.set_title(HEURISTIC_LABELS.get(h_name, h_name), fontsize=11)
        ax.set_xlabel("Suboptimality bound ε", fontsize=10)
        if log_y:
            ax.set_yscale("log")
        ax.set_xticks(epsilons)
        ax.grid(True, which="both", alpha=0.3, linestyle="--")

    axes[0].set_ylabel(ylabel, fontsize=10)
    axes[-1].legend(fontsize=9, loc="upper right")
    fig.suptitle(f"{title}  [{difficulty} instances]", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: suboptimality ratio vs epsilon
# ---------------------------------------------------------------------------

def plot_subopt_ratio(
    df:       pd.DataFrame,
    out_path: Path,
    difficulty: str = "medium",
) -> None:
    heuristics = [h for h in ["manhattan", "linear_conflict", "pdb"]
                  if h in df["heuristic"].unique()]
    n_cols = len(heuristics)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4), sharey=True)
    if n_cols == 1:
        axes = [axes]

    sub = df[(df["difficulty"] == difficulty) & df["subopt_ratio"].notna()]
    epsilons = sorted(df["epsilon"].unique())

    for ax, h_name in zip(axes, heuristics):
        sub_h = sub[sub["heuristic"] == h_name]

        # Draw the theoretical bound line
        ax.plot(
            epsilons, epsilons, "k--", linewidth=1,
            label="Theoretical bound ε", alpha=0.5,
        )

        for algo in ["weighted_astar", "focal_search", "ees"]:
            sub_a = sub_h[sub_h["algo"] == algo]
            if sub_a.empty:
                continue
            means, err_lo, err_hi = [], [], []
            for eps in epsilons:
                m, lo, hi = mean_ci(sub_a[sub_a["epsilon"] == eps]["subopt_ratio"])
                means.append(m)
                err_lo.append(lo)
                err_hi.append(hi)

            ax.errorbar(
                epsilons, means, yerr=[err_lo, err_hi],
                label=ALGO_LABELS[algo],
                color=ALGO_COLORS[algo],
                marker=MARKERS[algo],
                linewidth=1.8,
                markersize=6,
                capsize=4,
            )

        ax.set_title(HEURISTIC_LABELS.get(h_name, h_name), fontsize=11)
        ax.set_xlabel("ε", fontsize=10)
        ax.set_xticks(epsilons)
        ax.grid(True, alpha=0.3, linestyle="--")

    axes[0].set_ylabel("Returned cost / optimal cost", fontsize=10)
    axes[-1].legend(fontsize=9)
    fig.suptitle(f"Suboptimality ratio by ε  [{difficulty} instances]",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 3b / 3c: Pareto frontier — effort vs solution quality, parameterized by ε
# ---------------------------------------------------------------------------

def plot_pareto_frontier(
    df:         pd.DataFrame,
    metric:     str,            # "nodes_expanded" or "elapsed_ms"
    ylabel:     str,
    title:      str,
    out_path:   Path,
    difficulty: str = "medium",
    log_y:      bool = True,
    memory_mb:  dict[str, float] | None = None,
) -> None:
    """
    Pareto frontier: x = geometric-mean actual suboptimality ratio,
    y = geometric-mean search effort. One panel per heuristic, one curve per algorithm, points
    connected in increasing-ε order and labeled with their ε value.
    Lower-left is better (cheap AND close to optimal); points on the
    lower-left envelope of a panel dominate the rest.
    """
    heuristics = [h for h in ["manhattan", "linear_conflict", "pdb"]
                  if h in df["heuristic"].unique()]
    n_cols = len(heuristics)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4.5), sharey=True)
    if n_cols == 1:
        axes = [axes]

    sub = df[(df["difficulty"] == difficulty) & df["subopt_ratio"].notna()]
    epsilons = sorted(df["epsilon"].unique())

    for ax, h_name in zip(axes, heuristics):
        sub_h = sub[sub["heuristic"] == h_name]
        for algo in ["weighted_astar", "focal_search", "ees"]:
            sub_a = sub_h[sub_h["algo"] == algo]
            if sub_a.empty:
                continue
            xs, ys, eps_used = [], [], []
            for eps in epsilons:
                sub_e = sub_a[sub_a["epsilon"] == eps]
                if sub_e.empty:
                    continue
                x_mean, _, _ = mean_ci(sub_e["subopt_ratio"])
                y_mean, _, _ = mean_ci(sub_e[metric])
                xs.append(x_mean)
                ys.append(y_mean)
                eps_used.append(eps)

            if not xs:
                continue

            ax.plot(
                xs, ys,
                label=ALGO_LABELS[algo],
                color=ALGO_COLORS[algo],
                marker=MARKERS[algo],
                linewidth=1.8,
                markersize=7,
            )
            for x, y, eps in zip(xs, ys, eps_used):
                ax.annotate(
                    f"{eps:g}", (x, y),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=7, color=ALGO_COLORS[algo], alpha=0.85,
                )

        panel_title = HEURISTIC_LABELS.get(h_name, h_name)
        if memory_mb and h_name in memory_mb:
            mb = memory_mb[h_name]
            mem_label = f"{mb * 1024:.0f} KB" if mb < 1 else f"{mb:.0f} MB"
            panel_title += f"\n({mem_label} allocated)"
        ax.set_title(panel_title, fontsize=11)
        ax.set_xlabel("Returned cost / optimal cost", fontsize=9)
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3, linestyle="--")

    axes[0].set_ylabel(ylabel, fontsize=10)
    axes[-1].legend(fontsize=9, loc="upper right")
    fig.suptitle(
        f"{title}  [{difficulty} instances]",
        fontsize=12, y=1.03,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 4: heuristic comparison at fixed epsilon
# ---------------------------------------------------------------------------

def plot_heuristic_comparison(
    df:         pd.DataFrame,
    out_path:   Path,
    metric:     str = "nodes_expanded",
    ylabel:     str = "Nodes expanded (geometric mean)",
    fixed_eps:  float = 1.5,
    difficulty: str = "medium",
) -> None:
    heuristics = [h for h in ["manhattan", "linear_conflict", "pdb"]
                  if h in df["heuristic"].unique()]
    algos = ["weighted_astar", "focal_search", "ees"]

    sub = df[(df["difficulty"] == difficulty) & (df["epsilon"] == fixed_eps)]

    x = np.arange(len(heuristics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4))

    for i, algo in enumerate(algos):
        sub_a = sub[sub["algo"] == algo]
        means, err_lo, err_hi = [], [], []
        for h_name in heuristics:
            m, lo, hi = mean_ci(sub_a[sub_a["heuristic"] == h_name][metric])
            means.append(m)
            err_lo.append(lo)
            err_hi.append(hi)

        ax.bar(
            x + i * width,
            means,
            width,
            label=ALGO_LABELS[algo],
            color=ALGO_COLORS[algo],
            yerr=[err_lo, err_hi],
            capsize=4,
            alpha=0.85,
        )

    ax.set_yscale("log")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f"Heuristic comparison at ε={fixed_eps}  [{difficulty} instances]",
                 fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels([HEURISTIC_LABELS.get(h, h) for h in heuristics], fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 5: solve rate
# ---------------------------------------------------------------------------

def plot_solve_rate(
    df:       pd.DataFrame,
    out_path: Path,
) -> None:
    heuristics = [h for h in ["manhattan", "linear_conflict", "pdb"]
                  if h in df["heuristic"].unique()]
    algos = ["weighted_astar", "focal_search", "ees"]
    diffs = [d for d in DIFFICULTIES if d in df["difficulty"].unique()]
    epsilons = sorted(df["epsilon"].unique())

    n_rows = len(diffs)
    n_cols = len(heuristics)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 3.5 * n_rows),
                             sharey=True, sharex=True)
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]

    for r, diff in enumerate(diffs):
        for c, h_name in enumerate(heuristics):
            ax = axes[r][c]
            sub = df[(df["difficulty"] == diff) & (df["heuristic"] == h_name)]

            for algo in algos:
                sub_a = sub[sub["algo"] == algo]
                rates = []
                for eps in epsilons:
                    sub_e = sub_a[sub_a["epsilon"] == eps]
                    rate  = sub_e["solved"].mean() if not sub_e.empty else 0.0
                    rates.append(rate)

                ax.plot(
                    epsilons, rates,
                    label=ALGO_LABELS[algo],
                    color=ALGO_COLORS[algo],
                    marker=MARKERS[algo],
                    linewidth=1.8,
                    markersize=6,
                )

            ax.set_ylim(0, 1.05)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticks(epsilons)
            ax.grid(True, alpha=0.3, linestyle="--")

            if r == 0:
                ax.set_title(HEURISTIC_LABELS.get(h_name, h_name), fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{diff}\nSolve rate", fontsize=9)
            if r == n_rows - 1:
                ax.set_xlabel("ε", fontsize=9)

    axes[0][-1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Solve rate within timeout", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Memory footprint — traced Python allocations per heuristic (independent of the CSV;
# measured directly by loading each heuristic's persistent data structures)
# ---------------------------------------------------------------------------

def deep_getsizeof(obj, seen: set[int] | None = None) -> int:
    """
    Recursively sum sys.getsizeof over a container, without double-counting
    shared objects (e.g. CPython's cached small ints / interned strings).
    Exact but only practical for small object graphs.
    """
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            size += deep_getsizeof(k, seen)
            size += deep_getsizeof(v, seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            size += deep_getsizeof(item, seen)
    return size


def measure_pdb_memory(cache_path: str | Path) -> tuple[float, float, int]:
    """Return (allocated_MB, disk_MB, n_entries) for the Disjoint PDB."""
    from heuristics.pdb import DisjointPDB

    disk_bytes = Path(cache_path).stat().st_size

    gc.collect()
    tracemalloc.start()
    pdb = DisjointPDB.load(cache_path)
    allocated_bytes, _peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n_entries = sum(len(d) for d in pdb._dbs)
    return allocated_bytes / (1024 ** 2), disk_bytes / (1024 ** 2), n_entries


def measure_memory_footprint(pdb_cache: str | Path) -> dict[str, float]:
    """
    Measure traced Python allocations (MB) for each heuristic, keyed the same way as
    the CSV's `heuristic` column so the result can feed straight into the
    Pareto plots' `memory_mb` annotation.

    Manhattan Distance and Linear Conflict hold no precomputed lookup table
    of their own -- both only read domain.puzzle.GOAL_POS, a 16-entry dict
    shared by the whole project (KB-scale). The Disjoint PDB holds three
    large dict-based lookup tables (millions of entries) that must stay
    available for O(1) heuristic lookups. `tracemalloc` measures the Python
    allocations made while loading them; this is not an operating-system RSS
    measurement.
    """
    from domain.puzzle import GOAL_POS

    md_lc_mb = deep_getsizeof(GOAL_POS) / (1024 ** 2)
    pdb_mb, disk_mb, n_entries = measure_pdb_memory(pdb_cache)

    print(f"  Manhattan Distance : {md_lc_mb * 1024:.2f} KB  (GOAL_POS lookup table)")
    print(f"  Linear Conflict    : {md_lc_mb * 1024:.2f} KB  (same GOAL_POS table)")
    print(f"  Pattern Database   : {pdb_mb:.1f} MB allocated  "
          f"({disk_mb:.1f} MB on disk, {n_entries:,} entries)")

    return {"manhattan": md_lc_mb, "linear_conflict": md_lc_mb, "pdb": pdb_mb}


def plot_memory_footprint(memory_mb: dict[str, float], out_path: Path) -> None:
    heuristics = [h for h in ["manhattan", "linear_conflict", "pdb"] if h in memory_mb]
    labels = [HEURISTIC_LABELS.get(h, h) for h in heuristics]
    mem_mb  = [memory_mb[h] for h in heuristics]
    colors  = [HEURISTIC_COLORS.get(h, "#888888") for h in heuristics]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, mem_mb, color=colors, alpha=0.85)
    ax.set_yscale("log")
    ax.set_ylabel("Allocated Python memory (MB, log scale)", fontsize=10)
    ax.set_title("Heuristic memory footprint (tracemalloc)", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    for bar, mb in zip(bars, mem_mb):
        label = f"{mb * 1024:.1f} KB" if mb < 1 else f"{mb:.1f} MB"
        ax.annotate(
            label,
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            textcoords="offset points", xytext=(0, 5),
            ha="center", fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, difficulty: str = "medium") -> None:
    print(f"\n{'='*70}")
    print(f"Summary  [{difficulty} instances]")
    print(f"{'='*70}")

    sub = df[df["difficulty"] == difficulty]
    heuristics = [h for h in ["manhattan", "linear_conflict", "pdb"]
                  if h in df["heuristic"].unique()]
    epsilons = sorted(df["epsilon"].unique())

    for h_name in heuristics:
        print(f"\nHeuristic: {HEURISTIC_LABELS.get(h_name, h_name)}")
        header = f"{'Algorithm':<18}" + "".join(f"  ε={e:<5}" for e in epsilons)
        print(f"  {'':18}" + "  nodes_expanded (geometric mean)")
        print("  " + header)
        print("  " + "-" * len(header))

        sub_h = sub[sub["heuristic"] == h_name]
        for algo in ["weighted_astar", "focal_search", "ees"]:
            sub_a = sub_h[sub_h["algo"] == algo]
            row_str = f"  {ALGO_LABELS[algo]:<18}"
            for eps in epsilons:
                m, _, _ = mean_ci(sub_a[sub_a["epsilon"] == eps]["nodes_expanded"])
                row_str += f"  {m:>7.0f}"
            print(row_str)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse results CSV and produce plots"
    )
    parser.add_argument("csv", help="Path to results CSV from harness.py")
    parser.add_argument(
        "--out-dir", default=None,
        help="Directory for output plots (default: same dir as CSV)"
    )
    parser.add_argument(
        "--difficulty", default="medium",
        choices=["easy", "medium", "hard"],
        help="Which difficulty to focus on for per-heuristic plots"
    )
    parser.add_argument(
        "--fixed-eps", type=float, default=1.5,
        help="Fixed epsilon for heuristic comparison bar chart"
    )
    parser.add_argument(
        "--pdb-cache", default="pdb_cache.pkl",
        help="Path to a pre-built PDB cache, used for the memory-footprint "
             "plot and the Pareto plots' memory annotations"
    )
    parser.add_argument(
        "--skip-memory", action="store_true",
        help="Skip the memory-footprint measurement/plot (e.g. no PDB cache built yet)"
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir  = Path(args.out_dir) if args.out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {csv_path} ...")
    df = load(csv_path)
    print(f"  {len(df)} rows, columns: {list(df.columns)}")

    diff = args.difficulty

    memory_mb: dict[str, float] | None = None
    if not args.skip_memory:
        print(f"\nMeasuring heuristic memory footprint (PDB cache: {args.pdb_cache}) ...")
        try:
            memory_mb = measure_memory_footprint(args.pdb_cache)
        except FileNotFoundError:
            print(f"  PDB cache not found at {args.pdb_cache!r} -- skipping "
                  f"memory-footprint plot (pass --pdb-cache or --skip-memory)")

    print("\nGenerating plots ...")

    plot_metric_vs_epsilon(
        df, "nodes_expanded", "Nodes expanded (geometric mean, log scale)",
        "Nodes expanded vs ε",
        out_dir / "nodes_vs_epsilon.png",
        difficulty=diff, log_y=True,
    )
    plot_metric_vs_epsilon(
        df, "elapsed_ms", "Runtime ms (geometric mean, log scale)",
        "Runtime vs ε",
        out_dir / "time_vs_epsilon.png",
        difficulty=diff, log_y=True,
    )
    if df["subopt_ratio"].notna().any():
        plot_subopt_ratio(
            df, out_dir / "subopt_ratio.png", difficulty=diff,
        )
        plot_pareto_frontier(
            df, "nodes_expanded", "Nodes expanded (geometric mean, log scale)",
            "Nodes expanded vs. solution quality",
            out_dir / "pareto_nodes.png",
            difficulty=diff, log_y=True, memory_mb=memory_mb,
        )
        plot_pareto_frontier(
            df, "elapsed_ms", "Runtime ms (geometric mean, log scale)",
            "Runtime vs. solution quality",
            out_dir / "pareto_time.png",
            difficulty=diff, log_y=True, memory_mb=memory_mb,
        )
    if memory_mb is not None:
        plot_memory_footprint(memory_mb, out_dir / "memory_footprint.png")
    plot_heuristic_comparison(
        df, out_dir / "heuristic_comparison.png",
        fixed_eps=args.fixed_eps, difficulty=diff,
    )
    plot_solve_rate(df, out_dir / "solve_rate.png")

    print_summary(df, difficulty=diff)
    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()
