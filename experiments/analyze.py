"""
Analysis and plotting for bounded suboptimal search experiments.

Reads the CSV produced by harness.py and generates:

  1. nodes_vs_epsilon.png   — nodes expanded vs ε, grouped by algorithm,
                               one subplot per heuristic
  2. time_vs_epsilon.png    — runtime vs ε, same layout
  3. subopt_ratio.png       — actual suboptimality ratio vs ε (should stay ≤ ε)
  4. heuristic_comparison.png — nodes expanded by heuristic, at fixed ε
  5. solve_rate.png         — fraction of instances solved within timeout

Usage
-----
    cd puzzle_search
    python experiments/analyze.py experiments/results/results_<timestamp>.csv

    # save plots to a specific directory:
    python experiments/analyze.py results.csv --out-dir my_plots/
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

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

def mean_ci(series: pd.Series) -> tuple[float, float]:
    """Return (mean, half_width_of_95pct_CI)."""
    n  = series.dropna().count()
    if n == 0:
        return float("nan"), 0.0
    m  = series.mean()
    se = series.std(ddof=1) / np.sqrt(n)
    return m, 1.96 * se


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
            means, cis = [], []
            for eps in epsilons:
                m, ci = mean_ci(sub_a[sub_a["epsilon"] == eps][metric])
                means.append(m)
                cis.append(ci)

            ax.errorbar(
                epsilons, means, yerr=cis,
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
# Plot 3: actual suboptimality ratio vs epsilon
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
        ax.plot(epsilons, epsilons, "k--", linewidth=1, label="bound ε", alpha=0.5)

        for algo in ["weighted_astar", "focal_search", "ees"]:
            sub_a = sub_h[sub_h["algo"] == algo]
            if sub_a.empty:
                continue
            means, cis = [], []
            for eps in epsilons:
                m, ci = mean_ci(sub_a[sub_a["epsilon"] == eps]["subopt_ratio"])
                means.append(m)
                cis.append(ci)

            ax.errorbar(
                epsilons, means, yerr=cis,
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

    axes[0].set_ylabel("Actual cost / optimal cost", fontsize=10)
    axes[-1].legend(fontsize=9)
    fig.suptitle(f"Actual suboptimality ratio  [{difficulty} instances]",
                 fontsize=13, y=1.02)
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
    ylabel:     str = "Nodes expanded (mean)",
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
        means, cis = [], []
        for h_name in heuristics:
            m, ci = mean_ci(sub_a[sub_a["heuristic"] == h_name][metric])
            means.append(m)
            cis.append(ci)

        ax.bar(
            x + i * width,
            means,
            width,
            label=ALGO_LABELS[algo],
            color=ALGO_COLORS[algo],
            yerr=cis,
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
        print(f"  {'':18}" + "  nodes_expanded (mean)")
        print("  " + header)
        print("  " + "-" * len(header))

        sub_h = sub[sub["heuristic"] == h_name]
        for algo in ["weighted_astar", "focal_search", "ees"]:
            sub_a = sub_h[sub_h["algo"] == algo]
            row_str = f"  {ALGO_LABELS[algo]:<18}"
            for eps in epsilons:
                m, _ = mean_ci(sub_a[sub_a["epsilon"] == eps]["nodes_expanded"])
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
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir  = Path(args.out_dir) if args.out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {csv_path} ...")
    df = load(csv_path)
    print(f"  {len(df)} rows, columns: {list(df.columns)}")

    diff = args.difficulty

    print("\nGenerating plots ...")

    plot_metric_vs_epsilon(
        df, "nodes_expanded", "Nodes expanded (mean, log scale)",
        "Nodes expanded vs ε",
        out_dir / "nodes_vs_epsilon.png",
        difficulty=diff, log_y=True,
    )
    plot_metric_vs_epsilon(
        df, "elapsed_ms", "Runtime ms (mean, log scale)",
        "Runtime vs ε",
        out_dir / "time_vs_epsilon.png",
        difficulty=diff, log_y=True,
    )
    if df["subopt_ratio"].notna().any():
        plot_subopt_ratio(
            df, out_dir / "subopt_ratio.png", difficulty=diff,
        )
    plot_heuristic_comparison(
        df, out_dir / "heuristic_comparison.png",
        fixed_eps=args.fixed_eps, difficulty=diff,
    )
    plot_solve_rate(df, out_dir / "solve_rate.png")

    print_summary(df, difficulty=diff)
    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()