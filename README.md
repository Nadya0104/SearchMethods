# Bounded Suboptimal Search on the 15-Puzzle

**Comparing Weighted A\*, Focal Search (A\*ε), and Explicit Estimation Search (EES) across three admissible heuristic families**

---

## Abstract

Bounded suboptimal search algorithms trade solution quality for speed by
relaxing A\*'s optimality guarantee to a bound `w`: any solution returned
has cost at most `w · optimal`. This project implements and empirically
compares three such algorithms — **Weighted A\*** (Pohl, 1970),
**Focal Search / A\*ε** (Pearl & Kim, 1982), and **Explicit Estimation
Search** (Thayer & Ruml, 2011) — on the 15-puzzle, crossed with three
heuristics of increasing strength: **Manhattan Distance**, **Linear
Conflict**, and a **disjoint Pattern Database**. The goal is to isolate
how much of the algorithms' performance gap is explained by algorithm
choice versus heuristic quality, and to check whether EES's advantage
over Weighted A\* - which comes from distinguishing cost-to-go from
distance-to-go - actually materializes on a domain where cost and
distance coincide (unit-cost moves).

---

## Background

Given a search problem and a suboptimality bound `w ≥ 1`, a bounded
suboptimal algorithm must return a solution of cost ≤ `w · C*`, where
`C*` is the optimal cost. All three algorithms studied here provide this
guarantee, but they exploit it differently:

| Algorithm | Idea | Extra heuristic needed |
|---|---|---|
| **Weighted A\*** | Inflate `h` by `w` in `f = g + w·h`; greedier search, same guarantee as A\* scaled by `w`. | none |
| **Focal Search (A\*ε)** | Keep exploring OPEN by `f`, but expand from a FOCAL sublist of all nodes within `w · f_min`, chosen to minimize a secondary metric `d` (distance-to-go). | `d` (can reuse `h`) |
| **EES** | Maintain three synchronized queues (OPEN by `f̂`, FOCAL by `d̂`, CLEANUP by `f`) so it can pursue nodes that look close to the goal while still using the *admissible* `f` to certify the bound. | `ĥ`, `d̂` (inadmissible estimates) |

Because the 15-puzzle has unit-cost moves, cost-to-go and distance-to-go
are the same quantity — the exact scenario the EES paper identifies as
the *least* favorable for EES relative to Weighted A\*. This makes the
puzzle a useful negative/control case, not just a convenient benchmark.

### Heuristics

- **Manhattan Distance (MD)** — sum of per-tile grid distances to the goal. Admissible, consistent, cheap.
- **Linear Conflict (LC)** — MD plus a +2 penalty per pair of tiles that block each other in a shared row/column ([Hansson, Mayer & Yung, 1992](https://doi.org/10.1016/0004-3702(92)90015-3)). Strictly dominates MD.
- **Pattern Database (PDB)** — exact goal-distances for disjoint 5-5-5 tile groups, precomputed by backward 0-1 BFS from the goal and summed. Strictly dominates MD; incomparable with LC on individual states.

---

## Repository Structure

```
SearchMethods/
├── domain/
│   ├── puzzle.py             # State representation, moves, goal test, solvability
│   └── generator.py          # Random-walk instance generation by difficulty
├── heuristics/
│   ├── manhattan.py          # Manhattan Distance
│   ├── linear_conflict.py    # Linear Conflict (MD + row/col penalties)
│   ├── pdb.py                # Disjoint Pattern Database (5-5-5 partition)
│   └── d_hat.py              # Inadmissible distance-to-go estimators for EES
├── algorithms/
│   ├── weighted_astar.py     # Weighted A*  (f = g + w·h)
│   ├── focal_search.py       # Focal Search (OPEN + FOCAL lists)
│   └── ees.py                # Explicit Estimation Search (OPEN + FOCAL + CLEANUP)
├── experiments/
│   ├── harness.py            # Runs all algorithm × heuristic × ε × instance combinations
│   ├── analyze.py            # Loads results CSV, produces plots and summary tables
│   └── results/               # Generated CSVs and figures
├── tests/
│   ├── test_phase1.py        # Domain, heuristics, algorithms (28 tests)
│   ├── test_phase2.py        # Linear Conflict, PDB, d̂ correctness (30 tests)
│   └── test_phase3.py        # Harness + analysis pipeline (22 tests)
└── requirements.txt
```

---

## Setup

```bash
git clone <repo-url>
cd SearchMethods
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # pytest, matplotlib, pandas
```

## Tests

```bash
python -m pytest tests/ -v          # all 84 tests, ~1 minute

python -m pytest tests/test_phase1.py -v   # ~10s  — domain + algorithms
python -m pytest tests/test_phase2.py -v   # ~60s  — heuristics (builds PDB)
python -m pytest tests/test_phase3.py -v   # ~40s  — harness + plots
```

---

## Reproducing the Experiments

### 1. Smoke test

Sanity-checks the full pipeline with no PDB and a handful of instances (~1 minute):

```bash
python experiments/harness.py --smoke
```

### 2. Full experiment

```bash
python experiments/harness.py --pdb-cache pdb_cache.pkl
```

1. Builds the PDB on first run (~50s) and caches it to `pdb_cache.pkl` (subsequent runs load in ~1s).
2. Pre-computes optimal costs via A\* (`w=1`) per instance, to populate `subopt_ratio`.
3. Runs all combinations: **3 algorithms × 3 heuristics × 4 ε values × 20 instances × 3 difficulties = 2,160 solver calls.**
4. Writes results incrementally to CSV — safe to interrupt.

Expected runtime: 30–90 minutes (tight bounds with MD are the slowest cell).

### 3. Fast iteration (skip the PDB)

```bash
python experiments/harness.py --no-pdb
```

3 algorithms × 2 heuristics (MD, LC) × 4 ε × 60 instances = 1,440 calls, ~10–20 minutes.

### Other options

```bash
python experiments/harness.py --n-easy 30 --n-medium 30 --n-hard 30   # instance counts per difficulty
python experiments/harness.py --no-optimal                            # skip optimal-cost pass
python experiments/harness.py --output my_results.csv                 # custom output path
python experiments/harness.py --seed 123                              # reproducibility
```

Default ε sweep: `[1.1, 1.5, 2.0, 3.0]`. Per-cell timeout: 60s.

---

## Analysis

```bash
python experiments/analyze.py experiments/results/results_<timestamp>.csv
```

| Figure | What it shows |
|---|---|
| `nodes_vs_epsilon.png` | Nodes expanded vs. ε, one subplot per heuristic, one line per algorithm |
| `time_vs_epsilon.png` | Wall-clock time vs. ε, same layout |
| `subopt_ratio.png` | Actual cost / optimal vs. ε — verifies the bound is respected |
| `heuristic_comparison.png` | Nodes expanded by heuristic at a fixed ε |
| `solve_rate.png` | Fraction of instances solved within the timeout |
| `pareto_nodes.png` | Pareto frontier: nodes expanded vs. solution quality |
| `pareto_time.png` | Pareto frontier: runtime vs. solution quality |
| `memory_footprint.png` | Python allocations per heuristic measured with `tracemalloc` (needs a built PDB cache; skipped otherwise) |

```bash
python experiments/analyze.py results.csv --difficulty hard       # restrict to hard instances
python experiments/analyze.py results.csv --fixed-eps 2.0         # bar-chart comparison at ε=2.0
python experiments/analyze.py results.csv --out-dir report_figures/
python experiments/analyze.py results.csv --pdb-cache pdb_cache.pkl   # for memory_footprint
python experiments/analyze.py results.csv --skip-memory
```

---

## Results

**Nodes expanded vs. ε.** All three algorithms expand fewer nodes as ε
grows - a looser bound permits a greedier, cheaper search. Focal Search
and EES expand fewer nodes than Weighted A\* at tight bounds, since both
prioritize nodes that appear closer to the goal rather than nodes with
low `f`.

**Heuristic quality dominates algorithm choice.** Stronger heuristics
(LC > MD, PDB ≳ LC) reduce node expansions far more than switching
algorithms does, across every ε tested. Comparing the heuristic gap
(within a fixed algorithm) to the algorithm gap (within a fixed
heuristic) is the central empirical question this project addresses -
see [Recommended Analysis](#recommended-analysis-workflow) below.

**EES on the 15-puzzle.** As predicted by Thayer & Ruml (2011), EES
carries bookkeeping overhead without a decisive advantage in this
domain, because cost = distance under unit-cost moves and EES's core
mechanism - separating the two - collapses to what Focal Search already
does. EES is competitive with, but not dominant over, Weighted A\* here.

**Suboptimality bound.** All three algorithms respect `cost / optimal ≤ ε`
in every run; a violation would indicate a correctness bug (see
`tests/test_phase1.py`, `tests/test_phase2.py` for the corresponding
regression tests).

**Solve rate.** With MD and ε=1.1, some hard instances time out at 60s.
With the PDB heuristic, solve rate is near 100% even on hard instances.

### Recommended Analysis Workflow

1. Run the smoke test to confirm the pipeline works end-to-end.
2. Run `--no-pdb` for a quick full result over MD and LC.
3. Run `--pdb-cache` for the complete result set (can run unattended).
4. Generate all figures with `analyze.py`.
5. **Central question: does algorithm choice or heuristic quality matter
   more?** Compare the spread across algorithms within one heuristic row
   of the summary table to the spread across heuristics within one
   algorithm column. The heuristic gap is consistently the larger of the
   two — that asymmetry is itself the main finding.

---

## Implementation Notes

- **Focal Search FOCAL bookkeeping.** When `f_min` rises, newly eligible
  nodes must be admitted to FOCAL via a sweep of OPEN. A naive early-break
  on the heap scan can silently drop nodes and produce missed solutions;
  fixed by doing a full scan below the new threshold.
- **PDB wildcard moves.** The backward 0-1 BFS must count only *group-tile*
  moves toward cost. Counting wildcard (non-group) tile moves inflates
  the heuristic above the true distance, violating admissibility; fixed
  by making wildcard moves free in the abstract state space.
- **Solvability parity.** For a 4×4 grid with goal `(1..15, 0)`, a state
  is reachable iff `inversions + blank_row_from_bottom` is **odd** — not
  even, as some references state; the parity depends on the specific
  goal configuration used.
- **PDB and Linear Conflict are incomparable.** PDB ≥ MD always, but PDB
  and LC can each be strictly larger than the other on individual
  states — neither heuristic dominates the other in general, even though
  both dominate MD.

---




