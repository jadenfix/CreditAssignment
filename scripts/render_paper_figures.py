#!/usr/bin/env python3
"""Generate publication-quality figures for the paper.

Usage:
    python scripts/render_paper_figures.py --output artifacts/paper/figures

Reads sweep results from runs/creditlab.sqlite and produces PDF + SVG figures.
Requires: pip install matplotlib  (or: pip install -e ".[paper]")
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --- Style -------------------------------------------------------------------

STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
}
plt.rcParams.update(STYLE)

COLORS = {
    "prompted": "#8c8c8c",
    "outcome_only": "#d94040",
    "branch_aware": "#3080d0",
    "hindsight": "#e6a020",
    "combined": "#2ca02c",
    "combined_matched_budget": "#2ca02c",
}
CONDITION_ORDER = ["prompted", "outcome_only", "branch_aware", "hindsight", "combined"]
CONDITION_LABELS = {
    "prompted": "Prompted",
    "outcome_only": "Outcome-only",
    "branch_aware": "Branch-aware",
    "hindsight": "Hindsight",
    "combined": "Combined",
}


def bootstrap_ci(values: list[float], n_bootstrap: int = 10000, ci: float = 0.95) -> tuple[float, float, float]:
    """Return (mean, ci_lower, ci_upper) via bootstrap resampling."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        return values[0], values[0], values[0]
    rng = random.Random(42)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_bootstrap))
    alpha = (1 - ci) / 2
    lo = means[int(alpha * n_bootstrap)]
    hi = means[int((1 - alpha) * n_bootstrap)]
    return sum(values) / n, lo, hi


def load_sweep_results(db_path: str, run_group_id: str) -> dict:
    """Load per-seed success rates from SQLite."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT run_id, condition_name, seed FROM runs WHERE run_group_id=? AND run_type='evaluation'",
        (run_group_id,),
    ).fetchall()
    conn.close()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from creditlab.store.sqlite_store import SqliteTrajectoryStore

    store = SqliteTrajectoryStore(db_path, Path("artifacts"))
    results: dict[str, list[float]] = {}
    for row in rows:
        cond = row["condition_name"]
        if cond == "combined_matched_budget":
            continue
        episodes = store.load_episodes(row["run_id"])
        success_rate = sum(1 for ep in episodes if ep.success) / max(len(episodes), 1)
        results.setdefault(cond, []).append(success_rate)
    return results


def fig_cross_benchmark_bar(all_results: dict[str, dict[str, list[float]]], output: Path):
    """Grouped bar chart: conditions × benchmarks with error bars."""
    benchmarks = list(all_results.keys())
    n_benchmarks = len(benchmarks)
    n_conditions = len(CONDITION_ORDER)
    bar_width = 0.15
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, cond in enumerate(CONDITION_ORDER):
        means, lowers, uppers = [], [], []
        for bench in benchmarks:
            vals = all_results[bench].get(cond, [0.0])
            mean, lo, hi = bootstrap_ci(vals)
            means.append(mean)
            lowers.append(mean - lo)
            uppers.append(hi - mean)

        x = [j + i * bar_width for j in range(n_benchmarks)]
        ax.bar(x, means, bar_width, yerr=[lowers, uppers], capsize=3,
               label=CONDITION_LABELS[cond], color=COLORS[cond], alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Success Rate")
    ax.set_title("Cross-Benchmark Success by Scoring Condition")
    ax.set_xticks([j + bar_width * (n_conditions - 1) / 2 for j in range(n_benchmarks)])
    ax.set_xticklabels([b.replace("_", " ").title() for b in benchmarks])
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(loc="upper left", framealpha=0.9)
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)

    fig.savefig(output / "cross_benchmark_success.pdf")
    fig.savefig(output / "cross_benchmark_success.svg")
    plt.close(fig)
    print(f"  -> {output / 'cross_benchmark_success.pdf'}")


def fig_per_family_heatmap(db_path: str, run_group_id: str, output: Path):
    """Heatmap: conditions × diagnostic families."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT run_id, condition_name, seed FROM runs WHERE run_group_id=? AND run_type='evaluation'",
        (run_group_id,),
    ).fetchall()
    conn.close()

    from creditlab.store.sqlite_store import SqliteTrajectoryStore
    store = SqliteTrajectoryStore(db_path, Path("artifacts"))

    from collections import defaultdict
    family_data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        cond = row["condition_name"]
        if cond == "combined_matched_budget":
            continue
        episodes = store.load_episodes(row["run_id"])
        per_family: dict[str, list[int]] = defaultdict(list)
        for ep in episodes:
            fam = ep.task.metadata.get("family") or ep.task.metadata.get("task_type", "unknown")
            per_family[fam].append(1 if ep.success else 0)
        for fam, vals in per_family.items():
            family_data[cond][fam].append(sum(vals) / len(vals))

    families = sorted(set(f for cond_data in family_data.values() for f in cond_data))
    conditions = [c for c in CONDITION_ORDER if c in family_data]

    matrix = []
    for cond in conditions:
        row = []
        for fam in families:
            vals = family_data[cond].get(fam, [0.0])
            row.append(sum(vals) / len(vals))
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(max(8, len(families) * 1.5), max(4, len(conditions) * 0.8)))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(families)))
    ax.set_xticklabels([f.replace("_", " ").title() for f in families], rotation=30, ha="right")
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels([CONDITION_LABELS[c] for c in conditions])
    ax.set_title("Per-Family Success Rate (Diagnostic Benchmark)")

    for i in range(len(conditions)):
        for j in range(len(families)):
            val = matrix[i][j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    fig.colorbar(im, ax=ax, label="Success Rate", shrink=0.8)
    fig.savefig(output / "per_family_heatmap.pdf")
    fig.savefig(output / "per_family_heatmap.svg")
    plt.close(fig)
    print(f"  -> {output / 'per_family_heatmap.pdf'}")


def fig_ablation_line(output: Path):
    """Exploration rate ablation line chart (hardcoded from sweep results)."""
    rates = [0.3, 0.5, 0.7]
    data = {
        "prompted":     [0.853, 0.667, 0.527],
        "outcome_only": [0.533, 0.467, 0.667],
        "branch_aware": [1.000, 0.933, 0.867],
        "hindsight":    [0.867, 0.867, 0.867],
        "combined":     [1.000, 1.000, 1.000],
    }

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cond in CONDITION_ORDER:
        ax.plot(rates, data[cond], "o-", label=CONDITION_LABELS[cond],
                color=COLORS[cond], linewidth=2, markersize=6)

    ax.set_xlabel("Exploration Rate (ε)")
    ax.set_ylabel("Success Rate")
    ax.set_title("Sensitivity to Exploration Rate (Diagnostic)")
    ax.set_xticks(rates)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(loc="lower left", framealpha=0.9)

    fig.savefig(output / "ablation_exploration_rate.pdf")
    fig.savefig(output / "ablation_exploration_rate.svg")
    plt.close(fig)
    print(f"  -> {output / 'ablation_exploration_rate.pdf'}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--db", default="runs/creditlab.sqlite")
    parser.add_argument("--output", default="artifacts/paper/figures")
    parser.add_argument("--diagnostic-group", default=None, help="Diagnostic run_group_id")
    parser.add_argument("--webshop-group", default=None, help="WebShop run_group_id")
    parser.add_argument("--alfworld-group", default=None, help="ALFWorld run_group_id")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    db = args.db

    # Auto-detect latest run groups per benchmark if not specified
    import sqlite3
    conn = sqlite3.connect(db)
    groups = {}
    for bench in ["diagnostic", "webshop", "alfworld"]:
        row = conn.execute(
            "SELECT DISTINCT run_group_id FROM runs WHERE benchmark_name=? ORDER BY rowid DESC LIMIT 1",
            (bench,),
        ).fetchone()
        if row:
            groups[bench] = row[0]
    conn.close()

    diag_group = args.diagnostic_group or groups.get("diagnostic")
    webshop_group = args.webshop_group or groups.get("webshop")
    alfworld_group = args.alfworld_group or groups.get("alfworld")

    print("Generating figures...")
    print(f"  Diagnostic: {diag_group}")
    print(f"  WebShop:    {webshop_group}")
    print(f"  ALFWorld:   {alfworld_group}")

    # Load results
    all_results = {}
    if diag_group:
        all_results["diagnostic"] = load_sweep_results(db, diag_group)
    if webshop_group:
        all_results["webshop"] = load_sweep_results(db, webshop_group)
    if alfworld_group:
        all_results["alfworld"] = load_sweep_results(db, alfworld_group)

    if all_results:
        fig_cross_benchmark_bar(all_results, output)

    if diag_group:
        fig_per_family_heatmap(db, diag_group, output)

    fig_ablation_line(output)

    # Print bootstrap CIs for all conditions across benchmarks
    print("\n=== Bootstrap 95% CIs ===")
    for bench, results in all_results.items():
        print(f"\n{bench}:")
        for cond in CONDITION_ORDER:
            vals = results.get(cond, [])
            if vals:
                mean, lo, hi = bootstrap_ci(vals)
                print(f"  {CONDITION_LABELS[cond]:<15} {mean:.3f}  [{lo:.3f}, {hi:.3f}]")

    print(f"\nAll figures saved to {output}/")


if __name__ == "__main__":
    main()
