#!/usr/bin/env python3
"""Generate all publication-quality figures for the paper.

Output: paper/figures/*.pdf and paper/figures/*.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

# --- Academic style ---
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "prompted": "#999999",
    "outcome_only": "#e74c3c",
    "branch_aware": "#3498db",
    "hindsight": "#f39c12",
    "combined": "#2ecc71",
}

# ============================================================
# Figure 1: Stochastic Benchmark — Main Result (Table 1)
# ============================================================
def fig1_stochastic_main():
    conditions = ["Prompted", "Outcome-only", "Branch-aware", "Hindsight", "Combined"]
    success = [0.792, 0.240, 0.979, 0.695, 1.000]
    partial = [0.747, 0.375, 0.806, 0.739, 0.823]
    colors = [COLORS["prompted"], COLORS["outcome_only"], COLORS["branch_aware"],
              COLORS["hindsight"], COLORS["combined"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2), sharey=False)

    x = np.arange(len(conditions))
    w = 0.6
    bars1 = ax1.bar(x, success, w, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_ylabel("Success Rate")
    ax1.set_title("(a) Success Rate")
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, rotation=30, ha="right", fontsize=8)
    ax1.set_ylim(0, 1.12)
    ax1.axhline(y=0.792, color=COLORS["prompted"], linestyle="--", alpha=0.5, linewidth=0.8)
    for bar, val in zip(bars1, success):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

    bars2 = ax2.bar(x, partial, w, color=colors, edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("Partial Credit")
    ax2.set_title("(b) Partial Credit")
    ax2.set_xticks(x)
    ax2.set_xticklabels(conditions, rotation=30, ha="right", fontsize=8)
    ax2.set_ylim(0, 1.0)
    for bar, val in zip(bars2, partial):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

    fig.suptitle("Stochastic Benchmark (15 seeds, 50 rollouts)", fontsize=11, y=1.02)
    plt.tight_layout()
    fig.savefig(OUT / "fig1_stochastic_main.pdf")
    fig.savefig(OUT / "fig1_stochastic_main.png")
    plt.close(fig)
    print("  fig1_stochastic_main.pdf")


# ============================================================
# Figure 2: Per-Family Heatmap (Table 3)
# ============================================================
def fig2_perfamily_heatmap():
    families = ["Cascade", "Dead-end", "Enabling", "Multi-gate", "Recovery"]
    conditions = ["Outcome-only", "Branch-aware", "Hindsight", "Combined"]
    data = np.array([
        [0.143, 0.571, 0.714, 0.143, 1.000],
        [1.000, 1.000, 0.857, 1.000, 1.000],
        [0.286, 1.000, 1.000, 1.000, 1.000],
        [1.000, 1.000, 1.000, 1.000, 1.000],
    ])

    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(len(families)))
    ax.set_yticks(np.arange(len(conditions)))
    ax.set_xticklabels(families, fontsize=9)
    ax.set_yticklabels(conditions, fontsize=9)

    for i in range(len(conditions)):
        for j in range(len(families)):
            val = data[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8.5, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Success Rate", fontsize=9)
    ax.set_title("Per-Family Diagnostic Breakdown", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT / "fig2_perfamily_heatmap.pdf")
    fig.savefig(OUT / "fig2_perfamily_heatmap.png")
    plt.close(fig)
    print("  fig2_perfamily_heatmap.pdf")


# ============================================================
# Figure 3: Real ALFWorld — Scorer Differentiation (Tables 5-7)
# ============================================================
def fig3_alfworld_real():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2))

    # (a) Differentiation ratio
    models = ["Haiku\n(30% success)", "Qwen-7B\n(17% success)"]
    oo_pct = [13.3, 3.4]
    ba_pct = [34.8, 31.8]
    x = np.arange(len(models))
    w = 0.3

    b1 = ax1.bar(x - w/2, oo_pct, w, color=COLORS["outcome_only"], label="Outcome-only")
    b2 = ax1.bar(x + w/2, ba_pct, w, color=COLORS["branch_aware"], label="Branch-aware")

    ax1.set_ylabel("States Differentiated (%)")
    ax1.set_title("(a) Scorer Differentiation")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=9)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_ylim(0, 45)

    # Annotate ratios
    for i, (o, b) in enumerate(zip(oo_pct, ba_pct)):
        ratio = b / o
        ax1.annotate(f"{ratio:.1f}x", xy=(i + w/2, b + 1), fontsize=8,
                     ha="center", fontweight="bold", color=COLORS["branch_aware"])

    for bars, vals in [(b1, oo_pct), (b2, ba_pct)]:
        for bar, val in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=7.5)

    # (b) Action verb preference
    models2 = ["Haiku", "Qwen-7B"]
    oo_verb = [69, 54]
    ba_verb = [80, 82]
    x2 = np.arange(len(models2))

    b3 = ax2.bar(x2 - w/2, oo_verb, w, color=COLORS["outcome_only"], label="Outcome-only")
    b4 = ax2.bar(x2 + w/2, ba_verb, w, color=COLORS["branch_aware"], label="Branch-aware")
    ax2.set_ylabel("Task-Relevant Action Preference (%)")
    ax2.set_title("(b) Action Quality")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(models2, fontsize=9)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.set_ylim(0, 100)
    ax2.axhline(y=50, color="gray", linestyle=":", alpha=0.5, linewidth=0.8, label="Random (50%)")

    for bars, vals in [(b3, oo_verb), (b4, ba_verb)]:
        for bar, val in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{val}%", ha="center", va="bottom", fontsize=7.5)

    fig.suptitle("Real ALFWorld Benchmark (Auto-Extracted Signals)", fontsize=11, y=1.02)
    plt.tight_layout()
    fig.savefig(OUT / "fig3_alfworld_real.pdf")
    fig.savefig(OUT / "fig3_alfworld_real.png")
    plt.close(fig)
    print("  fig3_alfworld_real.pdf")


# ============================================================
# Figure 4: Cross-Benchmark Success (Table 2)
# ============================================================
def fig4_cross_benchmark():
    benchmarks = ["Diagnostic", "WebShop", "ALFWorld\n(manifest)"]
    conditions = ["Prompted", "Outcome-only", "Branch-aware", "Hindsight", "Combined"]
    data = {
        "Prompted":     [0.726, 0.771, 0.523],
        "Outcome-only": [0.514, 0.714, 0.000],
        "Branch-aware": [0.971, 1.000, 1.000],
        "Hindsight":    [0.857, 0.714, 0.000],
        "Combined":     [1.000, 1.000, 1.000],
    }

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(benchmarks))
    w = 0.15
    offsets = np.arange(len(conditions)) - (len(conditions)-1)/2

    for i, cond in enumerate(conditions):
        color = list(COLORS.values())[i]
        bars = ax.bar(x + offsets[i]*w, data[cond], w, color=color, label=cond, edgecolor="white", linewidth=0.3)

    ax.set_ylabel("Success Rate")
    ax.set_title("Deterministic Benchmarks (7 seeds, 50 rollouts)")
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT / "fig4_cross_benchmark.pdf")
    fig.savefig(OUT / "fig4_cross_benchmark.png")
    plt.close(fig)
    print("  fig4_cross_benchmark.pdf")


# ============================================================
# Figure 5: Exploration Rate Sensitivity (Table 8)
# ============================================================
def fig5_ablation():
    eps_rates = [0.3, 0.5, 0.7]
    data = {
        "Prompted":     [0.853, 0.667, 0.527],
        "Outcome-only": [0.533, 0.467, 0.667],
        "Branch-aware": [1.000, 0.933, 0.867],
        "Hindsight":    [0.867, 0.867, 0.867],
        "Combined":     [1.000, 1.000, 1.000],
    }

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for cond, vals in data.items():
        key = cond.lower().replace("-", "_").replace(" ", "_")
        color = COLORS.get(key, "#333333")
        marker = "o" if cond != "Prompted" else "s"
        ls = "--" if cond == "Prompted" else "-"
        ax.plot(eps_rates, vals, marker=marker, color=color, label=cond,
                linewidth=1.5, markersize=5, linestyle=ls)

    ax.set_xlabel("Exploration Rate (ε)")
    ax.set_ylabel("Success Rate")
    ax.set_title("Exploration Rate Sensitivity (Diagnostic)")
    ax.set_xticks(eps_rates)
    ax.set_ylim(0.3, 1.08)
    ax.legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    fig.savefig(OUT / "fig5_ablation.pdf")
    fig.savefig(OUT / "fig5_ablation.png")
    plt.close(fig)
    print("  fig5_ablation.pdf")


# ============================================================
# Figure 6: Signal Quality vs Model Capability
# ============================================================
def fig6_signal_vs_capability():
    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    # Data points: (success_rate, oo_diff%, ba_diff%)
    models = {
        "Qwen-3B\n(2%)":   (0.02,  1.5, 21.9),
        "Qwen-7B\n(17%)":  (0.167, 3.4, 31.8),
        "Haiku\n(30%)":    (0.300, 13.3, 34.8),
    }

    success_rates = [v[0] for v in models.values()]
    oo_diffs = [v[1] for v in models.values()]
    ba_diffs = [v[2] for v in models.values()]

    ax.plot(success_rates, oo_diffs, "o-", color=COLORS["outcome_only"],
            label="Outcome-only", markersize=8, linewidth=1.5)
    ax.plot(success_rates, ba_diffs, "s-", color=COLORS["branch_aware"],
            label="Branch-aware", markersize=8, linewidth=1.5)

    for (name, (sr, oo, ba)) in models.items():
        ax.annotate(name, (sr, ba + 1.5), fontsize=7.5, ha="center", color=COLORS["branch_aware"])
        ratio = ba / max(oo, 0.1)
        mid_y = (oo + ba) / 2
        ax.annotate(f"{ratio:.1f}x", (sr + 0.01, mid_y), fontsize=7.5,
                    fontweight="bold", color="#666666")

    # Fill the gap between curves
    ax.fill_between(success_rates, oo_diffs, ba_diffs, alpha=0.1, color=COLORS["branch_aware"])

    ax.set_xlabel("Model Success Rate")
    ax.set_ylabel("States Differentiated (%)")
    ax.set_title("Signal Advantage vs. Model Capability")
    ax.legend(fontsize=8)
    ax.set_xlim(-0.02, 0.35)
    ax.set_ylim(0, 42)
    plt.tight_layout()
    fig.savefig(OUT / "fig6_signal_vs_capability.pdf")
    fig.savefig(OUT / "fig6_signal_vs_capability.png")
    plt.close(fig)
    print("  fig6_signal_vs_capability.pdf")


if __name__ == "__main__":
    print("Generating paper figures...")
    fig1_stochastic_main()
    fig2_perfamily_heatmap()
    fig3_alfworld_real()
    fig4_cross_benchmark()
    fig5_ablation()
    fig6_signal_vs_capability()
    print(f"Done. All figures in {OUT}/")
