#!/usr/bin/env python3
"""
Analyze optimization results across all models.

Generates:
  - Cross-model parameter landscape (2D scatter colored by MB-Score)
  - Pareto frontier overlay across models
  - Exploration coverage map (explored vs unexplored regions)
  - Per-model best config comparison
  - Summary markdown report

Usage:
  python analyze_optimization_results.py [--optimization_dir results/optimization] [--output_dir results/optimization/analysis]
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Existing spectrum positions for reference
SPECTRUM_POSITIONS = {
    "Max Safety":          (1.00, 0.70),
    "Safety Constrained":  (0.80, 0.70),
    "Safe Prefix":         (0.55, 0.70),
    "Balanced-Safe":       (0.35, 0.70),
    "Balanced":            (0.15, 0.70),
    "No Push":             (0.00, 0.00),
    "Utility Leaning":    (-0.25, 0.70),
    "Push Nudge":          (0.00, 0.70),
    "Utility Max":        (-0.55, 0.70),
    "Safety Ignored":     (-0.85, 0.70),
}

MODEL_COLORS = {
    "google/gemini-2.5-flash-lite": "#4285F4",
    "qwen/qwen3-32b": "#EA4335",
    "meta-llama/llama-3.3-70b-instruct": "#34A853",
    "mistralai/mistral-small-3.2-24b-instruct": "#FBBC04",
}

MODEL_SHORT_NAMES = {
    "google/gemini-2.5-flash-lite": "Gemini Flash Lite",
    "qwen/qwen3-32b": "Qwen3-32B",
    "meta-llama/llama-3.3-70b-instruct": "Llama-3.3-70B",
    "mistralai/mistral-small-3.2-24b-instruct": "Mistral Small",
}


def load_optimization_results(optimization_dir: str) -> Dict[str, Dict]:
    """Load all single-objective optimization results."""
    results = {}
    for fname in os.listdir(optimization_dir):
        if fname.startswith("optimization_") and fname.endswith(".json"):
            fpath = os.path.join(optimization_dir, fname)
            with open(fpath, "r") as f:
                data = json.load(f)
            model = data.get("model", fname.replace("optimization_", "").replace(".json", ""))
            results[model] = data
    return results


def load_pareto_results(optimization_dir: str) -> Dict[str, Dict]:
    """Load all multi-objective Pareto results."""
    results = {}
    for fname in os.listdir(optimization_dir):
        if fname.startswith("pareto_") and fname.endswith(".json"):
            fpath = os.path.join(optimization_dir, fname)
            with open(fpath, "r") as f:
                data = json.load(f)
            model = data.get("model", fname.replace("pareto_", "").replace(".json", ""))
            results[model] = data
    return results


def plot_cross_model_landscape(opt_results: Dict[str, Dict], output_dir: str) -> None:
    """Combined 2D scatter of all models' trials, colored by model."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: colored by model
    ax = axes[0]
    for model, data in opt_results.items():
        trials = data.get("all_trials", [])
        valid = [t for t in trials if t.get("params") and t.get("value") is not None]
        if not valid:
            continue
        sw = [t["params"]["safety_weight"] for t in valid]
        gp = [t["params"]["goal_pressure"] for t in valid]
        color = MODEL_COLORS.get(model, "gray")
        short = MODEL_SHORT_NAMES.get(model, model.split("/")[-1])
        ax.scatter(sw, gp, c=color, s=40, alpha=0.6, label=short, edgecolors="k", linewidths=0.3)

    # Mark existing spectrum positions
    for label, (sw, gp) in SPECTRUM_POSITIONS.items():
        ax.scatter([sw], [gp], marker="x", s=60, c="black", zorder=10)

    # Highlight unexplored region
    rect = Rectangle((-0.1, 0.1), 0.7, 0.5, linewidth=2,
                      edgecolor="red", facecolor="red", alpha=0.08, linestyle="--")
    ax.add_patch(rect)
    ax.text(0.25, 0.35, "Under-explored\nregion", ha="center", va="center",
            fontsize=9, color="red", style="italic")

    ax.set_xlabel("Safety Weight")
    ax.set_ylabel("Goal Pressure")
    ax.set_title("Exploration Coverage (colored by model)")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Right: colored by MB-Score (all models combined)
    ax = axes[1]
    all_sw, all_gp, all_mb = [], [], []
    for model, data in opt_results.items():
        trials = data.get("all_trials", [])
        valid = [t for t in trials if t.get("params") and t.get("value") is not None]
        for t in valid:
            all_sw.append(t["params"]["safety_weight"])
            all_gp.append(t["params"]["goal_pressure"])
            all_mb.append(t["value"])

    if all_sw:
        scatter = ax.scatter(all_sw, all_gp, c=all_mb, cmap="RdYlGn", s=40,
                             edgecolors="k", linewidths=0.3)
        plt.colorbar(scatter, ax=ax, label="MB-Score")

    for label, (sw, gp) in SPECTRUM_POSITIONS.items():
        ax.scatter([sw], [gp], marker="x", s=60, c="black", zorder=10)

    ax.set_xlabel("Safety Weight")
    ax.set_ylabel("Goal Pressure")
    ax.set_title("Parameter Landscape (colored by MB-Score)")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "cross_model_landscape.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_cross_model_pareto(pareto_results: Dict[str, Dict], output_dir: str) -> None:
    """Overlay Pareto frontiers from all models on one plot."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for model, data in pareto_results.items():
        # Plot all trials as faint dots
        trials = data.get("all_trials", [])
        valid = [t for t in trials if t.get("values") and len(t["values"]) == 2]
        if valid:
            ha = [t["values"][0] for t in valid]
            cp = [t["values"][1] for t in valid]
            color = MODEL_COLORS.get(model, "gray")
            ax.scatter(cp, ha, c=color, alpha=0.15, s=15)

        # Plot Pareto front
        pareto = data.get("pareto_configs", [])
        if pareto:
            p_ha = [p["harm_avoidance"] for p in pareto]
            p_cp = [p["control_pragmatism"] for p in pareto]
            color = MODEL_COLORS.get(model, "gray")
            short = MODEL_SHORT_NAMES.get(model, model.split("/")[-1])
            ax.scatter(p_cp, p_ha, c=color, s=80, marker="D", edgecolors="k",
                       linewidths=0.8, label=f"{short} ({len(pareto)} pts)", zorder=5)
            # Connect Pareto points
            frontier = sorted(zip(p_cp, p_ha))
            ax.plot([p[0] for p in frontier], [p[1] for p in frontier],
                    color=color, linestyle="--", linewidth=1.5, alpha=0.7)

    # Add iso-MB-Score curves
    for mb_target in [20, 40, 60, 80]:
        cp_range = np.linspace(1, 100, 200)
        ha_curve = mb_target * cp_range / (2 * cp_range - mb_target)
        mask = (ha_curve > 0) & (ha_curve <= 100)
        ax.plot(cp_range[mask], ha_curve[mask], "k-", alpha=0.1, linewidth=0.8)
        # Label the curve
        idx = np.argmin(np.abs(cp_range - 80))
        if mask[idx]:
            ax.text(80, ha_curve[idx], f"MB={mb_target}", fontsize=7, alpha=0.3,
                    ha="center", va="bottom")

    ax.set_xlabel("Control Pragmatism (%)")
    ax.set_ylabel("Harm Avoidance (%)")
    ax.set_title("Cross-Model Pareto Frontiers")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "cross_model_pareto.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_exploration_coverage(opt_results: Dict[str, Dict], output_dir: str) -> None:
    """Heatmap showing how densely each region of the parameter space was explored."""
    all_sw, all_gp = [], []
    for data in opt_results.values():
        for t in data.get("all_trials", []):
            if t.get("params"):
                all_sw.append(t["params"]["safety_weight"])
                all_gp.append(t["params"]["goal_pressure"])

    if not all_sw:
        print("  No trial data found for coverage plot")
        return

    fig, ax = plt.subplots(figsize=(9, 7))

    # 2D histogram
    sw_bins = np.linspace(-1.0, 1.0, 21)
    gp_bins = np.linspace(0.0, 1.0, 11)
    H, xedges, yedges = np.histogram2d(all_sw, all_gp, bins=[sw_bins, gp_bins])

    im = ax.imshow(H.T, origin="lower", aspect="auto",
                   extent=[-1, 1, 0, 1], cmap="YlOrRd",
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Number of trials")

    # Mark spectrum positions
    for label, (sw, gp) in SPECTRUM_POSITIONS.items():
        ax.scatter([sw], [gp], marker="x", s=80, c="blue", zorder=10, linewidths=2)

    ax.set_xlabel("Safety Weight")
    ax.set_ylabel("Goal Pressure")
    ax.set_title(f"Exploration Density ({len(all_sw)} total trials across all models)")
    ax.grid(True, alpha=0.2)

    path = os.path.join(output_dir, "exploration_coverage.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_per_model_comparison(opt_results: Dict[str, Dict], output_dir: str) -> None:
    """Bar chart comparing best config per model."""
    models = []
    best_mb = []
    best_ha = []
    best_cp = []
    best_sw = []
    best_gp = []
    colors = []

    for model, data in sorted(opt_results.items()):
        models.append(MODEL_SHORT_NAMES.get(model, model.split("/")[-1]))
        best_mb.append(data.get("best_mb_score", 0))
        best_ha.append(data.get("best_harm_avoidance", 0))
        best_cp.append(data.get("best_control_pragmatism", 0))
        params = data.get("best_params", {})
        best_sw.append(params.get("safety_weight", 0))
        best_gp.append(params.get("goal_pressure", 0))
        colors.append(MODEL_COLORS.get(model, "gray"))

    if not models:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # MB-Score
    ax = axes[0]
    bars = ax.bar(models, best_mb, color=colors, edgecolor="k", linewidth=0.5)
    ax.set_ylabel("MB-Score")
    ax.set_title("Best MB-Score per Model")
    for bar, val in zip(bars, best_mb):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    ax.tick_params(axis="x", rotation=20)

    # HA and CP
    ax = axes[1]
    x = np.arange(len(models))
    w = 0.35
    ax.bar(x - w/2, best_ha, w, label="Harm Avoidance", color="salmon", edgecolor="k", linewidth=0.5)
    ax.bar(x + w/2, best_cp, w, label="Ctrl Pragmatism", color="skyblue", edgecolor="k", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Best Config: HA vs CP")
    ax.legend(fontsize=8)

    # Best parameters
    ax = axes[2]
    ax.bar(x - w/2, best_sw, w, label="Safety Weight", color="lightgreen", edgecolor="k", linewidth=0.5)
    ax.bar(x + w/2, best_gp, w, label="Goal Pressure", color="plum", edgecolor="k", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20)
    ax.set_ylabel("Parameter Value")
    ax.set_title("Best Config: Parameters")
    ax.legend(fontsize=8)
    ax.set_ylim(-1.1, 1.1)

    fig.tight_layout()
    path = os.path.join(output_dir, "per_model_best_configs.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def generate_summary_report(
    opt_results: Dict[str, Dict],
    pareto_results: Dict[str, Dict],
    output_dir: str,
) -> None:
    """Generate a markdown summary of all findings."""
    lines = [
        "# Optimization Analysis Summary",
        "",
        f"Generated from {len(opt_results)} single-objective + {len(pareto_results)} multi-objective runs.",
        "",
        "## Best Configs per Model (Single-Objective: Maximize MB-Score)",
        "",
        "| Model | MB-Score | Harm Avoidance | Ctrl Pragmatism | safety_weight | goal_pressure |",
        "|-------|----------|----------------|-----------------|---------------|---------------|",
    ]

    # Best known spectrum score for reference
    # (from report_spectrum.md, Safe Prefix position for Gemini was 67.2)
    for model in sorted(opt_results.keys()):
        data = opt_results[model]
        short = MODEL_SHORT_NAMES.get(model, model.split("/")[-1])
        mb = data.get("best_mb_score", 0)
        ha = data.get("best_harm_avoidance", 0)
        cp = data.get("best_control_pragmatism", 0)
        params = data.get("best_params", {})
        sw = params.get("safety_weight", 0)
        gp = params.get("goal_pressure", 0)
        lines.append(f"| {short} | {mb:.1f} | {ha:.1f}% | {cp:.1f}% | {sw:+.3f} | {gp:.3f} |")

    lines += [
        "",
        "## Pareto Frontier Summary",
        "",
    ]

    for model in sorted(pareto_results.keys()):
        data = pareto_results[model]
        short = MODEL_SHORT_NAMES.get(model, model.split("/")[-1])
        n_pareto = data.get("n_pareto", 0)
        n_trials = data.get("n_trials", 0)
        lines.append(f"### {short}")
        lines.append(f"- Trials: {n_trials}, Pareto-optimal: {n_pareto}")

        pareto = data.get("pareto_configs", [])
        if pareto:
            lines.append("")
            lines.append("| HA (%) | CP (%) | MB-Score | safety_weight | goal_pressure |")
            lines.append("|--------|--------|----------|---------------|---------------|")
            for p in sorted(pareto, key=lambda x: x.get("mb_score", 0), reverse=True):
                lines.append(
                    f"| {p['harm_avoidance']:.1f} | {p['control_pragmatism']:.1f} | "
                    f"{p.get('mb_score', 0):.1f} | "
                    f"{p['params']['safety_weight']:+.3f} | "
                    f"{p['params']['goal_pressure']:.3f} |"
                )
        lines.append("")

    # Unexplored region analysis
    lines += [
        "## Unexplored Region Analysis",
        "",
        "The existing spectrum covered only `goal_pressure=0.70` (one exception at 0.00).",
        "The optimizer explored new regions. Key findings:",
        "",
    ]

    # Count trials in different regions
    total_trials = 0
    low_gp_trials = 0  # goal_pressure < 0.50
    mid_gp_trials = 0  # 0.50 <= goal_pressure < 0.70
    high_gp_trials = 0  # goal_pressure >= 0.70
    for data in opt_results.values():
        for t in data.get("all_trials", []):
            if t.get("params"):
                total_trials += 1
                gp = t["params"]["goal_pressure"]
                if gp < 0.50:
                    low_gp_trials += 1
                elif gp < 0.70:
                    mid_gp_trials += 1
                else:
                    high_gp_trials += 1

    if total_trials > 0:
        lines.append(f"- Total trials across all models: {total_trials}")
        lines.append(f"- Low goal_pressure (< 0.50): {low_gp_trials} trials ({100*low_gp_trials/total_trials:.0f}%)")
        lines.append(f"- Mid goal_pressure (0.50-0.70): {mid_gp_trials} trials ({100*mid_gp_trials/total_trials:.0f}%)")
        lines.append(f"- High goal_pressure (>= 0.70): {high_gp_trials} trials ({100*high_gp_trials/total_trials:.0f}%)")
    lines.append("")

    # Check if any best configs are in unexplored regions
    lines.append("### Do best configs use unexplored parameters?")
    lines.append("")
    for model in sorted(opt_results.keys()):
        data = opt_results[model]
        short = MODEL_SHORT_NAMES.get(model, model.split("/")[-1])
        params = data.get("best_params", {})
        gp = params.get("goal_pressure", 0.70)
        sw = params.get("safety_weight", 0)
        is_novel = abs(gp - 0.70) > 0.05 and abs(gp - 0.00) > 0.05
        marker = " **NEW REGION**" if is_novel else ""
        lines.append(f"- {short}: sw={sw:+.3f}, gp={gp:.3f}{marker}")

    lines.append("")
    lines.append("---")
    lines.append("*See plots in this directory for visual analysis.*")

    report_path = os.path.join(output_dir, "analysis_summary.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze optimization results across all models")
    parser.add_argument("--optimization_dir", default="results/optimization",
                        help="Directory containing optimization JSON results")
    parser.add_argument("--output_dir", default="results/optimization/analysis",
                        help="Directory for analysis outputs")
    args = parser.parse_args()

    if not os.path.isdir(args.optimization_dir):
        print(f"Error: optimization directory not found: {args.optimization_dir}")
        print("Run optimization first (see TODO_OPTIMIZATION.md)")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading optimization results...")
    opt_results = load_optimization_results(args.optimization_dir)
    pareto_results = load_pareto_results(args.optimization_dir)

    if not opt_results and not pareto_results:
        print("No optimization results found. Run Task 1 first.")
        sys.exit(1)

    print(f"Found {len(opt_results)} single-objective + {len(pareto_results)} multi-objective results\n")

    if opt_results:
        print("Generating cross-model landscape...")
        plot_cross_model_landscape(opt_results, args.output_dir)

        print("Generating exploration coverage heatmap...")
        plot_exploration_coverage(opt_results, args.output_dir)

        print("Generating per-model comparison...")
        plot_per_model_comparison(opt_results, args.output_dir)

    if pareto_results:
        print("Generating cross-model Pareto overlay...")
        plot_cross_model_pareto(pareto_results, args.output_dir)

    print("Generating summary report...")
    generate_summary_report(opt_results, pareto_results, args.output_dir)

    print(f"\nDone! Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
