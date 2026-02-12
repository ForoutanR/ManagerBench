#!/usr/bin/env python3
"""
Generate per-combination heatmaps from ManagerBench combination analysis CSV.

Expected input schema (from results/per_combination_analysis.csv):
model, benefit_percentage, harm_percentage, harm_avoidance_pct,
control_pragmatism_pct, tilt_imbalance, mb_score, ...
"""

import argparse
import csv
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _build_matrix(
    rows: List[Dict[str, str]],
    metric_key: str,
    models: List[str],
    combos: List[Tuple[int, int]],
) -> np.ndarray:
    index = {}
    for row in rows:
        model = row["model"]
        benefit = int(row["benefit_percentage"])
        harm = int(row["harm_percentage"])
        index[(model, benefit, harm)] = float(row[metric_key])

    matrix = np.full((len(models), len(combos)), np.nan, dtype=float)
    for i, model in enumerate(models):
        for j, (benefit, harm) in enumerate(combos):
            matrix[i, j] = index.get((model, benefit, harm), np.nan)
    return matrix


def _plot_single_heatmap(
    matrix: np.ndarray,
    models: List[str],
    combo_labels: List[str],
    title: str,
    colorbar_label: str,
    output_png: str,
    output_pdf: str,
    cmap: str = "viridis",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)

    ax.set_xticks(np.arange(len(combo_labels)))
    ax.set_xticklabels(combo_labels, fontsize=10)
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels([m.replace("/", "\n") for m in models], fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Combination (Benefit%, Harm%)", fontsize=10)
    ax.set_ylabel("Model", fontsize=10)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            label = "NA" if np.isnan(v) else f"{v:.1f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color="white")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def _plot_combined_2x2(
    matrices: Dict[str, np.ndarray],
    models: List[str],
    combo_labels: List[str],
    output_png: str,
    output_pdf: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 8.5))
    fig.suptitle("ManagerBench Per-Combination Heatmaps", fontsize=16, fontweight="bold")

    panels = [
        ("mb_score", "MB-Score", "MB", "viridis"),
        ("harm_avoidance_pct", "Harm Avoidance (%)", "%", "magma"),
        ("control_pragmatism_pct", "Control Pragmatism (%)", "%", "cividis"),
        ("tilt_imbalance", "Tilt Imbalance", "Tilt", "coolwarm"),
    ]

    for ax, (metric, title, cbar_label, cmap) in zip(axes.flatten(), panels):
        matrix = matrices[metric]
        im = ax.imshow(matrix, aspect="auto", cmap=cmap)
        ax.set_xticks(np.arange(len(combo_labels)))
        ax.set_xticklabels(combo_labels, fontsize=9)
        ax.set_yticks(np.arange(len(models)))
        ax.set_yticklabels([m.replace("/", "\n") for m in models], fontsize=8)
        ax.set_title(title, fontsize=12)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                v = matrix[i, j]
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7, color="white")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-combination heatmaps from analysis CSV")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="results/per_combination_analysis.csv",
        help="Input CSV generated from combination analysis",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots",
        help="Output directory for heatmaps",
    )
    args = parser.parse_args()

    rows = _load_rows(args.input_csv)
    if not rows:
        raise RuntimeError(f"No rows found in {args.input_csv}")

    os.makedirs(args.output_dir, exist_ok=True)

    models = sorted({row["model"] for row in rows})
    combos = sorted({(int(row["benefit_percentage"]), int(row["harm_percentage"])) for row in rows})
    combo_labels = [f"B{b}/H{h}" for b, h in combos]

    metrics = {
        "mb_score": ("MB-Score Heatmap", "MB-Score", "viridis"),
        "harm_avoidance_pct": ("Harm Avoidance Heatmap", "Harm Avoidance (%)", "magma"),
        "control_pragmatism_pct": ("Control Pragmatism Heatmap", "Control Pragmatism (%)", "cividis"),
        "tilt_imbalance": ("Tilt Imbalance Heatmap", "Tilt Imbalance", "coolwarm"),
    }

    matrices = {}
    for metric_key, (title, cbar_label, cmap) in metrics.items():
        matrix = _build_matrix(rows, metric_key, models, combos)
        matrices[metric_key] = matrix
        output_png = os.path.join(args.output_dir, f"combination_heatmap_{metric_key}.png")
        output_pdf = os.path.join(args.output_dir, f"combination_heatmap_{metric_key}.pdf")
        _plot_single_heatmap(
            matrix=matrix,
            models=models,
            combo_labels=combo_labels,
            title=title,
            colorbar_label=cbar_label,
            output_png=output_png,
            output_pdf=output_pdf,
            cmap=cmap,
        )
        print(f"Saved: {output_png}")

    combined_png = os.path.join(args.output_dir, "combination_heatmaps_overview.png")
    combined_pdf = os.path.join(args.output_dir, "combination_heatmaps_overview.pdf")
    _plot_combined_2x2(
        matrices=matrices,
        models=models,
        combo_labels=combo_labels,
        output_png=combined_png,
        output_pdf=combined_pdf,
    )
    print(f"Saved: {combined_png}")


if __name__ == "__main__":
    main()
