#!/usr/bin/env python3
"""
Create model-comparison style plots for each (benefit, harm) combination.
Output:
  - one PNG/PDF per combination
  - one multi-page PDF with all combinations
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def load_rows(csv_path: str):
    with open(csv_path, "r", newline="") as f:
        return list(csv.DictReader(f))


def make_figure(rows, benefit: int, harm: int):
    models = [r["model"] for r in rows]
    harm_avoidance = [float(r["harm_avoidance_pct"]) for r in rows]
    control_pragmatism = [float(r["control_pragmatism_pct"]) for r in rows]
    mb_scores = [float(r["mb_score"]) for r in rows]
    tilt_imbalance = [float(r["tilt_imbalance"]) for r in rows]
    labels = [m.replace("/", "-").replace("_", "-") for m in models]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"ManagerBench Model Comparison (Benefit={benefit}%, Harm={harm}%)",
        fontsize=16,
        fontweight="bold",
    )

    ax1 = axes[0, 0]
    ax1.scatter(control_pragmatism, harm_avoidance, s=100, alpha=0.8, c=range(len(models)), cmap="viridis")
    for i, label in enumerate(labels):
        ax1.annotate(label, (control_pragmatism[i], harm_avoidance[i]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax1.set_xlabel("Control Pragmatism (%)", fontsize=11)
    ax1.set_ylabel("Harm Avoidance (%)", fontsize=11)
    ax1.set_title("Harm Avoidance vs Control Pragmatism", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=50, color="r", linestyle="--", alpha=0.5)
    ax1.axvline(x=50, color="r", linestyle="--", alpha=0.5)

    ax2 = axes[0, 1]
    bars = ax2.bar(range(len(models)), mb_scores, color="steelblue", alpha=0.8)
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("MB-Score", fontsize=11)
    ax2.set_title("MB-Score Comparison", fontsize=12)
    ax2.grid(True, alpha=0.3, axis="y")
    for bar, score in zip(bars, mb_scores):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{score:.1f}", ha="center", va="bottom", fontsize=9)

    ax3 = axes[1, 0]
    colors = ["green" if t < 0 else "red" for t in tilt_imbalance]
    bars = ax3.bar(range(len(models)), tilt_imbalance, color=colors, alpha=0.8)
    ax3.set_xticks(range(len(models)))
    ax3.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax3.set_ylabel("Tilt Imbalance", fontsize=11)
    ax3.set_title("Tilt Imbalance (Harm Avoidance - Control Pragmatism)", fontsize=12)
    ax3.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax3.grid(True, alpha=0.3, axis="y")
    for bar, tilt in zip(bars, tilt_imbalance):
        va = "bottom" if tilt >= 0 else "top"
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{tilt:.1f}", ha="center", va=va, fontsize=9)

    ax4 = axes[1, 1]
    x = np.arange(len(models))
    width = 0.35
    ax4.bar(x - width / 2, harm_avoidance, width, label="Harm Avoidance", alpha=0.8)
    ax4.bar(x + width / 2, control_pragmatism, width, label="Control Pragmatism", alpha=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax4.set_ylabel("Score (%)", fontsize=11)
    ax4.set_title("Harm Avoidance vs Control Pragmatism", fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot model comparison per combination")
    parser.add_argument(
        "--input_csv",
        default="results/per_combination_analysis.csv",
        help="Input CSV with per-combination metrics",
    )
    parser.add_argument("--output_dir", default="plots", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = load_rows(args.input_csv)
    grouped = defaultdict(list)
    for r in rows:
        grouped[(int(r["benefit_percentage"]), int(r["harm_percentage"]))].append(r)

    combined_pdf_path = os.path.join(args.output_dir, "model_comparison_by_combination.pdf")
    with PdfPages(combined_pdf_path) as pdf:
        for benefit, harm in sorted(grouped.keys()):
            combo_rows = sorted(grouped[(benefit, harm)], key=lambda x: x["model"])
            fig = make_figure(combo_rows, benefit, harm)

            base = os.path.join(args.output_dir, f"model_comparison_b{benefit}_h{harm}")
            png_path = f"{base}.png"
            pdf_path = f"{base}.pdf"
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            fig.savefig(pdf_path, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {png_path}")
            print(f"Saved: {pdf_path}")

    print(f"Saved: {combined_pdf_path}")


if __name__ == "__main__":
    main()
