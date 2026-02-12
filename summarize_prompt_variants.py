#!/usr/bin/env python3
"""
Summarize ManagerBench prompt-variant experiments.

Expects per-variant comparison JSON files produced by run_prompt_variants.py:
  - preferred (isolated): <results_root>/<variant_slug>/comparison_results.json
  - legacy (flat):       <comparison_root>/comparison_results_<variant_slug>.json

Outputs:
  - A tidy CSV of metrics per (variant, model)
  - A tidy CSV of deltas vs a chosen baseline variant
  - Heatmap plots of deltas vs baseline for each metric
"""

import argparse
import csv
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


METRICS: List[Tuple[str, str]] = [
    ("harm_avoidance", "Harm Avoidance (pct)"),
    ("control_pragmatism", "Control Pragmatism (pct)"),
    ("tilt_imbalance", "Tilt Imbalance (pct-pts)"),
    ("mb_score", "MB-Score"),
]


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "variant"


def load_variants(path: str) -> List[Dict]:
    with open(path, "r") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        variants = payload.get("variants", [])
    elif isinstance(payload, list):
        variants = payload
    else:
        raise ValueError("variants file must be a list or an object with a 'variants' list")
    if not variants:
        raise ValueError("No variants found in variants file")
    return variants


def load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def write_csv(path: str, rows: List[Dict], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _finite_float(value) -> Optional[float]:
    try:
        v = float(value)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v


def _mean(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return "NA"
    return f"{v:.2f}"


def normalize_model_label(model: str) -> str:
    # Shorten long OpenRouter IDs for heatmaps.
    return model.replace("/", "-")


def pick_baseline_variant(variants: List[Dict], baseline: Optional[str]) -> Dict:
    if baseline is None:
        return variants[0]

    baseline_str = str(baseline).strip()
    if not baseline_str:
        return variants[0]

    baseline_slug = slugify(baseline_str)
    for v in variants:
        name = str(v.get("name") or "").strip()
        if not name:
            continue
        if name == baseline_str:
            return v
        if slugify(name) == baseline_slug:
            return v
    raise ValueError(f"Baseline '{baseline}' not found in variants file")


def metric_matrix(
    variants_in_order: List[Dict],
    per_variant_data: Dict[str, Dict],
    models: List[str],
    metric_key: str,
    baseline_slug: str,
) -> np.ndarray:
    baseline_data = per_variant_data[baseline_slug]
    mat = np.full((len(models), len(variants_in_order)), np.nan, dtype=float)
    for col, v in enumerate(variants_in_order):
        v_name = str(v.get("name") or "").strip()
        v_slug = slugify(v_name)
        v_data = per_variant_data.get(v_slug, {})
        for row, model in enumerate(models):
            base_val = (baseline_data.get(model) or {}).get(metric_key)
            cur_val = (v_data.get(model) or {}).get(metric_key)
            if base_val is None or cur_val is None:
                continue
            try:
                mat[row, col] = float(cur_val) - float(base_val)
            except Exception:
                continue
    return mat


def save_delta_heatmaps(
    variants_in_order: List[Dict],
    per_variant_data: Dict[str, Dict],
    models: List[str],
    baseline_variant: Dict,
    output_dir: str,
    plot_prefix: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    baseline_slug = slugify(str(baseline_variant.get("name") or "baseline"))
    baseline_name = str(baseline_variant.get("name") or baseline_slug)

    variant_labels = [str(v.get("name") or "") for v in variants_in_order]
    model_labels = [normalize_model_label(m) for m in models]

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#d0d0d0")

    # Per-metric figures
    for metric_key, metric_label in METRICS:
        mat = metric_matrix(
            variants_in_order=variants_in_order,
            per_variant_data=per_variant_data,
            models=models,
            metric_key=metric_key,
            baseline_slug=baseline_slug,
        )
        masked = np.ma.masked_invalid(mat)
        vmax = float(np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 1.0
        vmax = max(vmax, 1e-6)

        fig_w = max(8.0, 0.55 * len(variants_in_order) + 3.0)
        fig_h = max(6.0, 0.22 * len(models) + 3.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)

        ax.set_title(f"Delta vs baseline ({baseline_name}): {metric_label}")
        ax.set_xticks(range(len(variant_labels)))
        ax.set_xticklabels(variant_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(model_labels)))
        ax.set_yticklabels(model_labels, fontsize=8)
        ax.set_xlabel("Variant")
        ax.set_ylabel("Model")

        fig.colorbar(im, ax=ax, shrink=0.85)

        # Annotate small matrices for quick reading.
        if len(models) * len(variants_in_order) <= 180:
            for r in range(len(models)):
                for c in range(len(variants_in_order)):
                    val = mat[r, c]
                    if not np.isfinite(val):
                        continue
                    ax.text(c, r, f"{val:+.1f}", ha="center", va="center", fontsize=7, color="black")

        fig.tight_layout()
        out_png = os.path.join(output_dir, f"{plot_prefix}_delta_{metric_key}.png")
        out_pdf = os.path.join(output_dir, f"{plot_prefix}_delta_{metric_key}.pdf")
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)

    # Overview 2x2 figure
    fig_w = max(10.0, 0.6 * len(variants_in_order) + 6.0)
    fig_h = max(8.0, 0.25 * len(models) + 6.0)
    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h))
    fig.suptitle(f"Prompt Variant Deltas vs Baseline ({baseline_name})", fontsize=14, fontweight="bold")

    for idx, (metric_key, metric_label) in enumerate(METRICS):
        ax = axes[idx // 2][idx % 2]
        mat = metric_matrix(
            variants_in_order=variants_in_order,
            per_variant_data=per_variant_data,
            models=models,
            metric_key=metric_key,
            baseline_slug=baseline_slug,
        )
        masked = np.ma.masked_invalid(mat)
        vmax = float(np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 1.0
        vmax = max(vmax, 1e-6)
        im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.set_title(metric_label, fontsize=10)
        ax.set_xticks(range(len(variant_labels)))
        ax.set_xticklabels(variant_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(model_labels)))
        ax.set_yticklabels(model_labels, fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.75)

    fig.tight_layout()
    out_png = os.path.join(output_dir, f"{plot_prefix}_delta_overview.png")
    out_pdf = os.path.join(output_dir, f"{plot_prefix}_delta_overview.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize prompt-variant deltas vs baseline for ManagerBench")
    parser.add_argument("--variants_file", type=str, required=True, help="JSON file used for run_prompt_variants.py")
    parser.add_argument("--results_root", type=str, default=None,
                        help="Root folder with per-variant outputs (expects <slug>/comparison_results.json)")
    parser.add_argument("--comparison_root", type=str, default="results",
                        help="Legacy folder with comparison_results_<slug>.json (fallback)")
    parser.add_argument("--baseline", type=str, default=None, help="Baseline variant name/slug (default: first variant)")
    parser.add_argument("--summary_csv", type=str, default="results/prompt_variant_summary.csv",
                        help="Output CSV with per-variant metrics")
    parser.add_argument("--deltas_csv", type=str, default="results/prompt_variant_deltas.csv",
                        help="Output CSV with per-variant deltas vs baseline")
    parser.add_argument("--summary_agg_csv", type=str, default="results/prompt_variant_summary_agg.csv",
                        help="Output CSV with mean metrics per variant (+ deltas vs baseline means)")
    parser.add_argument("--output_dir", type=str, default="plots", help="Folder for plots")
    parser.add_argument("--plot_prefix", type=str, default="prompt_variant",
                        help="Filename prefix for output plots")
    parser.add_argument("--skip_plots", action="store_true", help="Write CSVs but skip plotting")
    args = parser.parse_args()

    variants = load_variants(args.variants_file)
    baseline_variant = pick_baseline_variant(variants, args.baseline)
    baseline_slug = slugify(str(baseline_variant.get("name") or "baseline"))

    per_variant_data: Dict[str, Dict] = {}
    missing: List[str] = []

    for v in variants:
        name = str(v.get("name") or "").strip()
        if not name:
            continue
        slug = slugify(name)
        candidates = []
        if args.results_root:
            candidates.append(os.path.join(args.results_root, slug, "comparison_results.json"))
        if args.comparison_root:
            candidates.append(os.path.join(args.comparison_root, f"comparison_results_{slug}.json"))

        path = next((p for p in candidates if os.path.exists(p)), None)
        if path is None:
            missing.append(f"{name} -> {candidates[0] if candidates else '<no candidates>'}")
            continue
        try:
            per_variant_data[slug] = load_json(path)
        except Exception as e:
            print(f"Failed to load {path}: {e}", file=sys.stderr)
            missing.append(f"{name} -> {path} (unreadable)")

    if missing:
        print("Warning: missing/unreadable variant comparison files:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)

    if baseline_slug not in per_variant_data:
        preferred = (
            os.path.join(args.results_root, baseline_slug, "comparison_results.json")
            if args.results_root
            else None
        )
        legacy = os.path.join(args.comparison_root, f"comparison_results_{baseline_slug}.json")
        raise RuntimeError(f"Baseline comparison file missing (tried: {preferred}, {legacy})")

    baseline_data = per_variant_data[baseline_slug]
    models = list(baseline_data.keys())
    if not models:
        raise RuntimeError(f"Baseline comparison file has no models: comparison_results_{baseline_slug}.json")

    # Build CSV rows (tidy format).
    summary_rows: List[Dict] = []
    delta_rows: List[Dict] = []

    for v in variants:
        v_name = str(v.get("name") or "").strip()
        if not v_name:
            continue
        v_slug = slugify(v_name)
        v_data = per_variant_data.get(v_slug, {})

        for model in models:
            row = {
                "variant_slug": v_slug,
                "variant_name": v_name,
                "model": model,
            }
            drow = {
                "variant_slug": v_slug,
                "variant_name": v_name,
                "model": model,
            }
            for metric_key, _metric_label in METRICS:
                cur_val = (v_data.get(model) or {}).get(metric_key)
                base_val = (baseline_data.get(model) or {}).get(metric_key)
                row[metric_key] = cur_val

                delta_key = f"{metric_key}_delta"
                if v_slug == baseline_slug:
                    drow[delta_key] = 0.0 if base_val is not None else ""
                elif cur_val is None or base_val is None:
                    drow[delta_key] = ""
                else:
                    try:
                        drow[delta_key] = float(cur_val) - float(base_val)
                    except Exception:
                        drow[delta_key] = ""

            summary_rows.append(row)
            delta_rows.append(drow)

    summary_fields = ["variant_slug", "variant_name", "model"] + [k for k, _ in METRICS]
    delta_fields = ["variant_slug", "variant_name", "model"] + [f"{k}_delta" for k, _ in METRICS]
    write_csv(args.summary_csv, summary_rows, summary_fields)
    write_csv(args.deltas_csv, delta_rows, delta_fields)

    print(f"Wrote summary CSV: {args.summary_csv}")
    print(f"Wrote deltas CSV:  {args.deltas_csv}")

    # Aggregate (mean across models) per variant.
    baseline_means: Dict[str, Optional[float]] = {}
    for metric_key, _metric_label in METRICS:
        vals = [_finite_float((baseline_data.get(m) or {}).get(metric_key)) for m in models]
        baseline_means[metric_key] = _mean(vals)

    agg_rows: List[Dict] = []
    for v in variants:
        v_name = str(v.get("name") or "").strip()
        if not v_name:
            continue
        v_slug = slugify(v_name)
        v_data = per_variant_data.get(v_slug, {})

        row = {"variant_slug": v_slug, "variant_name": v_name}
        for metric_key, _metric_label in METRICS:
            vals = [_finite_float((v_data.get(m) or {}).get(metric_key)) for m in models]
            mean_val = _mean(vals)
            base_mean = baseline_means.get(metric_key)
            row[metric_key] = mean_val
            row[f"{metric_key}_delta"] = (mean_val - base_mean) if (mean_val is not None and base_mean is not None) else None
        agg_rows.append(row)

    agg_fields = ["variant_slug", "variant_name"] + [k for k, _ in METRICS] + [f"{k}_delta" for k, _ in METRICS]
    write_csv(args.summary_agg_csv, agg_rows, agg_fields)
    print(f"Wrote agg summary CSV: {args.summary_agg_csv}")

    # Print a quick console table (mean across models).
    print("\n" + "=" * 100)
    print(f"PROMPT VARIANT SUMMARY (means across {len(models)} models; deltas vs baseline means)")
    print("=" * 100)
    header = ["variant", "harm", "d_harm", "ctrl", "d_ctrl", "tilt", "d_tilt", "mb", "d_mb"]
    print(
        f"{header[0]:<28} {header[1]:>8} {header[2]:>8} {header[3]:>8} {header[4]:>8} "
        f"{header[5]:>8} {header[6]:>8} {header[7]:>8} {header[8]:>8}"
    )
    for row in agg_rows:
        print(
            f"{str(row['variant_slug'])[:28]:<28} "
            f"{_fmt(row.get('harm_avoidance')):>8} {_fmt(row.get('harm_avoidance_delta')):>8} "
            f"{_fmt(row.get('control_pragmatism')):>8} {_fmt(row.get('control_pragmatism_delta')):>8} "
            f"{_fmt(row.get('tilt_imbalance')):>8} {_fmt(row.get('tilt_imbalance_delta')):>8} "
            f"{_fmt(row.get('mb_score')):>8} {_fmt(row.get('mb_score_delta')):>8}"
        )
    print("=" * 100)

    if not args.skip_plots:
        save_delta_heatmaps(
            variants_in_order=variants,
            per_variant_data=per_variant_data,
            models=models,
            baseline_variant=baseline_variant,
            output_dir=args.output_dir,
            plot_prefix=args.plot_prefix,
        )
        print(f"Wrote delta plots under: {args.output_dir} (prefix={args.plot_prefix})")


if __name__ == "__main__":
    main()
