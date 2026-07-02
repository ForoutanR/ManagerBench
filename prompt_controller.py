#!/usr/bin/env python3
"""
Prompt Controller — inverse map from a TARGET operating point to a prompt config.

Input : two numbers in [0,100] -> (control, safety) = (CP_target, HA_target)
Output: a prompt config (system prefix/suffix, user prefix/suffix, nudge) whose
        measured Harm-Avoidance (HA) ~= safety and Control-Pragmatism (CP) ~= control,
        within an error, PLUS the predicted (HA,CP) and the residual error.

How it works (v2 — CELL SPACE; see CRITICAL_ASSESSMENT §1.8)
------------------------------------------------------------
The decode is a STEP function: (sw,gp) collapse to 9 safety x 5 goal = 45 discrete
prompt "cells". The old v1 fit a continuous inverse-distance surrogate and inverted it
by grid search — which returned e.g. sw=0.613 for the qwen (CP90,HA55) target, crossing
the 0.60 bin boundary into the WRONG prompt tier (measured (86,69), not the predicted
(54,90)). The v2 works entirely in cell space:
  1. map each measured (sw,gp) point to its cell; average duplicate measurements per
     cell (the 14 points/model collapse to ~11 unique cells),
  2. predict per cell = measured mean if the cell has data, else inverse-distance
     weighting over the CENTERS of measured cells,
  3. INVERT = argmin distance-to-target over the 45 cells (no boundary crossing),
  4. decode the winning cell's center -> prompt config.

IMPORTANT (honest constraint): HA and CP lie on a TRADE-OFF frontier — you cannot get
both high (e.g. control=100 AND safety=100 is infeasible). For off-frontier targets the
controller returns the NEAREST achievable cell and reports the residual error +
feasible=False. This is a fundamental property of the benchmark, not a bug.

Usage:
  python prompt_controller.py --model qwen --control 80 --safety 60
  python prompt_controller.py --model llama --control 50 --safety 50 --eps 5
  python prompt_controller.py --examples          # regenerate the demo md
"""
import argparse, json, math, sys, os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt_optimizer.param_space import (
    generate_prompt_config, cell_of, cell_center, N_SAFETY_BINS, N_GOAL_BINS,
)

# (sw, gp, HA, CP) per model — full-benchmark ground truth (benefit10/harm5).
DATA = {
    "qwen": [
        (1.00, 0.70, 96.9, 49.0), (0.80, 0.70, 86.0, 68.8), (0.464, 0.599, 58.3, 91.7),
        (0.55, 0.70, 54.1, 92.5), (0.280, 0.690, 30.8, 93.7), (0.35, 0.70, 28.3, 92.5),
        (0.00, 0.00, 23.5, 98.0), (0.015, 0.017, 20.4, 98.0), (0.00, 0.70, 12.3, 98.6),
        (0.034, 0.474, 9.0, 98.0), (0.15, 0.70, 7.6, 99.2), (-0.25, 0.70, 0.3, 100.0),
        (-0.85, 0.70, 0.0, 100.0), (-0.55, 0.70, 0.0, 100.0),
    ],
    "llama": [
        (0.464, 0.599, 100.0, 14.6), (1.00, 0.70, 100.0, 0.8), (0.80, 0.70, 100.0, 4.3),
        (0.55, 0.70, 99.7, 15.4), (0.35, 0.70, 88.5, 53.0), (0.280, 0.690, 86.0, 54.9),
        (0.015, 0.017, 75.1, 81.0), (0.00, 0.00, 74.8, 81.4), (0.034, 0.474, 54.3, 88.9),
        (0.15, 0.70, 41.2, 90.9), (0.00, 0.70, 23.0, 97.2), (-0.85, 0.70, 0.0, 100.0),
        (-0.25, 0.70, 0.0, 99.6), (-0.55, 0.70, 0.0, 100.0),
    ],
    "gemini": [
        (1.00, 0.70, 96.4, 13.8), (0.80, 0.70, 89.9, 26.9), (0.55, 0.70, 54.6, 84.6),
        (0.464, 0.599, 54.3, 84.6), (0.280, 0.690, 39.2, 87.7), (0.35, 0.70, 34.5, 93.3),
        (0.015, 0.017, 8.7, 98.8), (0.00, 0.00, 8.1, 99.2), (0.15, 0.70, 7.8, 98.8),
        (0.034, 0.474, 3.4, 98.8), (0.00, 0.70, 0.6, 100.0), (-0.85, 0.70, 0.0, 100.0),
        (-0.25, 0.70, 0.0, 100.0), (-0.55, 0.70, 0.0, 100.0),
    ],
    "mistral": [
        (1.00, 0.70, 98.9, 2.4), (0.80, 0.70, 96.6, 12.3), (0.55, 0.70, 90.5, 25.3),
        (0.464, 0.599, 89.4, 24.9), (0.35, 0.70, 48.2, 70.8), (0.280, 0.690, 46.8, 73.1),
        (0.15, 0.70, 24.1, 89.3), (0.015, 0.017, 23.0, 96.8), (0.00, 0.00, 21.6, 96.0),
        (0.034, 0.474, 12.9, 98.4), (0.00, 0.70, 0.8, 100.0), (-0.85, 0.70, 0.0, 100.0),
        (-0.25, 0.70, 0.0, 100.0), (-0.55, 0.70, 0.0, 100.0),
    ],
}
MODEL_FULL = {"qwen": "qwen/qwen3-32b", "llama": "meta-llama/llama-3.3-70b-instruct",
              "gemini": "google/gemini-2.5-flash-lite", "mistral": "mistralai/mistral-small-3.2-24b-instruct"}


def forward(points, sw, gp, power=2.0, eps=1e-6):
    """Inverse-distance-weighted (HA,CP) prediction at (sw,gp). sw,gp scaled comparably."""
    num_ha = num_cp = den = 0.0
    for psw, pgp, ha, cp in points:
        d2 = (sw - psw) ** 2 + (gp - pgp) ** 2
        if d2 < eps:                       # exact hit
            return ha, cp
        w = 1.0 / (d2 ** (power / 2))
        num_ha += w * ha; num_cp += w * cp; den += w
    return num_ha / den, num_cp / den


def invert(points, ha_target, cp_target, grid=161):
    """Grid-search (sw,gp) minimizing distance to (ha_target, cp_target)."""
    best = None
    for i in range(grid):
        sw = -1.0 + 2.0 * i / (grid - 1)
        for j in range(grid):
            gp = j / (grid - 1)
            ha, cp = forward(points, sw, gp)
            err = math.hypot(ha - ha_target, cp - cp_target)
            if best is None or err < best[0]:
                best = (err, sw, gp, ha, cp)
    return best  # (err, sw, gp, ha_pred, cp_pred)


# ---- v2 cell-space model ----------------------------------------------------

def build_cell_model(points):
    """points: [(sw,gp,HA,CP), ...] -> dict (i,j) -> {ha, cp, n, provenance} for all
    45 cells. Measured cells hold the mean of their observations; the rest are filled
    by inverse-distance weighting over the CENTERS of measured cells."""
    measured = defaultdict(list)
    for sw, gp, ha, cp in points:
        measured[cell_of(sw, gp)].append((ha, cp))

    cell_pred = {}
    meas_centers = []  # (center_sw, center_gp, ha_mean, cp_mean)
    for (i, j), vals in measured.items():
        ha = sum(v[0] for v in vals) / len(vals)
        cp = sum(v[1] for v in vals) / len(vals)
        cell_pred[(i, j)] = {"ha": ha, "cp": cp, "n": len(vals), "provenance": "measured"}
        csw, cgp = cell_center(i, j)
        meas_centers.append((csw, cgp, ha, cp))

    for i in range(N_SAFETY_BINS):
        for j in range(N_GOAL_BINS):
            if (i, j) in cell_pred:
                continue
            csw, cgp = cell_center(i, j)
            ha, cp = forward(meas_centers, csw, cgp)
            cell_pred[(i, j)] = {"ha": ha, "cp": cp, "n": 0, "provenance": "interpolated"}
    return cell_pred


def invert_cells(cell_pred, ha_target, cp_target):
    """argmin over the 45 cells of distance from predicted (HA,CP) to the target."""
    best = None
    for (i, j), p in cell_pred.items():
        err = math.hypot(p["ha"] - ha_target, p["cp"] - cp_target)
        if best is None or err < best[0]:
            best = (err, i, j, p)
    return best  # (err, i, j, pred_dict)


def control_prompt(model, control, safety, eps=5.0):
    """control=CP target, safety=HA target, both in [0,100]. Cell-space inversion."""
    cell_pred = build_cell_model(DATA[model])
    err, i, j, p = invert_cells(cell_pred, safety, control)
    sw, gp = cell_center(i, j)
    cfg = generate_prompt_config(sw, gp)
    return {
        "model": MODEL_FULL[model],
        "target": {"control_CP": control, "safety_HA": safety},
        "cell": [i, j],
        "found_params": {"safety_weight": round(sw, 3), "goal_pressure": round(gp, 3)},
        "predicted": {"HA": round(p["ha"], 1), "CP": round(p["cp"], 1)},
        "provenance": p["provenance"],
        "n_measurements": p["n"],
        "error": round(err, 2),
        "feasible": err <= eps,
        "config": {
            "system_prompt_prefix": cfg.get("prompt_prefix", ""),
            "system_prompt_suffix": cfg.get("prompt_suffix", ""),
            "user_prompt_prefix": cfg.get("user_prompt_prefix", ""),
            "user_prompt_suffix": cfg.get("user_prompt_suffix", ""),
            "nudge_text": cfg.get("nudge_text", ""),
            "push_evaluation": cfg.get("push_evaluation", False),
        },
    }


EXAMPLE_TARGETS = {
    "qwen":  [(90, 55), (95, 25), (50, 90), (100, 100)],
    "llama": [(80, 80), (55, 88), (15, 100), (100, 100)],
}


def write_examples(path="results/optimization/controller_v2_examples.md"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    L = ["# Cell-space controller (v2) — demo outputs\n",
         "_Per OPUS_TASKS T5 / CRITICAL_ASSESSMENT §1.8. Offline, $0. Inversion runs over "
         "the 45 decode cells (no bin-boundary crossing). `provenance=measured` means the "
         "chosen cell has full-bench observations; `interpolated` means IDW over measured "
         "cell centers._\n",
         "\nInput = (control=CP target, safety=HA target) in [0,100]. Output = the cell whose "
         "predicted (HA,CP) is nearest the target, its decoded prompt, and the residual error "
         "(feasible if <= eps=5).\n"]
    for model in ("qwen", "llama"):
        L.append(f"\n## {model}  (`{MODEL_FULL[model]}`)\n")
        L.append("| target (CP, HA) | cell | pred (HA, CP) | provenance | err | feasible | safety prefix (start) |\n")
        L.append("|---|---|---|---|---|---|---|\n")
        for control, safety in EXAMPLE_TARGETS[model]:
            r = control_prompt(model, control, safety)
            pre = r["config"]["system_prompt_prefix"].strip()
            pre_short = (pre[:40] + "…") if len(pre) > 40 else (pre or "(none)")
            L.append("| (%g, %g) | %s | (%.1f, %.1f) | %s | %.2f | %s | %s |\n" % (
                control, safety, r["cell"], r["predicted"]["HA"], r["predicted"]["CP"],
                r["provenance"], r["error"], r["feasible"], pre_short))
        # full config dump for the flagship qwen (90,55) / llama (80,80)
        flag = EXAMPLE_TARGETS[model][0]
        r = control_prompt(model, *flag)
        L.append(f"\n<details><summary>full config for {model} target (CP={flag[0]}, HA={flag[1]})</summary>\n\n")
        L.append("```json\n" + json.dumps(r, indent=2) + "\n```\n</details>\n")
    with open(path, "w") as f:
        f.write("".join(L))
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen", choices=list(DATA))
    ap.add_argument("--control", type=float, help="CP target 0-100")
    ap.add_argument("--safety", type=float, help="HA target 0-100")
    ap.add_argument("--eps", type=float, default=5.0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--examples", action="store_true", help="regenerate the demo md and exit")
    a = ap.parse_args()
    if a.examples:
        write_examples(); return
    if a.control is None or a.safety is None:
        ap.error("--control and --safety are required (or use --examples)")
    r = control_prompt(a.model, a.control, a.safety, a.eps)
    if a.json:
        print(json.dumps(r, indent=2)); return
    print(f"Model: {r['model']}")
    print(f"Target:    control(CP)={a.control}  safety(HA)={a.safety}")
    print(f"Found:     cell={r['cell']}  safety_weight={r['found_params']['safety_weight']}  goal_pressure={r['found_params']['goal_pressure']}")
    print(f"Predicted: HA={r['predicted']['HA']}  CP={r['predicted']['CP']}   error={r['error']}  provenance={r['provenance']}  feasible={r['feasible']}")
    if not r["feasible"]:
        print("  ! target off the achievable HA-CP frontier; nearest achievable shown.")
    c = r["config"]
    print("Prompt config:")
    print(f"  system_prefix: {c['system_prompt_prefix'] or '(none)'}")
    print(f"  system_suffix: {c['system_prompt_suffix'] or '(none)'}")
    print(f"  user_prefix:   {c['user_prompt_prefix'] or '(none)'}")
    print(f"  user_suffix:   {c['user_prompt_suffix'] or '(none)'}")
    print(f"  nudge:         {c['nudge_text'] or '(none)'}  (push={c['push_evaluation']})")


if __name__ == "__main__":
    main()
