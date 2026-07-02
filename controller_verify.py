#!/usr/bin/env python3
"""
Closed-loop verified controller (v2 — CELL SEARCH; see OPUS_TASKS T6 / CRITICAL_ASSESSMENT §1.8).

Offline `prompt_controller.py` picks a cell from a surrogate. This script then VERIFIES
and CORRECTS against the real model, searching the DISCRETE cell grid (not continuous
sw/gp) so every move lands in a genuinely different prompt:
  1. seed cell (i,j) from the cell-space controller inversion,
  2. measure the decoded config on the proxy (live, OpenRouter) -> measured (HA,CP),
  3. if error > eps, evaluate NEIGHBORING cells ((i±1,j),(i,j±1),diagonals), skipping
     cells already measured this run; move to the best-improving neighbor,
  4. repeat up to `rounds`, or until error <= eps, or the budget cap.

Output: the final cell + decoded config + MEASURED HA/CP + residual error (real).

--dry_run uses a deterministic SYNTHETIC evaluator (no API, $0) to exercise the search;
it asserts convergence to the synthetic optimum. This is BUILD-ONLY: a live run needs
explicit cost approval.

Run live on VPS:
    OPENROUTER_API_KEY=... .venv/bin/python controller_verify.py \
        --model qwen --control 90 --safety 55 --eps 4 --rounds 6
Dry-run (safe, no API):
    .venv/bin/python controller_verify.py --dry_run --control 70 --safety 60
"""
import argparse, json, math, os, sys, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt_optimizer.param_space import (
    generate_prompt_config, cell_center, N_SAFETY_BINS, N_GOAL_BINS,
)
from prompt_controller import DATA, MODEL_FULL, build_cell_model, invert_cells

HARD_CAP = float(os.environ.get("HARD_CAP", "10.0"))
PROXY = os.environ.get("PROXY", "bench_proxy/proxy_items.json")  # stratified as of T3


def usage(key):
    req = urllib.request.Request("https://openrouter.ai/api/v1/key",
                                 headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)["data"]["usage"]


def neighbor_cells(i, j):
    """8-neighborhood of cell (i,j), clamped to the grid."""
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < N_SAFETY_BINS and 0 <= nj < N_GOAL_BINS:
                yield (ni, nj)


class FakeEval:
    """Deterministic synthetic (HA,CP) per cell for --dry_run. Monotone safety frontier
    (HA up / CP down with the safety bin) plus a mild goal-pressure term, so the
    distance-to-target surface is convex and greedy cell search must reach the optimum."""
    needs_key = False

    def measure_cell(self, i, j):
        ha = max(0.0, min(100.0, 12.5 * i + 2.0 * (j - 2)))
        cp = max(0.0, min(100.0, 100.0 - 12.5 * i + 2.0 * (j - 2)))
        return ha, cp


class LiveEval:
    """Real proxy evaluation via OpenRouter. Imported lazily so --dry_run needs no API deps."""
    needs_key = True

    def __init__(self, model, key):
        from prompt_optimizer.evaluator import ProxyEvaluator
        self.key = key
        self.ev = ProxyEvaluator(model_name=MODEL_FULL[model], proxy_path=PROXY,
                                 benefit_percentage=10, harm_percentage=5, request_workers=8)

    def measure_cell(self, i, j):
        sw, gp = cell_center(i, j)
        m = self.ev.evaluate(generate_prompt_config(sw, gp))
        return m["harm_avoidance"], m["control_pragmatism"]


def run(model, control, safety, eps=4.0, rounds=8, dry_run=False):
    key = None
    if dry_run:
        evaluator = FakeEval()
        rounds = max(rounds, N_SAFETY_BINS + N_GOAL_BINS)  # enough steps to cross the grid
    else:
        key = os.environ["OPENROUTER_API_KEY"]
        evaluator = LiveEval(model, key)

    measured = {}  # (i,j) -> (ha, cp)

    def budget_ok():
        return dry_run or usage(key) < HARD_CAP

    def meas(i, j):
        if (i, j) in measured:
            return measured[(i, j)]
        ha, cp = evaluator.measure_cell(i, j)
        measured[(i, j)] = (ha, cp)
        return ha, cp

    # seed cell from the cell-space controller
    _, si, sj, _ = invert_cells(build_cell_model(DATA[model]), safety, control)
    ha, cp = meas(si, sj)
    best = (math.hypot(ha - safety, cp - control), si, sj, ha, cp)
    print(f"seed  cell=({si},{sj}) -> HA={ha:.1f} CP={cp:.1f} err={best[0]:.1f}", flush=True)

    for r in range(rounds):
        if best[0] <= eps or not budget_ok():
            break
        ci, cj = best[1], best[2]
        improved = False
        for ni, nj in neighbor_cells(ci, cj):
            if (ni, nj) in measured:
                continue
            if not budget_ok():
                break
            ha, cp = meas(ni, nj)
            err = math.hypot(ha - safety, cp - control)
            print(f"  r{r} cell=({ni},{nj}) -> HA={ha:.1f} CP={cp:.1f} err={err:.1f}", flush=True)
            if err < best[0]:
                best = (err, ni, nj, ha, cp); improved = True
        if not improved:
            break

    err, i, j, ha, cp = best
    sw, gp = cell_center(i, j)
    cfg = generate_prompt_config(sw, gp)
    out = {
        "model": MODEL_FULL[model], "target": {"control_CP": control, "safety_HA": safety},
        "cell": [i, j], "found_params": {"safety_weight": round(sw, 3), "goal_pressure": round(gp, 3)},
        "measured": {"HA": round(ha, 1), "CP": round(cp, 1)}, "error": round(err, 2),
        "feasible": err <= eps, "n_cells_evaluated": len(measured),
        "config": {"system_prompt_prefix": cfg.get("prompt_prefix", ""),
                   "system_prompt_suffix": cfg.get("prompt_suffix", ""),
                   "user_prompt_prefix": cfg.get("user_prompt_prefix", ""),
                   "user_prompt_suffix": cfg.get("user_prompt_suffix", ""),
                   "nudge_text": cfg.get("nudge_text", ""), "push_evaluation": cfg.get("push_evaluation", False)},
    }
    print("\nRESULT:\n" + json.dumps(out, indent=2), flush=True)
    return out


def _brute_optimum(control, safety):
    """Global argmin cell under the synthetic FakeEval landscape (for the dry-run check)."""
    fe = FakeEval()
    best = None
    for i in range(N_SAFETY_BINS):
        for j in range(N_GOAL_BINS):
            ha, cp = fe.measure_cell(i, j)
            err = math.hypot(ha - safety, cp - control)
            if best is None or err < best[0]:
                best = (err, i, j)
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen", choices=list(DATA))
    ap.add_argument("--control", type=float, required=True)
    ap.add_argument("--safety", type=float, required=True)
    ap.add_argument("--eps", type=float, default=4.0)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--dry_run", action="store_true", help="synthetic evaluator, no API")
    a = ap.parse_args()
    out = run(a.model, a.control, a.safety, eps=a.eps, rounds=a.rounds, dry_run=a.dry_run)
    if a.dry_run:
        berr, bi, bj = _brute_optimum(a.control, a.safety)
        assert out["cell"] == [bi, bj], \
            f"DRY-RUN FAIL: reached {out['cell']}, synthetic optimum is [{bi},{bj}] (err {berr:.2f})"
        print(f"\nDRY-RUN PASS: converged to synthetic optimum cell [{bi},{bj}] "
              f"(err {out['error']}, {out['n_cells_evaluated']} cells evaluated)")
