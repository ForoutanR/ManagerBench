#!/usr/bin/env python3
"""
Exhaustive 45-cell proxy-landscape runner (OPUS_TASKS T7).

Purpose: the approved-pending API package. For each model x each of the 45 decode cells,
decode the cell center to a prompt config and evaluate it on the STRATIFIED proxy
(bench_proxy/proxy_items.json, benefit10/harm5), recording HA/CP and counts. The result
is a ground-truth landscape per model that repairs the optimization story, sensitivity
marginals, controller surrogate, and enables BO/bandit evaluation by replay ($0 after).

Resume-safe: checkpoints after EVERY cell to results/grid/grid_<model_slug>.json and, on
start, skips cells already recorded — so a DPI-killed SSH session just re-runs and
continues. In-process budget guard polls the OpenRouter usage endpoint before each cell
and aborts cleanly at HARD_CAP (env, default $10).

BUILD ONLY — a live run needs explicit cost approval (~$4.5 at 90+90 proxy, 4 models).
Use --dry_run (synthetic evaluator, $0) to validate resume/checkpoint behavior; dry-run
writes to grid_<slug>_dryrun.json so it never clobbers the real artifact.

Examples:
    .venv/bin/python run_grid.py --dry_run                       # all 4 models x 45 cells, synthetic
    .venv/bin/python run_grid.py --dry_run --models qwen/qwen3-32b --cells 0-3,0-4
    OPENROUTER_API_KEY=... .venv/bin/python run_grid.py          # LIVE (needs approval)
"""
import argparse, json, os, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt_optimizer.param_space import (
    generate_prompt_config, cell_center, N_SAFETY_BINS, N_GOAL_BINS,
)

HARD_CAP = float(os.environ.get("HARD_CAP", "10.0"))
PROXY = os.environ.get("PROXY", "bench_proxy/proxy_items.json")  # stratified as of T3
GRID_DIR = "results/grid"


def usage(key):
    req = urllib.request.Request("https://openrouter.ai/api/v1/key",
                                 headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)["data"]["usage"]


def synth_eval(i, j):
    """Deterministic synthetic (HA,CP)+counts per cell for --dry_run (no API).
    Monotone safety frontier + mild goal term (mirrors controller_verify.FakeEval)."""
    ha = max(0.0, min(100.0, 12.5 * i + 2.0 * (j - 2)))
    cp = max(0.0, min(100.0, 100.0 - 12.5 * i + 2.0 * (j - 2)))
    return {"harm_avoidance": ha, "control_pragmatism": cp,
            "n_treatment": 90, "n_control": 90, "n_unparsed": 0, "n_total_calls": 180}


def parse_cells(spec):
    """'i0-i1,j0-j1' -> list of (i,j). Default: all 45."""
    if not spec:
        return [(i, j) for i in range(N_SAFETY_BINS) for j in range(N_GOAL_BINS)]
    isp, jsp = spec.split(",")
    i0, i1 = (int(x) for x in isp.split("-"))
    j0, j1 = (int(x) for x in jsp.split("-"))
    return [(i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)]


def slug(model):
    return model.replace("/", "_")


def checkpoint_path(model, dry_run):
    name = "grid_%s%s.json" % (slug(model), "_dryrun" if dry_run else "")
    return os.path.join(GRID_DIR, name)


def load_records(path):
    if not os.path.exists(path):
        return [], set()
    with open(path) as f:
        recs = json.load(f).get("records", [])
    done = {tuple(r["cell"]) for r in recs}
    return recs, done


def save_records(path, model, recs):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"model": model, "proxy": PROXY, "n_records": len(recs), "records": recs}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)  # atomic-ish: a killed write leaves the prior checkpoint intact


def run_model(model, cells, dry_run, key):
    path = checkpoint_path(model, dry_run)
    recs, done = load_records(path)
    evaluator = None if dry_run else _make_live(model)
    n_eval = n_skip = 0
    print("=== %s -> %s (%d cells, %d already done) ===" % (model, path, len(cells), len(done)), flush=True)

    for (i, j) in cells:
        if (i, j) in done:
            n_skip += 1
            continue
        if not dry_run:
            u = usage(key)
            if u >= HARD_CAP:
                print("  BUDGET CAP hit (usage $%.4f >= $%.2f) — stopping cleanly; "
                      "re-run to resume." % (u, HARD_CAP), flush=True)
                break
        sw, gp = cell_center(i, j)
        m = synth_eval(i, j) if dry_run else evaluator.evaluate(generate_prompt_config(sw, gp))
        rec = {"model": model, "cell": [i, j], "sw": sw, "gp": gp,
               "ha": round(m["harm_avoidance"], 4), "cp": round(m["control_pragmatism"], 4),
               "n_treatment": m["n_treatment"], "n_control": m["n_control"],
               "n_unparsed": m.get("n_unparsed", 0), "n_calls": m["n_total_calls"],
               "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        recs.append(rec)
        done.add((i, j))
        save_records(path, model, recs)  # checkpoint after EVERY cell
        n_eval += 1
        print("  cell (%d,%d) sw=%+.3f gp=%.3f -> HA=%.1f CP=%.1f" %
              (i, j, sw, gp, rec["ha"], rec["cp"]), flush=True)

    print("  done: %d evaluated this run, %d skipped (resumed), %d total records" %
          (n_eval, n_skip, len(recs)), flush=True)
    return recs


def _make_live(model):
    from prompt_optimizer.evaluator import ProxyEvaluator
    return ProxyEvaluator(model_name=model, proxy_path=PROXY,
                          benefit_percentage=10, harm_percentage=5, request_workers=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None,
                    help="comma-separated model slugs; default reads models.txt")
    ap.add_argument("--cells", default=None, help="chunk 'i0-i1,j0-j1'; default all 45")
    ap.add_argument("--dry_run", action="store_true", help="synthetic evaluator, no API")
    a = ap.parse_args()

    if a.models:
        models = [m.strip() for m in a.models.split(",") if m.strip()]
    else:
        with open("models.txt") as f:
            models = [ln.strip() for ln in f if ln.strip()]
    cells = parse_cells(a.cells)

    key = None
    if not a.dry_run:
        key = os.environ["OPENROUTER_API_KEY"]
        print("LIVE run. start usage=$%.4f cap=$%.2f proxy=%s" % (usage(key), HARD_CAP, PROXY), flush=True)

    for model in models:
        run_model(model, cells, a.dry_run, key)


if __name__ == "__main__":
    main()
