#!/usr/bin/env python3
"""Rigor batch (items 1 + 4): replication x3 of winner/best-hand cells, and
paraphrase-robustness evals of winner cells. Proxy-based, checkpointed per eval,
usage-guarded. Item 5 (stakes sweep) runs separately via run_prompt_variants.

Usage (VPS): OPENROUTER_API_KEY=... HARD_CAP=<abs $> .venv/bin/python run_rigor_batch.py
"""
import json, os, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt_optimizer.param_space import generate_prompt_config, cell_center
from prompt_optimizer.evaluator import ProxyEvaluator
from prompt_optimizer.paraphrases import PARAPHRASES

HARD_CAP = float(os.environ.get("HARD_CAP", "18.0"))
PROXY = os.environ.get("PROXY", "bench_proxy/proxy_items.json")
OUT = "results/optimization/rigor_batch.json"

# (model, cell, tag)
WINNERS = [
    ("qwen/qwen3-32b", (6, 0), "winner"), ("qwen/qwen3-32b", (6, 1), "winner"),
    ("meta-llama/llama-3.3-70b-instruct", (4, 0), "winner"),
    ("google/gemini-2.5-flash-lite", (6, 2), "winner"),
    ("mistralai/mistral-small-3.2-24b-instruct", (4, 1), "winner"),
    ("mistralai/mistral-small-3.2-24b-instruct", (5, 1), "winner"),
]
BEST_HAND = [
    ("qwen/qwen3-32b", (7, 3), "best-hand"),
    ("meta-llama/llama-3.3-70b-instruct", (3, 0), "best-hand"),
    ("google/gemini-2.5-flash-lite", (6, 3), "best-hand"),
    ("mistralai/mistral-small-3.2-24b-instruct", (5, 3), "best-hand"),
]
N_REPS = 3
# paraphrase evals: primary winner per model, 3 paraphrases of its safety tier
PARA = [
    ("qwen/qwen3-32b", (6, 1)), ("meta-llama/llama-3.3-70b-instruct", (4, 0)),
    ("google/gemini-2.5-flash-lite", (6, 2)),
    ("mistralai/mistral-small-3.2-24b-instruct", (4, 1)),
]

def usage(key):
    req = urllib.request.Request("https://openrouter.ai/api/v1/key",
                                 headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)["data"]["usage"]

def load():
    if os.path.exists(OUT):
        return json.load(open(OUT))
    return {"replication": [], "paraphrase": []}

def save(d):
    tmp = OUT + ".tmp"
    json.dump(d, open(tmp, "w"), indent=2)
    os.replace(tmp, OUT)

def main():
    key = os.environ["OPENROUTER_API_KEY"]
    data = load()
    done_rep = {(r["model"], tuple(r["cell"]), r["rep"]) for r in data["replication"]}
    done_par = {(r["model"], tuple(r["cell"]), r["variant"]) for r in data["paraphrase"]}
    evals = {}

    def ev(model):
        if model not in evals:
            evals[model] = ProxyEvaluator(model_name=model, proxy_path=PROXY,
                                          benefit_percentage=10, harm_percentage=5,
                                          request_workers=8)
        return evals[model]

    def guard():
        u = usage(key)
        if u >= HARD_CAP:
            print("CAP hit ($%.4f >= $%.2f) — stopping cleanly" % (u, HARD_CAP), flush=True)
            return False
        return True

    print("start usage=$%.4f cap=$%.2f" % (usage(key), HARD_CAP), flush=True)

    # item 1: replication x3
    for model, cell, tag in WINNERS + BEST_HAND:
        for rep in range(N_REPS):
            if (model, cell, rep) in done_rep:
                continue
            if not guard():
                return
            m = ev(model).evaluate(generate_prompt_config(*cell_center(*cell)))
            rec = {"model": model, "cell": list(cell), "tag": tag, "rep": rep,
                   "ha": m["harm_avoidance"], "cp": m["control_pragmatism"],
                   "n_unparsed": m.get("n_unparsed", 0),
                   "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
            data["replication"].append(rec); save(data)
            print("rep  %-14s cell %s r%d -> HA %.1f CP %.1f" %
                  (model.split("/")[1][:14], cell, rep, rec["ha"], rec["cp"]), flush=True)

    # item 4: paraphrase robustness (variants p1..p3; p0 covered by replication)
    for model, cell in PARA:
        i, j = cell
        base = generate_prompt_config(*cell_center(i, j))
        for v in (1, 2, 3):
            if (model, cell, v) in done_par:
                continue
            if not guard():
                return
            cfg = dict(base); cfg["prompt_prefix"] = PARAPHRASES[i][v]
            m = ev(model).evaluate(cfg)
            rec = {"model": model, "cell": list(cell), "variant": v,
                   "ha": m["harm_avoidance"], "cp": m["control_pragmatism"],
                   "n_unparsed": m.get("n_unparsed", 0),
                   "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
            data["paraphrase"].append(rec); save(data)
            print("para %-14s cell %s p%d -> HA %.1f CP %.1f" %
                  (model.split("/")[1][:14], cell, v, rec["ha"], rec["cp"]), flush=True)

    print("RIGOR BATCH DONE  usage=$%.4f" % usage(key), flush=True)

if __name__ == "__main__":
    main()
