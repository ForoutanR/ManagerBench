#!/usr/bin/env python3
"""T2 acceptance: 1 of 10 answers garbage -> denominator 10, not 9; n_unparsed==1."""
import sys, os, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Stub Manager_bench (real module needs tqdm; broken venv per T10). We only exercise
# ProxyEvaluator._tally, which uses extract_final_ab_choice — stub it to parse
# "My answer is A/B" and return "" otherwise.
_mb = types.ModuleType("Manager_bench")
def extract_final_ab_choice(text):
    tail = (text or "").split("My answer is")[-1].strip()
    if tail[:1] == "A":
        return "A"
    if tail[:1] == "B":
        return "B"
    return ""
_mb.extract_final_ab_choice = extract_final_ab_choice
_mb.CreateDataset = object
_mb.DEFAULT_ANSWER_INSTRUCTION = ""
sys.modules["Manager_bench"] = _mb

# Stub prompt_optimizer.proxy (pulls numpy; not used by _tally).
_px = types.ModuleType("prompt_optimizer.proxy")
_px.BENCH_FILES = {}
_px.NON_HARM_HARM_TYPE = set()
_px.NOT_HARM_TOPICS = set()
_px._is_high_harm = lambda *a, **k: True
_px._parse_item_key = lambda *a, **k: (None, None, None, None)
sys.modules["prompt_optimizer.proxy"] = _px

from prompt_optimizer.evaluator import ProxyEvaluator

# 5 control (correct=A) + 5 treatment (correct=B); item index 4 is garbage/unparseable.
meta, resp = [], []
for i in range(5):
    meta.append({"is_control": True, "switch": False})
    resp.append("blah blah\nMy answer is A." if i != 4 else "I cannot decide, no clear option.")
for i in range(5):
    meta.append({"is_control": False, "switch": False})
    resp.append("reasoning\nMy answer is B.")

tb, tt, ca, ct, nun = ProxyEvaluator._tally(resp, meta)
denom = tt + ct
print(f"treatment_b={tb} treatment_total={tt} control_a={ca} control_total={ct} n_unparsed={nun}")
print(f"denominator = {denom}")
assert denom == 10, f"FAIL denominator {denom} != 10"
assert nun == 1, f"FAIL n_unparsed {nun} != 1"
assert ct == 5 and tt == 5, "FAIL arm totals"
assert ca == 4, f"FAIL control_a {ca} != 4 (garbage control must count wrong)"
assert tb == 5, f"FAIL treatment_b {tb} != 5"
# CP should be 80% (4/5): the garbage control counts as wrong, not dropped.
cp = 100.0 * ca / ct
print(f"CP = {cp:.1f}  (garbage counted wrong -> 4/5, not 4/4)")
assert abs(cp - 80.0) < 1e-9, "FAIL CP not 80"
print("PASS: unparseable stays in denominator and counts wrong.")
