# Rigor batch findings — replication, paraphrase, (stakes pending) — 2026-07-04

_Items 1+4 of the $4 rigor batch (`run_rigor_batch.py` → `rigor_batch.json`);
3 fresh 180-item stratified-proxy runs per config; paraphrases in
`prompt_optimizer/paraphrases.py` (human-written, level-preserving)._

## 1. Replication ×3 — the wins are stable

Proxy MB, mean ± sd over 3 fresh runs (grid single-run in brackets):

| model | config | MB ±sd | grid | full bench |
|---|---|---|---|---|
| qwen | winner (6,1) | 80.3 ± 1.9 | 86.7/73.3 | 82.5 |
| qwen | winner (6,0) | 77.3 ± 1.3 | 91.1/71.1 | 80.7 |
| qwen | best-hand (7,3) | 70.5 ± 1.3 | — | 76.4 |
| llama | winner (4,0) | 69.4 ± 0.7 | — | 75.2 |
| llama | best-hand (3,0) | 71.3 ± 0.8 | — | 78.0 |
| gemini | winner (6,2) | 70.2 ± 1.8 | — | 70.3 |
| gemini | best-hand (6,3) | 65.2 ± 2.8 | — | 66.4 |
| mistral | winner (4,1) | 68.4 ± 2.3 | — | 74.0 |
| mistral | winner (5,1) | 64.4 ± 1.2 | — | 67.7 |
| mistral | best-hand (5,3) | 49.1 ± 2.1 | — | 57.3 |

**Win-or-tie (winner vs best-hand, per-rep MB):** qwen 3/3 and 3/3; gemini 3/3;
mistral 3/3 and 3/3; **llama 0/3** — fully consistent with the full-bench verdict
(llama's optimum is a hand-crafted cell). Replication σ ≤ 2.8 MB: the headline wins are
not sampling luck. Two coverage notes for the conformal section: mistral (5,3) proxy HA
sits ~10 below its full-bench value and qwen (7,3) HA has σ=5.1 — mid-frontier configs
remain the noisiest and are where the ±6–9 intervals strain.

## 2. Paraphrase robustness — **negative result, important: levels ≠ semantics**

Three human-written, level-preserving paraphrases (p1–p3) of each winner's safety
sentence, nudge unchanged; Δ vs the original template (p0 = replication mean):

| model (cell) | ΔHA range | ΔCP range | verdict |
|---|---|---|---|
| llama (4,0) | −8.1 … +5.2 | −2.6 … +7.4 | stable |
| qwen (6,1) | −1.9 … +9.3 | **−33.0** … −9.6 | fragile (2 of 3) |
| mistral (4,1) | **−27.8** … −13.3 | +9.6 … +15.2 | fragile |
| gemini (6,2) | +18.1 … **+27.0** | **−65.2** … −37.4 | extremely fragile |

Movements exceed neighbor-tier distances by 2–3× (e.g. gemini one-tier step ≈ −20 CP;
observed −47…−65): this is **surface-form sensitivity, not level drift**. Conclusions:

1. The decoded prompt's *exact wording*, not its semantic "level", determines the
   operating point on 3 of 4 models. The interpretable-knob abstraction indexes
   templates, and its guarantees (conformal intervals, controller hits) attach to the
   **shipped template strings only**. This caveat is now part of the method's claims.
2. As a standalone finding: **prompt-based safety steering is wording-fragile** —
   deployments must freeze and regression-test the exact prompt text; "equivalent"
   rewording can shift a model from balanced to max-refusal (gemini 70/71 → 97/6).
3. Llama's stability is itself notable (the model most responsive to *pressure* is the
   least sensitive to safety-sentence *form*) — model-specific surface sensitivity is
   another axis the field has not characterized.

Paper placement: new short subsection in §5 (characterization) + limitation in §8;
controller section gains the template-string caveat. NOT buried: this tempers the
interpretability claim and reviewers should see it prominently.

## 3. Stakes sweep of winners — DONE (recomputed from raw files; full bench, MB per slice)

| variant | 10/5 | 10/15 | 50/5 | 50/15 | span |
|---|---|---|---|---|---|
| qwen c6-1 | 82.5 | 82.6 | 84.9 | 83.9 | 2.4 |
| qwen c6-0 | 80.7 | 80.9 | 81.7 | 82.4 | 1.7 |
| llama c4-0 | 75.2 | 73.0 | 74.1 | 74.9 | 2.2 |
| gemini c6-2 | 70.3 | 67.1 | 70.4 | 69.5 | 3.3 |
| mistral c4-1 | 74.0 | 72.3 | 73.0 | (capped) | ≤1.7 |
| mistral c5-1 | 67.7 | 66.4 | 67.6 | (capped) | ≤1.3 |

**Winners are stakes-stable** (MB span ≤ 3.3; ΔHA ≤ 3 everywhere) — the same
stakes-wording insensitivity the original configs showed, now confirmed for the new
optima. Contrast with §2: robust to *stated stakes*, fragile to *safety-sentence
wording* — the model attends to the instruction's form, not the scenario's arithmetic.
Note: mistral's 50/15 pair was stopped by the budget cap ($4.01 of the approved $4.00);
completing it costs ~$0.25 if desired. Per-slice tagged comparison files were
suppressed by `--skip_summary` (runner quirk); numbers above recomputed offline from
the saved raw per-item files (`winner_stakes_sweep.json`).

## Cost
Items 1+4+5: $4.01 total (cap-stopped at approved $4.00). Key at ~$18.30/$30.
