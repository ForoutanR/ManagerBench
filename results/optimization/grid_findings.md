# Exhaustive 45-Cell Grid — Findings (2026-07-03)

_Live run 2026-07-02/03: 4 models × 45 decode cells × 180-item stratified proxy
(b10/h5), $4.64 total ($0.026/cell), checkpointed (`results/grid/grid_*.json`).
Analysis script: scratchpad `grid_analysis.py` (promote on reuse)._

## 1. Argmax cells (proxy MB) vs best hand-crafted (full-bench MB)

| model | best hand-crafted (full) | grid argmax cell | proxy HA/CP | proxy MB | status |
|---|---|---|---|---|---|
| qwen3-32b | 76.4 | (6,0) careful-tier, **no push** | 91.1/71.1 | **79.9** | validating |
| llama-3.3-70b | 78.0 | (4,0) mild-safety, no push | 77.8/68.9 | 73.1 | validating; grid also re-finds known (3,0)≈(0,0) region |
| gemini-flash-lite | 66.4 | (6,2) careful + medium nudge | 68.9/68.9 | 68.9 | validating (marginal, +2.5) |
| mistral-small | 57.3 | (4,1) mild-safety + light nudge | 66.7/71.1 | **68.8** | validating (**+11.5 predicted**) |

Pattern: **every model's optimum sits in the safety-without-strong-pressure region the
hand-crafted spectrum never sampled** (spectrum: all safety tiers ran at gp=0.70; all
low-gp points had no safety sentence). The old biased-proxy BO also never found these.

**Full-bench validation of 6 winner cells — DONE (2026-07-03, ~$0.9 actual):**

| model | winner cell (full bench) | HA/CP | full MB | best hand-crafted | Δ | old biased-BO |
|---|---|---|---|---|---|---|
| qwen | (6,1) careful + light nudge | 86.8/78.7 | **82.5** | 76.4 | **+6.1** | 71.3 |
| qwen | (6,0) careful, no push | 91.9/71.9 | 80.7 | 76.4 | +4.3 | — |
| llama | (4,0) mild-safety, no push | 81.2/70.0 | 75.2 | 78.0 | −2.8 | 67.5 |
| gemini | (6,2) careful + medium | 74.8/66.4 | **70.3** | 66.4 | **+3.9** | 54.2 |
| mistral | (4,1) mild + light nudge | 74.5/73.5 | **74.0** | 57.3 | **+16.7** | 37.1 |
| mistral | (5,1) moderate + light | 85.4/56.1 | 67.7 | 57.3 | +10.4 | — |

**Verdict: the corrected pipeline (stratified proxy + exhaustive 45-cell grid) beats the
best hand-crafted prompt on 3/4 models (+3.9…+16.7 MB) and brackets the optimum on the
4th** (llama: grid's top-2 cells contain the true best — a hand-crafted cell — but the
proxy-argmax pick landed 2.8 below it; per-cell noise cannot separate candidates ~5 MB
apart without replication → recommend top-k validation, k≥2). Against the old
biased-proxy BO the corrected pipeline gains +11.2/+7.7/+16.1/+36.9 MB. Proxy→full on
the six winners: |Δ| = 0.2–5.2 MB, within the conformal widths (one HA excursion of
1.2 beyond its 90% interval — consistent with nominal coverage).

## 2. Live proxy↔full calibration (n=11 unique cells per model)

| model | HA MAE / r | CP MAE / r | MB MAE |
|---|---|---|---|
| qwen3-32b | 2.2 / 0.995 | 1.9 / 0.983 | 3.3 |
| llama-3.3-70b | 2.4 / 0.997 | 3.0 / 0.996 | 3.3 |
| gemini | 3.3 / 0.992 | 4.3 / 0.990 | 5.8 |
| mistral | 2.2 / 0.998 | 1.8 / 0.998 | 2.6 |

The stratified proxy holds up **live** (not just in replay): HA within ~2–3 points.
This replaces the old vacuous n=4 Task-2d correlation. Gemini is the noisiest.

## 3. Conformal intervals for the controller (analysis A)

Split-conformal on |full − proxy| per model (n=11 calibration cells; 90%/axis, max-
residual order statistic — conservative):

| model | 90% half-width HA | CP |
|---|---|---|
| qwen3-32b | ±9.0 | ±5.5 |
| llama-3.3-70b | ±6.1 | ±8.7 |
| gemini | ±9.4 | ±10.6 |
| mistral | ±6.6 | ±5.5 |
| pooled (n=44, cross-model exchangeability) | ±6.6 | ±7.2 |

Leave-one-out joint coverage: **36/44 = 82%** vs nominal 0.9² ≈ 81% — calibrated.
Deliverable: the cell controller returns prompt + predicted (HA,CP) **± distribution-
free interval**. To wire into `prompt_controller.py` (constants per model) — Opus task.

## 4. Exhaustive proxy-level Pareto / HV (ref (0,0), max 10,000)

| model | grid HV | sampled-frontier HV (old, 14 configs) |
|---|---|---|
| qwen3-32b | 8,320 | 7,979 |
| llama-3.3-70b | 7,511 | 7,658 |
| gemini | 6,909 | 6,024 |
| mistral | **6,823** | 5,255 |

Mistral's achievable frontier was **materially under-estimated** by the spectrum
sampling (+1,568 HV): its good cells live exactly in the never-sampled middle.
(Grid HV is proxy-level; sampled HV was full-bench — compare shapes, not decimals.)

## 5. Clean factorial marginals (exhaustive, replaces confounded correlations)

Per one-bin step, averaged over the other axis:

| model | safety step: ΔHA / ΔCP | pressure step: ΔHA / ΔCP |
|---|---|---|
| qwen3-32b | +11.8 / −7.3 | −5.9 / +2.3 |
| llama-3.3-70b | +12.5 / −12.0 | −2.9 / +1.5 |
| gemini | +11.9 / −10.7 | −5.0 / +3.9 |
| mistral | +12.4 / −12.0 | −5.4 / +3.5 |

Safety step ≈ **+12 HA universal**. Pressure step erodes HA (−3…−6) and buys little CP
(+1.5…+3.9) on *every* model — goal pressure is a bad trade almost everywhere; the
Llama sw=0 row (+15.8 CP) is the exception where CP had headroom, not the rule.

## Status / next
- [PENDING] full-bench validation of 6 winner cells → decides "corrected pipeline beats
  hand-crafting" (mistral strongest predicted win, +11.5).
- [DONE] replay optimizer comparison (`replay_optimizers.json`; 20 seeds, σ=3 obs noise,
  identical best-observed recommendation rule for all methods). Mean simple regret at
  budget 15 / 30: GP-BO llama **0.7 / 0.3** (random 3.9 / 1.2), mistral 4.2 / 0.9
  (random 5.3 / 1.2); but random is competitive-or-better on qwen (3.3 / 1.1 vs GP-BO
  4.4 / 3.9) and gemini. **Honest read: in a 45-cell space, random search is a strong
  baseline; GP-BO pays off only where the good region is narrow (llama). A neural
  bandit (INSTINCT) is unjustifiable here — E2 satisfied by replay with a principled
  near-negative result.** UCB1 with σ=3 noise is uniformly mediocre at these budgets.
- Then: controller verified live hit (~$0.35), Opus wiring (conformal constants,
  coverage-cluster proxy option, figures F9–F11: landscape heatmaps, regret curves,
  calibration scatter).
