# When the Proxy Steers the Search: Biased Subset Evaluation Flips the Outcome of Black-Box Prompt Optimization for Safety

*Workshop cut (target: NeurIPS'26-cycle safety/evaluation workshop; ~5 pages + appendix,
non-archival). Figures [F#] to render — see Appendix map. All numbers reproducible from
the repo; benefit10/harm5 unless noted.*

## Abstract

Black-box prompt optimization needs cheap evaluation, so candidates are scored on a
subset of the benchmark. We show, on ManagerBench's safety–pragmatism trade-off, that
the *choice of subset* can silently decide the outcome of the search. Scoring prompts
on the intuitive "most-discriminating items" subset — and on the *boundary/uncertainty*
selections used by recent performance-guided methods — is a **biased estimator** of the
full benchmark (~13 points of systematic error vs 2.5 for random sampling of the same
budget), the bias is **config-conditional and cannot be removed by calibration**, and
it **mis-steers optimization**: Bayesian optimization scored by the biased proxy
returned prompts that *lose* to the best hand-crafted prompt on 4/4 models, with 3 of
its 4 "winners" decoding to byte-identical copies of existing prompts. Repairing only
the evaluation — representative sampling plus an exhaustive sweep of the 45-cell
decoded prompt space ($4.6 total) — **reverses the verdict**: validated winners beat
the best hand-crafted prompt on 3/4 models (+3.9 to +16.7 MB), all in a
safety-framing-without-strong-pressure region no baseline had sampled. We package the
repaired pipeline as a **controller with distribution-free guarantees**: given a target
(harm-avoidance, pragmatism) operating point, it returns a prompt, a prediction with
split-conformal intervals (±6–11 points at 90%, empirical coverage 82% vs 81% nominal),
and a closed-loop verified option that hit a live target within 2.1 points in 14
evaluations. Proxy design is not an implementation detail of prompt optimization — it
can flip the sign of the result.

## 1. Introduction

Autonomous LLMs trade **harm avoidance** (HA: refusing harmful-but-effective actions)
against **pragmatism** (CP: acting when action is safe). ManagerBench [Simhi 2025]
measures both; prompt optimization promises to *steer* a fixed model along this
frontier. Because each candidate evaluation costs hundreds of model calls, every
practical optimizer scores candidates on a **subset** — and the literature increasingly
recommends *informative* subsets: high-variance items, decision-boundary items,
performance-guided selections [IPOMP 2505.10736].

**Contributions.**
1. **A cautionary result with mechanism.** Informativeness-selected subsets
   (variance-, boundary-, and coverage+boundary-selected) are biased estimators of
   full-benchmark HA/CP; the bias is config-conditional (no affine calibration fixes
   it) and large (~13 MB points). Representative sampling of the same budget is nearly
   exact (2.5–4.2). Theory: variance/boundary selection over-keeps items with
   p ≈ 0.5, dragging subset means toward 50% in a config-dependent way [Active
   Testing 2103.05331].
2. **Demonstrated search damage — and repair.** The same optimizer, same budget scale,
   opposite verdicts: scored by the biased proxy, BO loses to the best hand-crafted
   prompt on 4/4 models; scored representatively over the (exhaustively enumerable)
   45-cell decoded space, it wins on 3/4 (+3.9…+16.7) and brackets the optimum on the
   4th. The winning prompts live in a region (safety framing *without* strong goal
   pressure) that neither hand-crafting nor the misled search ever sampled.
3. **A controller with guarantees.** The forward map (prompt cell → HA, CP) inverts
   into a deployment tool: target operating point in, prompt out, with split-conformal
   intervals on the achieved point and an optional closed-loop verification that hit a
   live target within 2.1 points.

## 2. Setup

**Benchmark & metrics.** ManagerBench poses managerial dilemmas: treatment items where
the goal-achieving option harms a person (safe choice ⇒ HA) and control items where the
effective option is harmless (effective choice ⇒ CP); MB = harmonic mean, reported only
as a summary. Main slice: 357 treatment-high-harm + 253 control items; unparseable
answers count as wrong.

**Prompt space.** Two interpretable ordinal knobs decoded by template: a
safety-framing sentence (9 levels, from anti-safety to maximal) × a goal-pressure nudge
(5 levels, none → "at all costs"). The decode is a step function: the continuous
parameterization collapses to **45 distinct prompts** — small enough to enumerate,
which gives exact ground truth for evaluating optimizers ex post.

**Proxy evaluation.** Candidates are scored on 90+90 items selected from the pools;
selection strategy is the object of study. Full-benchmark validation of winners is
always performed.

## 3. The bias result

**Estimation.** Leave-one-prompt-out replay over 56 configs (14 prompts × 4 models,
saved per-item answers; zero API): select the subset on the other prompts, predict the
held-out prompt's full HA/CP from one 90+90 draw.

| selection (180 items) | MB MAE [95% CI] | HA MAE | CP MAE | MB r |
|---|---|---|---|---|
| variance / discrimination | 12.9 [9.9, 16.0] | 11.0 | 8.5 | 0.81 |
| boundary (à la performance-guided methods) | **13.2** | 10.8 | 8.9 | — |
| coverage-clustering + boundary | 10.9 | 8.6 | 7.0 | — |
| difficulty-stratified | 4.2 [3.3, 5.3] | 3.6 | 3.1 | 0.98 |
| coverage-clustering alone | 3.0 | 2.4 | 1.7 | — |
| random | 2.5 [2.0, 2.9] | 1.8 | 1.4 | 0.999 |

Three observations. (i) The informative selections are the *worst on every metric,
including ranking* — the one thing they are meant for. (ii) The failure decomposes:
**coverage is fine, boundary is toxic** — adding a boundary term to a good coverage
selection triples its error. Pipelines with uncertainty/boundary components inherit the
bias. (iii) The two error types differ in kind: the biased selections' error is
systematic (identical under 1-draw and 5-draw-ensembled scoring), the representative
selections' is sampling noise (2.5 → 0.8 when ensembled). **No affine calibration
repairs the bias** (12.9 → 9.6 only): regression of proxy on full score gives slope
≈ 1.0 for *all* strategies — the bias is a config-conditional offset, not a shrinkage
law, so it must be fixed in selection, not post-hoc. [F4: MAE bars + CI]

**Why.** A binary item's across-config variance (and boundary proximity) peaks at
p ≈ 0.5, so these rules over-keep borderline items whose mean sits near 50% regardless
of the config's true score — the classic active-testing bias [2103.05331], here shown
inside a prompt-optimization loop.

## 4. The search damage — and the repair

**Damage.** 25-trial TPE runs per model, scored by the variance proxy (and, we found,
with the intended warm start silently absent — cold starts covered only 16–18 of 45
cells): the returned "optima" **lose to each model's best hand-crafted prompt on 4/4
models** (−5.2, −10.5, −12.2, −20.2 MB), and three of four decode to byte-identical
copies of hand-crafted prompts, so their apparent gains were re-measurement noise
(identical prompts re-measure within ΔHA ≤ 4.7). Where the search *did* visit the true
best cell, the biased proxy mis-ranked it (Mistral: true-best cell scored 41.8–44.3 vs
62.9 for the empty prompt whose true MB is 37.1).

**Repair.** Fix only the evaluation: stratified 90+90 proxy, full-bench-consistent
scoring, and — since the space is 45 cells — an exhaustive sweep (180 cell-evals,
$4.64). Validated winners on the full benchmark:

| model | grid winner (full bench) | best hand-crafted | Δ | biased-proxy BO |
|---|---|---|---|---|
| qwen3-32b | **82.5** (HA 86.8 / CP 78.7) | 76.4 | **+6.1** | 71.3 |
| gemini-flash-lite | **70.3** (74.8 / 66.4) | 66.4 | **+3.9** | 54.2 |
| mistral-small | **74.0** (74.5 / 73.5) | 57.3 | **+16.7** | 37.1 |
| llama-3.3-70b | 75.2 (81.2 / 70.0) | 78.0 | −2.8 | 67.5 |

Same benchmark, same kind of budget: **proxy design alone flips the sign of the
optimization result.** On llama the grid's top-2 cells bracket the true optimum (itself
a hand-crafted cell); the single argmax missed by a within-noise margin — validate
top-k, not top-1. Every winner sits at *moderate-to-careful safety framing with no or
light pressure*: the hand-crafted spectrum ran all its safety tiers under strong
pressure, and the biased search never got there. Mistral — "barely steerable" on the
sampled frontier (hypervolume 5,255) — has an exhaustive-frontier HV of 6,823: its poor
steerability was a sampling artifact. [F9: landscape heatmaps; F6: wins bar chart]

**Live proxy quality.** On the 11 cells per model with both measurements, the
stratified proxy tracks the full benchmark at HA MAE 2.2–3.3 (r ≥ 0.99) — including the
extrapolation regime: winner-cell proxy→full deltas were 0.2–5.2 MB.

**Is the optimizer even needed?** Replaying optimizers on the measured grids (20 seeds,
σ=3 observation noise, identical recommendation rule): random search is a strong
baseline in a 45-cell space (regret ≤ 1.6 at 30 evals on all models); GP-BO helps only
where the good region is narrow (llama: regret 0.7 at 15 evals vs 3.9 random); UCB1 is
uniformly mediocre at these budgets. The efficiency story of this pipeline is the
**cheap trustworthy proxy**, not optimizer sophistication — and a neural-bandit
optimizer is unjustifiable here. [F10: regret curves]

## 5. What the knobs actually do

**A useful lens: the model is a risk classifier, prompts move its threshold.** Treat
each dilemma as carrying a latent risk score the model refuses above some threshold.
Then HA is the true-positive refusal rate on harmful items, CP is one minus the
false-refusal rate on benign items, the measured frontier is the model's **ROC curve
for harm discrimination**, and hypervolume is its AUC analogue. Prompts shift the
*threshold* along a fixed curve; the curve's quality — how well the model separates
harmful from benign — is what differs between models. Qwen holding CP 92 under an
HA ≥ 50 floor while Mistral collapses is a *discrimination* difference, not a
knob-response difference; steerability is bounded by classifier quality.

Exhaustive marginals support the single-threshold picture (per one-level step,
averaged over the other axis): the safety knob moves HA by **+12 per step on every
model** with an opposite-signed CP cost (−7…−12); the pressure knob
*erodes* HA (−3…−6 per step) while buying little CP (+1.5…+3.9). The often-repeated
recipe "add goal pressure for effectiveness" is a bad trade almost everywhere: in the
controlled sw=0 dose–response, none→strong pressure costs HA −52 (llama) to −7.5
(gemini) while CP gains exceed +4 only where CP had headroom (llama, 81→97; ceiling
effect elsewhere). Operating points are properties of the *prompt*, not the items:
split-half item agreement r ≥ 0.989 (MAE ≤ 2.8), and stated benefit/harm magnitudes in
the scenario text shift behavior by < 5 points. Prompts do **not** transfer across
models (a qwen safety prompt drives llama to HA 100/CP 14.6). [F8: dose–response;
F3: marginals]

**But levels are not semantics.** Three human-written, level-preserving paraphrases of
each winner's safety sentence move the operating point by whole tiers on 3 of 4 models
(gemini: CP −37…−65, flipping a balanced 70/71 prompt to near-total refusal 97/6;
mistral HA −13…−28; qwen CP up to −33; llama stable within ±8) — far beyond
neighbor-level distances, so this is surface-form sensitivity, not level drift. The
knob abstraction indexes *specific template strings*; all guarantees in §6 attach to
those strings, and the replication study (3 fresh runs per config, win-or-tie 3/3 for
every winner on the three winning models) confirms the wins are stable **for the
frozen templates**. Practically: prompt-based safety steering is wording-fragile —
deployments should freeze and regression-test exact prompt text.

## 6. The controller

Inverting the measured map turns steering into a tool: given a target (HA*, CP*), pick
the cell minimizing distance in measured/interpolated (HA, CP), return the decoded
prompt with a **split-conformal interval** on the achieved point (per-model 90%
half-widths: HA ±6.1–9.4, CP ±5.5–10.6; leave-one-out joint coverage 82% vs 81%
nominal), plus a feasibility flag for off-frontier targets. A live closed-loop run
(qwen, target HA 88 / CP 75, ε=5) seeded at the wrong tier, corrected across cell
neighbors, and hit **(86.7, 76.7), error 2.1, in 14 evaluations (~$0.6)** — landing on
the same cell that independent full-bench validation scored at 86.8/78.7. To our
knowledge this is the first prompt controller for a safety operating point with
distribution-free guarantees. [F7: trajectory]

**External validity.** To check the operating point is not a ManagerBench artifact, we
ran each winner and its baseline on XSTest (over-refusal) and HarmBench's 200 standard
behaviours (attack-success), judged by an LLM (the string heuristic disagrees 40–55%, so
judge is primary). Every winner has a **lower HarmBench attack-success than its baseline**
on all four models (gemini 0.0 vs 1.5, llama 2.5 vs 5.0, mistral 3.0 vs 22.5, qwen 0.0 vs
2.5), at an XSTest over-refusal cost of at most +3.6 points. The prompts are externally
safer at a small, bounded over-refusal cost. Caveats: single LLM judge (self-preference
risk where it scores Gemini's own outputs), judge-based HarmBench ASR rather than the
official classifier.

## 7. Related work

Subset evaluation: tinyBenchmarks [2402.14992], metabench [2407.12844], Anchor Points
[2309.08638]; graph/submodular benchmark compression [2606.01400, 2605.02209] targets
ranking consistency across models — orthogonal to our per-config score-estimation
setting; Active Testing [2103.05331] proves informativeness-selection bias — we
demonstrate it *inside a search loop* and measure the damage; "Misses the Mark"
[2506.07673] warns proxies degrade under extrapolation — our stratified proxy held
(winner deltas ≤ 5.2). Performance-guided selection for prompt optimization: IPOMP
[2505.10736] — our ablation isolates its boundary principle as the toxic component.
Prompt optimization/BO: OPRO, ProTeGi, MIPRO [2406.11695], InstructZero/INSTINCT,
ReElicit [2605.19093] (single-objective GP twin; its own ablation supports fixed
interpretable spaces). Multi-objective: COM-BOM [2510.01178] (accuracy–calibration
Pareto; capability not safety). Safety–helpfulness frontiers: Safe RLHF [2310.12773],
Panacea [2402.02030] — ours is "frontier traversal in prompt space, training-free";
over-refusal as a real cost: XSTest [2308.01263], OR-Bench [2405.20947]. Conformal
methods for LLMs are active [2604.13991]; none target a prompt controller.

## 8. Limitations & conclusion

Four mid-size models, one benchmark, one decode template (45 cells; larger/learned
spaces re-open the search question our replay closes for small ones); n = 11
calibration cells per model make conformal intervals conservative, and mid-frontier
configs occasionally exceed them; same-prompt re-measurement noise (ΔHA up to 4.7)
bounds all single-point claims; **all guarantees attach to the exact template strings —
paraphrase sensitivity (§5) means semantic "levels" do not transfer across wordings**;
external-benchmark generalization untested.

Future work follows from the threshold lens (§5): enlarge the prompt space along axes
the 45 cells cannot express (tone, role, format, paraphrase variants as an explicit
robustness dimension), trace the larger front with noise-aware multi-objective BO
(qNEHVI, or ParEGO where dependencies must stay light), keep the stratified proxy +
top-k full-bench validation + conformal wrap, and test whether prompt-induced threshold
shifts transfer to an external agentic-safety benchmark.

The core lesson stands independent of scale: **subset evaluation is part of the
optimizer.** Selecting items for informativeness quietly biases scores, resists
calibration, and can steer a black-box search below hand-crafted baselines — while
representative sampling of the same budget, checked by exhaustive enumeration and
conformal validation, turns the same pipeline into a reliable safety-steering
controller.

---

## Appendix map (supplementary)
- A. Corrected-record audit trail (the biased-run forensics: warm-start absence,
  decode-identity of "optimized" prompts, cell coverage). Source: CRITICAL_ASSESSMENT.md.
- B. Full LOPO tables with CIs, parse-convention ablation, size sweep
  (proxy_selection_findings_v2.md); IRT negative result.
- C. Grid data + landscapes per model (results/grid/), calibration scatter.
- D. Replay protocol + regret tables (replay_optimizers.json).
- E. Conformal construction + coverage details; controller trajectory log
  (controller_hit.log).
- F. Dose–response and marginals tables (grid_findings.md §5); split-half details.
- G. Reproducibility: seeds, costs ($4.64 grid + $0.9 validation + $0.6 controller),
  scripts, VPS/throttle constraints.

## Figure map
- F3 marginals (bar pairs per model) · F4 LOPO MAE + CI · F6 wins vs hand-crafted ·
  F7 controller trajectory on the (HA,CP) plane · F8 pressure dose–response ·
  F9 grid heatmaps (HA, CP, MB per model) · F10 regret curves.
