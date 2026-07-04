# Chapter 5 — Search Damage and Repair

*Draft. Source: COMPLETE_REPORT §14–15 + addenda, grid_findings.md, replay_optimizers.json.
Figures: F6 (wins vs hand-crafted), F9 (grid heatmaps), F10 (regret curves).*

Chapter 4 established that an informativeness-selected proxy mis-estimates
full-benchmark scores. This chapter shows the consequence that makes the result matter:
scored by that proxy, a black-box optimizer returns prompts that lose to hand-written
baselines on every model — and repairing only the evaluation reverses the outcome.

## 5.1 The misled search

We ran 25-trial Tree-structured Parzen Estimator (TPE) optimization per model, scoring
each candidate on the discrimination proxy of Chapter 4. Three facts about these runs,
each verified from the saved trial records, explain what went wrong.

First, the intended warm start from the hand-crafted spectrum never executed: every
run's records show zero warm-start trials, and the first trial coordinates are identical
across models, i.e. a cold start from a fixed seed. The search began blind to every
hand-crafted result.

Second, the two-knob space decodes to only 45 distinct prompts (a step function of 9
safety levels × 5 pressure levels). The 25 cold trials visited only 16–18 of the 45
cells per model, and for two models the cell containing the true optimum was never
sampled.

Third, where the search did visit a good cell, the biased proxy mis-ranked it. On
Mistral, the cell that is in truth the best (full-benchmark MB 57.3) scored 41.8–44.3 on
the proxy, while the empty-prompt cell (true MB 37.1) scored an inflated 62.9 — so the
optimizer preferred the empty prompt.

The result is that the optimizer's returned "optima" **lose to each model's best
hand-crafted prompt on all four models** (Table 5.1, middle columns). Worse, by decode
identity, three of the four returned prompts are byte-for-byte identical to existing
spectrum prompts (the Qwen "optimum" is the safe-prefix prompt; Gemini's is
balanced-safe; Mistral's is the empty baseline). Their apparent gains over those
variants were re-measurement noise: re-running an identical prompt on the full benchmark
moves HA by up to 4.7 points and CP by up to 5.6.

## 5.2 The repair

We changed only the evaluation, not the model, the space, or the optimizer family:
(i) representative (difficulty-stratified) proxy selection, replacing discrimination;
(ii) a scoring convention that counts unparseable answers as wrong, matching the full
benchmark (the live proxy evaluator had previously excluded them from the denominator, a
second, independent source of over-estimation for parse-fragile models); and (iii),
because the space is only 45 cells, an exhaustive sweep of every cell on every model
rather than a 25-trial sample. The exhaustive grid cost $4.64 in total API spend
(measured $0.026 per cell), was checkpointed after every cell to survive the connection
blackouts described in Chapter 3, and completed without incident.

## 5.3 Validated winners

For each model we took the grid's argmax cell(s) by proxy MB and validated them on the
full benchmark. Table 5.1 gives the outcome.

**Table 5.1 — Full-benchmark MB: hand-crafted vs biased-proxy BO vs repaired pipeline.**

| model | best hand-crafted | biased-proxy BO | repaired-pipeline winner | Δ vs hand-crafted |
|---|---|---|---|---|
| Qwen3-32B | 76.4 | 71.3 | **82.5** (HA 86.8 / CP 78.7) | **+6.1** |
| Gemini-Flash-Lite | 66.4 | 54.2 | **70.3** (74.8 / 66.4) | **+3.9** |
| Mistral-Small | 57.3 | 37.1 | **74.0** (74.5 / 73.5) | **+16.7** |
| Llama-3.3-70B | 78.0 | 67.5 | 75.2 (81.2 / 70.0) | −2.8 |

The repaired pipeline beats the best hand-crafted prompt on three of four models, by
+3.9 to +16.7 MB, and against the biased-proxy BO it gains +7.7 to +36.9. On Llama it
falls 2.8 short: Llama's true optimum is itself a hand-crafted cell, and the grid's
top-two cells bracket it, but the single proxy-argmax pick landed just below — a
consequence of per-cell measurement noise, and the reason we recommend validating the
top-k cells rather than only the top one. Every winning prompt occupies the same region
of the space — a clear safety instruction combined with *no or light* goal pressure —
that neither the hand-crafted spectrum (whose safety tiers were all tested under strong
pressure) nor the biased search ever sampled.

Two supporting measurements confirm the winners are trustworthy. First, the stratified
proxy tracks the full benchmark *live* (not only in replay): on the 11 cells per model
with both a proxy and a full-benchmark measurement, HA MAE is 2.2–3.3 and CP MAE 1.8–4.3
(r ≥ 0.99), and the winner cells' proxy-to-full gaps are 0.2–5.2 MB — the extrapolation
regime held. Second, three fresh proxy re-runs of each winner give a win-or-tie against
the model's best hand-crafted prompt of 3/3 on all three winning models (and 0/3 on
Llama, consistent with the full-benchmark verdict), with a replication standard
deviation of at most 2.8 MB.

## 5.4 Is the optimizer even necessary?

Because we measured the whole grid, we can ask what any optimizer would have achieved by
replaying it: draw noisy observations (σ = 3, matching the ~90-item binomial noise) from
the measured cells and report the simple regret of each optimizer's recommendation
against the true grid optimum. Random search is a strong baseline in a 45-cell space
(regret ≤ 1.6 MB at 30 evaluations on every model); GP-BO helps decisively only where
the good region is narrow (Llama: regret 0.7 at 15 evaluations versus 3.9 for random),
and UCB1 is uniformly mediocre at these budgets. The efficiency of this pipeline
therefore comes from the *cheap, trustworthy proxy*, not from optimizer sophistication —
and a neural-bandit optimizer, the primary method of the original proposal, cannot be
justified at this scale. The comparison the proposal asked for is delivered here by
replay, at zero additional cost.

## 5.5 The claim

The same benchmark, the same class of optimizer, and the same order of API budget
produce opposite verdicts depending only on how the cheap evaluation subset is chosen:
under informativeness selection the optimizer loses to hand-writing on 4/4 models; under
representative selection it wins on 3/4. **Proxy design does not merely distort reported
numbers — it can flip the sign of the optimization result.** For safety-relevant prompt
optimization, where the objective is precisely to move a model's harm behaviour, the
evaluation subset must be treated as part of the optimizer and validated as carefully as
the search itself.
