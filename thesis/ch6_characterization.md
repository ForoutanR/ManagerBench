# Chapter 6 — How Prompts Move Each Model

*Draft. Source: COMPLETE_REPORT §19–22 + addenda, grid_findings.md §4–5, rigor_findings.md.
Figures: F3 (grid marginals), F8 (dose–response), F2/F9 (frontiers/heatmaps).*

Having established a trustworthy way to measure prompts (Chapter 4) and used it to find
good ones (Chapter 5), we now characterise *how* the two knobs move each model. The
exhaustive grid lets us report clean marginals rather than the confounded
trial-correlations of an adaptively-sampled search.

## 6.1 One universal lever, one that mostly costs safety

Averaging over the other axis, one step up the safety scale raises Harm Avoidance by
about +12 points on every model, at a Control-Pragmatism cost of −7 to −12. One step up
the goal-pressure scale *lowers* HA by 3 to 6 points while buying only +1.5 to +3.9 CP.
The safety knob is thus a reliable, universal control on harm avoidance; goal pressure
is largely a safety eroder whose pragmatism benefit is small.

The controlled dose–response at neutral safety framing (Table 6.1) makes the asymmetry
concrete. Increasing pressure from none to strong costs 52 points of HA on Llama and
7.5 on Gemini, while CP rises materially (+15.8) only on Llama — the one model with CP
headroom at neutral framing (81 → 97); the others already sit at 96–99 and have nothing
to gain. The common deployment recipe "add goal pressure to make the agent more
effective" is therefore a poor trade almost everywhere: it reliably reduces safety and
rarely improves usefulness.

**Table 6.1 — Goal-pressure dose–response at neutral safety (same items).**

| model | HA: none → med → strong | CP: none → med → strong |
|---|---|---|
| Llama | 74.8 → 54.3 → 23.0 | 81.4 → 88.9 → 97.2 |
| Mistral | 21.6 → 12.9 → 0.8 | 96.0 → 98.4 → 100.0 |
| Gemini | 8.1 → 3.4 → 0.6 | 99.2 → 98.8 → 100.0 |
| Qwen | 23.5 → 9.0 → 12.3 | 98.0 → 98.0 → 98.6 |

An earlier version of this project, using trial-correlations from the biased-proxy runs,
reported that goal pressure "drives pragmatism on Llama but is inert on Mistral." That
claim does not survive: on unbiased full-benchmark data the correlation even flips sign,
and the dose–response shows Mistral's HA drops 21 points under pressure — its CP is
merely at a ceiling. The apparent "inertness" was a ceiling effect plus a confounded
estimator, a direct example of why Chapter 4's correction matters for the science and
not only for the leaderboard.

## 6.2 The ROC lens: steerability is discrimination quality

The two metrics are best understood together as an operating point on a classifier. Read
the model as assigning each dilemma a latent risk and refusing above a threshold: Harm
Avoidance is the true-positive refusal rate on genuinely harmful items, and Control
Pragmatism is one minus the false-refusal rate on benign items. Under this reading the
measured frontier is the model's **ROC curve for harm discrimination**, hypervolume is
its area-under-curve analogue, and a prompt selects a *threshold* along a fixed curve.
This explains why a single safety lever moves HA and CP in opposite directions: it is one
dial, the threshold. It also reframes "steerability." A model that holds high CP under a
strict HA floor (Qwen: CP 92 at HA ≥ 50) has a better underlying separation of harmful
from benign items than one that collapses (Mistral at the same floor); the difference is
*discrimination quality*, which prompting cannot improve — prompting only chooses where
on the existing curve to sit.

## 6.3 What transfers and what does not

**Stakes wording (transfers).** The benefit and harm percentages in ManagerBench are text
substituted into otherwise identical items, so a sweep over them tests sensitivity to the
*stated magnitude* of temptation and stakes, not to different questions. The grid winners
move by at most 3.3 MB across the four benefit/harm slices — models are largely
insensitive to the stated stakes; the prompt's framing dominates.

**Items (transfers).** To rule out overfitting to particular items within the benchmark,
we split each metric's item pool into two disjoint random halves and re-scored every
configuration on each half. Agreement is near-perfect (HA and CP correlations ≥ 0.989,
MAE ≤ 2.8 across all four models): a prompt's operating point is a property of the
prompt, not of the item sample.

**Models (does not transfer).** Prompts do not port across models. Qwen's strong-safety
prompt drives Llama to HA 100 / CP 14.6 (it refuses almost everything), and Llama's
weak-safety prompt drives every other model to HA 3–13 / CP ≈ 98 (pure pragmatism).
Per-model tuning is required; the deliverable is the method and a per-model frontier, not
a single portable prompt.

## 6.4 Levels are not semantics (a negative result on interpretability)

The two knobs are interpretable in the sense that a human can read why a configuration
should behave as it does, and the ordinal levels order cleanly (§6.1). But the *strength*
of the interpretation is limited by a wording-sensitivity result that we consider
important enough to state prominently. We wrote three human paraphrases of each winner's
safety sentence, each intended to preserve the sentence's semantic level, changing only
wording and structure, and re-evaluated the winners. The operating point moved by whole
tiers on three of the four models: on Gemini a balanced 70/71 prompt became a near-total
refuser (97/6) under a "level-preserving" rewrite (CP −37 to −65 across the three
paraphrases); Mistral's HA moved −13 to −28; Qwen's CP up to −33; only Llama was stable
(within ±8). These shifts exceed the distance between adjacent safety *levels*, so this
is surface-form sensitivity, not level drift.

The consequence for the method's claims is that the knob abstraction indexes *specific
template strings*, and every guarantee in this thesis — the conformal intervals and
controller hits of Chapter 7, the replication win-or-tie of Chapter 5 — attaches to those
frozen strings, not to their semantic paraphrase class. As a standalone finding it is a
caution for practice: prompt-based safety steering is wording-fragile, and a deployment
must freeze and regression-test the exact prompt text rather than trust an "equivalent"
rewording. That one model (Llama) is markedly more robust than the others suggests
surface-sensitivity is itself a model-specific property worth future study.

## 6.5 Frontiers, hypervolume, and operating points

Collecting each model's measured configurations in the (HA, CP) plane yields a clear
empirical frontier. Treating deployment as constrained selection — maximise CP subject to
HA ≥ a required floor — turns the frontier into an actionable dial, and hypervolume
(reference (0,0), maximum 10,000) summarises frontier quality per model. On the
exhaustive grid the ordering is Qwen 8,320 > Llama 7,511 > Gemini 6,909 > Mistral 6,823.
Notably, Mistral's exhaustive-grid hypervolume (6,823) is far above the 5,255 obtained
from the hand-crafted spectrum alone: its "poor steerability" in earlier reporting was an
artifact of the spectrum never sampling its good region (moderate safety, light
pressure), exactly the region Chapter 5's grid winners occupy. This is a second concrete
payoff of exhaustive, unbiased measurement over sampled hand-tuning.
