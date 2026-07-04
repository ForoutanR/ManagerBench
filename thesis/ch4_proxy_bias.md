# Chapter 4 — The Proxy-Bias Result

*Draft. Source: COMPLETE_REPORT §16–18, proxy_selection_findings_v2.md, novelty_analyses.md.
Figures: F4 (LOPO MAE + CI). All numbers reproducible from the repository.*

Black-box prompt optimization is only affordable because candidates are scored on a
small subset of the benchmark rather than the whole. This chapter shows that the choice
of subset is not a harmless implementation detail: the intuitive "keep the most
informative items" rule is a *biased* estimator of the full-benchmark score, the bias
cannot be removed by calibration, and — as Chapter 5 will show — it can steer the entire
optimization to the wrong answer.

## 4.1 The symptom

The first version of our pipeline selected the 180 proxy items whose binary outcomes
varied most across prompt configurations ("discrimination" selection), on the standard
intuition that the most *informative* items should best track the benchmark. When the
optimizer's chosen configurations were validated on the full benchmark, their scores
came in systematically **below** their proxy scores — by 8 to 26 MB points, and by an
amount that differed per configuration. A join of proxy-predicted against full scores
gave a high Pearson correlation (r ≈ 0.95) but this was computed on only four points
(one per model); the same four points give a Spearman rank correlation of 1.0 on MB and
0.8 on HA. Correlation on n = 4 is not evidence, and — more importantly — the offline
study below shows the discrimination proxy is in fact the *worst* ranker of all the
strategies we tested, contradicting the very property it was chosen for.

## 4.2 The offline leave-one-prompt-out study

Because the pipeline had saved every model's per-item answer for 14 prompt
configurations across 4 models (56 configurations in total), we could measure the
quality of any selection strategy by *replay*, at zero API cost. The protocol is
leave-one-prompt-out, per model: to predict a held-out prompt's full-benchmark HA and
CP, the proxy items are selected using only the *other* prompts (so the held-out
configuration is never seen during selection), and the held-out prompt's score is then
estimated from those items. This is an out-of-sample estimate, not a fit.

A subtlety in the original analysis inflated the apparent advantage of representative
sampling: predictions were averaged over five random draws *before* the error was
computed. Five draws of 90 items from a pool of ~357 cover most of the pool, so the
"averaged" number describes a near-complete evaluation rather than a single deployable
proxy. We therefore report the honest **single-draw** error (the expected error of one
90 + 90 proxy, averaged over seeds), with the ensembled number kept only as a
diagnostic. Table 4.1 gives the corrected result.

**Table 4.1 — Single-draw LOPO error (180-item proxy, 56 configurations, 95% CI).**

| selection | MB MAE [95% CI] | HA MAE | CP MAE | MB r |
|---|---|---|---|---|
| variance / discrimination (original) | 12.9 [9.9, 16.0] | 11.0 | 8.5 | 0.81 |
| difficulty-stratified | 4.2 [3.3, 5.3] | 3.6 | 3.1 | 0.98 |
| random | 2.5 [2.0, 2.9] | 1.8 | 1.4 | 0.999 |

The discrimination proxy is the worst on every metric, *including ranking* (r 0.81 vs
0.999). The gap to representative sampling is roughly three-to-fivefold — not the "order
of magnitude" an earlier draft claimed once the unfair ensembling is removed. But the
raw ratio understates the real distinction, which is one of *kind*: the discrimination
proxy's ~13-point error is **bias** (it is identical under single-draw and five-draw
scoring — averaging does not reduce it), whereas the representative samplers' 2–4-point
error is **sampling noise** (random's error falls from 2.5 to 0.8 when five draws are
ensembled). Bias does not shrink with more measurement; noise does. This is why
representative sampling is the correct fix and no amount of re-running the biased proxy
would have helped.

## 4.3 Why informativeness biases the mean

The mechanism is a classical result from active testing (Kossen et al.). A binary
item's variance across configurations is maximised when its success probability is near
0.5, so a variance-maximising rule preferentially keeps *borderline* items. Borderline
items have outcome means near 50% almost by definition; restricting the subset to them
pulls the subset mean toward 50% and away from the true population mean. The size of the
distortion depends on how far the configuration's true score sits from 50%, which
differs per configuration — hence the bias is *configuration-conditional*, and a single
global correction cannot remove it. A representative sample, by contrast, keeps easy and
borderline items in their true proportions, so its mean is an unbiased estimate of the
population mean with only sampling noise.

## 4.4 The bias is a property of informativeness, not of one heuristic

Two further analyses (both zero-cost replay) sharpen the finding and connect it to
current practice.

**Selection-principle ablation.** Recent performance-guided selection methods for prompt
optimization (e.g. IPOMP) combine a *coverage* principle (spread the subset across the
space) with a *boundary* principle (prefer decision-boundary / high-uncertainty items).
Ablating the two principles in our harness (Table 4.2) shows that coverage is benign
while boundary is toxic: boundary selection alone reproduces the discrimination bias
(13.2 vs 12.9 MB MAE), and adding a boundary term to an otherwise unbiased
coverage-clustering selection triples its error (3.0 → 10.9). Coverage-clustering on its
own is not only unbiased but slightly better than our shipped stratified default.

**Table 4.2 — Selection-principle ablation (single-draw MB MAE).**

| principle | MB MAE |
|---|---|
| random | 2.5 |
| coverage-clustering | 3.0 |
| difficulty-stratified (shipped) | 4.2 |
| coverage + boundary | 10.9 |
| variance / discrimination | 12.9 |
| boundary alone | 13.2 |

The lesson is that the failure is intrinsic to *informativeness as a selection
objective for score estimation*, not to any particular heuristic. Methods that select
uncertain or boundary items — a common and otherwise reasonable choice for active
learning — inherit the bias whenever the selected subset is also used to *estimate* an
aggregate score.

**No calibration fix.** One might hope to correct the bias post hoc by regressing proxy
scores onto full scores. Fitting proxy = α · full + (1 − α) · 50 gives α ≈ 1.0 for every
strategy — the bias does not manifest as a global affine attenuation, because its offset
depends on the (unknown) true score of the configuration being predicted. Empirically, a
per-model affine calibration reduces the discrimination proxy's MB MAE only from 12.9 to
9.6, still three to four times worse than plain random sampling. The bias must be
prevented in selection, not repaired in scoring.

## 4.5 A principled estimator does not help at this scale (negative result)

For completeness we implemented the "correct" tinyBenchmarks/metabench approach: a
2-parameter Item Response Theory model fit to the configuration × item response matrix,
with anchor items chosen by quantile-stratified Fisher information and the held-out
configuration's score estimated from its latent ability. At our scale the IRT estimator
is *worse* than plain stratified or random sampling at every proxy size. The reason is
data scale: fitting hundreds of item parameters from only 56 configurations is
under-determined, producing noisy item parameters and hence a noisy ability estimate.
tinyBenchmarks-style estimators rely on hundreds of test-takers or a pre-fit item bank,
which we do not have. Below that data threshold, the mean of a representative sample is
hard to beat — consistent with the empirical finding of Perlitz et al. that a random
sample plus a simple estimator is a strong baseline.

## 4.6 Summary

Selecting benchmark items for informativeness biases the estimated aggregate score in a
configuration-dependent way that no calibration can undo; representative sampling of the
same budget is nearly unbiased; and a sophisticated IRT estimator does not help at our
data scale. The next chapter shows that this is not merely a reporting problem: the same
bias, sitting inside an optimization loop, changes which prompt the optimizer returns.
