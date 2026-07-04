# Chapter 9 — Discussion, Limitations, and Future Work

*Draft. Source: COMPLETE_REPORT §31b–33, workshop_paper §8, novelty_analyses.md.*

## 9.1 What the results mean

The through-line of this thesis is a single, somewhat uncomfortable claim: for
safety-relevant prompt optimisation, the cheap evaluation subset is part of the
optimiser, not a neutral measurement of it. The same benchmark, the same class of
optimiser, and the same order of budget produce opposite conclusions depending only on
how the subset is chosen. The failure is not exotic — it comes from the most natural
selection rule in the field (keep the informative items) and it recurs in the
decision-boundary component of current performance-guided methods. Because the objective
here is precisely a model's harm behaviour, a silent bias in the evaluation is not a
leaderboard nuisance; it is a safety-relevant defect that can certify the wrong prompt.

The positive results are equally a story about measurement. Once the evaluation is
representative and, at this scale, exhaustive, the same pipeline finds prompts that beat
careful hand-writing on three of four models, in a region hand-tuning never explored; a
model that looked "barely steerable" turns out to have been under-sampled; and the map is
regular enough to invert into a controller with a distribution-free guarantee. None of
this required a more sophisticated optimiser — a replay comparison shows random search is
competitive in a 45-cell space — only a trustworthy proxy.

## 9.2 Limitations

- **Scope.** Four mid-sized hosted models, one benchmark, and one decode template.
  Findings are stated as being about the *method* and *per-model* behaviour, not about a
  universal prompt; cross-model non-transfer is itself one of the results.
- **A small, discrete space.** The 45-cell decode makes exhaustive measurement possible
  but also makes the search problem easy, so the optimiser-efficiency claims are modest by
  construction; larger or learned prompt spaces re-open the search question that our replay
  closes here.
- **Wording fragility.** The interpretable knobs index specific template strings, not
  their semantic paraphrase class: level-preserving rewrites shift the operating point by
  whole tiers on three of four models. Every guarantee in the thesis attaches to the
  frozen strings. This both limits the interpretability claim and stands as a caution for
  practice.
- **Cold-start proxy.** Representative (stratified or coverage) selection needs per-item
  difficulty estimates, i.e. some prior full-benchmark responses. A brand-new model has no
  such history, so the proxy must be bootstrapped at real cost before it is cheap.
- **Statistical caution.** The conformal calibration uses only 11 cells per model, forcing
  conservative intervals that mid-frontier configurations occasionally still exceed; the
  measurement noise floor (identical prompts re-measuring within ΔHA ≈ 4.7) bounds all
  single-point claims; and the Llama result shows that a single proxy-argmax pick can miss
  a near-tied optimum, so top-k validation is needed.
- **External validity.** The operating points are defined on ManagerBench; the external
  XSTest/HarmBench checks (Chapter 7.5) test only the selected prompts, and the HarmBench
  scoring uses an LLM judge rather than the official classifier.

## 9.3 Future work

The natural next step follows directly from the classifier/threshold lens of Chapter 6.
If a prompt selects a threshold on a fixed harm-discrimination curve, then the way to
improve deployments is to (i) enlarge the prompt space along axes the two knobs cannot
express — tone, role, output format, and, given §6.4, paraphrase variants treated as an
explicit robustness dimension rather than noise — and (ii) trace the resulting,
no-longer-enumerable frontier with genuine multi-objective Bayesian optimisation. The
right tool there is a hypervolume-greedy, noise-aware acquisition such as qNEHVI (with
observation noise set to the measured proxy variance), falling back to random-scalarisation
ParEGO where dependencies must stay light; the stratified proxy, top-k full-benchmark
validation, and conformal wrapping carry over unchanged. Two further directions are worth
naming: testing whether prompt-induced threshold shifts transfer to an external agentic
benchmark (AgentHarm, or an operational-pressure suite), which would upgrade the external
validity of Chapter 7 from a spot-check to a claim; and a preference-conditioned
controller that amortises the inverse map into a single learned function from requested
trade-off to prompt — of which the present non-parametric controller is the tabular
special case. Finally, the cold-start limitation invites a multi-fidelity bootstrap:
spend a small full-benchmark budget to seed the difficulty estimates a representative
proxy needs, then run cheaply thereafter.

## 9.4 A note on the proposal

The approved proposal framed the task as constrained optimisation with a neural-bandit
optimiser, an external harm benchmark as the constraint, and a local white-box model. The
core — budgeted black-box prompt optimisation on ManagerBench with Pareto reporting and a
selected operating point — is delivered, and its budgeted-evaluation component became the
main contribution. Three deviations are argued from evidence rather than convenience: the
neural bandit is unjustifiable in a 45-cell space where random search is competitive (the
comparison is delivered by replay); the internal HA metric plays the constraint role more
faithfully than an external benchmark, since ManagerBench items are genuine dilemmas
without a single correct answer, with an optional external HarmBench spot-check retained;
and no GPU was available for a local white-box model, so all experiments are black-box via
API, which is also the realistic deployment setting. Appendix A records the full mapping.
