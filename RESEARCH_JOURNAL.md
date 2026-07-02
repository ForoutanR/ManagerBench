# Research Journal — Approaches Tried, Failed, and Replaced

> Chronological log of every approach we tried, whether it worked, *why* it failed,
> and what replaced it. Source material for the Methods / Limitations / "lessons"
> sections of the report. Newest entries at the bottom of each section.
> Convention: ❌ failed/rejected · ⚠️ partial/caveat · ✅ adopted.

---

## A. Proxy / cheap-evaluation design

### A1. Variance / discrimination item selection ❌ REJECTED
- **Tried:** pick the ~180 items whose binary outcome varies most across prompt
  variants/models (variance = "discrimination"); `compute_item_discrimination` +
  `select_proxy_items` in `prompt_optimizer/proxy.py`.
- **Rationale:** "most informative items best distinguish prompt configs."
- **Failed because:** it is a **biased estimator** of the full-benchmark score.
  Binary-outcome variance peaks at p≈0.5, so selection over-keeps borderline items
  whose mean ≠ population mean. Caused Task 2c full-bench scores to come in 8–26
  points *below* proxy (gemini −20, mistral −26); Task 2d MB MAE ≈ 15, HA MAE ≈ 28.
- **Evidence:** offline leave-one-prompt-out over 56 configs — v1 worst on *every*
  metric incl. ranking (MB r 0.81). p1 (arXiv 2604.08801) confirms the mechanism:
  raw-variance selection "biased toward p=0.5"; SNR selection biased to extremes.
- **Replaced by:** A2.
- **Docs:** `results/optimization/proxy_selection_findings.md`.

### A2. Stratified / random sampling ✅ ADOPTED (current)
- **Tried:** difficulty-stratified (bin by mean correctness, sample evenly) and pure
  random sampling of the same 180-item budget.
- **Result:** unbiased. Offline LOPO: random MB MAE **0.82** (r 0.999), stratified
  **3.0** (r 0.99), HA/CP MAE 2–3 — vs v1's 11/8.5. Even 60 items → MAE ~4.9.
- **Shipped:** `proxy.py` now defaults `method="stratified"` (also `"random"`,
  legacy `"discrimination"` kept as ablation baseline).

### A3. IRT proxy (2PL + Fisher-info anchors + ability estimator) ❌ TRIED, did NOT beat stratified
- **Tried:** `prompt_optimizer/irt_proxy.py` — fit 2PL IRT on the 56-config×item
  matrix (per axis), select quantile-stratified max-Fisher-info anchors, estimate a
  held-out config's full HA/CP from latent ability (tinyBenchmarks/metabench style).
- **Result (offline LOPO):** IRT WORSE than the simple samplers at every size —
  HA/CP MAE: IRT 2.75/1.83, stratified 1.10/0.54, random 1.01/0.63 (180 items);
  gap widens as anchors shrink.
- **Why:** with only ~56 configs, joint 2PL MLE over ~357 items × 2 params is
  under-determined → noisy item params → noisy ability → worse than averaging a
  representative subset. tinyBenchmarks/metabench rely on hundreds of models / a
  pre-fit item bank; we don't have that scale. Consistent with "Misses the Mark"
  (2506.07673): random-sample mean is hard to beat.
- **Decision:** keep **stratified/random** (A2) as the shipped proxy; report IRT as a
  negative result ("principled IRT does not help at this config scale").
- **Search-proxy (resampled minibatch + periodic full re-eval):** still planned, only
  matters once we run live BO again.

---

## B. Optimizer / surrogate

### B1. Optuna TPE ⚠️ WORKS, being upgraded
- **Used:** Optuna TPE, 25 trials/model, single + multi-objective, SQLite-checkpointed.
- **Caveat:** TPE is a density-ratio heuristic; no calibrated posterior, no explicit
  noise model — weak for our noisy proxy objective and our low-dim continuous space.
- **Planned replacement:** B2.

### B2. GP-based Bayesian optimization ✅ IMPLEMENTED (numpy)
- **Goal (converged across ReElicit, LLM-BO, MIPRO):** GP surrogate (Matérn-5/2 +
  ARD) over the 2D box, **explicit observation-noise term = measured proxy
  variance**, acquisition qLogNoisyEI or UCB(annealed κ 2.0→0.5).
- **BoTorch path ❌ NOT USED:** BoTorch needs torch (~1GB). VPS is 1 core / ~1GB RAM
  and was running the sweep — torch install risked OOM and competed for the single
  core. Decided not worth it for a 2D / ~30-point problem.
- **Adopted ✅:** a **numpy-only GP** — `prompt_optimizer/gp_bo.py`. Matérn-5/2 + ARD,
  hyperparameters by log-marginal-likelihood (derivative-free coordinate ascent, no
  scipy), UCB(annealed κ)/EI, explicit `noise` term. Offline self-test PASSES:
  single-obj finds a synthetic optimum to 0.049; ParEGO recovers a 37-point Pareto
  front (full spread).
- **Not yet:** live run against the real proxy evaluator (needs API budget; deferred
  while the sweep uses budget).

---

## C. Objective

### C1. Maximize MB (harmonic mean) ⚠️ DEMOTED to summary metric
- **Was:** optimize the single MB scalar.
- **Why changed (user reframe 2026-06-27 + all papers):** goal is to study the
  *effect* of prompt optimization on the safety↔pragmatism trade-off, not to win a
  number. MB collapses HA and CP and hides the trade-off.
- **Now:** report **HA and CP separately**; MB kept only as a convenience summary.
- **Pareto ✅ IMPLEMENTED:** HA–CP frontier traced directly via **ParEGO** (random
  Chebyshev scalarization per round) in `gp_bo.py:optimize_multi` + a
  non-dominated-filter (`pareto_front`). Chose ParEGO over qNEHVI because qNEHVI lives
  in BoTorch (torch, see B2); ParEGO needs only the numpy GP. Self-test PASSES.

---

## D. Compute / infrastructure

### D1. Kubernetes pod (prod, delivery ns) ❌ ABANDONED
- **Tried:** Ubuntu pod (4Gi/1core) to run experiments.
- **Failed because:** egress proxy throttled Ubuntu mirrors → `apt-get` hung ~10 min,
  git install / clone never completed (background task exit 1). Network to
  github/openrouter was fine; the Ubuntu-mirror egress was the bottleneck.
- **Replaced by:** D2.

### D2. Germany VPS (104.156.154.136, Ubuntu 24.04, 1 core, ~1GB RAM) ✅ ADOPTED
- Setup clean: git, python3.12, venv, deps, repo @ feat/Add-spectrum-Expriements.
- **Caveat ⚠️:** SSH from Iran hits **DPI that intermittently drops the SSH banner**
  mid-handshake ("Connection closed during banner exchange"). Mitigation: retry loops
  / VPN; keep remote commands short (long compound commands get cut off).

### D3. Remote budget-watchdog daemon ❌ ABANDONED
- **Tried:** `budget_guard.sh` polling OpenRouter usage, auto-kill sweep at $ ceiling.
- **Failed because:** could not be launched reliably over the DPI-throttled SSH —
  nohup/setsid launch commands kept getting cut off before the daemon detached;
  long-lived ssh sessions hung to the 2-min tool timeout.
- **Replaced by:** manual spend monitoring. Justified: measured rate ~$0.0076/min
  (~$0.46/hr), so ~10h of headroom to the $7 ceiling — no risk of a sudden jump
  between manual checks; key has a hard $30 cap as backstop.

### D4. Cost estimation ⚠️ CORRECTED
- **First estimate:** ~$0.0001/prompt → "lean run ~$0.25". Wrong (too low).
- **Corrected:** each item issues ~2 generations (push + control); output is tiny
  (~3 tokens, "My answer is X") so cost is **input-dominated** (long scenarios).
  Measured live ~$0.000084/prompt; full 4×4 slice ≈ $1.6; 3-slice sweep ≈ $4.8.

---

## E. Validation / experiment scope

### E1. Task 2c full-bench validation @ benefit10/harm5 (4×4) ✅ DONE
- Optimized beats handcrafted on 2/4 models: Llama 27→67.5 (+40), Qwen 70→71.3 (+1);
  loses Gemini (67→54), Mistral (39→37). Cross-model prompts do **not** transfer.

### E3. Proposal alignment — adopted only $0 paper-strengthening pieces ✅
- Analyzed `Final_proposal (1).pdf` (constrained system-prompt opt; HarmBench, INSTINCT
  neural bandit, white-box/black-box). Analysis: `results/optimization/proposal_alignment.md`.
- **Decision (user):** proposal is a means, not the goal — do only parts that help the
  main paper and cost $0; leave the rest.
- **Adopted ($0, on existing data, `constrained_analysis.py`):** constrained operating
  points (max CP s.t. HA≥thr) + threshold sweep, **Pareto hypervolume** per model
  (Qwen 7,979 > Llama 7,658 > Gemini 6,024 > Mistral 5,255), weighted-sum selection.
  Added to paper §5.6 — quantifies frontier quality + model-dependent *steerability*
  (Qwen holds CP=92.5 at HA≥50; Mistral collapses to 25.3).
- **Skipped (cost/build, not central):** HarmBench external safety, INSTINCT neural
  bandit, white-box/local model, XSTest/scope (FR/O).

### E2. benefit/harm difficulty sweep (slices 10/15, 50/5, 50/15, full 4×4) ✅ DONE
- **Result: optimized prompts GENERALIZE across difficulty.** Source-pair HA/CP are
  near-invariant across all 4 slices (per-model ΔHA ≈ ±2-5, ΔCP ≈ ±2). Mean across
  models per slice: HA 42-46, CP 90-92, MB 56-59 — essentially flat.
- **Interpretation:** a prompt config sets a stable operating point on the
  safety-pragmatism frontier; benefit/harm *difficulty* (item-selection filters)
  barely shifts it. So prompt-optimization effects transfer across difficulty —
  the opposite of cross-model transfer (2c), which fails.
- **Cost overrun ⚠️:** ran to **$7.82** vs approved $7 (+$0.82). Watchdog (D3) was
  dead and manual monitoring lapsed over the ~11h run; it stopped only on completion,
  not a cap. Key hard-cap $30 was the real backstop. Lesson: never rely on manual
  monitoring for long unattended runs — need a working in-process budget kill.
- Analysis: `analyze_sweep.py`; per-slice data `results/variants/optimized-v*/comparison_results_<b>_<h>.json`.

---

## G. 2026-07-02 full audit (Fable 5) — corrections to the record

### G1. Task 2c claim ❌ REVERSED
- Baseline mislabeled: "best hand-crafted" was safe-prefix = best *cross-model mean*,
  near-worst for Llama (MB 26.7). Vs per-model best hand-crafted (same slice/scoring):
  optimized loses ALL 4 (qwen −5.2, llama −10.5, gemini −12.2, mistral −20.2).
- Decode identity (verified `generate_prompt_config` equality): v1-qwen ≡ safe-prefix,
  v3-gemini ≡ balanced-safe, v4-mistral ≡ baseline-no-push. Only v2-llama (empty prefix
  + medium nudge) is a new prompt. Same-prompt re-measures: ΔHA ≤4.7, ΔCP ≤5.6 = noise floor.
- Warm start NEVER RAN: `n_warm_start: 0` all 4 optimization JSONs (cold TPE seed 42,
  identical first trials across models). Report/paper said "warm-started" — false.
- 25 trials covered 16–18 of 45 decode cells; llama/qwen true-best cells never visited;
  mistral's true-best cell visited but mis-ranked by biased proxy (41.8–44.3 vs 62.9 for
  the empty prompt; true MBs 57.3 vs 37.1). Biased proxy STEERED the search below
  hand-crafted baselines — the strongest form of the proxy-bias result.

### G2. Param space is 45 discrete cells ⚠️ REFRAME
- `param_space.py` decode = step function: 9 safety bins × 5 goal bins = 45 unique
  prompts. Continuous coords inside a cell are the same prompt. Exhaustive proxy grid
  ≈ $4.5 (90-item)/$9 (180-item) for 4 models; BO value then shown by replay.

### G3. Proxy LOPO scoring ⚠️ CORRECTED
- Published table ensembled 5 random/stratified draws before scoring (≈ full pool) while
  variance ran single-shot. Fair single-draw (10 variants local): variance 10.2 /
  stratified 3.0 / random 2.0 MB-MAE → gap ~3–5×, not "order of magnitude". Bias-vs-noise
  distinction is the correct framing. Script also had wrong ROOT (crashed; loaded 0
  variants) and the 4 optimized variants' raw files are VPS-only (published 14-config
  numbers not repo-reproducible).
- Second bias source found: ProxyEvaluator excluded unparseable answers from the
  denominator; full bench counts them wrong. Fixed (OPUS_TASKS T2).
- `bench_proxy/proxy_items.json` was still discrimination_variance at audit time — the
  stratified "fix" changed the code default only, never the artifact (T3).

### G4. Sensitivity claim ❌→✅ REPLACED
- gp→CP "model-specific / Mistral inert" = artifact (biased proxy + TPE confounding;
  sign flips on full bench). Replaced by controlled sw=0 dose-response: pressure erodes
  HA everywhere (llama −52 … gemini −7.5), CP gain only where headroom exists (llama
  81.4→97.2; others at 96–99 ceiling). sw→HA universal (+0.87…+0.89 full bench) — survives.

### G5. Difficulty sweep ⚠️ REFRAMED
- benefit/harm % are TEXT substitutions into the same 357/253 items (identical slice
  denominators), not item filters. Sweep = stakes-wording insensitivity, not item
  generalization; slices are correlated re-measurements.

### G6. Controller demo miss ✅ EXPLAINED (bug, not noise)
- Offline controller returned sw=0.613 for qwen (CP90/HA55) → crosses 0.60 template
  boundary → decodes to safety-first tier (measured 86/68.8) while surrogate predicted
  54.3/89.9; live verify measured ≈84/53. Fix: predict/invert over 45 cells; cell-
  neighbor search in the closed loop (OPUS_TASKS T4–T6).

### G7. Handoffs
- Corrections applied: COMPLETE_REPORT (§5.2, 14–17, 19, 20, 22, 23, 23a, 24, 28, 31b,
  32, 34), paper draft (abstract, §4, 5.2–5.7, 6, 7). Full audit: CRITICAL_ASSESSMENT.md.
- Mechanical/infra work specced for Opus 4.8: OPUS_TASKS.md (T1–T10). API grid package
  (~$8.4) awaits explicit user cost approval.

### G8. Opus 4.8 execution of OPUS_TASKS (2026-07-02)
- T1–T8 done, $0, no API. venv rebuilt (T10.1; numpy 2.0.2). Commits: proxy-lab LOPO
  repair (single_draw vs ensemble5 + parse-convention ablation + bootstrap CIs);
  evaluator counts unparseable as wrong (+n_unparsed); proxy_items.json regenerated
  stratified 90+90 (old kept as *_discrimination.json ablation baseline); param_space
  cell utilities (cell_of/cell_center/enumerate_cells, 45 cells); cell-space controller
  v2; controller_verify cell-neighbor search + --dry_run; run_grid.py (build only);
  compare_proxy_vs_full per-slice + per-model-best.
- Verified numbers: LOPO single_draw MB-MAE variance 10.22 / stratified 2.95 /
  random 2.03. Per-model best hand-crafted MB 76.4/78.0/66.4/57.3 — optimized loses
  on all 4 (2c reversal holds).
- **T10.2 VPS raw-file pull: initially FAILED, then SUCCEEDED same day.** First 5 ssh
  attempts all `Connection timed out during banner exchange` (Iran DPI banner-drop);
  when DPI lifted, pulled all 16 files (4 optimized variants × 4 models) via streamed
  `ssh … tar cf - | tar xf -` (scp remote-glob escaping fought us; tar was robust).
  All 16 parse OK; optimized-v1 qwen raw recompute (HA 58.3 / CP 91.7 / MB 71.3)
  matches its comparison_results_10_5.json exactly.
- **LOPO rerun at 14 variants/model** (was 10): single_draw MB-MAE variance 12.91 /
  stratified 4.20 / random 2.45 (ensemble5 random 0.82). Direction unchanged
  (variance ≫ stratified > random); magnitudes match the audit's 14-config figures.
  proxy_selection_findings_v2.{md,json} regenerated.
- T9 (figures) gated on Fable's doc edits (COMPLETE_REPORT §19) + user go-ahead; not run.
