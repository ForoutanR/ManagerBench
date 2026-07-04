# B.Eng. Thesis — Structure & Content Map

**Title (working):** *Steering the Safety–Pragmatism Trade-off of Autonomous LLMs by
Prompt Optimization: Why the Cheap Evaluation Is Part of the Optimizer*

**Author:** Foroutan Rahmani · **Institution:** BIHE, Computer Engineering ·
**Supervisor:** Shahrad Mohammadzadeh

> This outline maps every thesis section to the source material already written, so
> drafting is assembly + tightening, not new writing. Source keys:
> **CR** = `COMPLETE_REPORT.md` · **WP** = `paper/workshop_paper.md` ·
> **RJ** = `RESEARCH_JOURNAL.md` · findings md = `results/optimization/*.md`.
> Target length ~40–55 pp. Status per section: ☐ todo · ◑ draft-from-source · ✅ done.

---

## Front matter
- Title page, abstract (½ pp — adapt WP abstract, less compressed), acknowledgements,
  table of contents, list of figures/tables. ☐

## Chapter 1 — Introduction (~4 pp) ◑ from CR §1–3
- 1.1 Motivation: autonomous LLM agents, the harmful-but-effective action, two failure
  modes (over-pragmatism, over-caution). *(CR §1)*
- 1.2 Problem statement: place a fixed model at a chosen safety/pragmatism point by
  prompt alone; do it cheaply and with a guarantee. *(CR §2)*
- 1.3 Contributions (5): (i) prompt-as-optimization method on an interpretable space;
  (ii) the proxy-bias result — informativeness selection is biased *and mis-steers the
  search*; (iii) repaired pipeline that beats hand-crafting 3/4; (iv) per-model
  characterization (safety universal, pressure erodes, ROC lens); (v) a conformal
  controller + verified hit. *(CR §3, WP §1)*
- 1.4 Thesis roadmap.

## Chapter 2 — Background (~8 pp) ◑ from CR §4–9 (teaching-level, keep)
- 2.1 LLMs, system vs user prompts, autonomous decisions. *(CR §4)*
- 2.2 ManagerBench: items, HA/CP/MB, push/nudge, worked example. *(CR §5, §5.1a)*
- 2.3 Prompt design as optimization: the two knobs, template decode, the 45-cell
  discretization (state up front — it is load-bearing). *(CR §6, updated)*
- 2.4 Black-box / Bayesian optimization: GP surrogate, acquisition, TPE vs GP. *(CR §7)*
- 2.5 Multi-objective optimization: dominance, Pareto, hypervolume, ParEGO; the ROC
  lens (frontier = ROC, HV = AUC). *(CR §8 + §22 addendum)*
- 2.6 Cheap evaluation: subset proxies, when a subset predicts the whole, IRT, the
  extrapolation caution. *(CR §9)*
- 2.7 Conformal prediction (short primer — new, for the controller guarantees).

## Chapter 3 — Method (~6 pp) ◑ from CR §10–11, WP §2–3
- 3.1 Pipeline overview (the diagram). *(CR §10)*
- 3.2 Parameter space & decode; ordinal-levels framing; 45 cells. 
- 3.3 Proxy selection: stratified vs discrimination; the scoring convention
  (unparseable = wrong). 
- 3.4 Optimizer: numpy GP-BO (why not BoTorch), ParEGO. *(CR §11)*
- 3.5 Controller: inverse map over cells, conformal intervals, closed-loop verify.
- 3.6 Reproducibility & infra (VPS, OpenRouter, budget, DPI). *(CR §12)*

## Chapter 4 — The Proxy-Bias Result (~7 pp) ✅ core novelty, from CR §16–18, findings
- 4.1 The symptom: proxy over-estimates; the n=4 warning. *(CR §16)*
- 4.2 Offline LOPO study: fair single-draw table (variance 12.9 / stratified 4.2 /
  random 2.5), bias-vs-noise, CIs. *(proxy_selection_findings_v2.md)*
- 4.3 Mechanism: variance/boundary peak at p≈0.5; Active Testing tie-in. *(CR §9.1)*
- 4.4 Selection-principle ablation: boundary reproduces the bias, coverage does not;
  no affine fix. *(novelty_analyses.md B, C)*
- 4.5 IRT negative result. *(CR §18)*

## Chapter 5 — Search Damage and Repair (~6 pp) ✅ headline, from CR §14–15 + grid
- 5.1 The misled search: cold start, 45-cell coverage, decode identity of "optima",
  loss to hand-crafting 4/4. *(CR §14–15)*
- 5.2 The repair: stratified proxy + exhaustive 45-cell grid ($4.64). *(grid_findings)*
- 5.3 Validated winners: beat hand-crafting 3/4 (+3.9…+16.7); llama brackets optimum;
  live proxy↔full calibration. *(grid_findings §1–2, CR §15 addendum)*
- 5.4 Replay optimizer comparison: random strong at 45 cells, GP-BO wins narrow-region;
  INSTINCT descope justified. *(grid_findings §replay, replay_optimizers.json)*
- 5.5 "Proxy design flips the sign of the result" — the thesis's one-sentence claim.

## Chapter 6 — Characterization (~6 pp) ◑ from CR §19–22 + rigor
- 6.1 Knob sensitivity: safety universal (+12/step), pressure erodes; dose–response.
  *(CR §19)*
- 6.2 ROC/threshold lens: steerability = discrimination quality. *(CR §22 addendum)*
- 6.3 Transfer: stakes-wording insensitive, item-split stable, cross-model NO. *(CR §20–21)*
- 6.4 Wording fragility (the negative result): levels ≠ semantics; freeze templates.
  *(rigor_findings §2)*
- 6.5 Frontiers, hypervolume, constrained operating points; mistral re-graded. *(CR §22)*

## Chapter 7 — The Controller (~4 pp) ◑ from CR §23 + conformal
- 7.1 Inverse map & feasibility flag. *(CR §23)*
- 7.2 Conformal intervals: construction, per-model widths, 82% coverage. *(grid_findings §3)*
- 7.3 Closed-loop verified hit (2.1 error, 14 evals). *(CR §23 addendum)*
- 7.4 Demo CLI (`demo_cli.py`) — the deployment artifact; 10% risk margin.
- 7.5 External validity: XSTest + HarmBench results. *(results/external/ — pending batch B)*

## Chapter 8 — Related Work (~3 pp) ◑ from WP §7 + CR §31 (trim)
Subset eval, prompt-opt/BO, multi-objective, safety–pragmatism frontiers, conformal.

## Chapter 9 — Discussion, Limitations, Future Work (~4 pp) ◑ from CR §31b–33, WP §8
- Limitations: 45-cell scope, wording fragility, cold-start proxy, n=11 conformal,
  4 models/1 benchmark. Future: scaled MOO (qNEHVI + proxy + conformal + top-k),
  learned/larger spaces, external agentic-safety transfer.

## Chapter 10 — Conclusion (~1 pp) ◑ from CR §34 / WP conclusion.

## Appendices
- A. Proposal alignment & deviations (descopes). *(proposal_alignment.md, SUPERVISOR_REPORT)*
- B. Full tables (LOPO, grid, rigor, stakes, external). 
- C. Reproducibility: seeds, cost ledger, artifact index, commands. *(CR App A–B)*
- D. Glossary. *(CR App C)*
- E. Chronological research journal (the failures). *(RJ)*

---

## Drafting order (fastest path to a complete draft)
1. Ch 4 + Ch 5 first — the novel core, fully supported, mostly assembly. 
2. Ch 6 + Ch 7 — characterization + controller (wait on batch B for §7.5).
3. Ch 2 + Ch 3 — background/method (teaching content already in CR).
4. Ch 1, 8, 9, 10 — frame around the finished core.
5. Appendices + front matter last.

## Open dependencies
- §7.5 external validity → batch B (running, Opus).
- Figures → Opus T9/T11 (F9/F3 done; F6/F7/F10 in progress).
- Appendix A descopes → supervisor sign-off (pending meeting).
