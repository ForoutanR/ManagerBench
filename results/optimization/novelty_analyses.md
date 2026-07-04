# Novelty analyses B, C, D (replay, $0) — 2026-07-02

_Data: 56 configs (14/model × 4 models), b10/h5 raw replay, unparseable=wrong,
90+90 proxy, single-draw LOPO scoring (same harness as `proxy_selection_findings_v2`)._

## B. Selection-principle ablation — which ingredient of "smart" selection is toxic?

Motivated by IPOMP (arXiv 2505.10736), which selects prompt-optimization eval subsets
via semantic clustering + **boundary analysis** + performance-guided refinement. We
ablate the two principles in our LOPO harness:

| selection principle | MB MAE | HA MAE | CP MAE |
|---|---|---|---|
| random | 2.46 | 1.77 | 1.33 |
| coverage-cluster (k-means on config-signatures, centroid-nearest) | **2.99** | 2.41 | 1.70 |
| difficulty-stratified (shipped default) | 4.20 | 3.56 | 3.06 |
| coverage + boundary (IPOMP-style combo) | **10.94** | 8.56 | 7.02 |
| variance / discrimination (original bug) | 12.91 | 10.98 | 8.50 |
| boundary alone (|mean−0.5| minimizing) | **13.23** | 10.80 | 8.92 |

**Findings.**
1. **The boundary principle reproduces the discrimination bias** (13.2 ≈ 12.9): items
   nearest the decision boundary are the borderline items whose mean diverges from the
   population mean. Performance-guided pipelines that include a boundary/uncertainty
   term inherit the bias — even when combined with good coverage selection
   (coverage+boundary: 10.9 vs coverage alone: 3.0).
2. **Coverage-clustering alone is unbiased and beats our shipped stratified default**
   (3.0 vs 4.2 MB MAE) — consistent with Anchor-Points (2309.08638). Candidate future
   default; same data requirement as stratified (needs multi-config history).
3. Caveat: this ablates the *principles*, not the full IPOMP implementation (their
   semantic clustering uses text embeddings; ours clusters correctness signatures;
   their refinement is iterative/online). State as "IPOMP-style boundary component",
   not "IPOMP".

## C. Is the bias a global shrinkage law? — No (negative result, useful)

Hypothesis tested: proxy ≈ α·full + (1−α)·50 with α≪1 for informativeness-selected
subsets. **Rejected:** OLS slope α ≈ 0.97–1.02 for every strategy (CP axis mildly
expansive, 1.14, for boundary/variance). The bias does not manifest as global affine
attenuation — it is **config-conditional** (large residuals at slope ≈ 1; the offset
depends on where each config's item-level p sits relative to the selected borderline
set, and its sign varies by config).

**Consequence (the useful part):** no global affine calibration can repair a
boundary/variance-selected proxy — confirmed empirically by the calibration ablation
(variance 12.9 → 9.6 with per-model affine fit; still 3–4× worse than plain random).
The fix must be in *selection*, not post-hoc score correction. This strengthens the
paper's central recommendation.

## D. Held-out item split — operating points generalize across items

Per config, HA/CP computed on two disjoint random halves of the item pools:

| model | HA r | HA MAE | CP r | CP MAE | n |
|---|---|---|---|---|---|
| gemini-2.5-flash-lite | 0.998 | 1.5 | 0.994 | 2.3 | 14 |
| qwen3-32b | 0.995 | 2.0 | 0.989 | 1.6 | 14 |
| llama-3.3-70b | 0.995 | 2.8 | 0.997 | 2.1 | 14 |
| mistral-small | 0.998 | 1.6 | 0.998 | 1.8 | 14 |

A prompt's operating point is a stable property across disjoint item sets (r ≥ 0.989,
MAE ≤ 2.8) — the stakes-sweep's "same items" caveat does not hide item-level
overfitting. (External-benchmark generalization remains future work.)

## Status
- B/C/D land in the workshop paper: B as a results subsection (new ablation), C as one
  paragraph in the proxy section (calibration cannot fix it), D as one line + appendix
  table in the transfer section.
- Script: scratchpad `novelty_analyses.py`; promote into repo if reused.
- Pending same-week: A = conformal intervals for the cell controller on the 45-cell
  grid (blocked on grid completion).
