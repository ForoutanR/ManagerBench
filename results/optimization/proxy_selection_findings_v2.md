# Proxy-selection LOPO study (v2)
_Repaired per CRITICAL_ASSESSMENT §1.1 / OPUS_TASKS T1. Offline replay, $0. Does not overwrite the original `proxy_selection_findings.md`._
**Setup.** Leave-one-prompt-out, per model, over 40 full-bench configs (variants/model: gemini-2.5-flash-lite=10, qwen3-32b=10, llama-3.3-70b-instruct=10, mistral-small-3.2-24b-instruct=10). Proxy size = 90 HA + 90 CP items. Seeds [0, 1, 2, 3, 4]. Bootstrap 2000 reps for 95% CIs.
- **single_draw** = expected error of ONE 90+90 subset (per-seed error, averaged over seeds). This is the deployable number.
- **ensemble5** = average 5 seeds' predictions *before* scoring — diagnostic only; ~5×90 draws from a ~357-item pool approach the whole pool, flattering stochastic strategies. Reported to expose the original table's optimism.

## Main table (convention = count_wrong; matches full-bench denominator)
| strategy | MB-MAE single_draw [95% CI] | MB-MAE ensemble5 [95% CI] | MB_r | HA-MAE | CP-MAE |
|---|---|---|---|---|---|
| random | 2.03 [1.54, 2.55] | 0.71 [0.49, 0.99] | 0.999 | 1.35 | 1.13 |
| variance | 10.22 [7.27, 13.48] | 10.22 [7.27, 13.48] | 0.875 | 8.33 | 6.17 |
| stratified | 2.95 [2.15, 3.81] | 2.53 [1.76, 3.40] | 0.991 | 1.97 | 2.14 |

Variance error is **bias** (irreducible by averaging — single_draw≈ensemble5); random/stratified error is **sampling noise** (reducible — ensemble5 << single_draw).

## Parse-convention ablation (single_draw MB-MAE [95% CI])
| strategy | count_wrong (full-bench) | exclude (ProxyEvaluator) |
|---|---|---|
| random | 2.03 [1.54, 2.55] | 2.05 [1.56, 2.58] |
| variance | 10.22 [7.27, 13.48] | 10.31 [7.31, 13.61] |
| stratified | 2.95 [2.15, 3.81] | 2.96 [2.15, 3.82] |

## + affine calibration (count_wrong, single_draw MB-MAE)
| strategy | MB-MAE [95% CI] |
|---|---|
| variance+calibration | 8.95 [5.71, 12.38] |
| stratified+calibration | 2.65 [1.97, 3.45] |

## Size sweep (stratified, count_wrong, single_draw MB-MAE)
| size (HA+CP) | MB-MAE [95% CI] |
|---|---|
| 30+30 | 5.54 [4.13, 7.11] |
| 60+60 | 3.64 [2.60, 4.76] |
| 90+90 | 2.95 [2.15, 3.81] |
| 120+120 | 2.46 [1.81, 3.18] |
