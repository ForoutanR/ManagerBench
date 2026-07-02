# Proxy-selection LOPO study (v2)
_Repaired per CRITICAL_ASSESSMENT §1.1 / OPUS_TASKS T1. Offline replay, $0. Does not overwrite the original `proxy_selection_findings.md`._
**Setup.** Leave-one-prompt-out, per model, over 56 full-bench configs (variants/model: gemini-2.5-flash-lite=14, qwen3-32b=14, llama-3.3-70b-instruct=14, mistral-small-3.2-24b-instruct=14). Proxy size = 90 HA + 90 CP items. Seeds [0, 1, 2, 3, 4]. Bootstrap 2000 reps for 95% CIs.
- **single_draw** = expected error of ONE 90+90 subset (per-seed error, averaged over seeds). This is the deployable number.
- **ensemble5** = average 5 seeds' predictions *before* scoring — diagnostic only; ~5×90 draws from a ~357-item pool approach the whole pool, flattering stochastic strategies. Reported to expose the original table's optimism.

## Main table (convention = count_wrong; matches full-bench denominator)
| strategy | MB-MAE single_draw [95% CI] | MB-MAE ensemble5 [95% CI] | MB_r | HA-MAE | CP-MAE |
|---|---|---|---|---|---|
| random | 2.45 [2.02, 2.89] | 0.82 [0.62, 1.04] | 0.999 | 1.76 | 1.35 |
| variance | 12.91 [9.87, 16.00] | 12.91 [9.87, 16.00] | 0.813 | 10.98 | 8.50 |
| stratified | 4.20 [3.25, 5.25] | 3.86 [2.88, 4.93] | 0.980 | 3.56 | 3.06 |

Variance error is **bias** (irreducible by averaging — single_draw≈ensemble5); random/stratified error is **sampling noise** (reducible — ensemble5 << single_draw).

## Parse-convention ablation (single_draw MB-MAE [95% CI])
| strategy | count_wrong (full-bench) | exclude (ProxyEvaluator) |
|---|---|---|
| random | 2.45 [2.02, 2.89] | 2.46 [2.03, 2.91] |
| variance | 12.91 [9.87, 16.00] | 12.97 [9.93, 16.07] |
| stratified | 4.20 [3.25, 5.25] | 4.22 [3.27, 5.28] |

## + affine calibration (count_wrong, single_draw MB-MAE)
| strategy | MB-MAE [95% CI] |
|---|---|
| variance+calibration | 9.61 [6.72, 12.78] |
| stratified+calibration | 3.48 [2.78, 4.25] |

## Size sweep (stratified, count_wrong, single_draw MB-MAE)
| size (HA+CP) | MB-MAE [95% CI] |
|---|---|
| 30+30 | 6.54 [5.14, 8.09] |
| 60+60 | 5.21 [4.16, 6.38] |
| 90+90 | 4.20 [3.25, 5.25] |
| 120+120 | 3.49 [2.71, 4.40] |
