# Proxy vs Full-Benchmark Validation (Task 2d, corrected)

**Variants validated:** 4/4. Full-bench source per variant: 10_5 (per-slice `10_5` preferred; `aggregate` = 4-slice fallback).


## Per-variant proxy vs full (optimized prompts)

| Variant | Model | Proxy MB | Full MB | ΔMB | Full HA | Full CP | src |
|---|---|---|---|---|---|---|---|
| optimized_v1_qwen3-32b | qwen3-32b | 79.5 | 71.3 | -8.3 | 58.3 | 91.7 | 10_5 |
| optimized_v2_llama-3.3-70b-instruct | llama-3.3-70b-instruct | 75.0 | 67.5 | -7.6 | 54.3 | 88.9 | 10_5 |
| optimized_v3_gemini-2.5-flash-lite | gemini-2.5-flash-lite | 74.1 | 54.2 | -19.9 | 39.2 | 87.7 | 10_5 |
| optimized_v4_mistral-small-3.2-24b-instruct | mistral-small-3.2-24b-instruct | 62.9 | 37.1 | -25.8 | 23.0 | 96.8 | 10_5 |

## Proxy → full transfer

- **MB**: Pearson r = 0.954, MAE = 15.38  (**n=4 models** — statistically indicative only, not a powered estimate)
- **HA**: Pearson r = 0.969, MAE = 28.39  (**n=4 models** — statistically indicative only, not a powered estimate)
- **CP**: Pearson r = 0.840, MAE = 13.77  (**n=4 models** — statistically indicative only, not a powered estimate)

## Optimized vs hand-crafted (per-slice b10/h5, raw recompute, unparseable=wrong)

| Model | safe-prefix baseline (best cross-model mean) | per-model best hand-crafted | best optimized full MB | beats safe-prefix? | beats per-model best? |
|---|---|---|---|---|---|
| qwen3-32b | 68.2 | 76.4 (spectrum-safety-constrained) | 71.3 | yes | **no** |
| llama-3.3-70b-instruct | 26.7 | 78.0 (baseline-no-push) | 67.5 | yes | **no** |
| gemini-2.5-flash-lite | 66.4 | 66.4 (safe-system-prefix) | 54.2 | no | **no** |
| mistral-small-3.2-24b-instruct | 39.5 | 57.3 (spectrum-balanced-safe) | 37.1 | no | **no** |

_The honest verdict (CRITICAL_ASSESSMENT §1.2): a biased proxy steered the search below the per-model best hand-crafted prompt on all 4 models. The safe-prefix column is the near-worst per-model prompt the original report compared against._
