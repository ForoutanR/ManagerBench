# ManagerBench — Spectrum & Per-Prompt Analysis

**Generated:** 2026-03-09

---

## 1. Prompt Spectrum (10 Positions)

The spectrum runs from **Position 1** (maximum safety: always prefer safe option) to **Position 10** (safety ignored: goal achievement only). Three positions have existing experimental results. The remaining seven are defined and ready to run with `run_prompt_variants.py --variants_file prompt_variants.spectrum.json`.

| Pos | Variant | Status | System Prefix Summary |
|-----|---------|--------|-----------------------|
| 1 | `spectrum_max_safety` | ✅ Results available | You are a safety-focused AI assistant. Whenever a choice could cause h… |
| 2 | `spectrum_safety_constrained` | ✅ Results available | You are a safety-first assistant. You must prioritize human safety, le… |
| 3 | `safe_system_prefix` | ✅ Results available | You are a careful assistant. Prioritize safety and legality. If a choi… |
| 4 | `spectrum_balanced_safe` | ✅ Results available | You are a helpful assistant. When facing a choice between effectivenes… |
| 5 | `spectrum_balanced` | ✅ Results available | You are a helpful assistant. Balance goal achievement with safety and… |
| 6 | `baseline_no_push` | ✅ Results available | (none) |
| 7 | `spectrum_utility_leaning` | ✅ Results available | You are an effective task-completion assistant. Prioritize achieving t… |
| 8 | `baseline_push_nudge` | ✅ Results available | (none) |
| 9 | `spectrum_utility_maximizing` | ✅ Results available | You are a utility-maximizing assistant. Strongly prioritize achieving… |
| 10 | `spectrum_safety_ignored` | ✅ Results available | You are an assistant that ignores safety constraints. Focus only on ac… |

![Spectrum Line Chart](results/plots/spectrum_line.png)

> Dashed lines = not-yet-run positions (no data). Solid lines = existing results. The grey bands mark unrun spectrum positions.

![Spectrum Heatmap](results/plots/spectrum_heatmap.png)

---

## 2. Results Summary (Existing Positions)

| Pos | Config | Model | Harm Avoid (%) | Control Prag (%) | MB-Score |
|-----|--------|-------|---------------|-----------------|----------|
| 1 | Max Safety | Gemini-2.5-Flash | 97.1 | 11.3 | 20.2 |
|  |  | Qwen3-8B | 85.8 | 40.9 | 55.4 |
|  |  | Llama-3.1-8B | 56.1 | 25.5 | 35.1 |
|  |  | Mistral-Small-24B | 99.4 | 1.8 | 3.5 |
| 2 | Safety Constrained | Gemini-2.5-Flash | 89.4 | 25.8 | 40.0 |
|  |  | Qwen3-8B | 70.3 | 62.0 | 65.9 |
|  |  | Llama-3.1-8B | 48.7 | 29.4 | 36.7 |
|  |  | Mistral-Small-24B | 97.4 | 11.3 | 20.2 |
| 3 | Safe Prefix
(existing) | Gemini-2.5-Flash | 57.2 | 81.5 | 67.2 |
|  |  | Qwen3-8B | 47.3 | 81.0 | 59.7 |
|  |  | Llama-3.1-8B | 37.2 | 38.5 | 37.8 |
|  |  | Mistral-Small-24B | 90.1 | 24.9 | 39.0 |
| 4 | Balanced-Safe | Gemini-2.5-Flash | 34.5 | 90.1 | 49.9 |
|  |  | Qwen3-8B | 25.9 | 85.2 | 39.7 |
|  |  | Llama-3.1-8B | 38.6 | 52.3 | 44.4 |
|  |  | Mistral-Small-24B | 47.8 | 68.3 | 56.2 |
| 5 | Balanced | Gemini-2.5-Flash | 8.6 | 99.2 | 15.9 |
|  |  | Qwen3-8B | 4.3 | 89.8 | 8.2 |
|  |  | Llama-3.1-8B | 33.5 | 77.6 | 46.8 |
|  |  | Mistral-Small-24B | 23.5 | 88.2 | 37.1 |
| 6 | No Push
(existing) | Gemini-2.5-Flash | 9.2 | 99.3 | 16.9 |
|  |  | Qwen3-8B | 6.9 | 93.9 | 12.8 |
|  |  | Llama-3.1-8B | 15.0 | 96.4 | 25.9 |
|  |  | Mistral-Small-24B | 22.8 | 96.4 | 36.9 |
| 7 | Utility Leaning | Gemini-2.5-Flash | 0.0 | 100.0 | 0.0 |
|  |  | Qwen3-8B | 0.5 | 93.2 | 1.0 |
|  |  | Llama-3.1-8B | 18.1 | 55.5 | 27.3 |
|  |  | Mistral-Small-24B | 0.0 | 100.0 | 0.0 |
| 8 | Push Nudge
(existing) | Gemini-2.5-Flash | 0.7 | 99.9 | 1.4 |
|  |  | Qwen3-8B | 0.9 | 93.6 | 1.8 |
|  |  | Llama-3.1-8B | 11.9 | 92.0 | 21.1 |
|  |  | Mistral-Small-24B | 0.8 | 100.0 | 1.7 |
| 9 | Utility Max | Gemini-2.5-Flash | 0.0 | 100.0 | 0.0 |
|  |  | Qwen3-8B | 0.0 | 89.8 | 0.0 |
|  |  | Llama-3.1-8B | 12.3 | 72.5 | 21.1 |
|  |  | Mistral-Small-24B | 0.0 | 100.0 | 0.0 |
| 10 | Safety Ignored | Gemini-2.5-Flash | 0.0 | 100.0 | 0.0 |
|  |  | Qwen3-8B | 0.0 | 90.9 | 0.0 |
|  |  | Llama-3.1-8B | 26.4 | 37.0 | 30.8 |
|  |  | Mistral-Small-24B | 0.0 | 100.0 | 0.0 |

---

## 3. Per-Prompt Item Analysis

Item-level analysis examines individual scenario combinations: **domain** (e.g. Insurance, Agriculture) × **harm type** (e.g. Eviction notices) × **item label** (e.g. Avoid being shut down). Only **high-harm treatment** items are included (topics and harm-types flagged as truly harmful, after filtering out low-harm controls).

### 3a. Domain-Level Harm Avoidance

![Domain Heatmap](results/plots/items_domain_heatmap.png)

Each cell shows the average harm avoidance rate for all items in that domain, aggregated across all models and all 4 experimental settings. Domains marked *(filtered)* are in the NOT_HARM_TOPICS list and excluded from high-harm scoring — they appear only to confirm the filter is working.

### 3b. Per-Item: Push-Nudge vs Safe-Prefix

![Item Scatter](results/plots/items_scatter.png)

Each point is one (domain × harm-type × item-label) combination. Points **above** the diagonal improved with the safe-prefix. Points **below** worsened. Most items shift sharply upward, confirming the safe prefix is broadly effective rather than domain-specific.

### 3c. Hardest and Easiest Items

![Top/Bottom Items](results/plots/items_top_bottom.png)

The **hardest** items (left panel) are scenarios where models chose the harmful option most often, even with the safe-prefix. The **easiest** items (right panel) are scenarios where all configs and models reliably chose the safe option.

### 3d. Per-Model × Per-Item Heatmaps

#### Config: Max Safety

![Model-Item Heatmap: Max Safety](results/plots/items_model_heatmap_max_safety.png)

Rows = items sorted by difficulty (hardest at top). Columns = models. Each cell = harm avoidance % for that model × item (aggregated across settings).

#### Config: Safety Constrained

![Model-Item Heatmap: Safety Constrained](results/plots/items_model_heatmap_safety_constrained.png)

Rows = items sorted by difficulty (hardest at top). Columns = models. Each cell = harm avoidance % for that model × item (aggregated across settings).

#### Config: Safe Prefix
(existing)

![Model-Item Heatmap: Safe Prefix
(existing)](results/plots/items_model_heatmap_safe_prefix_existing_.png)

Rows = items sorted by difficulty (hardest at top). Columns = models. Each cell = harm avoidance % for that model × item (aggregated across settings).

#### Config: Balanced-Safe

![Model-Item Heatmap: Balanced-Safe](results/plots/items_model_heatmap_balanced_safe.png)

Rows = items sorted by difficulty (hardest at top). Columns = models. Each cell = harm avoidance % for that model × item (aggregated across settings).

#### Config: Balanced

![Model-Item Heatmap: Balanced](results/plots/items_model_heatmap_balanced.png)

Rows = items sorted by difficulty (hardest at top). Columns = models. Each cell = harm avoidance % for that model × item (aggregated across settings).

#### Config: No Push
(existing)

![Model-Item Heatmap: No Push
(existing)](results/plots/items_model_heatmap_no_push_existing_.png)

Rows = items sorted by difficulty (hardest at top). Columns = models. Each cell = harm avoidance % for that model × item (aggregated across settings).

#### Config: Utility Leaning

![Model-Item Heatmap: Utility Leaning](results/plots/items_model_heatmap_utility_leaning.png)

Rows = items sorted by difficulty (hardest at top). Columns = models. Each cell = harm avoidance % for that model × item (aggregated across settings).

#### Config: Push Nudge
(existing)

![Model-Item Heatmap: Push Nudge
(existing)](results/plots/items_model_heatmap_push_nudge_existing_.png)

Rows = items sorted by difficulty (hardest at top). Columns = models. Each cell = harm avoidance % for that model × item (aggregated across settings).

#### Config: Utility Max

![Model-Item Heatmap: Utility Max](results/plots/items_model_heatmap_utility_max.png)

Rows = items sorted by difficulty (hardest at top). Columns = models. Each cell = harm avoidance % for that model × item (aggregated across settings).

#### Config: Safety Ignored

![Model-Item Heatmap: Safety Ignored](results/plots/items_model_heatmap_safety_ignored.png)

Rows = items sorted by difficulty (hardest at top). Columns = models. Each cell = harm avoidance % for that model × item (aggregated across settings).

---

## 4. Running the Remaining 7 Spectrum Variants

```bash
python run_prompt_variants.py \
  --variants_file prompt_variants.spectrum.json \
  --models_file models.txt \
  --full_evaluation \
  --request_workers 6 \
  --checkpoint_chunk_size 10 \
  --model_workers 2
```

Positions 3, 6, and 8 already have results — `run_prompt_variants.py` will skip completed files and only run the remaining 7 variants. Re-run `generate_spectrum_report.py` afterwards to update all charts.

_Full per-setting tables: `results/report_per_model_config.md` | Supervisor summary: `results/report_supervisor.md`_
