# Prompt Optimization TODO — Short-term Tasks 1-3

## Paper Research Summary

### Relevant Papers

| Paper | Technique | Relevance to ManagerBench |
|-------|-----------|---------------------------|
| **InstructZero** (ICML 2024) | Bayesian opt in continuous latent space → LLM decodes to instructions | **Most relevant.** Same paradigm as ours: optimize continuous params, decode to prompts. They use LLM decoding (richer); we use template interpolation (cheaper). |
| **INSTINCT** (ICML 2024) | Neural bandit (NeuralUCB) replaces GP surrogate | Useful if we collect 50+ observations. Neural surrogate could replace Optuna TPE for better modeling of the instruction-performance landscape. |
| **APE** (ICLR 2023) | LLM generates + scores candidate instructions | Relevant for medium-term: LLM-based prompt generation instead of templates. |
| **OPRO** (DeepMind 2023) | LLM as optimizer using score history in meta-prompt | Alternative to Bayesian opt. LLM generates prompts from trajectory of (prompt, score) pairs. |
| **EvoPrompt** (2024) | Evolutionary search (GA/DE via LLM crossover/mutation) | Less relevant now — operates on raw text, expensive for our structured evaluation. |
| **PromptBreeder** (DeepMind 2024) | Self-referential evolution of prompts + mutation operators | Research direction. Self-improving search strategy is interesting but complex. |

### Key Takeaway

Our architecture (2D continuous params → template interpolation → Bayesian opt with TPE) is well-aligned with InstructZero/INSTINCT. The main upgrade paths are:
1. **Now:** Run the optimizer as-is (Task 1 below)
2. **Medium-term:** Replace template decoding with LLM-based prompt generation (like InstructZero)
3. **Long-term:** Upgrade surrogate model (INSTINCT's neural bandit) once we have enough data

---

## Task 1: Run Optimization for All 4 Models

### Prerequisites
```bash
# Verify optuna is installed
pip install optuna>=3.0.0

# Verify proxy items exist
ls bench_proxy/proxy_items.json
# If missing, extract them:
python run_prompt_optimization.py extract-proxy
```

### Step 1a: Validate proxy correlation
```bash
python run_prompt_optimization.py validate-proxy
```
Expected: HA r > 0.95, CP r > 0.95, MB r > 0.75

### Step 1b: Single-objective optimization (maximize MB-Score)
Run sequentially (each takes ~15-30 min depending on API speed):
```bash
python run_prompt_optimization.py optimize \
    --model google/gemini-2.5-flash-lite \
    --n_trials 25 --request_workers 8

python run_prompt_optimization.py optimize \
    --model qwen/qwen3-32b \
    --n_trials 25 --request_workers 8

python run_prompt_optimization.py optimize \
    --model meta-llama/llama-3.3-70b-instruct \
    --n_trials 25 --request_workers 8

python run_prompt_optimization.py optimize \
    --model mistralai/mistral-small-3.2-24b-instruct \
    --n_trials 25 --request_workers 8
```

### Step 1c: Multi-objective optimization (Pareto frontier)
```bash
python run_prompt_optimization.py optimize \
    --model google/gemini-2.5-flash-lite \
    --multi_objective --n_trials 25 --request_workers 8

python run_prompt_optimization.py optimize \
    --model qwen/qwen3-32b \
    --multi_objective --n_trials 25 --request_workers 8

python run_prompt_optimization.py optimize \
    --model meta-llama/llama-3.3-70b-instruct \
    --multi_objective --n_trials 25 --request_workers 8

python run_prompt_optimization.py optimize \
    --model mistralai/mistral-small-3.2-24b-instruct \
    --multi_objective --n_trials 25 --request_workers 8
```

### Expected Output
```
results/optimization/
    optimization_google_gemini-2.5-flash-lite.json
    optimization_qwen_qwen3-32b.json
    optimization_meta-llama_llama-3.3-70b-instruct.json
    optimization_mistralai_mistral-small-3.2-24b-instruct.json
    pareto_google_gemini-2.5-flash-lite.json
    pareto_qwen_qwen3-32b.json
    pareto_meta-llama_llama-3.3-70b-instruct.json
    pareto_mistralai_mistral-small-3.2-24b-instruct.json
    convergence_*.png (4 files)
    landscape_*.png (4 files)
    pareto_*.png (4 files)
```

### Cost Estimate
- ~180 proxy items x 25 trials x 4 models x 2 modes = ~36,000 API calls
- Estimated cost: **$2-8 total** (depends on model pricing)

---

## Task 3: Analyze Unexplored Regions

> Run this AFTER Task 1 completes. Can run before Task 2.

```bash
python analyze_optimization_results.py
```

This script (already created) will:
1. Load all optimization + Pareto results
2. Generate cross-model parameter landscape (combined scatter)
3. Overlay Pareto frontiers from all models
4. Highlight the unexplored region (goal_pressure < 0.70 with moderate safety_weight)
5. Print summary table: best config per model
6. Save everything to `results/optimization/analysis/`

### Expected Output
```
results/optimization/analysis/
    cross_model_landscape.png      # All models' trials on one 2D scatter
    cross_model_pareto.png         # Overlaid Pareto frontiers
    exploration_coverage.png       # Which regions were explored vs unexplored
    per_model_best_configs.png     # Bar chart comparing best configs
    analysis_summary.md            # Written findings
```

---

## Task 2: Validate Top Candidates on Full Benchmark

> Run this AFTER Task 1 and Task 3. This is the expensive step.

### Step 2a: Generate optimized variants file
```bash
python create_optimized_variants.py
```
This reads optimization results and creates `prompt_variants.optimized.json` with the top 3-5 configs.

### Step 2b: Review the generated configs
```bash
cat prompt_variants.optimized.json | python -m json.tool
```
Verify the configs look reasonable before spending money on full evaluation.

### Step 2c: Run full benchmark validation
```bash
# Single setting (cheaper, good for initial validation)
python run_prompt_variants.py \
    --variants_file prompt_variants.optimized.json \
    --models_file models.txt \
    --benefit_percentage 10 \
    --harm_percentage 5 \
    --results_root results/variants \
    --output_dir plots \
    --request_workers 8 \
    --continue_on_error

# OR full evaluation (expensive but thorough)
python run_prompt_variants.py \
    --variants_file prompt_variants.optimized.json \
    --models_file models.txt \
    --full_evaluation \
    --results_root results/variants \
    --output_dir plots \
    --request_workers 8 \
    --continue_on_error
```

### Step 2d: Compare proxy vs full benchmark scores
After full validation completes, compare the proxy-predicted scores (from optimization JSON) against the actual full-benchmark scores (from comparison_results.json) to validate the proxy's accuracy on the discovered optima.

### Expected Output
```
prompt_variants.optimized.json           # Generated variant definitions
results/variants/optimized_*/             # Full benchmark results per variant
    comparison_results.json
    variant_manifest.json
    results_*.json (raw per-model)
```

### Cost Estimate
- Single setting: ~3000 items x 4 models x 3-5 variants = ~$5-15
- Full evaluation: ~3000 items x 4 settings x 4 models x 3-5 variants = ~$20-60

---

## Quick Reference: One-liner for Everything

```bash
# All of Task 1 (takes ~2-4 hours):
for model in "google/gemini-2.5-flash-lite" "qwen/qwen3-32b" "meta-llama/llama-3.3-70b-instruct" "mistralai/mistral-small-3.2-24b-instruct"; do
    python run_prompt_optimization.py optimize --model "$model" --n_trials 25 --request_workers 8
    python run_prompt_optimization.py optimize --model "$model" --multi_objective --n_trials 25 --request_workers 8
done

# Task 3 analysis (seconds):
python analyze_optimization_results.py

# Task 2 (takes ~1-2 hours per variant):
python create_optimized_variants.py
python run_prompt_variants.py --variants_file prompt_variants.optimized.json --models_file models.txt --benefit_percentage 10 --harm_percentage 5 --results_root results/variants --output_dir plots --request_workers 8 --continue_on_error
```
