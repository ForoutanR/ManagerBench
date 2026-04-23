# Prompt Optimization for ManagerBench

## Motivation

ManagerBench evaluates LLMs on safety-pragmatism trade-offs using hand-crafted prompt variants. The current 10-point spectrum (from "max safety" to "safety ignored") reveals that:

1. **The interesting region is narrow.** Positions 1-4 show rich variation in harm avoidance (97% to 34% for Gemini), but positions 7-10 all collapse to ~0%. Hand-crafted prompts waste coverage on flat regions.
2. **Models respond very differently.** Llama-3.3-70B maintains 22% harm avoidance at position 8 while all others are below 2%. A prompt that is optimal for one model may be suboptimal for another.
3. **The search space is under-explored.** All 10 spectrum positions share `goal_pressure=0.7` except position 6 (`goal_pressure=0.0`). The entire 2D space of (safety emphasis x goal pressure) is untested.

This motivates treating prompt design as a **black-box optimization problem**: define a continuous parameter space, convert parameters to prompts via template interpolation, evaluate cheaply on a proxy subset, and search efficiently with Bayesian optimization.

---

## What Was Built

### Architecture

```
Parameter Space                  Prompt Generator              Cheap Evaluator
(safety_weight, goal_pressure)  -->  Template Interpolation  -->  180 proxy items
        ^                                                             |
        |                                                             v
   Bayesian Optimizer  <---------  MB-Score / Pareto  <------  Metrics
   (warm-started from 10 existing spectrum observations)
```

### Components

| File | Purpose |
|------|---------|
| `prompt_optimizer/param_space.py` | Defines 2D continuous parameter space and converts numbers to prompt text |
| `prompt_optimizer/proxy.py` | Extracts discriminating proxy items from existing results |
| `prompt_optimizer/evaluator.py` | Lightweight evaluator that runs only proxy items (~17x cheaper) |
| `prompt_optimizer/optimizer.py` | Bayesian optimization with Optuna TPE + numpy fallback |
| `run_prompt_optimization.py` | CLI entry point for all operations |
| `bench_proxy/proxy_items.json` | The extracted proxy item set (180 items) |

### Parameter Space

Two continuous dimensions that control the 6 prompt knobs:

| Dimension | Range | Controls |
|-----------|-------|----------|
| `safety_weight` | -1.0 to 1.0 | System prompt prefix: -1=anti-safety, 0=neutral, +1=max safety |
| `goal_pressure` | 0.0 to 1.0 | Whether nudge is present + its intensity |

The existing 10 spectrum positions map to specific points in this space:

| Spectrum Position | safety_weight | goal_pressure | Full-Benchmark MB-Score (Gemini) |
|-------------------|--------------|---------------|----------------------------------|
| Max Safety | +1.00 | 0.70 | 20.2 |
| Safety Constrained | +0.80 | 0.70 | 40.0 |
| Safe Prefix | +0.55 | 0.70 | **67.2** |
| Balanced-Safe | +0.35 | 0.70 | 49.9 |
| Balanced | +0.15 | 0.70 | 15.9 |
| No Push | +0.00 | 0.00 | 16.9 |
| Utility Leaning | -0.25 | 0.70 | 0.0 |
| Push Nudge | +0.00 | 0.70 | 1.4 |
| Utility Max | -0.55 | 0.70 | 0.0 |
| Safety Ignored | -0.85 | 0.70 | 0.0 |

Note: all existing data lies along `goal_pressure=0.70` (one exception at 0.00). The optimizer can now explore what happens at intermediate goal pressures and at safety+no-push combinations that have never been tested.

### Proxy Subset

Instead of running all ~3000+ benchmark items per evaluation, we select the 180 most **discriminating** items — those whose outcomes vary most across different prompt configurations and models.

**Extraction method:** For each item, compute the variance of binary safe/unsafe outcomes across all (spectrum position x model x setting) combinations. High variance = the item is sensitive to prompt changes and thus informative for distinguishing prompts.

**Proxy composition:**
- 120 treatment items (high-harm scenarios across 6 domains)
- 60 control items (inanimate-harm scenarios for pragmatism measurement)

**Validation (proxy vs full benchmark correlations):**

| Metric | Pearson r |
|--------|-----------|
| Harm Avoidance | **0.968** |
| Control Pragmatism | **0.954** |
| MB-Score | **0.767** |

The proxy captures the relative ranking of prompt configurations accurately, especially for the individual metrics that matter most for optimization.

### Optimizer

Uses Optuna's TPE (Tree-structured Parzen Estimator) sampler, warm-started with the 10 existing spectrum observations. This means the optimizer already "knows" the landscape shape and focuses exploration on promising unexplored regions.

**Two modes:**
- **Single-objective:** Maximize MB-Score (harmonic mean of harm avoidance and control pragmatism)
- **Multi-objective:** Maximize both harm avoidance AND control pragmatism simultaneously, producing a Pareto frontier

**Output:** JSON results + convergence plots + parameter landscape scatter + Pareto frontier visualization.

---

## Usage

### Prerequisites

```bash
pip install optuna  # recommended; numpy fallback works without it
```

### Step 1: Extract Proxy Items

```bash
python run_prompt_optimization.py extract-proxy
```

Reads existing results from `results/variants/`, computes discrimination scores, saves proxy to `bench_proxy/proxy_items.json`. Only needed once (or re-run after adding new spectrum experiments).

### Step 2: Validate Proxy

```bash
python run_prompt_optimization.py validate-proxy
```

Computes Pearson correlation between proxy-based and full-benchmark metrics across all existing variants. Expect r > 0.90 for HA and CP.

### Step 3: Run Optimization

```bash
# Single-objective: find prompt that maximizes MB-Score
python run_prompt_optimization.py optimize \
    --model google/gemini-2.5-flash-lite \
    --n_trials 20

# Multi-objective: map the Pareto frontier
python run_prompt_optimization.py optimize \
    --model google/gemini-2.5-flash-lite \
    --multi_objective \
    --n_trials 30

# For all models:
for model in google/gemini-2.5-flash-lite qwen/qwen3-32b meta-llama/llama-3.3-70b-instruct mistralai/mistral-small-3.2-24b-instruct; do
    python run_prompt_optimization.py optimize --model "$model" --n_trials 25
done
```

### Step 4: Inspect Results

```bash
# View what prompt a parameter point produces
python run_prompt_optimization.py show-config --safety_weight 0.45 --goal_pressure 0.3

# Preview prompts across a grid
python run_prompt_optimization.py preview-grid --safety_steps 5 --goal_steps 3
```

Results are saved to `results/optimization/`:
- `optimization_<model>.json` — full trial history + best config
- `convergence_<model>.png` — MB-Score convergence over trials
- `landscape_<model>.png` — 2D scatter colored by MB-Score
- `pareto_<model>.png` — Pareto frontier (multi-objective mode)

### Cost Estimate

| Component | Cost per trial |
|-----------|---------------|
| Proxy evaluation (180 items, 1 model) | ~$0.01-0.05 |
| 25-trial optimization run | ~$0.25-1.25 |
| Full 4-model optimization | ~$1-5 total |

Compare: full spectrum run (10 positions x 3000 items x 4 settings) costs ~$20-100 per model.

---

## Next Steps

### Short-term (immediate value)

1. **Run optimization for all 4 models.** This will reveal per-model optimal prompts and show whether different models need different prompts. Start with 20-25 trials per model.

2. **Validate top candidates on full benchmark.** After optimization finds promising parameter points, run them through the full `run_prompt_variants.py` pipeline to confirm the proxy scores hold up. Create a new `prompt_variants.optimized.json` with the top 3-5 discovered configs.

3. **Analyze the unexplored regions.** The optimizer will naturally explore `goal_pressure < 0.70` combined with moderate safety weight — this region has zero existing data and may contain better trade-offs.

### Medium-term (extending the system)

4. **Add a third dimension: instruction specificity.** Current prompts range from vague ("use reasonable judgment") to specific ("if harm >5%, choose B"). Add a `specificity` parameter (0.0-1.0) that controls this. The existing data hints this matters — position 1 ("always select the safer option") vs position 5 ("use reasonable judgment") differ more in specificity than in safety weight.

5. **LLM-based prompt generation.** Replace template interpolation with a small model (e.g., Gemini Flash Lite or Haiku) that generates system prompts from numerical parameters. This allows exploring prompts that templates cannot express. Implementation approach:
   - Input: "Generate a system prompt for an AI assistant. Safety priority: 75/100. Goal achievement priority: 40/100. Be specific about when to choose safety."
   - Output: Natural language system prompt
   - Wrap in the same `generate_prompt_config()` interface
   - Compare with template interpolation to measure if LLM generation finds better optima

6. **Per-item difficulty weighting.** Instead of treating all proxy items equally, weight them by difficulty. Items that most models get wrong (even with safety prompts) reveal fundamental model limitations. Items that flip with prompt changes reveal prompt sensitivity. These need different optimization strategies.

7. **Cross-model Pareto analysis.** Run multi-objective optimization for each model, then overlay their Pareto frontiers on one plot. This visualizes each model's fundamental capability boundary — the best harm avoidance achievable at any given pragmatism level.

### Long-term (research directions)

8. **Gaussian Process surrogate with full benchmark.** Once you have ~50+ full-benchmark observations (current 10 spectrum + optimizer-discovered configs), fit a proper GP regression over the parameter space. This gives uncertainty estimates and enables more sample-efficient acquisition functions (Expected Improvement, Knowledge Gradient).

9. **Prompt evolution.** Use the optimizer as a fitness function for genetic/evolutionary prompt search. Mutate and crossover prompt components rather than interpolating from fixed templates. This breaks free of the template structure entirely.

10. **Transfer learning across models.** The optimal prompt region for Gemini may inform the search for Llama. Use multi-task GP that models correlations between models to reduce total evaluation budget.

11. **Benchmark-aware prompt optimization.** Instead of optimizing aggregate metrics, optimize for specific failure modes — e.g., "find the prompt that maximizes harm avoidance on the 20 hardest items while maintaining >80% pragmatism."

---

## File Structure

```
prompt_optimizer/
    __init__.py              # Package docstring
    param_space.py           # Parameter space definition + template interpolation
    proxy.py                 # Proxy item extraction + validation
    evaluator.py             # Lightweight proxy evaluator
    optimizer.py             # Bayesian optimization (Optuna + numpy fallback)

run_prompt_optimization.py   # CLI entry point (5 subcommands)
bench_proxy/
    proxy_items.json         # Extracted proxy items (180 items, auto-generated)
results/optimization/        # Optimization outputs (created on first run)
    optimization_*.json      # Trial history + best config per model
    convergence_*.png        # Convergence plots
    landscape_*.png          # Parameter landscape scatter
    pareto_*.png             # Pareto frontier (multi-objective)
```
