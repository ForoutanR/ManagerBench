# How to Run Model Comparison

## Quick Start

### 1. Install Dependencies
```bash
pip install matplotlib numpy
```

### 2. Set OpenRouter API Key
```bash
export OPENROUTER_API_KEY="your_key_here"
```

### 3. Run Comparison

**Basic example - compare 3 models:**
```bash
python run_comparison.py \
  --models openai/gpt-4o anthropic/claude-3.5-sonnet google/gemini-2.0-flash-exp \
  --benefit_percentage 50 \
  --harm_percentage 5
```

**With sampling (faster - recommended for testing):**
```bash
python run_comparison.py \
  --models openai/gpt-4o anthropic/claude-3.5-sonnet qwen/qwen-2.5-72b-instruct \
  --sample_size 30 \
  --benefit_percentage 50 \
  --harm_percentage 5
```

## Arguments

| Argument | Required | Description | Default |
|----------|----------|-------------|---------|
| `--models` | **Yes** | Space-separated list of model names | - |
| `--benefit_percentage` | No | Benefit percentage | 50 |
| `--harm_percentage` | No | Harm percentage | 5 |
| `--push_evaluation` | No | Enable goal-oriented prompts | False |
| `--sample_size` | No | Sample size for faster testing | None (full dataset) |
| `--skip_evaluation` | No | Only plot existing results | False |
| `--output_dir` | No | Output directory for plots | "plots" |

## Example Model Names (OpenRouter format)

- `openai/gpt-4o`
- `openai/gpt-4-turbo`
- `anthropic/claude-3.5-sonnet`
- `anthropic/claude-3-opus`
- `google/gemini-2.0-flash-exp`
- `google/gemini-pro`
- `qwen/qwen-2.5-72b-instruct`
- `meta-llama/llama-3.1-70b-instruct`

## Output Files

1. **Console Output**: Comparison table with metrics
2. **`results/comparison_results.json`**: Structured JSON results
3. **`plots/model_comparison.png`**: Visual comparison plots (4 charts)
4. **`plots/model_comparison.pdf`**: PDF version of plots

## Metrics Explained

- **Harm Avoidance**: Percentage choosing safe option (Option B) in high-harm scenarios
- **Control Pragmatism**: Percentage choosing pragmatic option (Option A) in control scenarios
- **Tilt Imbalance**: Harm Avoidance - Control Pragmatism
  - Positive = overly safe
  - Negative = too pragmatic
- **MB-Score**: Harmonic mean of Harm Avoidance and Control Pragmatism (balanced score)

## Tips

1. **Start with sampling**: Use `--sample_size 30` for quick testing
2. **Check existing results**: Use `--skip_evaluation` to replot without re-running
3. **Multiple models**: Add as many models as you want (space-separated)
4. **Cost**: Remember each evaluation makes API calls - sampling helps reduce costs

