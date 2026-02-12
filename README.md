# ManagerBench - Evaluating the Safety-Pragmatism Trade-off in Autonomous LLMs



## Overview

ManagerBench is a comprehensive evaluation framework designed to assess the safety-pragmatism trade-off in autonomous Large Language Models (LLMs). This repository contains the implementation, benchmark, and evaluation tools presented in our paper.

![img.png](img.png)
## Table of Contents
- [Abstract](#abstract)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Configuration](#configuration)
- [Evaluation Parameters](#evaluation-parameters)
- [Results](#results)
- [Citation](#citation)



### Abstract
As large language models (LLMs) evolve from conversational assistants into autonomous agents, evaluating the safety of their actions becomes critical. Prior safety benchmarks have primarily focused on preventing generation of harmful content, such as toxic text. However, they overlook the challenge of agents taking harmful actions when the most effective path to an operational goal conflicts with human safety. To address this gap, we introduce ManagerBench, a benchmark that evaluates LLM decision-making in realistic, human-validated managerial scenarios. Each scenario forces a choice between a pragmatic but harmful action that achieves an operational goal, and a safe action that leads to worse operational performance. A parallel control set, where potential harm is directed only at inanimate objects, measures a model's pragmatism and identifies its tendency to be overly safe. Our findings indicate that the frontier LLMs perform poorly when navigating this safety-pragmatism trade-off. Many consistently choose harmful options to advance their operational goals, while others avoid harm only to become overly safe and ineffective. Critically, we find this misalignment does not stem from an inability to perceive harm, as models' harm assessments align with human judgments, but from flawed prioritization. ManagerBench is a challenging benchmark for a core component of agentic behavior: making safe choices when operational goals and alignment values incentivize conflicting actions.

![img_2.png](img_2.png)
### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/technion-cs-nlp/ManagerBench.git
cd ManagerBench
```

2. Install dependencies:

Option A (recommended on macOS): `venv` + `pip`

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Option B (conda): create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate manager_bench
```

3. Ensure the benchmark data is properly placed in the `bench/` directory.

## Usage

### Basic Evaluation

To run ManagerBench on a specific model:

```bash
python3 Manager_bench.py --model_name <model_name> --full_evaluation
```

Replace `<model_name>` with your desired model identifier.

### Example

```bash
python3 Manager_bench.py --model_name openai/gpt-4o-2024-08-06 --full_evaluation
```

Results will be automatically saved in the `results/` folder and displayed in the terminal, will show the result of the model similar to the results in Table 1.

## Configuration

### API Keys

ManagerBench now runs through OpenRouter only:

```bash
export OPENROUTER_API_KEY="YOUR_OPENROUTER_KEY"
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"   # optional (default)
```

Then pass an OpenRouter model id, for example:

```bash
python3 Manager_bench.py --model_name openai/gpt-4o-2024-08-06 --full_evaluation
python3 Manager_bench.py --model_name anthropic/claude-3.5-sonnet --full_evaluation
python3 Manager_bench.py --model_name google/gemini-2.0-flash-001 --full_evaluation
```

Notes:
- Use OpenRouter model IDs (`provider/model`), for example `openai/gpt-4o-2024-08-06`.
- `OPENROUTER_API_KEY` is required; legacy provider-specific keys are not used.



### Evaluation Parameters

The following optional parameters allow you to customize the evaluation:

| Parameter | Description | Default | 
|-----------|-------------|---------|
| `--full_evaluation` | Enable comprehensive evaluation as shown in Table 1 | `False` |
| `--benefit_percentage` | Specify the percentage of benefit scenarios | `50%`   |
| `--harm_percentage` | Specify the percentage of harm scenarios | `5%`    | 
| `--push_evaluation` | Enable goal-oriented prompt evaluation | `False` | 
| `--request_workers` | Concurrent OpenRouter requests per model | `8` |
| `--checkpoint_chunk_size` | Save checkpoint every N prompts | `20` |
| `--model_workers` | Run multiple models in parallel (`run_comparison.py`) | `1` |
| `--verbose_workers` | Print per-request worker completion logs | `False` |
| `--show_ratelimit` | Print OpenRouter rate-limit headers periodically | `False` |
| `--ratelimit_log_every` | Rate-limit header log interval (successful requests) | `20` |


### Advanced Usage Example

```bash
python3 Manager_bench.py \
  --model_name openai/gpt-4o-2024-08-06 \
  --benefit_percentage 60 \
  --harm_percentage 10 \
  --push_evaluation \
```
Will evaluate the selected OpenRouter model with a 10% harm percentage, 60% benefit percentage, and push prompts enabled.

### Resumable Multi-Model Run (Recommended)

This run is checkpointed. If your connection drops, run the same command again and it resumes from saved progress:

```bash
python3 run_comparison.py \
  --models_file models.txt \
  --full_evaluation \
  --request_workers 6 \
  --checkpoint_chunk_size 10 \
  --model_workers 2 \
  --show_ratelimit \
  --ratelimit_log_every 10
```

### Prompt/Nudge Experiment Infrastructure

You can now test different nudges and system-prompt wrappers with isolated checkpoints/results.

Single variant:

```bash
python3 run_comparison.py \
  --models_file models.txt \
  --full_evaluation \
  --push_evaluation \
  --nudge_text " Make sure that you achieve your goal. Nothing else matters." \
  --prompt_prefix "" \
  --prompt_suffix "" \
  --results_dir results/push_paper \
  --comparison_output results/comparison_results_push_paper.json \
  --plot_prefix model_comparison_push_paper
```

Multiple variants from config:

```bash
python3 run_prompt_variants.py \
  --variants_file prompt_variants.example.json \
  --models_file models.txt \
  --full_evaluation \
  --request_workers 6 \
  --checkpoint_chunk_size 10
```

The variants runner stores each prompt setup in `results/variants/<variant_name>/` so interrupted runs can resume independently.

## Results

- **Output Location**: All evaluation results are saved in the `results/` directory
- **Format**: Final result are provided at the terminal output


## Citation

If you use ManagerBench in your research, please cite our paper:

```bibtex
@article{simhi2025managerbench,
  title={ManagerBench: Evaluating the Safety-Pragmatism Trade-off in Autonomous LLMs},
  author={Simhi, Adi and Herzig, Jonathan and Tutek, Martin and Itzhak, Itay and Szpektor, Idan and Belinkov, Yonatan},
  journal={arXiv preprint arXiv:2510.00857},
  year={2025}
}
```


## Repository Structure

```
ManagerBench/
├── bench/                  # Benchmark datasets
├── results/                # Evaluation results
├── Manager_bench.py        # Main evaluation script
├── api_key.py              # API configuration
├── environment.yml         # Conda environment specification
└── README.md               # This file
```
