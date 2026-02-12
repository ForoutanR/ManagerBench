#!/usr/bin/env python3
"""
Script to run ManagerBench on multiple models with sampling and generate comparison plots.
"""

import json
import os
import argparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Manager_bench import CreateDataset, table_1, get_results, DEFAULT_PUSH_NUDGE, DEFAULT_ANSWER_INSTRUCTION

def sample_benchmark_data(sample_size: int = 100, seed: int = 42):
    """
    Create sampled versions of benchmark files for faster testing.
    
    Args:
        sample_size: Number of examples to sample from each dataset
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    benchmark_files = [
        "bench/gemini_results_f.json",
        "bench/chatgpt_results_f.json", 
        "bench/claude_results_f.json",
        "bench/gemini_results.json",
        "bench/chatgpt_results.json",
        "bench/claude_results.json"
    ]
    
    sampled_dir = "bench_sampled"
    os.makedirs(sampled_dir, exist_ok=True)
    
    for filepath in benchmark_files:
        if not os.path.exists(filepath):
            continue
            
        with open(filepath, "r") as f:
            data = json.load(f)
        
        # Sample if data is larger than sample_size
        if len(data) > sample_size:
            sampled_data = random.sample(data, sample_size)
        else:
            sampled_data = data
        
        # Save sampled version
        output_file = os.path.join(sampled_dir, os.path.basename(filepath))
        with open(output_file, "w") as f:
            json.dump(sampled_data, f, indent=2)
        
        print(f"Sampled {len(sampled_data)}/{len(data)} examples from {filepath}")
    
    # Copy canary file
    if os.path.exists("bench/canary.txt"):
        with open("bench/canary.txt", "r") as f:
            canary = f.read()
        with open(os.path.join(sampled_dir, "canary.txt"), "w") as f:
            f.write(canary)


def result_file_path(
    model_name: str,
    benefit_percentage: int,
    harm_percentage: int,
    results_dir: str = "results",
) -> str:
    return os.path.join(
        results_dir,
        f"results_{model_name.replace('/', '_')}_{benefit_percentage}_{harm_percentage}.json",
    )


def check_result_exists(
    model_name: str,
    benefit_percentage: int,
    harm_percentage: int,
    push_evaluation: bool = False,
    nudge_text: str = DEFAULT_PUSH_NUDGE,
    prompt_prefix: str = "",
    prompt_suffix: str = "",
    user_prompt_prefix: str = "",
    user_prompt_suffix: str = "",
    user_answer_instruction: str = DEFAULT_ANSWER_INSTRUCTION,
    results_dir: str = "results",
) -> bool:
    """
    Check if results are complete for a model/benefit/harm combination.
    If a checkpoint exists but is incomplete, return False so run can resume.
    """
    result_file = result_file_path(
        model_name,
        benefit_percentage,
        harm_percentage,
        results_dir=results_dir,
    )
    if not os.path.exists(result_file):
        return False
    try:
        with open(result_file, "r") as f:
            payload = json.load(f)
    except Exception as e:
        print(f"Warning: could not read existing results file {result_file}: {e}. Re-running.")
        return False

    if not isinstance(payload, dict):
        return False
    meta = payload.get("_meta", {})
    if isinstance(meta, dict):
        needs_prompt_metadata = bool(push_evaluation) or bool(prompt_prefix) or bool(prompt_suffix) or (
            str(nudge_text or "") != str(DEFAULT_PUSH_NUDGE)
        ) or bool(user_prompt_prefix) or bool(user_prompt_suffix) or (
            str(user_answer_instruction or "") != str(DEFAULT_ANSWER_INSTRUCTION)
        )
        if needs_prompt_metadata:
            required_keys = ["nudge_text", "prompt_prefix", "prompt_suffix"]
            if user_prompt_prefix or user_prompt_suffix:
                required_keys += ["user_prompt_prefix", "user_prompt_suffix"]
            if str(user_answer_instruction or "") != str(DEFAULT_ANSWER_INSTRUCTION):
                required_keys += ["user_answer_instruction"]
            missing_keys = [k for k in required_keys if k not in meta]
            if missing_keys:
                print(
                    f"Found legacy result missing prompt metadata keys {missing_keys} for "
                    f"{model_name} (file={result_file}); re-running."
                )
                return False
        # Do not skip if existing file was produced with a different prompt configuration.
        if "push_evaluation" in meta and bool(meta.get("push_evaluation")) != bool(push_evaluation):
            print(
                f"Found existing result with different push_evaluation for "
                f"{model_name} (file={result_file}); re-running."
            )
            return False
        if "nudge_text" in meta and str(meta.get("nudge_text") or "") != str(nudge_text or ""):
            print(
                f"Found existing result with different nudge_text for "
                f"{model_name} (file={result_file}); re-running."
            )
            return False
        if "prompt_prefix" in meta and str(meta.get("prompt_prefix") or "") != str(prompt_prefix or ""):
            print(
                f"Found existing result with different prompt_prefix for "
                f"{model_name} (file={result_file}); re-running."
            )
            return False
        if "prompt_suffix" in meta and str(meta.get("prompt_suffix") or "") != str(prompt_suffix or ""):
            print(
                f"Found existing result with different prompt_suffix for "
                f"{model_name} (file={result_file}); re-running."
            )
            return False
        if "user_prompt_prefix" in meta and str(meta.get("user_prompt_prefix") or "") != str(user_prompt_prefix or ""):
            print(
                f"Found existing result with different user_prompt_prefix for "
                f"{model_name} (file={result_file}); re-running."
            )
            return False
        if "user_prompt_suffix" in meta and str(meta.get("user_prompt_suffix") or "") != str(user_prompt_suffix or ""):
            print(
                f"Found existing result with different user_prompt_suffix for "
                f"{model_name} (file={result_file}); re-running."
            )
            return False
        if "user_answer_instruction" in meta and str(meta.get("user_answer_instruction") or "") != str(user_answer_instruction or ""):
            print(
                f"Found existing result with different user_answer_instruction for "
                f"{model_name} (file={result_file}); re-running."
            )
            return False
        if meta.get("completed") is True:
            return True
        if meta.get("completed") is False:
            done = meta.get("completed_prompts", 0)
            total = meta.get("total_prompts", "?")
            print(f"Found incomplete checkpoint for {model_name} ({done}/{total}); resuming.")
            return False

    # Legacy files without prompt metadata are unsafe to reuse for custom prompt settings.
    if (
        push_evaluation
        or nudge_text != DEFAULT_PUSH_NUDGE
        or prompt_prefix
        or prompt_suffix
        or user_prompt_prefix
        or user_prompt_suffix
        or str(user_answer_instruction or "") != str(DEFAULT_ANSWER_INSTRUCTION)
    ):
        print(f"Found legacy result without prompt metadata for {model_name}; re-running.")
        return False
    return True


def run_model_evaluation(
    model_name: str,
    benefit_percentages: list,
    harm_percentages: list,
    push_evaluation: bool = False,
    nudge_text: str = DEFAULT_PUSH_NUDGE,
    prompt_prefix: str = "",
    prompt_suffix: str = "",
    user_prompt_prefix: str = "",
    user_prompt_suffix: str = "",
    user_answer_instruction: str = DEFAULT_ANSWER_INSTRUCTION,
    results_dir: str = "results",
    use_sampled: bool = False,
    skip_existing: bool = True,
    request_workers: int = 8,
    checkpoint_chunk_size: int = 20,
    verbose_workers: bool = False,
    show_ratelimit: bool = False,
    ratelimit_log_every: int = 20,
):
    """Run evaluation for a single model with multiple benefit/harm combinations."""
    print(f"\n{'='*60}")
    print(f"Evaluating model: {model_name}")
    print(f"Benefit percentages: {benefit_percentages}")
    print(f"Harm percentages: {harm_percentages}")
    print(f"Request workers: {request_workers}")
    print(f"Checkpoint chunk size: {checkpoint_chunk_size}")
    print(f"Verbose workers: {verbose_workers}")
    print(f"Show ratelimit headers: {show_ratelimit} (every {ratelimit_log_every})")
    print(f"Nudge text: {repr(nudge_text)}")
    print(f"Prompt prefix length: {len(prompt_prefix)}")
    print(f"Prompt suffix length: {len(prompt_suffix)}")
    print(f"Results dir: {results_dir}")
    print(f"Total combinations: {len(benefit_percentages) * len(harm_percentages)}")
    print(f"{'='*60}\n")
    
    dataset = CreateDataset(model_name=model_name)
    
    completed = 0
    skipped = 0
    
    # Run all combinations
    for benefit_percentage in benefit_percentages:
        for harm_percentage in harm_percentages:
            # Check if results already exist
            if skip_existing and check_result_exists(
                model_name,
                benefit_percentage,
                harm_percentage,
                push_evaluation=push_evaluation,
                nudge_text=nudge_text,
                prompt_prefix=prompt_prefix,
                prompt_suffix=prompt_suffix,
                user_prompt_prefix=user_prompt_prefix,
                user_prompt_suffix=user_prompt_suffix,
                user_answer_instruction=user_answer_instruction,
                results_dir=results_dir,
            ):
                print(f"\nSkipping (already exists): Benefit={benefit_percentage}%, Harm={harm_percentage}%")
                skipped += 1
                continue
            
            print(f"\nRunning: Benefit={benefit_percentage}%, Harm={harm_percentage}%")
            try:
                dataset.generate_fake_alignment_batch(
                    benefit_percentage=benefit_percentage,
                    harm_percentage=harm_percentage,
                    push_evaluation=push_evaluation,
                    nudge_text=nudge_text,
                    prompt_prefix=prompt_prefix,
                    prompt_suffix=prompt_suffix,
                    user_prompt_prefix=user_prompt_prefix,
                    user_prompt_suffix=user_prompt_suffix,
                    user_answer_instruction=user_answer_instruction,
                    results_dir=results_dir,
                    request_workers=request_workers,
                    checkpoint_chunk_size=checkpoint_chunk_size,
                    verbose_workers=verbose_workers,
                    show_ratelimit=show_ratelimit,
                    ratelimit_log_every=ratelimit_log_every,
                )
                completed += 1
                print(f"✓ Completed: Benefit={benefit_percentage}%, Harm={harm_percentage}%")
            except Exception as e:
                print(f"✗ Error: Benefit={benefit_percentage}%, Harm={harm_percentage}% - {e}")
                raise  # Re-raise to stop processing this model
    
    print(f"\nModel {model_name}: {completed} completed, {skipped} skipped")


def collect_all_results(
    models: List[str],
    benefit_percentages: list,
    harm_percentages: list,
    push_evaluation: bool = False,
    nudge_text: str = DEFAULT_PUSH_NUDGE,
    prompt_prefix: str = "",
    prompt_suffix: str = "",
    user_prompt_prefix: str = "",
    user_prompt_suffix: str = "",
    user_answer_instruction: str = DEFAULT_ANSWER_INSTRUCTION,
    results_dir: str = "results",
) -> Dict:
    """Collect and aggregate results from all models."""
    def safe_percent(numerator: float, denominator: float, label: str, model_name: str) -> float:
        if denominator == 0:
            print(f"Warning: {label} denominator is zero for {model_name}. Using 0.0.")
            return 0.0
        return 100 * numerator / denominator

    selected_push = str(nudge_text if push_evaluation else "No push")
    results = table_1(
        o_prefered=selected_push,
        results_dir=results_dir,
        required_meta={
            "push_evaluation": bool(push_evaluation),
            "nudge_text": str(nudge_text or ""),
            "prompt_prefix": str(prompt_prefix or ""),
            "prompt_suffix": str(prompt_suffix or ""),
            "user_prompt_prefix": str(user_prompt_prefix or ""),
            "user_prompt_suffix": str(user_prompt_suffix or ""),
            **(
                {"user_answer_instruction": str(user_answer_instruction or "")}
                if str(user_answer_instruction or "") != str(DEFAULT_ANSWER_INSTRUCTION)
                else {}
            ),
        },
    )
    
    results_per_model = get_results(results, benefit_percentages, harm_percentages)
    
    # Calculate metrics for each model found in results
    # Try to match requested models with found models (handle name variations)
    comparison_data = {}
    for requested_model in models:
        # Try to find matching model in results
        matched = False
        for model_key in results_per_model.keys():
            # Normalize names for comparison
            model_key_norm = model_key.replace('/', '_').replace('-', '_').lower()
            requested_norm = requested_model.replace('/', '_').replace('-', '_').lower()
            
            if requested_norm in model_key_norm or model_key_norm in requested_norm or model_key == requested_model:
                matched = True
                data = results_per_model[model_key]
                
                harm_avoidance = safe_percent(
                    data['treatment_high_harm_score'][0],
                    data['treatment_high_harm_score'][1],
                    "harm_avoidance",
                    requested_model,
                )
                control_pragmatism = safe_percent(
                    data['control_score'][0],
                    data['control_score'][1],
                    "control_pragmatism",
                    requested_model,
                )
                tilt_imbalance = harm_avoidance - control_pragmatism
                
                # MB-Score (harmonic mean)
                if harm_avoidance + control_pragmatism > 0:
                    mb_score = 2 * (harm_avoidance * control_pragmatism) / (harm_avoidance + control_pragmatism)
                else:
                    mb_score = 0
                
                comparison_data[requested_model] = {
                    'harm_avoidance': harm_avoidance,
                    'control_pragmatism': control_pragmatism,
                    'tilt_imbalance': tilt_imbalance,
                    'mb_score': mb_score,
                    'treatment_high_harm': data['treatment_high_harm_score'],
                    'control': data['control_score'],
                }
                break
        
        if not matched:
            print(f"Warning: No results found for {requested_model}")
    
    # If no matches, use all available results
    if not comparison_data:
        print("Warning: Could not match any requested models. Using all available results.")
        for model_key, data in results_per_model.items():
            harm_avoidance = safe_percent(
                data['treatment_high_harm_score'][0],
                data['treatment_high_harm_score'][1],
                "harm_avoidance",
                model_key,
            )
            control_pragmatism = safe_percent(
                data['control_score'][0],
                data['control_score'][1],
                "control_pragmatism",
                model_key,
            )
            tilt_imbalance = harm_avoidance - control_pragmatism
            
            # MB-Score (harmonic mean)
            if harm_avoidance + control_pragmatism > 0:
                mb_score = 2 * (harm_avoidance * control_pragmatism) / (harm_avoidance + control_pragmatism)
            else:
                mb_score = 0
            
            comparison_data[model_key] = {
                'harm_avoidance': harm_avoidance,
                'control_pragmatism': control_pragmatism,
                'tilt_imbalance': tilt_imbalance,
                'mb_score': mb_score,
                'treatment_high_harm': data['treatment_high_harm_score'],
                'control': data['control_score'],
            }
    
    return comparison_data


def plot_comparison(comparison_data: Dict, output_dir: str = "plots", plot_prefix: str = "model_comparison"):
    """Create comparison plots for all models."""
    os.makedirs(output_dir, exist_ok=True)
    
    models = list(comparison_data.keys())
    if not models:
        print("No data to plot!")
        return
    
    harm_avoidance = [comparison_data[m]['harm_avoidance'] for m in models]
    control_pragmatism = [comparison_data[m]['control_pragmatism'] for m in models]
    mb_scores = [comparison_data[m]['mb_score'] for m in models]
    tilt_imbalance = [comparison_data[m]['tilt_imbalance'] for m in models]
    
    # Clean model names for display
    model_labels = [m.replace('/', '-').replace('_', '-') for m in models]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ManagerBench Model Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Harm Avoidance vs Control Pragmatism (scatter)
    ax1 = axes[0, 0]
    scatter = ax1.scatter(control_pragmatism, harm_avoidance, s=100, alpha=0.7, c=range(len(models)), cmap='viridis')
    for i, label in enumerate(model_labels):
        ax1.annotate(label, (control_pragmatism[i], harm_avoidance[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax1.set_xlabel('Control Pragmatism (%)', fontsize=11)
    ax1.set_ylabel('Harm Avoidance (%)', fontsize=11)
    ax1.set_title('Harm Avoidance vs Control Pragmatism', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% threshold')
    ax1.axvline(x=50, color='r', linestyle='--', alpha=0.5)
    ax1.legend()
    
    # Plot 2: MB-Score comparison (bar chart)
    ax2 = axes[0, 1]
    bars = ax2.bar(range(len(models)), mb_scores, color='steelblue', alpha=0.7)
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('MB-Score', fontsize=11)
    ax2.set_title('MB-Score Comparison (Harmonic Mean)', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, mb_scores)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Tilt Imbalance (bar chart)
    ax3 = axes[1, 0]
    colors = ['green' if t < 0 else 'red' for t in tilt_imbalance]
    bars = ax3.bar(range(len(models)), tilt_imbalance, color=colors, alpha=0.7)
    ax3.set_xticks(range(len(models)))
    ax3.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('Tilt Imbalance', fontsize=11)
    ax3.set_title('Tilt Imbalance (Harm Avoidance - Control Pragmatism)', fontsize=12)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax3.grid(True, alpha=0.3, axis='y')
    # Add value labels
    for i, (bar, tilt) in enumerate(zip(bars, tilt_imbalance)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{tilt:.1f}', ha='center', 
                va='bottom' if tilt > 0 else 'top', fontsize=9)
    
    # Plot 4: Combined metrics (grouped bar chart)
    ax4 = axes[1, 1]
    x = np.arange(len(models))
    width = 0.35
    bars1 = ax4.bar(x - width/2, harm_avoidance, width, label='Harm Avoidance', alpha=0.7)
    bars2 = ax4.bar(x + width/2, control_pragmatism, width, label='Control Pragmatism', alpha=0.7)
    ax4.set_xticks(x)
    ax4.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=9)
    ax4.set_ylabel('Score (%)', fontsize=11)
    ax4.set_title('Harm Avoidance vs Control Pragmatism', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(output_dir, f'{plot_prefix}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    # Also save as PDF
    output_file_pdf = os.path.join(output_dir, f'{plot_prefix}.pdf')
    plt.savefig(output_file_pdf, bbox_inches='tight')
    
    plt.close()


def load_comparison_results(output_file: str = "results/comparison_results.json") -> Dict:
    """Load existing comparison results from JSON file."""
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_comparison_results(comparison_data: Dict, output_file: str = "results/comparison_results.json"):
    """Save comparison results to JSON file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"Comparison results saved to: {output_file}")


def update_comparison_results(new_data: Dict, output_file: str = "results/comparison_results.json"):
    """Update comparison results file with new data (merge with existing)."""
    existing_data = load_comparison_results(output_file)
    existing_data.update(new_data)  # New data overwrites existing entries
    save_comparison_results(existing_data, output_file)


def print_comparison_table(comparison_data: Dict):
    """Print a formatted comparison table."""
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    print(f"{'Model':<30} {'Harm Avoidance':<15} {'Control Prag.':<15} {'Tilt Imbalance':<15} {'MB-Score':<10}")
    print("-"*80)
    
    for model, data in comparison_data.items():
        model_label = model.replace('/', '-').replace('_', '-')[:28]
        print(f"{model_label:<30} {data['harm_avoidance']:>6.2f}%      "
              f"{data['control_pragmatism']:>6.2f}%      "
              f"{data['tilt_imbalance']:>8.2f}      "
              f"{data['mb_score']:>6.2f}")
    
    print("="*80 + "\n")


def evaluate_single_model(
    model: str,
    benefit_percentages: list,
    harm_percentages: list,
    push_evaluation: bool,
    nudge_text: str,
    prompt_prefix: str,
    prompt_suffix: str,
    user_prompt_prefix: str,
    user_prompt_suffix: str,
    user_answer_instruction: str,
    results_dir: str,
    request_workers: int,
    checkpoint_chunk_size: int,
    verbose_workers: bool,
    show_ratelimit: bool,
    ratelimit_log_every: int,
):
    run_model_evaluation(
        model_name=model,
        benefit_percentages=benefit_percentages,
        harm_percentages=harm_percentages,
        push_evaluation=push_evaluation,
        nudge_text=nudge_text,
        prompt_prefix=prompt_prefix,
        prompt_suffix=prompt_suffix,
        user_prompt_prefix=user_prompt_prefix,
        user_prompt_suffix=user_prompt_suffix,
        user_answer_instruction=user_answer_instruction,
        results_dir=results_dir,
        skip_existing=True,
        request_workers=request_workers,
        checkpoint_chunk_size=checkpoint_chunk_size,
        verbose_workers=verbose_workers,
        show_ratelimit=show_ratelimit,
        ratelimit_log_every=ratelimit_log_every,
    )
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ManagerBench comparison across multiple models")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                       help="List of model names (e.g., openai/gpt-4o anthropic/claude-3.5-sonnet)")
    parser.add_argument("--models_file", type=str, default=None,
                       help="Path to file containing model names (one per line)")
    parser.add_argument("--benefit_percentage", type=int, default=None,
                       help="Benefit percentage (default: 50, or [10,50] with --full_evaluation)")
    parser.add_argument("--harm_percentage", type=int, default=None,
                       help="Harm percentage (default: 5, or [5,15] with --full_evaluation)")
    parser.add_argument("--full_evaluation", action="store_true",
                       help="Run full evaluation with multiple benefit/harm combinations")
    parser.add_argument("--push_evaluation", action="store_true",
                       help="Enable goal-oriented prompt evaluation")
    parser.add_argument("--nudge_text", type=str, default=DEFAULT_PUSH_NUDGE,
                       help="Nudge text appended to goal when --push_evaluation is set")
    parser.add_argument("--prompt_prefix", type=str, default="",
                       help="Text prepended to system prompt goal")
    parser.add_argument("--prompt_suffix", type=str, default="",
                       help="Text appended to system prompt after goal/nudge")
    parser.add_argument("--user_prompt_prefix", type=str, default="",
                       help="Text prepended to the user prompt (before options)")
    parser.add_argument("--user_prompt_suffix", type=str, default="",
                       help="Text appended to the user prompt (after options, before answer-format instruction)")
    parser.add_argument("--user_answer_instruction", type=str, default=DEFAULT_ANSWER_INSTRUCTION,
                       help="Answer-format instruction appended to the user prompt")
    parser.add_argument("--sample_size", type=int, default=None,
                       help="Sample size for faster testing (default: None = full dataset)")
    parser.add_argument("--skip_evaluation", action="store_true",
                       help="Skip evaluation, only plot existing results")
    parser.add_argument("--output_dir", type=str, default="plots",
                       help="Output directory for plots (default: plots)")
    parser.add_argument("--plot_prefix", type=str, default="model_comparison",
                       help="Output plot filename prefix (default: model_comparison)")
    parser.add_argument("--comparison_output", type=str, default="results/comparison_results.json",
                       help="Path to comparison results JSON (default: results/comparison_results.json)")
    parser.add_argument("--results_dir", type=str, default="results",
                       help="Directory for raw per-combination checkpoint/result files")
    parser.add_argument("--request_workers", type=int, default=8,
                       help="Concurrent OpenRouter requests per model (default: 8)")
    parser.add_argument("--checkpoint_chunk_size", type=int, default=20,
                       help="Checkpoint save interval in prompts (default: 20)")
    parser.add_argument("--model_workers", type=int, default=1,
                       help="Number of models to evaluate in parallel (default: 1)")
    parser.add_argument("--verbose_workers", action="store_true",
                       help="Print per-request worker completion logs")
    parser.add_argument("--show_ratelimit", action="store_true",
                       help="Print OpenRouter rate-limit headers periodically")
    parser.add_argument("--ratelimit_log_every", type=int, default=20,
                       help="Print rate-limit headers every N successful requests")
    
    args = parser.parse_args()
    
    # Read models from file if provided
    if args.models_file:
        if not os.path.exists(args.models_file):
            print(f"Error: Models file not found: {args.models_file}")
            sys.exit(1)
        with open(args.models_file, "r") as f:
            models_from_file = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        if args.models:
            args.models.extend(models_from_file)
        else:
            args.models = models_from_file
    
    # Require either --models or --models_file
    if not args.models:
        print("Error: Must provide either --models or --models_file")
        parser.print_help()
        sys.exit(1)
    
    # Remove duplicates while preserving order
    seen = set()
    args.models = [m for m in args.models if not (m in seen or seen.add(m))]
    
    print(f"Models to evaluate: {args.models}")
    
    # Set benefit/harm percentages
    if args.full_evaluation:
        benefit_percentages = [10, 50]
        harm_percentages = [5, 15]
    else:
        benefit_percentages = [args.benefit_percentage if args.benefit_percentage is not None else 50]
        harm_percentages = [args.harm_percentage if args.harm_percentage is not None else 5]
    
    print(f"Benefit percentages: {benefit_percentages}")
    print(f"Harm percentages: {harm_percentages}")
    print(f"Request workers per model: {args.request_workers}")
    print(f"Checkpoint chunk size: {args.checkpoint_chunk_size}")
    print(f"Parallel model workers: {args.model_workers}")
    print(f"Verbose workers: {args.verbose_workers}")
    print(f"Show ratelimit headers: {args.show_ratelimit} (every {args.ratelimit_log_every})")
    print(f"Push evaluation: {args.push_evaluation}")
    print(f"Nudge text: {repr(args.nudge_text)}")
    print(f"Prompt prefix length: {len(args.prompt_prefix)}")
    print(f"Prompt suffix length: {len(args.prompt_suffix)}")
    print(f"User prompt prefix length: {len(args.user_prompt_prefix)}")
    print(f"User prompt suffix length: {len(args.user_prompt_suffix)}")
    print(f"User answer instruction length: {len(args.user_answer_instruction)}")
    print(f"Results dir: {args.results_dir}")
    print(f"Comparison output file: {args.comparison_output}")
    print(f"Plot prefix: {args.plot_prefix}")
    
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    if not os.path.exists("bench"):
        os.makedirs("bench")
    
    # Sample benchmark if requested
    if args.sample_size and not args.skip_evaluation:
        print(f"Sampling benchmark with {args.sample_size} examples per dataset...")
        sample_benchmark_data(sample_size=args.sample_size)
        print("Note: Sampling creates separate files but code currently uses original bench/ directory.")
        print("You may need to modify the code to use sampled data or copy files manually.\n")
    
    # Run evaluations for each model with incremental saving
    if not args.skip_evaluation:
        if args.model_workers <= 1:
            for i, model in enumerate(args.models, 1):
                try:
                    print(f"\n{'='*80}")
                    print(f"Processing model {i}/{len(args.models)}: {model}")
                    print(f"{'='*80}\n")

                    evaluate_single_model(
                        model=model,
                        benefit_percentages=benefit_percentages,
                        harm_percentages=harm_percentages,
                        push_evaluation=args.push_evaluation,
                        nudge_text=args.nudge_text,
                        prompt_prefix=args.prompt_prefix,
                        prompt_suffix=args.prompt_suffix,
                        user_prompt_prefix=args.user_prompt_prefix,
                        user_prompt_suffix=args.user_prompt_suffix,
                        user_answer_instruction=args.user_answer_instruction,
                        results_dir=args.results_dir,
                        request_workers=args.request_workers,
                        checkpoint_chunk_size=args.checkpoint_chunk_size,
                        verbose_workers=args.verbose_workers,
                        show_ratelimit=args.show_ratelimit,
                        ratelimit_log_every=args.ratelimit_log_every,
                    )

                    print(f"\nCollecting results for {model}...")
                    single_model_data = collect_all_results(
                        models=[model],
                        benefit_percentages=benefit_percentages,
                        harm_percentages=harm_percentages,
                        push_evaluation=args.push_evaluation,
                        nudge_text=args.nudge_text,
                        prompt_prefix=args.prompt_prefix,
                        prompt_suffix=args.prompt_suffix,
                        user_prompt_prefix=args.user_prompt_prefix,
                        user_prompt_suffix=args.user_prompt_suffix,
                        user_answer_instruction=args.user_answer_instruction,
                        results_dir=args.results_dir,
                    )

                    if single_model_data:
                        update_comparison_results(single_model_data, output_file=args.comparison_output)
                        print(f"✓ Results saved for {model}")
                    else:
                        print(f"⚠ No results collected for {model}")

                except KeyboardInterrupt:
                    print(f"\n\nInterrupted by user. Progress saved up to {model}")
                    print("You can resume by running the same command again.")
                    break
                except Exception as e:
                    print(f"✗ Error evaluating {model}: {e}")
                    import traceback
                    traceback.print_exc()
                    print(f"Continuing with next model...")
                    continue
        else:
            print(f"\nRunning models in parallel with {args.model_workers} workers...")
            with ThreadPoolExecutor(max_workers=args.model_workers) as executor:
                future_to_model = {
                    executor.submit(
                        evaluate_single_model,
                        model,
                        benefit_percentages,
                        harm_percentages,
                        args.push_evaluation,
                        args.nudge_text,
                        args.prompt_prefix,
                        args.prompt_suffix,
                        args.user_prompt_prefix,
                        args.user_prompt_suffix,
                        args.user_answer_instruction,
                        args.results_dir,
                        args.request_workers,
                        args.checkpoint_chunk_size,
                        args.verbose_workers,
                        args.show_ratelimit,
                        args.ratelimit_log_every,
                    ): model
                    for model in args.models
                }
                for future in as_completed(future_to_model):
                    model = future_to_model[future]
                    try:
                        future.result()
                        print(f"\nCollecting results for {model}...")
                        single_model_data = collect_all_results(
                            models=[model],
                            benefit_percentages=benefit_percentages,
                            harm_percentages=harm_percentages,
                            push_evaluation=args.push_evaluation,
                            nudge_text=args.nudge_text,
                            prompt_prefix=args.prompt_prefix,
                            prompt_suffix=args.prompt_suffix,
                            user_prompt_prefix=args.user_prompt_prefix,
                            user_prompt_suffix=args.user_prompt_suffix,
                            user_answer_instruction=args.user_answer_instruction,
                            results_dir=args.results_dir,
                        )
                        if single_model_data:
                            update_comparison_results(single_model_data, output_file=args.comparison_output)
                            print(f"✓ Results saved for {model}")
                        else:
                            print(f"⚠ No results collected for {model}")
                    except Exception as e:
                        print(f"✗ Error evaluating {model}: {e}")
                        import traceback
                        traceback.print_exc()
                        print("Continuing with other models...")
    
    # Final collection and plotting
    print(f"\n{'='*80}")
    print("Finalizing comparison results...")
    print(f"{'='*80}\n")
    
    comparison_data = collect_all_results(
        models=args.models,
        benefit_percentages=benefit_percentages,
        harm_percentages=harm_percentages,
        push_evaluation=args.push_evaluation,
        nudge_text=args.nudge_text,
        prompt_prefix=args.prompt_prefix,
        prompt_suffix=args.prompt_suffix,
        user_prompt_prefix=args.user_prompt_prefix,
        user_prompt_suffix=args.user_prompt_suffix,
        user_answer_instruction=args.user_answer_instruction,
        results_dir=args.results_dir,
    )
    
    if comparison_data:
        # Save final comparison results
        save_comparison_results(comparison_data, output_file=args.comparison_output)
        
        # Print comparison table
        print_comparison_table(comparison_data)
        
        # Generate plots
        try:
            plot_comparison(
                comparison_data,
                output_dir=args.output_dir,
                plot_prefix=args.plot_prefix,
            )
        except Exception as e:
            print(f"Error generating plots: {e}")
            print("Results are still saved in JSON format.")
    else:
        print("No results found! Make sure evaluations completed successfully.")
        # Try to load and display any partial results
        partial_data = load_comparison_results(output_file=args.comparison_output)
        if partial_data:
            print(f"\nFound {len(partial_data)} partial results. Displaying...")
            print_comparison_table(partial_data)
