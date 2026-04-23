#!/usr/bin/env python3
"""
Create prompt_variants.optimized.json from optimization results.

Reads the best configs from single-objective optimization results and
generates a variants file in the same format as prompt_variants.spectrum.json,
ready for full-benchmark validation via run_prompt_variants.py.

Usage:
  python create_optimized_variants.py [--optimization_dir results/optimization] [--output prompt_variants.optimized.json]

  # Then validate:
  python run_prompt_variants.py --variants_file prompt_variants.optimized.json --models_file models.txt --benefit_percentage 10 --harm_percentage 5
"""

import argparse
import json
import os
import sys
from typing import Dict, List


def load_best_configs(optimization_dir: str) -> List[Dict]:
    """Load best configs from all single-objective optimization results."""
    configs = []
    for fname in sorted(os.listdir(optimization_dir)):
        if not (fname.startswith("optimization_") and fname.endswith(".json")):
            continue
        fpath = os.path.join(optimization_dir, fname)
        with open(fpath, "r") as f:
            data = json.load(f)

        model = data.get("model", "unknown")
        configs.append({
            "model": model,
            "params": data.get("best_params", {}),
            "mb_score": data.get("best_mb_score", 0),
            "harm_avoidance": data.get("best_harm_avoidance", 0),
            "control_pragmatism": data.get("best_control_pragmatism", 0),
            "prompt_config": data.get("best_prompt_config", {}),
        })
    return configs


def select_top_variants(configs: List[Dict], max_variants: int = 5) -> List[Dict]:
    """Select top unique configs for full-benchmark validation.

    Strategy:
    1. Best overall MB-Score config
    2. Best per-model configs (if different from #1)
    3. Deduplicate configs that map to the same prompt (similar parameters)
    """
    if not configs:
        return []

    # Sort by MB-Score descending
    sorted_configs = sorted(configs, key=lambda c: c["mb_score"], reverse=True)

    selected = []
    seen_params = set()

    for cfg in sorted_configs:
        # Round params to avoid near-duplicates
        sw = round(cfg["params"].get("safety_weight", 0), 2)
        gp = round(cfg["params"].get("goal_pressure", 0), 2)
        param_key = (sw, gp)

        if param_key in seen_params:
            continue
        seen_params.add(param_key)

        selected.append(cfg)
        if len(selected) >= max_variants:
            break

    return selected


def config_to_variant(cfg: Dict, index: int) -> Dict:
    """Convert an optimization config to a prompt variant definition."""
    prompt_config = cfg.get("prompt_config", {})
    sw = cfg["params"].get("safety_weight", 0)
    gp = cfg["params"].get("goal_pressure", 0)
    model_short = cfg.get("model", "unknown").split("/")[-1]

    return {
        "name": f"optimized_v{index}_{model_short}",
        "description": (
            f"Optimized config from {model_short}: "
            f"sw={sw:+.2f}, gp={gp:.2f}, "
            f"proxy MB-Score={cfg.get('mb_score', 0):.1f}"
        ),
        "push_evaluation": prompt_config.get("push_evaluation", False),
        "nudge_text": prompt_config.get("nudge_text", ""),
        "prompt_prefix": prompt_config.get("prompt_prefix", ""),
        "prompt_suffix": prompt_config.get("prompt_suffix", ""),
        "user_prompt_prefix": prompt_config.get("user_prompt_prefix", ""),
        "user_prompt_suffix": prompt_config.get("user_prompt_suffix", ""),
        "_optimization_metadata": {
            "source_model": cfg.get("model"),
            "safety_weight": sw,
            "goal_pressure": gp,
            "proxy_mb_score": cfg.get("mb_score"),
            "proxy_harm_avoidance": cfg.get("harm_avoidance"),
            "proxy_control_pragmatism": cfg.get("control_pragmatism"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create optimized variants file from optimization results"
    )
    parser.add_argument("--optimization_dir", default="results/optimization",
                        help="Directory with optimization_*.json files")
    parser.add_argument("--output", default="prompt_variants.optimized.json",
                        help="Output variants file path")
    parser.add_argument("--max_variants", type=int, default=5,
                        help="Maximum number of variants to include")
    args = parser.parse_args()

    if not os.path.isdir(args.optimization_dir):
        print(f"Error: optimization directory not found: {args.optimization_dir}")
        print("Run optimization first (see TODO_OPTIMIZATION.md)")
        sys.exit(1)

    print(f"Loading optimization results from {args.optimization_dir}...")
    configs = load_best_configs(args.optimization_dir)

    if not configs:
        print("No optimization results found.")
        sys.exit(1)

    print(f"Found {len(configs)} model optimization results:")
    for cfg in configs:
        model_short = cfg["model"].split("/")[-1]
        print(f"  {model_short}: MB={cfg['mb_score']:.1f}, "
              f"HA={cfg['harm_avoidance']:.1f}%, CP={cfg['control_pragmatism']:.1f}%, "
              f"sw={cfg['params'].get('safety_weight', 0):+.3f}, "
              f"gp={cfg['params'].get('goal_pressure', 0):.3f}")

    top = select_top_variants(configs, max_variants=args.max_variants)
    print(f"\nSelected {len(top)} unique configs for validation:")

    variants = []
    for i, cfg in enumerate(top, 1):
        variant = config_to_variant(cfg, i)
        variants.append(variant)
        print(f"  [{i}] {variant['name']}: {variant['description']}")

    output = {"variants": variants}
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")
    print(f"\nNext step:")
    print(f"  python run_prompt_variants.py \\")
    print(f"      --variants_file {args.output} \\")
    print(f"      --models_file models.txt \\")
    print(f"      --benefit_percentage 10 --harm_percentage 5 \\")
    print(f"      --results_root results/variants --output_dir plots \\")
    print(f"      --request_workers 8 --continue_on_error")


if __name__ == "__main__":
    main()
