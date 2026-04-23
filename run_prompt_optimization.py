#!/usr/bin/env python3
"""
CLI entry point for ManagerBench prompt optimisation.

Subcommands:
  extract-proxy    Extract discriminating proxy items from existing results
  validate-proxy   Validate proxy-vs-full correlation
  optimize         Run Bayesian optimisation for a model
  show-config      Show the prompt config for given parameter values
  preview-grid     Preview prompt configs across a parameter grid

Examples:
  # 1. Extract proxy items from existing spectrum data
  python run_prompt_optimization.py extract-proxy

  # 2. Validate proxy correlates with full benchmark
  python run_prompt_optimization.py validate-proxy

  # 3. Run single-objective optimisation (maximise MB-Score)
  python run_prompt_optimization.py optimize \\
      --model google/gemini-2.5-flash-lite --n_trials 20

  # 4. Run multi-objective optimisation (Pareto frontier)
  python run_prompt_optimization.py optimize \\
      --model google/gemini-2.5-flash-lite --multi_objective --n_trials 30

  # 5. Show what prompt a parameter point produces
  python run_prompt_optimization.py show-config --safety_weight 0.6 --goal_pressure 0.3

  # 6. Preview prompts across a grid
  python run_prompt_optimization.py preview-grid --safety_steps 5 --goal_steps 3
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cmd_extract_proxy(args: argparse.Namespace) -> None:
    from prompt_optimizer.proxy import extract_and_save_proxy

    extract_and_save_proxy(
        variants_dir=args.variants_dir,
        bench_dir=args.bench_dir,
        output_path=args.output,
        n_treatment=args.n_treatment,
        n_control=args.n_control,
        min_observations=args.min_observations,
    )


def cmd_validate_proxy(args: argparse.Namespace) -> None:
    from prompt_optimizer.proxy import validate_proxy_correlation

    correlations = validate_proxy_correlation(
        proxy_path=args.proxy_path,
        variants_dir=args.variants_dir,
    )
    print("\nProxy vs Full-Benchmark Correlations:")
    for metric, r in correlations.items():
        if r is not None:
            print(f"  {metric}: r = {r:.4f}")
        else:
            print(f"  {metric}: insufficient data")


def cmd_optimize(args: argparse.Namespace) -> None:
    if args.multi_objective:
        from prompt_optimizer.optimizer import run_multi_objective
        result = run_multi_objective(
            model_name=args.model,
            proxy_path=args.proxy_path,
            variants_dir=args.variants_dir,
            n_trials=args.n_trials,
            request_workers=args.request_workers,
            benefit_percentage=args.benefit_percentage,
            harm_percentage=args.harm_percentage,
            output_dir=args.output_dir,
            warm_start=not args.no_warm_start,
            seed=args.seed,
        )
        print(f"\nPareto-optimal configs found: {result['n_pareto']}")
        for i, cfg in enumerate(result["pareto_configs"]):
            print(f"  [{i+1}] HA={cfg['harm_avoidance']:.1f}%, "
                  f"CP={cfg['control_pragmatism']:.1f}%, "
                  f"MB={cfg['mb_score']:.1f} | "
                  f"sw={cfg['params']['safety_weight']:+.3f}, "
                  f"gp={cfg['params']['goal_pressure']:.3f}")
    else:
        from prompt_optimizer.optimizer import run_single_objective
        result = run_single_objective(
            model_name=args.model,
            proxy_path=args.proxy_path,
            variants_dir=args.variants_dir,
            n_trials=args.n_trials,
            request_workers=args.request_workers,
            benefit_percentage=args.benefit_percentage,
            harm_percentage=args.harm_percentage,
            output_dir=args.output_dir,
            warm_start=not args.no_warm_start,
            seed=args.seed,
        )
        print(f"\nBest MB-Score: {result['best_mb_score']:.2f}")
        print(f"  safety_weight:      {result['best_params']['safety_weight']:+.4f}")
        print(f"  goal_pressure:      {result['best_params']['goal_pressure']:.4f}")
        print(f"  harm_avoidance:     {result['best_harm_avoidance']:.1f}%")
        print(f"  control_pragmatism: {result['best_control_pragmatism']:.1f}%")
        print(f"\nGenerated prompt config:")
        for k, v in result["best_prompt_config"].items():
            print(f"  {k}: {repr(v)}")


def cmd_show_config(args: argparse.Namespace) -> None:
    from prompt_optimizer.param_space import generate_prompt_config

    config = generate_prompt_config(args.safety_weight, args.goal_pressure)
    print(f"Parameters: safety_weight={args.safety_weight:+.4f}, goal_pressure={args.goal_pressure:.4f}")
    print()
    for k, v in config.items():
        print(f"  {k}: {repr(v)}")


def cmd_preview_grid(args: argparse.Namespace) -> None:
    from prompt_optimizer.param_space import generate_prompt_config, enumerate_grid

    points = enumerate_grid(args.safety_steps, args.goal_steps)
    seen_prefixes = set()
    print(f"Grid: {args.safety_steps} x {args.goal_steps} = {len(points)} points\n")

    for pt in points:
        cfg = generate_prompt_config(pt["safety_weight"], pt["goal_pressure"])
        prefix_short = cfg["prompt_prefix"][:80] + "..." if len(cfg["prompt_prefix"]) > 80 else cfg["prompt_prefix"]
        push = "PUSH" if cfg["push_evaluation"] else "no-push"
        nudge_short = cfg["nudge_text"][:50] + "..." if len(cfg["nudge_text"]) > 50 else cfg["nudge_text"]

        label = f"sw={pt['safety_weight']:+.2f} gp={pt['goal_pressure']:.2f}"
        print(f"  {label}  [{push}] prefix={repr(prefix_short)} nudge={repr(nudge_short)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ManagerBench Prompt Optimisation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- extract-proxy ---
    p_extract = subparsers.add_parser("extract-proxy", help="Extract proxy items from existing results")
    p_extract.add_argument("--variants_dir", default="results/variants")
    p_extract.add_argument("--bench_dir", default="bench")
    p_extract.add_argument("--output", default="bench_proxy/proxy_items.json")
    p_extract.add_argument("--n_treatment", type=int, default=120)
    p_extract.add_argument("--n_control", type=int, default=60)
    p_extract.add_argument("--min_observations", type=int, default=4)

    # --- validate-proxy ---
    p_validate = subparsers.add_parser("validate-proxy", help="Validate proxy vs full correlation")
    p_validate.add_argument("--proxy_path", default="bench_proxy/proxy_items.json")
    p_validate.add_argument("--variants_dir", default="results/variants")

    # --- optimize ---
    p_opt = subparsers.add_parser("optimize", help="Run Bayesian optimisation")
    p_opt.add_argument("--model", required=True, help="OpenRouter model ID")
    p_opt.add_argument("--proxy_path", default="bench_proxy/proxy_items.json")
    p_opt.add_argument("--variants_dir", default="results/variants")
    p_opt.add_argument("--n_trials", type=int, default=30, help="Number of optimisation trials")
    p_opt.add_argument("--request_workers", type=int, default=8)
    p_opt.add_argument("--benefit_percentage", type=int, default=10)
    p_opt.add_argument("--harm_percentage", type=int, default=5)
    p_opt.add_argument("--output_dir", default="results/optimization")
    p_opt.add_argument("--multi_objective", action="store_true",
                       help="Maximise both harm_avoidance and control_pragmatism (Pareto)")
    p_opt.add_argument("--no_warm_start", action="store_true",
                       help="Do not seed from existing spectrum results")
    p_opt.add_argument("--seed", type=int, default=42)

    # --- show-config ---
    p_show = subparsers.add_parser("show-config", help="Show prompt config for parameters")
    p_show.add_argument("--safety_weight", type=float, required=True)
    p_show.add_argument("--goal_pressure", type=float, required=True)

    # --- preview-grid ---
    p_grid = subparsers.add_parser("preview-grid", help="Preview prompts across parameter grid")
    p_grid.add_argument("--safety_steps", type=int, default=5)
    p_grid.add_argument("--goal_steps", type=int, default=3)

    args = parser.parse_args()

    dispatch = {
        "extract-proxy": cmd_extract_proxy,
        "validate-proxy": cmd_validate_proxy,
        "optimize": cmd_optimize,
        "show-config": cmd_show_config,
        "preview-grid": cmd_preview_grid,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
