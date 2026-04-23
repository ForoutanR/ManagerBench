"""
Bayesian optimization over the ManagerBench prompt parameter space.

Uses Optuna when available (TPE sampler). Falls back to a numpy-only
surrogate-assisted random search when Optuna is not installed.

Supports:
  - Warm-starting from existing spectrum experiment results
  - Single-objective optimisation (maximise MB-Score)
  - Multi-objective optimisation (maximise harm_avoidance AND control_pragmatism)
  - Convergence plots and Pareto-frontier visualisation
"""

from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import optuna
    from optuna.trial import create_trial
    from optuna.distributions import FloatDistribution
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prompt_optimizer.param_space import (
    SPECTRUM_TO_PARAMS,
    generate_prompt_config,
)
from prompt_optimizer.evaluator import ProxyEvaluator


# ---------------------------------------------------------------------------
# Warm-start: load existing spectrum results as prior observations
# ---------------------------------------------------------------------------

def load_warm_start_data(
    variants_dir: str = "results/variants",
    target_model: Optional[str] = None,
) -> List[Dict]:
    """
    Load metric data from existing spectrum variants.

    Returns list of dicts with keys:
        safety_weight, goal_pressure, harm_avoidance, control_pragmatism, mb_score, variant
    """
    observations = []

    for variant_slug, params in SPECTRUM_TO_PARAMS.items():
        comp_path = os.path.join(variants_dir, variant_slug, "comparison_results.json")
        if not os.path.exists(comp_path):
            continue
        try:
            with open(comp_path, "r") as f:
                comp = json.load(f)
        except Exception:
            continue

        for model, metrics in comp.items():
            if target_model and model != target_model:
                continue
            if not isinstance(metrics, dict):
                continue
            obs = {
                "safety_weight": params["safety_weight"],
                "goal_pressure": params["goal_pressure"],
                "harm_avoidance": metrics.get("harm_avoidance", 0.0),
                "control_pragmatism": metrics.get("control_pragmatism", 0.0),
                "mb_score": metrics.get("mb_score", 0.0),
                "variant": variant_slug,
                "model": model,
            }
            observations.append(obs)

    return observations


# ===================================================================
# Backend: Optuna (preferred)
# ===================================================================

def _seed_study_single(study, warm_data: List[Dict]) -> int:
    added = 0
    for obs in warm_data:
        try:
            study.add_trial(
                create_trial(
                    params={"safety_weight": obs["safety_weight"], "goal_pressure": obs["goal_pressure"]},
                    distributions={
                        "safety_weight": FloatDistribution(-1.0, 1.0),
                        "goal_pressure": FloatDistribution(0.0, 1.0),
                    },
                    values=[obs["mb_score"]],
                )
            )
            added += 1
        except Exception as e:
            print(f"  Warning: could not add warm-start trial: {e}")
    return added


def _seed_study_multi(study, warm_data: List[Dict]) -> int:
    added = 0
    for obs in warm_data:
        try:
            study.add_trial(
                create_trial(
                    params={"safety_weight": obs["safety_weight"], "goal_pressure": obs["goal_pressure"]},
                    distributions={
                        "safety_weight": FloatDistribution(-1.0, 1.0),
                        "goal_pressure": FloatDistribution(0.0, 1.0),
                    },
                    values=[obs["harm_avoidance"], obs["control_pragmatism"]],
                )
            )
            added += 1
        except Exception as e:
            print(f"  Warning: could not add warm-start trial: {e}")
    return added


def _run_optuna_single(
    evaluator: ProxyEvaluator,
    n_trials: int,
    warm_data: List[Dict],
    seed: int,
) -> Dict:
    """Run single-objective with Optuna TPE."""
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    n_warm = _seed_study_single(study, warm_data) if warm_data else 0
    if n_warm:
        print(f"  Warm-started with {n_warm} existing observations")

    eval_count = {"n": 0}

    def objective(trial):
        sw = trial.suggest_float("safety_weight", -1.0, 1.0)
        gp = trial.suggest_float("goal_pressure", 0.0, 1.0)
        config = generate_prompt_config(sw, gp)
        eval_count["n"] += 1
        print(f"  Trial {eval_count['n']}/{n_trials}: sw={sw:+.3f}, gp={gp:.3f}")
        metrics = evaluator.evaluate(config)
        print(f"    -> HA={metrics['harm_avoidance']:.1f}%, CP={metrics['control_pragmatism']:.1f}%, MB={metrics['mb_score']:.1f}")
        trial.set_user_attr("harm_avoidance", metrics["harm_avoidance"])
        trial.set_user_attr("control_pragmatism", metrics["control_pragmatism"])
        trial.set_user_attr("tilt_imbalance", metrics["tilt_imbalance"])
        trial.set_user_attr("prompt_config", config)
        return metrics["mb_score"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    trials_list = []
    for t in study.trials:
        trials_list.append({
            "number": t.number,
            "params": t.params,
            "value": t.value,
            "harm_avoidance": t.user_attrs.get("harm_avoidance"),
            "control_pragmatism": t.user_attrs.get("control_pragmatism"),
        })

    return {
        "best_params": best.params,
        "best_mb_score": best.value,
        "best_harm_avoidance": best.user_attrs.get("harm_avoidance"),
        "best_control_pragmatism": best.user_attrs.get("control_pragmatism"),
        "n_warm_start": n_warm,
        "all_trials": trials_list,
        "_study": study,
    }


def _run_optuna_multi(
    evaluator: ProxyEvaluator,
    n_trials: int,
    warm_data: List[Dict],
    seed: int,
) -> Dict:
    """Run multi-objective with Optuna TPE."""
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(directions=["maximize", "maximize"], sampler=sampler)

    n_warm = _seed_study_multi(study, warm_data) if warm_data else 0
    if n_warm:
        print(f"  Warm-started with {n_warm} existing observations")

    eval_count = {"n": 0}

    def objective(trial):
        sw = trial.suggest_float("safety_weight", -1.0, 1.0)
        gp = trial.suggest_float("goal_pressure", 0.0, 1.0)
        config = generate_prompt_config(sw, gp)
        eval_count["n"] += 1
        print(f"  Trial {eval_count['n']}/{n_trials}: sw={sw:+.3f}, gp={gp:.3f}")
        metrics = evaluator.evaluate(config)
        print(f"    -> HA={metrics['harm_avoidance']:.1f}%, CP={metrics['control_pragmatism']:.1f}%")
        trial.set_user_attr("mb_score", metrics["mb_score"])
        trial.set_user_attr("prompt_config", config)
        return metrics["harm_avoidance"], metrics["control_pragmatism"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    pareto_configs = []
    for t in study.best_trials:
        pareto_configs.append({
            "params": t.params,
            "harm_avoidance": t.values[0],
            "control_pragmatism": t.values[1],
            "mb_score": t.user_attrs.get("mb_score"),
            "prompt_config": t.user_attrs.get("prompt_config"),
        })

    trials_list = []
    for t in study.trials:
        trials_list.append({
            "number": t.number,
            "params": t.params,
            "values": list(t.values) if t.values else None,
            "mb_score": t.user_attrs.get("mb_score"),
        })

    return {
        "pareto_configs": pareto_configs,
        "n_pareto": len(pareto_configs),
        "n_warm_start": n_warm,
        "all_trials": trials_list,
        "_study": study,
    }


# ===================================================================
# Backend: numpy-only fallback (no external optimisation library)
# ===================================================================

class _SurrogateOptimizer:
    """
    Minimal surrogate-assisted optimiser using RBF interpolation.

    Uses warm-start data + Latin Hypercube exploration + greedy exploitation
    near the current best. No external dependencies beyond numpy.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.X: List[np.ndarray] = []  # observed params [[sw, gp], ...]
        self.Y: List[float] = []       # observed values

    def seed(self, warm_data: List[Dict], value_key: str = "mb_score") -> int:
        added = 0
        for obs in warm_data:
            self.X.append(np.array([obs["safety_weight"], obs["goal_pressure"]]))
            self.Y.append(obs[value_key])
            added += 1
        return added

    def _rbf_predict(self, x: np.ndarray) -> float:
        """Simple RBF-kernel weighted average prediction."""
        if not self.X:
            return 0.0
        X_arr = np.array(self.X)
        Y_arr = np.array(self.Y)
        # Squared distances, scaled
        dists = np.sum((X_arr - x) ** 2, axis=1)
        eps = 0.05  # length-scale
        weights = np.exp(-dists / (2 * eps ** 2))
        total_w = weights.sum()
        if total_w < 1e-12:
            return float(Y_arr.mean())
        return float(np.dot(weights, Y_arr) / total_w)

    def suggest(self) -> np.ndarray:
        """Suggest next point: mix of exploration and exploitation."""
        n = len(self.X)

        # First 5 trials: Latin Hypercube exploration
        if n < 5:
            sw = self.rng.uniform(-1.0, 1.0)
            gp = self.rng.uniform(0.0, 1.0)
            return np.array([sw, gp])

        # After that: 50% exploit near best, 50% explore
        if self.rng.random() < 0.5:
            # Exploit: perturb best point
            best_idx = int(np.argmax(self.Y))
            best_x = self.X[best_idx]
            noise = self.rng.normal(0, 0.15, size=2)
            x = best_x + noise
            x[0] = np.clip(x[0], -1.0, 1.0)
            x[1] = np.clip(x[1], 0.0, 1.0)
            return x
        else:
            # Explore: pick from candidates with highest predicted value + uncertainty bonus
            candidates = []
            for _ in range(50):
                sw = self.rng.uniform(-1.0, 1.0)
                gp = self.rng.uniform(0.0, 1.0)
                candidates.append(np.array([sw, gp]))

            scores = []
            for c in candidates:
                pred = self._rbf_predict(c)
                # Distance-based uncertainty bonus (explore far from observed)
                X_arr = np.array(self.X)
                min_dist = np.min(np.sum((X_arr - c) ** 2, axis=1))
                uncertainty = np.sqrt(min_dist) * 10
                scores.append(pred + uncertainty)

            best_candidate = candidates[int(np.argmax(scores))]
            return best_candidate

    def observe(self, x: np.ndarray, y: float) -> None:
        self.X.append(x.copy())
        self.Y.append(y)


def _run_numpy_single(
    evaluator: ProxyEvaluator,
    n_trials: int,
    warm_data: List[Dict],
    seed: int,
) -> Dict:
    """Run single-objective with numpy fallback."""
    opt = _SurrogateOptimizer(seed=seed)
    n_warm = opt.seed(warm_data) if warm_data else 0
    if n_warm:
        print(f"  Warm-started with {n_warm} existing observations")

    all_trials = []
    # Include warm-start as trials
    for i, obs in enumerate(warm_data or []):
        all_trials.append({
            "number": i,
            "params": {"safety_weight": obs["safety_weight"], "goal_pressure": obs["goal_pressure"]},
            "value": obs["mb_score"],
            "harm_avoidance": obs["harm_avoidance"],
            "control_pragmatism": obs["control_pragmatism"],
        })

    best_mb = max((t["value"] for t in all_trials), default=-1)
    best_trial = max(all_trials, key=lambda t: t["value"]) if all_trials else None

    for i in range(n_trials):
        x = opt.suggest()
        sw, gp = float(x[0]), float(x[1])
        config = generate_prompt_config(sw, gp)
        print(f"  Trial {i+1}/{n_trials}: sw={sw:+.3f}, gp={gp:.3f}")

        metrics = evaluator.evaluate(config)
        mb = metrics["mb_score"]
        print(f"    -> HA={metrics['harm_avoidance']:.1f}%, CP={metrics['control_pragmatism']:.1f}%, MB={mb:.1f}")

        opt.observe(x, mb)

        trial_data = {
            "number": n_warm + i,
            "params": {"safety_weight": sw, "goal_pressure": gp},
            "value": mb,
            "harm_avoidance": metrics["harm_avoidance"],
            "control_pragmatism": metrics["control_pragmatism"],
        }
        all_trials.append(trial_data)
        if mb > best_mb:
            best_mb = mb
            best_trial = trial_data

    return {
        "best_params": best_trial["params"] if best_trial else {},
        "best_mb_score": best_mb,
        "best_harm_avoidance": best_trial["harm_avoidance"] if best_trial else None,
        "best_control_pragmatism": best_trial["control_pragmatism"] if best_trial else None,
        "n_warm_start": n_warm,
        "all_trials": all_trials,
        "_study": None,
    }


def _run_numpy_multi(
    evaluator: ProxyEvaluator,
    n_trials: int,
    warm_data: List[Dict],
    seed: int,
) -> Dict:
    """Run multi-objective with numpy fallback (random search + Pareto filter)."""
    rng = np.random.RandomState(seed)

    all_trials = []
    for i, obs in enumerate(warm_data or []):
        all_trials.append({
            "number": i,
            "params": {"safety_weight": obs["safety_weight"], "goal_pressure": obs["goal_pressure"]},
            "values": [obs["harm_avoidance"], obs["control_pragmatism"]],
            "mb_score": obs["mb_score"],
        })

    n_warm = len(warm_data) if warm_data else 0
    if n_warm:
        print(f"  Warm-started with {n_warm} existing observations")

    for i in range(n_trials):
        sw = float(rng.uniform(-1.0, 1.0))
        gp = float(rng.uniform(0.0, 1.0))
        config = generate_prompt_config(sw, gp)
        print(f"  Trial {i+1}/{n_trials}: sw={sw:+.3f}, gp={gp:.3f}")

        metrics = evaluator.evaluate(config)
        print(f"    -> HA={metrics['harm_avoidance']:.1f}%, CP={metrics['control_pragmatism']:.1f}%")

        all_trials.append({
            "number": n_warm + i,
            "params": {"safety_weight": sw, "goal_pressure": gp},
            "values": [metrics["harm_avoidance"], metrics["control_pragmatism"]],
            "mb_score": metrics["mb_score"],
        })

    # Compute Pareto front
    pareto_configs = _compute_pareto(all_trials)

    return {
        "pareto_configs": pareto_configs,
        "n_pareto": len(pareto_configs),
        "n_warm_start": n_warm,
        "all_trials": all_trials,
        "_study": None,
    }


def _compute_pareto(trials: List[Dict]) -> List[Dict]:
    """Extract Pareto-optimal trials (maximising both objectives)."""
    valid = [t for t in trials if t.get("values") and len(t["values"]) == 2]
    if not valid:
        return []

    pareto = []
    for t in valid:
        dominated = False
        for other in valid:
            if other is t:
                continue
            if (other["values"][0] >= t["values"][0] and
                other["values"][1] >= t["values"][1] and
                (other["values"][0] > t["values"][0] or other["values"][1] > t["values"][1])):
                dominated = True
                break
        if not dominated:
            pareto.append({
                "params": t["params"],
                "harm_avoidance": t["values"][0],
                "control_pragmatism": t["values"][1],
                "mb_score": t.get("mb_score"),
                "prompt_config": generate_prompt_config(
                    t["params"]["safety_weight"], t["params"]["goal_pressure"]
                ),
            })
    return pareto


# ===================================================================
# Public API
# ===================================================================

def run_single_objective(
    model_name: str,
    proxy_path: str = "bench_proxy/proxy_items.json",
    variants_dir: str = "results/variants",
    n_trials: int = 30,
    request_workers: int = 8,
    benefit_percentage: int = 10,
    harm_percentage: int = 5,
    output_dir: str = "results/optimization",
    warm_start: bool = True,
    seed: int = 42,
) -> Dict:
    """Optimise MB-Score over (safety_weight, goal_pressure) for a given model."""
    os.makedirs(output_dir, exist_ok=True)

    evaluator = ProxyEvaluator(
        model_name=model_name,
        proxy_path=proxy_path,
        benefit_percentage=benefit_percentage,
        harm_percentage=harm_percentage,
        request_workers=request_workers,
    )

    warm_data = load_warm_start_data(variants_dir, target_model=model_name) if warm_start else []

    backend = "optuna" if HAS_OPTUNA else "numpy"
    print(f"\nStarting single-objective optimisation ({n_trials} trials, backend={backend}) for {model_name}")

    if HAS_OPTUNA:
        raw = _run_optuna_single(evaluator, n_trials, warm_data, seed)
    else:
        raw = _run_numpy_single(evaluator, n_trials, warm_data, seed)

    best_config = generate_prompt_config(
        raw["best_params"].get("safety_weight", 0),
        raw["best_params"].get("goal_pressure", 0),
    )

    result = {
        "model": model_name,
        "mode": "single_objective",
        "backend": backend,
        "best_params": raw["best_params"],
        "best_mb_score": raw["best_mb_score"],
        "best_harm_avoidance": raw["best_harm_avoidance"],
        "best_control_pragmatism": raw["best_control_pragmatism"],
        "best_prompt_config": best_config,
        "n_trials": n_trials,
        "n_warm_start": raw["n_warm_start"],
        "all_trials": raw["all_trials"],
    }

    result_path = os.path.join(output_dir, f"optimization_{model_name.replace('/', '_')}.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {result_path}")

    _plot_convergence_from_trials(raw["all_trials"], output_dir, model_name, raw["n_warm_start"])
    _plot_landscape_from_trials(raw["all_trials"], output_dir, model_name)

    return result


def run_multi_objective(
    model_name: str,
    proxy_path: str = "bench_proxy/proxy_items.json",
    variants_dir: str = "results/variants",
    n_trials: int = 30,
    request_workers: int = 8,
    benefit_percentage: int = 10,
    harm_percentage: int = 5,
    output_dir: str = "results/optimization",
    warm_start: bool = True,
    seed: int = 42,
) -> Dict:
    """Multi-objective: maximise both harm_avoidance and control_pragmatism (Pareto)."""
    os.makedirs(output_dir, exist_ok=True)

    evaluator = ProxyEvaluator(
        model_name=model_name,
        proxy_path=proxy_path,
        benefit_percentage=benefit_percentage,
        harm_percentage=harm_percentage,
        request_workers=request_workers,
    )

    warm_data = load_warm_start_data(variants_dir, target_model=model_name) if warm_start else []

    backend = "optuna" if HAS_OPTUNA else "numpy"
    print(f"\nStarting multi-objective optimisation ({n_trials} trials, backend={backend}) for {model_name}")

    if HAS_OPTUNA:
        raw = _run_optuna_multi(evaluator, n_trials, warm_data, seed)
    else:
        raw = _run_numpy_multi(evaluator, n_trials, warm_data, seed)

    result = {
        "model": model_name,
        "mode": "multi_objective",
        "backend": backend,
        "pareto_configs": raw["pareto_configs"],
        "n_pareto": raw["n_pareto"],
        "n_trials": n_trials,
        "n_warm_start": raw["n_warm_start"],
        "all_trials": raw["all_trials"],
    }

    result_path = os.path.join(output_dir, f"pareto_{model_name.replace('/', '_')}.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {result_path}")

    _plot_pareto_from_trials(raw["all_trials"], raw["pareto_configs"], output_dir, model_name)

    return result


# ===================================================================
# Backend-agnostic plotting (works with trial dicts, no optuna objects)
# ===================================================================

def _plot_convergence_from_trials(
    trials: List[Dict], output_dir: str, model_name: str, n_warm: int = 0,
) -> None:
    valid = [t for t in trials if t.get("value") is not None]
    if not valid:
        return

    numbers = list(range(len(valid)))
    values = [t["value"] for t in valid]
    best_so_far = []
    current_best = -float("inf")
    for v in values:
        current_best = max(current_best, v)
        best_so_far.append(current_best)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(numbers, values, alpha=0.5, s=20, label="Trial MB-Score")
    ax.plot(numbers, best_so_far, "r-", linewidth=2, label="Best so far")
    if n_warm > 0:
        ax.axvline(x=n_warm - 0.5, color="grey", linestyle="--", alpha=0.5, label=f"Warm-start ({n_warm})")
    ax.set_xlabel("Trial")
    ax.set_ylabel("MB-Score")
    ax.set_title(f"Optimisation Convergence — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, f"convergence_{model_name.replace('/', '_')}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Convergence plot: {path}")


def _plot_landscape_from_trials(
    trials: List[Dict], output_dir: str, model_name: str,
) -> None:
    valid = [t for t in trials if t.get("value") is not None and t.get("params")]
    if not valid:
        return

    sw = [t["params"]["safety_weight"] for t in valid]
    gp = [t["params"]["goal_pressure"] for t in valid]
    mb = [t["value"] for t in valid]

    best_idx = int(np.argmax(mb))

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(sw, gp, c=mb, cmap="RdYlGn", s=60, edgecolors="k", linewidths=0.5)
    plt.colorbar(scatter, ax=ax, label="MB-Score")
    ax.scatter([sw[best_idx]], [gp[best_idx]], marker="*", s=300, c="blue",
               edgecolors="k", linewidths=1, zorder=5, label=f"Best (MB={mb[best_idx]:.1f})")
    ax.set_xlabel("Safety Weight")
    ax.set_ylabel("Goal Pressure")
    ax.set_title(f"Parameter Landscape — {model_name}")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, f"landscape_{model_name.replace('/', '_')}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Landscape plot: {path}")


def _plot_pareto_from_trials(
    trials: List[Dict], pareto_configs: List[Dict], output_dir: str, model_name: str,
) -> None:
    valid = [t for t in trials if t.get("values") and len(t["values"]) == 2]
    if not valid:
        return

    ha = [t["values"][0] for t in valid]
    cp = [t["values"][1] for t in valid]
    p_ha = [p["harm_avoidance"] for p in pareto_configs]
    p_cp = [p["control_pragmatism"] for p in pareto_configs]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(cp, ha, alpha=0.4, s=30, label="All trials")
    if p_cp:
        ax.scatter(p_cp, p_ha, c="red", s=80, marker="D", edgecolors="k",
                   label=f"Pareto front ({len(p_ha)})")
        frontier = sorted(zip(p_cp, p_ha))
        ax.plot([p[0] for p in frontier], [p[1] for p in frontier], "r--", alpha=0.7)

    ax.set_xlabel("Control Pragmatism (%)")
    ax.set_ylabel("Harm Avoidance (%)")
    ax.set_title(f"Pareto Frontier — {model_name}")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, f"pareto_{model_name.replace('/', '_')}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Pareto plot: {path}")
