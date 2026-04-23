"""
Prompt Optimizer for ManagerBench.

Treats prompt design as a black-box optimization problem: define a continuous
parameter space (safety_weight, goal_pressure), use template interpolation to
convert parameters to natural-language prompts, evaluate cheaply on a proxy
subset of discriminating items, and search with Bayesian optimization.
"""
