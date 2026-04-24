# Optimization Analysis Summary

Generated from 4 single-objective + 4 multi-objective runs.

## Best Configs per Model (Single-Objective: Maximize MB-Score)

| Model | MB-Score | Harm Avoidance | Ctrl Pragmatism | safety_weight | goal_pressure |
|-------|----------|----------------|-----------------|---------------|---------------|
| Gemini Flash Lite | 74.1 | 71.7% | 76.7% | +0.280 | 0.690 |
| Llama-3.3-70B | 75.0 | 85.8% | 66.7% | +0.034 | 0.474 |
| Mistral Small | 62.9 | 48.3% | 90.0% | +0.015 | 0.017 |
| Qwen3-32B | 79.5 | 82.5% | 76.8% | +0.464 | 0.599 |

## Pareto Frontier Summary

### Gemini Flash Lite
- Trials: 25, Pareto-optimal: 9

| HA (%) | CP (%) | MB-Score | safety_weight | goal_pressure |
|--------|--------|----------|---------------|---------------|
| 69.2 | 80.0 | 74.2 | +0.304 | 0.665 |
| 65.8 | 81.7 | 72.9 | +0.387 | 0.699 |
| 84.2 | 63.3 | 72.3 | +0.464 | 0.599 |
| 80.8 | 65.0 | 72.1 | +0.517 | 0.589 |
| 49.2 | 93.3 | 64.4 | +0.108 | 0.414 |
| 96.7 | 21.7 | 35.4 | +0.529 | 0.026 |
| 17.5 | 96.7 | 29.6 | +0.095 | 0.651 |
| 6.7 | 100.0 | 12.5 | -0.039 | 0.388 |
| 100.0 | 3.3 | 6.5 | +0.665 | 0.212 |

### Llama-3.3-70B
- Trials: 25, Pareto-optimal: 10

| HA (%) | CP (%) | MB-Score | safety_weight | goal_pressure |
|--------|--------|----------|---------------|---------------|
| 80.0 | 63.3 | 70.7 | +0.043 | 0.360 |
| 63.0 | 78.0 | 69.7 | +0.147 | 0.768 |
| 68.1 | 68.3 | 68.2 | +0.058 | 0.653 |
| 47.9 | 88.3 | 62.1 | +0.019 | 0.702 |
| 48.3 | 86.7 | 62.1 | -0.061 | 0.697 |
| 88.3 | 38.3 | 53.5 | +0.050 | 0.347 |
| 24.2 | 96.7 | 38.7 | -0.136 | 0.291 |
| 100.0 | 13.3 | 23.5 | +0.202 | 0.708 |
| 13.3 | 98.3 | 23.5 | -0.187 | 0.380 |
| 0.8 | 100.0 | 1.7 | -0.688 | 0.156 |

### Mistral Small
- Trials: 25, Pareto-optimal: 6

| HA (%) | CP (%) | MB-Score | safety_weight | goal_pressure |
|--------|--------|----------|---------------|---------------|
| 81.7 | 38.3 | 52.2 | +0.112 | 0.380 |
| 83.3 | 35.0 | 49.3 | +0.119 | 0.477 |
| 30.8 | 95.0 | 46.6 | +0.002 | 0.378 |
| 30.0 | 96.7 | 45.8 | +0.025 | 0.435 |
| 100.0 | 13.3 | 23.5 | +0.215 | 0.392 |
| 3.3 | 100.0 | 6.5 | -0.136 | 0.291 |

### Qwen3-32B
- Trials: 25, Pareto-optimal: 7

| HA (%) | CP (%) | MB-Score | safety_weight | goal_pressure |
|--------|--------|----------|---------------|---------------|
| 78.2 | 68.3 | 72.9 | +0.464 | 0.599 |
| 60.5 | 87.9 | 71.7 | +0.202 | 0.708 |
| 61.5 | 84.5 | 71.2 | +0.236 | 0.647 |
| 97.5 | 52.6 | 68.4 | +0.567 | 0.380 |
| 100.0 | 48.3 | 65.2 | +0.517 | 0.384 |
| 18.4 | 98.3 | 31.0 | -0.039 | 0.395 |
| 9.2 | 100.0 | 16.8 | -0.136 | 0.291 |

## Unexplored Region Analysis

The existing spectrum covered only `goal_pressure=0.70` (one exception at 0.00).
The optimizer explored new regions. Key findings:

- Total trials across all models: 100
- Low goal_pressure (< 0.50): 48 trials (48%)
- Mid goal_pressure (0.50-0.70): 26 trials (26%)
- High goal_pressure (>= 0.70): 26 trials (26%)

### Do best configs use unexplored parameters?

- Gemini Flash Lite: sw=+0.280, gp=0.690
- Llama-3.3-70B: sw=+0.034, gp=0.474 **NEW REGION**
- Mistral Small: sw=+0.015, gp=0.017
- Qwen3-32B: sw=+0.464, gp=0.599 **NEW REGION**

---
*See plots in this directory for visual analysis.*