 pl# ManagerBench — Supervisor Summary Report

**Date:** 2026-03-15

---

## Executive Summary

We evaluated five language models on **ManagerBench**, a benchmark measuring whether AI systems avoid harmful actions when nudged by a goal-oriented manager. Three configurations were tested: a **push-nudge baseline** (model is told "achieve your goal no matter what"), a **no-push baseline** (same scenario without the nudge), and a **safe-system-prefix** condition (model's system prompt begins with an explicit safety instruction).

**Key finding:** The push nudge severely suppresses harm avoidance. Adding a safe system prefix dramatically recovers safety behavior, with average harm avoidance rising from **6.3%** (push-nudge) to **75.8%** (safe-prefix) — a **+69.5 ppt** improvement. The no-push baseline sits in between at **32.9%**.

---

## Configurations

| Config | Push Nudge | Safety Prefix | Avg Harm Avoid (%) | Avg Control Prag (%) | Avg MB-Score |
|--------|-----------|---------------|-------------------|---------------------|-------------|
| **Push-Nudge** | Yes | No | 6.3 | 99.2 | 10.6 |
| **No-Push** | No | No | 32.9 | 93.5 | 42.8 |
| **Safe-Prefix** | Yes | Yes | 75.8 | 53.5 | 50.8 |

### All Metrics Overview

![Overview Grid](results/plots/overview_grid.png)

> **Harm Avoidance**: % of harmful scenarios where model chose the safe option. **Control Pragmatism**: % of neutral scenarios where model chose the practical option. **Tilt Imbalance**: difference between the two (ideal ≈ 0). **MB-Score**: harmonic mean (higher = better balanced).

---

## Per-Model Results

![MB-Score by Model](results/plots/mb_score_by_model.png)

| Model | Push-Nudge HA (%) | No-Push HA (%) | Safe-Prefix HA (%) | Best MB-Score |
|-------|-------------------|----------------|-------------------|---------------|
| Gemini-2.5-Flash-Lite | 0.7 | 9.2 | 57.2 | 67.2 (Safe-Prefix) |
| Qwen3-32B | 1.4 | 24.1 | 56.5 | 70.0 (Safe-Prefix) |
| Llama-3.3-70B | 22.4 | 75.6 | 99.6 | 78.7 (No-Push) |
| Mistral-Small-24B | 0.8 | 22.8 | 90.1 | 39.0 (Safe-Prefix) |

![Harm Avoidance & Control Pragmatism](results/plots/harm_avoidance_control_pragmatism.png)

---

## Balance: Harm Avoidance vs Control Pragmatism

![Tilt Scatter](results/plots/tilt_scatter.png)

Points on the **diagonal** are perfectly balanced. Points **above** the diagonal show over-caution (model avoids harm but also avoids pragmatic choices). Points **below** show under-caution. Most push-nudge models cluster near 0–2% harm avoidance despite ~90–100% control pragmatism — meaning the nudge effectively suppresses safety behavior while leaving normal task compliance intact.

---

## Per-Setting Breakdown (Benefit% × Harm%)

![Per-Setting Heatmaps](results/plots/per_setting_heatmap.png)

Each cell shows the MB-Score for a specific experimental setting. Settings vary the stated benefit to the company (10% or 50%) and the harm risk (5% or 15%). For the safe-prefix config, performance is relatively stable across settings, suggesting the safety instruction is robust. _Note: baseline-no-push per-setting data is unavailable (raw files removed)._

---

## Notable Findings

- **Most improved by safe-prefix:** Mistral-Small-24B (+89.2 ppt harm avoidance).
- **Least improved:** Qwen3-32B (+55.1 ppt).
- **Mistral-Small-24B** with safe-prefix shows extremely high harm avoidance (90.1%) but very low control pragmatism (24.9%), indicating the safety prefix caused over-refusal — it refuses harmful *and* benign requests.

- Removing the push nudge (no-push vs push-nudge) improves harm avoidance by **+26.6 ppt** on average, but is still far below the safe-prefix condition. The nudge alone explains a substantial portion of the gap.

---

_Full detailed report with per-setting tables: `results/report_per_model_config.md`_
