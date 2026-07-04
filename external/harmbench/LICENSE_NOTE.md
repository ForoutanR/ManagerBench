# HarmBench — vendored data provenance

- Source: https://github.com/centerforaisafety/HarmBench
  (file `data/behavior_datasets/harmbench_behaviors_text_all.csv`, main branch)
- Paper: Mazeika et al., "HarmBench: A Standardized Evaluation Framework for Automated
  Red Teaming and Robust Refusal", ICML 2024, arXiv:2402.04249.
- License: MIT (see upstream repo). Vendored unmodified for reproducibility.
- Use here: the 200 `standard` (non-copyright, non-contextual) text behaviors only.
- DEVIATION: the official HarmBench classifier needs a GPU (unavailable). We score
  attack-success with a two-tier heuristic + gemini-flash-lite judge (see run_harmbench.py).
- Fetched: 2026-07-04 for OPUS_TASKS T13.
