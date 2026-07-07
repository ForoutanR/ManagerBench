# External-benchmark validation summary — winners vs baseline (T12/T13)
_Judge-based numbers (gemini-2.5-flash-lite) are primary; the string heuristic is reported for transparency but is unreliable here (heuristic↔judge disagreement 17–55%; on UNSAFE prompts the heuristic detects 1–14% refusal vs the judge's 95%+, i.e. it badly under-detects refusals phrased without its keyword families). Single-judge limitation applies._

## Per winner: Δ vs its model's baseline-no-push (judge)
| model | winner | ΔASR HarmBench (↓ good) | Δover-refusal XSTest safe (↑ = cost) | winner ASR | base ASR | winner over-ref | base over-ref |
|---|---|---|---|---|---|---|---|
| gemini | grid-w-gemini-c6-2 | -1.5 | +3.2 | 0.0 | 1.5 | 16.0 | 12.8 |
| llama | grid-w-llama-c4-0 | -2.5 | +0.4 | 2.5 | 5.0 | 10.4 | 10.0 |
| mistral | grid-w-mistral-c4-1 | -19.5 | +1.6 | 3.0 | 22.5 | 12.8 | 11.2 |
| mistral | grid-w-mistral-c5-1 | -19.0 | +3.6 | 3.5 | 22.5 | 14.8 | 11.2 |
| qwen | grid-w-qwen3-c6-0 | -2.5 | -0.4 | 0.0 | 2.5 | 35.6 | 36.0 |
| qwen | grid-w-qwen3-c6-1 | -2.5 | +2.0 | 0.0 | 2.5 | 38.0 | 36.0 |

## Takeaway
- **External harm-avoidance generalizes:** every grid-winner config lowers HarmBench judge-ASR vs its baseline-no-push (Δ −1.5 to −19.5 pts; mistral 22.5→3.0 the largest). Safety steering learned on ManagerBench transfers to a *different* harmful-request set.
- **At a modest over-refusal cost:** XSTest over-refusal on safe prompts rises only 0–3.6 pts (judge) for the winners — the same safety↔pragmatism trade-off, now external. qwen is over-refusal-prone at baseline (~36%) independent of steering.
- **Scorer caveat:** trust the judge column; the keyword heuristic under-detects refusals and should not be read as ground truth.
