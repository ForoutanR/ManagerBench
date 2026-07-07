# XSTest over-refusal — external validation (T12)
_Over-refusal = full refusal rate on the 250 SAFE prompts (lower is better). Refusal on the 200 UNSAFE prompts should stay high. Two scorers: string heuristic and a gemini-2.5-flash-lite judge._

| model | config | over-refusal SAFE (heur / judge) | refusal UNSAFE (heur / judge) | h/j disagree |
|---|---|---|---|---|
| gemini | grid-w-gemini-c6-2 | 7.2 / 16.0 | 14.0 / 96.0 | 46.0 |
| gemini | baseline-no-push-gemini (base) | 2.4 / 12.8 | 21.0 / 95.5 | 43.1 |
| llama | grid-w-llama-c4-0 | 2.4 / 10.4 | 74.0 / 94.5 | 17.6 |
| llama | baseline-no-push-llama (base) | 1.2 / 10.0 | 73.5 / 94.0 | 18.4 |
| mistral | grid-w-mistral-c4-1 | 1.6 / 12.8 | 1.0 / 96.0 | 54.7 |
| mistral | grid-w-mistral-c5-1 | 1.2 / 14.8 | 2.5 / 95.0 | 54.0 |
| mistral | baseline-no-push-mistral (base) | 0.0 / 11.2 | 2.0 / 91.0 | 47.6 |
| qwen | grid-w-qwen3-c6-0 | 56.4 / 35.6 | 50.5 / 97.0 | 40.9 |
| qwen | grid-w-qwen3-c6-1 | 64.0 / 38.0 | 54.5 / 97.5 | 39.8 |
| qwen | baseline-no-push-qwen (base) | 65.6 / 36.0 | 62.5 / 97.0 | 38.7 |
