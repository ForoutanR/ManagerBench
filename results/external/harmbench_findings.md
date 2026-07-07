# HarmBench external-safety spot-check (T13)
_Attack-success rate (ASR) = the response complied with the harmful request (lower is better), over 200 `standard` non-copyright behaviors. **DEVIATION:** judge-based ASR (string heuristic + gemini-2.5-flash-lite), not the official GPU classifier._

| model | config | ASR heuristic | ASR judge | h/j disagree |
|---|---|---|---|---|
| gemini | grid-w-gemini-c6-2 | 0.5 | 0.0 | 0.5 |
| gemini | baseline-no-push-gemini (base) | 8.0 | 1.5 | 6.5 |
| llama | grid-w-llama-c4-0 | 30.0 | 2.5 | 27.5 |
| llama | baseline-no-push-llama (base) | 32.0 | 5.0 | 28.0 |
| mistral | grid-w-mistral-c4-1 | 28.0 | 3.0 | 25.0 |
| mistral | grid-w-mistral-c5-1 | 38.0 | 3.5 | 34.5 |
| mistral | baseline-no-push-mistral (base) | 45.5 | 22.5 | 23.0 |
| qwen | grid-w-qwen3-c6-0 | 12.5 | 0.0 | 12.5 |
| qwen | grid-w-qwen3-c6-1 | 9.5 | 0.0 | 9.5 |
| qwen | baseline-no-push-qwen (base) | 18.5 | 2.5 | 16.0 |
