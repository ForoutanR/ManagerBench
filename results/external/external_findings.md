# External validation — XSTest + HarmBench (batch B, real runs 2026-07-07)

_10 configs (6 grid winners + 4 per-model baselines) × 2 benchmarks × 4 models.
Two scorers: a string-heuristic and a gemini-2.5-flash-lite judge. **The judge is the
primary scorer**; the heuristic disagrees with it by 40–55% on XSTest and up to 34% on
HarmBench (it under-detects implicit refusals and over-counts non-refusal text as
compliance), so heuristic numbers are reported only as a noisy cross-check._

## HarmBench — attack-success rate (judge ASR, lower = safer)

| model | winner ASR | baseline ASR | Δ |
|---|---|---|---|
| gemini (c6-2) | 0.0 | 1.5 | −1.5 |
| llama (c4-0) | 2.5 | 5.0 | −2.5 |
| mistral (c4-1) | 3.0 | 22.5 | **−19.5** |
| mistral (c5-1) | 3.5 | 22.5 | −19.0 |
| qwen (c6-0) | 0.0 | 2.5 | −2.5 |
| qwen (c6-1) | 0.0 | 2.5 | −2.5 |

**The ManagerBench-tuned safety operating point transfers to an external harm benchmark:
every grid winner has a lower judge-ASR than its model's neutral baseline**, dramatically
so on Mistral (22.5 → 3.0), the model whose ManagerBench gain was also largest. This is
the key external-validity result.

## XSTest — over-refusal on SAFE prompts (judge, lower = better)

| model | winner over-refusal | baseline | Δ |
|---|---|---|---|
| gemini (c6-2) | 16.0 | 12.8 | +3.2 |
| llama (c4-0) | 10.4 | 10.0 | +0.4 |
| mistral (c4-1) | 12.8 | 11.2 | +1.6 |
| mistral (c5-1) | 14.8 | 11.2 | +3.6 |
| qwen (c6-0) | 35.6 | 36.0 | −0.4 |
| qwen (c6-1) | 38.0 | 36.0 | +2.0 |

**The safety framing costs at most +3.6 points of over-refusal versus baseline** — a small
price for the HarmBench safety gain. Qwen is a high-refuser regardless of prompt (~36% on
safe prompts, consistent with its high ManagerBench HA); the winner does not make this
materially worse. On the UNSAFE XSTest prompts the judge sees ≥91% refusal for every
config (the heuristic's 1–21% here is simply wrong — it misses implicit refusals).

## Reading

Together: the prompts our repaired pipeline selected on ManagerBench are **externally
safer** (lower harmful-compliance on HarmBench, all 4 models) at a **small, bounded
over-refusal cost** on XSTest. The operating point is not a ManagerBench artifact.

## Caveats (state in the thesis)

1. **Judge reliability.** A single LLM judge (gemini-2.5-flash-lite) is the primary
   scorer; the string heuristic disagrees heavily and is not trustworthy here. A
   multi-judge or human-audited subset would strengthen this.
2. **Self-judging.** The judge model also judges Gemini's own outputs; Gemini's near-zero
   ASR should be read with that self-preference risk in mind. Qwen/Llama/Mistral results
   do not have this issue.
3. **HarmBench scoring deviation.** Judge-based ASR, not the official GPU classifier
   (unavailable on our infrastructure) — already disclosed.
4. **Spot-check scope.** Only the selected winners + baselines, not the full grid.

Cost: batch B ≈ \$0.82 (cheap target + judge models). Key usage \$19.18 / \$30.
