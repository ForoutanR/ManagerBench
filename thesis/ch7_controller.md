# Chapter 7 — The Controller: From Description to Prescription

*Draft. Source: COMPLETE_REPORT §23 + addenda, grid_findings.md §3, demo_cli.py,
controller_hit.log. §7.5 external validity pending batch B (XSTest/HarmBench).
Figures: F7 (controller trajectory).*

The preceding chapters describe *where* a model sits and *how* prompts move it. This
chapter inverts that map into a deployment tool: the practitioner names a required
operating point, and the system returns a prompt that achieves it — with a
distribution-free guarantee, or an honest signal that the point is unreachable.

## 7.1 The inverse map

The forward map from a prompt cell to its measured (HA, CP) is predictable enough to
invert. Given a target (HA*, CP*), the controller predicts each of the 45 cells'
operating point — using the full-benchmark measurement where one exists, otherwise an
inverse-distance interpolation over the measured cells' centres — and returns the cell
whose predicted point is nearest the target, decoded to its prompt. Crucially the search
is over the 45 discrete cells, not over the continuous parameter box: an earlier
continuous formulation returned coordinates that crossed a template bin boundary and
decoded to a different, much safer prompt than intended (predicted HA 54 / CP 90,
measured 86 / 69), a bug that vanishes once prediction and inversion both respect the
cells. Off-frontier targets (e.g. HA 100 and CP 100 simultaneously) are reported as
infeasible with the nearest achievable point, because the trade-off is a property of the
benchmark, not a limitation of the search.

## 7.2 Distribution-free guarantees by split conformal prediction

A point prediction is not enough for a safety deployment; the practitioner needs to know
how far the achieved operating point may fall from the prediction. We attach a
split-conformal interval. Calibrating on the per-model residuals |full − proxy| over the
cells with both measurements (n = 11 per model), the 90% half-widths are HA ±6.1 to ±9.4
and CP ±5.5 to ±10.6. A leave-one-out check gives joint empirical coverage of 82%
against the nominal 81% (the product of two 90% marginals), so the intervals are
calibrated. The intervals are conservative because n = 11 forces the conformal quantile
to the maximum residual, and mid-frontier cells — the noisiest, per Chapter 5 —
occasionally strain them; both are honest consequences of a small calibration set and are
stated as limitations. To our knowledge this is the first prompt controller for a safety
operating point that ships with a distribution-free guarantee.

## 7.3 Closed-loop verification

When a tighter guarantee than the offline interval is required, the controller can verify
against the live model: seed from the inverse map, measure the seed cell on the proxy,
and if the measured error exceeds a tolerance ε, search neighbouring cells, keeping the
best measured point. A live demonstration on Qwen (target HA 88 / CP 75, ε = 5) seeded at
the wrong tier (error 20.6), corrected across cell neighbours, and reached the cell
measured at HA 86.7 / CP 76.7 — error 2.1 — in 14 evaluations (~$0.6, ~50 minutes under
throttle). The cell it converged to is the same one that Chapter 5's full-benchmark
validation independently measured at 86.8 / 78.7, closing the loop between the controller
and the ground-truth grid. The handoff item that motivated this chapter — a clean live
verified hit, unachieved in earlier work because the continuous surrogate seeded far from
the target — is thereby delivered.

## 7.4 The demo tool

The controller is packaged as a small command-line tool (`demo_cli.py`, offline,
zero-cost) intended for the thesis defence and for practitioners. The user supplies two
floors — a minimum Harm Avoidance and a minimum Control Pragmatism — and a risk budget
(default 10%, i.e. the 90% conformal lower bounds must clear both floors). The tool
returns the highest-MB prompt that *guarantees* both floors at that risk level, printing
the exact prompt text, the guaranteed lower bounds, and whether the prediction rests on a
full-benchmark or proxy measurement; if no prompt can guarantee the request it reports
INFEASIBLE with the nearest achievable point and alternatives. For example, a Qwen request
for HA ≥ 80, CP ≥ 65 returns the careful-tier no-pressure prompt with guaranteed floors
HA ≥ 82.9, CP ≥ 66.4; a Gemini request for HA ≥ 90 and CP ≥ 90 is correctly reported
infeasible. This converts the study from descriptive to prescriptive: a use-case states
the safety and usefulness it needs and receives either a prompt that provably meets it or
an explicit statement that its requirement is off the model's achievable frontier.

## 7.5 External validity [PENDING batch B]

*To be completed from `results/external/` once the XSTest and HarmBench runs finish.*
The controller's operating points are defined on ManagerBench; this section will test
whether a selected prompt's safety/pragmatism behaviour holds on independent benchmarks —
over-refusal on XSTest (safe prompts that should be answered) and attack-success on
HarmBench standard behaviours (harmful requests that should be refused) — comparing each
winner prompt against its model's neutral baseline. The HarmBench scoring uses an
LLM-judge rather than the official GPU classifier (unavailable on our infrastructure);
this deviation will be stated with the results.
