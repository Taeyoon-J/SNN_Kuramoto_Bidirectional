# SW0169 final result: rejected

2026-10-15. Both registered 64-update pilots and all fixed320 predictions completed.
The public CPU validator independently recomputed every metric, prediction binding,
paired interval and decision from the frozen predictions and canonical modal32 GT.
Root separately rechecked all 320-value means, paired bootstrap intervals and code hashes.

| Seed1 fixed320/native32 endpoint | FG-ARI | Foreground IoU | Matched object IoU |
|---|---:|---:|---:|
| history_live_conditional | 0.03837044 | 0.61499343 | 0.05815510 |
| history_live_standalone | 0.00000000 | 0.00000000 | 0.00000000 |
| history_detached_conditional | 0.03858640 | 0.61499343 | 0.05905823 |
| history_detached_standalone | 0.00007265 | 0.00009091 | 0.00008681 |
| SW0164 | 0.77093210 | 0.61499343 | 0.63052348 |
| Source-T | 0.75527308 | 0.61499343 | 0.57951379 |
| Slot | 0.84181929 | 0.20341671 | 0.22211516 |

The registered co-primary gate failed. Live did not improve either output over
SW164 or its detached-history control. Conditional foreground is preserved exactly,
but object assignments collapse; the standalone live endpoint scores zero on all
three metrics. Nonconstant firing and nonzero gradients were insufficient for
object binding. This is a negative supervised pilot, not evidence that memory
or hard reset is generally useless. Sparse temporal coding/readout mismatch is
a hypothesis supported by TRAIN diagnostics, not an established causal explanation.

Close this recipe without threshold/time-scale/LR/seed sweeps, 70k expansion or
the positive-gate-only history/reset interventions. SW165 remains the verified
three-seed incumbent (.804298915 / .609975448 / .651148975); do not compare these
seed1 pilot results with its three-seed mean as if their scopes were identical.

The conditional synaptic-current idea in PROSPECTIVE_NEXT_STEP.md can now receive
a separate prospective design and cheap TRAIN-only feasibility test. It is not
a successful result and cannot change this rejection.

Evidence: results_archive/completed_score_20261015/server_records.json and
independent_verification.json. Score SHA256: `896562821faa67bb91acdc0612b2cf56a9bd594d63dcf944d002326b06d5875c`.
