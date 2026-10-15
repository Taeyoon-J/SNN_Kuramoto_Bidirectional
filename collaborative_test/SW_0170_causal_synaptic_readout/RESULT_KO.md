# SW0170 actual GPU TRAIN8 result: rejected before training

2026-10-15. The owned GPU1 admission completed cleanly: supervisor3734244,
worker3734297. Both exited normally with `complete_rejected`, not an operational
failure. Local/server CPU tests passed 22/22; actual native B4+B4 CPU Q/QCC,
source inputs, CLI, full fingerprints and exclusive worker ownership were checked.

| Fixed TRAIN8, optimizer zero | Identity point events | Causal synaptic current |
|---|---:|---:|
| Mean constrained maximin loss (lower better) | .4962020501 | .5034186066 |
| Mean routed positive-minus-negative affinity | .0014866169 | -.1286328143 |

The filtered QCC had nonempty foreground on all eight images, so noncollapse
passed. However both predeclared task criteria failed: loss increased and routed
class separation worsened. Source parameters and raw S were unchanged during
the readout comparison. Smoothing recovered foreground activity without useful
object discrimination in this probe. This does not identify the universal cause
of sparse-coding failure or prove filtering can never help another model.

No disposable optimizer updates, 64-update training, validation scoring, seed
fanout or 70k run were performed. Do not rescue this recipe with width/threshold
or sample changes. The early rejection prevented another long failed pilot.
Root independently reconstructed weighted critical-edge means and float32-square
losses from all saved edge/pair masses, checked all eight-row aggregates and
the exact registered failure list. This is TRAIN feasibility evidence, not FG-ARI.

SW165 remains the verified three-seed incumbent. The next direction must address
task representation or optimization; activity/nonzero gradients alone are not
accepted as evidence of useful binding. Evidence: completed_rejection_20261015
server_records.json and independent_verification.json under results_archive.
