# SW0141 native32 source-preservation objective pilot

Compare C-only and C+source-Q from fresh identical source1/Adam states. Both remove the old phase/Q statistical objectives and train encoder plus graph on the same ordered4096 images/256 logicalB16 updates. Native core/scalars stay frozen. The canonical source-Q teacher uses no masks.

Local and server CPU20/20 checks passed, including the actual checkpoint→protocol→prediction validator file roundtrip. Actual source RGB/gamma is bit-exact. Supervisor2908798 and cache workers2908822/2908823 were verified live on leased GPU0/1. Calibration, disposable GPU preflights, both training arms, frozen native32 predictions and GT-last fixed320 scoring are queued in the ten-task DAG. Both caches, source0 calibration and both disposable GPU preflights subsequently passed. Both training arms started on leased GPU0/1 (PIDs2926522/2926523) and were verified live; see [training_start.json](results_archive/training_start.json). Both training arms and native32 predictions completed. The original CPU score failed before GT access on a missing legacy Slot metadata key; a CPU-only in-memory hash-handoff repair completed scoring without retraining.

This is a one-seed inherited70k-lineage continuation pilot. It cannot establish three-seed Slot superiority, increasing-data gains or all-gating necessity. Only a candidate that beats source and C-only with positive paired95% FG lower bounds may justify seed/70k expansion. See protocol.json and results_archive receipts.

## Completed seed1 result (2026-10-11)

| Model | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| Source97 | .754135 | .606900 | .582201 |
| C-only | .752123 | .566258 | .580526 |
| C+source-Q | .752108 | .607767 | .581191 |
| Old+C (SW0139) | .738505 | .592047 | .566686 |
| Slot32 seed1 | .841819 | .203417 | .222115 |

Both FG expansion gates failed. Close both fixed recipes with no lambda-grid, seed or70k expansion. Removing old statistical priors recovered the old+C FG decline; source-Q recovered foreground IoU without object-grouping improvement. SW0137 remains the native32 incumbent. No three-seed win, data-scaling or all-gate claim.

[Raw evaluation](results_archive/score_recovery_20261011/evaluation.json), [recovery receipt](results_archive/score_recovery_20261011/score_recovery_receipt.json), [independent replay](results_archive/score_recovery_20261011/independent_endpoint_analysis.json), and [recovery code](score_recovery.py). Source and Slot per-image references exactly reproduce SW0137. The protocol retains its original registration-time status; this README and best-model tracker record the endpoint.
