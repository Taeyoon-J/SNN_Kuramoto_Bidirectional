# SW0099 stage-flow diagnosis

SW0099 is a validation-only trace of the frozen SW0095 final, matched SW0097
positive-frozen controls, and SW0098 long-window checkpoints for seeds 0/1/2.
It scored validation IDs 1320–1335 in two original batch-8 forwards per
checkpoint (nine checkpoints total). No optimizer, parameter update, holdout
read, threshold tuning, or checkpoint write occurred.

The readout was held to the peer classifier: 1024 inference steps, settle 512,
positive per-component Pearson-product spike affinity, threshold .50, minimum
component size 2, with the largest connected component assigned to background.
Ground-truth patch labels were loaded only after each forward for diagnostic
scoring. Temporal affinities use steps 512–1023. The trace records graph
adjacency, theta product PLV, sinusoidal carrier, delayed raw gate, the exact
`sin(theta) * gate` carrier, dendritic output, positive-product membrane
affinity, and the fixed spike affinity. Pair discrimination is summarized by
foreground pair AUC and same-minus-different similarity, overall and within
fixed Euclidean patch-distance bins `(0,2]`, `(2,5]`, `(5,10]`, and `(10,∞]`.
Membrane/spike affinity also has fixed-.50 edge and per-object connectivity
summaries.

## Result

The corrected diagnostic endpoint means over these 16 images were:

| Checkpoint set | FG-ARI | Foreground IoU | Matched-object IoU |
|---|---:|---:|---:|
| SW0095 | .811208 | .557463 | .645968 |
| SW0097 | .810841 | .561339 | .642008 |
| SW0098 | .810284 | .555407 | .641283 |

SW0098 minus its matched SW0097 control was −.000557/−.005932/−.000725.
These 16-image endpoint values are diagnostic only, not the full-320 evaluation
contract.

Graph-generator states were bitwise equal across all three checkpoint sets and
seeds. Theta product-PLV mean foreground pair AUC rose by .001687 from SW0097
to SW0098. The raw carrier AUC was lower for all three seeds overall, but that
decline was not consistent within fixed-distance bins. The earliest
distance-controlled decline was at the delayed raw gate: SW0098 AUC fell in
the near bin for all seeds (−.0128/−.0183/−.0271) and in the mid bin for all
seeds (−.0410/−.0182/−.0348); the near/mid decline persisted at the spike
stage. This localizes a repeatable association in these traces, not a causal
mechanism or proof that the gate is the model bottleneck. At the fixed .50
connectivity threshold, seed 0 worsened while seeds 1 and 2 mostly improved;
the 16-image sample does not support a universal fragmentation claim. Full
per-stage, per-seed, distance-stratified and endpoint records are in
`summary.json`; the 5.6 Sol interpretation is reflected above.

## Execution and audit trail

Before the nine-checkpoint suite, SW0095 seed 0 on IDs 1320–1327 exactly
matched the archived batch-8 per-image endpoint metrics. The first smoke
attempt failed before endpoint scoring because connected-component
postprocessing mixed CUDA affinity with CPU traversal state; the completed
smoke moved those already-computed classifier tensors to CPU. `smoke_audit.json`
preserves the failure and the successful comparison.

The first nine-checkpoint run is retained in `results/`, but its dendritic
trace was invalid because folded `[B*D,T,N]` hook records were reinterpreted as
`[T,B,D,N]`. Do not use those dendritic values. All other stages and endpoint
records from that run were independently compared with the corrected run and
were exactly equal. The corrected runner passed a synthetic fold-order
sentinel; the corrected smoke again matched all endpoint metrics exactly and
unaffected stage summaries exactly. The corrected nine-checkpoint records are
in `results_corrected_dendrite_order/`, with their runner SHA embedded in each
record. `execution_audit.json`, both smoke audits, and `graph_frozen_audit.json`
retain this provenance.

The corrected queue completed on GPU 0 after checking GPU ownership; GPUs 1–3
were occupied by other users. The user had authorized GPUs 0–3, but only a
genuinely free device was used. The executed coordinator wrote tagged progress
to shared `state.json` instead of the tag-specific state file. Terminal
completion and all nine isolated corrected outputs were verified; the local
coordinator now writes `state_corrected_dendrite_order.json` for tagged runs.
The local code reflects that fix, while `summary.json` records the SHA of the
coordinator that actually executed.

`diagnose.py` runs a checkpoint and `coordinator.py` queues the nine registered
comparisons. This small diagnostic subset did not read holdouts or tune model
settings. It cannot establish full-validation performance or causation.

## Synthetic phase-representation audit

After the image-trace analysis, a deterministic CPU-only audit called the
repository's actual `sinusoidal_gating` function at each time step, with
finite float64 phases of shape `[1,5,3,4]` and `gate_mode="raw"`. The audit's
vectorized helper matched the actual function exactly (zero maximum difference)
for all four phase histories. Adding `2π` to one component across its history left its
`sin(theta)` carrier unchanged to `1.2e-15`, but changed the delayed raw gate
by up to `0.6992` and the gated drive by up to `0.5332`. Adding `2π` to all
components preserved gate and drive within `1.1e-15`; permuting components
preserved the gate and permuted drive within `4.5e-16`. This establishes a
representation sensitivity of the implemented formula to a single-component
phase lift, while its common-shift and component-permutation controls pass.
It is synthetic mathematical evidence only: it reads no checkpoint or image
and says nothing by itself about model performance or a design change. See
`sinusoidal_gate_periodicity_audit.py` and its JSON output for exact deltas,
the actual gating source SHA, and audit source hashes.
