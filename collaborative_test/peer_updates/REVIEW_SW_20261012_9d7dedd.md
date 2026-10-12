# Peer review: patch_v2 at 9d7dedd (2026-10-12)

Previous inspected peer head: `697eeb67dc031325b63153e678cea8cbc602da31`.
Read-only `ls-remote` and fetch verified new head
`9d7dedd200fe02bb1a175b55c47fc098e8b511ce`. Fourteen new commits were inspected
through their published experiment records; no changes were merged into the frozen
SW0142 runner or its dependencies. SW0142 joint-training PID361308 remains live.

## Positive candidate: selective graph distance, inference only

PV2_0076 reports replacing adjacent dissimilarity `1-cosine` with its square in
the geodesic step cost, while setting effective contrast to 8. The leading base
cost stays 1. Reported three-seed test means at fixed spectral k14 are
**0.7603 / 0.3373 / 0.6598**, versus the previous
**0.7533 / 0.3358 / 0.6549**. The unchanged connected-component readout also
improves, from **0.7258 / 0.7124 / 0.5029** to
**0.7300 / 0.7132 / 0.5050**. Those values are peer-reported, not reproduced here.

The transferable hypothesis is that boundary-crossing costs should grow more
than within-object shading costs. It does not require using masks to construct
the graph. Our native32 chunked geodesic operator has the same linear
dissimilarity term, so a separately versioned operator can test this mechanism.
Keep the spike-derived QCC classifier, threshold, background rule and fixed320
evaluation contract unchanged. Do not copy k14: the peer's spectral FG gain is
partly a foreground-extent tradeoff, and object-count learning is not demonstrated
by choosing a fixed cluster count.

## Negative evidence to incorporate

- PV2_0077: training under power2 scored **0.4268 / 0.4115 / 0.3035** on its
  validation contract, below the linear-trained baseline. Therefore port the
  inference intervention first; do not change the current training objective.
- PV2_0078: lowering the base step cost did not improve the calibrated
  connected-component result. Its spectral FG increase traded away foreground
  IoU. Keep base1; do not queue a base-cost sweep.
- PV2_0068: RGB reconstruction reduced loss without improving object grouping.
  This supports evaluating SW0142 by segmentation and its paired control, not
  by reconstruction loss or nonzero gradients alone. Its objective differs from
  SW0142 masked completion through actual spikes, so it is not a substitute for
  finishing our registered comparison.
- PV2_0072 withdraws the claim that oracle object codes establish attainable
  unsupervised accuracy: oracle codes transcribe ground-truth identity. Random
  patch corruption was repaired, while coherent object merges persisted. Oracle
  performance is a diagnostic bound, not evidence of a discoverable solution.

## Contract and evidence limits

Peer images are 240x320 with a 16x16 patch grid, 6,000 training IDs, validation300
and test1000. Our current contract uses 128x128 images, a 32x32 grid, fixed320
evaluation and source models previously trained on our70k at the older grid.
Absolute scores and Slot baselines cannot be pooled. The peer's three models
share one graph; its Slot baseline is trained for fewer epochs than its stated
official recipe; the test set was read a second time for PV2_0076.

Actual Git inspection also found the peer model package, including
`snn_kuramoto_bidirectional/graph_generator.py`, deleted from the current
published tree at commit36c59e2. The published evaluator does not expose the new
power argument. Thus the README equation and metric CSV are available, but the
claimed complete implementation is not independently inspectable at this head.
Any local port must be implemented and regression-verified independently.

## Next action

Continue SW0142 unchanged. Have the design review specify a narrowly queued
native32 inference experiment, with a power1 regression check and a contrast-only
control separating the square-cost effect from changing contrast. Preserve all
existing scientific file hashes and source checkpoints. Promotion still requires
our own three-seed comparison; an inference gain does not establish core training,
all-gating contribution, object-count learning or fresh native32 70k scaling.

Evidence:
[PV2_0076](https://github.com/Taeyoon-J/SNN_Kuramoto_Bidirectional/blob/9d7dedd200fe02bb1a175b55c47fc098e8b511ce/collaborative_test/PV2_0076_geodesic_power/README.md),
[PV2_0077](https://github.com/Taeyoon-J/SNN_Kuramoto_Bidirectional/blob/9d7dedd200fe02bb1a175b55c47fc098e8b511ce/collaborative_test/PV2_0077_train_power/README.md),
[PV2_0078](https://github.com/Taeyoon-J/SNN_Kuramoto_Bidirectional/blob/9d7dedd200fe02bb1a175b55c47fc098e8b511ce/collaborative_test/PV2_0078_geodesic_base/README.md),
[PV2_0072](https://github.com/Taeyoon-J/SNN_Kuramoto_Bidirectional/blob/9d7dedd200fe02bb1a175b55c47fc098e8b511ce/collaborative_test/PV2_0072_transfer/README.md).
