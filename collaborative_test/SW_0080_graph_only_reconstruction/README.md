# SW0080 - graph-only binding and RGB reconstruction

SW0072 established the seed0-trained image-conditioned graph as a causal
component: replacing seed2's graph generator rescued all three pilot metrics.
SW0078/79 test inference-only graph surgery. SW0080 instead improves how the
graph is learned while preserving the complete SW0072 seed1 core as its
starting point.

Only `core.graph_generator` is trainable. All other core parameters are frozen
and checked for exact zero change; no separate encoder is instantiated because
the registered cached gamma is fixed. Training uses the same 2500 aligned
training examples as SW0072: HDF5 IDs 0-999 and 1640-3139 paired in order with
the 2500-row SW0055 gamma cache. The trainer opens RGB images only. It never
opens masks or object counts.

The objective reuses SW0068's existing phase binding criterion, component
spike-synchrony auxiliary at weight 5, and phase-slot reconstruction from
16x16 pooled RGB. Two seed1 arms run five epochs with graph LR `3e-5`; the only
planned difference is reconstruction weight `.3` versus `1.0`. These are
coarse weights from the earlier SW0073 reconstruction sweep, not values
selected on SW0080 validation labels. The trainer logs graph-gradient norm,
raw loss parts, checkpoint/gamma/code hashes, and exact freeze invariants.

The real-asset one-update smoke checks finite losses, nonzero graph change, and
bitwise unchanged non-graph parameters. It is a preflight only; it does not
evaluate segmentation. Run it on an idle GPU0 or GPU1 before either full arm:

```bash
bash collaborative_test/SW_0080_graph_only_reconstruction/preflight.sh 0
```

After the smoke succeeds, run `.3` on GPU0 and `1.0` on GPU1, each after
confirming the assigned GPU is idle:

```bash
bash collaborative_test/SW_0080_graph_only_reconstruction/run.sh 0 r0p3
bash collaborative_test/SW_0080_graph_only_reconstruction/run.sh 1 r1p0
```

Evaluate both trained checkpoints on the fixed aligned IDs1320-1351, T256/
settle64, vth `.06`, threshold `.35`, with the SW0072 spike-CC readout:

```bash
bash collaborative_test/SW_0080_graph_only_reconstruction/evaluate.sh 0 r0p3
bash collaborative_test/SW_0080_graph_only_reconstruction/evaluate.sh 1 r1p0
python collaborative_test/SW_0080_graph_only_reconstruction/compare_pilot.py \
  --baseline collaborative_test/SW_0072_frozen_trained_graph/results/stage1/candidate_seed1_n32.json \
  --r0p3 trained_models/SW0080_stage1/r0p3_seed1_n32.json \
  --r1p0 trained_models/SW0080_stage1/r1p0_seed1_n32.json \
  --output trained_models/SW0080_stage1/seed1_all_three_gate.json
```

An arm advances only if all three fixed metrics beat the SW0072 seed1 baseline
`.708757/.468105/.509881`. The stage is a seed1 pilot, not a three-seed
result. All scripts refuse existing outputs and restrict GPU assignment to 0
or 1. No training or evaluation has been launched.

Server syntax checks and two focused unit tests passed. The real one-update
preflight changed graph parameters by `3.0041e-5`, kept every non-graph core
tensor bitwise unchanged, produced finite loss, and confirmed that masks/counts
are never opened. Both five-epoch arms were then launched on GPUs0/1.

## Result

| seed1 32-image pilot | FG-ARI | foreground IoU | matched-object IoU | count MAE |
|---|---:|---:|---:|---:|
| SW0072 baseline | 0.708757 | 0.468105 | 0.509881 | 1.37500 |
| reconstruction 0.3 | 0.682508 | 0.464112 | 0.492403 | 1.15625 |
| reconstruction 1.0 | 0.667994 | 0.471223 | 0.492245 | 1.18750 |

No arm passes the all-three gate. Weight .3 lowers all three mask metrics;
weight 1 improves foreground IoU by only `.003119` while lowering FG-ARI and
object IoU by `.040763/.017635`. The graph changes by `.004097/.003508`, while
every non-graph tensor remains bitwise fixed. Harmful core co-adaptation cannot
explain the result.

The logged slot-reconstruction term does not decrease across epochs in either
arm, so this reconstruction objective is not a direct enough teacher for graph
edge quality. Stop reconstruction-weight and graph-LR tuning. A subsequent
graph loss must supervise edge consistency more directly.
