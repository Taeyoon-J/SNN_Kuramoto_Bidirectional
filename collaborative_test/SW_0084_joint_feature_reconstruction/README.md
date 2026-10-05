# SW0084: jointly learn RGB features, graph, and downstream core

SW0068 trained the depth-1 native RGB encoder with the core, but spent two
epochs on cached gamma before joint updates and used no phase-slot RGB
reconstruction. SW0073 added reconstruction weights `.3` and `1.0`, but froze
the SW0072 graph; its component-swap results showed useful learned features
paired with harmful downstream co-adaptation. SW0084 tests the combined
mechanism: phase binding, 5x component-spike synchrony, and phase-slot RGB
reconstruction are active together from epoch1 while the encoder, learned
graph, and complete downstream core are all trainable.

All arms use seed1, 10 epochs, batch16, 2500 registered examples in the exact
order HDF5 IDs `0-999` then `1640-3139`, cached gamma in corresponding row
order, core LR `3e-4`, encoder LR `3e-5`, graph init seed0, seven phase slots,
temperature `.3`, and zero warmup. That is 25,000 image exposures and 1,570
updates (157 per epoch, with the final batch of four retained). RGB targets
are 16x16 pooled images. The trainer opens only the HDF5 `image` field; it
never loads masks or object counts.

The registered depth-1 RGB encoder and its matching normalization statistics
initialize every arm. This keeps the verified feature basis and allows a
preflight check that its output reproduces cached gamma. Fresh arms construct a
new S2NetCore and initialize its graph with seed0. Arm C instead starts from
the full SW0072 seed1 core; its graph remains trainable. This separates the
initialization effect from SW0073's frozen-graph condition.

| Arm | Core/graph initialization | Slot reconstruction weight | Purpose |
|:---:|---|---:|---|
| A | Fresh core, graph seed0 | 1.0 | Main combined-mechanism port |
| B | Fresh core, graph seed0 | 0.0 | No-reconstruction control |
| C | SW0072 seed1 full core; graph trainable | 1.0 | Stable-core initialization control |
| D | Fresh core, graph seed0 | 0.3 | Coarse reconstruction-scale control |

The peer PV2_0039 implementation is not present in this local checkout, so
its initialization details could not be independently inspected. SW0084 uses
the existing registered depth-1 encoder plus a fresh jointly trained core for
the primary arm, while preserving the exact local data order and S2Net
hyperparameters. The manifests make this initialization choice explicit.

## Run

First perform the real one-update preflight separately for each arm on an idle
GPU. It requires finite losses, a finite nonzero reconstruction gradient to
the encoder when reconstruction is enabled, and nonzero changes to encoder,
graph, and full core. It verifies cached-gamma reproduction, arm identity,
source hashes, and that only images are read:

```bash
bash collaborative_test/SW_0084_joint_feature_reconstruction/preflight.sh 0 A
bash collaborative_test/SW_0084_joint_feature_reconstruction/preflight.sh 1 B
bash collaborative_test/SW_0084_joint_feature_reconstruction/preflight.sh 2 C
bash collaborative_test/SW_0084_joint_feature_reconstruction/preflight.sh 3 D
```

After all four preflights pass, the launcher rechecks all GPU assignments and
starts the four arms on GPUs0-3. Each arm independently revalidates its
preflight hashes and refuses existing outputs:

```bash
bash collaborative_test/SW_0084_joint_feature_reconstruction/launch_parallel.sh
```

The trainer saves exact combined core/encoder checkpoints after epochs 5 and
10; the run script exports validation gamma for each saved encoder using the
same registered feature/statistics path. To compare against the SW0072 seed1
baseline, run the fixed T256/settle64, vth `.06`, spike-CC `.35` evaluator on
the baseline and each arm at both epochs:

```bash
bash collaborative_test/SW_0084_joint_feature_reconstruction/evaluate.sh 0 baseline 10
for arm in A B C D; do
  for epoch in 05 10; do
    bash collaborative_test/SW_0084_joint_feature_reconstruction/evaluate.sh 0 "$arm" "$epoch"
  done
done
python collaborative_test/SW_0084_joint_feature_reconstruction/summarize.py \
  --results-dir trained_models/SW0084_fixed_pilot \
  --output-json trained_models/SW0084_fixed_pilot/summary.json \
  --output-md trained_models/SW0084_fixed_pilot/summary.md
```

The summary reports all three fixed mask metrics and deltas for every arm and
epoch. No validation label informs training or a per-image prediction choice.
This is a seed1 experiment, not a multi-seed result. No preflight, training,
or evaluation was launched while preparing the scripts.

If all GPUs are occupied, register the durable evaluation queue instead. It
waits without model interaction, never restarts training, resumes already
completed evaluation JSONs, and writes a machine-readable queue state. It uses
any idle GPU0--3 unless a `tkim1` GPU process exists, in which case it restricts
itself to GPU0--1:

```bash
nohup bash collaborative_test/SW_0084_joint_feature_reconstruction/queue_evaluation.sh \
  </dev/null >trained_models/SW0084_EVALUATION_QUEUE.log 2>&1 &
```

When SW0086 was added while every GPU remained occupied, the original SW0084
waiter was replaced by `resume_after_sw0086.sh`. This gives the newer requested
cross-contract evaluation the first newly idle GPU and resumes every SW0084 job
afterward, avoiding a race in which two independent waiters claim one GPU.
