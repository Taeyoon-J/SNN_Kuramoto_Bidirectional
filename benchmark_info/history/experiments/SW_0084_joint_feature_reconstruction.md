# SW_0084_joint_feature_reconstruction

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `820cc69b0de43e6607fd12a130d2fda729d6ec2ff9c1005d22f7e7ffa9bb76a1`

````markdown
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

## Completed-training static diagnostic

All four arms completed 1,570 updates and wrote validated epoch5/10
checkpoints. Before the queued mask evaluation, the final manifests show:

| Arm | Final total loss | Final primary | Final spike | Gamma RMS drift | Graph max change | Core max change |
|:---:|---:|---:|---:|---:|---:|---:|
| A | 21.1286 | 4.4749 | 3.3307 | .1551 | .07995 | .33946 |
| B | 20.7217 | 3.7606 | 3.3922 | .1244 | .07068 | .35197 |
| C | **19.2333** | 3.9501 | **3.0566** | **.0231** | **.06529** | .34328 |
| D | 20.4952 | **3.7398** | 3.3511 | .1346 | .08559 | .40289 |

Arm C, initialized from SW0072, changes the feature basis and graph least while
ending with the lowest total and component-spike losses. This is evidence that
the stable initialization constrains co-adaptation, but it is not performance
evidence: prior experiments show that lower training loss need not predict the
three mask metrics.

The reconstruction term itself is nearly flat. It changes from `.6277` to
`.6261` in A, `.6398` to `.6461` in C, and `.6288` to `.6242` in D. Therefore
the queued fixed-mask evaluation remains necessary, and the next conditional
test (SW0088) preserves every downstream core tensor while allowing only the
encoder and graph to co-adapt.

````

</details>

## results/summary.md

<details>
<summary>summary.md</summary>

Source SHA256: `c24449bef3fec59ae1482de38457e574014b8f9b384d3f96972844f5a023633a`

````markdown
# SW0084 fixed seed1 pilot

| Arm | Epoch | FG-ARI | Δ | FG IoU | Δ | Object IoU | Δ | All three improve |
|:---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| A | 05 | 0.573703 | -0.135053 | 0.443988 | -0.024117 | 0.370688 | -0.139192 | no |
| A | 10 | 0.646381 | -0.062376 | 0.489311 | +0.021206 | 0.453914 | -0.055967 | no |
| B | 05 | 0.556742 | -0.152015 | 0.458074 | -0.010031 | 0.370865 | -0.139015 | no |
| B | 10 | 0.613345 | -0.095412 | 0.472576 | +0.004471 | 0.439784 | -0.070096 | no |
| C | 05 | 0.576709 | -0.132048 | 0.530123 | +0.062018 | 0.448355 | -0.061526 | no |
| C | 10 | 0.608609 | -0.100148 | 0.630744 | +0.162639 | 0.471259 | -0.038621 | no |
| D | 05 | 0.608041 | -0.100716 | 0.370725 | -0.097380 | 0.402920 | -0.106960 | no |
| D | 10 | 0.622237 | -0.086520 | 0.397004 | -0.071100 | 0.455198 | -0.054683 | no |

aligned IDs1320-1351, T256/settle64, vth .06, spike CC threshold .35
report fixed epoch checkpoints; no per-image or threshold selection
Interpret joint-loss and initialization controls together; this is one seed's pilot.

````

</details>
