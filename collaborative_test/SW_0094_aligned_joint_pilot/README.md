# SW0094 — aligned spike loss with graph and encoder adaptation

User restarted goal mode on 2026-10-06. Final target remains all three
spike/membrane mask metrics above comparable Slot Attention for seeds0/1/2,
including training on the full70,000 unique images. Short pilots are selection
evidence only, never a substitute for that target.

## Four registered pilot arms

| Arm | Spike-loss affinity | Graph | Encoder |
|---|---|---|---|
| absolute_frozen | Existing absolute-correlation product | Frozen | Frozen |
| positive_frozen | Actual classifier's positive-correlation product | Frozen | Frozen |
| positive_graph | Actual classifier's positive-correlation product | Trainable | Frozen |
| positive_joint | Actual classifier's positive-correlation product | Trainable | Trainable |

All continue the **entire** SW0090 seed0 epoch1 core, not just its graph. This is
the best measured70k single-seed checkpoint, not representative three-seed proof.
Common downstream/graph LR3e-5; joint encoder LR3e-6; Adam; clip1; 256 updates,
batch16; fixed shuffle seed17; same4,096 unique training examples sampled without
replacement from the existing70k pool. Full pool IDs0–999 plus1640–70639;
held-out IDs1000–1639 never enter training. The images/data contract is unchanged.
The script records every actual training ID and initial checkpoint hash.

Primary phase loss and all global affinity weights remain unchanged from
SW0090. Only spike-affinity semantics and parameter eligibility differ across
arms. No reconstruction term or GT instance loss is added. The joint arm uses
native RGB -> encoder -> scalar saved normalization/clip -> differentiable patch
pooling -> graph/core. It is intentionally not the earlier reconstruction-only
joint-training recipe. Encoder preprocessing statistics remain fixed to make
the comparison interpretable; learning those statistics is not tested here.

Training retains64 steps, settle32. Evaluation retains1024 steps, settle512,
threshold .50, largest component as background, spike component-positive
correlation product. Evaluate validation IDs1320–1399 (80 images), no readout
sweep. Joint evaluation uses **its updated encoder** to regenerate all320 cached
validation gamma rows; other arms retain the existing gamma. These are one-source
pilot scores, not full320 or three-seed results. The unchanged source's first80
per-image scores provide the immediate continuation control.

## Preflight and execution

`test_affinity.py` checks the sign counterexample, nonzero gradient on correlated
signals and finite forward/backward on constant traces. The coordinator also
requires all four actual-data CPU forward/backward/update preflights before GPU
launch. Each verifies finite loss/gradients, changed downstream parameters,
unchanged frozen graph, graph gradients when trainable, encoder gradients when
trainable, held-out exclusion and RGB/cache correspondence. Unused maximal-clique
group detection is skipped only during training because object-overlap loss is
zero; mask evaluation uses the actual unmodified classifier.

The first synthetic preflight caught and corrected a filename collision with
Python's standard `queue` module. The first asset import caught and corrected
the legacy core's package search-path requirement. Both happened before any GPU
training. Preserve these checks to avoid wasting training on startup errors.

Server:

```
/Data0/kevinswk/envs/snn/bin/python collaborative_test/SW_0094_aligned_joint_pilot/test_affinity.py
/Data0/kevinswk/envs/snn/bin/python collaborative_test/SW_0094_aligned_joint_pilot/coordinator.py --preflight
/Data0/kevinswk/envs/snn/bin/python collaborative_test/SW_0094_aligned_joint_pilot/coordinator.py --daemon
```

Outputs: `trained_models/SW0094_aligned_joint_pilot/`, including process PID,
atomic queue state, per-arm logs/manifests/history/checkpoints/evaluations and
final summary. The queue checks real GPU compute processes every60 seconds and
starts only on completely free GPU0/1. The colleague is currently using GPU3,
so the earlier user's restriction to0/1 applies. No busy-GPU sharing or stopping
other users. GPU/queue locks prevent duplicate copies of this queue. A failed
arm is terminal and requires diagnosis; the queue does not silently retry it.
Already-live arms may finish after another arm fails. If restarted, completed
arms are skipped; incomplete output directories require explicit inspection.

## Peer incorporation

Fetched `origin/patch_v2` through `e2389f5` in this cycle. Read
`PV2_0045_e2e_frozen_graph` and `PV2_0047_matched_slot`.

- Peer found severe collapse when adapting encoder inputs while freezing the
  image-conditioned graph. Therefore our encoder arm also trains graph parameters.
- Their newly trained Slot scores are on their own6,000-image split, two seeds,
  60,000 steps; they must not replace our matched70k three-seed baseline.
- Their reconstruction/end-to-end failures support testing actual spike
  affinity alignment first, rather than repeating only reconstruction weights.

## Next decision

### Completed CPU temporal diagnosis

`diagnose_windows.py` completed on unchanged SW0090 seed0/2 epoch1/10 cores,
validation IDs1320–1323 only. It uses no GPU, no optimization, and no GT in the
forward path. GT foreground pairs enter post-forward diagnostic scoring only.
Foreground pair AUC is computed using tie-averaged ranks (scipy), so no new
package installation is required. Raw32 image/checkpoint/window rows are in
`window_diagnosis.json`. This diagnostic is not an evaluation of the queued arms.

Mean actual classifier positive-product measurements on these4 images:

| Seed / epoch | Window / settle | FG pair AUC | Same-object edges above .50 | Different-object edges above .50 |
|---|---|---:|---:|---:|
| 0 / 1 | 64 / 32 | .9549 | .8958 | .1031 |
| 0 / 1 | 1024 / 512 | .9697 | .9253 | .0128 |
| 0 / 10 | 64 / 32 | .9269 | .8968 | .3649 |
| 0 / 10 | 1024 / 512 | .9575 | .8935 | .0057 |
| 2 / 1 | 64 / 32 | .8894 | .7891 | .2270 |
| 2 / 1 | 1024 / 512 | .9510 | .9109 | .0090 |
| 2 / 10 | 64 / 32 | .8642 | .3620 | .0295 |
| 2 / 10 | 1024 / 512 | .9613 | .4802 | .0007 |

Seed2's long-window AUC stays high while same-object edge recall roughly halves:
rank separation alone does not ensure sufficient edges for connected-component
grouping. This supports checking affinity calibration/within-object connectivity
as well as pair AUC. It does not establish activation death or prove which layer
caused the change. No threshold is selected from these scores.

Training absolute-affinity vs classifier-positive-affinity threshold disagreement
over **all** off-diagonal patch pairs is3.00–22.58% in the short window, but
0.00–3.54% in the long window. Thus the sign mismatch is meaningful in the actual
training window; its long-window size can be small. We must not attribute all
observed degradation to it. Short-window and long-window different-object edge
rates also differ substantially, motivating a later controlled temporal-window
or temporal-consistency loss test if the current aligned-loss arms cannot
preserve connectivity. This four-image diagnostic is too narrow to conclude
which mechanism dominates across the full validation set.

The four queued GPU arms remain unchanged, so their controlled comparison is
preserved. Extend this diagnosis to a larger fixed subset and per-layer traces
when the pilot results identify which arm needs explanation.

All four CPU asset preflights passed. RGB/cached gamma maximum error was zero;
the modified downstream parameters include Kuramoto, dendritic and membrane
tensors. The joint arm's first-step pre-clipping gradient norms were187.46 for
the core,21.21 for the graph and1182.82 for the encoder. Norm clipping is active;
nonzero gradients establish connectivity, not stable learning over many updates.
Training-vs-classifier thresholded edge disagreement was9.66% on this two-image
preflight; this is real checkpoint evidence of the sign mismatch, not a
representative held-out measurement or a proven cause of the metric decline.
Queue PID3546691 was verified live and waiting for free GPU0/1. Preflight
manifests/history and a timestamped queue snapshot are committed in this folder.
Synthetic checks and evaluator argument parsing also passed.

Compare positive_frozen against absolute_frozen for the objective change;
positive_graph against positive_frozen for graph adaptation; positive_joint
against positive_graph for encoder adaptation. Inspect gradient/edge disagreement
alongside all three metrics. Prefer balanced gains but retain useful FG-ARI or
mask-quality tradeoffs per SELECTION_POLICY. Expand promising arms to all320
validation images and three seed-specific starting cores, then full70k training.
Use independent test evidence for final success; training/validation pilot gains
cannot complete the goal. Do not discard a method based solely on one pilot seed.
