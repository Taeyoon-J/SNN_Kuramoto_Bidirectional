# SW0052 - seed2 checkpoint trajectory

SW0050 showed that both a lower learning rate and a cross-sample activity
diversity loss rescue the collapsed seed2 final checkpoint, with different
metric tradeoffs. This experiment evaluates the already-saved epochs 20, 25,
30, and 35 to test whether either trajectory contains a better early-stopping
point before epoch 40.

The first screen uses the fixed aligned HDF5 validation IDs 1320-1639 and the
short T256/settle64 window. Every checkpoint must first pass a real four-image
forward/scoring smoke test. The smoke validates checkpoint loadability, finite
model tensors, five finite readout rows, the requested checkpoint path, held-out
IDs, and `ground_truth_used_for_prediction=false`. Full outputs refuse
overwrite. The best predeclared short candidates will then receive the long
T1024/settle512 evaluation; epoch 40 is already available from SW0050.

Three workers can screen the eight intermediate checkpoints on idle GPUs:

```bash
bash collaborative_test/SW_0052_checkpoint_trajectory/launch_short_parallel.sh 1 2 3
```

Before reading the screen, long-window promotion is fixed to the common
threshold `.35`: promote every non-dominated epoch across the three mask
metrics, plus any per-metric maximum not already included. Epoch 40 from
SW0050 participates in this selection without being rerun at short window.
No new model is trained and no test split is used for selection.

## Results

The common-threshold `.35` short screen selected exactly three Pareto rows:

| Arm / epoch | Short FG-ARI | Short foreground IoU | Short object IoU |
|---|---:|---:|---:|
| low LR / 25 | **.360243** | .174951 | **.125087** |
| diversity / 30 | .330192 | .210209 | .099338 |
| diversity / 35 | .301196 | **.231187** | .094342 |

Their long T1024/settle512 results at the same `.35` threshold are:

| Arm / epoch | Long FG-ARI | Long foreground IoU | Long object IoU |
|---|---:|---:|---:|
| low LR / 25 | **.395707** | .189112 | **.159085** |
| diversity / 30 | .330248 | .222187 | .157874 |
| diversity / 35 | .302799 | **.231287** | .132006 |

Low-LR epoch 25 strictly dominates its epoch-40 result
`.347156/.167773/.135250`. This identifies early stopping within the low-LR
trajectory as the strongest current seed2 intervention. Diversity epochs
30/35 retain the foreground-IoU track but do not dominate low LR on the two
grouping metrics. Full short/long JSONs are under `results/`.
