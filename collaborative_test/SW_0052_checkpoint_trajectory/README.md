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

No new model is trained and no test split is used for selection.
