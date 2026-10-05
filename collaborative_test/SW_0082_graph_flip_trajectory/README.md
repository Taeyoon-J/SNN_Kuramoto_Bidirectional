# SW0082: SW0081 graph-equivariance epoch trajectory

This is a single seed1 run of the SW0081 0.1x-gradient arm with an exact
checkpoint saved at the end of each of its five epochs. The starting SW0072
core, registered encoder/statistics, cached gamma, shuffled training order,
phase binding objective, 5x component-spike objective, learning rate
`3e-5`, and equivariance weight are unchanged. The only training-loop addition
is writing the epoch snapshots. The trainer compares the final checkpoint with
epoch 5 bitwise and fails if they differ.

Before training, the launcher validates the SW0081 one-update preflight marker
against the original source checkpoint, gamma, encoder, statistics, SW0081
trainer/helper, and all registered model/objective code dependencies. It also
recomputes the 0.1x weight formula from the recorded gradient norms. This
prevents reuse of a stale calibration marker. The run is GPU0-only and refuses
existing trajectory outputs.

Run on an idle GPU0 only after the SW0081 real-asset preflight is present:

```bash
bash collaborative_test/SW_0082_graph_flip_trajectory/run.sh 0
```

Evaluate SW0072 baseline and checkpoints from epochs 1-5 on the same fixed
seed1 pilot slice IDs1320-1351, with T256/settle64 and spike connected
components threshold `.35`:

```bash
bash collaborative_test/SW_0082_graph_flip_trajectory/evaluate.sh 0 baseline
bash collaborative_test/SW_0082_graph_flip_trajectory/evaluate.sh 0 sw0081
for e in 1 2 3 4 5; do
  bash collaborative_test/SW_0082_graph_flip_trajectory/evaluate.sh 0 "$e"
done
python collaborative_test/SW_0082_graph_flip_trajectory/summarize.py \
  --results-dir trained_models/SW0082_fixed_pilot \
  --sw0081-reference trained_models/SW0082_fixed_pilot/sw0081_seed1_n32.json \
  --output-json trained_models/SW0082_fixed_pilot/trajectory_summary.json \
  --output-md trained_models/SW0082_fixed_pilot/trajectory_summary.md
```

The summary reports every epoch's FG-ARI, foreground IoU, matched-object IoU,
and deltas from baseline. An epoch advances only if all three deltas are
strictly positive; the summary does not choose a threshold or alter training.
Training opens RGB images and cached gamma only, never masks or object counts.
Scoring labels are used only after each checkpoint's predictions are formed.
No GPU preflight, training, or evaluation has been launched for SW0082.
