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

## Result

Server syntax checks, four focused unit tests, and the immutable SW0081
preflight validator passed. Training completed 785 updates on GPU0. The encoder
and every non-graph core tensor stayed bitwise unchanged; the graph maximum
absolute change was `.00790281`. Epoch 5 matched the saved final checkpoint
bitwise and reproduced the independent SW0081 0.1x evaluation exactly.

| checkpoint | FG-ARI | foreground IoU | matched-object IoU |
|---|---:|---:|---:|
| SW0072 baseline | .708757 | .468105 | .509881 |
| epoch 1 | .699289 | .463458 | .504113 |
| epoch 2 | .697220 | .467795 | .507192 |
| epoch 3 | .686575 | .465248 | .507033 |
| epoch 4 | .682034 | .463918 | .501284 |
| epoch 5 | .706200 | .464029 | .513687 |

No epoch improves all three metrics. Epoch 2 comes closest on foreground IoU
but remains below baseline on every metric, and epoch 5 improves only
matched-object IoU. Horizontal-flip graph equivariance is therefore stopped:
there is no further weight or epoch sweep and no full-320 promotion.
