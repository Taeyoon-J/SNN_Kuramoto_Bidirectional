# SW_0082_graph_flip_trajectory

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `6df30e1fb4b2ea1cdbd19e367aea51164d762e68a7554d62de06f29232abc826`

````markdown
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

````

</details>

## results/trajectory_summary.md

<details>
<summary>trajectory_summary.md</summary>

Source SHA256: `ec22082f3789d1547199f060ba51c0604348f9f87b6bde11acbf1321a81ca4bf`

````markdown
# SW0082 fixed pilot trajectory

| Epoch | FG-ARI | Δ | Foreground IoU | Δ | Matched-object IoU | Δ | Advance |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 0.699289 | -0.009468 | 0.463458 | -0.004646 | 0.504113 | -0.005768 | no |
| 2 | 0.697220 | -0.011536 | 0.467795 | -0.000310 | 0.507192 | -0.002689 | no |
| 3 | 0.686575 | -0.022182 | 0.465248 | -0.002857 | 0.507033 | -0.002848 | no |
| 4 | 0.682034 | -0.026723 | 0.463918 | -0.004186 | 0.501284 | -0.008596 | no |
| 5 | 0.706200 | -0.002556 | 0.464029 | -0.004075 | 0.513687 | +0.003806 | no |

aligned IDs1320-1351, T256/settle64, spike CC threshold .35
an epoch advances only when all three deltas are strictly positive
No threshold or epoch was selected using training labels.

````

</details>
