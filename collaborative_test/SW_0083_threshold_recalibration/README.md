# SW0083: frozen-core membrane-threshold recalibration

This is an inference-only diagnostic on the frozen SW0072 seed1 core. Its
motivation is the reported 98.82% binary crossing rate at vth `.06`; SW0036
also found that vth `2.0` restored event variation on an earlier frozen
checkpoint, but that result predates the stable SW0072 graph. This sweep checks
whether the same intervention changes the current fixed spike readout.

Every run uses aligned validation IDs1320-1351, the same cached gamma, 256
steps, settle64, the learned graph and all other loaded weights unchanged,
spike connected-components at fixed synchrony threshold `.35`, minimum group
size2, and largest-component background. The only changed parameter is the
single global membrane threshold: `.06` control, `.5`, `1.0`, or `2.0`. There
is no per-image adaptation or threshold selection from labels.

The evaluator's opt-in `--event-diagnostics` hooks each membrane-layer update
and records the hard crossing `(membrane > current v_th)` before gated spike
amplitude is applied. Captures are checked and reshaped to
`[batch, component, region, time]`; the post-settle report contains binary event
rate, always-on fraction, constant-history fraction, and mean temporal standard
deviation. These statistics use binary crossings only and are computed
independently of ground-truth labels. Segmentation
metrics are scored only after prediction formation. The experiment does not
claim a trained improvement: the checkpoint was trained with vth `.06`.

Run each fixed condition sequentially on an idle GPU0 or GPU1; the script
checks GPU occupancy and refuses existing JSON/log outputs:

```bash
bash collaborative_test/SW_0083_threshold_recalibration/run_all.sh 0
```

Summarize the four results:

```bash
python collaborative_test/SW_0083_threshold_recalibration/summarize.py \
  --results-dir trained_models/SW0083_threshold_recalibration \
  --output-json trained_models/SW0083_threshold_recalibration/summary.json \
  --output-md trained_models/SW0083_threshold_recalibration/summary.md
```

The table reports fixed-readout FG-ARI, foreground IoU, matched-object IoU,
and binary event diagnostics relative to vth `.06`. It is a seed1 32-image
pilot and does not select a threshold for deployment or retraining. No GPU
evaluation uses a validation-adaptive threshold.

## Result

Server syntax checks, four focused unit tests, and a real one-image checkpoint
smoke passed before the fixed sweep. The smoke caught and corrected the hook's
actual `[batch*components, regions]` layout before the registered results were
created.

| vth | binary event rate | always on | FG-ARI | foreground IoU | matched-object IoU |
|---:|---:|---:|---:|---:|---:|
| .06 | 1.000000 | 1.000000 | .708757 | .468105 | .509881 |
| .5 | .938791 | .701843 | .595236 | .330631 | .481278 |
| 1.0 | .136033 | 0 | .016168 | .029373 | .026015 |
| 2.0 | 0 | 0 | 0 | 0 | 0 |

Raising vth to `.5` restores some temporal variation but lowers all three
metrics. At `1.0` the readout nearly collapses, and at `2.0` no crossing remains.
Thus SW0072's useful synchrony is carried primarily by continuous gate amplitude
under this configuration; inference-time threshold recalibration cannot recover
an additional binary-timing signal. Stop this direction without a finer sweep or
full-320 promotion.
