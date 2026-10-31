# SW_0083_threshold_recalibration

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `03803d4ebe2cdeeda6a01c9e8c2e92597618b040fd97453c7bf3d36dfa2be6cd`

````markdown
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

````

</details>

## results/summary.md

<details>
<summary>summary.md</summary>

Source SHA256: `778c81f7173c26cc005a64a96c598a53ed6ce955aeb069c6eb4bfade41525458`

````markdown
# SW0083 membrane-threshold recalibration

| vth | Event rate | Always-on | Constant histories | FG-ARI | FG IoU | Object IoU | All metrics improve |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.06 | 1.000000 | 1.000000 | 1.000000 | 0.708757 | 0.468105 | 0.509881 | no |
| 0.5 | 0.938791 | 0.701843 | 0.703094 | 0.595236 | 0.330631 | 0.481278 | no |
| 1 | 0.136033 | 0.000000 | 0.229675 | 0.016168 | 0.029373 | 0.026015 | no |
| 2 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | no |

Fixed aligned 32-image pilot; no per-image threshold adaptation.
Event diagnostics are computed from binary component spike histories after settle, before scoring.

````

</details>
