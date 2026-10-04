# SW_0026: calibrate per-component spike-product threshold

Status: complete; seed-0 classifier-only validation, no production merge.

## 결과와 판단

동료의 낮은 product threshold가 우리 checkpoint에는 그대로 이전되지 않았습니다.
같은 320장과 같은 checkpoint에서 SW_0004는 threshold 0.50이 세 지표 모두
0.10 이하보다 좋았습니다. SW_0011은 0.10에서 foreground IoU가 높아지는
별도 후보가 있지만, FG-ARI와 object IoU는 0.50보다 낮습니다. 따라서 낮은
threshold를 전면 채택하지 않고, foreground-IoU track의 후보로 보존합니다.

| checkpoint / threshold | FG-ARI ↑ | foreground IoU ↑ | matched-object IoU ↑ | predicted groups/image |
|---|---:|---:|---:|---:|
| SW_0003 / 0.50 control | 0.076784 | 0.092284 | 0.036012 | 1.69 |
| SW_0003 / 0.01 | 0.034007 | 0.042532 | 0.013385 | 0.51 |
| SW_0004 / 0.50 control | 0.195269 | 0.275711 | 0.238803 | 30.99 |
| SW_0004 / 0.01 | 0.120160 | 0.246947 | 0.153101 | 11.03 |
| SW_0011 / 0.50 control | 0.179760 | 0.271468 | 0.295250 | 77.03 |
| SW_0011 / 0.10, foreground-IoU track | 0.098892 | 0.331330 | 0.180793 | 14.14 |

The true mean foreground-object count is 6.20. A lower threshold joins
more oscillators, often merging objects or swallowing them into the largest
component designated background. These are validation-only seed-0 results;
the peer's different render/split cannot be compared score-for-score. The
full threshold tables are in `sw0003_validation320.json`,
`sw0004_validation320.json`, and `sw0011_validation320.json`.

## 쉽게 설명하면

동료가 네 개 spike 성분의 유사도를 곱하면 값이 아주 작아진다는 점을
발견했습니다. 우리가 너무 높은 threshold로 분류해서 유용한 spike
그룹을 잘라냈을 가능성이 있습니다. 같은 학습된 코어를 다시 학습하지
않고, 낮은 threshold만 시험합니다.

## Exact setup

- Frozen existing seed-0 checkpoints SW_0003 (membrane PLV 40ep), SW_0004
  (phase PLV + membrane graph teacher 10ep), and SW_0011 (same 40ep).
- Actual per-component **spike** histories, 256 steps with 64 discarded;
  existing signed centered correlation product, connected components, and
  largest component background. No GT or oracle object count in prediction.
- Thresholds .0005/.002/.005/.01/.02/.04/.08/.1/.2/.5; .5 is the old
  control. All 320 fixed HDF5 validation IDs 1320-1639 and the shared 16x16
  patch evaluator. No reference-test selection.
- Compare FG-ARI first, but keep both IoUs and group counts. Any improvement
  remains seed-0 validation until frozen settings pass three-seed testing.

## Code changes

`SW_0006_peer_spike_components/evaluate.py` gains optional `--thresholds`
while preserving its old defaults exactly. This folder adds only the
reproducible sweep command and report. No core, training, loss, or production
classifier code is modified.
