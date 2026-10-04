# SW_0002 — membrane-sourced synchrony pilot

Status: training and validation complete. This pilot does not meet the goal.

## 쉽게 설명하면

기존 학습은 Kuramoto의 위상(theta)이 잘 묶이는지만 loss로 봤습니다. 그래서
그 뒤에 있는 dendritic·membrane 층에는 학습 신호가 가지 않는다는 것을
확인했습니다. 이번에는 loss가 **membrane의 시간 패턴**을 보도록 바꾸고
10 epoch만 시험했습니다. 모델 구조와 최종 spike 분류법은 그대로입니다.

| 같은 검증 이미지 320장, spike 분류 | 물체 구분 FG-ARI ↑ | 전경 IoU ↑ | 물체별 IoU ↑ |
|---|---:|---:|---:|
| 기존 모델, 40 epoch | 0.1363 | 0.2720 | 0.1456 |
| membrane 기준 학습, 10 epoch | 0.2703 | 0.2355 | 0.1233 |

**해석:** 물체끼리 구분하는 점수는 올랐지만, 전경과 각 물체의 실제 위치를
맞추는 두 점수는 떨어졌습니다. 따라서 전체 개선은 아닙니다. 학습 길이도
40 대 10 epoch로 달라, 이 숫자만으로 loss 변경의 순수한 효과라고 할 수
없습니다. SW_0003에서 둘 다 40 epoch로 맞춰 비교합니다.

## Reason

Under the starting `plv_source=phase` objective, all PLV terms read Kuramoto
theta directly. A gradient-connectivity check on four training examples showed
zero connected gradient tensors in `dendric_layer` (0/3) and `membrane_layer`
(0/1), despite nonzero gradients in gamma projection, graph, and Kuramoto.
Thus the phase loss was not training the spike path. This is a causal property
of the current computation graph, not an inference from spike rate alone.

## Controlled change

Train seed 0 for 10 epochs with baseline model settings fixed except
`--plv-source membrane` instead of `--plv-source phase`. The starting baseline
checkpoint ran 40 epochs, so a direct score difference is confounded by training
duration; this is a short pilot, not a matched-epoch causal ablation. It measures centered
membrane-trace synchrony (not spike threshold events), connecting the objective
to the SNN layers. The unchanged `--spike-per-component` still produces the
final spike history. This pilot is not a replacement for the 40-epoch baseline.

The exact command is in `run.sh`. The server checkpoint is
`/Data0/kevinswk/patch_v2_sw/trained_models/SW_0002_membrane_plv_pilot_seed0/core.pt`.
The training objective fell from 3.495306 to 2.764854; that alone is not an
object-grouping result.

## Evaluation

`evaluate.sh` evaluated the original seed-0 checkpoint and this pilot on
exactly the same 320 validation images (IDs 1320–1639), with both the original
spatial-components readout and the fixed-k=8 aggregate-spike-synchrony readout
from SW_0001. The metric calculation uses the HDF5 instance mask converted to
16×16 modal-ID patches; no true object count is used in clustering.

| Seed-0 readout, validation 320 | FG-ARI | Foreground IoU | Matched object IoU | Actual spike rate |
|---|---:|---:|---:|---:|
| Baseline 40ep, spatial components | 0.000000 | 0.216821 | 0.014756 | 0.484850 |
| Pilot 10ep, spatial components | 0.000000 | 0.216821 | 0.014756 | 0.448761 |
| Baseline 40ep, spike synchrony k=8 | 0.136286 | 0.271997 | 0.145604 | 0.484850 |
| Pilot 10ep, spike synchrony k=8 | 0.270323 | 0.235452 | 0.123274 | 0.448761 |

The spike readout improves foreground ARI by 0.134037 but worsens foreground
IoU by 0.036545 and matched object IoU by 0.022331 relative to the starting
checkpoint. This is a tradeoff, not an all-metric improvement. The original
spatial-components classifier still assigns the entire image to one group.
No test-split metrics were used to select this pilot.

The full per-image results and patch-label predictions are on the server at
`$BASE/validation_fixed_split/{summary.json,patch_masks.pt}` and
`$PILOT/validation_fixed_split/{summary.json,patch_masks.pt}` (paths defined
in `evaluate.sh`).
