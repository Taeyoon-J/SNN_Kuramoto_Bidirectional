# mod_benchmark/RESULTS_360_KO.md

<details>
<summary>RESULTS_360_KO.md</summary>

Source SHA256: `4123b30695ab893d02638d3d690637856b6a403749d46ef55d1b13450924ed22`

````markdown
# SW0250: 고정 TEST 360장 평가

완료: **12/12개 조건**. 모든 조건의 seed 0·1·2 평가 완료.

두 데이터셋 모두 고정 TEST ID 7000–7359를 사용했습니다. benchmark f9301ed의 원본 preflight와 pixel 평가 함수를 사용하며, 360장 결과로 checkpoint를 선택하거나 모델을 수정하지 않았습니다.

## Seed별 실제 결과

| 데이터셋 | 방식 | seed | fg_ari | foreground_iou | matched_object_iou | mbo |
|---|---|---:|---:|---:|---:|---:|
| multi_dsprites | finetune | 0 | 0.787439 | 0.866768 | 0.799352 | 0.806465 |
| multi_dsprites | finetune | 1 | 0.792153 | 0.788048 | 0.768108 | 0.776052 |
| multi_dsprites | finetune | 2 | 0.778410 | 0.805875 | 0.754038 | 0.761169 |
| multi_dsprites | zero_shot | 0 | 0.719519 | 0.802697 | 0.739914 | 0.747041 |
| multi_dsprites | zero_shot | 1 | 0.723435 | 0.638377 | 0.707899 | 0.718773 |
| multi_dsprites | zero_shot | 2 | 0.650676 | 0.530324 | 0.709432 | 0.716904 |
| tetrominoes | finetune | 0 | 0.804415 | 0.747060 | 0.831822 | 0.839511 |
| tetrominoes | finetune | 1 | 0.797903 | 0.722025 | 0.821156 | 0.829565 |
| tetrominoes | finetune | 2 | 0.797107 | 0.649681 | 0.806625 | 0.815562 |
| tetrominoes | zero_shot | 0 | 0.740911 | 0.524468 | 0.781968 | 0.789285 |
| tetrominoes | zero_shot | 1 | 0.679965 | 0.464121 | 0.715001 | 0.722460 |
| tetrominoes | zero_shot | 2 | 0.428329 | 0.372172 | 0.514825 | 0.516841 |

## 세 seed 평균 ± 표준편차

표준편차는 seed 간 표본 표준편차(ddof=1)입니다.

| 데이터셋 / 방식 | fg_ari | foreground_iou | matched_object_iou | mbo |
|---|---:|---:|---:|---:|
| multi_dsprites/zero_shot | 0.697877 ± 0.040924 | 0.657132 ± 0.137152 | 0.719082 ± 0.018058 | 0.727572 ± 0.016886 |
| multi_dsprites/finetune | 0.786001 ± 0.006983 | 0.820230 ± 0.041277 | 0.773833 ± 0.023193 | 0.781229 ± 0.023087 |
| tetrominoes/zero_shot | 0.616402 ± 0.165702 | 0.453587 ± 0.076693 | 0.670598 ± 0.138997 | 0.676196 ± 0.141992 |
| tetrominoes/finetune | 0.799808 ± 0.004009 | 0.706255 ± 0.050569 | 0.819868 ± 0.012648 | 0.828213 ± 0.012032 |

## 해석 및 재현 조건

- zero_shot: 기존 CLEVR 학습 SW0240을 새 데이터셋에서 추가 학습 없이 평가.
- finetune: 같은 seed의 SW0240에서 TRAIN 6,000장 × 10 epoch (batch 16, 3,750 updates)를 추가 학습한 최종 checkpoint 평가.
- 기존 SW0240 및 추가 학습에는 mask supervision이 있으므로 unsupervised 결과로 해석하지 않습니다.
- 모델 native grid는 두 데이터셋 모두 64입니다. Multi-dSprites는 grid64, Tetrominoes는 원본 mask_cells 규칙으로 grid32에 투영한 뒤 native pixel로 평가합니다.
- MBO가 표시되어 있다면 matched-object IoU와 별개의 보조 지표입니다.
- 나머지 TEST ID 7360–7999의 640장과 전체 1,000장 결과는 이 표와 구분합니다.
- 서버 subset360/<dataset>/<mode>/seedN/의 prediction_protocol.json, preflight/, per_image_scores.npz, result.json에 실행 근거가 남아 있습니다.

````

</details>
