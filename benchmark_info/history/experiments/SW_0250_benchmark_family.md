# SW0250 모델/benchmark 계열

## mod_benchmark/DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `4cb2168b76d684645279d563b92e131b9d9d96f7c53801c30f1e531d4eb7a9f8`

````markdown
# SW0250: SW0240의 두 데이터셋 전이 benchmark

사용자 승인: 기존 240번의 세 독립 seed에서 추가 학습한다. benchmark/와 benchmark.md는 f9301ed 원문을 그대로 고정하며 코드·데이터 adapter는 이 폴더에만 작성한다. 보호 파일 11개의 SHA는 BENCHMARK_LOCK.json으로 실행 전후 확인한다.

데이터는 /work/USERS/tkim1/mod의 올바른 multi_dsprites(colored_on_colored, HWE) 및 tetrominoes(EHW) 계열이다. mdsprites 계열은 사용하지 않는다. 원래 manifest의 TRAIN 0..5999, validation 6000..6999, TEST 7000..7999를 유지한다. 별도 다운로드나 무작위 재분할은 하지 않는다.

각 데이터셋에 대해 zero_shot / finetune × seed0/1/2, 총 12개 작업이다. 기존 portable SW0240의 SW0249_matched35000_terminal에서 시작한다. zero_shot은 학습 없이 기존 전체 checkpoint를 사용한다. finetune은 TRAIN 6,000장×10회(60,000노출, batch16, 3,750 actual Adam updates), fresh Adam, LR 3e-5/3e-4/3e-4, gradient clip1로 encoder 전체와 실제 sensory projection만 추가 학습한다. 나머지 graph/core/integration/readout과 feature 통계는 고정한다. 기존 frozen teacher의 frequency denoising + gamma anchor mask-supervised objective를 유지한다. Teacher encoder/frequency 연산은 student와 같은 GPU에서 no_grad로 수행하여 CPU 병목을 줄인다. 새 encoder나 precomputed gamma로 checkpoint의 encoder를 교체하지 않는다.

고정 10회 종료 checkpoint만 평가하며 validation 선택·early stopping·loss 탐색은 하지 않는다. TEST mask는 학습에 사용하지 않는다. 각 1,000장 TEST prediction protocol을 완성한 다음 scoring에서 pixel truth를 읽는다. 원본 preflight.py는 실제 평가를 시작할 때마다 변경 없이 실행한다.

입력은 원본 benchmark/make128.py로 nearest 128×128 RGB를 준비한다. Native model은 두 데이터셋 모두 실제 64×64 SW0240을 유지한다. Multi-dSprites는 native64 예측을 그대로 평가한다. Tetrominoes는 원본 35픽셀에서 빈 cell이 생기는 grid64 GT를 만들지 않는다. 학습 시 기존 grid32 TRAIN partition만 2×2 반복하여 모델의 64×64 frequency에 대응시킨다. 평가 시 native64 예측을 원본 benchmark/make_targets_grid.py의 modal 규칙으로 32×32로 변환하고, 그 grid32를 같은 원본 함수의 35픽셀 cell edge로 확장한다. 이 고정 readout adapter를 zero_shot과 finetune에 동일 적용한다. Native32로 core를 재구성하거나 node parameter를 변환한 별도 모델 실험은 하지 않는다.

Pixel FG-ARI / foreground IoU / Hungarian-matched object IoU는 보호된 benchmark/evaluation.py의 원본 함수를 직접 호출한다. 추가로 pixel MBO(일대일 Hungarian matching과 구분), pred_fg, true_fg, native output group 수를 보고할 수 있다. 실제 TEST split의 oracle grid ceiling을 원본 modal/expand/evaluation 함수로 측정한다. 원래 validation300에서 측정한 0.9409/1.0000과 TEST ceiling을 혼동하지 않는다. 세 seed 평균과 sample standard deviation을 보고한다. 추가 학습 없는 모델도 source CLEVR 학습이 mask-supervised였다는 사실을 명시한다. 공개 논문 baseline 재현 없이 이 결과만으로 논문 모델을 이겼다고 주장하지 않는다.

실행 전 Linux 단위 검사·실제 두 데이터셋 CPU gradient 검사·TRAIN image0의 실제 CUDA forward/readout smoke를 통과해야 한다. 작업별 immutable binding, resumable optimizer/encoder/projection boundary, row별 prediction SHA, terminal model SHA와 dataset/file SHA를 보관한다. 원본 데이터·원본 checkpoint·다른 사용자의 프로세스는 수정하지 않는다. 여유 메모리 8GiB 이상의 GPU에서 공유 실행하며 외부 사용자가 나타났다는 이유로 실행 중인 작업을 자동 중단하지 않는다. 완료 후 구조/loss 탐색을 이어가지 않는다.

````

</details>

## mod_benchmark/RESULTS_360_KO.md

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

## mod_benchmark/RESULTS_FINAL_KO.md

<details>
<summary>RESULTS_FINAL_KO.md</summary>

Source SHA256: `df2e6443f76c51924eb72df97559ae3f661652ff20d8c9ac33dd4567d417301d`

````markdown
# SW0250: SW0240 두 데이터셋 benchmark 최종 결과

**12개 조건의 학습·평가 완료.** 12,000개 예측과 checkpoint의 독립 검증도 통과했습니다.

각 데이터셋의 분리된 고정 TEST 1,000장(ID 7000–7999), seed 0·1·2를 사용했습니다. 수치는 pixel metric의 세 seed 평균 ± seed 간 표본 표준편차(ddof=1)입니다.

## 전체 TEST 1,000장

| 데이터셋 | 방식 | FG-ARI | Foreground IoU | Matched Object IoU |
|---|---|---:|---:|---:|
| Multi-dSprites | 기존 checkpoint | 0.683310 ± 0.038195 | 0.662551 ± 0.140285 | 0.722921 ± 0.024252 |
| Multi-dSprites | 해당 데이터로 추가 학습 | 0.770147 ± 0.006693 | 0.826195 ± 0.038971 | 0.777474 ± 0.023378 |
| Tetrominoes | 기존 checkpoint | 0.617843 ± 0.168734 | 0.460512 ± 0.075859 | 0.674918 ± 0.135794 |
| Tetrominoes | 해당 데이터로 추가 학습 | 0.801673 ± 0.003663 | 0.712728 ± 0.048643 | 0.819070 ± 0.010636 |

## 먼저 요청한 360장: ID 7000–7359

| 데이터셋 | 방식 | FG-ARI | Foreground IoU | Matched Object IoU |
|---|---|---:|---:|---:|
| Multi-dSprites | 기존 checkpoint | 0.697877 ± 0.040924 | 0.657132 ± 0.137152 | 0.719082 ± 0.018058 |
| Multi-dSprites | 해당 데이터로 추가 학습 | 0.786001 ± 0.006983 | 0.820230 ± 0.041277 | 0.773833 ± 0.023193 |
| Tetrominoes | 기존 checkpoint | 0.616402 ± 0.165702 | 0.453587 ± 0.076693 | 0.670598 ± 0.138997 |
| Tetrominoes | 해당 데이터로 추가 학습 | 0.799808 ± 0.004009 | 0.706255 ± 0.050569 | 0.819868 ± 0.012648 |

## 추가 640장: ID 7360–7999

| 데이터셋 | 방식 | FG-ARI | Foreground IoU | Matched Object IoU |
|---|---|---:|---:|---:|
| Multi-dSprites | 기존 checkpoint | 0.675117 ± 0.036695 | 0.665599 ± 0.142047 | 0.725080 ± 0.028639 |
| Multi-dSprites | 해당 데이터로 추가 학습 | 0.761230 ± 0.006595 | 0.829550 ± 0.037714 | 0.779523 ± 0.023902 |
| Tetrominoes | 기존 checkpoint | 0.618654 ± 0.170441 | 0.464407 ± 0.075756 | 0.677348 ± 0.134012 |
| Tetrominoes | 해당 데이터로 추가 학습 | 0.802723 ± 0.003554 | 0.716369 ± 0.047732 | 0.818622 ± 0.009578 |

## 전체 1,000장 seed별 결과

| 데이터셋 | 방식 | seed | FG-ARI | Foreground IoU | Matched Object IoU |
|---|---|---:|---:|---:|---:|
| Multi-dSprites | 해당 데이터로 추가 학습 | 0 | 0.770416 | 0.870532 | 0.800997 |
| Multi-dSprites | 해당 데이터로 추가 학습 | 1 | 0.776702 | 0.797362 | 0.777183 |
| Multi-dSprites | 해당 데이터로 추가 학습 | 2 | 0.763324 | 0.810689 | 0.754243 |
| Multi-dSprites | 기존 checkpoint | 0 | 0.705195 | 0.811516 | 0.749475 |
| Multi-dSprites | 기존 checkpoint | 1 | 0.705528 | 0.643176 | 0.717345 |
| Multi-dSprites | 기존 checkpoint | 2 | 0.639208 | 0.532961 | 0.701942 |
| Tetrominoes | 해당 데이터로 추가 학습 | 0 | 0.805693 | 0.755327 | 0.829831 |
| Tetrominoes | 해당 데이터로 추가 학습 | 1 | 0.800802 | 0.723132 | 0.818816 |
| Tetrominoes | 해당 데이터로 추가 학습 | 2 | 0.798524 | 0.659725 | 0.808564 |
| Tetrominoes | 기존 checkpoint | 0 | 0.745325 | 0.524785 | 0.781662 |
| Tetrominoes | 기존 checkpoint | 1 | 0.681705 | 0.479914 | 0.721014 |
| Tetrominoes | 기존 checkpoint | 2 | 0.426500 | 0.376836 | 0.522076 |

## 해석 및 고정 조건

- 두 데이터셋 모두 추가 학습 후 세 지표의 평균이 개선됐습니다. 서로 다른 데이터셋의 점수 차이만으로 과적합 여부를 판정하지 않습니다.
- 기존 checkpoint는 CLEVR에서 학습된 240의 세 독립 seed입니다. zero_shot은 새 데이터셋에서의 추가 학습이 없다는 뜻이며, 원래 240도 mask-supervised 모델입니다.
- 추가 학습은 각 seed의 240에서 fresh Adam으로 encoder 전체와 sensory projection만 학습했습니다. graph/core/integration/readout과 feature 통계는 고정했습니다. 처음부터 전체 모델을 재학습한 실험이 아닙니다.
- TRAIN ID 0–5999의 6,000장 × 10 epoch, batch 16, 3,750 updates, 60,000회 노출입니다. LR은 encoder 3e-5, context encoder·projection 각각 3e-4, gradient clip 1입니다.
- 고정 마지막 epoch checkpoint를 평가했습니다. TEST를 이용한 checkpoint 선택·loss 탐색은 하지 않았습니다.
- Multi-dSprites는 colored_on_colored의 올바른 multi_dsprites 파일만 사용했습니다. 잘못된 축 순서의 mdsprites 계열은 사용하지 않았습니다.
- native model은 64×64입니다. Multi-dSprites는 grid64 그대로, Tetrominoes는 원본 mask_cells의 modal 규칙으로 grid32에 변환한 뒤 원본 cell_index로 35×35에 확장했습니다. Tetrominoes TRAIN grid32 타겟은 2×2 반복으로 native64에 대응했습니다.
- 입력은 원본 make128.py의 nearest 128×128 RGB입니다. 제공된 별도 encoder나 사전 계산 gamma로 240 encoder를 교체하지 않았습니다.
- 평가 전마다 원본 preflight.py를 실행했습니다. benchmark.md와 benchmark/의 보호 파일 11개는 f9301ed SHA를 유지했습니다. 점수는 원본 evaluation.py로 native pixel에서 계산했습니다.
- MBO는 Hungarian-matched Object IoU와 다른 보조 지표이며 results_final.json에 별도로 보관했습니다.
- 같은 규약으로 외부 모델을 재평가하지 않았으므로 이 표만으로 Slot Attention 또는 다른 논문 모델을 이겼다고 주장하지 않습니다.
- 360장 보고 후 나머지 평가를 재개했습니다. GPU 상태 조회 시간 초과로 coordinator가 종료된 이후에도 기존 worker는 계속 실행됐습니다. 학습·예측 재시작 없이 완료 파일을 검증하고 원래 summarizer로 종료 기록을 정리했습니다.

## 재현 및 실행 증거

과학 코드: 이 폴더의 common.py, worker.py, queue.py, prepare.py, smoke.py, bootstrap.py와 ../snn_kuramoto_bidirectional/. 보호된 평가 코드는 ../benchmark/입니다.

동일한 경로·의존성·원본 checkpoint가 준비된 환경의 실행 명령:

```bash
python -u -B -m mod_benchmark.bootstrap --archive <new_output_directory>
```

실제 서버 출력 경로: /Data0/kevinswk/patch_v2_sw/collaborative_testing/generated/SW0250_mod_benchmark/. jobs/<dataset>/<mode>/seedN/ 아래 binding.json, training_report.json(finetune), preflight/, prediction_protocol.json, rows/, per_image_scores.npz, result.json에 원본 실행 증거가 있습니다.

로컬 results_final.json에는 세 평가 구간·seed별 수치·파일 SHA를 보관했습니다. 독립 검증 원문은 생성 파일 폴더의 SW0250_mod_benchmark/final_audit.json입니다.

````

</details>

## mod_benchmark/RESULTS_SEED0_KO.md

<details>
<summary>RESULTS_SEED0_KO.md</summary>

Source SHA256: `2bde5f02a8f7026f6e43df3f3648f64318a64090a5d7ae40e50409d69d77e043`

````markdown
# SW0250 seed0 ?? ??

**?? ??: seed0? ? ??? ????. seed1?2? ? seed ??? ?? ????.**

??? TEST ID 7000?7999? ??? 1,000?? ????. ? 360?? ??? 640?? ?? ?? ?? ?? 1,000?? ???. ?? ???? ???? threshold ??? ?? ???.

## ? 360?: ID 7000?7359

| ???? | ?? | FG-ARI | Foreground IoU | Matched Object IoU |
|---|---|---:|---:|---:|
| multi_dsprites | zero_shot | 0.719519 | 0.802697 | 0.739914 |
| multi_dsprites | finetune | 0.787439 | 0.866768 | 0.799352 |
| tetrominoes | zero_shot | 0.740911 | 0.524468 | 0.781968 |
| tetrominoes | finetune | 0.804415 | 0.747060 | 0.831822 |

## ??? 640?: ID 7360?7999

| ???? | ?? | FG-ARI | Foreground IoU | Matched Object IoU |
|---|---|---:|---:|---:|
| multi_dsprites | zero_shot | 0.697138 | 0.816477 | 0.754853 |
| multi_dsprites | finetune | 0.760840 | 0.872650 | 0.801922 |
| tetrominoes | zero_shot | 0.747809 | 0.524963 | 0.781491 |
| tetrominoes | finetune | 0.806412 | 0.759976 | 0.828711 |

## ?? 1,000?: ID 7000?7999

| ???? | ?? | FG-ARI | Foreground IoU | Matched Object IoU |
|---|---|---:|---:|---:|
| multi_dsprites | zero_shot | 0.705195 | 0.811516 | 0.749475 |
| multi_dsprites | finetune | 0.770416 | 0.870532 | 0.800997 |
| tetrominoes | zero_shot | 0.745325 | 0.524785 | 0.781662 |
| tetrominoes | finetune | 0.805693 | 0.755327 | 0.829831 |

zero_shot? ?? ????? ?? ?? ??? ??. finetune? ?? SW0240?? TRAIN 6,000? ? 10?, batch16, Adam 3,750 updates? encoder? sensory projection? ?? ????. frozen core? ?? mask-supervised loss? ??? supervised ????.

Multi-dSprites? ??? colored_on_colored multi_dsprites ??, grid64? ????. Tetrominoes? checkpoint native64? ???? ?? mask_cells? modal rule? grid32 ???? ???? pixel metric? ????. benchmark f9301ed ???? ???? ??? ? ??? ?? preflight? ????.

?? ??? result/protocol/per-image score SHA, ?? benchmark SHA, ?? TEST ID, preflight ??, ?? ? 360? ?? ??? ????. ?? 12? ??? ?? ? checkpoint ??? ?? pixel evaluator ???? ??? ?? ?? ??? ??? ????.

````

</details>
