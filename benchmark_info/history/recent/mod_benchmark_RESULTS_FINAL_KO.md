# mod_benchmark/RESULTS_FINAL_KO.md

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
