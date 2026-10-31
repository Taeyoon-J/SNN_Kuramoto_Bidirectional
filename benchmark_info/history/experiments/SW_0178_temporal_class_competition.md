# SW_0178_temporal_class_competition

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `acbda72c31e568ff19fdb1a314f7af3a360bcb0edb922b99e927c80f8e5d1507`

````markdown
# SW0178: Temporal class competition

## Question

Does class-balanced leave-one-out competition over full temporal labels improve
the registered native32 task, and does using signed component mean affinity
help relative to the matched production positive-product affinity?

## Frozen comparison

Two fresh model instances start from identical pretrained SW0097 SourceT
weights and zero-context parameters. Neither arm reuses trained weights,
heads, or rollout hooks from SW0175, SW0176, or SW0177. Both use the same SW0175
four optimizer families, rates, shuffle order, and native rollout:

- `mean_rho_ce`: for each image, use the mean of the four signed component
  correlations as pair affinity.
- `product_q_ce`: use the frozen production affinity, the product of the four
  component correlations after each component is clamped below at zero.

Both arms train on all 512 registered TRAIN images for two passes (64 logical
updates, B16 with microbatches of four), using seed 171001 plus the zero-based
logical update index for each update. The two NumPy shuffle permutations are
drawn consecutively from `RandomState(171001)`. The
rollout is the original SourceT recurrence with 1024 steps, settle point 512,
and live tail 64. This experiment does not reuse learned weights from earlier
namespaces and does not add a SourceF restriction to the training objective.

## Objective

For each image, each patch is labeled with its full modal target, including
background label 0. For a query pixel, each class prototype is the mean
affinity to pixels of that class. The query is omitted from its own class
prototype. A singleton query is omitted because it has no leave-one-out own
prototype; singleton classes remain eligible as competitors for other queries.
For each query, cross-entropy uses temperature 0.1 over the available classes.
Losses are averaged within each query class, then over classes with valid
queries, then over images. An image with no valid class contributes zero to
the batch mean. GT classes/counts are used only by this TRAIN loss; inference
does not use GT count or a GT-derived K.

The `mean_rho_ce` arm uses the signed mean of four live component correlations;
the `product_q_ce` control uses the exact native positive-product Q. The latter
is computed with the frozen production operation order. No additional loss,
lambda, threshold sweep, or source-target foreground restriction is added.

## Readout and decision

Inference uses actual emitted spikes S, native production Q, and the original
SW0165 solver. Conditional readout uses the registered SourceT foreground;
standalone readout uses its own native QCC foreground. All predictions for
both arms are frozen and validated before opening validation ground truth.
Each arm is assessed against SW0164 seed 1 (registered means: FG-ARI
0.7709320986336607, ForegroundIoU 0.6149934302630647, MatchedObjectIoU
0.6305234820304301) with paired 10,000-resample bootstrap, seed 178. At both
endpoints, mean FG-ARI must be strictly above the SW0164 mean and the paired
95% confidence lower bound must be above zero; mean ForegroundIoU and
MatchedObjectIoU must each meet or exceed their SW0164 means. The mean-rho
mechanism is promoted over product-Q only if it also passes these same gates
against that matched control. Validation is fixed to registered IDs
1320..1639; reserve data is never opened. If both arms fail, close the line
without sweeps.

## Cache boundary and scope

The SW0161 cache stores float32 component spikes with shape `[4,1024,512]`;
these are already settled traces. Cache-boundary tests consume them as settled
inputs with `settle=0` and must not slice a second 512-step warmup. Live
rollouts use 1024 steps with `settle=512`. CPU-only preparation is limited to
the first eight ordered TRAIN rows and their full 1024-node modal labels.
Cached-trace gradients are not evidence of model credit. Ground-truth classes
and counts are used only in the TRAIN loss; inference does not consume GT
count or a GT-derived K. No validation/reserve rows, GPU rollout, training, or
performance claim is part of this initial implementation slice.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `1e24179e1e34aed5e2fbb3628941899c544cb1194f836c263881f8ac01c40653`

````markdown
# SW0178: temporal class competition

완료·독립 검증된 seed1 결과입니다. **QCE standalone FG-ARI 0.8180 / FIoU 0.4375 /
Object IoU 0.6864**. FG와 Object IoU는 기준 대비 개선됐지만 foreground false
positives가 늘었습니다. 원래 사전 gate는 실패로 종료하며 기존 unsupervised
세-seed 최고 모델은 유지합니다. 후속 SW0179에서 변경 없는 QCE 후보의
source seed0/2 재현성을 확인합니다.

전체 수치·비교·원인 진단·해석 범위는 [RESULT_KO.md](RESULT_KO.md), 사전 설계는
[DESIGN_KO.md](DESIGN_KO.md)에 있습니다. 이 실험은 TRAIN modal mask를 사용하는
supervised 추가 학습(512장×2)이며 native32 70,000장 신규 학습이나 gate 인과
기여 증명으로 해석하지 않습니다.

기록: `results_archive/cpu_v1`(실제 입력/모델/복원), `gpu_v1`(첫 두 실제 updates),
`train_v1`(최종64updates와 loss 검증), `eval_v1`(평가 준비/등록),
`pred_v1`(GT-free 예측 검사/audit 예약), `score_v1`(최종 점수/독립 검산/진단).
모든 학습·예측·점수·audit 작업이 완료됐으며 재시작하지 않았습니다.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `190f2a90b786e4db2741b3cb1b0424db40917a2a7bf5670cde6903e6c06a7d26`

````markdown
# SW0178 최종 결과

**QCE standalone에서 FG-ARI 개선을 확인했지만 원래 사전 판정은 실패했습니다.**
기존 unsupervised SW0165 세-seed 최고 모델은 유지하며, QCE는 후속 재현성
검증을 위한 supervised 후보로 보존합니다. seed1 한 개의 결과입니다.

## 실험과 평가 조건

동일한 fresh pretrained SW0097 SourceT + zero-context 모델에서 loss만 비교했습니다.
Signed mean-rho CE는 네 component signed correlation의 평균을, Product-Q CE는
기존 production positive-product Q를 이용해 물체 class 간 경쟁을 학습합니다.
정답 full modal mask(배경 포함)는 TRAIN loss에만 사용합니다. 새 mask decoder,
SW0176 gradient hook, SW0177 polarity head는 없습니다.

TRAIN512장 × 2 passes, B16/micro4, 64 updates, 1024 exposures입니다.
이는 이 설정으로 native32/70,000장을 새로 학습한 결과가 아닙니다.
추론은 실제 emitted S → native Q → SW0165 readout을 동일하게 사용합니다.
Conditional은 기존 SourceT foreground를, standalone은 자기 native QCC foreground를
사용합니다. 모든 예측을 고정한 뒤 동일한 fixed320 IDs1320..1639에서 평가했습니다.
Reserve는 읽지 않았습니다. 평균·per-image 지표·seed178 bootstrap·판정·foreground
규칙·hash를 별도 CPU 검증기로 재계산해 확인했습니다.

## 결과

| 모델 / readout | FG-ARI ↑ | Foreground IoU ↑ | Object IoU ↑ |
|---|---:|---:|---:|
| 기준 SW0164 seed1 | 0.7709 | 0.6150 | 0.6305 |
| Signed mean-rho CE · conditional | 0.7658 | 0.6150 | 0.6340 |
| Signed mean-rho CE · standalone | 0.7818 | 0.6097 | 0.6534 |
| Product-Q CE · conditional | 0.7774 | 0.6150 | 0.6403 |
| Product-Q CE · standalone | 0.8180 | 0.4375 | 0.6864 |

QCE standalone의 기준 대비 FG-ARI 차이는 **+0.0471**이며,
paired 95% CI는 **[+0.0348, +0.0597]**입니다.
Object IoU도 개선됐습니다. 하지만 Foreground IoU가 낮아졌고, conditional
FG 개선의 CI도 0을 포함합니다. 따라서 원래의 두 endpoint FG CI 양수·두
IoU 유지 조건은 어느 arm도 통과하지 못했습니다. 이 사전 판정을 변경하지
않고 SW0178 recipe는 종료합니다. Mean-rho가 QCE보다 좋다는 가설도 이번
조건에서 지지되지 않았습니다. 이는 clamp의 일반적인 미분 특성 전체에
대한 반증은 아닙니다.

## Foreground 손실 진단

기존 frozen predictions와 fixed320 GT만으로 재계산했습니다. 새 forward,
optimizer update, threshold/min8/BG 정책 변경은 없습니다.

| 이진 foreground 진단 | 기준 SW0164 | QCE standalone |
|---|---:|---:|
| 이미지별 precision 평균 | 0.6886 | 0.4575 |
| 이미지별 recall 평균 | 0.8696 | 0.9192 |
| 배경을 물체로 포함한 FP patch 합 | 46060 | 88461 |
| 물체를 놓친 FN patch 합 | 8996 | 5752 |
| 예측 foreground 비율 | 0.3315 | 0.4708 |

GT foreground 비율은 0.2184입니다. 물체를
놓치는 양은 줄었지만 배경을 물체로 포함하는 양이 늘었습니다. 이진 support는
native QCC와 최종 standalone mask에서 동일하므로 SW0165 merge가 foreground
membership를 바꾼 결과는 아닙니다. 저장된 최종 support만으로 largest-background
component와 discarded small groups의 개별 원인을 나눠 단정할 수는 없습니다.
Object 개수 classifier의 안정성이나 각 gate의 인과 기여를 입증한 결과도 아닙니다.

## 후속 결정

사용자의 FG-ARI 우선 목표에 유용한 후보를 잃지 않도록 별도 SW0179에서
QCE standalone의 **변경 없는 seed0/2 재현성**을 먼저 확인합니다. 이는 seed1
결과를 보고 선택한 posthoc supervised lineage이며, SW0178의 사전 성공이나
unsupervised 성과로 표시하지 않습니다. seed1 checkpoint와 점수는 그대로
재사용합니다. 새 loss/온도/LR/passes/BG 정책을 섞지 않습니다.

새 실용 후보의 세-seed standalone 평균 FG가 SW0165와 matched Slot을
넘고, 두 IoU 평균이 matched Slot을 엄격히 넘는지 확인합니다. Shared-image
bootstrap에서는 세 seed의 이미지별 평균 320개를 사용하며 960개를 독립
표본으로 취급하지 않습니다. 단일 seed를 Slot 세-seed 평균과 비교해 승리로
주장하지 않습니다. FG .87, native70k scaling, gating 기여 증명은 남아 있습니다.

## 근거

- `results_archive/score_v1/evaluation.json`: 고정 점수와 per-image 원문.
- `verification.json`: 별도 재계산과 판정 검증.
- `foreground.json`: 이진 support·FP/FN·group 수와 크기 진단.
- `rho_protocol.json`, `qce_protocol.json`, `*_queue.json`, `*_model.json`: 고정 예측과 모델 불변성.
- `verify_score.py`, `diagnose_foreground.py`: 검증·분석 코드.

점수 SHA256: `f7c404615edef11c57ac6e1d71dd450ededab7f2e9307b54996d6365a51106df`.

````

</details>
