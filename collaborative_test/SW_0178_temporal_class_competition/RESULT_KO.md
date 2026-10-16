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
