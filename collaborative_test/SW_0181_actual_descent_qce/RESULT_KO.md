# SW0181 결과 — 실제 loss 감소를 확인한 업데이트만 반영

고정 native32(32×32 patch), 공통 평가 320장, seed 0/1/2 평균입니다.
실제 spike → native Q → production QCC → SW0165 standalone mask를 평가했습니다.
세 seed의 예측과 protocol을 모두 확정한 뒤 평가 정답을 읽었습니다.

| 설정 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| SW0181, mask-supervised | **0.814836** | **0.589302** | **0.671681** |
| SW0178/179, 동일 seed unconditional QCE 대조군 | 0.761620 | 0.469308 | 0.584656 |
| SW0165, 기존 unsupervised 기준점 | 0.804299 | 0.609975 | 0.651149 |
| Slot Attention, 우리 70k 학습 epoch10 기준 | 0.781652 | 0.204360 | 0.203165 |

SW0181은 Slot Attention을 세 지표 모두 초과했고 등록한 primary gate를
통과했습니다. SW0165보다 FG-ARI와 Object IoU는 높지만 Foreground IoU는 낮습니다.
정답 mask를 사용한 추가 학습 결과이므로 unsupervised 모델의 승리로 합치지 않습니다.

## Seed별 결과와 신뢰구간

| seed | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 0 | 0.848596 | 0.724047 | 0.699976 |
| 1 | 0.796650 | 0.592035 | 0.678430 |
| 2 | 0.799263 | 0.451825 | 0.636637 |

세 seed의 같은 이미지별 점수를 먼저 평균하고, 공통 320장을 10,000회
paired bootstrap했습니다(RandomState seed 178).

- FG-ARI 차이, SW0181 − SW0165: **+0.010537**, 95% CI **[+0.005757, +0.015223]**.
- FG-ARI 차이, SW0181 − Slot: **+0.033184**, 95% CI **[+0.014304, +0.051693]**.
- FG-ARI 차이, SW0181 − unconditional QCE: **+0.053216**, 95% CI **[+0.044362, +0.062337]**.

Unconditional QCE 대비 seed 0/1은 각각 0.849988→0.848596,
0.818029→0.796650으로 낮아졌고, seed 2는 0.616842→0.799263으로 회복했습니다.
따라서 모든 seed가 좋아졌다는 결과는 아닙니다. 다만 동일 정책으로 세 seed를
실행했을 때 평균과 하위 seed의 붕괴 문제가 개선됐습니다. 최종 FG 분할은
minibatch QCE 감소만으로 보장되지 않습니다.

SourceT foreground를 고정한 conditional 진단 평균은
**0.802259 / 0.609975 / 0.655217**입니다. 이 endpoint는 foreground membership을
고정하므로 standalone과 구분합니다.

## 변경 내용과 비용

기존 SW0179와 같은 fresh SourceT+zero-context, 공통 TRAIN512, 두 permutation,
B16/micro4, T1024/settle512/live-tail64, QCE temp0.1, 네 optimizer family와 LR를
사용했습니다. 변경은 같은 batch·RNG에서 실제 post loss가 사전 guard보다 더
감소한 Adam proposal만 반영한 것입니다. 거절 시 가중치·Adam·RNG를 복원했습니다.

64 proposals 중 반영 수는 seed별 **32 / 38 / 34**, 거절 수는 **32 / 26 / 30**입니다.
거절된 batch를 대체하거나 학습 예산을 늘리지 않았습니다. 각 seed는 gradient
forward 64회 외에 post 검사 64회와 pre 검사 2회(B16)를 추가했습니다.
따라서 총 130 logical full forward, 2,080 image forwards가 필요했습니다.
기존 대조군과 committed Adam step 수 및 연산 비용이 같다는 주장은 하지 않습니다.

## 검증과 범위

서버의 실제 모델·Adam 산출물 검증, 첫 두 proposals의 실제 spike와 네 family의
gradient, no-grad loss parity, model/optimizer rollback 및 checkpoint cursor를
확인했습니다. 최종 독립 검산은 세 seed·두 endpoint의 실제 예측에서 모든
per-image metric, seed 평균, paired CI, 판정을 다시 계산했습니다. 평가 전후
예측·protocol·모델·optimizer·score hash도 동일했습니다.

이 결과는 **TRAIN512 mask-supervised 추가 학습**입니다. Native32에서 처음부터
70,000장 전체를 학습한 결과가 아니며, FG-ARI 0.87, 데이터 증가에 따른 향상,
개별 gate의 필요성·학습 기여는 아직 증명하지 않았습니다. Unsupervised 기준점은
SW0165를 유지합니다.

원본 점수 SHA256:
`3f69a74a5729341b5fd11f2bd74f2d51ff245c0839f76053b8733af9523ddd54`.

- [고정 평가 원본](results_archive/score_v1/evaluation.json)
- [독립 검산 결과](results_archive/score_v1/independent_audit.json)
- [독립 검산 코드](results_archive/score_v1/independent_verifier.py)
- [동일 seed QCE 대조군 비교](results_archive/score_v1/unconditional_comparison.json)
- [사전 등록 설계](DESIGN_KO.md)
