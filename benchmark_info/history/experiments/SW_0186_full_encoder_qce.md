# SW_0186_full_encoder_qce

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `dca432c264afbf79dbdd1f94a01cacbc55ccb799b26f9615bc3a5dbcc5bdfe03`

````markdown
# SW0186: 원본 RGB encoder 공동 학습 + 기존 QCE

185번 공간 보조 loss는 기존181 대비 FG 개선 gate를 통과하지 못했으므로
종료했다. 이번에는181의 QCE-only/readout와 초기 tensor를 유지하면서,
현재 source CNN을 명시적으로 frozen으로 설정한 제약 하나를 바꾼다.

Source097→native32→SourceT + 동일 zero-final RF31 context 초기화에서
원본8-channel RGB CNN의 requires_grad를 켜고 독립 source_base Adam group을
추가한다. 기존 네 group/rate는 유지하며 source_base LR은 이전160 설계의
3e−5로 고정한다. QCE 온도.1, TRAIN512×2, B16/micro4,64 proposals,
T1024/settle512/live-tail64, clip1, strict actual full-forward descent/rollback,
세 source seed0/1/2와 입력 순서도181과 동일하다. 185 보조 loss는 사용하지
않는다. CNN의 초기 tensor/forward는 원래 모델과 정확히 같아야 한다.

Source160의 다른 loss/readout 조건에서 joint encoder가 .7590이었던 결과는
이번 QCE 조건의 성공 근거가 아니다. 이번 결과로 discrete gamma ordering
최적성/순서 불변성이나 많은 데이터의 향상 효과가 입증됐다고 하지 않는다.
TRAIN mask를 사용하는 supervised continuation이며 fresh70k/unsupervised가
아니다. 원래 .87/70k scaling 및 각 gating의 유용성 목표는 계속 남는다.

실행 전 실제 세 seed CPU factory와 기존181 초기 model SHA 일치, empty Adam,
CNN forward parity와 gradient→source CNN→실제 Adam 변경, five-family 중복
없음, checkpoint restore 및 composite가 아닌 QCE full-forward parity를
검사한다. 첫 두 실제 GPU proposals에서는 loss→S, 다섯 family의 finite
positive gradient/proposed delta와 full-forward parity를 검사한다. 최종
모델은 원본 CNN의 초기/최종 SHA와 실제 Adam steps도 기록·검증한다.
Frozen 비교에서는 의도적으로 trainable인 base_encoder만 제외하며,
feature mean/std/clip, patcher, geodesic_contrast는 계속 고정한다.

평가는 기존 actual S→native Q→QCC→SW165 readout, fixed320 native32 modal
계약이다. 세 seed와 conditional/standalone 예측을 모두 확정한 뒤만 GT를
채점한다. Primary gate는 standalone FG mean>SW181 mean 및 paired-image
95%CI 하한>0, Slot 대비FG/FIoU/ObjectIoU 세 mean 엄격 초과다. 원본 CNN의
실제 변경/Adam credit도 확인해야 한다. Conditional과 모든 seed를 공개한다.
독립 검산은185 v2의 명시적 Slot archive loader를 사용한다.

실패하면 이 조건의 작은 LR/epoch sweep으로 연장하지 않고 종료한다.
GPU는 exclusive idle 장치만 사용하고 타 사용자가 오면 해당 장치를 양보한다.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `240633f53bd1d74f9ff88c7e7ec141ee671537cd42d2742e66aed489f91f14c4`

````markdown
# SW0186 ? full source RGB encoder plus unchanged QCE

Completed and independently audited. All three training runs and fixed320
native32 conditional/standalone predictions are complete. The original CNN
actually learned in every seed. Standalone three-seed means are
**0.8346805 FG-ARI / 0.5690396 foreground IoU / 0.6898022 object IoU**.

The preregistered FG gate versus SW0181 passed, including a positive paired-image
95% CI. All three metrics exceed comparable Slot70k. Foreground IoU decreased
versus SW0181, so this is the supervised FG incumbent rather than an improvement
in every metric. The unsupervised incumbent remains SW0165.

This is supervised continuation from Source097 with TRAIN512x2/64 proposals,
not fresh70k training. FG0.87, data scaling and gate causality remain unproven.
See [the full result](RESULT_KO.md) and [registered design](DESIGN_KO.md).

CPU admissions, actual GPU credit, complete training manifests/histories, frozen
prediction receipts, the raw score and independent audit are under
`results_archive`. No final metric was used to select prediction thresholds.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `74e63d667187b1e4d879fb1235d6b139a1dea52bcf84709ecb11ff56a798f06f`

````markdown
# SW0186 ?? ??: ?? RGB encoder ?? ??

??320 / native32 / seed0?1?2 ???? FG-ARI ?? supervised ??? ????. ?? QCE, ?? tensor, RF31 context, spike?Q?QCC readout? TRAIN512?2/64 proposal ??? ???? ?? RGB CNN? LR3e-5? ?? ????.

| ?? | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| SW0186 ?? encoder | 0.834681 | 0.569040 | 0.689802 |
| SW0181 ?? ?? FG | 0.814836 | 0.589302 | 0.671681 |
| Slot Attention, ??70k ?? epoch10 | 0.781652 | 0.204360 | 0.203165 |

| seed | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 0 | 0.856752 | 0.788567 | 0.706667 |
| 1 | 0.815539 | 0.460093 | 0.685784 |
| 2 | 0.831751 | 0.458459 | 0.676955 |

??181 ?? FG ??? +0.019844, paired-image 95% CI? [0.013290, 0.026281]?. ? seed ?? FG? ????. Object IoU? ????? Foreground IoU? -0.020263? ????. ? ?? ?? ?? ???? ????? ??? ?? ???.

Conditional ??? 0.808423 / 0.609975 / 0.662083??. Primary endpoint? ??? ?? standalone??, ?? ? readout?threshold??? ??? ??? ???.

?? ??? ?? candidate 1920 endpoint-image ?? Slot 960 image ?, modal GT, ? seed ??, paired CI? ??? ?????. ?? ??? 3.3306690738754696e-16??. ??? accepted update? seed0/1/2?? 34/36/43???, ? seed ?? ?? CNN ??? ??? Adam ??? ????.

??? 4096??? ??? Source097 ??? supervised TRAIN512?2 ?? ????. 70,000? fresh ??, ??? ??? ?? ??, unsupervised ?? ??, discrete gamma ordering? ???, ? gating? ??? ???, FG-ARI 0.87 ??? ???? ???. ?? unsupervised ??165? ??? ????.

?? ?? ??? ?? encoder ???? unique1024? 512 ??? ?? ???proposal ???? ???? ???. ?? ??? ?? prefix?? ??? ??? ?? ??? ? ?? ???? ??? ????. ? ??? ???? ? ? ?? ???? ????.

?? ?? SHA256: `bf4e0964d4683687f4e15302eb0ef1e9283ca380aa38b1d2f4061ea1f03daded`

[?? ??](results_archive/score_v1/evaluation.json), [?? ??](results_archive/score_v1/completion_audit.json), [archive receipt](results_archive/score_v1/archive_receipt.json).

Training-budget clarification: Source097 already inherits full70k RGB training; 4,096 is its final additional stage. The 512/1024 counts describe additional supervised adaptation only. See [verified foundation provenance](../SW_0187_joint_encoder_unique_scaling/FOUNDATION_PROVENANCE.md).

````

</details>
