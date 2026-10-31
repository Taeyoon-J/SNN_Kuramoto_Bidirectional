# SW_0185_qce_spatial_connectivity

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `ab741f29a94f6995064019384df5575d5ba5dd3d5427228add0020e1f2e46c3d`

````markdown
# SW0185: QCE + local spike-connectivity margins

184번 TRAIN16은 단일 원인을 입증하지 못했다. 다만 같은 물체/배경의 Q
연결이 실제 QCC .50 기준을 넘도록 하는 학습 신호가 QCE에 명시되어 있지
않다. 상대 class competition은 유지하면서 실제 공간 graph의 연결/경계에
credit을 주는 보조 목적을 별도 테스트한다. 175번 forest loss 단독 정책이나
종료한182/183의 rescue sweep으로 취급하지 않는다.

목적: 기존 class-balanced leave-one-out production-Q CE + 공간 보조 loss.
4-neighbor native32 grid에서 TRAIN modal labels가 같은 edge는 네 signed
component rho 각각을 최소 .55**(1/4)로 유도한다. 그러면 그 edge의 positive
product Q는 .55 이상이 된다. 다른 class의 edge는 production Q를 .45
이하로 유도한다. 각각 squared hinge, positive class별/negative unordered
class-pair별 평균, 두 항 평균, 보조 weight1로 고정한다. BG class0 포함.
QCE 온도.1, 기존 모델/optimizer/batch/clip/readout는 유지한다.

모델은 기존 Source097 초기화+zero context에서 seed0/1/2별 fresh 시작한다.
TRAIN512×2/B16/64 proposals, strict actual full-forward loss descent와
rollback을 사용한다. 첫2 proposals에서 실제 S→loss/four-family credit와
full-forward parity, 최종 실제 Adam 및 모델 tensor 변경을 검사한다.
기존181 QCE-only는 고정 비교 대상으로 보존한다. accepted update 수가 다를
수 있음을 명시하고, 70k 신규 학습으로 부르지 않는다.

평가는 기존 native32 fixed320 actual S→Q→QCC→SW165 readout를 그대로
사용한다. 세 seed·두 endpoint 예측이 확정되기 전 GT scoring 금지.
primary standalone 평균FG가181보다 높고 paired-image95%CI 하한>0,
Slot 대비FG/FIoU/ObjectIoU 평균 모두 엄격히 초과하면 확대한다. Conditional
FG와 세seed각각, foreground관련오류도 공개한다. 원래 .87/70k/gating 인과
기여 목표는 별도이며 이 pilot의 성공으로 대체하지 않는다.

실패하면 margin/weight/threshold 작은 sweep으로 이 recipe를 연장하지 않는다.
현재 이 문서는 preregistration이며 GPU 학습 시작 증거는 별도 기록한다.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `ea123eb12781c01668afcd73ba22a0bcc0d736435ee0caf6c51a5aad5bb6a6bf`

````markdown
# SW0185 — completed, no FG improvement over SW181

Three-seed fixed native32 scores: **.813681 / .591928 / .676498**.
Slot is exceeded in all three metrics, but the preregistered FG improvement
gate against SW181 fails. All training/predictions/scoring are complete.
The failed audit v1 is preserved; separate v2 independently verified actual
candidate and Slot masks, means, confidence intervals and gates.

See [full results, limits and next direction](RESULT_KO.md).
No incumbent change or .87/fresh70k/gate-causality claim is made.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `6f1692178c8541351db135b0c4720bed33383120a396739019c403d1ec9d73f5`

````markdown
# SW0185 완료: 공간 연결 보조 loss의 FG 개선 미확인

Native32×32, 고정320장, seed0/1/2 평균. Source097에서 시작한 supervised
TRAIN512×2/64-proposal continuation이다. 70k 신규 학습/unsupervised 결과가 아니다.

| 모델 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| SW185 QCE + spatial margins | .813681 | .591928 | .676498 |
| 기존 최고 supervised SW181 | .814836 | .589302 | .671681 |
| 기존 최고 unsupervised SW165 | .804299 | .609975 | .651149 |
| Slot Attention, 우리70k 학습 | .781652 | .204360 | .203165 |

| SW185 seed | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 0 | .842880 | .708906 | .691570 |
| 1 | .786678 | .605848 | .664304 |
| 2 | .811483 | .461031 | .673621 |

SW181 대비 FG 차이는 −.001156, paired-image95%CI [−.006110,+.003752].
Foreground IoU 차이 +.002626의 CI는 [−.003957,+.009528], Object IoU 차이
+.004817의 CI는 [−.000886,+.010453]이다. IoU의 수치는 높지만 세 지표 모두
이 비교 CI가0을 포함한다. 따라서 통계적으로 확인된 개선이라고 주장하지
않고, 사전 등록한 primary gate 실패로 종료한다. Slot은 세 지표에서 넘지만
기존 최고FG 교체 및 .87/70k 확대 근거가 되지는 않는다.

실제 accepted updates는33/40/43이다. 추가 공간 loss의 proposal 평균은
각각 .05684/.07323/.05630으로 실행됐으며 첫 두 proposal의 emitted-S 및
네 family gradient/변화, full-forward parity와 최종 Adam 상태를 확인했다.
이 credit 증거는 각 gating의 정확도 기여나 인과 증명이 아니다.

원 채점은 모든3seed 예측 확정 후 GT를 읽어 정상 완료됐다. 별도 audit v1은
native179의 references에 Slot이 있다고 잘못 가정하여 KeyError('slot')로
실패했다. 학습/예측/점수와 v1 코드·실패 기록을 보존했고 GPU를 재실행하지
않았다. 별도v2는 세 Slot archive의 checkpoint/protocol/ID/SHA와 native128
labels를 직접 검증한 뒤 independent modal pooling 및 NumPy/SciPy 지표로
candidate1920 endpoint/image rows와Slot960 rows를 재계산했다. 최대 오차는
3.34e−16 미만이고 모든 seed평균/pairedCI/primary gate가 일치했다. v2의
실제 세 Slot archive 사전 검사는 GT를 읽기 전에 완료됐다.

작은 margin/weight/threshold sweep으로 이 recipe를 연장하지 않는다. 다음
방향은 기준 QCE와 readout를 유지하면서, 현재 명시적으로 frozen인 원본 RGB
encoder도 공동 학습하는 비교다. 이전160번의 다른 loss/readout 결과를 이번
QCE 조건의 성공 증거로 사용하지 않는다. 70k scaling과 gating utility는 계속
별도 검증해야 한다. SW181/SW165 incumbent는 유지한다.

증거: [최종 점수](results_archive/score_v1/evaluation.json),
[독립 검산v2](results_archive/score_v1/completion_audit_v2.json),
[보존한 audit v1 실패](results_archive/score_v1/failed_audit_v1_terminal.json).

````

</details>
