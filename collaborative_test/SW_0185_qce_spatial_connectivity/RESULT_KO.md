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
