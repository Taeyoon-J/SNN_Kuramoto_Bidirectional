# SW0193 완료: seed0 native64 미니 평가

고정 development32장(IDs1320..1351), native64 직접 spike-derived mask 기준이다.
이는 전체320장·세 seed 평균 결과가 아니며, 최종 FG-ARI .90 목표 달성이 아니다.

| 모델/같은32장 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| SW0186 learned model의 native64 전환, 추가 학습 전 | .909537 | .827147 | .734442 |
| SW0192 native64 TRAIN8/2-proposal 추가 학습 후 | .918586 | .832867 | .733686 |
| Slot seed0 | .689062 | .180979 | .209370 |
| Slot 세 seed 평균 (참고) | .821571 | .173315 | .203988 |

우리 모델은 seed0 하나다. Slot 세 seed 평균과의 비교는 참고이며, matched
three-seed superiority claim으로 사용하지 않는다. 미니 subset의 Slot 평균도
기존 전체320장 Slot64 평균 .786902보다 높으므로, 이 subset 점수를 전체 평가
점수로 외삽하지 않는다.

추가 학습 후-minus-before paired-image 차이와95%CI:

- FG-ARI +.009049, CI[+.001647,+.019472]
- Foreground IoU +.005721, CI[-.004043,+.018492]
- Object IoU -.000756, CI[-.018451,+.013969]

FG 변화의 paired-image CI는 양수지만, 등록한 minimum gain .01 기준에 미달해
screen_gate_passed=false다. 기준을 사후 완화하거나 이 결과를 새 incumbent로
승격하지 않는다. 데이터양, 업데이트 수, gating 인과성, seed 일반화는 입증하지
않는다. 양의 미니 신호를 기록하되, 더 큰 TRAIN64 미니 continuation은 원래
SW0192에 등록한 별도 학습 예산 검증으로 다뤄야 한다. 70k 학습은 예약하지 않았다.

실제 원본 CNN/graph/core/integration이 학습된 것은 SW0192 checkpoint와 Adam
step2/gradient 증빙으로 확인했다. 본 실험의 optimizer update는0이다. 두 모델
모두 같은 per-image inference RNG, T1024/settle512, SourceT graph, production QCC
및 foreground-induced merge를 사용했다. 두 모델의32개 mask를 먼저 동결한 후
GT를 읽었고, Slot native128 및 GT는 직접2x2 modal로 변환했다.

분리된 read-only audit에서 실제 후보64 image rows와 Slot96 rows, 독립 modal,
ARI/IoU/assignment/mean/CI/gate를 다시 계산했다. 최대 metric 오차는
2.220446049250313e-16이다. 원본 이미지별 점수는 비공개로 보관한다.

유효한 추론 시간은 baseline32장1138.79초, candidate32장1152.98초로, 각각
이미지당35.59초/36.03초였다. 초기 import 오류의 첫 이미지 forward는 이 합계에
포함되지 않는다. 오류·기존 source·admission·로그를 보존하고 실제64 mask 변환
회귀검사를 추가해 일곱 server tests를 통과한 후 재개했다. 학습은 반복하지 않았다.

공개 집계 summary SHA256:
`6e8e036718c520eec04226fc75e6356644de73bcea74e338bb7c174be9cbd4eb`

증빙: results_archive/score_v1/summary.json, completion_audit.json,
frozen_v1 및 completed_v1.
