# SW0218 완료: energy-ranking loss는 FG 개선 없음

동일 native64/fixed16/seed0, 실제 warm192 model/Adam/RNG에서 시작해204와 같은
새 TRAIN8/B4 두 proposal을 수행했다. 모델과 CNN이 바뀌고 실제 Adam step4가
저장됐다. 원래1024-step spike readout과 encoder/core/graph/gating 구조는 유지했다.

| 설정 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 새 energy ranking | .898687 | .858750 | .729999 |
| matched204 component QCE | .900539 | .860577 | .727622 |
| 기존 warm192 | .924401 | .859896 | .727638 |

204 대비 FG 차이-.001852,95% paired CI[-.005725,+.001296]이다.
Warm 대비-.025715, CI[-.073678,-.000170]로 개선되지 않았다. ObjectIoU의
작은 양의 차이도 두 CI 모두0을 포함한다. 확대 gate는 false다. 이 loss의
작은 weight/margin/LR 탐색은 진행하지 않는다. Full320 최고 평균
.848615/.592533/.672047과 .90 목표 미달 상태는 그대로다.

## 실제 학습과 진단

Auxiliary 자체의 emitted-spike gradient가8개 새 TRAIN 이미지에서 양수였다.
각 proposal 첫 active 이미지에서 context encoder/graph/native core/integration/
원래 CNN의 실제 auxiliary parameter gradient도 모두 양수였다. Gradient가
존재하고 파라미터가 바뀌었다는 사실이 분할 성능 개선을 보장하지 않았다.
같은 RNG의 no-grad/gradient loss 차이는2.61e-8 및1.49e-8이었다.

첫 proposal의 실제 combined TRAIN loss는2.819465->2.947188로 상승했고,
두 번째는2.554252->2.478408로 감소했다. 두 proposal은 다른 TRAIN 이미지이므로
이 수치를 서로 이어진 loss curve로 해석하지 않는다. 유한 uphill step도
유지하는 사전 고정 정책을 따른 것이다. 새 목적함수에서도 이전 목적함수의
Adam 상태를 이어 쓰는 영향이 있는지는 아직 미검증이다. 다음에는 동일한
gradient/모델/이미지에서 Adam 상태 유지와 초기화를 비교하는 한 번의 제한된
진단을 검토한다. 현재 결과만으로 Adam 상태가 원인이라고 단정하지 않는다.

## 집계 오류와 복구 범위

학습과16장 candidate 예측은 정상 완료했다. 이후 evaluator의 candidate arm을
문자열 치환하면서 control package까지 SW_0204_native64_energy_ranking으로
잘못 바꾼 탓에 CPU 집계 import가 실패했다. Local/server22 tests는 이 전체
집계 분기의 실제 import를 실행하지 못했다. 서버 차단이나 모델 학습 실패가 아니다.

원래218 학습/추론 소스와 fingerprint, model/Adam/checkpoint/예측을 유지했다.
별도 recover_score.py가 실제204 control을 명시적으로 import하고48개 고정
mask를 검증한 뒤 GT를 읽어 집계만 복구했다. 전체 집계 분기를 실행하는
새4개 local/server tests와 actual frozen-mask CPU admission을 통과했다.
재학습/추가 native forward/update는0이며 원래 queue의 failed 기록도 보존했다.

별도 독립 감사에서 실제 model/CNN/Adam/RNG/학습 이력/auxiliary credit과48개
mask endpoint의 세 지표/paired CI를 재검산했다. 최대 metric 차이2.22e-16,
새 source model SHA는6735c80ceb83404c8a4748421be330a076603dd20d3377d403f9003e40bab2da다.
결과는 supervised mini이며 full320/세 seed .90/신규 native64 TRAIN70k나
각 physical gate의 인과 기여 증명으로 표기하지 않는다.

근거: results_archive/completed_v1/score_v1/summary.json, completion_audit.json,
training/report.json 및 score_recovery_v1/receipt.json.
