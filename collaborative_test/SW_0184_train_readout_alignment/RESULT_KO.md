# SW0184 완료: TRAIN16 손실/분할 진단

고정 TRAIN16(old8+new8), frozen183 두 모델을 비교했다. 추가 학습 없이
실제 S→native Q→QCC→SW165 분할을 세 seed에서 실행했다.

| Seed | QCE 512 → 1024 | Standalone FG-ARI 512 → 1024 | Foreground IoU 512 → 1024 |
|---|---:|---:|---:|
| 0 | .244963 → .294692 | .913664 → .895067 | .684570 → .819372 |
| 1 | .408745 → .361419 | .835867 → .837792 | .611928 → .598299 |
| 2 | .365583 → .369496 | .864376 → .873369 | .526855 → .488036 |

이 숫자는 validation 성능이 아니며 .87 목표 달성으로 세지 않는다.
seed0는 QCE/FG가 함께 악화되고 seed1은 함께 개선됐다. seed2는 QCE가
조금 높아졌지만 FG는 개선됐다. 따라서 모든 seed에 공통인 단일 원인을
확정하지 않는다. 기존183의 full320 결과 및 최고 모델도 바꾸지 않는다.

같은 BG patch 쌍의 Q>=.50 비율은 candidate seed0/1/2에서 각각
.8255/.6268/.3498이다. FG same-object 쌍은 .8536/.8081/.8442이다.
이는 특히 seed2의 배경 연결이 약한 현상과 일치하지만, 모든 쌍의 평균을
공간 graph의 connectivity나 오류의 인과 증명으로 취급하지 않는다.
BG 비율은 기록된 all-class/FG-class balanced 평균과 class 수로 역산했고
모든16행에서 정확히 BG 한 class 차이가 있음을 검사했다.

다음 모델 변경은 QCE를 유지하면서 TRAIN mask의 공간 이웃 same-class
연결과 cross-class 경계가 실제 QCC .50 기준 양쪽에 놓이도록 보조 loss를
주는 것이다. 양의 연결은 per-component signed rho에 직접 credit을 주고,
경계 억제는 production Q를 사용한다. 이전 forest-loss 단독 실패와 구분해
새 실험으로 등록하고, 실제 분할이 좋아질 때만 확대한다.

원본 보고서의 no-update model tensor hash, cohort/producer SHA와 산술은
검증했다. 이 사후 검사는 두 번째 mask 추론이나 독립 metric 재계산은
아니다. scope는16장에 한정한다.

원본과 집계: [result_v1](results_archive/result_v1/).
