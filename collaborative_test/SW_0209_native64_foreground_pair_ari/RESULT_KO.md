# SW0209 완료: pair-ARI surrogate는 실제 hard-mask FG 개선을 입증하지 못함

고정16장·seed0·native64, 동일한 warm192 실제 model/Adam/CPU-CUDA RNG에서
같은 TRAIN8 추가·B4 두 proposal을 수행했다. 비교 control202도 같은 추가
학습량이다. 모델/gating/encoder/graph/readout 구조와 평가 데이터는 그대로다.
Supervised pilot이며 새로운70k 학습이나 전체320 평가가 아니다.

| 모델 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| QCE +foreground pair-ARI surrogate | 0.901220 | 0.862279 | 0.736022 |
| 동일 학습량 원본 QCE202 | 0.900148 | 0.859586 | 0.728138 |
| 추가 학습 전 warm192 | 0.924401 | 0.859896 | 0.727638 |

Candidate−control FG 차이 +0.001072, paired95% CI[−0.000057,+0.003147].
Object IoU 차이 +0.007884, CI[−0.000653,+0.024115]. 세 지표 평균은 control보다
조금 높지만 모든 CI가0을 포함한다. Warm 대비 FG는−0.023182,
CI[−0.069988,+0.000940]다. 등록된 확대 gate는 실패했으며 이 계수의 loss
recipe를 닫는다. 작은 LR/계수 sweep이나 이 모델의70k 확대를 진행하지 않는다.
향후 다른 접근에서 재검토할 수 있도록 checkpoint는 보존한다.

## 실제 loss 전달은 확인됨

두 새 proposal이 모두 적용되고 실제 TRAIN objective가 감소했다.
Adam 누적 step은4(상속2+새2), 실제 CNN도 바뀌었다. 새 pair-ARI 항만의
emitted-spike gradient norm은0.005706/0.006321이며 전체 loss의 gradient를
새 항의 기여로 잘못 해석한 것이 아니다. Pair-ARI TRAIN loss는 각각
.249533→.237221, .207382→.197430으로 감소했다.

따라서 이번 결과를 gradient 단절이나 encoder 고정 문제라고 부를 수 없다.
그러나 gradient가 있다는 사실은 gating의 인과적 유용성이나 hard readout
성능 향상을 증명하지 않는다. Soft Q pair agreement와 최종 discrete
partition이 다르다는 한계는 그대로다. 다음 후보는 계수 조정보다 물체별
assignment 학습을 다루되 기존 foreground를 함부로 제거하지 않는 방식이다.

독립 검산은 실제 완료 checkpoint/model/Adam/history, 초기 tensor transaction,
상속 CPU-CUDA RNG,48개 실제 masks, 세 metric/CI/count 및 새 항의 spike
gradient 기록을 확인했다. 최대 metric 오차2.22e-16.
Model SHA `9b497900ca05e7a1bcab6e26f97eb527c936e6df44b99ccdcb487a048694075d`.
Boundary SHA `9fb38d06400acbce4210183d373ebbee21afac1cfbed0fa693acfe749d3b031f`.
첫 서버 검산-test 의존성 누락은 학습 시작 전에 수정됐고 원본 실패 로그를
보존했다. 이 결과를 얻기 위한 GPU 학습 재시도는 없었다.

현재 full320 기준은208 warm192의 .877709/.791224/.696284다. 위16장
.901220을 새 full320 .90 달성이나 새 최고 모델로 표기하지 않는다.
