# SW0168 본 학습과 자동 평가 큐 실행

2026-10-15. GPU 선행 검증 통과 후 실제512장 TRAIN RGB/HDF mask/source-cache/target 입력 검증을 완료했다. 서버 본 학습 CPU14개 검사와 실제512 입력 검증, 실제 큐 admission 함수의3-tuple handoff가 모두 통과했다. 본 학습 소스는 독립 검토를 거쳤으며 배포 후 변경하지 않았다.

| Arm | Supervisor | Worker | 실제 GPU |
|---|---:|---:|---:|
| Constrained maximin | 3671729 | 3671833 | 2 |
| Matched local edge | 3671730 | 3671820 | 1 |

두 worker의 `/proc` 명령줄과 NVIDIA GPU 소유 PID를 확인했다. 각 worker가 실제 optimizer update를 수행한 로그도 확인했다. 이는 큐 state만으로 학습을 추정한 기록이 아니다. GPU0의 외부 PID2653200은 사용하거나 중지하지 않았다.

두 arm은 source seed1, 동일512장, 동일 초기값·shuffle168001·LR·Adam·64 logical updates를 사용한다. Logical B16은 microbatch4 네 번을 imagewise 평균한 gradient로 한 번 update한다.512장 두 pass/1024회 이미지 노출이며, fresh70k 학습이나 seed0/2 재현은 아니다. Encoder·native graph·core·기존12 integration scalar를 학습하고 새로운 MLP/residual encoder는 추가하지 않는다.

각 update의 durable checkpoint에 model/optimizer/RNG/order/history를 저장한다. 다른 GPU 사용자가 나타나면 자기 child만 중지하고 lease를 반환한 뒤 같은 durable boundary에서 재개한다. 사용하지 않은 parameter는 optimizer receipt에 명시하며 가짜 update를 만들지 않는다. Initial/final 고정 TRAIN8의 실제 spike·gate·margin·clamp·Q·conditional/standalone partition 진단도 별도로 저장한다.

학습 완료 후 두 co-primary mask를 자동 생성하도록 prediction 두 개와 CPU GT-last score 큐를 미리 등록했다.

| 큐 | 실제 supervisor PID | 등록 시 상태 |
|---|---:|---|
| Maximin fixed320 prediction | 3674744 | waiting_for_training |
| Local-control fixed320 prediction | 3674760 | waiting_for_training |
| 두 prediction 이후 GT-last score | 3674763 | waiting_for_both_predictions |

평가 코드의 서버9개 CPU 검사, 실제 SourceT/SW164 고정320 reference·RGB cache·baseline score SHA 검사, 실제 실행 중인 두 training supervisor 의존성 검사, prediction→score의 실제 process handoff가 모두 통과했다. 등록된7개 training/prediction/score process가 실제 live인 것을 확인했다. VAL mask는 아직 읽지 않았다.

Prediction은 양쪽 training artifact가 모두 완료돼야 시작한다. Conditional은 기존 SourceT foreground를 사용하고, standalone은 같은 learned rollout의 production native-CPU QCC에서 foreground를 다시 얻는다. 둘 다 같은 actual GPU-Q/A와 기존 signed induced merge를 사용한다. 모든 prediction을 고정한 뒤에만 기존 고정320/modal32 GT와 세 production 지표를 계산한다. Bootstrap은10000회, 공통320 이미지, RNG168이며 사전등록한 두 co-primary gate를 모두 적용한다.

**최종 FG-ARI 결과는 아직 없다.** SW165가 incumbent이며 .87 평균·gating 기여·fresh70k/data scaling은 이 실행 시작으로 달성됐다고 주장하지 않는다.

증거: `results_archive/training_started_20261015/server_records.json`. 생성 파일은 기존 junction을 통해 `collaborative_testing/generated/`에 저장한다.
