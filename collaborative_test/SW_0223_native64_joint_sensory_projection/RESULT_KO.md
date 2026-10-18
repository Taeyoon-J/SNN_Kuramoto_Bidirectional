# SW0223: 세 지표 평균 개선, source 대비FG 확신은 아직 부족

기존 source/teacher/TRAIN64/128 updates/B16/gamma anchor/encoder LR을 유지하고
실제 gamma_channel_proj.weight32개만 core에서 추가 학습했다. omega/frequency gain/
phase gain/graph 및 나머지 gate 상태는 고정했다. 원래64x64/1024 settle512
readout, 동일 DEV16과2px modal GT를 사용했다. 세 candidate 및222 control
protocol을 모두 동결·검증한 뒤 GT를 읽었다.

| 세 seed 평균 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 기존 source | .842891 | .600692 | .702161 |
| 222 encoder-only | .854272 | .631545 | .695736 |
| 223 공동 학습 | .859634 | .645611 | .706340 |

source 대비 차이는 +.016743/+.044919/+.004179다. FG 차이의 paired image95% CI
[-.000361,.040940]가0을 포함하므로 사전 FG/balanced full320 gate는 모두 실패했다.
Foreground IoU CI[.003369,.092662]는0보다 높다.222 대비 FG 차이+.005362,
CI[.000062,.012050]로 이 고정16장·세 모델에서는 공동 학습의 FG 평균이 높았다.
이 CI는 고정 세 모델에 대한 image bootstrap이며 seed 모집단 또는 모델 선택을
보정한 불확실성은 아니다. 확정적인 일반화나 .90 달성으로 승격하지 않는다.

실제 projection/encoder Adam128 및 첫 projection gradient 양수, projection SHA
변경, 나머지 core SHA 보존을 확인했다. source/222/223의144개 raw mask를
독립 metric으로 재평가했고 평균·CI 오차 최대2.22e-16, 로컬 평균/CI 차이는0이다.
DEV 내부 주파수 분산은 세 seed 모두222보다 추가로 줄었다. 이 입력 통계가
recurrent rhythm이나 각 gate의 인과 기여를 증명하는 것은 아니다.

다음 확인은223 checkpoint를 고정하고 평가만64장으로 늘리는 별도225번이다.
223의16장 gate 실패는 그대로 유지하며, source 코드를 수정하거나 더 학습하지 않는다.
정해진64장 validation을 한 번만 확인하고 성능이 불확실하면 자동으로70k나
full320 확대를 하지 않는다. untouched test 또는 전체 학습 데이터 scaling 시험이 아니다.

native64 전체320 최고 세 seed 평균 .848615/.592533/.672047은 유지한다.
증거: results_archive/completed_v1/{summary,completion_audit,archive_receipt}.json.
이 모델도 상속한70k/core 및 native32 TRAIN512 supervision을 포함하는 supervised
계열이다. 새 native64 70k 학습은 수행하지 않았다.
