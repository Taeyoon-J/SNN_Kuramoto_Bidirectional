# SW0222: 내부 주파수 일관성 개선, FG 평균 소폭 상승은 아직 불확실

기존 세 native64 source에서 source의 TRAIN 물체별 주파수 평균을 teacher로
고정하고 gamma anchor와 함께 encoder만 학습했다. TRAIN64,128 update/B16,
기존 core/gates/feature 통계/1024-step spike readout을 유지했다. 세 최종 protocol을
모두 동결한 뒤 기존 DEV16, canonical128/no crop,2px modal GT로 평가했다.

| 세 seed 평균 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 기존 source | .842891 | .600692 | .702161 |
| source teacher denoising | .854272 | .631545 | .695736 |
| 차이 | +.011381 | +.030853 | -.006425 |

FG 차이의 paired image 95% CI는 [-.004210,.034446], Foreground IoU는
[-.006984,.075435], Object IoU는 [-.038799,.032782]로 모두 0을 포함한다.
사전 full320 gate(FG+.01/CI하한>0/Object IoU 비감소)는 실패했다. FG track 후보는
유지하지만 확실한 개선, 새로운 full320 최고, .90 달성으로 승격하지 않는다.
이 단계에서 70,000장 확대나 LR/epoch/anchor 계수 sweep을 하지 않는다.

| seed | 기존 FG | denoising FG |
|---|---:|---:|
| 0 | .924401 | .925242 |
| 1 | .815856 | .807814 |
| 2 | .788414 | .829759 |

평균 상승은 주로 seed2에서 나온다. seed1은 하락했고 seed0의 변화는 작다.
seed0의 .925242는 반복 사용한 16장 mini 결과이며 전체320 결과가 아니다.

221번과 달리 이번에는 실제 DEV 내부 주파수 분산이 세 seed 모두 감소했다.
seed0 .005064 -> .003530, seed1 .017437 -> .014881,
seed2 .007019 -> .005821로 약15~30% 감소했다. 반면 접촉 class 사이 평균 거리도
줄었고, 내부 일관성 개선만으로 모든 seed의 mask가 향상되지는 않았다.
source 물체 중심 유지 MSE 및 TRAIN gamma anchor는 감사 결과에 함께 보관했다.
이는 입력 주파수 통계이며 recurrent rhythm, 각 gate의 인과 기여, 데이터 scaling의
증명이 아니다.

실제 세 encoder checkpoint/Adam128/history/고정 teacher 및96개 raw mask를
검증하고 독립 metric으로 재평가했다. metric/평균/paired CI 최대 차이2.22e-16,
별도 로컬 평균/CI 재계산 차이는0이다. frozen scorer가 실제 source+최종 encoder
모델을 복원한 검증을 재사용했고, 감사에서 비싼 full-model factory나 추가 forward를
반복하지 않았다. 집계 증거는 results_archive/completed_v1에 있다.

상속한70k/core 및 native32 TRAIN512 supervision이 있는 supervised 추가 단계다.
새 native64 70k 학습이나 unsupervised 성능으로 취급하지 않는다.
native64 전체320 최고 세 seed 평균 .848615/.592533/.672047은 그대로 보존한다.
다음 구조 후보는 encoder에 더해 실제 gamma-to-drive projection을 함께 학습해,
주파수 일관성의 개선이 spike 구분으로 전달되는지를 같은 조건에서 확인하는 것이다.
이 후보는 아직 구현·실행 결과가 없다.
