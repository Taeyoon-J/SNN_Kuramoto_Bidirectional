# SW0221: 분리 margin을 강제한 encoder 학습은 성능 하락

기존 최고 native64 세 source 모델에 같은 128 update/B16, fresh encoder Adam을
적용했다. core/gates/readout/feature 통계는 고정했다. 여섯 최종 mask protocol을
모두 동결한 뒤 기존 DEV16(1320,1322,...1350)의 2px modal GT를 읽었다.

| 추가 TRAIN 단계 | 세 seed 평균 FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 기존 source | .842891 | .600692 | .702161 |
| TRAIN64 | .705741 | .243287 | .581092 |
| TRAIN512 | .693004 | .239999 | .574074 |

세 seed 각각에서도 두 arm 모두 세 지표가 내려갔다. 기준 대비 FG 차이는
TRAIN64 -.137150, paired image 95% CI[-.222561,-.065948],
TRAIN512 -.149887, CI[-.229969,-.079548]이다. 따라서 둘 다 확대 gate 실패다.
512-64 차이 -.012737의 CI[-.034828,.010643]는 0을 포함하므로,
이 결과를 일반적인 데이터 증가의 역효과라고 해석하지 않는다.

중요한 구별: TRAIN total loss는 감소했지만, 물체 내부 주파수 분산은 감소하지 않았다.
TRAIN64의 seed0/1/2 내부 분산은 .004660/.013782/.005229에서
.040941/.074095/.047621로 증가했다. TRAIN512도
.004777/.014256/.005424에서 .041751/.081332/.048474로 증가했다.
기존 입력의 내부 분산은 이미 작았고, 총 loss 감소는 주로 이웃 class 사이의
unit-distance margin 위반을 줄이는 항에서 나왔다. 총 loss 감소를
물체 내부 rhythm 일관성 개선이라고 주장할 수 없다.

해당 unit-margin encoder branch는 종료한다. LR/epoch/margin 수치 sweep이나
이 설정의 70,000장 확대는 하지 않는다. 다음 후보는 기존 source의 실제 물체별
주파수 중심을 보존하면서 내부 변동만 줄이고 원래 gamma 입력을 anchor로 유지하는
학습이다. 이는 검증 전 가설이며, 아직 새 모델 성능으로 취급하지 않는다.

이 시험은 mask-supervised 추가 native64 encoder 단계다. 상속한 70k/core 및
native32 TRAIN512 supervision이 있으므로 신규 고유 데이터 64/512 전체 학습이나
unsupervised scaling 증명이 아니다. native64 전체320 최고 세 seed 평균
.848615/.592533/.672047은 유지되며 .90 목표는 아직 미달이다.

후속 읽기 전용 frequency 진단도 완료했다. DEV의 물체 내부 분산은 TRAIN64의
seed0/1/2에서 .005064/.017437/.007019 -> .054533/.078117/.048307,
TRAIN512에서는 .052330/.088710/.050701로 증가했다. 두 arm 모두 DEV의
물체간 margin 항도 줄었으므로 단순히 TRAIN objective만 암기한 결과는 아니다.
기존에 작은 주파수 대역을 큰 폭으로 넓히면서, 물체 내부 변동도 함께 커졌다.
이 변화와 mask 하락은 확인했지만 개별 gate의 붕괴나 유일한 인과 원인을
이 진단만으로 특정하지 않는다.

실제 여섯 encoder checkpoint의 SHA/cursor/Adam128/history를 대조했고, 독립
post-score 진단에서 원래 source+최종 encoder 모델 여섯 개와 mask를 재검증했다.
144개 mask 평가 벡터의 평균/paired bootstrap CI 재계산 차이는 0,
두 metric 구현 최대 오차는 2.22e-16이다. 새 model forward/추가 학습은 0이다.
증거는 results_archive/completed_v1의 summary.json, frequency_summary.json,
completion_audit.json이다. 원시 imagewise vector와 checkpoint는 공개하지 않는다.
