# SW0221: 실제 감각 주파수의 물체별 일관성을 학습

기존 최고 모델(seed0 warm192, seed1/2 warm205)을 그대로 시작점으로 쓴다.
전체 encoder(base와 context)를 학습하고 core, gate, graph 파라미터, gamma 통계,
기존 spike/membrane readout은 고정한다. 최종 mask는 원래 1024-step 경로만 사용한다.
별도 RGB mask decoder나 Slot Attention head를 추가하지 않는다.

학습 대상은 raw gamma가 아니라 실제 gamma_to_drive의 projection, 공간 표준화,
tanh를 모두 통과한 (omega + freq_gain * drive) / (abs(freq_gain) * pi)다.
실제 freq_gain이 0이 아니므로 원형 위상 거리만으로 같은 주파수라고 취급하지 않는다.
GT 물체 내부 분산과 4-neighbor 접촉 물체 평균 간 거리의 unit-margin 제곱 hinge를
동일 가중치로 최소화한다. 배경도 하나의 class다. 멀리 떨어진 동일 외형 물체를
억지로 분리하지 않는다. 171번의 graph projection CE와 다른 입력 물리량이며,
당시 고정했던 base encoder도 이번에는 학습한다. 성공을 가정하지 않는다.

세 seed 각각 TRAIN64/512 두 arm, 128 update, B16, 노출 2048회로 고정한다.
기존 TRAIN512 순서의 prefix를 쓴다. base LR=3e-5/context LR=3e-4, clip=1,
fresh Adam, 마지막 128 step만 평가한다. 이는 mask-supervised encoder pretraining이며
core 공동 학습이나 새 native64 70,000장 학습이 아니다. 이전 70k/core와 native32
TRAIN512 supervision을 상속한다. 64/512 비교는 추가 native64 학습 단계의 데이터
다양성 비교이며 최초 학습 전체의 scaling 또는 새로운 고유 데이터 증가 증명이 아니다.

DEV는 기존 16장 IDs1320,1322,...1350, RNG193000+offset, canonical128/no crop,
2px modal GT. 여섯 arm의 코드와 최종 mask protocol을 모두 고정한 후 DEV GT를 읽는다.
학습 loss 감소와 입력 주파수 개선만으로 최종 mask 향상이나 gating 인과성을 주장하지 않는다.
FG-ARI 평균 +.01, paired image bootstrap CI 하한>0, Object IoU 평균 비감소인
arm만 전체320 검증 후보로 삼는다. 전체320 검증 전 기존 최고 기록과 .90 목표는 유지한다.
CPU admission, 실제 encoder checkpoint/resume, full scoring branch 검증 후 배포한다.
