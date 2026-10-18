# SW0209: FG pair-ARI surrogate를 실제 spike loss에 추가

208 전체320 seed0의 기준 warm192 .877709/.791224/.696284를 보존한다.
새 학습은 같은 실제192 model/Adam/CPU-CUDA RNG에서 분기한다. Core,
CNN, gamma ordering, graph,1024-step dynamics/settle512/native64 readout은
변경하지 않는다. 원본 QCE와 모든5개 family 학습을 유지한다.

QCE는 class 평균 affinity의 상대 순위를 학습한다. 이번에는 GT foreground
안에서 Q를 soft pair coassignment로 보고, chance-corrected pair agreement
loss를 weight1로 추가한다. GT background pair와 diagonal은 제외한다.
Q가 hard partition의 동치 행렬이면 이 값은 실제 FG-ARI와 정확히 같지만,
일반 Q는 partition이 아니므로 **실제 최종 mask ARI의 미분 가능한 동일식이
아니다.** 그 proxy가 최종 classifier에도 도움이 되는지 검증하는 실험이다.

M=foreground unordered pairs, T=GT same-object pairs, P=sum(Q_ij),
I=sum(GT_same*Q_ij), E=T*P/M 이다. loss=1−(I−E)/(.5*(T+P)−E).
상하 Q는 평균하고 float64로 reduce한다. 기존 QCE temperature.1과
계수1을 고정하며 validation threshold/LR/가중치 sweep은 하지 않는다.

추가 TRAIN은 기존204/202와 같은 고정8개(indices8..15), B4 두 proposal,
micro1/live-tail64다. 유한 Adam step은 유지하고 실제 descent 여부를 따로
기록한다. Source의 TRAIN8을 포함한 논리적 추가 adaptation은16장이다.
실제 masks를 만들 때 GT, GT object count, 이 surrogate를 사용하지 않는다.

비교는 완료된202 finite-Adam 원본 QCE control과 warm192 모두를 사용한다.
고정16 development offsets0,2,..30/seed0, inference RNG193000+offset,
실제 masks의 세 metric과 paired95%CI를 평가한다. Candidate−control FG≥.01,
CI lower>0, candidate−warm FG≥0, 두 ObjectIoU delta≥0, 실제 CNN update와
new accepted update가 있어야 확대한다. 실패하면 같은계수의 작은 rescue
sweep은 닫는다. 70k 신규 학습이나 세 seed native64 .90이라고 부르지 않는다.

기존191의 discrete error-mined squared penalties,199의 absolute calibration,
204의 component QCE와 다르다. 이번에는 foreground pair counts의 chance
correction/FP-FN 균형이 직접 loss를 결정한다. 과거 실패를 성공으로 재분류하지
않으며, 실제 readout과 proxy가 불일치할 가능성도 결과에 명시한다.
