# SW0219: 실제 Q만 사용하는 readout 가중 ablation

217의 actual Q pair AUC는 native graph-weighted energy보다 높았다. Graph가
spike를 생성하는 dynamics는 유지하고, decoder에서 A가 energy 크기에 주는
추가 가중만 제거하면 어떠한지 확인한다. 물리적 graph/gate 전체 제거 실험이
아니며 gating 인과 증명이라고 하지 않는다.

실제 best-reference seed0/1/2, 같은16 offsets0,2,..30와217의 실제 private
Q_F/A_F/gain을 재사용한다. 새 rollout, encoder/core update, GPU 사용이 없다.
Q_F 밖은 neutral Q=.5/A=0으로 채워 full-N 연산을 호출한다. F 내부 actual
Q/A는 그대로이고, source 전경 밖은 merge에 참여하지 않으므로 정규화를
바꾸지 않는다. 원래 SW0164 helper를 호출해 baseline mask를 먼저 bitwise
재현해야 한다. 이후 같은 helper에서 A만0으로 두어 logit(Q)의 signed
agglomeration을 실행한다. Threshold .5, 4-connectivity, positive cross-sum,
tie 순서, singleton 보존, 원래 전경과 NATIVE_NODES1024는 유지한다.

198의1024->4096 normalizer와 다르게 이번에는 readout A weighting의0/1
비교다. 중간 alpha/threshold sweep을 하지 않는다. 163의 learned context
assignment Q_only head와도 다르다. 단일 사전 고정 후보이며217 진단 이후의
exploratory validation이다. 반복 사용한16장은 untouched test가 아니다.

모든96 endpoint를 먼저 고정/검증하고 GT를 읽어 세 지표, seed별 결과,
paired-image bootstrap10000 및 세 고정 모델 평균을 계산한다. 확대 gate는
평균 FG+.01 이상,95% CI lower>0, ObjectIoU 감소 없음. FG-only 유용 후보는
따로 보존한다. Mini .90을 전체320 세 seed .90 또는70k로 바꾸어 말하지 않는다.
원래 source baseline/CNN/core/학습 이력과 최고 full320 .848615는 그대로다.
