# SW176: 전체 관측 구간에 local phase gradient 전달

## 근거와 가설

SW175 critical-component loss는 고정320의 conditional/standalone FG와 IoU 개선에 실패했다. SW173에서 full1024 gradient는 tail64 대비 매우 크고 방향도 달랐으며, SW174에서 event surrogate 경로만 제거해도 이 현상이 남았다. 이번에는 loss나 forward 구조가 아니라 긴 Kuramoto 재귀를 거친 gradient 전달 정책 하나를 바꾼다.

각 step의 Kuramoto `theta_prev`만 detach하고, 같은 drive/A/coupling과 원래 연산으로 theta output을 계산한다. history에 저장한 각 output은 live로 유지해 delayed gate에서 해당 step parameter로의 gradient를 남긴다. D/M/event/reset recurrence는 유지한다. 전체1024 frames를 grad-enabled로 계산해 마지막512 settled frames의 objective가 각 local phase step에 직접 gradient를 줄 수 있도록 한다. 이는 exact full gradient가 아닌 명시적인 approximate gradient다. 모든 긴 recurrence를 제거한다거나 효과가 입증됐다고 주장하지 않는다.

## 변경과 대조

forward, gate 구조, threshold, 실제 component spike, production Q, 실제 A와 learned coupling gain, 추론 readout을 바꾸지 않는다. supervised critical-component objective, negative Q 항, 기존 source checkpoint+seeded zero-final context, frozen base encoder, TRAIN512×동일 두 순서, B16/micro4, optimizer/LR/clip/projection,64 updates를 SW175 critical-component arm과 맞춘다. 후보는 fresh source에서 시작한다.

완료된 SW175 critical-component arm의 초기 state, source/preprocessing/frozen 구성, TRAIN 순서, optimizer 설정, update cursor/경계, RNG 조건을 대조군으로 재사용할 수 있는지 먼저 검증한다. 일치하지 않으면 fresh matched tail64 대조군이 필요하다. 학습 정책을 제외한 차이가 남으면 직접 인과 비교를 주장하지 않는다.

## 실행 전 검증

CPU graph 검사는 동일 forward, 이전 theta로의 gradient 차단, local parameter 및 delayed output으로의 gradient 보존, context manager 정상/예외 복원을 확인한다. 실제 TRAIN 캐시와 완료된 control provenance로 CPU admission을 통과해야 한다.

짧은 GPU preflight는 TRAIN의 첫 고정 이미지 한 장에서 parameter update 없이 원본과 후보의 six traces, actual Q/A, conditional/standalone masks 동일성을 검증한다. 전 과정에서 VAL/reserve 정답은 읽지 않는다. fullwindow gradient의 유한성, 실제 S와 context/graph/Kuramoto/integration 각 family의 credit, 시간과 peak memory를 기록한다. forward/provenance 불일치나 비유한 값은 구현/가설 문제를 해결하기 전 학습하지 않는다. 작은 진단 loss의 하락 자체를 학습 확대 조건으로 사용하지 않는다.

64-update 학습을 구현할 때 첫 두 committed updates에서 실제 finite gradient·parameter 변화와 메모리를 확인하며, 이를 별도의 disposable update로 중복 실행하지 않는다. 그래프의 한계로 B4가 실행되지 않으면 조건 변경을 숨기지 않고 새 설계를 등록한다. 현재 문서는 trainer 또는 GPU preflight가 실행됐다는 증거가 아니다.

## 평가와 종료

같은 native32 고정320, seed1의 conditional Source-T foreground와 standalone native QCC foreground를 사용한다. 두 모델의 모든 예측을 확정한 뒤에만 정답을 읽는다. FG-ARI / foreground IoU / matched-object IoU를 모두 기록하고 candidate FG가 SW164 seed1과 matched tail64 control보다 각각 높으며 paired bootstrap10000회(seed176)의 95% lower CI가 0보다 높아야 한다. 두 IoU 평균도 각 비교군 이상이어야 한다. 두 endpoint가 모두 조건을 만족할 때에만 seed0/2 및 더 큰 고유 TRAIN 데이터로 확장한다.

실패하면 이 정책을 종료하고 epsilon/LR/threshold/time-window 미세조정으로 구제하지 않는다. 성공해도 gating의 인과적 필요성, 세 seed 평균의 Slot Attention 초과, FG-ARI .87, native32 70k fresh training과 positive data scaling은 각각 별도 검증이 필요하다. 기존 최고 모델은 결과가 검증될 때까지 유지한다.
