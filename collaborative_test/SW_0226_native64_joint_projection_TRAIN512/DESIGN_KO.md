# SW0226: 공동학습의 고유 TRAIN64 → TRAIN512 비교

223번과 같은 원래 seed0/1/2 checkpoint에서 시작한다. mask-supervised 고정 source
object-frequency teacher와 gamma anchor, encoder 전체 및 실제 4×8 core projection
학습을 유지한다. 다른 core/graph/gate는 고정한다. TRAIN pool의 첫512개 중 64개만
반복한 223번과 비교하며 128 Adam updates, batch16, 총2048 exposures, LR/clip은 같다.
512장은 64장의 정확한 prefix를 포함하고 그 prefix RGB/GT/teacher SHA를 검증한다.
초기 모델과 DEV16 RGB/baseline도 동일해야 한다. 221번 unit-margin loss 재시도가 아니다.

223번 TRAIN64의 학습과 mask는 재사용한다. 새로운 TRAIN512 세 seed의 최종 mask를
모두 동결한 후 IDs1320,1322,..1350의 원래 native64 modal GT를 읽는다. 1024 steps,
settle512 및 원래 spike/membrane readout을 유지한다. paired image bootstrap은 기존
recipe와 같은 RNG221/10000회이다. full320 후보 기준은 source 대비 FG≥.01, CI하한>0,
TRAIN64보다 높은 FG이고, balanced 후보는 두 IoU도 source 대비 비감소여야 한다.
별도로 TRAIN512−TRAIN64 FG 차이 CI하한>0일 때에만 이 고정모델·미니범위의 data
scaling 증거로 기록한다. seed 모집단, 70000장 scaling, untouched test의 증명이 아니다.
전체 목표는 full320에서 세 seed 평균 all3 Slot 초과와 FG .90이며 mini로 축소하지 않는다.

225번에서 진단된 THP 지연을 피하도록 Linux 큐/worker에만 huge page를 끈다.
전역 설정과 연산식은 변경하지 않는다. GPU0..3 중 다른 compute 소유자가 없는
장치만 공유 lease로 예약하고 외부 작업이 나타나면 본인 worker만 yield한다.
