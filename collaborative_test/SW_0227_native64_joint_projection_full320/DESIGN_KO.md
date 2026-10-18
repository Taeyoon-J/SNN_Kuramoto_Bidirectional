# SW0227: 고정 공동학습 모델의 전체320장 확인

225번의 사전 정의된 balanced 확대 기준을 통과한 223번 seed0/1/2 checkpoint를
고정한다. 225번 64장 평균은 .877164/.597239/.670309이고 FG 개선 CI는 양수다.
원래16장 gate 실패 및64장 선택 이력을 유지한다. 추가 학습은0회다.

canonical128/no crop, native64, IDs1320..1639, 원래2px modal GT, 1024 steps와
settle512, RNG193000+offset 및 원래 spike/membrane readout을 유지한다.
225번의 첫64개 mask와 frequency를 hash 검증 후 재사용하며 나머지256장을
새로 예측한다. offset0을 추가로 실행해 mask와 frequency 모두 bitwise 재현해야
진행한다. checkpoint/전체 RGB/64장 prefix/원래16장/control baseline을 CPU에서
검증한다. Linux 큐와 worker에만 THP를 끄고 전역 설정은 변경하지 않는다.

세320장 protocol과 원래 comparable Slot 세 seed의 prediction/protocol/hash/IDs를
모두 검증한 후 GT를 읽는다. Slot은 기존154/210의 우리데이터 학습 reference이며
공개 checkpoint의 다른 계약 .8946 수치와 혼동하지 않는다. source 및 Slot 재계산
평균은 기존210의 전체320장 집계와1e-12 이내로 같아야 한다. paired image
bootstrap10000회 RNG227로 candidate−source 및 candidate−Slot을 함께 기록한다.
같은 전체320장 계약에서 세seed 평균이 Slot을 세 지표 모두 엄격히 넘는지와
FG≥.90 여부를 별도 판정한다. seed 모집단의 CI나 untouched test라고 주장하지 않는다.

70,000장 재학습은 없다. mask-supervised lineage와 다른 core/graph/gate 고정 상태를
유지하며 모든 gating의 인과 기여나 data scaling을 이 확인만으로 주장하지 않는다.
226번 TRAIN64→512 비교는 별도로 계속한다. GPU0..3을 독점 shared lease로 예약하고
foreign compute가 나타나면 본인 worker만 yield한다. 기존 작업을 중복 등록하지 않는다.
