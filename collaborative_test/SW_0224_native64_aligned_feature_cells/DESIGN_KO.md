# SW0224: encoder feature 중심과 기존2px 평가 cell의 정렬

별도 구조 가설이다. 기존 encoder는128 RGB에서 valid3x3 및 context crop 후
126x126 feature를 낸다. 기존64 adaptive pooling은 축당64개 중60개 cell에서
feature3개를 평균하고62개 인접 경계에서 겹친다. receptive-field center 기준으로
기존 균일2px GT cell 중심 대비 평균 절대 오차는 축당 .5px다.
이는 기하 확인이며 성능 하락 원인이라고 아직 주장하지 않는다.

기존 feature 값/정규화/clip은 그대로 두고 가장자리 feature를 한 줄 복제해
128x128로 만든 뒤 같은64 patcher를 쓴다. 내부 cell은 정확히 비중첩2x2 평균과
대응한다. 경계는 가장 가까운 기존 feature를 복제한다. RGB/GT/HDF5/ID/평가 규칙,
CNN/core 가중치,1024 settle512 spike readout은 바꾸지 않는다. 추가 학습은0이다.
모델 식별자는 같은 parameter SHA와 새 adapter 코드 SHA를 함께 쓴다.

warm192/205 세 seed 공통. DEV16 IDs1320,1322,...1350, RNG193000+offset,
canonical128/no crop/2px modal GT. 기존 동일 모델의 mask가 control이다.
세 최종 protocol을 모두 검증한 뒤 GT를 읽는다. source checkpoint bytes와 모델
state SHA가 유지되고 source statistics가 변하지 않는지도 확인한다.
이 방법은 기존 구현과 수치적으로 동등하다고 주장하지 않는다.

FG track full320 후보는 기존 source 대비 FG+.01 및 paired image CI하한>0,
balanced 후보는 여기에 두 IoU 비감소를 요구한다. tradeoff를 함께 기록하고
full320/.90 성공을 mini 결과로 대체하지 않는다. 새70k 학습은 하지 않는다.
source의 mask-supervised lineage는 그대로 상속한다. 각 gate의 인과 기여나
데이터 scaling을 이 시험만으로 입증하지 않는다.
