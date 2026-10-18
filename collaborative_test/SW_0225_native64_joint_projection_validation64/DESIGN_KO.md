# SW0225: 고정223 모델의 평가 범위를64장으로 확인

223의 DEV16 세 seed 평균 .859634/.645611/.706340은 source 대비 세 지표가
올랐지만 FG 차이 CI[-.000361,.040940]가0을 포함해 사전 full320 gate는 실패했다.
encoder-only 대비 FG 차이는+.005362 CI[.000062,.012050]였다. 모델을 더
학습하거나 LR/loss를 바꾸지 않고 한 번만 더 넓은 validation으로 확인한다.
이 별도 확인은223의16장 gate 실패나224 결과를 변경하지 않는다.

223의 최종 seed0/1/2 checkpoint를 고정한다. IDs1320..1383의 첫64장,
canonical128/no crop/2px modal GT/1024 settle512 원래 spike readout,
RNG193000+offset이다. 기존16장 offset0,2,..30 mask는 hash 및 RGB를 대조해
재사용한다. offset0은 native forward로 재생해 bitwise mask 일치를 확인한다.
추가 native forward는 seed당49장(새48장+재생1장), 추가 학습은0이다.
baseline은 동일 source 모델의 기존 full320 mask에서 같은64장을 쓴다.

세 모델/64장 최종 protocol을 모두 동결한 뒤 GT를 읽는다. paired image
bootstrap10,000회 RNG225. FG source 대비+.01/CI하한>0이면 full320 FG-track
후보, 여기에 두 IoU 비감소면 balanced 후보로 기록한다. final 목표는 같은
full320 평가의 세 seed 평균으로 Slot의 세 지표를 엄격히 초과하고 FG .90도
검증하는 것이다. 이64장 확인만으로 final 성공이나70k scaling을 주장하지 않는다.
고정 validation을 반복 사용하는 연구 단계이며 untouched test가 아니다.
