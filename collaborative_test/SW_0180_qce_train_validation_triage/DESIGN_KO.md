# SW0180 — 같은 TRAIN8의 학습 전후 paired replay

SW0179 QCE의 고정320 FG-ARI는 seed 0/1/2가
0.849988 / 0.818029 / 0.616842였습니다. Conditional SourceT foreground를
유지한 seed 2도 0.681013이므로 배경 정책 하나로 실패를 설명하지 않습니다.
기존 SW0165를 유지하고, 실패한 QCE 설정의 70,000장 확대·seed 2 단독
재학습·온도/LR/배경 정책 sweep을 하지 않습니다.

이번에는 공통 SW0171 TRAIN512의 고정 positions 256..263, 총 8장을 씁니다.
각 seed 0/1/2에 대해 fresh same-seed SourceT+zero-context와 학습 완료 QCE를
같은 RGB·정답·B4·RNG에서 비교합니다. 두 checkpoint 상태 × 8장 × 3 seed,
총 48 image forwards이며 T1024 / settle512 / live-tail0, no-grad·no-optimizer입니다.
추가 학습이나 validation/reserve 이미지·정답 읽기는 하지 않습니다.

Fresh 모델은 SW0179의 seed별 초기화 factory를 사용합니다. Seed 1 fresh
tensor SHA는 기존 SW0178 초기 상태와 일치해야 합니다. Final seed 0·2는
SW0179, final seed 1은 변경하지 않은 SW0178 product-QCE checkpoint입니다.
같은 seed의 실제 모델을 strict-load하고 CPU admission 후 GPU를 초기화합니다.
데이터의 seed 1 producer provenance와 각 모델의 source seed를 분리합니다.
Seed 0·2에서 seed 1 cached spike/support를 대신 사용하지 않습니다.

모든 mask는 실제 emitted component spikes → native positive product Q →
production QCC → 기존 SW0165 classifier 경로로 만듭니다. Fresh SourceT의
TRAIN8 QCC foreground를 conditional support로 보존하고 final Q/A와 trained
graph gain으로 분류합니다. Standalone은 각 모델 자신의 QCC support를 씁니다.
Validation에서 저장된 foreground는 TRAIN 입력으로 쓰지 않습니다.

이미지별로 등록된 product-QCE loss(temp 0.1, full modal GT/BG0, LOO,
class-balanced mean), true-class minus best-other-class margin, 두 readout의
FG-ARI/Foreground IoU/Object IoU, foreground precision/recall을 기록합니다.
표본/클래스 집계와 singleton의 유효성은 기존 QCE loss 정의에 맞춥니다.
Source/final hash, 공통 RGB/정답/IDs hash, RNG, GPU, model/state 불변과
optimizer update 0을 기록합니다. 다른 사용자 GPU 작업은 건드리지 않습니다.

결과를 본 뒤 다음 방향을 고르는 작은 진단이며 원인 확정이나 성능 승격이
아닙니다. 같은 이미지의 raw CE·margin·mask 변화와 기존 고정320 결과를
함께 보고 다음 세 가설을 구분합니다.

- TRAIN에서도 CE 상승·mask FG 하락: optimization/representation instability 쪽 증거.
- CE 하락에도 TRAIN mask FG 하락: loss와 QCC/최종 readout 불일치 쪽 증거.
- CE 하락·TRAIN mask FG 상승에도 고정320 하락: 작은 학습 표본에 대한 적합/일반화 쪽 증거.

혼합되거나 변화가 작으면 불확실하다고 기록합니다. 8장으로 가설을 입증하지
않으며 nonzero gradient만으로 gating의 유용성을 주장하지 않습니다.
