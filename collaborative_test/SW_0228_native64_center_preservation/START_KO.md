# SW0228 예약: 원래 주파수 중심 보존 loss 비교

로컬14개·Linux14개 테스트와3개 CLI 검사를 통과했다. uniform center drift의
loss/gradient만16배로 강해지고, 중심을 움직이지 않는 내부 편차에 대한 계수는
유지되는 것을 확인했다. class ID를 바꿔도 loss가 같으며 임의의 class/channel
순서를 만들지 않는다. 기존 source teacher3개 배열이 동결된223번 대조군과 같고,
세 실제 source 모델의 projection gradient가 유한·양수이며 초기 자연 분산을
1.05e-9 이내로 재현함을 CPU에서 검증했다.

세 seed supervisor는 실제 살아 있고 waiting_for_full320_first_attempts 상태다.
**아직 GPU 학습이나 성능 결과는 없다.** 227번의 각seed가 첫 GPU 배정을 받은
기록이 생긴 뒤에만228 큐가 빈 GPU0..3을 요청한다. 기존 전체320장 확인을 먼저
배정하면서 이후 빈 장치를 자동으로 사용할 예약이다. 이 조건은227 완료나 성능을
의미하지 않는다. 외부 GPU 작업이 나타나면 본인 worker만 양보한다.

223번과 같은 초기 모델·TRAIN64·128 Adam updates·batch16·LR/clip·encoder 및
32-weight core projection을 유지한다. 바꾸는 것은 source-center MSE의 계수1→16뿐이다.
class_labels는 TRAIN teacher에만 추가하며 GT 없이 예측하는 readout에는 주지 않는다.
다른 core/graph/gates/statistics, native64/RNG/readout/DEV16은 그대로다.

mask를 모두 동결한 후 세 지표와 source-center drift/within/인접class 통계를 함께
확인한다. 계수 sweep이나70,000장 재학습은 이번에 하지 않는다. 목표는 전체320장에서
세seed 평균이 같은 Slot을 세 지표 모두 넘고 FG≥.90인지 검증하는 것이며, 이번
mini/CPU gradient 검사로 그 목표나 모든 gating의 인과 기여를 달성했다고 보지 않는다.
