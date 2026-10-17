# SW0195: native64 학습/검증 변화 분리

상태: 진단 코드 준비 중. 학습 또는 GPU 진단을 실행했다는 뜻이 아니다.

SW0194를 변경하지 않는다. 완료된 TRAIN8 및 TRAIN64 모델의 같은 production
spike/Q readout을 사용한다. 추가 학습, loss 변경, threshold 탐색은 하지 않는다.

TRAIN64의 각 B4 proposal에서 첫 이미지 하나씩, 인덱스 0,4,...,60 총16장을
미리 선택한다. 두 모델에서 같은 이미지와 RNG(195000+offset)를 사용한다.
TRAIN8 모델에는 이 중 두 장만 native64 추가 학습에 사용되었다. 따라서 이 비교는
전체 모델의 최초 학습 전후 비교가 아니라 56장 추가 adaptation 전후 비교이다.
기존 SW0194의 32장 development 결과를 재사용한다. 두 cohort의 image 난이도가
같다고 가정하지 않고, 각 cohort 내부의 paired 변화 및 그 차이를 보고한다.

세 지표: FG-ARI, foreground IoU, matched object IoU. TRAIN 및 development 모두
같은 64x64 modal GT와 production mask 계산을 사용한다. 모든 TRAIN prediction을
먼저 고정한 후 GT로 점수를 계산한다. imagewise 점수/GT는 공개하지 않는다.

각 cohort에서 paired bootstrap 10,000회, 두 cohort의 resampling은 독립이다.
TRAIN의 FG 변화 CI 하한>0이면서 development 변화 CI 상한<0일 때만
이 adaptation에서 과적합과 일치하는 신호라고 표시한다. 원인 규명으로 해석하지
않는다. raw TRAIN>development 점수차나 minibatch loss 감소만으로 판정하지 않는다.
이 작은 cohort는 최종 held-out test나 세 seed 검증을 대체하지 않는다.

실행 전 요건: SW0194 완료 report/checkpoint/validation protocol 검증,
두 모델 digest 확인, CPU 검사 통과, 독점 idle GPU 배정. 진행 중인194 소스와
fingerprint는 변경하지 않는다. 단순한 대기 timeout으로 재학습하지 않는다.
