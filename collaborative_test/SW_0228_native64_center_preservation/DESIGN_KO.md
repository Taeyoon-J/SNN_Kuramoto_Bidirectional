# SW0228: 원래 주파수 중심 보존 강화

223/226 진단에서 물체 내부 주파수 분산과 함께 인접 class 중심 거리도 줄었다.
특히 약한 seed1/2에서는 내부 분산과 중심 거리의 감소율이 비슷했다. 데이터512
확장 효과도 작고 CI가0을 포함했다. 이는 단일 원인 증명이 아닌 다음 loss 가설이다.

223번과 같은 원래 초기 checkpoint, TRAIN64,128 Adam updates, batch16,LR/clip,
encoder 전체 및 실제32-weight 입력 projection 공동학습을 유지한다. 다른 core,
graph/gates/statistics/readout은 고정한다. 기존 source teacher의 class별 주파수
중심을 그대로 사용하며 임의의 단위1 margin이나 class/channel 순서를 만들지 않는다.

기존 teacher MSE는 class-balanced within variance + source-center MSE로 분해된다.
이번 loss는 within variance +16×source-center MSE +원래 gamma anchor다. source
중심에서 벗어나는 벌점만15배 추가하고 within 항의 계수1은 유지한다. 계수16은
현재 validation 진단에서 선택한 하나의 사전 고정 비교이며 계수 sweep은 하지 않는다.
TRAIN class_labels는 teacher에만 추가하고 예측/GT 이전 readout에는 주지 않는다.
기존3개 teacher 배열의 SHA가 동결된223번 대조군과 같아야 admission을 통과한다.

기존223의 계수1 mask를 재사용한다. 세seed 최종16장 mask를 모두 동결한 후
IDs1320,1322,..1350의 원래 native64 modal GT를 읽는다.1024 steps/settle512와
RNG/readout을 유지한다. source 대비 FG≥.01/CI하한>0이고 계수1보다 높은 FG면
확대 후보이며, 두 IoU 비감소면 balanced 후보로 기록한다. paired bootstrap은
RNG221/10000회다. source-center drift와 within/adjacent-class 통계도 함께 보고하되
모든 gating의 인과 기여나 최종 FG≥.90를 mini로 주장하지 않는다.

227번 전체320장 확인을 우선한다. 각227 seed가 최소한 한 번 실제 GPU allocation을
받은 기록이 생긴 후에만228 큐가 GPU를 요청한다. 그 뒤 빈GPU0..3을 독점 lease로
사용한다. 이 allocation 조건은 완료/성능 증거가 아닌 예약 우선순위다. Linux 각
프로세스의 THP만 끄고 전역 설정은 변경하지 않는다.70,000장 재학습은 없다.
