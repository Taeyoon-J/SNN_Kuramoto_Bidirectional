# SW0197: native64 기준 모델의 seed1/2 병렬 확인

모든 GPU를 대상으로 독립 테스트를 예약하라는 사용자 지시에 따라, 현재 비어 있는
GPU 하나를 먼저 확보하는 작업 두 개를 각각 seed1과2에 예약한다. GPU lease는
기존194/195/196과 공유하여 다른 사용자 또는 우리 작업과 중복 점유하지 않는다.
seed1/2 작업 사이에는 과학적 선행 의존성이 없으므로 동시에 실행할 수 있다.

SW0186의 각 seed별 실제 전체 모델과 학습된 CNN을 SW0192의 엄격한32→64 매핑으로
읽는다. SourceT squared contrast8/chunk1, graph top-k512, T1024/settle512 및
production spike/Q mask readout을 그대로 쓴다. 새 optimizer update는0회이다.
이는 추가 TRAIN8/64 학습이 아니라 그 이전의 native64 baseline 확인이다.

SW0193과 같은 고정32장 development 이미지1320..1351, RNG193000+offset을
사용한다. seed0 baseline과 Slot3mean은 이미 검증된193번 frozen reference를
재사용한다. 두 seed의 예측을 고정하고 모델·RGB·배열 해시를 확인한 후 modal
GT 및 독립 수식으로 세 지표를 계산한다. imagewise 점수는 비공개이다.

완료 시 세 seed 평균을 낸다.32장 development 결과로 전체320장 또는 최종
FG-ARI>=.90 목표를 달성했다고 표시하지 않는다. 기존 최고 모델도 자동 교체하지
않는다. 이 결과는 native64에서 seed별 전이 상태와 다음 학습 비교의 기준점이다.

194번 학습이나195번의 필수194완료 의존성을 바꾸지 않는다.196번 계산 검증은
독립적으로 ready queue로 전환한다. 모든 작업은 GPU0,1,2,3 중 먼저 완전히 비는
GPU에서 실행하며, 외부 사용자가 합류하면 우리 worker만 양보하고 저장된 행부터
재개한다. 각 GPU에 같은 seed를 복제하여 평가 표본을 늘리지 않는다.
