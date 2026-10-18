# SW0222: source 주파수 중심을 유지하는 encoder denoising

221번의 unit-margin branch는 종료한다. 실제 TRAIN/DEV 모두 내부 분산이
증가했으므로 unit-distance repulsion을 제거한다. 기존 source 모델이 TRAIN
이미지에서 만든 실제 normalized frequency의 GT 물체별 평균을 고정 teacher로
쓴다. 물체별 평균을 새로 벌리지 않고 기존 평균 방향으로 내부 변동을 줄인다.
동시에 source gamma 자체와의 MSE anchor를 둬 projection 밖의 feature/graph 입력
변화도 제약한다. loss는 class-balanced frequency MSE + gamma MSE, 계수는 각 1이다.
이는 검증 전 후보이며, 더 작은 margin이나 LR/epoch sweep이 아니다.

기존 세 source seed warm192/205 각각 TRAIN64 prefix, 128 encoder update/B16,
base LR3e-5/context3e-4, clip1, fresh Adam을 유지한다. core/gates/feature 통계와
기존 1024/settle512 spike readout은 유지한다. source teacher는 TRAIN만 사용해
학습 전에 고정한다. source object 중심은 정답 label을 사용하므로 supervised다.
상속한 70k/core 및 native32 TRAIN512 supervision은 별도로 명시한다.

DEV16 IDs1320,1322,...1350, canonical128/no crop, 2px modal GT,
RNG193000+offset. 세 최종 protocol 모두 동결 후 GT를 읽는다. 최종128 step만
평가하고 DEV를 이용해 checkpoint나 가중치를 고르지 않는다. FG 평균+.01,
paired image CI 하한>0, Object IoU 비감소인 경우만 full320 확대 후보로 삼는다.
native64 전체320 세 seed FG .90 목표와 현재 최고 .848615는 그대로 유지한다.
모델 성능이나 물체 내부 개선을 학습 전 성공으로 주장하지 않는다.
