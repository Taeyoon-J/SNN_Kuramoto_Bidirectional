# SW0210: native64 기준 모델의 독립 세 seed 전체320 평가

208의 warm192 seed0 .877709/.791224/.696284만으로 세 seed 평균 승리를
주장할 수 없다. 같은 두 B4/native64 TRAIN8 적응 recipe의 독립 seed1/2,
즉 완료된205 checkpoint들을 전체320으로 평가한다. 205의 mini 개선 gate는
실패했으며 이 실험은 그 실패를 구제하거나 학습 개선으로 재분류하지 않는다.
현재 선택한 공통 baseline의 seed 편차와 절대 성능을 확인하는 평가다.
seed별로 더 좋은 checkpoint를 골라 섞지 않는다.

Seed0는208의 검증된 실제320개 warm 예측을 그대로 재사용한다.
seed1/2 각각은 canonical128 RGB/no crop/IDs1320..1639/native64/1024steps,
settle512, 기존 readout을 사용한다. RNG는 전 seed에서193000+image offset.
독립성은 학습된 모델 seed의 독립성이며 inference noise는 paired하다.
Source205 mini16과 겹치는 새 mask32개는 정확히 재현해야 한다.

추가 학습은 없다. 두 seed supervisor는 각각4개의80-image shard를 순차
실행하고 매 shard마다 shared exclusive GPU lease를 해제한다. GPU0..3에
대기하고 외부 사용자가 들어오면 같은 행별 checkpoint에서 재개한다.
긴 평가가 하나의 GPU lease를 수 시간 연속 차지하지 않도록 분할했다.

모든960개 우리 mask와 Slot 세 seed를 검증한 뒤 GT를 읽는다. 2px modal,
동률 작은ID, 같은 세 metric 및 image-paired bootstrap(10000회, seed210).
모든 세 지표의 우리3seed 평균이 Slot3seed 평균을 초과하는지와 FG평균≥.90을
별도로 기록한다. Supervised, inherited70k lineage/additional native64 TRAIN8,
validation-selected 모델이다. 새70k 학습, unsupervised 승리, scaling 또는
gating 인과 기여의 증거로 사용하지 않는다. 209 학습 모델은 섞지 않는다.
