# SW0184: TRAIN-only QCE / actual QCC alignment diagnostic

183번은 unique-data FG gate를 통과하지 못했다. 실패 recipe를 연장하지 않고,
동일 frozen control512/candidate1024에서 상대적인 QCE와 실제 threshold .50
QCC/foreground-induced merge 출력의 일치 여부를 비교한다.

각 seed0/1/2의 ordered TRAIN1024에서 old positions 0,64,...448 및
new positions 512,576,...960을 사용한다. 총16장, 두 모델, 세 seed로
96 image forwards, B4, T1024/settle512, 같은 batch RNG를 사용한다.
추가 optimizer update, validation/reserve 이미지/정답 접근은 없다.
모델의 모든 tensor hash는 추론 전후 같아야 한다. 기존183 파일은 불변이다.

각 이미지에 실제 native QCE/true-vs-best class margin, Q>=.50 same-object
recall/cross-object rate(BG 포함/제외 구분), per-component rho 통계, 실제
spike QCC 및 SW165 partition의 세 지표를 기록한다. Conditional은 control의
QCC foreground를 고정하고 standalone은 각 모델 자신의 QCC를 사용한다.
정답은 TRAIN 진단 지표에만 사용하며 mask/readout 입력에는 넣지 않는다.

QCE가 감소하고 same-object threshold recall 혹은 실제 FG가 악화되면
loss/readout mismatch의 작은-cohort 증거다. QCE 자체도 악화되면 고정 probe
전역 목적/optimization 문제가 남아 있다는 증거다. 전경 고정 시와 자체
전경 시의 차이도 따로 보고한다. 동률/상반된 seed는 그대로 공개한다.
이 작은 cohort의 평균을 전체 TRAIN/70k/validation 개선으로 일반화하지 않는다.
validation threshold 검색, 후보 승격, 본 실험만으로 gate 인과 주장도 하지 않는다.
한 번의 진단 후 관측된 병목에 맞는 새 loss/학습 방식을 별도 등록한다.

GPU는 기존 owner lease를 사용하고 타 사용자가 있으면 해당 장치를 피한다.
