# SW_0184_train_readout_alignment

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `a43b347932914ed7670617871b9ddfeaf2b902ae5135c1af14f068a166f2e79d`

````markdown
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

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `b30e47f3f49edd4de88fda0c9c28a795b348e5c150b78a7ce5a990a84e8d6f2d`

````markdown
# SW0184 — completed TRAIN-only alignment diagnostic

All three seed replays completed with zero optimizer updates and unchanged model tensors.
The evidence is mixed across seeds; no single universal cause is proven.
See [results and next direction](RESULT_KO.md). No fixed320 score or incumbent changed.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `30333b74bff069fb4939813296506b031f3773a34c282b033b950b4c5d7b8cd1`

````markdown
# SW0184 완료: TRAIN16 손실/분할 진단

고정 TRAIN16(old8+new8), frozen183 두 모델을 비교했다. 추가 학습 없이
실제 S→native Q→QCC→SW165 분할을 세 seed에서 실행했다.

| Seed | QCE 512 → 1024 | Standalone FG-ARI 512 → 1024 | Foreground IoU 512 → 1024 |
|---|---:|---:|---:|
| 0 | .244963 → .294692 | .913664 → .895067 | .684570 → .819372 |
| 1 | .408745 → .361419 | .835867 → .837792 | .611928 → .598299 |
| 2 | .365583 → .369496 | .864376 → .873369 | .526855 → .488036 |

이 숫자는 validation 성능이 아니며 .87 목표 달성으로 세지 않는다.
seed0는 QCE/FG가 함께 악화되고 seed1은 함께 개선됐다. seed2는 QCE가
조금 높아졌지만 FG는 개선됐다. 따라서 모든 seed에 공통인 단일 원인을
확정하지 않는다. 기존183의 full320 결과 및 최고 모델도 바꾸지 않는다.

같은 BG patch 쌍의 Q>=.50 비율은 candidate seed0/1/2에서 각각
.8255/.6268/.3498이다. FG same-object 쌍은 .8536/.8081/.8442이다.
이는 특히 seed2의 배경 연결이 약한 현상과 일치하지만, 모든 쌍의 평균을
공간 graph의 connectivity나 오류의 인과 증명으로 취급하지 않는다.
BG 비율은 기록된 all-class/FG-class balanced 평균과 class 수로 역산했고
모든16행에서 정확히 BG 한 class 차이가 있음을 검사했다.

다음 모델 변경은 QCE를 유지하면서 TRAIN mask의 공간 이웃 same-class
연결과 cross-class 경계가 실제 QCC .50 기준 양쪽에 놓이도록 보조 loss를
주는 것이다. 양의 연결은 per-component signed rho에 직접 credit을 주고,
경계 억제는 production Q를 사용한다. 이전 forest-loss 단독 실패와 구분해
새 실험으로 등록하고, 실제 분할이 좋아질 때만 확대한다.

원본 보고서의 no-update model tensor hash, cohort/producer SHA와 산술은
검증했다. 이 사후 검사는 두 번째 mask 추론이나 독립 metric 재계산은
아니다. scope는16장에 한정한다.

원본과 집계: [result_v1](results_archive/result_v1/).

````

</details>
