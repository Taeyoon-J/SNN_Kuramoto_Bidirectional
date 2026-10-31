# SW_0229_native64_partition_diagnosis

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `d41cd02c2a3cccdce143e2985af7dde7449670aedaba90f5cdc0960ef16cd78e`

````markdown
# SW0229: 동결된225 mask의 오류 분해, CPU 진단만 수행

GPU를 추가 사용하지 않고 이미 검증된225의64장·세seed 원래/공동학습 mask를 읽는다.
모든 prediction/protocol/checkpoint와225의 독립 검증을 확인한 후 같은 native64 GT를
읽으며, 세 지표를 다시 계산해225 집계와1e-12 이내로 같아야 분석한다. 모델·loss·
데이터·mask를 바꾸지 않고227/228 실행에도 관여하지 않는다.

FG-ARI가 보는 GT전경 patch 쌍에서 서로 다른 GT가 같은 예측 label에 들어간 경우를
false merge, 같은 GT가 다른 예측 label에 들어간 경우를 false split으로 센다.
merge는 예측0과 양수label 내부로 나눈다. 여기서 예측0 merge는 실제 배경 pixel의
오류가 아니라 여러 실제 물체가 예측 배경0에 합쳐진 경우다. split은 한쪽만 예측
배경0인 support 경계와, 양쪽 모두 양수지만 다른 foreground label인 경우로 나눈다.
전경 precision/recall과 GT물체가 통째로 예측0에 들어간 비율도 별도 보고한다.
쌍의 합계 비율과 이미지별 평균을 모두 기록하며 단일 원인이나 causal gate로 해석하지 않는다.

GT 각 물체의4-connected 가시 조각 수와 작은 조각의 patch질량을 계산한다.
예측 양수 cluster의4-connectivity도 실제 확인한다. 이는 topology 설명이며 모델의
FG-ARI 성능이나 가능한 ARI의 상한을 생성하지 않는다. source readout의 연결 제한이
오류를 얼마나 설명하는지 판단할 자료이며 oracle 예측을 배포하지 않는다.

분석용 이미지별 수치는 private_rows.json에만 두고 공유할 결과는 집계와 코드다.
이 CPU 진단을 새로운 모델 결과·전체320장 점수·70,000장 scaling·모든 gating의
의미 증명으로 주장하지 않는다. 기존 전체320장 incumbent는 유지한다.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `155ca6139d6590078979333479275faa62c6a004b6be7463ac49fac5f67a3560`

````markdown
# SW0229 COMPLETE

CPU-only diagnosis of frozen SW0225 masks. Splits decreased, merges slightly increased;
extra foreground/background and connectivity accounting is in RESULT_KO.md.
No new model score, GPU forward or learning; full320 incumbent unchanged.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `9460755c6cef723f66e9d5724ba9dd7f21627a6a8d0287a955ac5ce3b3681360`

````markdown
# SW0229 완료: 현재 개선과 남은 mask 오류

이미 검증된225의64장·세seed mask를 CPU로만 분석했다. 추가 GPU forward, 학습,
mask·데이터 변경은0회다. 원래225 지표와의 최대 차이는 약1.11e-16이고384개 분석 row의
정수 분해/집계/비율을 로컬에서도 확인했다. 이것은 새 모델 성능 결과가 아니다.

| GT전경 patch 쌍의 이미지별 평균 오류율 | 원래 모델 | 공동학습223/225 |
|---|---:|---:|
| 같은 물체가 다른 label로 쪼개짐 | 8.32% | **6.10%** |
| 서로 다른 물체가 같은 label로 합쳐짐 | **4.27%** | 4.62% |

두 오류율은 각각 GT같은물체/다른물체 pair 수로 나눈 값으로, 같은 분모의 양이나
FG-ARI 기여도라고 직접 비교할 수 없다. 전체 pair 질량을 합친 비율에서도 split은
8.46%→5.98%로 줄고 merge는4.29%→4.39%로 소폭 증가했다. 입력 공동학습의 개선은
덜 쪼개지는 현상과 동반되지만, 합쳐짐까지 함께 해결한 것은 아니다. 어느 gate의
인과적 기여나 유일한 원인을 이 통계만으로 입증하지 않는다.

공동학습의 전경 precision은 이미지별 평균62.6%, recall은94.8%다. GT전경은 대부분
포함하지만 실제 배경 patch도 전경으로 많이 남는다. pair 질량 기준 잘못된 merge의
약10.9%는 여러 실제 물체가 예측배경0에 들어간 경우이며, 나머지는 양수label 내부의
merge다. split 중 약40.1%는 예측배경0/양수label 경계, 나머지는 foreground label
사이에서 발생했다. 이 비중의 증가를 support 오류 자체의 증가로 해석하지 않는다.

예측 양수군집은 평균14.65개→12.95개, GT modal-patch 유효물체는 평균6.20개다.
양수군집에는 false-positive 배경도 들어가므로 이 비교를 학습된 object-count
classifier의 정확도라고 주장하지 않는다. 여전히 과도한 군집이 남는다는 진단이다.

이64장의 GT물체397개 중4-connected 가시영역이 둘 이상인 것은2개(0.50%)였고,
가장 큰 조각 외 foreground patch는3개/51194개(0.0059%)뿐이었다. 예측 양수군집은
원래/공동학습 모두 실제로4-connected였다. 이 cohort에서는 가려짐으로 인한
분리 조각을 합치지 못하는 제약만으로 현재 오류를 설명하기 어렵다. 다른 데이터나
전체320장으로 일반화하지 않으며, ARI의 수학적 상한을 계산한 것도 아니다.

따라서 원인 확인 없이 nonlocal/topology 변경을 우선하지 않는다. 현재 예약된228번의
source-center 보존 비교로 응집과 분리의 tradeoff를 먼저 확인한다. 이 가설이 틀리면
해당 결과를 기록하고 다른 구조/읽기 방식으로 전환한다.227번 전체320장 평가는 유지하며
이 진단으로 incumbent를 바꾸거나 FG≥.90 목표를 달성했다고 판정하지 않는다.

근거: results_archive/completed_v1/summary.json과 archive_receipt.json.
이미지별 분석 row와 원본 mask/GT는 공개하지 않는다.

````

</details>
