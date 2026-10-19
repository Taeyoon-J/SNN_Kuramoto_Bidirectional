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
