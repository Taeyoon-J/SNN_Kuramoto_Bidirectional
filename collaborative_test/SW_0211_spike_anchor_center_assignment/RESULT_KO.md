# SW0211: center head pilot 실패, 기존 기준점 유지

고정16·seed2·native64 결과다. 실제 head Adam128, source checkpoint 불변,
TRAIN-only normalization,32/64/128 snapshot과96개 mask endpoint를 독립
검증했다. Core/encoder/gates를 이 실험에서 학습하지 않았다.

| 설정 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 기존 warm205 | .788414 | .445474 | .674335 |
| center head32 | .034678 | .445474 | .044414 |
| center head64 | .034678 | .445474 | .044414 |
| primary128 | .034678 | .445474 | .044414 |

모든DEV 이미지에서 center가 검출되지 않아 단일 center fallback을 사용했다.
평균 정답 물체5.375개에 예측1개로 collapse했다. 전경은 원래 mask와 정확히
같으므로 Foreground IoU가 유지된 것이 grouping 성공을 의미하지 않는다.
Mean-input/moments-only 비교도 대부분 같은 fallback이라 최종 mask 점수로
입력 신호의 유용성이나 gating 기여를 판단할 수 없다. 확대 gate는 실패했다.

별도 CPU 진단에서 TRAIN32도 모두 fallback이었다. 정답 center가 고정
확률 기준 .3에 도달한 비율은0이고, 정답 center 확률 평균은 .06354였다.
즉 validation 성능만 나쁜 경우가 아니라 학습 데이터에서 center 검출기
자체가 아직 작동하지 않은 상태다. Head128의 full TRAIN loss는9.7335에서
4.2920으로 줄었고 모델·Adam은 실제 업데이트됐다.

한편 object-balanced offset 오차는 무변위4.2434 patch 대비 실제 입력에서
1.5185였다. 같은 frozen head에 TRAIN-mean 입력을 넣으면4.7278,
moments-only 입력은3.6219였다. 이 값은 TRAIN 진단이며 validation 성능이나
전체 물리적 gating의 인과 기여가 아니다. Offset 학습은 있지만 sparse
center 검출이 endpoint를 막는다는 후속 검증 근거로 사용한다.

211의 고정128 판정을 바꾸거나 threshold/LR/width를 sweep하지 않는다.
후속 실험에서는 동일 head/Adam을 이어 학습하는 단일 예산 확장을 별도로
등록해 TRAIN center 검출이 살아나는지와 DEV 성능이 함께 개선되는지 본다.
추가 core forward가 필요 없는 cached-feature 학습이다. 실패가 계속되면
학습량을 반복해서 늘리지 않고 readout 방향을 전환한다.

기존 전체320·세 seed 기준점 .848615/.592533/.672047은 그대로 유지한다.
증빙: completed_v1/completion_audit.json, score_v1/summary.json,
head_training/report.json 및 head_diagnosis_v1/summary.json.
마지막 진단의 source_model_sha256은 frozen classifier head SHA를 가리킨다.
