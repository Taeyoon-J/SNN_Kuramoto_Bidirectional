# SW0212: 학습 전경 결합도 개선 실패

추가 학습 없이211 primary128 head를 사용했다. CPU에서 원래 전경을 사용한
모든16 예측이211 GPU endpoint와 정확히 같음을 확인하고, 새 mask를 모두
고정한 뒤 GT를 읽었다. Head SHA,48개 mask의 독립 metric과 paired CI를
재검증했다. seed2·고정16·native64의 supervised readout 비교다.

| 설정 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 원래 warm205 | .788414 | .445474 | .674335 |
| 211 원래 전경+center head | .034678 | .445474 | .044414 |
| 212 학습 전경+같은 head | .000000 | .292532 | .022362 |

추가 전경 판별도 세 지표를 개선하지 못했다. 모든 이미지에서 center
fallback과 예측1개 상태가 유지됐고 전경 변경 비율은39.68%였다.
같은 head의 TRAIN 진단에서도 center 검출이 작동하지 않았으므로 이 결과로
학습 전경과 비국소 assignment의 일반적인 가능성을 부정하지 않는다.
이128-step checkpoint에 대한 고정 .5 전경 readout의 확대는 중단하며
threshold를 validation에 맞춰 바꾸지 않는다.

새 최고 결과나 전체320 성능 개선은 없다. 기존 기준점은 유지한다.
증빙: results_archive/completed_v1/score_v1/summary.json,
predictions/protocol.json, completion_audit.json.
