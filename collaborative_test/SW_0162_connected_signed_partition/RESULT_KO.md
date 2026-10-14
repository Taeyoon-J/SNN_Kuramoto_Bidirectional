# SW0162: 연결된 signed partition 비교 결과

32×32 patch, seed1, 고정 평가 이미지 320장. 학습이나 정답 mask를 사용하지 않고, 기존 SourceT의 실제 spike 동기화 Q와 실제 native coupling A로 patch 묶음 전체의 관계를 평가하는 inference/readout 실험이다.

| 방식 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| SW0162 connected signed | 0.775385 | 0.598386 | 0.626825 |
| 기존 SourceT / 동일 rollout control | 0.755273 | 0.614993 | 0.579514 |
| 동일 조건 Slot Attention seed1 | 0.841819 | 0.203417 | 0.222115 |

기존 SourceT 대비 FG-ARI는 +0.020112 (paired 95% CI [0.012060, 0.028894]), Object IoU는 +0.047311 [0.037916, 0.056818] 개선됐다. Foreground IoU는 −0.016608 [−0.025845, −0.007694] 하락했다. Slot Attention보다 FG-ARI가 낮으며 0.87 목표에 도달하지 않았다.

등록된 확장 기준은 두 IoU를 유지하면서 FG-ARI가 유의하게 개선되는 것이므로 통과하지 않았다. 이 고정 recipe의 seed0/2 추가 실행 및 계수 탐색은 진행하지 않는다. 관계를 이용한 물체 구분의 개선 신호는 다음 SW0163의 학습된 assignment 비교를 위한 근거로 보존하되, 모든 gating의 기여나 데이터 증가 효과가 증명됐다고 해석하지 않는다.

서버의 `validate_score()`가 예측 산출물과 reference hash를 검증하고 실제 평가 함수를 통해 모든 지표를 독립 재계산했다. 동일 rollout의 기존 control은 보관된 SourceT 320개 mask와 정확히 일치했다. 평가 원본 SHA256: `723bf11a5fbf499f868d9fa7877be6bb2e5e1156069abc5db62af6b039ceaa4f`. 원본 및 완료된 큐 기록은 `results_archive/completed_score_20261014/server_records.json`에 보관했다. seed1 결과이며 세 seed 평균이나 새로운 70,000장 학습 결과가 아니다.
