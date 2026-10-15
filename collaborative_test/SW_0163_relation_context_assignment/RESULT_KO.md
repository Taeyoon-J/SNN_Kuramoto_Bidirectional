# SW0163 최종 결과: classifier 확장 중단

동일한 native32·fixed320·seed1 평가다. 네 모델 모두 같은 frozen source의 실제 spike와 같은 학습 순서로 512번 업데이트했다. 평가 복구에서는 추가 학습하지 않았다. 최종 예측 다섯 묶음의 public validator와 score 재계산을 통과한 뒤 두 큐가 정상 완료됐다.

| 모델 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| raw_only | 0.624734 | 0.457351 | 0.400735 |
| Q_only | 0.634939 | 0.489591 | 0.410119 |
| A_only | 0.663123 | 0.466450 | 0.434423 |
| Q_A | 0.660608 | 0.498060 | 0.432283 |
| 동일 rollout SourceT control | 0.755273 | 0.614993 | 0.579514 |
| 비교 Slot Attention seed1 | 0.841819 | 0.203417 | 0.222115 |

Q+A는 raw 대비 세 지표 모두 개선했다. FG-ARI 차이는 +0.035874, image-paired 95% CI는 [+0.017619, +0.054757]이다. A_only의 FG 개선도 +0.038389, CI [+0.019800, +0.057119]다. 이는 이 supervised classifier에서 관계 입력이 도움이 된다는 제한된 근거이며, 각 물리적 gating의 인과적 기여나 전체 모델 개선의 증명은 아니다.

Q+A는 SourceT보다 FG-ARI가 0.094665 낮고, 두 IoU도 낮다. 등록된 확장 gate는 실패했다. 이 classifier recipe의 seed0/2 복제 및 대규모 학습은 진행하지 않는다. 별도로 이미 검증된 SW165 seed1의 0.770932/0.614993/0.630523과 세 seed 평균 최고 모델을 유지한다.

다음 후보는 SW165의 foreground와 연결 partition을 유지하면서 split·merge 결정을 직접 학습하는 SW167이다. CPU/GPU Q 재계산 차이 때문에 신규 pair-feature 경로의 실제 backend 검증이 아직 필요하다. 그 검증을 통과하기 전에는 학습을 시작하지 않는다.

기존 hash-family 오류와 실패 기록을 보존했다. 복구 결과의 SHA-256은 `f189d83190e5134b00a76fb81a72202456e437d41af76233b1159c1d5575468a`다. 최종 report·두 큐 상태·실제 프로세스 확인은 `results_archive/source_lineage_prediction_frozen_20261015/public_validation_complete.json`에 있다. 이 결과는 supervised seed1 pilot이며 unsupervised 결과나 세 seed 평균으로 표기하지 않는다.
