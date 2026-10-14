# SW159 결과: 임계값 초기화 변경은 성능을 낮춤

seed1, native32, 동일한 2,048장 TRAIN·128 updates의 supervised continuation을 고정 320장으로 평가했다. 실제 spike QCC와 기존 평가 계약을 유지했다.

| 모델 | FG-ARI | foreground IoU | matched-object IoU |
|---|---:|---:|---:|
| sourceT | 0.755273 | 0.614993 | 0.579514 |
| 기존 초기화 비교군 | 0.754049 | 0.609159 | 0.580244 |
| source-calibrated 임계값 초기화 | 0.668768 | 0.460783 | 0.541587 |

후보−sourceT FG 차이는 −0.086506, paired-image 95% CI는 [−0.101517, −0.071847]이다. 후보−비교군 차이는 −0.085281, CI는 [−0.100557, −0.070390]이다. 후보는 세 지표 모두 sourceT와 비교군보다 낮다.

사전 검증에서는 항상 켜져 있던 event가 시간에 따라 켜졌다 꺼졌고 membrane과 기존 b가 실제로 업데이트됐다. 그러나 그 변화가 물체 구분 개선으로 이어지지는 않았다. b는 event cutoff와 reset을 동시에 바꾸므로, 이 결과만으로 어느 효과가 원인인지 분리할 수 없다.

이 recipe는 종료한다. 임계값/LR sweep과 seed0/2 확대는 하지 않으며 기존 최고 모델을 유지한다. 다음 SW160은 기존 sourceT 초기화에서 실제 source encoder를 공동 학습하는 별도 비교다. 이번 결과는 fresh70k 학습·data scaling·unsupervised 성능 또는 gating 기여의 증명이 아니다.

검증된 최종 score SHA: `abbc35a80124f8898e3e444a4dd38f7c5c9ca01e2454c4c6163a04f4dd7b95b6`. 실제 scoring queue의 독립 재계산과 로컬 해시/fingerprint 검증을 통과했다. 증거는 `results_archive/completed_score_20261014/`에 보관했다.
