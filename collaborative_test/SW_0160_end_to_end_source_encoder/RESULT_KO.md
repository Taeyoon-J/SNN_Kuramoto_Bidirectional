# SW160 — 실제 source encoder 공동 학습 비교

Native32, seed1, 고정320장 modal4px 평가. 같은 source97→T 기반 모델에 같은 TRAIN2048장/128updates를 추가 학습했다. 양쪽 graph/core/residual/integration은 학습하고, candidate만 실제 source RGB encoder도 학습한다. TRAIN mask를 사용한 supervised pilot이며, 새 70,000장 학습이나 데이터 증가 효과의 검증은 아니다.

| 모델 | FG-ARI | foreground IoU | object IoU |
|---|---:|---:|---:|
| 기존 sourceT | 0.755273 | 0.614993 | 0.579514 |
| encoder base 고정 control | 0.754095 | 0.607244 | 0.578872 |
| encoder base 공동 학습 candidate | 0.759042 | 0.626936 | 0.586715 |

Candidate의 실제 encoder 가중치가 바뀌었고 해당 Adam group도128 updates를 완료했다. Control의 source encoder는 불변이며 Adam updates는0이다. 전체 학습 모델과 학습된 encoder를 그대로 사용해 예측했다.

Candidate의 FG-ARI 차이는 sourceT 대비 +0.003769, paired95% CI[-0.003595,0.011391], control 대비 +0.004947, CI[-0.002277,0.012128]이다. 두 FG 신뢰구간이0을 포함하므로 사전 등록한 seed 복제 조건은 통과하지 못했다. Foreground IoU의 control 대비 차이만 CI[0.008002,0.032480]로 양수이다. .87 목표나 세 seed 평균 우위, 모든 gating의 기여를 입증한 결과로 해석하지 않는다.

이 pilot은 종료한다. LR/작은 설정 sweep과 seed0/2 복제는 추가하지 않고, SW161의 실제 spike 기반 learned assignment와 물체별 mask objective 비교로 진행한다. Checkpoint는 후속 분석용으로 보존하며 기존 leaderboard는 변경하지 않는다.

320장 예측 생성은 완료됐으나 producer가 redundant source-base hash fields2개를 protocol 최상위에 누락하여 마지막 검증이 실패했다. 이미 검증된 학습 manifest의 hash2개만 보충하고 기존 검증기로 재확인했다. Model, prediction NPZ, row ledger, 평가 기준과 고정 source code fingerprint는 그대로이며 GPU 재학습/예측은 하지 않았다. 원래 protocol·실패 queue state는 복구 archive에 보존했다. 두 모델 예측을 검증한 후에만 평가 GT를 읽었고, 기존 score validator의 독립 재계산도 통과했다.

최종 score SHA: `0b2deb211716584e231e80194fc23adae740c125a683456ff6aad7e917732036`.
증빙: `results_archive/completed_training_20261014/root_review.json`, `results_archive/completed_score_20261014/root_review.json`.
