# SW0164 결과: foreground를 유지한 물체 분리

고정 native32/320장, seed1, 실제 spike Q 및 native coupling A를 사용한 frozen-source readout이다. 학습 업데이트는 0회이며 예측에 정답 mask나 물체 개수를 사용하지 않았다.

| 방법 | FG-ARI | Foreground IoU | Matched object IoU |
|---|---:|---:|---:|
| SW0164 foreground-induced partition | .770932 | .614993 | .630523 |
| 동일 rollout SourceT | .755273 | .614993 | .579514 |
| 동일 seed Slot Attention | .841819 | .203417 | .222115 |

SourceT 대비 FG-ARI +.015659, paired95%CI [.009116,.022883]. Object IoU +.051010, CI [.041976,.060664]. Foreground membership과 foreground IoU는 이미지마다 SourceT와 정확히 같았다. 기존 SW0162에서 관찰한 foreground IoU 저하 없이 물체별 분리가 개선됐다.

독립 public `validate_score()`가 저장된 예측에서 production metric 전체를 다시 계산하여 통과했다. score SHA256: `978a9d97658837eb19f414a7fb7dbb35d86c6135cb1213e12e63e2837fd65df6`. 전체 점수·paired CI·queue 종료 증거는 `results_archive/completed_score_20261014/server_records.json`에 보존한다.

등록된 seed1 확장 기준은 모두 통과했다. 다만 Slot의 FG-ARI는 여전히 더 높으며, seed1만의 결과이므로 세 seed 평균 승리나 .87 달성을 주장하지 않는다. Frozen recipe의 seed0/2 재현은 SW0165에서 별도로 수행한다. 이 결과는 추가 데이터 학습 효과나 모든 gating의 인과적 기여를 입증하지 않는다.
