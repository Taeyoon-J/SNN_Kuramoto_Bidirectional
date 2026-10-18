# SW0225 완료: 고정 모델의 64장 확대 평가

223번 encoder/core 입력 projection 공동학습 checkpoint를 고정하고 canonical128,
native64, IDs1320..1383에서 세 seed를 평가했다. 추가 학습은 0회다.

| 모델 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 같은 이미지의 원래 모델 | .864270 | .571015 | .669769 |
| 223번 공동학습 모델 | **.877164** | **.597239** | **.670309** |

FG 차이 +.012894의 paired95% CI는 [.004403,.022515], FIoU 차이 +.026224는
[.002066,.051886]이다. Object IoU 차이 +.000540의 CI[-.013955,.015579]는 0을
포함하므로 그 지표의 개선은 불확실하다. 세 평균은 모두 비감소이고, 사전 정의된
FG 및 balanced full320 후보 기준을 통과했다. 이는 고정된 세 모델에 대한 이미지
bootstrap이며 seed 모집단의 불확실성 구간은 아니다.

| seed | 원래 FG-ARI | 공동학습 FG-ARI | 공동학습 FIoU | 공동학습 Object IoU |
|---|---:|---:|---:|---:|
| 0 | .911894 | .911601 | .807920 | .694423 |
| 1 | .841229 | .860112 | .521367 | .666115 |
| 2 | .839686 | .859778 | .462432 | .650389 |

FG 개선은 주로 seed1/2에서 나타났다. seed0은 거의 같고 Object IoU는 소폭 낮아졌다.
이것만으로 모든 gating의 인과적 기여나 데이터 scaling을 입증하지 않는다.

원래 16장 gate 실패 기록은 유지한다. 이번 별도64 확인의 범위와 기준은 GT를 읽기
전에 정했으며, 기존 평가에서 좋았던 모델을 더 넓게 확인한 exploratory validation이다.
반복 사용한 validation이므로 untouched test라고 주장하지 않는다.

세 frozen protocol 및 실제 checkpoint SHA를 모두 확인한 후 GT를 읽었다.
candidate/baseline 총384개 mask를 독립 재계산했고 최대 지표 차이는 2.22e-16,
로컬 평균·CI 재계산 차이는 0이었다. 16개 원래 mask/seed의 재사용과 bitwise replay도
검증했다. seed2의 THP 복구 이력·원래 실패 상태·추가 replay는 보존했다.

**64장 평균 .8772를 전체320장 최고 .8486과 직접 비교해 최고 기록을 갱신하지 않는다.**
전체320장·세 seed의 FG .90 목표는 아직 검증되지 않았다. 같은 checkpoint의 나머지
256장/seed를 확인할 후보로 선정했다. 70,000장 재학습은 이번 평가에 포함하지 않았다.
226번은 별도로 같은 업데이트 예산에서 고유 TRAIN64→512 효과를 평가 중이다.

근거: results_archive/completed_v1/summary.json, completion_audit.json,
archive_receipt.json 및 best_model_tracking_20261008.json의 별도 secondary64 항목.
