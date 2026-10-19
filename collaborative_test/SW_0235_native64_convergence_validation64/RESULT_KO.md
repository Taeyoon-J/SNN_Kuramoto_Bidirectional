# SW0235: 추가 학습의 개선이 확장64장에서도 유지됨

추가 학습 없이 실제 SW0232 누적512회 모델을 기존 first64 validation에서
확인했다. 고유 추가 TRAIN64장, native64, seed0/1/2 평균이다. 원래16장 gate
실패는 보존하며, 이64장을 untouched test라고 부르지 않는다.

| 같은64장 평가 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 원래 기준 | .864270 | .571015 | .669769 |
| 실제128회 / SW0223 | .877164 | .597239 | .670309 |
| 실제512회 / SW0232 | **.893972** | **.625892** | **.680334** |

128회 대비 차이는 FG +.016808, paired image95% CI[.006572,.028011],
Foreground IoU +.028652, CI[.009232,.050311],
Object IoU +.010025, CI[.000594,.020334]이다. 이 고정 cohort에서는 세 지표
개선이 모두 지지된다. 원래 기준 대비 FG도 +.029702,
CI[.015205,.046431]이며 등록한 FG/balanced 후보 gate를 모두 통과했다.
CI는 세 고정 모델에 조건부인 이미지 bootstrap이며 seed 모집단 CI가 아니다.

같은 checkpoint의 원래16장 prediction을 보존하고 seed별48장을 추가로
추론했다. offset0 mask와 frequency를 bitwise 재현했다. 세 source/checkpoint와
prediction protocol을 모두 검증한 뒤 DEV GT를 읽고, 세 비교군의 실제576개
mask를 독립 재채점했다. 원시 점수·평균·CI와 로컬 재계산 오차는1e-12 미만이다.
추가 optimizer update는0이며 모든 gating의 인과 기여를 증명한 결과는 아니다.

다음 SW0237에서 같은 세 checkpoint를 전체320장으로 확인한다. 검증된64장은
재사용하고 나머지256장과 exact replay만 추론한다. 새 학습이나 checkpoint 선택은
하지 않는다. 현재 전체320장 최고 .858916/.605286/.674313을 이64장 점수로
교체하지 않으며, 전체320장 FG≥.90이나 신규70,000장 학습을 달성했다고 주장하지 않는다.

근거: `results_archive/completed_v1/{summary.json,completion_audit.json,archive_receipt.json}`.
원본 데이터·checkpoint·이미지별 벡터는 공개하지 않는다.
