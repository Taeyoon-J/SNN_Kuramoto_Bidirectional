# SW_0235_native64_convergence_validation64

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `603f1dd33f64d0e971972c638bece7f9ada2bd5d916131313a52653abc4c9570`

````markdown
# SW0235: 고정512회 모델의 secondary64 확인

232번 고정16장 FG .884622는 128회 대비 평균 +.024988이나 CI에0이 포함된다.
이 원래 gate=false는 보존한다. 추가 학습·loss 조정 없이 실제512회 모델을
기존 first64 validation(IDs1320–1383)에서 확인한다. 원본16장은 그대로
재사용하고 seed마다48장만 새로 추론한다. offset0 mask와 frequency를
bitwise 재현해야 한다. GPU0–3의 독점 idle lease와 기존 checkpoint를 유지한다.

비교군은 동일64장의 원래 기준과 실제128회 SW0223/SW0225다. 세 비교군의
protocol과 actual512 checkpoint를 모두 확인한 뒤 GT를 읽는다.
FG gate는 원래 기준 대비+.01 이상 및 paired image CI 하한>0,
그리고128회 모델의 FG 평균 초과다. balanced gate는 두 비교군 각각에
대해 두 IoU 평균도 낮아지지 않아야 한다. CI RNG235,10000 draws다.

기존16장 gate를 바꾸거나 secondary64를 untouched test로 부르지 않는다.
전체320장 최고는 SW0227 .858916으로 유지한다. 이번 결과만으로 FG .90
목표나70,000장 신규 학습을 주장하지 않는다. 모델 선택이나 threshold sweep은 없다.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `d629684f43b082c0f81ef53bbefc0b16e8a395500f200cdc59b4281c7f4d7970`

````markdown
# SW0235: fixed232 convergence models, secondary64 validation

No additional training. Actual accepted512 checkpoints; retain the original16
gate failure, reuse16 masks and infer48 additional images per seed. Require exact
mask and frequency replay. Compare against the same64 baseline and actual128
control. All three candidate/control protocols precede GT. This is reused
validation, not untouched test or a fresh70000-image training result.

Completed and independently audited: [results](RESULT_KO.md). Fixed64 three-seed
mean .893972/.625892/.680334; registered FG and balanced gates passed. Expand the
same checkpoint in SW0237, preserving the original16 failure and full320 incumbent.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `69e5b19d4c3571c7ee1137b503917c022c232e9d2ae40cd2a5245337a3cfd83f`

````markdown
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

````

</details>

## START_KO.md

<details>
<summary>START_KO.md</summary>

Source SHA256: `667b898a4730991b55563844bf360dbd5b64ebbb91697dc814c1c648b24b48de`

````markdown
# SW0235 시작

고정16장에서 평균 상승이 관찰된 실제 SW0232 512회 모델을 기존 first64 validation에서 확인한다. 추가 학습은 없다. 세 seed 각각 기존16장 mask 재사용,48장 새 추론,원본 mask/frequency 재현 검사1회다.

원래 모델과 실제128회 모델을 같은64장 조건에서 비교한다. 로컬8개·Linux8개 검사와 CLI2개 검사를 통과했으며, 실제 세 checkpoint CPU 복원 검증이 끝난 뒤에만 GPU0–3 독점 대기 큐를 등록하도록 했다. 현재 확인 단계와 정확한 소스 SHA는 START_EVIDENCE.json에 기록한다.

SW0232의 원래16장 FG gate=false는 보존한다. 이번64장 확인 역시 반복 사용한 validation이고, 전체320장 최고 기록은 SW0227 .858916으로 유지한다. 새70,000장 학습, 전체 .90 달성 또는 gating 기여를 주장하지 않는다.

````

</details>
