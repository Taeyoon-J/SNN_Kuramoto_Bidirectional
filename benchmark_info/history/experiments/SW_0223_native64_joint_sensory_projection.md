# SW_0223_native64_joint_sensory_projection

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `30bc7ad5ff8afd6d8681d2383906c39e012cde2c68bba954dc68cd193fc40d61`

````markdown
# SW0223: encoder와 실제 core 감각 projection 공동 학습

222번은 내부 주파수 분산을 줄였지만 FG 평균 +.0114의 CI가0을 포함했고
Object IoU가 하락했다. 이번에는 동일 source/teacher/TRAIN64/128 updates/B16/
encoder LR/목표함수를 유지하고 실제 gamma_channel_proj.weight(4x8,32개)를
함께 학습한다. projection LR은 기존 context LR3e-4로 고정한다. omega,
frequency gain, phase gain, 나머지 graph/gate/core 파라미터는 모두 고정한다.
추가 projection을 끼우거나 채널 순열을 DEV에서 고르지 않는다.

고정 teacher는 원래 source로 먼저 계산한다. gamma anchor와 자연 물체별
주파수 중심 목표는 그대로다. 전체 학습 파라미터에 global norm clip1을 쓴다.
gradient와 실제 projection 변경을 기록하고, 나머지 core 상태의 바이트 SHA가
유지되는지 확인한다. 작은 projection+encoder만 매 update checkpoint에 저장한다.
이는 smooth 감각 입력 pretraining이며 hard spike/gate loss의 end-to-end 학습이나
각 gate의 인과 기여 증명으로 취급하지 않는다.

seed0/1/2 공통 recipe. 고정 DEV16/IDs1320,1322,...1350/canonical128/no crop/
2px modal GT/RNG193000+offset/1024 settle512 기존 spike readout을 유지한다.
세 candidate protocol 및 기존222 control mask를 모두 검증한 뒤 GT를 읽는다.
222 encoder-only와 기존 source를 함께 비교하며 control protocol SHA를 CPU
admission에서 미리 고정한다. 개별 seed를 다른 recipe로 골라 평균내지 않는다.

사용자의 FG 우선순위와 SELECTION_POLICY의 FG/balanced track을 구별한다.
FG track full320 확대 후보 조건은 source 대비 FG+.01/paired image CI하한>0,
그리고 FG 평균이222보다 높음이다. IoU tradeoff를 반드시 같이 기록한다.
Balanced gate는 여기에 source 대비 두 IoU 비감소를 요구한다. 이는 사전 후보
선정 규칙이며 최종 성공은 아니다. 최종 full320 세 seed 평균은 동일 Slot 계약의
세 지표를 모두 엄격히 넘고 최신 사용자 목표인 FG .90도 검증해야 한다.
221/222의 기존 gate 실패는 변경하지 않는다. 새70k 학습은 하지 않는다.

````

</details>

## PEER_REVIEW_KO.md

<details>
<summary>PEER_REVIEW_KO.md</summary>

Source SHA256: `6acd5f6e84bac19cf052185c7f26aa75bd1a4d0c9f002b5609c6b71f84fc8174`

````markdown
# 동료 연구 확인 — 2026-10-18

223번 시작 전 origin/patch_v2를 다시 fetch했다. HEAD는
7a21632b93805f4f1da69a2e77a9ddd95c624d0a로 변동이 없다.
새로운 검증 완료 결과를 발견하지 못했으므로 기존 PV2_0102 검토를 유지한다.
이번 실험의 직접 근거는222의 세 seed 입력 분산 감소와 불확실한 FG 평균 상승이다.
단일 seed .9252나 MINI 평균 .8543을 full320 최고 성적으로 승격하지 않는다.
기존 source와222 encoder-only 모델을 동시에 고정 control로 사용한다.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `84c36fe2b5d228f41189db6dc45df2f59b315948b16eba02fd124510d59a3b5e`

````markdown
# SW0223

Prospective native64 joint encoder and actual32-weight sensory projection test.
Same frozen source object-frequency teacher, gamma anchor, TRAIN64,128 updates
and production spike readout as222. All remaining core state stays frozen.
Three independent seeds; compare against both source and frozen222 controls.
See DESIGN_KO.md. No performance claim before execution and final scoring.

Actual start2026-10-18:9 local/9 Linux tests,3 CLI checks and actual three-seed teacher/control/32-weight projection-gradient CPU admission passed without source updates. Three exclusive idle GPU queues registered. START_EVIDENCE.json binds actual NVIDIA workers, fixed222 control protocols, and positive projection gradients. Encoder and actual core input projection learn; every other core state remains frozen. No new segmentation result; full320 incumbent .848615/.592533/.672047 unchanged.

Completed: fixed16 three-seed source .842891/.600692/.702161; encoder-only222 .854272/.631545/.695736; joint223 .859634/.645611/.706340. All mean metrics rise, but source FG CI[-.000361,.040940] includes0; original expansion gate False. Actual learned32 projection, remaining core invariance and144 masks/independent metrics/CI audited. Separate fixed-model64 validation225 will check stability; source16 gate and full320 incumbent unchanged.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `bcb18a976c7f90a03daea47e1a111cf350b4315406e0680cf6f23f601367422f`

````markdown
# SW0223: 세 지표 평균 개선, source 대비FG 확신은 아직 부족

기존 source/teacher/TRAIN64/128 updates/B16/gamma anchor/encoder LR을 유지하고
실제 gamma_channel_proj.weight32개만 core에서 추가 학습했다. omega/frequency gain/
phase gain/graph 및 나머지 gate 상태는 고정했다. 원래64x64/1024 settle512
readout, 동일 DEV16과2px modal GT를 사용했다. 세 candidate 및222 control
protocol을 모두 동결·검증한 뒤 GT를 읽었다.

| 세 seed 평균 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 기존 source | .842891 | .600692 | .702161 |
| 222 encoder-only | .854272 | .631545 | .695736 |
| 223 공동 학습 | .859634 | .645611 | .706340 |

source 대비 차이는 +.016743/+.044919/+.004179다. FG 차이의 paired image95% CI
[-.000361,.040940]가0을 포함하므로 사전 FG/balanced full320 gate는 모두 실패했다.
Foreground IoU CI[.003369,.092662]는0보다 높다.222 대비 FG 차이+.005362,
CI[.000062,.012050]로 이 고정16장·세 모델에서는 공동 학습의 FG 평균이 높았다.
이 CI는 고정 세 모델에 대한 image bootstrap이며 seed 모집단 또는 모델 선택을
보정한 불확실성은 아니다. 확정적인 일반화나 .90 달성으로 승격하지 않는다.

실제 projection/encoder Adam128 및 첫 projection gradient 양수, projection SHA
변경, 나머지 core SHA 보존을 확인했다. source/222/223의144개 raw mask를
독립 metric으로 재평가했고 평균·CI 오차 최대2.22e-16, 로컬 평균/CI 차이는0이다.
DEV 내부 주파수 분산은 세 seed 모두222보다 추가로 줄었다. 이 입력 통계가
recurrent rhythm이나 각 gate의 인과 기여를 증명하는 것은 아니다.

다음 확인은223 checkpoint를 고정하고 평가만64장으로 늘리는 별도225번이다.
223의16장 gate 실패는 그대로 유지하며, source 코드를 수정하거나 더 학습하지 않는다.
정해진64장 validation을 한 번만 확인하고 성능이 불확실하면 자동으로70k나
full320 확대를 하지 않는다. untouched test 또는 전체 학습 데이터 scaling 시험이 아니다.

native64 전체320 최고 세 seed 평균 .848615/.592533/.672047은 유지한다.
증거: results_archive/completed_v1/{summary,completion_audit,archive_receipt}.json.
이 모델도 상속한70k/core 및 native32 TRAIN512 supervision을 포함하는 supervised
계열이다. 새 native64 70k 학습은 수행하지 않았다.

````

</details>
