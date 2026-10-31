# SW_0222_native64_frequency_teacher_denoising

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `a9e09ed4424d635a90939ccaef2cd2552d3d32ec3e0a6830264af19a16f06ef2`

````markdown
# SW0222: source 주파수 중심을 유지하는 encoder denoising

221번의 unit-margin branch는 종료한다. 실제 TRAIN/DEV 모두 내부 분산이
증가했으므로 unit-distance repulsion을 제거한다. 기존 source 모델이 TRAIN
이미지에서 만든 실제 normalized frequency의 GT 물체별 평균을 고정 teacher로
쓴다. 물체별 평균을 새로 벌리지 않고 기존 평균 방향으로 내부 변동을 줄인다.
동시에 source gamma 자체와의 MSE anchor를 둬 projection 밖의 feature/graph 입력
변화도 제약한다. loss는 class-balanced frequency MSE + gamma MSE, 계수는 각 1이다.
이는 검증 전 후보이며, 더 작은 margin이나 LR/epoch sweep이 아니다.

기존 세 source seed warm192/205 각각 TRAIN64 prefix, 128 encoder update/B16,
base LR3e-5/context3e-4, clip1, fresh Adam을 유지한다. core/gates/feature 통계와
기존 1024/settle512 spike readout은 유지한다. source teacher는 TRAIN만 사용해
학습 전에 고정한다. source object 중심은 정답 label을 사용하므로 supervised다.
상속한 70k/core 및 native32 TRAIN512 supervision은 별도로 명시한다.

DEV16 IDs1320,1322,...1350, canonical128/no crop, 2px modal GT,
RNG193000+offset. 세 최종 protocol 모두 동결 후 GT를 읽는다. 최종128 step만
평가하고 DEV를 이용해 checkpoint나 가중치를 고르지 않는다. FG 평균+.01,
paired image CI 하한>0, Object IoU 비감소인 경우만 full320 확대 후보로 삼는다.
native64 전체320 세 seed FG .90 목표와 현재 최고 .848615는 그대로 유지한다.
모델 성능이나 물체 내부 개선을 학습 전 성공으로 주장하지 않는다.

````

</details>

## PEER_REVIEW_KO.md

<details>
<summary>PEER_REVIEW_KO.md</summary>

Source SHA256: `2b2a758594ba26a8c636e43ed4256c58a64cd2af4f7e2e9205d33d1d8d25e253`

````markdown
# 동료 연구 확인 — 2026-10-18

이번 222번 사이클에서도 origin/patch_v2를 fetch했다.
HEAD 7a21632b93805f4f1da69a2e77a9ddd95c624d0a로 새 변경은 없다.
PV2_0102의 seed 편차 및 초기 headline 철회 관련 기존 검토를 유지한다.
세 독립 source 모델의 결과를 모두 동결한 뒤 평가하며, 단일 seed나 TRAIN loss
감소를 segmentation 개선으로 대체하지 않는다. 이번 수정 근거는 221번에서
실제로 확인한 TRAIN/DEV 내부 주파수 분산 증가다.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `8d56fb99992158f33cf100139dc4465af52ffe514c4eed91adbe062dcfc42add`

````markdown
# SW0222

Prospective three-seed native64 teacher-denoising model test. Existing source
object-frequency means and gamma are frozen TRAIN teachers. No arbitrary
inter-object frequency margin; original core/gates/readout retained.
See DESIGN_KO.md. No segmentation result claimed before execution.

Actual start2026-10-18:9 local/9 Linux tests,3 local CLI checks and all3 actual source/TRAIN/teacher CPU admissions passed. Frozen teacher initial loss matches source within-object variance; original gamma anchor is zero. Three exclusive idle GPU queues registered; START_EVIDENCE.json records actual NVIDIA-bound workers. No segmentation improvement claimed before all three final endpoints freeze and score. Full320 incumbent .848615/.592533/.672047 unchanged.

Completed: fixed16 three-seed .854272/.631545/.695736 vs source .842891/.600692/.702161. FG +.011381 CI[-.004210,.034446], Object IoU down; full320 expansion gate failed. Retain as uncertain FG-track candidate. DEV within-object frequency variance decreased on all3 seeds. Actual3 checkpoints/teacher/96 masks and independent metrics/CI audited, with no repeated full-model factory or extra forward. Full320 incumbent remains .848615/.592533/.672047. See RESULT_KO.md and completed_v1.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `9f2fe5ebca6e458dfa19d2e73c0cd724477de36b12b9721cec8c49bf104a0bf9`

````markdown
# SW0222: 내부 주파수 일관성 개선, FG 평균 소폭 상승은 아직 불확실

기존 세 native64 source에서 source의 TRAIN 물체별 주파수 평균을 teacher로
고정하고 gamma anchor와 함께 encoder만 학습했다. TRAIN64,128 update/B16,
기존 core/gates/feature 통계/1024-step spike readout을 유지했다. 세 최종 protocol을
모두 동결한 뒤 기존 DEV16, canonical128/no crop,2px modal GT로 평가했다.

| 세 seed 평균 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 기존 source | .842891 | .600692 | .702161 |
| source teacher denoising | .854272 | .631545 | .695736 |
| 차이 | +.011381 | +.030853 | -.006425 |

FG 차이의 paired image 95% CI는 [-.004210,.034446], Foreground IoU는
[-.006984,.075435], Object IoU는 [-.038799,.032782]로 모두 0을 포함한다.
사전 full320 gate(FG+.01/CI하한>0/Object IoU 비감소)는 실패했다. FG track 후보는
유지하지만 확실한 개선, 새로운 full320 최고, .90 달성으로 승격하지 않는다.
이 단계에서 70,000장 확대나 LR/epoch/anchor 계수 sweep을 하지 않는다.

| seed | 기존 FG | denoising FG |
|---|---:|---:|
| 0 | .924401 | .925242 |
| 1 | .815856 | .807814 |
| 2 | .788414 | .829759 |

평균 상승은 주로 seed2에서 나온다. seed1은 하락했고 seed0의 변화는 작다.
seed0의 .925242는 반복 사용한 16장 mini 결과이며 전체320 결과가 아니다.

221번과 달리 이번에는 실제 DEV 내부 주파수 분산이 세 seed 모두 감소했다.
seed0 .005064 -> .003530, seed1 .017437 -> .014881,
seed2 .007019 -> .005821로 약15~30% 감소했다. 반면 접촉 class 사이 평균 거리도
줄었고, 내부 일관성 개선만으로 모든 seed의 mask가 향상되지는 않았다.
source 물체 중심 유지 MSE 및 TRAIN gamma anchor는 감사 결과에 함께 보관했다.
이는 입력 주파수 통계이며 recurrent rhythm, 각 gate의 인과 기여, 데이터 scaling의
증명이 아니다.

실제 세 encoder checkpoint/Adam128/history/고정 teacher 및96개 raw mask를
검증하고 독립 metric으로 재평가했다. metric/평균/paired CI 최대 차이2.22e-16,
별도 로컬 평균/CI 재계산 차이는0이다. frozen scorer가 실제 source+최종 encoder
모델을 복원한 검증을 재사용했고, 감사에서 비싼 full-model factory나 추가 forward를
반복하지 않았다. 집계 증거는 results_archive/completed_v1에 있다.

상속한70k/core 및 native32 TRAIN512 supervision이 있는 supervised 추가 단계다.
새 native64 70k 학습이나 unsupervised 성능으로 취급하지 않는다.
native64 전체320 최고 세 seed 평균 .848615/.592533/.672047은 그대로 보존한다.
다음 구조 후보는 encoder에 더해 실제 gamma-to-drive projection을 함께 학습해,
주파수 일관성의 개선이 spike 구분으로 전달되는지를 같은 조건에서 확인하는 것이다.
이 후보는 아직 구현·실행 결과가 없다.

````

</details>
