# SW_0224_native64_aligned_feature_cells

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `012f164f38c7c5db5c67f69e76165573ee1c6d6aaf2e0fe900b5d4e8ae0bef49`

````markdown
# SW0224: encoder feature 중심과 기존2px 평가 cell의 정렬

별도 구조 가설이다. 기존 encoder는128 RGB에서 valid3x3 및 context crop 후
126x126 feature를 낸다. 기존64 adaptive pooling은 축당64개 중60개 cell에서
feature3개를 평균하고62개 인접 경계에서 겹친다. receptive-field center 기준으로
기존 균일2px GT cell 중심 대비 평균 절대 오차는 축당 .5px다.
이는 기하 확인이며 성능 하락 원인이라고 아직 주장하지 않는다.

기존 feature 값/정규화/clip은 그대로 두고 가장자리 feature를 한 줄 복제해
128x128로 만든 뒤 같은64 patcher를 쓴다. 내부 cell은 정확히 비중첩2x2 평균과
대응한다. 경계는 가장 가까운 기존 feature를 복제한다. RGB/GT/HDF5/ID/평가 규칙,
CNN/core 가중치,1024 settle512 spike readout은 바꾸지 않는다. 추가 학습은0이다.
모델 식별자는 같은 parameter SHA와 새 adapter 코드 SHA를 함께 쓴다.

warm192/205 세 seed 공통. DEV16 IDs1320,1322,...1350, RNG193000+offset,
canonical128/no crop/2px modal GT. 기존 동일 모델의 mask가 control이다.
세 최종 protocol을 모두 검증한 뒤 GT를 읽는다. source checkpoint bytes와 모델
state SHA가 유지되고 source statistics가 변하지 않는지도 확인한다.
이 방법은 기존 구현과 수치적으로 동등하다고 주장하지 않는다.

FG track full320 후보는 기존 source 대비 FG+.01 및 paired image CI하한>0,
balanced 후보는 여기에 두 IoU 비감소를 요구한다. tradeoff를 함께 기록하고
full320/.90 성공을 mini 결과로 대체하지 않는다. 새70k 학습은 하지 않는다.
source의 mask-supervised lineage는 그대로 상속한다. 각 gate의 인과 기여나
데이터 scaling을 이 시험만으로 입증하지 않는다.

````

</details>

## PEER_REVIEW_KO.md

<details>
<summary>PEER_REVIEW_KO.md</summary>

Source SHA256: `a02c0fe41be07ee42f26c5096116bf7811b72c02b5b367725dcf3016ca0ac6c9`

````markdown
# 동료 연구 확인 — 2026-10-18

224번 배포 후에도 origin/patch_v2를 fetch했고 HEAD는
7a21632b93805f4f1da69a2e77a9ddd95c624d0a로 변동이 없었다.
이번 가설은 실제 encoder의 valid3x3/context crop126 출력과64 adaptive pooling의
cell 중심을 대조한 자체 기하 진단에서 나왔다. 동료의 미검증 제안이나 단일 seed
headline을 포팅한 결과로 취급하지 않는다. 세 기존 source의 가중치를 유지해
동일 RGB/GT/ID/readout에서 새 입력 adapter만 비교한다.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `45e11c2ddb32a9d806df3641bd8e27276b330e3a4df5dc025f274d91ccb8834c`

````markdown
# SW0224

Independent zero-update feature-cell alignment model test. Existing126x126
features get one replicated border before native64 pooling. Original128 RGB,
2px modal GT, source parameters and spike readout remain fixed. This changes
the inference adapter and is not a numerically equivalent optimization.
Three source seeds, same16 images. See DESIGN_KO.md; no result claimed yet.

Actual start2026-10-18:7 local/7 Linux tests and all3 source/immutable-checkpoint/actual126-feature CPU admissions passed. Three exclusive idle GPU queues registered; START_EVIDENCE.json binds actual NVIDIA workers. Source parameters/statistics/RGB/GT remain fixed; only the declared feature-border/pooling adapter changes. Zero optimizer updates. No new segmentation result; full320 incumbent .848615/.592533/.672047 unchanged.

Completed: fixed16 three-seed aligned .838302/.586019/.713245 vs source .842891/.600692/.702161. FG and Foreground IoU decline; Object IoU rises with CI including0. FG/balanced gates False. Keep uncertain Object IoU track; do not expand the zero-update adapter for FG. Source checkpoint/adapter/96 masks and independent metrics/CI audited, no extra forward or training. Full320 incumbent unchanged.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `5847e0c724111d1681c08fea04675e8bd013cf35e07cfa1a963dfed9ab8f1a57`

````markdown
# SW0224: cell 정렬만으로 FG 개선은 확인되지 않음

CNN/core/source 통계/128 RGB/2px modal GT/ID/readout을 고정하고126 feature에
복제 경계 한 줄을 붙여64 pooling cell을 정렬했다. 추가 학습0이다. 같은 source
세 모델과DEV16에서 비교했고 세 protocol 모두 동결한 뒤 GT를 읽었다.

| 세 seed 평균 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 기존 source | .842891 | .600692 | .702161 |
| cell 정렬 | .838302 | .586019 | .713245 |
| 차이 | -.004588 | -.014672 | +.011084 |

FG paired image95% CI[-.035006,.037387], Foreground IoU[-.051109,.014607],
Object IoU[-.022370,.044649]로 모두0을 포함한다. FG/balanced 확대 gate는 실패했다.
Object IoU track 후보는 보존하지만 확실한 개선이나 새 최고 모델로 승격하지 않는다.
FG 우선순위에 따라 이 무학습 adapter의 확대·padding/stride sweep은 하지 않는다.

축당 평균 feature cell 중심 오차 .5px -> .015625px라는 기하 확인과,
세 seed 평균 segmentation 성능 개선 여부는 다른 질문이었다. 첫 두 seed의FG는
하락했고 seed2는 상승했다. 이 결과만으로 모든 정렬 방법이나 학습 후 효과까지
부정하거나, 원래 성능 문제의 원인이 pooling 하나였다고 주장하지 않는다.

원본 checkpoint byte SHA와 모델 parameter SHA/통계 유지, adapter 코드 SHA,
96개 raw mask의 독립 metric 및 모든 평균·paired CI를 감사했다. 최대 오차는
2.22e-16이며 별도 로컬 평균/CI 재계산 차이는0이다. 감사의 추가 forward/학습0,
RGB/GT resize/crop0. 증거는 results_archive/completed_v1에 있다.

source의 mask-supervised lineage를 상속한다. 추가 학습이 없다는 이유로 이
전체 계열을 unsupervised 모델로 부르지 않는다. native64 전체320 최고
.848615/.592533/.672047과223의16장 결과는 변경하지 않는다.

````

</details>
