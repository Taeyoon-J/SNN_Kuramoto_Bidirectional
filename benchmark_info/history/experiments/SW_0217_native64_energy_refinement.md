# SW_0217_native64_energy_refinement

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `af5cae1180e1090435a1de8359a286568566249704fb0116178033e65cad0938`

````markdown
# SW0217 prospective: native signed-energy split and remerge

실행 전 고정한 구현 명세다.216까지의 merge-only classifier는
개선되지 않았다. 현재 baseline 각 group은 이미4-connected이므로 단순
connected-component 분리는 반복하지 않는다. Native greedy agglomeration의
초기 잘못된 merge를 실제 동일 energy에서 되돌릴 수 있는지 시험한다.

## 고정 source와 평가 범위

- 실제210 기준점의 독립 source0/1/2를 사용한다.0 warm192 model SHA
  2dd2b0d64602b85d4082bcaddcd21dfe8c60d34af1ea78a2ed29bf03c8ba4b6e;
  1 warm205 e1e16281612b2171b6822e0ca70dee051f095ad7ebc9239a9675b8e8d9b391e0;
  2 warm205 dff381ebcd5280a2a7bf7b99ef8ee3b5af30e91f3384962ef47c6d89b7da53d0.
- Seed0 factory: SW0193.prepare_inputs_cpu의 found에 candidate_state를 실제
  복원한다. Seed1/2: SW0210.prepare_cpu(seed)를 그대로 재사용한다.
- 같은16 offsets0,2,..30/IDs1320..1350, canonical128 RGB/no crop/native64,
  inference RNG193000+offset,1024 steps/settle512/live_tail0. 학습 update0.
- 원래 baseline mask를 모든 image에서 bitwise 재현해야 후보를 허용한다.
  Core/encoder/graph/gating은 유지하며 실제 model SHA를 전후 확인한다.

## 에너지와 새 optimizer

같은 actual Q/A/gain에서 SW0162.signed_pair_energy를 그대로 호출한다.
NATIVE_NODES 상수나 graph normalization을 이번에 바꾸지 않는다.268-channel
anchor cache로 full energy를 복구했다고 하지 않는다. 새 core forward가 필요하다.

기존 group마다 E_block의 signed Laplacian diag(E.sum(1))-E를 계산한다.
최소 eigenvector의 zero cut과 sorted-vector cut 중 cross-energy가 최소인
prefix cut을 고정 후보로 삼는다. 각 side를 공간4-connected pieces로
나눈 뒤, 실제 전체 within-group energy가 증가하는 후보만 수락한다.
동일성/tie는 row-major 순서로 처리한다. 허용오차는 float64 machine epsilon과
실제 abs-energy 합에서만 정하고 GT/metric으로 고르지 않는다. 원래 group당
한 번만 proposal한다. 추가 recursive/eigenvector/threshold sweep은 없다.

Secondary split-only와 primary split-then-remerge를 함께 등록한다. Remerge는
split된4-connected groups에서 시작해 기존과 같은 full cross-energy 합이
양수인 인접 group을 최대 gain/canonical tie 순으로 합친다. 원래 source FG는
모두 보존하며 singleton도 버리지 않는다. BG 재선택/min-size/count cap 없음.
모든 group4-connectivity, FG exact equality, energy nondecrease를 검증한다.

## 검증과 저장

Operator CPU tests: synthetic negative cross-energy split/positive clique identity,
connectivity after eigen cut, exact foreground, label-ID permutation invariance,
deterministic tie, split/remerge energy monotonicity, empty/singleton, invalid inputs.
Native source/CUDA 전 실제 CPU factory admission과 source reference hashes를 확인한다.

GPU0..3 exclusive lease/per-image checkpoint를 사용하고 외부 owner에게 양보한다.
각 row는 baseline/split/refined masks와 full-N 계산 후 자른 실제 Q_F/A_F,
F indices/gain을 dtype 손실 없이 server-private NPZ에 저장한다. Full Q/A와
energy hashes도 기록한다. 정규화를 F 크기로 바꾸거나 foreground oscillator를
다시 실행하지 않는다. 저장 전에 실제 free disk와48행 최악 크기를 확인한다.
외부 auditor는 원래 operator 상수로 Q_F/A_F에서 E_F를 재계산하고 실제
mask energy와 operator replay를 검증할 수 있어야 한다. Local/GitHub에는
metadata만 내보내며 Q/A/mask arrays/checkpoint/RGB를 공개하지 않는다.

세 seed의 모든 endpoint를 고정한 뒤 GT를 읽고 세 지표·count·energy gain·
accepted split 수·paired CI를 보고한다. 같은16의 세 seed 평균이므로 full320
.90 또는70k 완료 주장이 아니다. No split/energy gain이면 후보를 찾지 못한
것이며 global optimum/모든 energy cut 불가능을 증명했다고 하지 않는다.
Energy가 좋아져도 mask metric이 좋아진다고 가정하지 않는다.

확대 후보는 primary의 세 seed mini 평균 FG delta>=.01, paired95% CI lower>0,
ObjectIoU 평균 감소 없음으로 판정한다. FG-only 후보도 별도 보존하지만 작은
양의 point estimate만으로 전체320/70k를 확장하지 않는다.

개선이 크고 재현되면 전체320에서만 최종 비교한다. Gate 필요성이나 data
scaling은 이 readout-only 시험으로 입증되지 않는다. 기존 .848615 reference와
216의 실패 판정은 유지한다.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `abcfd535aa2ab2a4cc9e08dad2c00dffc1e045c8498f3f8f3edcf96d00e59c25`

````markdown
# SW0217 native64 signed-energy split/refinement

Use the actual best-reference models0/1/2, same fixed16 images and native64
baseline readout. Spectral proposals may undo a source merge only if the
original signed Q/A/gain energy improves after enforcing4-connectivity.
Split-only is secondary; split-then-warm-remerge is primary. No model update,
source foreground change, GT count, dropped component or new normalization.

New native rollouts are required:268-channel anchor caches cannot reconstruct
full energy. Actual induced Q/A remain server-private for exact energy and
operator replay audit. Baseline masks must bitwise reproduce the saved reference.
See DESIGN_KO.md. Local tests cover optimizer invariants and full-N induction;
actual CPU source/disk admission precedes exclusive-GPU queue registration.
This is a mini test; the full320 three-seed .848615 incumbent remains unchanged.

## Actual admission and start (2026-10-18)

Eight local and eight server tests passed, followed by real CPU model/data/disk
admission for each of three independent seeds. Seed0 and seed1 started on
exclusive GPUs1 and3; seed2 is registered to acquire the next idle device.
Actual process identities and NVIDIA rows are archived in started_v1; registration
and source SHA bindings are in deployment_v1/registration_receipt.json.
Peer patch_v2 was rechecked at a0a39dfb081c1012e1beeae1136c4e7a9ef6073b.

No new result is claimed at registration. This tests whether the existing native
signed energy can correct grouping by splitting and remerging, rather than
training another unconstrained merger. It does not establish gating causality,
data scaling, fresh native64 70k training, or full320 FG-ARI above .90.

## CPU posthoc diagnosis registered

After all three seeds' predictions and scores are fixed, diagnose_energy.py
reuses private Q/A and checks whether same-object versus different-object pairs
are ordered correctly by Q and by the exact signed energy. It separately counts
positive evidence at existing false merges and negative evidence at false splits.
An explicitly GT-assisted, connected purification reports its energy change;
this is a diagnostic action, not a prediction, optimum or performance ceiling.
No threshold search, core forward, model update or running source change occurs.
Four local and four Linux tests passed; registration is recorded under
results_archive/posthoc_diagnosis_registration_v1/. An initial deployment probe
looked for a receipt that existed only locally and stopped before any file write;
passing that receipt explicitly resolved the preparation error. GPU jobs were
unaffected. Pair signs/AUC cannot prove a global partition optimum or gate causality.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `c5ae96e02bf25c01258626f782aefde3abc7a533be0219535d5c238f7ff8062a`

````markdown
# SW0217 결과: 원래 에너지의 분할·재결합은 FG-ARI 개선 없음

64x64 patch, 같은 고정16장, 독립 seed0/1/2, 원래1024-step spike readout이다.
원래 전경과 모델을 고정하고 actual Q/A의 signed energy를 높이는 spectral
split과 인접 재결합을 적용했다. 학습0 update, fresh70k 학습 없음.

| 같은16장의 세 seed 평균 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 기존 기준 모델 | .842891 | .600692 | .702161 |
| split-only | .842891 | .600692 | .702180 |
| split 후 재결합 | .842891 | .600692 | .702180 |

FG 차이0, paired image95% CI[0,0]. ObjectIoU 차이+.00001847로 확대 근거가
아니다. seed0/1/2 FG는 각각 .924401/.815856/.788414로 동일하다.
원래 에너지는 seed1/2 일부 이미지에서 증가했지만 FG가 개선되지 않았다.
이것은 모든 가능한 분할의 불가능성이나 global optimum의 증명이 아니다.
이 방향의 추가 eigenvector/recursive/threshold 소규모 탐색은 진행하지 않는다.

실제48개 Q/A 행에서 full-N energy를 재구성하고 optimizer를 다시 실행했다.
144 mask endpoint와 독립 세 지표/paired CI를 감사했고 최대 metric 차이는
2.22e-16이다. 실제 세 GPU worker와 큐는 정상 완료했다. 전경 정확 보존,
모든 그룹4-connected, source model SHA 불변과 energy nondecrease를 확인했다.

## 완료 후 GT를 이용한 에너지 정렬 진단

추가 core forward/update 없이 저장한 Q/A를 사용했다. 아래 pair AUC는 source
전경에 잡힌 GT 전경 patch 쌍에서 같은 물체를 더 높게 정렬하는 정도다.
잘못 합쳐진 쌍의 양수 에너지 비율은 해당 오류가 존재하는 이미지 평균이다.

| seed | Q pair AUC | energy pair AUC | 잘못 합쳐진 쌍의 양수 energy 비율 |
|---|---:|---:|---:|
| 0 | .9778 | .9737 | 99.58% (10장) |
| 1 | .9237 | .8859 | 98.18% (9장) |
| 2 | .9366 | .9138 | 89.59% (13장) |

반대로 잘못 쪼개진 같은 물체 쌍의 양수 energy 비율은3.35/14.22/4.85%다.
GT를 알고 각 기존 그룹을 GT별4-connected 조각으로 나누는 구체적 참고
행동은 모든 seed에서 평균 energy를 낮춘다. 참고 행동은 predictor, 최적해,
학습 가능한 상한이 아니다. Pair AUC도 최적 partition 성능의 증명이 아니다.

Q에 물체 구분 순위 정보는 남지만 현재 부호와 graph 가중 energy가 실제
잘못된 묶음을 지지하는 경우가 많다. 따라서 동일 energy optimizer만 바꾸는
시험을 종료한다. 다음은 실제 native Q/graph evidence와 연결 판정의 TRAIN-only
정렬을 검토한다. 206의4-rho MLP,199의core Bernoulli loss,214-216의component
merger/odds/target 시험은 이미 실패했으므로 그대로 반복하지 않는다.
DEV GT를 이용한 threshold 선택이나 fitting은 허용하지 않는다.

Full320의 세 seed 기준 .848615/.592533/.672047은 그대로다. 고정16장의
점수를 전체320, native64 신규70k, unsupervised, gating 인과 기여 또는 .90
목표 달성으로 표기하지 않는다. 모든 source/결과는 기존 patch_v2_sw에 공유한다.

근거: results_archive/completed_v1/score_v1/summary.json,
completion_audit.json 및 energy_diagnosis_v1/aggregate_summary.json.

````

</details>
