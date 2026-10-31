# SW_0218_native64_energy_ranking

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `474e808a59f7e6c82224b0ebbbb12409a0d811c3d03adcd1df136e272c56dbd3`

````markdown
# SW0218: 실제 production partition의 energy ranking 학습

217에서 actual Q의 같은-object pair 정렬 AUC는 .9237-.9778이나, false merge의
양수 native energy 비율은89.59-99.58%였다. 같은 에너지를 높이는 분할은 FG를
개선하지 못했다. 고정한 에너지 optimizer 추가 탐색 대신 실제 readout이 쓰는
Q/A/gain의 에너지 순서를 TRAIN 정답과 맞춘다.

## 기존 실패와 차이

191은 잘못된 쌍의 component rho 제곱/production Q 제곱 surrogate였다.
199는 최종 Q의 pair-balanced Bernoulli loss,204는 component별 prototype QCE,
206과214-216은 별도 관계 classifier였다. 이번에는 별도 head나 decision
threshold를 만들지 않고 원래 signed energy 자체를 미분한다. 이것이 더 좋다는
주장은 실험 전 가설이다. 작은 margin/LR/epoch rescue sweep은 하지 않는다.

## 정확히 맞춘 source와 control

204와 동일한192 TRAIN8 seed0 model, 실제 Adam moments/steps2, CPU/CUDA RNG에서
시작한다. 새 TRAIN indices8..15의 B4 두 proposal, micro1, 같은5개 learning
rate/clip1/projection, finite-Adam policy(유한 uphill step도 유지)를 사용한다.
Component objective는204의 실제 함수를 재사용하고 변경하지 않는다.
Control은 이미 완료된204이며 TRAIN/초기 model/Adam/RNG/예산이 일치해야 한다.
기존 최고 warm192와도 비교한다. 이 seed0 mini는 독립 세 seed 목표의 대체물이
아니다. 효과가 확인되기 전에 전체320/70k를 추가 등록하지 않는다.

## Frozen TRAIN reference와 objective

각 proposal의 before-step no-grad native rollout에서 production spike readout을
실제로 실행해 예측 전경과 partition을 얻는다. 그 전경 안에서 각 GT label의
4-connected 조각을 reference로 삼는다. GT0인 false-positive patch도 그대로
포함한다. 전경 누락을 정답으로 복구하거나 GT object count를 inference에 넣지
않는다. Reference는 TRAIN GT를 이용한 구체적 feasible partition이며 optimum,
성능 상한 또는 predictor가 아니다.

해당 전경에서 D_ij = same(pred)_ij - same(reference)_ij를 정한다. Q/A/gain은
원래 signed_pair_energy와 같은 식, float64, Q_EPS1e-6 및 기존NATIVE_NODES1024를
사용한다. 4096이나 전경 크기로 정규화 상수를 바꾸지 않는다.

Loss = component_QCE204 + ReLU(1 + sum(E_current * D) / scale).
scale = max(sum(abs(E_before) * abs(D)), 2 * disagreement_pairs * Q_EPS).
단위 margin과 coefficient1을 고정한다. D=0이면 미분 가능한0을 반환한다.
분자/분모 모두 unordered pair를 두 번 세므로 factor2는 상쇄된다. 실제 source
partition, reference, D와 scale은 before/gradient/after에서 고정하고 parameter
변경 후 다시 mining하지 않는다. 이는 ranking surrogate이며 structured-inference
bound나 미분 가능한 FG-ARI라고 하지 않는다.

원래 production inference, graph, encoder, 모든 gating은 유지한다. Gamma를
core와 별도로 고정하지 않는다. 기존5개 parameter family를 실제 함께 update한다.
T1024/settle512/live-tail64, gradient/no-grad 같은 RNG loss parity를 검증한다.
추가 gradient-only 데이터가 없고 logical TRAIN exposure는 inherited8+new8=16이다.
Auxiliary 자체의 emitted-spike gradient와 각 proposal의 첫 active image에서
실제 parameter-family gradient RSS도 확인한다. Auxiliary의 실제 S/parameter
credit이 적용된 step에서 확인되지 않으면 에너지 학습 성공으로 보지 않는다.
이는 gate별 인과 기여 증명이 아니다.

## 실행 및 판정

소스 전송 전 CPU operator/finite-difference/context/transaction/GT barrier tests,
서버에서도 같은 tests와 actual model/Adam/RNG CPU admission을 통과해야 한다.
GPU0..3 exclusive lease, foreign owner 양보와 proposal/이미지 checkpoint resume.
기존 파일과 이미 실행 중인 ancestor source는 바꾸지 않는다.

동일 native64 canonical128/no crop/2px modal GT, fixed16 offsets0,2,..30,
RNG193000+offset, 원래1024-step spike inference. Candidate protocol을 고정한 뒤
DEV GT를 읽고 기존204 control/warm과 paired10000 bootstrap으로 비교한다.
확대 gate는 control 대비 FG+.01 이상 및95% CI lower>0, warm 대비 FG 감소 없음,
두 비교의 ObjectIoU 감소 없음, 실제 새 update/CNN 변화/auxiliary parameter
credit이다. FG-only 유용 후보는 별도 보존한다. 최종 .90 판정은 전체320의
독립 세 seed 평균에서만 한다. Mask-supervised 실험이며 unsupervised 승리,
신규 native64 TRAIN70k 또는 데이터 scaling 증거라고 하지 않는다.

````

</details>

## PEER_REVIEW_KO.md

<details>
<summary>PEER_REVIEW_KO.md</summary>

Source SHA256: `772b64266b9d8bd472ffa800bcf5d7d52b515d2be727eed15d395623fcb6f042`

````markdown
# 이번 사이클의 동료 연구 확인

origin/patch_v2를2026-10-18에 다시 fetch했고, 이전a0a39df에서
7a21632b93805f4f1da69a2e77a9ddd95c624d0a로 갱신된 것을 확인했다.
해당 commit의 PV2_0102_grid32_seeds/README.md, metrics.csv와101 정정을 읽었다.

동료는8 epoch/native32 seed0 .7699/.7531/.7017이라는 결과가 세 seed에서
재현되지 않아 기존 headline을 철회했다. w0의 FG는 .7699/.2916/.7068,
평균 .5894, 표본 표준편차 .2599다. group-count w5도 .7803/.2118/.7617,
평균 .5846, 표본 표준편차 .3230으로 안정화되지 않았다. 비교 대상 .7239는
동료 자신의 published16 설정이며 Slot Attention 수치가 아니다.

동료는 foreground 과다 예측과 단일-group 동기화 실패를 구분했고, collapse
통계 하나로 seed 실패를 예측할 수 있다는 해석도 철회했다. 다음 graph-freeze
시험은 제안되어 있지만 이 commit에 완료 결과는 없다. 해당 contract의
고정 graph를 native64의 확정 개선으로 가져오지 않는다.

적용한 원칙은 동일 seed/source/학습 예산 비교, 세 seed 전체 평가 유지,
proxy 하나가 아닌 실제 mask/foreground/instance 오류 확인이다. 218은
완료된204와 같은 warm192 및 실제 Adam/RNG에서 시작하며 새 loss만 바꾼다.
실행 중 fingerprint를 바꾸거나 동료의 count term/8 epoch를 그대로 추가하지 않는다.
현재 우리의 full320 .848615와 동료300장/다른 계약의 .5894를 직접 순위화하지 않는다.

추가 주의: 동료의 seed0 8-vs40 epoch 비교는 +.1406의 근거가 있지만,
seed2 .7068을 seed0의40 epoch .6293과 비교한 사실만으로 seed-independent
overtraining이 증명되지는 않는다. 데이터 scaling과 epoch budget은 같은
seed, 데이터, readout을 맞춘 비교로 계속 구분한다.

근거: https://github.com/Taeyoon-J/SNN_Kuramoto_Bidirectional/tree/7a21632b93805f4f1da69a2e77a9ddd95c624d0a/collaborative_test/PV2_0102_grid32_seeds

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `c8256852a9245befc4889ad6dec5081c64d3f6c87fea007dc66c4405eb0ce7c9`

````markdown
# SW0218 native64 energy-ranking training

Compare against completed SW0204 from the exact same warm192 model, Adam/RNG
and eight new TRAIN images. Add a loss that ranks actual native Q/A/gain energy
of the mined production partition below a TRAIN-GT connected reference. Keep
the source support, frozen before-step context, all gates, joint encoder/core
updates and the production spike readout. No decoder threshold or head fitting.

See DESIGN_KO.md for formulas, parity/credit checks, exact control, fixed16
screen and the full320 three-seed .90 goal. No result is claimed before execution.

Actual start2026-10-18: 22 local and22 Linux tests passed; actual CPU model/Adam/RNG admission passed. Supervisor2368483, worker2368524 on exclusive GPU1. Registration and actual NVIDIA/process evidence are archived. Peer patch_v2 was rechecked at7a21632; see PEER_REVIEW_KO.md. No new score yet.

Completed: FG .898687 vs matched204 .900539 and warm .924401; no expansion. Training and all predictions completed, but a renamed control import broke original CPU scoring. Separate recover_score.py restores only scoring without changing frozen source, models, Adam or masks. Four new local/server tests exercise the complete scoring branch; actual48 endpoints/model/CNN/Adam/RNG were independently audited. See RESULT_KO.md and completed_v1. The original failed queue record is preserved.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `527d2a8c6ef34375f692d0388f4db6dcb3614cdf22425b8674257bdfd1c537d8`

````markdown
# SW0218 완료: energy-ranking loss는 FG 개선 없음

동일 native64/fixed16/seed0, 실제 warm192 model/Adam/RNG에서 시작해204와 같은
새 TRAIN8/B4 두 proposal을 수행했다. 모델과 CNN이 바뀌고 실제 Adam step4가
저장됐다. 원래1024-step spike readout과 encoder/core/graph/gating 구조는 유지했다.

| 설정 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 새 energy ranking | .898687 | .858750 | .729999 |
| matched204 component QCE | .900539 | .860577 | .727622 |
| 기존 warm192 | .924401 | .859896 | .727638 |

204 대비 FG 차이-.001852,95% paired CI[-.005725,+.001296]이다.
Warm 대비-.025715, CI[-.073678,-.000170]로 개선되지 않았다. ObjectIoU의
작은 양의 차이도 두 CI 모두0을 포함한다. 확대 gate는 false다. 이 loss의
작은 weight/margin/LR 탐색은 진행하지 않는다. Full320 최고 평균
.848615/.592533/.672047과 .90 목표 미달 상태는 그대로다.

## 실제 학습과 진단

Auxiliary 자체의 emitted-spike gradient가8개 새 TRAIN 이미지에서 양수였다.
각 proposal 첫 active 이미지에서 context encoder/graph/native core/integration/
원래 CNN의 실제 auxiliary parameter gradient도 모두 양수였다. Gradient가
존재하고 파라미터가 바뀌었다는 사실이 분할 성능 개선을 보장하지 않았다.
같은 RNG의 no-grad/gradient loss 차이는2.61e-8 및1.49e-8이었다.

첫 proposal의 실제 combined TRAIN loss는2.819465->2.947188로 상승했고,
두 번째는2.554252->2.478408로 감소했다. 두 proposal은 다른 TRAIN 이미지이므로
이 수치를 서로 이어진 loss curve로 해석하지 않는다. 유한 uphill step도
유지하는 사전 고정 정책을 따른 것이다. 새 목적함수에서도 이전 목적함수의
Adam 상태를 이어 쓰는 영향이 있는지는 아직 미검증이다. 다음에는 동일한
gradient/모델/이미지에서 Adam 상태 유지와 초기화를 비교하는 한 번의 제한된
진단을 검토한다. 현재 결과만으로 Adam 상태가 원인이라고 단정하지 않는다.

## 집계 오류와 복구 범위

학습과16장 candidate 예측은 정상 완료했다. 이후 evaluator의 candidate arm을
문자열 치환하면서 control package까지 SW_0204_native64_energy_ranking으로
잘못 바꾼 탓에 CPU 집계 import가 실패했다. Local/server22 tests는 이 전체
집계 분기의 실제 import를 실행하지 못했다. 서버 차단이나 모델 학습 실패가 아니다.

원래218 학습/추론 소스와 fingerprint, model/Adam/checkpoint/예측을 유지했다.
별도 recover_score.py가 실제204 control을 명시적으로 import하고48개 고정
mask를 검증한 뒤 GT를 읽어 집계만 복구했다. 전체 집계 분기를 실행하는
새4개 local/server tests와 actual frozen-mask CPU admission을 통과했다.
재학습/추가 native forward/update는0이며 원래 queue의 failed 기록도 보존했다.

별도 독립 감사에서 실제 model/CNN/Adam/RNG/학습 이력/auxiliary credit과48개
mask endpoint의 세 지표/paired CI를 재검산했다. 최대 metric 차이2.22e-16,
새 source model SHA는6735c80ceb83404c8a4748421be330a076603dd20d3377d403f9003e40bab2da다.
결과는 supervised mini이며 full320/세 seed .90/신규 native64 TRAIN70k나
각 physical gate의 인과 기여 증명으로 표기하지 않는다.

근거: results_archive/completed_v1/score_v1/summary.json, completion_audit.json,
training/report.json 및 score_recovery_v1/receipt.json.

````

</details>
