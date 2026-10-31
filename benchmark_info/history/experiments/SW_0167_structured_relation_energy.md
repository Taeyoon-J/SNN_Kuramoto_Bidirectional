# SW_0167_structured_relation_energy

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## ADMISSION_RESULT.md

<details>
<summary>ADMISSION_RESULT.md</summary>

Source SHA256: `4f9efff1ff75dcbdb196e9402e44e95a61da8d134c7a278e9ef2772f49821534`

````markdown
# Actual TRAIN/GPU admission, 2026-10-15

The frozen seed-1 source passed the actual eight-image TRAIN admission on an exclusively leased GPU1. Both the structured-partition objective and matched local-margin control completed two disposable Adam updates. This admits the fixed 256-update comparison; it is **not a performance result** and does not replace SW165 as incumbent.

All eight rows retained the original cached Q/A/gain and exact zero-residual baseline energy, merge log and output labels. Foreground support came from production CPU-B4 QCC on the original saved component spikes. GPU-B1 rho was combined with original cached context Q, with exact feature parity against the old wrapper checked separately on same-mode inputs. The two arms used identical features and identical initial head state (`489211c0dbb25db3729364642cd2e45590b7c5003d1436b23f11ed45238b0a98`). No VAL images or masks were read.

| Objective | Disposable loss, step 1 → 2 | Output-layer gradient, step 1 | Hidden-layer gradient, step 2 |
|---|---:|---:|---:|
| Structured partition | .277337 → .277266 | .072757 | .000101931 |
| Local margin | .548849 → .548685 | .132403 | .000236500 |

Gradient figures sum weight and bias norms; these differently normalized losses must not be compared as model scores. Actual output parameters changed on update1 and hidden parameters changed on update2 in both arms. Independent Sol5.6 review found no concrete math, gradient, device or provenance blocker in the final implementation.

The first admission attempt stopped during CPU target preparation: the existing mask loader returns a Tensor, while the caller used `torch.from_numpy`. No GPU child or training was started by that attempt. Its source and failure evidence remain preserved. The separate `preflight_v2.py` uses `torch.as_tensor`; before relaunch, all ten CPU contract checks and an actual eight-row TRAIN mask/support smoke check passed on the server. The v2 queue then completed successfully with supervisor PID3647147, child PID3647205 and report SHA256 `31757d610b799fef70a28a9096ca71f69cc49f5abf1627cfb4f27732e98e3217`.

Evidence: `results_archive/actual_train_admission_20261015/server_records.json` and `first_attempt_failure.json`. Training remains a separate next stage: same 512 TRAIN images, eight passes, 256 updates, fresh identical initialization, fixed learning rate and final-only evaluation. No hyperparameter or validation selection was performed during admission.

````

</details>

## CPU_CACHE_DIAGNOSIS.md

<details>
<summary>CPU_CACHE_DIAGNOSIS.md</summary>

Source SHA256: `baaf628d6e703a104f6710d4007b325fd5693b2eb1b7937a83295fcfd6314020`

````markdown
# First eight TRAIN rows: CPU arithmetic diagnosis

Read-only server inspection verified source/context manifests, ordered ledgers, selected RGB bytes and spike/Q/A payload hashes for the first eight registered TRAIN cache rows. Both source-state receipts use the same SW0161 digest family and equal `b205d21a917299089b9b5d373eb94c9eb5eb971471e9428072e0fc0229c70749`. The recorded source coupling gain is 7.623262882232666. No mask/GT arrays, validation data, rollouts or optimizer updates were used.

Production CPU Q was recomputed in the original B4 groups. Its maximum absolute difference from the cached GPU Q ranged from 0.0000053644 to 0.0000063777. All eight rows had **zero** threshold-edge disagreements at 0.50, zero QCC label differences and zero foreground-support differences. These observations do not explain an accuracy loss or establish cross-backend bitwise equivalence.

The new solver exactly matched SW0164 energy, labels and complete merge logs on all eight rows using identical cached Q/A and the same CPU-derived support. However, the pair-feature constructor rejected all eight CPU reconstructions against the GPU cache at its unchanged 0.000002 consistency tolerance. Therefore **full preflight failed**: solver-only parity must not be presented as a completed model preflight.

The next arithmetic check compares GPU B4, GPU B1 and cached Q under an idle-GPU ownership lease before attempting the real zero-residual MLP path. The tolerance, cached Q, original source and scientific selection gate remain unchanged. No SW0167 training has started; SW0163 final scoring still determines the next scientific experiment.

Evidence: `results_archive/train_cache_cpu_diagnosis_20261015/server_records.json`. The reproducible read-only script is `audit_train_cache.py`.

````

</details>

## CPU_PREPARATION.md

<details>
<summary>CPU_PREPARATION.md</summary>

Source SHA256: `33c536280383223ceaeb94347edce9cb02f2ec1668049833c3cc6ae2ed8d5941`

````markdown
# Provisional CPU preparation

The reusable solver, pair feature/MLP and loss components are implemented locally while SW0163 evaluation runs. This updates the original design-only implementation status; the scientific admission condition is unchanged. No training/cache generation, server deployment or GPU experiment has started for SW0167. If SW0163 passes its registered gate, replication of that recipe takes priority.

The provisional implementation retains the source foreground and variable object count. A zero-initialized pair correction starts at the registered float64 signed energy. The structured candidate trains through scores of detached greedy partitions; the matched local-margin control uses the same network and inference solver. Ground-truth projected target components include background pixels inside the fixed source support. This is a supervised proposal, not an unsupervised performance claim.

CPU checks cover exact original solver energy/merge-log/label parity on small grids, production-order component correlations and Q reconstruction, disconnected support, feasible targets, gradients and two disposable optimizer updates. They do not establish full native32 TRAIN-cache parity, GPU arithmetic, actual training signal over all eligible TRAIN images, performance, 70k scaling or gating causality. Those admission checks remain required before any experiment.

The implementation preserves original grid coordinates through the oracle, computes normalization over unordered foreground pairs and leaves the existing SW0164 solver unchanged. Additional analytical gradient checks are recorded with the final CPU receipt. No validation masks are used by these synthetic tests.

A separate pure-Q helper now reproduces the production SW0136 QCC readout from an already-computed affinity: full-graph `Q >= 0.50`, connected components, minimum size eight, stable descending-size order, first largest group as background, and the registered native32 label converter. Its synthetic spike test matches the actual production affinity/component path at an exact 0.50 edge and tied largest groups. It accepts only float32 Q, preserving source-cache threshold semantics. This is synthetic CPU readout parity only; parity against the actual SW0163 native32 TRAIN cache, including its saved Q and labels, remains outstanding. No GT or object-count input is used by the helper.

````

</details>

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `43c01261963edb7c0f2427b46e35ddf4341d18c0e1f908ed426b66558ff2493a`

````markdown
# SW0167 조건부 설계 — 실제 연결 partition의 구조학습

2026-10-15. **설계만 작성; 구현·cache 생성·GPU 실행 승인 문서가 아니다.** SW163의 네 arm 최종 검증 결과가 아직 없으므로 지금 실행하지 않는다. SW165/166 및 다른 sealed source는 변경하지 않는다. 이 실험은 **TRAIN mask supervised track**이며 unsupervised 성적표에 합치지 않는다.

## 선택 이유와 진입 조건

완료 기록 `SW_0165_foreground_partition_replication/results_archive/completed_all_seeds_20261015/server_records.json`에서 fixed three-seed means는 FG/foregroundIoU/objectIoU = .804298915/.609975448/.651148975이고 Slot = .781651972/.204360128/.203164986이다. SourceT 대비 FG +.021254909 및 objectIoU +.054951846의 재현 개선이다. 이 성공한 가변 K, 실제 spike Q/actual coupling A, 4-connected foreground partition을 유지한다.

`SW_0166_fixed_support_ari_diagnosis/results_archive/completed_diagnosis_20261015/server_records.json`의 GT-assisted 순차 merge/split 교정 이득은 각각 .05530949/.05555731이다. 한 종류만 지배한다고 판단하지 않는다. .91516572는 connectivity 등을 완화한 **인증 상한**이며 이 solver로 도달 가능한 점수나 .87 보장이 아니다. GT attachment .87261955도 predictor가 아니다. 이 진단의 VAL GT를 TRAIN target이나 seed/계수 선택에 재사용하지 않는다.

SW163 Q_A가 등록된 gate를 통과하면 먼저 그 고정 recipe의 seed0/2 복제를 검토하고 이 후보는 대기한다. SW163가 통과하지 못하면 이 후보 하나를 수행한다. 슬롯 수·binder·RGB loss를 다시 바꾸는 대신, SW165에서 이미 효과가 확인된 전역 그룹화의 **학습 목적과 inference를 일치**시키는 가설이다. SW155의 pair MSE나 SW161/163의 슬롯 마스크 loss 반복이 아니다.

## 모델과 그대로 두는 항목

Seed1 SourceT encoder/graph/core/12 integration scalars는 모두 frozen. 실제 emitted component spikes에서 production 순서로 계산한 signed component correlations, Q와 native prepare_coupling tuple[0] A만 사용한다. RGB/gamma/theta/GT count는 classifier 입력이 아니다. SourceT QCC의 F={label>0}도 frozen. F 밖은0, F 안은 singleton까지 모두 positive label로 보존한다. 따라서 foregroundIoU는 per-image SW165와 정확히 같아야 한다.

SW164 `foreground_induced_merge`의 4-neighbor adjacency, strict-positive all-cross-pair greedy merge, canonical ties와 가변 K를 유지한다. 새 min8/background 재선택/attachment는 없다. 구현은 sealed solver를 바꾸지 않고 새 pure energy-input solver leaf를 만들며, 기존 operator와 전체 merge log 및 labels의 zero-residual 일치를 증명한다.

원래 signed energy E0_ij = wA_ij logit(Qbar_ij), wA_ij = 1+1024(Aij+Aji)/(2 gain). Qbar의 clipping과 gain은 sealed SW162/164와 동일하다. 공유 pair MLP는 10→32 ReLU→1, 최종 weight/bias=0이다. 대칭 입력은 [component rho 4개, logit(Qbar), log1p(1024 Asym/gain), 두 node의 Q row-mean 평균/절대차, Q row-std 평균/절대차]이다. Row 통계는 full1024, diagonal 제외이며 그 뒤 F를 유도한다. r_ij는 두 방향 평균으로 대칭화하고 Etheta=E0+wA*r, diagonal0. source를 담는 마지막0 합산까지 초기 energy/merge log/labels의 exact parity가 필요하다. 두 학습 arm은 파라미터 shape·초기값·입력·inference solver가 같다.

## 학습 target과 비미분 solver의 명시적 범위

TRAIN modal32 mask만 사용한다. F 안의 각 GT ID별 induced4-connected components를 분리해 feasible partition G*를 만든다. 같은 object라도 F 때문에 끊긴 부분은 별도 target component다. **F 안의 GT-background도 버리지 않고** ID0의 connected components로 target을 만든다; 이들은 inference 때 여전히 positive labels이며 GT로 background에 옮기지 않는다. 이는 fixed support의 한계를 감추지 않는 projected target이고 전체 GT partition과 같다고 주장하지 않는다.

Unordered pairs i<j in F에서 Y*=same feasible component. Positive/negative pair 집합이 모두 있으면 각 집합 총 weight를1/2로, 한 집합만 있으면 그 집합 총 weight를1로 둔다. Empty/singleton F는 두 loss 모두0이고 별도로 집계한다. 임의의 feasible partition G에 대해 D(G,G*)=sum a_ij |Y_G-Y*|, score S_theta(G)=sum Etheta_ij Y_G / Z, Z=max(sum |E0_ij|,1); Z는 frozen source에서 계산해 detach한다.

**Candidate: structured hinge.** Loss-augmented energy Eaug=Etheta+Z*a*(1-2Y*)를 detached CPU greedy solver에 넣는다. 그 결과와 G* 두 개 중 S_theta(G)+D(G,G*)가 큰 것을 Ghat으로 고른다(동률 G*). Loss = S_theta(Ghat)+D(Ghat,G*)-S_theta(G*). Memberships는 detach하고 선택된 두 partition의 score만 live tensor로 재계산한다. 따라서 gradient는 sum (Y_Ghat-Y*)/Z에서 pair MLP로 흐른다. Greedy oracle는 exact maximizer가 아니므로 **근사 structured surrogate**이며 true structured-hinge upper bound나 ARI의 미분이라고 부르지 않는다. G* 포함은 loss>=0을 보장할 뿐 최적 oracle을 보장하지 않는다.

**Matched control: local margin learning.** 같은 target weights로 sum a_ij ReLU(1-(2Y*-1)*Etheta_ij/c), c=max(mean |E0_ij|,1) detached. 같은 pair MLP를 학습하지만 실제 전역 partition을 loss 계산에 쓰지 않는다. Candidate/control 모두 최종 inference는 동일 global greedy partition이다. 이 대조가 새 capacity 자체와 실제 partition-level credit의 효과를 구분한다. Loss 혼합/추가 lambda/온도 sweep은 없다.

## 저비용 cache와 preflight

SW163의 검증된 seed1 frozen SourceT 512 TRAIN actual float32 S/Q/A cache가 source checkpoint, T power2/contrast8, RGB/order/encoder/stats, source-state hashes까지 일치할 때만 read-only 재사용한다. Q는 캐시 actual S의 production positive-clamped product다. Rho 추가 계산은 실제 production centering/normalization 순서로 하며 Q 재구성이 일치해야 한다. 새로운 teacher rollout이나 GT-derived affinity는 없다.

B4 TRAIN cache를 B1과 bitwise 같다고 가정하지 않는다. 원래 cache의 firstB4 replay/A capture proof를 검증하고 B4를 TRAIN arithmetic으로 등록한다. VAL은 SW165와 같은 B1/T1024/settle512/live0이다. Cache 안의 8 TRAIN에서 zero residual의 원래 SW164 partition 정확 재현을 확인한다. 별도 native B1 검증은 conditional admission 때 필요한 최소 4 TRAIN만 사용하며 기존 source hash·Q/A 경로와 연결한다; B4=B1 주장은 하지 않는다. GT는 TRAIN target 생성에만 사용한다.

CPU tiny connected examples에서 feasible target/BG fragments/singleton, loss-augmentation 부호, chosen-partition score의 analytical gradient, nonnegative loss를 확인한다. 두 disposable logical updates에서 초기 forward parity, 첫 final-layer update, 두 번째 hidden-layer credit 및 실제 changed weights를 확인한다. 일부 이미지의 hinge0는 정상이다. 모든 eligible TRAIN image에서 candidate gradient가0이면 actionable signal이 없으므로 중지하고 계수 변경으로 통과시키지 않는다. 16 TRAIN source examples로 CPU solver 시간/메모리를 측정한 뒤 전체 비용을 공개한다. 새1024-step rollout cache 수집 없이 시작하는 head-only 실험이다.

## 고정 pilot와 판정

두 arm 모두 같은 512 TRAIN, 8 complete passes=256 B16 updates, fixed shuffle167001/head167001, Adam LR3e-4, global clip1, fresh optimizer. 최종256 checkpoint 하나만 평가하고 validation checkpoint 선택은 없다. GPU micro4로 energy/MLP를 계산하고 CPU oracle는 detached energy만 받는다. optimizer/RNG/순서 포함 logical-update checkpoint를 사용한다. 예측 입력과 provenance를 freeze한 뒤 두 arm/SW165/SourceT/Slot의 fixed320 GT를 연다. Scene K/GT count/VAL oracle는 inference에 없다. Reserve90640..90959는 읽지 않는다.

Seed1 candidate FG가 SW164 seed1 및 matched local-control보다 모두 높고 두 paired95% CI lower>0, objectIoU>=SW164, foregroundIoU per-image exact일 때만 seed0/2 복제를 검토한다. CI는 common320 image indices/10000 draws/RNG167이며 seed-population CI가 아니다. 통과 실패 시 fixed recipe를 닫고 margin/MLP width/loss mix/seed sweep을 하지 않는다. 통과해도 .87, unsupervised 승리, all-gate causality, fresh native32 70k 또는 data scaling 증명이 아니다. 실제 full70k 확장은 all3 재현과 별도 supervised-versus-unsupervised claim 검토 후 새 등록으로만 진행한다.

현재 출력물은 이 DESIGN 하나다. 이후 생성되는 cache/prediction/checkpoint는 `collaborative_testing/prepare_output.ps1`로 중앙 output junction을 먼저 준비하고 provenance archive로 연결한다. 이 문서는 현재 SW163 실행이나 sealed165/166 source를 바꾸지 않는다.

````

</details>

## EVALUATION_QUEUE.md

<details>
<summary>EVALUATION_QUEUE.md</summary>

Source SHA256: `d6da150af7af6cadcd9404582eb3580cd02831ba47978394d3c24a8a8c76e7a2`

````markdown
# Automatic evaluation handoff, 2026-10-15

Two dependency queues are registered while the original two training workers continue unchanged. Prediction supervisor PID3653194 is live with exact module `collaborative_test.SW_0167_structured_relation_energy.prediction_queue`, waiting for both final 256-update artifacts. CPU score supervisor PID3653228 is live with exact `score_queue` module, waiting for validated frozen predictions. The actual prediction-to-score process-module handoff passed before the score queue was launched.

All eleven evaluator/queue checks passed locally and on the server with zero skips. Independent Sol5.6 review found no concrete runtime, mathematical or schema blocker. The actual training-dependency CPU smoke confirmed that both upstream training supervisors were live, that training was still incomplete, and that CUDA remained uninitialized. No VAL images or masks were read during queue registration.

After both endpoints freeze, inference uses a single shared native-B1 Source-T rollout per fixed320 image: T1024, settle512 and live-tail0. Both heads receive the same actual spike-derived features, actual source Q/A and gain. Foreground support remains the production CPU-QCC support. The fresh QCC control must reproduce archived Source-T; the distinct zero-residual global partition must reproduce SW164. These are separate controls. Every candidate label up to1024 remains valid; no object-count cap or GT count is used. Source identity uses the SW161 state-digest family before and after inference.

Both candidate predictions, the Source-T control and all reference hashes must freeze before canonical GT masks are opened. Metrics use the existing native32 contract. SW164's comparator score is pinned to SHA256 `978a9d97658837eb19f414a7fb7dbb35d86c6135cb1213e12e63e2837fd65df6`; its metric vectors and means are independently recomputed from its frozen labels under the same GT. Repeated legacy full-score validation is unnecessary. Final score validation independently recomputes the full report and paired intervals.

The preregistered expansion gate requires structured FG-ARI above SW164 seed1 and above the matched local-margin arm, with positive lower paired95% bounds against both; object IoU must not decrease relative to SW164, and foreground IoU must remain exactly equal per image. Bootstrap uses common320 indices,10000 draws and seed167. This pilot does not establish all-seed improvement, FG-ARI .87, physical-gate causality or70k scaling. SW165 remains incumbent until stronger verified results exist.

Evidence: `results_archive/evaluation_queue_registered_20261015/server_records.json`. Queue runtimes: `results_archive/prediction_queue_20261015/` and `results_archive/score_queue_20261015/`. No inference or scoring restart was issued.

````

</details>

## GPU_ARITHMETIC_RESULT.md

<details>
<summary>GPU_ARITHMETIC_RESULT.md</summary>

Source SHA256: `a690298e0d496f47f3935f84044f3e5a2c3b2d8baa9ad828b65e2975c041b59f`

````markdown
# GPU arithmetic check and feature-mode decision

The owner-guarded GPU1 probe completed on the same first eight registered TRAIN cache rows. Four CPU checks passed before launch; source/order/row hashes were verified under an exclusive lease. No source rollout, mask/validation read or optimizer update occurred.

| Feature consistency comparison | Rows passing unchanged 2e-6 guard |
|---|---:|
| Per-image GPU rho against cached GPU-B4 Q | 0/8 |
| Per-image GPU rho against recomputed GPU-B4 Q | 0/8 |
| Per-image GPU rho against same-mode GPU-B1 Q | 8/8 |

Recomputed GPU-B4 Q matched cached Q exactly on rows4–7; rows0–3 differed by at most 3.04e-6. GPU-B1 Q differed from cached Q by about 5.36e-6–6.20e-6. All comparisons had zero threshold-edge disagreements at0.50. This does **not** justify replacing cached Q: the logit-weighted baseline energy changed by as much as49.1614 when computed from B1 Q. The first cache-generation batch and later batches also use different tensor construction paths; that observation is not a proven causal explanation of the numerical differences, and no further stride exploration is needed for this pilot.

Sol6.1 reviewed the actual implementation and report and approved an explicit separation of arithmetic contracts. Keep cached TRAIN Q/A/gain and baseline E0/wA unchanged. Derive rho from the identical saved component spikes using a declared GPU-B1 mode. A new feature-building API combines these independently provenance-validated inputs without asserting cross-mode bitwise factorization. The existing same-mode wrapper and2e-6 check stay unchanged. Both loss arms use identical feature bytes and modes. TRAIN foreground remains production CPU-B4 QCC from saved spikes; VAL uses actual native-B1 source output and its existing CPU-QCC readout.

The subsequent actual eight-row TRAIN/GPU admission completed all required checks: new-builder parity with the old wrapper on same-mode inputs, source/cache/order hashes, exact corrected-E0/source-support/merge-log/label parity against the cached-Q baseline at zero residual, and two-update gradient credit for both objectives. See `ADMISSION_RESULT.md`. Neither this arithmetic probe nor disposable admission updates are performance evidence. The scientific choice of SW167 follows SW163's failed registered gate, not favorable numeric variants.

Evidence and ownership receipts: `results_archive/gpu_arithmetic_diagnosis_20261015/`. The completed probe code remains immutable; later admission uses a separate entrypoint.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `87ca873621b0a3b72176f314968d91c44f03859a6bf003f2c63533703ea889a0`

````markdown
# SW0167 결과: 분할 loss 학습은 성능 개선으로 이어지지 않음

2026-10-15. 고정320장, native32, seed1 평가를 완료했다. 두 학습은 동일한512 TRAIN 이미지·초기값·순서·385-parameter pair MLP를 사용하고 loss만 다르다. 각256 Adam 업데이트와4096 노출, 모든 parameter의 실제 optimizer step256, 최종 head와 durable boundary 일치를 검증했다. 두 학습의 binding은 arm 이름 외에 정확히 같았다. Source encoder·graph·core는 학습하지 않은 **supervised head-only 실험**이다.

| 방식 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| SW164 / SW165의 동일 seed1 설정 | .770932099 | .614993430 | **.630523482** |
| Structured partition hinge | .770932717 | .614993430 | .627593144 |
| Matched local margin | **.771162710** | .614993430 | .629836618 |
| 동일 rollout의 Source-T QCC | .755273083 | .614993430 | .579513786 |
| matched Slot70k seed1 | .841819292 | .203416709 | .222115156 |

Structured FG-ARI의 기존 partition 대비 차이는+.000000618, paired95%CI[-.000540090,+.000509511]이다. Local margin은+.000230612, CI[-.000119001,+.000637749]다. Structured-minus-local은-.000229994, CI[-.000622098,+.000042253]다. CI는 고정320 이미지의 common paired bootstrap10000회/RNG167이며 seed 모집단의 CI가 아니다. Foreground IoU는 모든 이미지에서 Source-T와 정확히 같았지만 두 새 head의 Object IoU는 기존 partition보다 낮았다. **등록한 expansion gate 실패: 추가 seed·70k 확장이나 이 recipe의 loss/width/계수 sweep을 하지 않는다.**

학습 자체가 멈춘 결과는 아니다. Epoch1→8 평균 TRAIN loss는 structured .293436→.267798, local .551978→.490236으로 감소했고 모든256 업데이트에서 gradient norm이 양수였다. 서로 다른 loss 값의 크기를 성능 점수로 비교하지 않는다.

실제 분할도 일부 변했다. Label 번호에 무관한 pair-membership 비교에서 structured는54/320 이미지, local은43/320 이미지에서 기존 partition과 달랐다. 평균 unordered-pair disagreement는 각각 .001828970/.000423163이다. 다만 FG-ARI는 각각292/298 이미지에서 같았다. 이것은 관측된 출력 변화량이며 encoder가 원인이라는 증명이나 단순 학습 부족의 증거는 아니다.

두 head와 reference를 먼저 고정한 뒤에만 VAL GT를 읽었다. 모든320 이미지에서 fresh Source-T QCC와 별도의 zero-residual SW164 partition 재현을 확인했고, source state는 전후 동일한 SW161 digest family로 검증했다. Prediction/score 큐는 모두 정상 완료했다. 공개 `validate_score()`가 전체 metric·paired CI·gate를 독립 재계산해 통과했고, root가 다시 실행한 검증도 같은 score SHA를 재현했다: `714deb2ab8a0c944d5b5fbedfe09b568433de54c959b5d486414e0629dcb4ab4`.

**최고 모델은 SW165 유지:** 세 seed 평균 .804298915/.609975448/.651148975. 위 seed1 pilot과 이 세-seed 평균을 섞어 성능 향상으로 해석하지 않는다. .87, physical-gate 기여 입증, data/70k scaling은 아직 해결하지 못했다. 다음 후보는 이 frozen-source readout을 세부 조정하기보다 encoder/graph/native core까지 실제로 학습할 수 있는 다른 가설을 별도 설계로 검토한다.

근거: `results_archive/completed_training_20261015/`, `results_archive/completed_score_20261015/server_records.json`, `independent_verification.json`.

````

</details>

## TRAINING_LAUNCH.md

<details>
<summary>TRAINING_LAUNCH.md</summary>

Source SHA256: `eea048cf7ceec87dfdd58fa720a275868855881c07fc33d32f9b434715ad7c6c`

````markdown
# Fixed two-objective training queue, 2026-10-15

After the completed real-eight GPU admission, the actual full 512-image CPU preparation passed with CUDA uninitialized. Both full caches, ordered TRAIN IDs, frozen source checkpoint, original-B4 CPU source foreground, feasible targets and first-eight receipt parity were verified. No VAL images or masks were used. Five focused trainer/queue CPU tests also passed on the server; Sol5.6 independently reviewed the final trainer and queue.

Two independent owner-guarded queues are registered, using only exclusively idle GPUs:

| Objective | Supervisor PID | Worker PID | Leased physical GPU |
|---|---:|---:|---:|
| Structured partition | 3648428 | 3648536 | 2 |
| Matched local margin | 3648429 | 3648525 | 1 |

At the archived launch observation the worker processes were alive and preparing their inputs before CUDA allocation; **an optimizer update was not yet verified**. Queue status alone is insufficient to claim GPU training progress. Later durable boundaries and actual GPU ownership provide that evidence. Foreign GPU0 PID2653200 was left running.

Subsequent direct observation confirmed actual CUDA training: structured worker3648536 owned GPU2 (842MiB) with durable update2; local-margin worker3648525 owned GPU1 (838MiB) with durable update4. Both exact registered worker command lines were live, both supervisors remained running, and no error was recorded. These counts are a start receipt, not a final endpoint or a model score. Evidence: `results_archive/actual_gpu_training_started_20261015/server_records.json`.

Both arms use the same 512 TRAIN images, fixed eight-pass order, identical fresh head initialization, 256 logical B16 Adam updates with micro4, learning rate3e-4 and gradient clip1. Only the loss differs. Every completed update has an atomic resumable boundary. Final validation checks exact final head/boundary agreement and all four Adam parameter step counters at256. Neither encoder nor physical source gates are optimized in this head-only supervised pilot.

Both final endpoints must freeze before shared-rollout fixed320 evaluation. Compare the structured candidate against both SW164 seed1 and the matched local control, using the registered paired confidence intervals and exact foreground preservation gate. This launch is not evidence of an improvement, data scaling or gate causality; SW165 remains incumbent.

Trainer SHA256: `3c69d9acb505bf635f72309505c1bf9fe3d951791ddeafaa22e813cda6789642`. Queue SHA256: `0359d2ff800157dd2313f604b9fdf908cd31a89e5559199669b12a049f14126a`.

Evidence: `results_archive/training_launch_20261015/server_records.json`. Server runtime: `results_archive/training_queue_20261015/` and `results_archive/training_20261015/`.

````

</details>
