# SW_0139_native32_representation

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## causal_stage_review.md

<details>
<summary>causal_stage_review.md</summary>

Source SHA256: `879b043e58482d85e3acd1fa63198e98dc42576a916ddec17e720f58bfd011ea`

````markdown
# Prospective native32 causal stage screen

Sol6.1 code/design review, 2026-10-10. This document records the design; implementation and execution are recorded separately. Complete all existing SW0139 arms unchanged before considering further training. SW0138 compared readout inputs on fixed trajectories: actual means `.78191867/.60542713/.59701381`, retained-gate `.78188260/.60577083/.59658259`; FG interval includes zero. This does not remove the event selector from the recurrent dynamics or establish stage necessity.

## Actual architecture, not five interchangeable gates

The native32 foundation uses live image-conditioned graph, static gamma drive and factorized Kuramoto coupling, with no graph feedback or spike pulse. Kuramoto contributes an interaction term `(K/N)*interaction` to Euler phase integration; it is not a scalar gate. Raw sinusoidal gating produces per-component carrier `sin(theta_t)*g_t` and delayed common gate `g_t=.5*(1+sin(mean_D(theta_{t-2})))`.

`SW0134/rollout.py` folds rows in image/component order `b*4+d`, uses carrier in the shared dendritic projection with the **previous emitted spike**, applies dendritic retention/integration, membrane voltage retention/integration/reset, then emits `S_t=1[m_t>v_d]*g_t`. It stores emitted S as the next previous_spike, not the unscaled binary event. Gate holds occur only at exactly g=0; near-zero is not zero. Source integration coefficients a_D/a_M/b are zero, and SW0139 freezes them and all native parameters other than graph. Thus gradients through downstream operations do not mean those native parameters are being trained or their history is necessary.

## Local intervention sites and exact meanings

Use an isolated instance-local diagnostic rollout with an intervention enum; production files/checkpoints stay untouched. Its NONE path must reproduce the existing native32 reference on identical inputs/layout. Execute each intervention from t=0 through1023, **not just the settled output band**. Recompute all descendants and carry changed states forward; retain the original graph/drive/initial theta for each condition.

| Condition | Exact local change | What it tests |
|---|---|---|
| Kuramoto interaction off | In `kuramoto_layer.forward`, replace only coupling_term with zero before `theta_dot=omega_eff+coupling_term+drive_term`; keep drive/kappa/omega/dt/init unchanged. An isolated K=0 instance is equivalent for this static/no-pulse source. | Need for interpatch interaction at this checkpoint, not necessity of every oscillator operation. |
| Delayed gate constant | For the registered raw mode, replace g_t by.5 and carrier by `sin(theta_t)*.5`; use this same replacement in folded gate, integration u, emission and exact-zero hold test. | Combined delayed phase-conditioned modulation, preserving current-phase carrier. No mixture of native carrier gate and replacement emission gate. |
| Dendritic h-history off | Define `j_D=(1-beta_D)*dense` and replace h by `(1+a_D*(u-1))*j_D`; remove every explicit previous_h contribution while keeping beta_D's current-input gain and a_D's current-input modulation. Keep dense's previous-spike input and its sum-to-membrane operation. | Dendritic self-history rather than a gain-changing beta=0 replacement; previous-spike input remains, so this is not removal of the whole dendritic layer or every history path. |
| Membrane voltage-history off | Define `j_M=(1-alpha)*R_m*sum(h)-v_d*S_prev` and candidate `(1+a_M*(u-1))*j_M`; remove every explicit previous_mem contribution while preserving current-input gain and spike reset. Apply existing `where(g==0,previous_mem,candidate)` hold afterwards. | Voltage-history paths apart from the separately retained exact-zero hold. Not a gain-changing alpha=0 replacement or complete membrane bypass. |
| Spike reset off | Delete only `-v_d*S_prev` from the membrane candidate; retain alpha/a_M, threshold, holds, dendrite input and event selection. | Membrane reset feedback, distinct from voltage retention. |
| Binary events forced on | Compute normal h/m/threshold, replace the event result with ones **before** `S=events*g`, and assign that S to membrane.spike for subsequent dense-input/reset updates. | Closed-loop event selection contribution. Different from SW0138's fixed-trajectory head-input replacement; membrane state can still be computed but cannot select output. |

All conditions keep v_d=`.06*exp(b_d)` except that selection is bypassed in the final condition. Thresholds, taus and coefficients are never recalibrated to make an ablation look useful. No θ/gamma mask readout or GT teacher is introduced. Retention-off arms isolate named memory paths; a harmful effect cannot be advertised as proof that the complete dendrite/membrane architecture is indispensable.

The history formulas are exact path knockouts. Native h can be written `(1+a_D*(u-1))*j_D + [beta_D*(1+a_D*(u-1))-a_D*(u-1)]*previous_h`; membrane has the corresponding expression with alpha and j_M before holds. Delete only the latter history coefficient contribution. For source97/139, a_D/a_M/b are identically zero, so these reduce to h=`(1-beta_D)*dense` and pre-hold m=`(1-alpha)*R_m*sum(h)-v_d*S_prev`. No result from this source screen can establish usefulness of introduced a_D/a_M scalars. This differs from a legacy 'set beta/alpha to zero' ablation that also changes feed-through amplitude, and must be labeled accordingly.

## State and screening semantics

Reset theta from the registered same gamma initialization, h/m/S to zero and delayed history to an empty list for every image/condition; use the exact batch-major component folding and production stack/permute/reshape order. Do not initialize an ablation with the baseline's settled state. Freeze all parameters, use T1024/settle512/batch1/native1024 features, and capture literal events from their forward decision, gate and emitted S. No global autograd-function monkeypatch or division-by-near-zero reconstruction of events.

First inspect the same first4 TRAIN images per seed, all three source97 seeds. Verify NONE parity at T1024 for all six existing trace families: theta, mean membrane/spikes, component membrane/spikes/gates; preserve production strides/reduction order and the unchanged registered source parity rule. Verify finite states, intended operand replacement, exact `S=E*g` and reset/emission feedback equations. Record phase/gate/dense/h/m/event/S changes, event occupancy/transition counts, exact-zero gate hold counts and recurrence coefficients. Compare earliest differing stage with the declared intervention: downstream interventions must leave theta unchanged in this no-feedback/no-pulse source. This is a real-flow/resource screen, not a mask-precision or arbitrary effect-size eligibility test. An unchanged output after a correctly executed intervention is an informative null, not a reason to invent a new threshold.

Reuse each image's frozen gamma/graph/coupling across conditions when mathematically unchanged, with hashes and independent state reset; do not cache mutable trajectory state. Record peak memory/runtime and use existing global exclusive GPU leases. No N³ unchunked operator, shortened horizon, fewer patches or foreign-owner sharing. Technical errors stop only the affected diagnostic branch and preserve records.

## Fixed endpoint and closure

Freeze the complete prediction arrays before GT and evaluate every technically valid condition on all320 IDs1320..1639, ordinary4px modal/min-ID GT and the unchanged actual-spike QCC `.50/min8/largest BG`. Use all three seeds, common image indices for paired bootstrap and conditionally fixed-seed intervals; report every metric, seed, valid count and null/beneficial ablation. Apply the same screen to a SW0139 learned candidate only after it passes its own preregistered source/control development gate, with checkpoint and preprocessing fixed beforehand. Do not choose an intervention/checkpoint by GT or claim source ablations prove the learned candidate's mechanism.

A frozen ablation's reproducible loss of FG/object quality supports conditional reliance on that operation; severe distribution shift and inability of remaining weights to compensate remain limitations. Neutral retention/reset/event effects refute a demonstrated benefit on this endpoint, while finite surrogate derivatives merely describe backward credit. Sparse events are not themselves object-binding evidence. Coupling/gate collapse cannot rescue unsupported dendrite/membrane claims.

No matched retraining is launched until SW0139 results and this diagnostic are reviewed. A later necessity study must preregister equal-source/data/update full-model and exact ablated-model training, report both frozen and adapted endpoints, and allow the ablated model to compensate. If a downstream path remains neutral or beneficial to remove, close the 'every current gate is useful' claim for this recipe rather than tuning an ablation to obtain the desired result. Final native32 Slot comparison,70k continuation and independently controlled unique-data scaling remain required and are not established by this screen; reserve90640..90959 remains unread.

````

</details>

## partial_failure_review.md

<details>
<summary>partial_failure_review.md</summary>

Source SHA256: `5237a6c71b6f2454ddeeaabf6a995275e4828f052bb41485a7d1d7ad0c7483a7`

````markdown
# SW0139 partial failure review

Sol6.1 design review, 2026-10-10. This document uses completed control/context records only. Cross-view training and SW0140 full320 causal endpoints remain pending; their outcomes are not inferred here. No running recipe, source file, queue or guard is changed.

## Established outcome and implementation scope

The fixed native32 seed1 source FG/foreground-IoU/object-IoU is `.754135/.606900/.582201`. Old-loss encoder/graph continuation gives `.726187/.592371/.557006`; context residual gives `.666608/.650807/.477581`. Control-minus-source FG is `−.027948`, paired CI `[−.039005,−.017387]`. Context-minus-control FG is `−.059579`, CI `[−.077038,−.042826]`; object-IoU also falls, while foreground-IoU increases. Close this context recipe: no other-seed expansion, dilation/LR refinements or 70k continuation.

Both completed histories have the same 4096 ordered images, 256 logicalB16/B4×4 updates and native1024/settle512/tail64 rollout. Actual `trainable_backbone` releases encoder and graph only; context adds its zero-output multiscale residual. Native drive/Kuramoto/dendrite/membrane and all 12 integration/threshold scalars remain frozen. Therefore inputs and learned coupling can drift substantially while downstream state parameters remain unchanged. These results do not establish a defective membrane learning rule or a useful event selector.

Evaluation regenerates gamma from each trained encoder plus its context module, then uses actual spikes and unchanged QCC. It is not scoring an auxiliary RGB head. Residual features are clamped to the registered range before pool32. The completed numerical/lineage checks establish a faithfully evaluated failed recipe, not a demonstrated implementation mistake.

## What explains the pattern, and what does not

The implemented objective is phase criterion plus 5×actual positive-clamped product-Q criterion. Each criterion applies bimodality6, density balance10 toward `.867`, collapse barrier1 and coherence1. The coherence term smooths each node's **mean synchrony with all other nodes** across the grid; it does not directly require correct object boundaries or within-instance relations. These global statistics allow many distinct partitions. Objective/instance mismatch is therefore plausible, but density alone is not identified as the cause, and no target should be chosen from GT.

Descriptively, first32/last32 old-loss means are `26.190→25.064` for control and `27.737→18.622` for context. The batches differ, so these are not paired loss improvements. Nonetheless endpoint degradation alongside strong progress through that objective provides no reason to enlarge the same context mechanism. Higher foreground IoU with worse FG-ARI/object IoU warns of partition errors; merge versus split versus background reassignment must come from Luna's pending CPU diagnosis, not this metric pattern alone.

Every update clips the unique parameter union at1. Median preclip norms are `5.486/7.602`, multipliers `.1823/.1315`. Context adds parameter capacity, a different LR and a gradient direction to that union, so its comparison combines representational change and optimization effects. These measurements neither locate family-specific credit nor prove clipping caused the failure. With Adam, those multipliers are not simply effective LR ratios. No clip/LR retry is justified by the partial results.

## Conditional next priority

First retain SW0137 as incumbent and finish cross-view and all six SW0140 interventions on all three seeds. No new training admission now. Do not duplicate the requested CPU partition diagnosis or inspect reserved IDs.

If cross-view preserves source FG and provides credible paired benefit, and SW0140 shows native downstream history/events provide no mask benefit, the next mechanism question is the already documented **task event-credit versus terminal event-detach pair**, releasing only native membrane retention and component thresholds in addition to encoder/graph. Keep actual1024 spikes, native recurrence and QCC; require literal E×g forward parity, finite selector credit and actual event-decision changes on fixed TRAIN audits. A gradient or lower event rate cannot establish useful gating.

If cross-view also degrades source, do not promote the same objective to more capacity or release downstream selectors automatically. The next broad objective question should instead preserve source binding while learning view stability: a fresh prospective comparison of the completed old+C continuation against **C plus detached frozen-source actual-Q preservation**, replacing the global old-prior optimization as a single declared objective change. This preserves a numerical incumbent relation, not a claim that source relations are true object labels; SW0128's failed pseudo-instance precision remains relevant. No confidence-threshold teacher, GT tuning or RGB decoder is introduced. Exact weighting and budget would require registration after the pending outcomes, not selection now.

Any eventual candidate must improve the original native32 source and a matched control before all-seed/70k expansion. Fixed320 modal4px metrics, actual-spike masks and comparable Slot scoring remain unchanged. Conditional seed1 evidence is not all-gate necessity, three-seed superiority or positive unique-data scaling; the source already inherits70k training.

````

</details>

## partition_failure_analysis.md

<details>
<summary>partition_failure_analysis.md</summary>

Source SHA256: `cf34e7c8da0b07304f641dcf723f47900296e16e46269ca85c589953c2014b49`

````markdown
# SW0139 seed 1 partition-error analysis

This read-only, post-hoc decomposition covers the same 320 validation images (IDs 1320–1639) and the frozen source97, control, and context_residual predictions. It used the saved predictions; it did not rerun inference or alter training. The source evaluation and prediction hashes are recorded in the [diagnostic result](results_archive/partition_diagnostic_seed1.json) (SHA-256 `27d15a4d2a225cdf37a6bf81f1923e76f46919e3c82672b835957ddebdde4ab0`).

| Arm | False-merge pairs / all true-FG pairs | False-split pairs / all true-FG pairs | Missed true-FG patches | Predicted nonzero labels | True objects | BG-column merge pairs / all true-FG pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| source97 | 0.047292 | 0.046832 | 0.130703 | 5.815625 | 6.312500 | 0.017335 |
| control | 0.054390 | 0.051094 | 0.155309 | 5.687500 | 6.312500 | 0.022439 |
| context_residual | 0.086560 | 0.047700 | 0.256129 | 4.503125 | 6.312500 | 0.055274 |

The reported pair fractions are means of per-image ratios over true-foreground patch pairs. They are not fractions of all image patches. False merges count pairs of true-foreground patches from different true objects that share a predicted label; false splits count pairs from the same true object assigned different predicted labels. The background-column merge fraction is the subset of false-merge pairs assigned predicted label 0. Predicted-label counts are image means; the nonzero-label count is measured on the full prediction, while the true-object count comes from the target.

Relative to source97, context_residual increases the false-merge fraction by 0.039268, of which 0.037939 is in the predicted-background column (96.6%). Its false-split fraction changes by only 0.000868, while the missed-foreground fraction rises by 0.125426 and the mean number of nonzero predicted labels falls by 1.312500. This points to foreground loss and background-label merging as the main observed error pattern, rather than a comparable increase in object fragmentation. Control shows smaller increases in both merge and split fractions and a missed-foreground increase of 0.024606 relative to source97.

This is a descriptive decomposition on one seed, not a causal-layer result or a model-promotion test. It does not establish gating or scaling, and it does not change the source97 incumbent. Use it only to inform a separately preregistered next hypothesis after the SW0140 endpoint is complete; do not change the closed SW0139 recipes based on this post-hoc analysis.
````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `fdd984e6ae8a71692dc152b862e273d21c3122670bccaf7ccacbd701af4fe2d5`

````markdown
# SW0139 native32 representation pilot

Compare old-loss control, trainable context residual encoder, and cross-view spike consistency. All arms use genuine 32x32 patches, the same ordered4096 TRAIN images and256 logical updates. Encoder and learned graph train together; oscillator, dendrite, membrane and integration parameters are frozen. No TRAIN masks, object counts or learned assignment head. Cross-view doubles view exposures and is not FLOP matched.

Local/server CPU tests18/18, actual source RGB gamma parity and optimizer groups verified. Supervisor2711934 and three GPU disposable preflight/calibration workers verified live. Fresh training follows successful disposable preflights; fixed320 endpoint predictions precede GT scoring. This is a seed1 pilot, not fresh70000-image training or evidence of gating necessity. Publication remains pending scope approval.

Control training PID2712399 GPU1 and context training PID2712528 GPU2 subsequently verified live after disposable preflights passed. Cross-view calibration remains running, its training queued. See training_start_20261010.json.

Control and context_residual seed1 training completed and checkpoint/optimizer/history hashes verified. Archived histories independently confirm the same4096 unique ordered images,256 B16 updates/B4x4 and one pass without GT. Both mask evaluations have started; cross_view training continues. Actual run times were5127.19s/5152.43s, about20.03s/20.13s per logical update. The roughly24.34h/24.46h70k one-pass projections are extrapolations only. All256 updates in both arms activated global clip1; median clip multipliers.1823/.1315. This is a potential optimization diagnostic, not a causal explanation or task result. See results_archive/completed_control_context_training_analysis.json.

Completed control/context predictions were frozen and scored with the registered evaluator in an additional CUDA-hidden CPU process2859188; final all-arm evaluation path is preserved for the queue. Per-image source metrics exactly reproduceSW0137; means and paired10k image intervals were independently recomputed. Seed1 FG/foregroundIoU/objectIoU: source.754135/.606900/.582201; control.726187/.592371/.557006; context.666608/.650807/.477581. Context minus source FG is-.087528 (95%[-.104987,-.070645]); minus control-.059579 (95%[-.077038,-.042826]). Better foreground coverage accompanies worse object assignment. The registered context expansion gate failed; do not expand this fixed recipe to other seeds or70k. Cross_view remains pending, not failed. No incumbent change or three-seed superiority claim. See results_archive/control_context_analysis.json and evaluation_seed1/evaluation_control_context.json.


## Seed1 partition-error diagnostic

The completed control/context output was decomposed on true-foreground patch pairs. The context arm shows a larger background-label merge component and higher missed-foreground fraction; see [the analysis](partition_failure_analysis.md) and the hash-bound [diagnostic JSON](results_archive/partition_diagnostic_seed1.json). This is descriptive and does not change the source97 incumbent or establish a causal layer claim.


## Completed all-arm seed1 endpoint

All11 queue tasks completed successfully; terminal state/log snapshots and all3 prediction protocols are archived. Fixed320 per-image means and common paired10k bootstrap intervals were independently reproduced.

| Arm | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| control | 0.726187 | 0.592371 | 0.557006 |
| context_residual | 0.666608 | 0.650807 | 0.477581 |
| cross_view | 0.738505 | 0.592047 | 0.566686 |
| source97 | 0.754135 | 0.606900 | 0.582201 |

Cross-view improves FG over old-loss control by.012319 (paired95% CI[.004287,.021231]) but remains below source by.015630 (CI[-.024289,-.007376]). Both registered candidate expansion gates fail. Close these exact recipes without seed/70k expansion; incumbent staysSW0137. The conditional C-only/C+source-Q objective pilot is now justified for preparation, with actual integration/preflight required before training. No new pilot has run. This endpoint is one-seed continuation, not fresh70k native32 training or a scaling proof.

````

</details>

## results_archive/partition_diagnostic_code/README.md

<details>
<summary>README.md</summary>

Source SHA256: `27fb92478c50b8afd4017591fd73e0b6531557687dc06eedcd03429a82e15a62`

````markdown
# SW0139 seed1 partition-error diagnostic

This is a byte-identical archive of the read-only diagnostic and its local tests. It decomposes partition errors for the completed seed1 SW0139 control and context-residual predictions against source97. It does not change prediction or scoring contracts and does not establish a causal-layer claim.

On the server, run with the repository path explicitly set (especially when feeding the script on stdin):

```sh
PATCH_V2_REPO=/Data0/kevinswk/patch_v2_sw python - --output /Data0/kevinswk/patch_v2_sw/collaborative_test/SW_0139_native32_representation/results_archive/partition_diagnostic_seed1.json < /path/to/diagnose_repr32_partition.py
```

The diagnostic validates the SW0139 partial-complete report, the control/context prediction protocols and hashes, SW0139 training/checkpoint provenance under `trained_models/SW0139_native32_representation`, and the source97 fixed320 prediction before it opens the HDF5 mask slice for IDs 1320 through 1639. It then checks the saved production metrics against a native32 replay and writes one create-only JSON. It forces `CUDA_VISIBLE_DEVICES` empty and sets OMP/MKL thread counts to one.

The collector `collect_repr32_partition.py` in the private deployment folder reads that one JSON, checks its fixed seed1/320/32x32/GT-scope schema, and emits a SHA256-bound base64 transfer payload.

````

</details>

## results_archive/source_preservation_preparation/README.md

<details>
<summary>README.md</summary>

Source SHA256: `18d466cc1c093d01eccb1be7518eed1647de4f2ad4afa15bab2fe92265608dbe`

````markdown
# Source preservation helper preparation

Prepared locally on 2026-10-10 from the conditional [design](../../source_preservation_design.md). This is a loss and provenance-validation library, not a registered experiment or a complete training implementation. No teacher cache, rollout, calibration probe, server deployment or new GPU training has run.

The helper computes float32 off-diagonal actual-Q MSE with a detached teacher. CPU fixtures check exact values, zero identity loss, live student gradients, unchanged inputs, malformed affinities, cache identity and grouping, and the four-batch TRAIN-only calibration rule. Actual checkpoint-to-cache and rollout-to-loss integration remain unverified.

Root independently reproduced 5/5 tests with CUDA hidden, OMP_NUM_THREADS=1, and C:/Users/SangWoo Kim/anaconda3/python.exe; import numpy before unittest loads the test module. The default local Python failed with duplicate OpenMP initialization; no KMP_DUPLICATE_LIB_OK workaround was used. Both implementations remain identical to the tested private originals.

- source_preservation_loss.py SHA256: 28d6fd98bc7be9d57b7a72a8af9d6bb76e063eb88a8ffbf02291e807d1f8ea0a
- test_source_preservation_loss.py SHA256: 12708f91c163120411f2fe1abdd8b6f85d5d59525ab20942c0b6cc673a7ad962

Admission and full implementation depend on the pending SW0139 cross-view and SW0140 causal endpoint results. The incumbent and all running protocols are unchanged.

````

</details>

## source_preservation_design.md

<details>
<summary>source_preservation_design.md</summary>

Source SHA256: `76410b32e7520a49483c52ef34b7700edf86337291635550e95ec3e933d17dcc`

````markdown
# 실제 spike 관계 보존 + cross-view 목적함수: 조건부 설계

2026-10-10, Sol6.1 검토. 실행 등록이나 실험번호 할당이 아니다. SW0139 cross_view 및 SW0140의 all3/full320 결과를 먼저 완료한다. 현 실험·guard·queue는 변경하지 않는다.

## 근거와 질문

완료된 partition 진단에서 context의 false-merge 증가 .039268 중 .037939(96.6%)는 predicted BG column이다. Missed FG는 .130703→.256129, false-split 증가는 .000868이다. 따라서 전경을 BG에 흡수하는 관계 drift가 주요 관찰 패턴이다. 이것이 density prior나 clipping의 단독 인과효과라는 뜻은 아니다.

다음 질문은 **기존 관계를 훼손하는 통계적 목적 최적화 대신, 원본 spike 관계를 보존하며 view 안정성을 학습할 수 있는가**이다. Source는 정답 teacher가 아니다. SW0128의 실패한 binary pseudo-instance precision을 우회하거나 재해석하지 않는다.

## 동일 모델의 최소 대조

동일 seed1 SW0137 native32/source97 strict-mapped core와 registered encoder/statistics에서 시작한다. Encoder와 graph만 trainable; native dynamics와 12 scalars는 고정한다. Context/decoder/classifier 추가 없이 실제 [B,4,1024,1024] spike trace를 사용한다.

두 새 arm은 모두 old phase+5Q 통계적 목적을 **완전히 제거**한다.

- C-only: `L=λ_C C`.
- Source-preserving: `L=λ_C C+λ_A A`.

완료된 SW0139 old+λ_C C를 같은 seed/order/optimizer/augmentation의 기존 reference로 별도 비교한다. C-only↔C+A는 anchor의 추가효과, old+C↔C-only는 phase 및 Q priors 전체 교체의 효과를 구분한다. 이 비교로 balance/density 하나의 원인이나 phase prior만의 효과를 주장할 수 없다. 신규 두 arm을 old+C 하나와만 비교하면 삭제와 anchor 추가가 섞이므로 충분한 인과 대조가 아니다.

C는 실제 SW0139 helper 그대로: settled512의 raw 4×512 spike를 flatten/L2 normalize, 같은 patch의 밝기·contrast 변형 view를 positive로 하고 B4 내 다른 image patch32개만 negative, symmetric InfoNCE τ=.1. Augmentation/negative RNG nonce도 원래 fixed order에 결박한다. 같은 image의 다른 object를 negative라고 가정하지 않는다. 단, 이 C 자체가 instance 감독을 제공하는 것은 아니다.

## Soft numerical anchor와 teacher 비용

원본 frozen same-seed encoder/graph/core의 **canonical RGB** rollout에서 production `spike_synchrony_affinity(..., settle=512, affinity_mode='spike')`로 `Q_T`를 만든다. 각 component correlation의 positive clamp와 product를 정확히 유지한다. Teacher에는 GT, threshold, Hungarian label, H/group/BG 판정이 없다.

학생도 같은 canonical RGB에서 `Q_S`를 계산하고,

`A = mean_images [ sum_{i≠j}(Q_S[i,j]-stopgrad(Q_T[i,j]))² / (1024×1023) ]`.

대칭 양방향을 그대로 평균하고 diagonal은 제외한다. Augmented view는 C에만 사용한다. Uniform pair 평균은 source의 희소한 FG 관계를 약하게 반영할 수 있고, 작은 MSE가 .5 CC의 bridge 안정성을 보장하지도 않는다. 따라서 A 감소나 teacher 충실도만으로 object 개선을 선언하지 않는다.

권장 방식은 고정 4096 canonical TRAIN Q를 **float32 전체 matrix**로 한 번 생성하는 것이다: 4096×1024²×4 bytes = 16 GiB payload. Float16/quantization/binary confidence filtering은 사용하지 않는다. 비교할 두 새 arm이 같은 immutable cache를 재사용한다. 생성은 no-grad full1024/512, 실제 B4 및 동일 microbatch grouping으로 수행한다. Native recurrence의 batch-shape 수치 차이를 없애기 위해 B1로 만든 teacher를 B4 학생 reference로 조용히 대체하지 않는다.

두 번째 teacher rollout을 매 update 반복하는 방식은 storage는 줄지만 두 view 학생에 canonical teacher 비용까지 계속 더한다. Shared teacher cache는 두 pilot에서 그 비용을 한 번만 지불하므로 우선한다. Source0 보정용 첫64 teacher Q는 256 MiB이며 별도 source0 provenance를 유지한다. 실제 생성·I/O·학생 two-view update 시간을 먼저 측정하고 비용을 보고한다; 기존 one-view 약20초/logical update를 two-view 비용으로 가정하지 않는다.

Cache ledger는 source core/encoder/statistics/graph 설정 및 helper SHA, RGB bytes/preprocessing, ordered image IDs와 pool rows, B4 grouping, full1024/512 horizon, matrix dtype/shape와 chunk hashes에 결박한다. 각 image의 Q row/column은 동일 native32 row-major patch ordering이다. Teacher/source는 optimizer를 갖거나 학생 checkpoint에서 갱신하지 않는다. 첫 actual batch의 cache↔동일 teacher 재계산과 source 학생 Q parity를 확인한다.

## 정확한 TRAIN-only weight 규칙

`λ_C`는 SW0139 source0 첫4 logical TRAIN batches의 고정 calibration 값과 provenance를 재사용한다. 두 새 arm에서 동일하다. 이는 old gradient를 기준으로 정해진 기존 scale이라는 한계를 기록하고, validation에서 다시 맞추지 않는다.

동일 source 초기에는 A=0, ∇A=0이므로 초기 gradient ratio로 λ_A를 만들면 안 된다. 다음 고정 disposable 절차를 등록한다.

1. Fresh source0와 fresh Adam, encoder3e-6/graph3e-5, logicalB16/B4×4, unique-union clip1로 **첫 TRAIN B16의 λ_C C만 1회 update**한다. 실제 objective와 동일 RNG를 사용한다.
2. 이 고정 probe 상태에서 첫4 logical TRAIN batches를 재평가한다. 각 batch의 B4 gradient vectors를 1/4씩 먼저 누적한 뒤 encoder+graph 전체 norm을 계산한다.
3. `λ_A = median_batches( ||∇(λ_C C)||₂ / ||∇A||₂ )`로 설정한다. 이는 한 source0 C-induced displacement에서 두 목적의 credit scale을 같게 하는 단일 규칙이다. 분모의 zero/nonfinite는 calibration 실패로 기록하며 epsilon으로 통과시키거나 다른 step/seed로 재보정하지 않는다.
4. Probe weights와 Adam은 폐기한다. 두 실제 arm은 원본 source와 fresh Adam에서 시작하고 λ_A는 다른 seed에도 그대로 동결한다. 첫 update에서 anchor gradient가 0인 것은 정상이다. 보정 원본/상태 hash, 각 목적 family norm과 cosine, 최종 λ를 보존한다.

보정 이후 두 목적 gradient는 parameter union에서 합산하여 한 번 clip/Adam한다. λ를 family별로 추가 조정하거나 event occupancy에 맞추지 않는다. 단계2의 credit 검사와 source/cache fidelity가 충분한 preflight이며, 새로운 GT precision guard를 만들지 않는다.

## 예산·종료·논문 한계

Pilot은 기존과 동일 seed1, ordered4096 unique images, 256 logicalB16/B4×4, full1024/settle512/live64, fresh Adam, encoder3e-6/graph3e-5, clip1이다. 두 image views는 8192 view exposures로 공개한다. Train은 GT-free이고 endpoint는 immutable native32 modal4px GT와 actual-spike QCC fixed320이다. 저장된 final checkpoint 하나만 평가한다.

확장을 고려하려면 C+A의 FG가 source 및 C-only를 모두 넘어 common-image paired95% lower bound가 양수여야 한다. 완료 old+C와의 비교 및 세 IoU/ARI 지표도 함께 보고한다. 실패하면 이 고정 recipe를 닫고 λ/teacher threshold/precision/seed 변경으로 반복하지 않는다. 단일 seed 결과로 all3 Slot superiority를 주장하지 않는다.

Source를 복제하는 것만으로 모든 gate의 usefulness나 scaling이 해결되지 않는다. 특히 본 설계는 membrane/event parameters를 학습하지 않는다. SW0140 frozen intervention은 필요성 경고이고 retrained gate 대조와 다르다. 유익한 representation 학습이 확인될 때만 별도 event-credit 대조의 우선순위를 검토한다. 최종 all3 comparable Slot32 세 지표 및 increasing-unique-data 개선은 별도 목표다. 70k lineage continuation을 fresh-data scaling으로 부르지 않으며, 비용과 three-seed 성능을 확인하기 전 full70k 확장하지 않는다. Reserve90640–90959는 읽지 않는다.

````

</details>
