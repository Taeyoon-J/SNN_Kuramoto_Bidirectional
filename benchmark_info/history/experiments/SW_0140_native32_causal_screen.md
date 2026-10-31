# SW_0140_native32_causal_screen

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## causal_endpoint_review.md

<details>
<summary>causal_endpoint_review.md</summary>

Source SHA256: `23e61ce41b1f9d8ffdd3bce249481420fbd711181731e319afbf22da2568f322`

````markdown
# SW0140: 완료된 native32 causal endpoint 검토

2026-10-10, Sol6.1. `causal_endpoint_analysis.json`, `causal_evaluation.json`, prediction protocols 및 실제 `causal_rollout.py`를 검토했다. 고정 source97 all3/320, native1024 patches, full1024/settle512, actual-spike QCC 결과이다. 학습·queue·incumbent를 변경하지 않는다. SW0139 cross_view는 아직 진행 중이다.

## 현재 frozen 모델에서 확인된 기여

| Frozen 조건 | 평균 FG-ARI | Source 대비 FG 차이 | 해석 |
| --- | ---: | ---: | --- |
| Source | .78191867 | — | 현재 incumbent |
| Kuramoto interaction off | .05607437 | −.72584430, CI [−.741371,−.709412] | 현재 결합 상호작용에 강한 의존 |
| Delayed gate=.5 | .00000000 | −.78191867 | 해당 sinusoidal gate 경로의 강한 의존 |
| Dendrite h-history off | .77164959 | −.01026909, CI [−.012928,−.007722] | h-history가 현재 grouping에 기여 |
| Membrane voltage-history off | .78222498 | +.00030630, CI [−.000702,+.001454] | FG 이득 입증 없음 |
| Spike reset off | .78188260 | −.00003608, CI [−.000584,+.000452] | 실질적 FG 이득 입증 없음 |
| Binary events forced on | .78188260 | 위와 동일 | 실질적 FG 이득 입증 없음 |

D-history 제거는 foreground IoU도 −.00800344, object IoU도 −.01635863이며 두 CI 모두 음수다. 짧은 TRAIN trace에서 seed1 효과가 작았어도 실제320 masks에는 평균 손실이 나타났다. 작은 raw trace 차이가 CC에 영향을 줄 수 있으므로 TRAIN effect size로 조건을 걸러서는 안 된다는 점도 확인된다.

M-history 제거의 object IoU는 오히려 +.00145341, CI [+.000465,+.002587]이다. Reset/event 제거는 foreground IoU가 +.00034369, CI [+.000015,+.000741]이고 FG/object CI는 0을 포함한다. 이 작은 이득을 새로운 incumbent나 GT-selected ablation recipe로 채택하지 않는다. 현 결과는 모든 downstream 연산의 성능 기여를 긍정하는 근거가 아니다.

## Intervention의 정확한 범위

K-off는 coupling interaction만 0으로 두고 native drive/다른 경로를 유지한다. Gate=.5는 carrier=`sin(theta)*.5`, integration gate, spike emission과 exact-zero hold 조건을 함께 바꾼다. 따라서 **현재 delayed sinusoidal gate 전체 경로**의 유용성 근거이며, delay 길이만의 효과나 각 carrier/integration/emission 역할의 독립 증명은 아니다.

D-history off는 이전 h 항만 제거하면서 현재 dense 입력 gain `(1−sigmoid(tau_n))`을 유지한다. 이전 emitted spike의 dense 입력은 남는다. M-history off도 이전 voltage 항만 제거하고 현재 입력 gain/reset/hold를 유지한다. D-history 결과는 해당 recurrence의 frozen 기여이지, dendritic layer 전체를 제거한 뒤 재학습해도 회복할 수 없다는 뜻이 아니다. Source의 추가 a_D/a_M scalars는 0이므로 그 새 integration 조절의 기여를 이 실험으로 주장할 수 없다.

Forced E=1은 readout factor만 바꾸지 않고 다음 dendrite/reset에 전달되는 emitted S도 바꾼다. Reset-off는 reset 항만 삭제한다. 세 seed 각각 두 조건의 prediction NPZ SHA가 **완전히 동일**하고, 320장의 세 metric per-image arrays도 동일하다. 따라서 저장된 labels/IDs 수준에서 동일한 예측 artifact라는 근거가 있다. Raw membrane/E/full trajectory의 동일성은 검증되지 않았으며 서로 다른 state가 같은 settled labels를 만들 수 있다.

Forced-event 조건의 E occupancy=1은 intervention 정의다. 그것만으로 원본 전체 validation E가 100%였다고 추론하지 않는다. 원본 첫4 TRAIN/full1024 occupancy `.99901056/.99992502/.99685615`는 높은 포화의 별도 관찰이고, settled512/all320 occupancy와 동일한 통계가 아니다. MG Gaussian surrogate의 backward credit는 포화 binary forward에서도 존재할 수 있다. Nonzero gradient, 파라미터 변화, event 결정 변화, grouping 이득은 각각 다른 증거다.

## 다음 학습 우선순위는 여전히 조건부

현재 가장 정확한 연구 문장은 **이 frozen native32 predictor는 K coupling, delayed sinusoidal route와 dendritic h-history로 이득을 얻지만 membrane voltage history/reset/binary selector의 추가 grouping 이득은 확인되지 않았다**이다. Frozen intervention은 현재 사용 경로의 효과이며 retrained layer necessity나 모든 gate의 보편적 필수성을 입증하지 않는다. Image bootstrap CI는 고정 세 seed에 조건부이고 seed 모집단 주장이 아니다.

1. 먼저 진행 중인 cross_view를 완료하고 source 및 matched old-loss control과 비교한다. Source를 보존하며 FG 개선을 보이면 `event_selection_review.md`의 task event-credit 대 terminal event-detach pair를 다음 후보로 검토한다. 같은 native forward/recurrence와 실제 S=E×g를 보존하고 b/tau_m만 추가 학습 가능하게 한다. GT-free fixed TRAIN에서 실제 E decision 변화와 candidate/control credit 차이를 확인하며 event-rate target을 만들지 않는다. 학습된 candidate의 closed-loop event-forcing mask 악화까지 있어야 새로운 event 의존 근거가 된다.
2. Cross_view도 source를 훼손하면, 실패한 global-prior objective에 selector capacity를 바로 더하지 않는다. `SW0139/source_preservation_design.md`의 C-only 대 C+soft source actual-Q anchor가 먼저 검토할 목적함수 변경이다. Binary pseudo-instance teacher 실패를 재개하지 않으며 source fidelity를 object truth나 gating 해결로 부르지 않는다. 이 역시 신규 등록 전 후보이고 자동 실행 계획이 아니다.

어느 조건에서도 completed context/causal recipe의 threshold·LR·rate를 재조정하지 않는다. 향후 gate 학습의 이득은 실제 masks와 matched training 대조로 확인해야 한다. 현재 source의 Slot32 FG 평균 차이는 +.00026670에 불과하고 두 seed는 Slot에 진다. 최종 all3 comparable Slot 세 지표 초과와 unique-data 증가에 따른 개선 목표는 유지한다. Source70k lineage continuation/현재 frozen ablation은 native32 training 또는 fresh scaling 증거가 아니며, 유망한 all3 학습 recipe가 확보되기 전 full70k 확장을 약속하지 않는다. Reserve90640–90959는 읽지 않는다.

````

</details>

## event_selection_review.md

<details>
<summary>event_selection_review.md</summary>

Source SHA256: `027e7b418029058239805a4b7245be796f22c7a4a75ec45f81562652471354bd`

````markdown
# Conditional event-selection direction after the causal screen

Sol6.1 read-only design review, 2026-10-10. All three technical screens are complete. Do not modify or interrupt SW0139, alter SW0140, or launch training from this document. Review completed SW0139 and the prospective fixed320 causal endpoints before deciding on this conditional follow-up.

## What the actual TRAIN screens establish

The archived seed0/1/2 files and `results_archive/causal_screen_analysis.json` establish `screen_complete`, exact NONE full1024 six-trace parity, and unchanged source/encoder hashes for all three seeds. Mean full1024-step baseline event occupancy is `.99901056/.99992502/.99685615`. In seed1's four TRAIN images, occupancy is `.999889–.999976`. Dendritic h-history removal changes mean membrane by `.387–.408`, but mean emitted-spike difference is only `1.09e-6–1.66e-5`; membrane voltage-history removal similarly changes membrane by `.382–.406` with emitted-spike difference `1.91e-7–2.53e-6`. Reset removal changes mean membrane by about `.036–.037` while **emitted spikes remain exactly identical** on all four images.

The corresponding seed0 history effects are larger: mean spike differences are `.02432` for dendritic history and `.00698` for membrane voltage history. Seed2 has per-image mean spike differences `.03191–.03573` for dendritic history, `.00627–.00669` for membrane voltage history, `.00256–.00421` for reset removal and `.00256–.00421` for forced events. Its corresponding membrane differences are `.20005–.20316`, `.14454–.14604`, `.03448–.03617` and `.000576–.000948`. Thus history has measurable forward effects in seeds0/2; seed1's substantial state changes rarely cross the binary event boundary into S=E*g. This is a seed-dependent information bottleneck, not evidence that the entire layer computes nothing. Rare exactly-zero gates/holds do occur and remain part of the registered dynamics.

Native `ActFun_adp` uses the MG Gaussian surrogate with a positive central lobe and negative broad lobes, not a rectangular hard-gradient window. Almost-always-on binary events do **not** imply zero backward credit. Finite surrogate gradients also do not establish changed event decisions, useful task information or better masks.

These are whole1024-trajectory statistics, including initial transients; they are not settled512 event statistics or patch-mask scores. Even sparse changed patches could affect connected components. All six interventions are technically eligible; evaluate the unchanged registered all320/all3 conditions without selecting them by effect size or GT. Their endpoint performance remains pending. Existing SW0138 compared retained gate with actual spikes on unchanged trajectories, so it supplies complementary factor-information evidence, not this closed-loop intervention proof. Source a_D/a_M are zero, so these results do not credit the introduced integration scalars.

## Minimal conditional training question

If the complete endpoints confirm no useful native event contribution, ask whether **task-dependent event-selector learning credit**, rather than a chosen target event rate, can make the native downstream state informative while maintaining incumbent masks. Do not recode membrane history as SW0126, replace QCC with a new RGB classifier, calibrate thresholds to sparsity, or train from GT/pseudo-instance relations rejected by SW0128.

Use the original native32 source97 seed1, full1024/settle512/live64 and original QCC. Keep encoder/graph trainable as in SW0139 and additionally release **only membrane tau_m and the existing four component threshold b values**, v_d=`.06*exp(b_d)` initialized at zero. Native Kuramoto/drive/dendritic parameters and a_D/a_M remain frozen. Retention, reset and event equations remain native. This tests learning access to an existing selector, not a claim that every dendritic gate is already useful.

Compare two fresh-source matched arms, with identical trainable parameter capacity:

- **Task event credit:** canonical old phase+5Q objective plus the exact registered SW0139 two-view temporal InfoNCE C computed from actual S.
- **Event-credit control:** the same old objective and C **values**, but C consumes `S_stop=stopgrad(E)*g`, using the actual forward's literal E and g. Keep native S in every recurrent/reset update and old-Q loss. This removes only C's terminal event-selector credit; it does not change the training forward, globally detach the custom event function, or modify old surrogate credit.

Because the source has no graph feedback/spike pulse, g/theta are independent of membrane state; this control must have exactly zero C gradient to b/tau_m while retaining C's encoder/graph gate-path gradients. Candidate C should have finite nonzero selector credit. Capture factor tensors locally from the native event calculation; prove S=E*g and identical old/C/Q/trace values before the first update. Use the same original old tensors/expressions, not a separately recomputed primary that introduces another numerical path. No arbitrary all-family or target-occupancy guard is needed.

Reuse SW0139's frozen TRAIN-calibrated C coefficient in both arms, explicitly recording that calibration originally used encoder/graph gradients. Do not recalibrate it against validation, selector rates or desired significance. The architecture/parameter-release and C-selector credit are new questions, not promotion or reopening of a failed139 recipe. If139 itself provides no evidence of useful C, record that as a major limitation on this follow-up's priority; first review the completed139/140 outcomes instead of automatically scheduling it.

## Budget and prospective decision

Both arms use identical4096 IDs/order,256 logicalB16 updates/B4x4, fresh Adam/clip1, encoder3e-6, graph3e-5, membrane tau_m3e-5 and threshold b1e-2. No warm head/decoder, occupancy penalty, homeostatic teacher, threshold/LR grid or intermediate GT selection. Measure one actual two-view logical update under an exclusive lease before admission; use139's measured two-view cost rather than assuming the one-view33.858s figure applies. All counters/calibration/data/implementation and prediction provenance remain mandatory, but additional speculative precision tests are not.

The threshold LR is an explicitly new, shared prospective parameter-group policy; it is not a validated remedy for saturation. Initial b=0 gives v=.06, and membrane tau_m at3e-5 can remain nearly unchanged over256 steps. Before admission, two disposable TRAIN updates must establish finite b/tau_m credit and actual Adam parameter changes, with the control's C selector gradients zero and the candidate's finite/nonzero. On the same fixed four TRAIN images, record baseline and post-update component thresholds, sigmoid(tau_m), membrane-margin distributions, changed binary-event counts, settled512 occupancy, and actual S/Q changes. Separate selector changes from upstream gate changes using captured E. A zero event-change count is a dormant forward response, not a reason to choose a new LR, threshold or target rate.

Register the same no-GT forward audit at update32 and256 of the uninterrupted fixed-budget run, retaining optimizer/history and reporting both arms. This is a mechanism observation, not checkpoint selection or a sparsity gate. Do not declare event learning from gradient norms or parameter deltas alone; if the trained selector never changes its forward decisions on the audit images, report the missing forward evidence even if upstream learning improves an endpoint. No automatic extension or numerical/rate tuning follows a null audit.

Evaluate both final endpoints on the same native32 320 contract. Require actual task credit to improve FG over both source and paired control with positive shared-image paired intervals to justify other seeds. Report all three metrics, event transitions and settled occupancy without using rate as success. Also register a frozen full closed-loop forced-event endpoint for the trained candidate: FG/object deterioration versus its actual predictor would support learned event reliance, subject to intervention shift. A low event rate, nonzero gradient, or candidate/control difference alone proves neither object relevance nor necessity after retraining.

If source preservation or selector benefit fails, close this fixed recipe; do not force a sparse state, retry thresholds or rename nulls as gating evidence. Three-seed comparable Slot32 metrics and increasing-unique-data performance remain separate goals. Source pretrained70k initialization makes this continuation learning, not fresh scaling evidence. No reserve90640..90959 access is permitted.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `1305ac3efd0a75a2a9bf2b34a8ea8b49506bcf6d1a11f8488af96cad9ffe2a8e`

````markdown
# SW0140 native32 causal screen

Frozen source97, native1024 patches, full1024-step dynamics. Three seed jobs with first four ordered TRAIN RGB images each; no GT or optimizer. Compare unchanged dynamics and six isolated interventions: Kuramoto interaction, delayed sinusoidal gate, dendritic history, membrane voltage history, spike reset and event selection. Changed emitted spikes feed later dendrite/reset recurrence. Exact zero-gate holds remain as specified.

Local/server9 CPU tests, actual three-source order and CPU native1024 T64 baseline parity passed. Final two-file return-contract correction was verified before GPU launch. Supervisor2726916, seed0 PID2726941 on GPU3 verified live; seed1/2 queued. Global exclusive GPU leases, CUDA-hidden supervisor, no retries. This is a trace/resource screen, not segmentation performance or proof of stage necessity. Fixed endpoint evaluation follows technically valid screens. Current introduced integration scalars are zero; their usefulness cannot be credited by this source screen. Publication remains pending scope approval.

All three GT-free screens completed and independently verified: native1024 full1024 NONE six-trace exact parity, source/encoder state preserved, downstream theta invariance. All six interventions remain eligible for fixed320 endpoint evaluation; no GT or effect-size filter. See results_archive/causal_screen_analysis.json.

Fixed320 endpoint queue launched after15 CPU tests and actual three-seed source/Slot/RGB verification. Supervisor2808950; Kuramoto-off seed0 PID2808974 verified live on physicalGPU3. Six conditions x three seeds, genuine1024-patch actual-spike QCC; all18 predictions must validate and freeze before GT scoring. Other jobs wait for exclusive global leases and run automatically when GPUs become available. No optimizer or new training in140;139 training continues independently. Endpoint scores are complete; the fixed320 results and interpretation are summarized below. See endpoint_execution_start_20261010.json.

## Fixed320 causal endpoint results

The endpoint queue completed **19/19 tasks**. It validated and froze all 18 intervention predictions (six conditions x three seeds) before opening targets. Scoring used the fixed native32, 320-image endpoint (IDs 1320 to 1639); prediction did not use ground truth and the endpoint made no optimizer updates. The six intervention conditions start from the frozen SW0097 source dynamics and change only the listed operation; source97 and Slot32 are fixed references.

| Arm | FG-ARI | Foreground IoU | Matched object IoU | delta FG-ARI vs source97 (paired 95% CI) |
| --- | ---: | ---: | ---: | ---: |
| Source97 | 0.781919 | 0.605427 | 0.597014 | n/a |
| Slot32 reference | 0.781652 | 0.204360 | 0.203165 | n/a |
| Kuramoto interaction off (K=0) | 0.056074 | 0.235064 | 0.085188 | -0.725844 [-0.741371, -0.709412] |
| Delayed gate = 0.5 | 0.000000 | 0.000000 | 0.000000 | -0.781919 [-0.797199, -0.765770] |
| Dendrite history off | 0.771650 | 0.597424 | 0.580655 | -0.010269 [-0.012928, -0.007722] |
| Membrane voltage history off | 0.782225 | 0.604921 | 0.598467 | +0.000306 [-0.000702, +0.001454] |
| Spike reset off | 0.781883 | 0.605771 | 0.596583 | -0.000036 [-0.000584, +0.000452] |
| Events forced on | 0.781883 | 0.605771 | 0.596583 | -0.000036 [-0.000584, +0.000452] |

Intervals are paired over the same images with 10,000 `RandomState(140)` bootstrap resamples; the table reports FG-ARI differences from source97. The Slot32 reference is included for comparison. Root-verified per-seed source and Slot prediction hashes, all intervention prediction hashes, queue evidence, and evaluation hashes are recorded in the [endpoint execution receipt](results_archive/causal_endpoint_execution_receipt_20261010.json).

The frozen intervention effects differ by operation. Removing Kuramoto interaction (`K=0`) sharply lowers all three metrics; replacing the delayed gate with 0.5 in the carrier, integration, emission, and hold path collapses the readout to zero. These show strong sensitivity to those particular frozen-source interventions. Turning off dendritic history causes a smaller but consistent FG-ARI decline. Turning off membrane-voltage history gives a tiny higher FG-ARI mean, but its paired interval crosses zero; matched object IoU rises only about 0.00145. Turning off spike reset has essentially no FG-ARI change. Forced-on events and reset-off produce identical saved prediction payloads for each seed; that establishes equality of saved labels/IDs, not equality of raw trajectories.

This screen does not prove that every gate is useful or necessary after retraining, and it is not a data-scaling result. SW0139 cross-view is still incomplete, so there is no promotion or incumbent change from this endpoint. The source97 incumbent remains unchanged. See the [full evaluation](results_archive/causal_evaluation.json) and [independently audited analysis](results_archive/causal_endpoint_analysis.json).

````

</details>
