# SW_0135_native32_spike_binding

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## batched_resource_review.md

<details>
<summary>batched_resource_review.md</summary>

Source SHA256: `be69ba1387179d4ccf1349bb5ca5557de7a8f78394f966974b0bcfea629d9274`

````markdown
# B1/B4 resource comparison: admission remains pending

Sol6.1 read-only review, 2026-10-10. The archived B1 and B4 observations have identical16 image IDs, source core SHA, RGB/encoder/statistics asset bindings and every shared implementation fingerprint. Both use fresh identical seeded heads, lambda1 and one disposable logical update. No scientific training result follows.

B4 reduced measured logical-update time56.880->33.858 seconds, with reserved memory1.124->4.500GB. Its old accumulated gradient norm3.18275->2.76813 differs by13.03%; RGB accumulated norm.00267161->.00263747 differs by1.28%. RGB graph norm increased41.26%, but the absolute difference is only7.39e-6. Norm comparisons alone cannot establish vector equivalence or a harmless numerical error.

## Actual objective has no cross-image term

The activated PLV/Q bimodality, balance, collapse and spatial-coherence terms reduce each image's own off-diagonal matrix first, then take a batch mean. The density target remains per-image. In the B4 probe, multiplying a microbatch mean by its image count and dividing accumulated gradients by16 matches the B1 mathematical objective. No sample-diversity term is activated. Kuramoto coupling min/max normalization is per-image; theta initializes from gamma without random noise; the registered encoder has no batch normalization. No code evidence currently supports a changed cross-batch objective.

This does not prove numerical equivalence. Batch-dependent convolution/bmm kernels can alter gamma/graph inputs slightly. Graph top-k ties/selection, clamp and absolute-value branches, nearly constant spike correlation normalization, and1024 recurrent updates can amplify those differences. These are plausible causes, not an established attribution. The prior SW0120 flat-trace amplification result motivates measuring centered spike norms, not applying its failed recipe or a new normalization guard here.

## Minimal decisive diagnostic: one fixed4-image TRAIN batch

Use the first4 IDs from the archived probe; no GT, updates, new model recipe or threshold changes. Preserve actual1024 nodes/T1024/settle512/live64 and immutable source/head initial state.

1. **Separate loss reduction from rollout.** On the actual B4 Q/theta tensors, independently compute the registered batch criterion and the mean of four one-image criteria. Report phase and5Q parts, total differences and direct gradients with respect to Q and theta. This comparison deliberately fixes traces to test the reduction algebra; it is not sufficient as the source/rollout parity proof. A substantive discrepancy here is a loss/gradient-assembly blocker.
2. **Independent rollout comparison.** Run the same four images individually through fresh same-weight source instances and compare with the native B4 path before either optimizer step. Record gamma, graph/coupling values and top-k selection disagreement, theta checkpoints including512/960/1024, component membrane/spikes, actual event disagreement, Q, phase/5Q losses and centered component-trace minimum/low-norm counts. Report per-image old/R gradients and their accumulated vector relative difference/cosine, not only aggregate norms. This identifies the first differing stage and whether a small number of near-flat units dominate old credit.

If needed to locate the first difference, reuse the B4-generated gamma slices in the independent B1 rollout **as a clearly labeled secondary input-isolation check**, retaining the original end-to-end comparison. Do not replace it with two calls through the same B4 path or redefine source parity to make a failed comparison pass.

Do not require mathematical batching equivalence to be bitwise CUDA equality; also do not retrospectively widen guards merely because13% was observed. Register numerical comparisons before execution and preserve raw differences. If objective reduction agrees but actual traces/gradients change materially, B4 is an execution-dependent numeric path requiring explicit prospective registration and its own untouched native32 reference parity/warm32/calibration/scientific preflight. Resource rc0 alone admits neither B4 nor full training. If the discrepancy is an implementation/data/order defect, fix and re-review that defect first.

The evidence currently supports continuing this bounded diagnostic and preserving the measured throughput benefit; it does not support blind38.5-hour joint training, scientific failure, a claimed Slot32 improvement, or a new gate/epsilon/threshold intervention.

## Completed diagnostic and prospective decision

The archived `validation_queue_observation.json` now contains three passed native32 source/zero-adapter parity probes at B4, plus the completed independent source0 batching diagnostic. All are TRAIN-only with no scientific optimizer updates or admission.

On identical fixed B4 traces, batch old loss versus mean image losses differed by1.19209e-6; direct theta-gradient relative difference was8.05090e-7 and direct Q-gradient difference5.13087e-8. This supports the registered imagewise reduction algebra and excludes a material hidden cross-image old-loss term on the audited batch.

The independent end-to-end B1/B4 comparison had exactly equal gamma. The first observed difference was the prepared graph: max5.96046e-7, relative1.27138e-7, with similarly small prepared-coupling differences. After the1024 rollout, Q relative difference was1.01697%; complete old parameter-gradient relative difference6.08993%, cosine.998974. Some localized forward differences were substantial: component-spike max difference nearly1 and Q max difference.9570. Therefore describe this as a numerically sensitive execution path, not globally negligible error or bitwise/equivalent training. The diagnostic does not uniquely attribute each amplification to graph kernels, thresholds or correlation normalization. Binary event occupancy remained99.777%–99.999%, so this evidence supplies no new downstream gating-usefulness claim.

**Recommendation: stop further batching diagnosis and prospectively adopt B4 as the declared native32 training execution configuration**, subject to unchanged scientific admission. B4 is a different float32 execution trajectory of the same mathematical objective, not a substitute proof of B1 trajectory equivalence. The independently passed B4 source/adaptor guards remain unchanged; no failed guard is being relaxed. Preserve both historical resource records and these raw diagnostic differences.

Perform all warm32, fresh source0 first-four-logical-batch lambda calibration and real family-credit/disposable-update preflight with B4x4 accumulation. Use B4 consistently in both actual_joint and actual_frozen training arms, identical logicalB16 orders/means/clipping/Adam semantics, and record this amendment/hash in every artifact. Keep endpoint evaluation B1 fixed for both arms and their mapped-source reference; regenerate each trained encoder's native32 gamma. B4 selection does not itself admit full training or choose an endpoint.

The separate SW0136 evaluation-only transfer screen can proceed independently under exclusive ownership without changing SW0135. Use its result to make the next compute allocation decision transparently; do not silently amend the registered135 budget or promote a single transferred seed as three-seed native32 success. The immediate evidence required now is useful native32 predictions versus comparable Slot, followed by the already registered scientific preflight—not another batch, epsilon or event-threshold sweep.

````

</details>

## design_review.md

<details>
<summary>design_review.md</summary>

Source SHA256: `9f098a0f82b461f96d4a6b65d0cd3bb3c5ef42aaafaa06f14968255506620745`

````markdown
# SW0135: genuine native32 spike binding

Prospective Sol6.1 design review, 2026-10-10. This document authorizes no deployment and claims no new result. Implementation and real GPU verification must precede training admission. The governing metric/data contract is [evaluation_contract32.md](../evaluation_contract32.md).

## Decision and source

Use the same-seed immutable SW0097 positive_frozen whole core plus registered encoder/statistics, strictly mapped from 256 to 1024 nodes. Prefer this common three-seed foundation to SW0114's trained32 seed0 checkpoint: the latter already received another objective-specific continuation, has frozen encoder/graph, and supplies no matching seed1/2 foundation. Reuse audited SW0114 mapping/geometry ideas, not its training factory unchanged: `resolution.build_core` explicitly freezes the graph.

SW0114's genuine native32 seed0 result was FG-ARI/foreground-IoU/object-IoU `.817187/.760535/.641405`. Its pooled16 result is a different endpoint and neither result establishes a native32 three-seed Slot victory. The source97 pretrained70k lineage and all subsequent continuation exposure must be disclosed; the proposed 4096-unique-image pilot is neither training from scratch nor a new70k result.

Keep already-running SW0134 immutable and let it finish as historical16 work. No new16 training is admitted. SW0135 preparation and its exclusive GPU preflight have no scientific dependency on SW0134's endpoint; availability and ownership govern admission.

## Native32 foundation

Read native128 RGB, divide uint8 by255 exactly once, regenerate features through the registered encoder/statistics and adaptive-average-pool **32**, obtaining gamma `[B,8,1024]`. Do not expand gamma256 or copy16 predictions. At evaluation, regenerate native32 gamma with the actual trained encoder. Pilot selected4096 RGB images and validation320 suffice; a full70k gamma32 cache is unnecessary for live encoder training.

For row-major target `(r,c)`, parent `p=16*floor(r/2)+floor(c/2)`. Replicate omega, kappa, dendritic tau_n and membrane tau_m along their node axis, and direction_learner along both axes via p. This is spatial2x2 replication, not flattened repeat_interleave4. Preserve all shared tensors exactly; regenerate SC identity1024 and grid distances at0.5 old-patch units per new cell. Strict-load the complete declared target keys/shapes and reject undeclared conversion.

Register common resolution adaptations: K=1024 preserves K/N; graph top_k128 approximates the previous physical neighborhood; distance radius/cap remain in old-patch units; three geodesic steps retain their source settings and add temperature*log4 at each soft-min relaxation as SW0114's duplicated-intermediate entropy correction. This correction changes the mathematical operator relative to uncorrected16; it is not proof that32 dynamics equal replicated16 dynamics. Old coherence coefficient becomes1 rather than.5 to account for halved spatial spacing, equally in both arms; the other old-loss coefficients, including density target.867, remain unchanged. Secondary QCC uses threshold.50/minimum8 patches/largest-component background, preserving the previous minimum physical area.

Native zero-initialized SW0130 integration must match a separately constructed mapped32 native reference on identical native32 gamma at T64 and T1024. Do not compare32 output with upsampled16 traces. Use the frozen source integration equations and12 scalar initialization; no event/threshold tuning.

## Differentiable graph memory

The source relaxation forms `[B,N,N,N]`; at N1024 one float32 B1 tensor alone is4GiB. Plain forward row chunking is insufficient when graph is trainable: autograd would retain all row intermediates. Implement row chunks32 with **nonreentrant checkpoint of the complete row relaxation**, reducing across all1024 intermediate nodes. Checkpoint closures must bind row bounds and current relaxation inputs explicitly. Keep projection, contrast and distance gradients live; no detach, truncated intermediate-node reduction, neighborhood approximation or replacement graph.

Compare chunked/checkpointed values and gradients with the registered log4-corrected mathematical reference. Chunking changes memory, not cubic arithmetic cost; checkpointing increases recomputation. SW0114 only demonstrated the frozen-graph path. Actual1024 joint backward is a new mandatory proof.

Use microbatch1 and accumulate16 images for each logicalB16 update. Objectives are imagewise means; scale each microgradient by1/16, accumulate once, clip each optimizer union once, step once. No batch normalization, per-micro clipping or16 optimizer steps. Calibration uses the norm of the accumulated gradient vector, not an average of per-image norms. Release every micrograph before processing the next image.

Before main training, measure a complete actual1024 logicalB16 optimizer update, peak allocated/reserved memory, forward/backward time and projected full-budget runtime on an exclusively leased genuinely free GPU. Verify finite/nonzero reconstruction credit and real updates to every declared joint family. OOM or infeasible runtime closes resource admission for this implementation; do not silently freeze graph, shorten T, reduce nodes, or claim success from a forward-only test. Any subsequent resource redesign needs its own prospective registration.

## Binder and decoder skeleton

Create independent135 modules; do not mutate frozen134 constants. GRID32, PATCH4, IMAGE128, PATCHES1024. Binder input is actual emitted `[B,4,1024,512]`, preserving full settled history. Flatten only component/time per patch,2048->64+LayerNorm; keep11 exchangeable slots, shared64-dimensional key/query/value, three competitive attention/GRU refinements and the same residual64->128->64 MLP. Normalize competitive assignments across slots and slot updates across **all1024** nodes. Empty hard slots are allowed. No gamma/RGB/theta/absoluteXY/QCC groups/count enter the binder; Slot Attention-style binding is explicitly borrowed, not claimed as a new algorithm.

Use shared decoder66->64->64->3 with relative coordinates derived live from P and slot64 latents. P is the sole mask mixture; no independent alpha head. Map each genuine patch P to its own4x4 pixels. Keep native128 pixel centers; compute32-grid patch centers and intra-cell variance `((4^2-1)/12)*(2/128)^2 = .00030517578125`. Nonreentrant pixel-chunk1024 checkpointing includes the complete rendering and mixture. Target is full128 RGB MSE, with no GT or object-count information.

Retain native1024 rollout: no-grad960 prefix and live64 tail, head input includes all512 settled frames. Prepare live static gamma/coupling before burn-in, reuse them in the tail, and detach recurrent state/delayed history at the boundary. This provides truncated temporal credit, not full1024 BPTT.

## Smallest informative learning comparison

First verify all three mapped sources technically at1024; then admit a **seed1 two-arm pilot**, actual_joint versus actual_frozen. Both classify actual spikes with the same new32 binder/decoder; actual_joint trains encoder, graph, native core and12 integration scalars, while actual_frozen trains only binder/decoder. Share exact fresh initialization, frozen-backbone warm32 logicalB16 weights and head/decoder Adam32 moments within seed. Frozen native32 QCC from the same mapped source is an untrained reference, not a new16 arm. A later gate-only causal arm must be separately registered; these first two arms cannot establish gate necessity.

Use the registered4096 IDs, logicalB16,16 complete passes=4096 joint/head updates=65536 image exposures, plus disclosed warm512 exposures. A new random recurrent binder deserves the same prospectively adequate budget as amended134, rather than rejection after one pass. Pass0 uses source permutation117+seed; later passes use fixed RandomState(135000+100*seed+pass_index). Record every actual-ID order hash. No intermediate GT scoring/checkpoint selection.

Keep core/graph LR3e-5, encoder3e-6, integration1e-2 with registered projection, head/decoder3e-4; fresh joint Adam, restored warm head/decoder Adam, separate unique-union clipping1. Joint gradients are old phase+5*actual positive-product Q loss plus lambda*R; head/decoder receive **unweighted** R once. Calibrate fresh native32 lambda on source0 after its own actual-spike warm32: .25*median(old/R accumulated joint-gradient-norm ratio) over the first4 logical TRAIN batches, excluding head/decoder. Freeze it across seeds/arms; reuse neither16 lambda nor SW0114 optimizer states. Warm/calibration/data/implementation artifacts are hash-bound and saved before disposable verification.

At the final endpoint evaluate both arms, including failures. Require actual_joint FG to exceed mapped-source QCC and actual_frozen with positive paired-image95% lower bounds to justify the remaining two seeds; report all IoUs and the comparable Slot32 deltas. This development gate is not final success. If it fails, close this fixed recipe without slot, decoder, LR, threshold or checkpoint sweeps. If it passes, complete registered seeds0/2 before any three-seed claim or70k expansion.

## Evaluation and scientific limits

Primary masks are direct32-grid argmax P with frozen prediction-only largest-slot background and deterministic ties. Secondary actual-spike QCC is separately labeled and cannot substitute for failed primary. Freeze predictions before masks; IDs1320..1639 use production4x4 modal GT/min-ID ties and the unchanged three metrics. No pooled16 primary result.

Comparable own70k Slot epoch10 files `seed{0,1,2}_epoch10/predictions.npz` already contain native128 hard pixel labels. Preserve their audited perimeter-background mapping and apply4x4 modal/min-ID ties. **This is hard-label voting, not averaged probability argmax.** Explicitly disclose that our largest-slot and Slot perimeter background algorithms differ, despite common data/GT/metrics. Reproduce historical16 baseline scores only as provenance verification, not as a new16 experiment.

Ultimate success requires three-seed means above this comparable Slot32 on all three metrics. Pilot learning, finite gradients, one seed, secondaryQCC, or pretrained continuation cannot demonstrate it. Every-stage causal usefulness needs matched retrained/intervention controls; unique-data scaling requires independently registered exposure/update-controlled sizes after a useful native32 recipe exists. Reserved90640..90959 remains unread.

````

</details>

## next_hypotheses_review.md

<details>
<summary>next_hypotheses_review.md</summary>

Source SHA256: `d4f27e3ec439b7ad609ced0922e44d990150e7100acbc5ed8711dc47c85e35e9`

````markdown
# Ranked next native32 experiments

Prospective Sol6.1 review, 2026-10-10. Preserve running SW0136 and frozen SW0135; no implementation, launch or protocol change follows this document. Primary development target is native32 FG-ARI, while final numerical success still requires three-seed means above comparable Slot32 on **all three** metrics. Its established means are `.78165197/.20436013/.20316499`. Gating usefulness and increasing-unique-data benefit remain separate scientific requirements.

## 1. Native32 source97 QCC: cheapest direct candidate and common reference

Hypothesis: the already learned source dynamics retain useful instance relations when features and dynamics are genuinely recomputed at1024 nodes, without paying for another fresh classifier.

Evaluate immutable sources0/1/2 with the registered strict spatial mapping, native RGB encoder/statistics/pool32, T1024/512, batch1, actual positive-clamped component-product Q and threshold.50/min8/largest-component background. Reuse SW0136's source1 prediction artifact only if all these bindings match. Freeze predictions before GT, evaluate every seed on1320..1639 with4px modal labels, and report all per-image metrics/counts and unweighted means. No training, best-seed selection, new threshold or reserve access.

This is a valid genuine1024-node candidate despite its16-grid training origin; disclose its entire pretrained70k lineage and subsequent continuations. If all three means beat Slot32, it establishes the specified numerical native32 comparison, **not** native32 training, increasing-data benefit, independent generalization or every gate's usefulness. If it misses FG, preserve this strong common reference and do not tune its readout. In either case it anchors all subsequent development against a known native32 model rather than historical16 scores.

## 2. Learned spike binding with joint representation: conditional compute investment

Hypothesis: a trained imagewise binder can exploit actual512-frame spike relations, and reconstruction-driven encoder/graph/core adaptation can improve native32 object binding beyond source QCC. SW0136 first tests transfer of the already completed134 joint/frozen classifiers at substantially lower cost than new training. All source/geometry changes and the missing gate-arm endpoint must remain explicit.

Before allocating a roughly38.5-hour new joint arm, compare both transferred endpoints with source1 native32 QCC and Slot32. A prospective favorable allocation signal is joint primary FG above both frozen transfer and source QCC, with shared-image paired intervals supporting improvement; report both IoUs regardless. This is an allocation criterion for a **new** training decision, not a rewrite of136's outcome or automatic promotion. Joint better than a weak frozen head while still below source QCC is insufficient reason for immediate full-budget fanout.

If transfer is favorable, perform the already registered135 B4 warm32/fresh source0 lambda/scientific admission, then its fixed seed1 joint/frozen learning comparison. A genuinely trained32 endpoint is required; copied16 masks and pooled16 scores remain invalid. Keep full512 spike input, imagewise competition, P-only reconstruction mixtures and no RGB/gamma/theta head bypass. The source reference and both arms use one fixed native32 evaluation configuration. Only a prospectively successful paired pilot justifies remaining seeds and eventual70k continuation.

If both transferred heads substantially underperform source QCC after their completed16-pass training, deprioritize this expensive fresh-head recipe. Close the transfer result without slot/noise/temperature/decoder or early-checkpoint selection. A source-domain mismatch can limit transfer, so this does not prove all learned binders impossible; it does remove evidence for immediate multi-day investment. No additional weak color-mean objective, untrained-readout or loss-coefficient sweep is recommended.

## 3. Selected-model gating falsification: diagnose useful event information

Hypothesis: a *performance-useful* native32 model learns object information downstream of the sinusoidal amplitude, rather than merely preserving finite dendrite/membrane gradients. At present99.8–99.999% event occupancy and prior16 forced-event results provide no positive evidence for this hypothesis.

After selecting a useful frozen native32 recipe, register one matched stage screen on all three seeds: actual spike input versus forced-on-event **head-input counterfactual** on the same frozen trajectories and same classifier. Capture native binary events and gates from their actual forward equation; prove reconstruction `S=E*g` exactly. Feed the retained native gate in place of S only for this explicitly diagnostic condition, without changing weights, geometry, graph, masks or background rules. This is a factor-information comparison; it does not remove membrane feedback or reset dynamics. Do not promote gate-only masks as the final actual-spike candidate.

If actual input does not improve FG/object grouping relative to this counterfactual, stop the claim that event selection is useful; do not manufacture sparsity or retry thresholds to obtain the desired ablation. If a benefit appears, it motivates a separately preregistered matched retrained stage-ablation test, including dendritic/membrane retention or integration controls and equal source/data/updates. Coupling/gate collapse, activation occupancy or nonzero gradients alone cannot prove every stage necessary. This screen is lower priority than establishing native32 instance performance and introduces no new model family.

## Scaling only after a useful recipe exists

Freeze one useful architecture/objective, then preregister three unique-data counts2500/10000/70000, the same three starting seeds and no GT-selected checkpoints. First compare equal image exposures/optimizer updates, reporting repeats and complete ID orders; separately evaluate additional passes where needed to distinguish diversity from optimization budget. Warmup exposures and pretrained source data are included in accounting. With inherited70k weights this is a **continuation scaling** experiment, not a fresh-data-size causal claim; fresh-initialization scaling requires separate matched training. Existing109's one-seed flat equal-update curve does not establish scaling capability or incapability.

Reserve90640..90959 remains unread until final recipe and confirmation plan are frozen. Report every seed/failed branch, preserve incumbent native32 QCC, and evaluate comparable Slot at the same immutable4px contract throughout.

````

</details>

## peer_review_20261010.md

<details>
<summary>peer_review_20261010.md</summary>

Source SHA256: `b663797157284e4c8ae53a09c3815018272f776e521f63a00043e1b640a43644`

````markdown
# Peer review before native32 validation

Fetched both research branches on 2026-10-10. `origin/patch_v2_sw` remains at approved publication c3e2434. The colleague's `origin/patch_v2` latest collaborative evidence is commit 697eeb67dc031325b63153e678cea8cbc602da31 (2026-10-08).

PV2_0049 reports validation means .7250/.7188/.4953 for their model versus .6635/.1184/.0869 for matched Slot Attention. Their comparison has 300 validation images, shared frozen graph across model seeds and a 320-epoch Slot baseline versus the official 457-epoch schedule. It does not replace our 320-image native32 baseline or establish our victory.

PV2_0051 measured feature geometry and cancelled a proposed full-convergence scaling run. It supports examining encoder quality, but it does not establish that adequately converged larger-data training cannot help. Keep the requested fresh-data scaling study open.

PV2_0052's longer integration window improvement was only .0005 under their graph-frozen recipe. Preserve the current T1024 protocol rather than spending training budget on a speculative longer window.

PV2_0053 diagnoses same-appearance object confusion in graph features. Adding coordinates reduced their oracle-defined unseparable-image fraction from 38.8% to 18.1% at weight1. This is a mask-based feature diagnostic, not trained segmentation evidence. Their next coordinate-augmented training PV2_0054 has no committed result in this snapshot. Our current native32 graph already retains a physical spatial prior and full geodesic operator; neither is equivalent to coordinates inside graph feature projection. Position-aware learned graph features remain a separately registered candidate if the present actual-spike joint pilot fails. Do not inject coordinates into the spike-only binder or silently change the frozen SW0135 protocol.

Immediate incorporation: retain jointly trainable encoder/graph, fixed integration window, strict matched native32 baseline and explicit scaling controls. Re-check peer results before the next model hypothesis; no unsupported port of their scores or oracle geometry.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `aa724ffd0747b965ae98ebf89e4a542b4315707775617a8d04a11f0f7889b845`

````markdown
# SW0135 native32 spike binding

Preparation started following the user's instruction to use32x32=1024
patches for all new tests. The immediate target is to exceed comparable Slot
Attention under the same native32 patch evaluation contract; the final
three-seed/all-three-metric requirement remains intact.

See [native32 contract](../evaluation_contract32.md). No native32 learned-binder
training or new performance result is claimed yet. SW0134's already-running
16-grid training is kept immutable while this new experiment is prepared.

Server inspection confirmed all three own70k Slot epoch10 prediction files:
`trained_models/SW0092_slot_our70000_eval/seed{seed}_epoch10/predictions.npz`.
Each contains320 native128x128 pixel-label images for IDs1320..1639. These
can be rescored directly at4-pixel patches without rerunning Slot training
or substituting upsampled16 patch labels.

Implementation review must address native1024 feature extraction, strict
source-node parameter conversion, differentiable graph memory, full spike
history input,4-pixel decoder geometry, and a real GPU optimizer update
before main training. Grid-size constants alone are insufficient.

The previous SW0114 candidate's native32 seed0 scores were
`.817187/.760535/.641405`, with frozen encoder and graph. They are historical
single-seed observations, not a native32 Slot victory or this experiment's
result. Its pooled16 result is not the primary contract for the new work.

## Comparable native32 Slot baseline

CPU scoring completed on2026-10-10; original16 scores reproduced for all
three seeds from the same frozen native128 predictions. Native32 seed means
are FG-ARI .781651972, foreground IoU .204360128 and matched-object IoU
.203164986. All320 images are valid for each metric and seed.

See results_archive/slot70k_native32_baseline.json and
results_archive/slot32_execution_record.json. This is baseline scoring,
not new candidate training or evidence of a native32 victory.

## Native32 implementation

Implemented genuine1024 raw-spike binder,4px decoder geometry, strict spatial
source conversion and a differentiable checkpointed full-reduction graph.
Eight focused CPU tests passed, including graph values/gradients against a
dense registered reference and a real native32 core construction with live
graph parameters. See implementation_review.json and protocol.json.

Actual1024 GPU rollout/optimizer/resource checks remain pending; CPU tests
do not certify main-training readiness or any candidate performance.

Native32 encoder/source foundation is implemented and four focused CPU
checks passed. It regenerates gamma from registered uint8 RGB, maps the
complete source before the integration wrapper, and uses native32 losses.
See foundation_review.json. Actual server assets,1024 GPU rollout/parity
and logical-batch resource admission remain pending.

## Exclusive GPU resource execution

All21 model/foundation/resource CPU checks passed on Frontier. Registered
seed0 native1024 resource probe is actually running on exclusive GPU0,
supervisor2658960 / child2658986 (observed2026-10-10 05:59 UTC). See
resource_execution_start_20261010.json. This disposable logicalB16 Adam
measurement uses lambda1 before warmup; it is not candidate training,
scientific calibration, temporal-parity certification or performance.

Resource-only probe completed exit0: logicalB16 microB1 update56.880s;
peak allocated .869GiB/reserved1.047GiB. All20 joint and23 head/decoder
parameter tensors changed; all declared joint RGB gradient families are
finite/nonzero. This does not establish gating usefulness or FG-ARI.
Naive4096-update extrapolation is64.72h, so throughput must be measured
with larger image batches before committing to main training. Original
source/protocol/scientific budget stays recorded; see archived result,
queue/log SHA and resource_probe_analysis.json.

````

</details>

## throughput_amendment.md

<details>
<summary>throughput_amendment.md</summary>

Source SHA256: `9d37316720968715787e5f7c68870c98636ab787f08a30fcc54574dfaf9b79fb`

````markdown
# SW0135 prospective throughput amendment

Sol6.1 review, 2026-10-10. Resource implementation change only; no model/objective/endpoint/budget change. Register before the new probe; preserve the existing B1 observation and artifact hashes.

## Observed foundation

The actual native32 disposable lambda1 probe completed rc0: logicalB16 via16 microB1 contributions took56.87983477s, peak allocated933536768 bytes/reserved1124073472 bytes. All20 declared joint tensors and23 head tensors changed; reconstruction credit was nonzero in the reported families. These are resource/path proofs, not segmentation or gate-usefulness evidence. Simple extrapolation gives about64.7h per4096-update arm, excluding warmup, calibration, evaluation and other overhead. It is not an adequate reason to launch that full budget blindly.

## Recommended next measurement

Measure **microB4 with four accumulation calls per logicalB16** first, on the same exclusive-GPU policy, fresh source0, identical16 TRAIN IDs/order and disposable lambda1. Keep genuine1024 nodes, native gamma32, T1024/settle512/live-tail64, full intermediate-node geodesic reduction/checkpointing and the unchanged binder/decoder. No scientific training admission follows this probe.

If memory permits and throughput warrants it, prospective resource measurements at microB8 or16 may follow. Choose the execution batch from measured capacity/throughput only, never endpoint masks or scores. Record CUDA synchronization boundaries, peak allocated/reserved memory, joint/head gradient and optimizer-state allocation, wall time and projected warm32/calibration/4096-update/evaluation cost. Include initialization separately; one-step timing remains a forecast. The frozen arm has a different cost and requires its own estimate.

## Same logical objective and optimizer semantics

For each microbatch of size m, multiply its imagewise mean old loss and R contribution by **m/16**. Equivalently accumulate the sum of each image loss divided by16. The joint receives old+lambda*R; head/decoder receive unweighted R exactly once through their separate gradient computation. Zero gradients once, accumulate all16 images, clip each unique optimizer union once, perform one Adam step per union, and project integration coefficients once. Restore the same warm Adam32 moments when scientific training begins. Do not clip or step per microbatch.

The source encoder/binder use no inter-image normalization, and old/R losses reduce independently over images. Thus B4 accumulation represents the same logicalB16 mathematical objective. Calibration must use norms of the **accumulated joint gradient vector**, preserving first4 logical batches, rather than averaging microbatch gradient norms. Warm32 remains32 logical updates/512 images, not128 B4 updates.

Before GPU execution, verify batching/accumulation on a small deterministic CPU fixture, including old/R means, separate joint/head gradients, one clipping/Adam step and parameter update. Compare with explicit per-image accumulation using prespecified numerical tolerances, not bitwise equality. Actual GPU resource checks must additionally record batch-layout trace/Q/loss and gradient differences against the retained B1 reference. Float32 kernels and recurrent threshold crossings can amplify small batching differences: mathematical independence is not a claim of identical CUDA trajectories.

## Scientific admission remains separate

Always mark the disposable probe `resource_only=true`, `training_admission=false`, `ground_truth_used=false`, with source, IDs, implementation, microbatch/accumulation and optimizer configuration hashes. Do not reuse its changed source, head or optimizer as warm/training artifacts. A zero R-family gradient in the random-head resource probe is recorded, not used to reject the model hypothesis.

After a practical execution batch is selected, run native32 source/adaptor parity at that batch configuration, frozen-backbone warm32, fresh source0 lambda calibration, and the original real family-credit/update preflight. Preserve every existing scientific guard; do not redefine parity to accept a failed input/source path. Record the throughput amendment in warm/preflight/training provenance. Only then admit the registered seed1 paired pilot. Its final4096-update endpoint and native32 evaluation contract remain unchanged.

If no measured batch makes the registered budget practical, report the measured limitation and require a separately registered budget/resource decision before launching. Do not silently shorten the sequence, freeze the graph, reduce1024 patches or replace the final endpoint with an early checkpoint. No new16 experiment or server action is authorized by this review document.

````

</details>

## transfer_design_review.md

<details>
<summary>transfer_design_review.md</summary>

Source SHA256: `f4c24b65f244469e97a4ac172d97dc51f6a565df95b61a08db27a963964c5f9d`

````markdown
# Separate native32 transfer screen from completed SW0134

Prospective Sol6.1 review, 2026-10-10. **Recommend this evaluation-only screen before committing to two new4096-update native32 arms.** Keep SW0135's source97 training protocol and running parity queue unchanged. A separate registered transfer task must record its own initialization, implementation and prediction hashes. No new16 test or training is needed.

## What can transfer exactly

SW0134 `train.py` saves `checkpoint.pt` containing `wrapped_state_dict`, `encoder_state_dict`, `binder_state_dict`, `decoder_state_dict`, source SHA, seed, arm and lambda. The two completed seed1 arms are available:

- actual_joint checkpoint SHA `7bb7efcaf3640c6de3ca4ac5b8f8f265f16f6e75a2ffb24b0515862930e49beb`.
- actual_frozen checkpoint SHA `361b426179a4eb3f2d39169398557e0b898c995afab62c68cdcb025677b783ed`.

These are chosen because both registered actual-spike arms completed, not because of a new validation ranking. The interrupted gate_joint has no completed checkpoint; no three-arm/gate-usefulness conclusion is available.

Validate each immutable completion marker, manifest/checkpoint/history hashes,4096 update sequence,16 passes and original source/preflight/optimizer provenance using the original frozen134 validator. Preserve all historical artifacts. If those bindings do not validate, do not infer completion merely from a checkpoint file.

Split the wrapped state **explicitly** into `core.*` native tensors and exactly the three learned integration tensors `a_d`, `a_m`, `b`. Strip only the registered `core.` prefix, map the native tensors with the declared135 spatial2x2 conversion, strict-load the1024 core, and strict-load the learned12 integration scalars without resetting them to zero. Carry every learned graph/native/shared parameter; no partial-load fallback or source97 replacement of inconvenient tensors.

Strict-load each arm's trained encoder and the original registered feature statistics; recompute genuine gamma `[B,8,1024]` from native128 RGB with adaptive pooling32. Strict-load the entire binder state, including **the persistent134 initial_noise buffer**, learned shared slot mu/log-scale, projection2048->64, attention/GRU/MLP and normalization weights. The patch count is not a learned dimension, so these weight shapes are compatible with the135 binder. Initializing a135 head without replacing its noise buffer would change the trained classifier. Strict-load decoder weights too; they are shape-compatible, although RGB decoding is unnecessary to generate masks.

## Genuine32 execution

Use the independently implemented32 binder,1024 patch normalization, original11 slots/three refinements and full actual `[B,4,1024,512]` settled spike input. No16 feature/prediction expansion. Use the same prospectively registered32 physical foundation: K1024, top_k128, grid spacing.5, corrected log4 geodesic/full-node reduction and1024 SC. These adaptations change input dynamics; strict weight transfer does not promise exact equivalence to the16 function.

If rendering RGB diagnostics, use PATCH4 relative geometry and intra-cell variance `.00030517578125`; carry the original trained decoder rather than warm/retrain it. At evaluation put all modules in eval/no-grad mode, explicitly as inference, with full T1024/settle512. Do not treat inference freezing as evidence that a later native32 training path has correct gradients. Start batch1 to minimize both memory and batch-induced numerical changes. Hash the complete converted state and source/mapping/configuration before prediction.

One no-GT real32 batch establishes strict load, finite actual traces, genuine1024 P and output32 labels; then evaluate **both** transferred arms on all320 IDs1320..1639, freezing predictions before GT. Primary is the learned32 P readout with its frozen background/tie rule; report native32 QCC separately using.50/min8. Include the same-seed mapped-source97 QCC reference if not already available from a correctly bound native32 evaluation. Score all three metrics with the registered4px modal GT and report the fixed comparable Slot32 baseline. No held-out reserve access, checkpoint selection, slot/temperature/geometry tuning or automatic training extension.

## Interpretation

This screen can cheaply identify whether the already learned classifier transfers useful instance information at32, and whether joint versus frozen **16-training history** transfers differently. It cannot establish a native32 joint-training causal effect, native32 data scaling, all-gate necessity, a fresh70k training comparison or three-seed success. Resolution, graph adaptation and larger patch competition can degrade a useful16 head; a transfer failure does not itself reject the fresh135 recipe. Conversely, a promising single seed does not amend135 or satisfy the final objective: any native32 fine-tuning/three-seed extension needs its own prospective protocol and preserved pretrained exposure accounting.

````

</details>

## two_training_hypotheses_review.md

<details>
<summary>two_training_hypotheses_review.md</summary>

Source SHA256: `61e78ddcd4a2b6bd49090bbab6d5c6a4f557aaf15e0383d14a7f592b0f5bab04`

````markdown
# Two bounded native32 training hypotheses after SW0136/0137

Prospective Sol6.1 review, 2026-10-10. This registers a design recommendation only; no existing protocol, code or task is changed. Prefer these source-preserving tests to immediate fresh-binder4096-update fanout.

## Evidence and incumbent

The actual SW0136 report shows seed1 transferred joint head primary FG `.296267`, frozen head `.115019`; joint-backbone QCC `.659521`, while frozen/source QCC `.754135`. This independently implicates both classifier transfer and backbone drift; another decoder/head budget is poorly justified. It does not establish which learned layer caused that drift.

SW0137 completed actual1024 source QCC means `.781918674/.605427134/.597013807`, compared with Slot32 `.781651972/.204360128/.203164986`. Preserve this numerical all-three-metric incumbent. The FG margin is only `.000266702`; individual seed1/2 lose to Slot, so describe no robust/generalized victory, native32 training, all-gate usefulness or positive data scaling. Final predictions remain actual emitted-spike positive-clamped product QCC, not theta or a new weak classifier.

The registered encoder is only an eight-channel **single valid3x3 convolution**. More training images cannot supply missing spatial context to this architecture. Peer geometry diagnostics motivate testing context but do not prove an object-grouping mechanism; SW0110's failed XY-only graph route is not evidence against a learned receptive-field expansion. Conversely, existing joint RGB and late-credit failures argue against again updating every native state parameter with reconstruction.

## Common controlled source and short budget

Use native32 mapped same-seed source97, frozen statistics, fresh actual RGB->pool32, the existing exact geodesic/checkpoint and physical32 settings. Freeze native oscillator drive/Kuramoto/dendrite/membrane parameters and all12 zero-initialized integration scalars; **re-enable the native graph parameters and source encoder**. Frozen parameters do not mean detaching the computational path: gradients must pass through actual membrane/events and the live64 rollout tail to gamma/graph. Keep full1024 forward and full512 actual-spike old objective/readout.

Register seed1 as the common representative pilot before new results; run control and both candidates regardless of each other's scores unless a technical failure prevents that arm. Use the same4096 unique source-order IDs, logicalB16 via B4x4 accumulation,256 updates, no warm head or decoder. Control trains encoder/graph with unchanged source phase-primary+5*actual-Q objective, including native32 coherence1 and the existing density prior. Encoder LR3e-6, graph3e-5, fresh Adam, unique-union clip1 once per logical update. Native frozen reference is the bound SW0137 source1 artifact.

Only final256 is evaluated. Preserve all arm outcomes. At B4's measured33.858s/logical update, a conservative one-view projection is2.41h/256-update arm; actual old-only training may be cheaper because it omits RGB and multiple gradient calls. This is a forecast, not a resource guarantee. Measure one actual declared update for each new route first, with exclusive leases and unchanged1024/T1024/512/live64. No4096-update commitment,16-grid fallback or hidden graph freeze.

## A — Context residual encoder, native dynamics fixed

Hypothesis: context-aware patch representation can separate same-appearance instances through native graph/spike relations while preserving the useful source temporal operator.

Let F be the registered clipped standardized eight-channel126x126 feature map. Add four parallel Conv2d8->16, kernel3/dilation `(1,4,16,32)`, padding equal to dilation, followed by GELU; concatenate64 channels and apply Conv1x1(64->8). Initialize the last projection weights and bias exactly zero, keep branch initialization fixed/private, and use `clamp(F+residual,-3,3)` before the existing adaptive pooling32. Keep128 input and registered statistics; no XY or RGB path into the classifier. The largest branch supplies67-pixel input receptive field through the source convolution; this is a capacity/context intervention, not an explicit coordinate label.

Candidate trains original encoder/graph plus adapter, using the **same old objective as control**, no new RGB loss, teacher labels, classifier or coefficient calibration. New adapter LR3e-4; original family rates unchanged. Compare source, control and candidate to distinguish a useful capacity change from merely beating a degraded continuation control.

At initialization require exact source gamma/traces/Q on the same B4 input. The zero output layer necessarily makes hidden branch gradients zero on step1. A **two-step disposable TRAIN check** should show finite output-projection credit/update first and subsequent live hidden-branch credit; do not manufacture a first-step all-parameter gradient requirement. Save no disposable optimizer state as training initialization.

Failure closes this fixed256-update context recipe without dilation/width/LR grids. It does not prove that every contextual encoder is ineffective or that a fresh adapter is sufficiently optimized at all budgets.

## B — Cross-view temporal correspondence, original architecture

Hypothesis: directly making actual spike relations stable to benign photometric nuisance provides useful encoder/graph learning credit that the global density prior lacks. This uses **known same-patch augmentation correspondence**, not source-Q pseudo-instance labels; it does not reopen the rejected SW0128 teacher branch.

Use the original eight-channel encoder/graph architecture, no context adapter or learnable head. Canonical view x remains unchanged. Make a second native128 view `x' = clamp(c*(x-.5)+.5+b,0,1)`, c uniform[.9,1.1], b uniform[-.03,.03], one scalar pair per image from a fixed recorded TRAIN RNG. No crop, resize, hue permutation or instance mask. Both views compute genuine1024 dynamics at the same horizon.

For each patch, flatten its raw emitted4x512 settled spike history to2048, then L2-normalize using the existing epsilon1e-8. Do **not** time-average, history-center, replace spikes with phases/gates, or learn a disposable projection. Positive similarity is the dot product of corresponding patch vectors across views. For each anchor use32 deterministically sampled patch vectors from **other images in the same B4 microbatch**, with no within-image negatives. Use ordinary symmetric two-direction InfoNCE, temperature.10, mean over patches/images. Keep both view embeddings live. Record that same-appearance/background false negatives remain possible: this objective learns nuisance/correspondence consistency, not an object supervision guarantee.

Joint loss is canonical old objective + lambda*C; the endpoint is still the original actual-spike QCC from the canonical image. Calibrate fresh lambda on source0 first4 logical TRAIN batches as `.25*median(norm(old_encoder_graph_grad)/norm(C_encoder_graph_grad))`; use accumulated B4x4 gradient-vector norms. Freeze the coefficient across seeds, no validation tuning. Require finite nonzero new loss credit to encoder/graph on real native32 input; no pre-training GT accuracy/precision guard is relevant to exact geometric correspondence. The source old loss and its numerical guards are unchanged.

Control uses the same canonical old objective/source/order/updates. Candidate additionally sees a second view; disclose8192 transformed training exposures versus control4096, but the same4096 unique images. This is an objective-plus-augmentation intervention, **not FLOP-matched training**. No extra control retraining is required beyond the common canonical control. Conservatively budget up to about4.82h for the256-update two-view arm, then replace that estimate with measured runtime; no automatic full-epoch sweep.

## Endpoint, extension and gating limits

For each candidate separately, require primary native32 FG above both source1 and paired control, with shared-image paired95% lower bounds positive, to justify registered seeds0/2. Report all three metrics and comparable Slot32 deltas, no checkpoint or readout selection. All-three-seed final means and per-seed consistency must precede any70k continuation; a pilot is not the final goal. Do not retrospectively choose which historical seed counts as replication.

These tests intentionally protect native dynamics while learning their input/graph representation. They cannot establish membrane event usefulness merely because a surrogate path is live. SW0138's separate factor-information screen and subsequent matched stage controls remain necessary; occupancy99.8–99.999% is still a paper limitation. Source97's70k lineage makes these continuation experiments, not fresh scaling evidence. Only after a useful native32 model exists register2500/10000/70000 unique-data curves with equal exposure/update controls and separately disclosed extra passes. Reserve90640..90959 remains unread.

Required provenance is source/checkpoint, data IDs/order, preprocessing, mapped architecture, implementation, augmentation RNG, coefficient, optimizer/update count and predictions frozen before GT. Do not add speculative decoder-utility thresholds, binary teacher QA or arbitrary first-step family guards to these decoder-free experiments.

````

</details>
