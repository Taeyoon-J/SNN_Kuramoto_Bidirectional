# SW_0133_soft_partition_rgb

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## design_review_after_soft_pilot_20261009.md

<details>
<summary>design_review_after_soft_pilot_20261009.md</summary>

Source SHA256: `6be8746a3caa000f853d5855e68e84066d963419c49879e275062e197b4d200c`

````markdown
# Review after the first SW0133 soft-partition pilot

Prospective conditional recommendation only. Do not alter SW0133, stop its controls, or launch this proposal before their registered results are complete. SW0133 phase_live already fails its source-expansion criterion; the remaining controls are needed for attribution. This document is a design review, not a deployment authorization or an achieved scientific result.

## Evidence and the selected bottleneck

The completed seed1 phase_live endpoint is FG-ARI/foreground-IoU/object-IoU 0.745244705/0.599191152/0.585315489, versus source97 0.742658188/0.590283199/0.584964043. Its paired FG interval [-0.0048105, 0.0095690] includes zero. Source and fixed metric contracts remain unchanged.

In the 256-update history, the first32/last32 mean RGB loss is 0.0114636/0.0068126, and mean foreground-group counts are 1.76172/1.70703. These are different TRAIN batches, so neither comparison proves learning on fixed images. Final a_D=(0.42658,0,0.23523,0), a_M=(0.48263,0,0.29139,0), b=(0.44334,-0.34966,-0.24399,-0.73884). Integration parameters did move; inability to update them is not the established bottleneck. Binary event occupancy was not measured in this history, so non-saturation is unproven.

Every SW0133 reconstruction uses detached current QCC H to choose K and initialize all assignment columns. Soft P has genuine gradients conditional on that H, but cannot add a new slot within a forward pass; K=1 has zero competitive assignment credit. The observed small TRAIN group count makes this a plausible restriction, not proof of the cause of endpoint failure. The old loss still targets global pair density 0.867, which is not an instance-label objective. Keep it fixed for the next comparison rather than introducing another coefficient change.

## One new hypothesis: native-spike imagewise competitive binding

Conditionally register a new experiment, tentatively SW0134. Learn imagewise instance assignments independently of current QCC H/K, while retaining native source97 dynamics and the zero-initialized SW0130 integration adapter. No history-centering or new membrane recoding is permitted. The head receives raw actual component spikes only; no RGB, gamma, theta, absolute coordinates, or current QCC labels are supplied to assignment.

Input is exactly the complete settled actual S[B,4,256,512]. Flatten its component/time axes per patch to 2048 values; Linear(2048,64) followed by LayerNorm creates features. Do not average time, center the emitted-spike signal, or replace it by binary events. This preserves the available phase/amplitude information in the head input. SW0123 diagnosed the mean-time horizon collapse; SW0126/127 diagnosed substantial loss from the history adapter before learned classification. Those changes are absent here.

Use an explicit Slot Attention-style binder: 11 exchangeable initial slots, 64 dimensions, three shared refinement iterations. Fixed iid initialization noise from seed134, shared across arms, is transformed by shared learned mean/scale; no per-slot category classifier is used. Shared key/query/value projections produce patch-slot logits scaled by sqrt(64). Softmax is across slots; normalize each slot's resulting weights across patches for its value update. A shared GRUCell and residual shared 64->128->64 MLP update each slot. Recompute final patch-to-slot P after iteration3. Slots may be unused; do not impose equal occupancy, a fixed foreground count, or GT counts. This adopts the competitive binding mechanism of Slot Attention transparently; it is not presented as a novel slot algorithm.

Decode each imagewise slot using its 64-D latent and P-derived relative pixel coordinates through a shared 66->64->64->3 ReLU/ReLU/sigmoid decoder. Use the same centroid/variance convention and exact intra-cell variance as SW0132, with P live in geometry. Repeat patch P over its8x8 pixels and mix decoded slot RGB with P. There is no independent alpha head. Pixel reconstruction alone may still learn appearance partitions or an assignment-independent field; this remains a falsifiable risk. Use fixed1024-pixel render-and-mixture chunks with nonreentrant checkpointing.

Inference masks are argmax slot labels. Choose the largest hard slot as background; ties resolve by first occupied patch, then slot index. Ignore empty slots and canonicalize other IDs by first occupied patch. Thus the new readout is declared explicitly, uses only actual spikes, and can have a variable number of occupied instances. Also evaluate untouched production actual-spike QCC on the same trained backbone as a secondary attribution endpoint. Never substitute that secondary for a failed primary, or compare new primary scores as if the readout were unchanged.

## Minimal paired experiment and temporal credit

Use three seed1 arms from the same immutable source97 and encoder/statistics:

1. Actual-spike head; encoder/graph/native core/integration adapter jointly trainable.
2. Gate-only head; the identical trainable backbone, but the head receives the actual gate repeated over four channels. No membrane, dendrite, event, or other hidden-state input is permitted. Native internal recurrence and old actual-Q loss continue unchanged.
3. Actual-spike head; source encoder and entire core/adapter frozen. Only binder/decoder learn.

The second arm tests whether downstream dynamics help beyond the carrier's gate; the third tests whether jointly learning the representation helps beyond a learned readout. These are different questions. All arms must be reported, even if a technical or scientific outcome is unfavorable.

Train and evaluate the head on the same full512 settled samples of a1024-step rollout. For joint credit use an exact no-grad960 prefix and live64 tail. Prepare live static gamma drive/coupling before burn-in, detach recurrent boundary states including delayed phase history, and retain live use of drive/coupling in the tail. Preserve native forward order and source values. This is a resource method, not a claim that longer credit alone helps: SW0125 did not improve FG. Prove actual1024 native zero-init trace/Q parity before training. Use B16; do not silently switch objective aggregation to microbatches.

Per arm use matched4096 source-order TRAIN IDs, 256 joint/head updates, and shuffle118. Before that, freeze the backbone for32 head/decoder warm updates on the first512 IDs. Actual/joint and actual/frozen share their warm artifact and optimizer moments; gate-only receives an independently warmed artifact from identical initial weights and the same32 images/updates. This gives the gate control its own equal-budget opportunity rather than using a decoder warmed specifically for actual spikes.

Joint arms retain the original phase-primary+5*positive-Q loss plus lambda*R. Binder/decoder gradients are unweighted R. Core/graph LR3e-5, encoder3e-6, integration scalars1e-2; head/decoder Adam3e-4. Clip the unique joint union to1 and the head/decoder union separately to1; preserve post-step a_D/a_M projection. Fresh source0 calibration after the new actual-spike warmup sets lambda=.25*median(old/R joint-gradient norm) over first4 TRAIN batches, excluding head parameters. Freeze that value for all arms and later seeds.

Technical release requires strict source/hash/data binding, exact source forward parity, identical paired initialization, finite real update paths, and no accidental reconstruction credit to downstream states in the gate-only arm. Record P usage, occupancy, event activity, and RGB slot-scramble counterfactuals, but do not reuse the pretrained positive-scramble eligibility rule as a necessary condition for successful joint learning.

## Endpoints and stop rule

Freeze predictions before opening target masks. Preserve IDs1320..1639, native128 RGB, modal8x8 GT,16grid and all three metrics. Full evaluation remains B8,T1024/settle512. Bootstrap paired common image indices10000 times with preregistered seed134. The primary actual/joint FG must exceed source97, gate-only/joint and actual/frozen with positive paired95% lower bounds for all comparisons before extending to seeds0/2. Report every metric and both readouts; add no new IoU cutoff. Failed promotion closes this recipe without slot-count, decoder, learning-rate or temperature sweeps.

Even primary success would not prove every gate useful. Gate-only parity would specifically defeat a downstream-usefulness claim. Later frozen stage interventions and matched retrained controls are needed for individual dendritic, membrane and event contributions. Fixed-update and adequately trained unique-data comparisons are separate requirements, followed by the same70k/all3-seed comparable Slot goal. Reserve90640..90959 remains unread.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `d6aeffd39296934decaff5938350dfb2da7ac21a67d9fce2f06c6af75790c93c`

````markdown
Completed 2026-10-09: candidate exceeded both controls on FG with positive paired-image intervals, but source97 interval crossed zero; expansion gate failed. No all-gating/scaling claim or incumbent promotion. See [terminal evidence](results_archive/terminal_pilot_20261009/summary.json). Registration text below is historical.

# SW0133 genuine soft-partition RGB learning

SW0132 failed its registered initial decoder-usage screen and remains closed. This new experiment tests actual joint learning with soft competitive assignments in both the RGB forward and gradient paths. Hard group seeds still come from actual spike QCC; the final evaluation classifier remains the original hard QCC.

The seed1 pilot compares phase-conditioned integration with constant integration and a decoder-only RGB-credit control. This separates assignment learning from decoder improvement and tests phase-dependent state integration. All three seeds first receive source parity and finite-update checks; the initial row-shuffle diagnostic is recorded without a positivity eligibility cutoff in this new registration.

See [prospective protocol](protocol.json). No training or performance results exist yet. Soft assignments do not guarantee object separation, and this pilot alone cannot prove every gating stage matters or that larger datasets help.

````

</details>

## scaling_design_review_20261009.md

<details>
<summary>scaling_design_review_20261009.md</summary>

Source SHA256: `6a53b9fc92eef188b4fa9278b4c77547a5ac3f8ab282ca53eabbbcd5a585efd4`

````markdown
# Prospective data-scaling design for a successful SW0133 candidate

This is a design review only. It does not change or launch the active SW0133
seed1 pilot. Scaling begins only if the registered SW0133 pilot and seed0/2
expansion gates pass with the recipe unchanged.

## Why two curves are required

SW0109's recovered seed0 checkpoints used nested 2,500/10,000/70,000 pools,
4,375 updates and 70,000 additional presentations per condition. Their
FG-ARI was `.822080/.819832/.818679`; the paired intervals included zero.
This older frozen-encoder/frozen-graph recipe therefore showed no data-scale
gain. It also gave the smaller pools 28 and 7 passes but the 70,000 pool only
one pass. Peer PV2_0048 found the same qualitative fixed-compute failure at
roughly 240,000 presentations: increasing unique images reduced repetitions,
and the large-data loss was still descending. Neither result tests whether a
jointly learned encoder/graph benefits when each pool receives the same
registered number of repetitions.

Continuing SW0097 or a completed SW0133 checkpoint is not a valid unique-count
experiment: that source already consumed the 70,000-image pool. It can test
continuation, but cannot identify the effect of exposing a model to 2,500
versus 10,000 versus 70,000 unique scenes.

## Frozen candidate and common initialization

Freeze the successful SW0133 architecture, phase-live assignment rule, losses,
lambda, decoder warmup rule, optimizer families, learning rates, clipping,
64/32 training rollout, and original hard-QCC evaluation before scaling.
Do not tune any of them per pool size.

For seeds 0/1/2, construct one seed-specific initialization before any exposure
to images outside the common 2,500-image prefix. Clone that exact initialization
into all pool-size and compute arms. The encoder, graph, native core, integration
parameters, and decoder must all be eligible exactly as in the frozen candidate;
the experiment must not freeze encoder/graph as SW0109 did. Record hashes before
the first update and prove equality across the six arms within each seed.

If the only reproducible initialization has already seen all 70,000 images,
stop and label any resulting study a continuation comparison. Reconstruct a
clean deterministic initialization instead of presenting it as data scaling.
Disclose any shared encoder pretraining and the common 2,500-image exposure;
the resulting claim is conditional scaling beyond that common history, not
from-scratch representation learning.

Use nested ordered pools from the existing training set:

- N2,500: the registered common prefix.
- N10,000: N2,500 plus the next 7,500 registered training IDs.
- N70,000: N10,000 plus the remaining 60,000 registered training IDs.

Masks and object counts never enter training. Persist every actual ID and each
cycle's independently seeded without-replacement permutation.

## Curve A: matched compute

Train every pool for exactly 70,000 presentations: 4,375 full batch-16 updates.
This gives 28, 7, and 1 complete passes for N2,500, N10,000, and N70,000. It
measures the return from diversity at fixed optimizer compute. Repetition is an
intentional part of this question and must not be interpreted as adequate
training of the 70,000 arm.

## Curve B: matched repetition

From the same initial hashes, train each pool for ten complete passes, retaining
the final partial batch rather than dropping examples. This gives 25,000,
100,000, and 700,000 presentations. Because every pass retains its own final
partial batch, the update count is `10*ceil(N/16)`: 1,570, 6,250, and 43,750
updates respectively. Ten passes are a registered equal-repetition budget; they
do not guarantee convergence or sufficient optimization. Save fixed pass-1,
pass-3, and pass-10 checkpoints for trajectory reporting, but use pass 10 as the
preregistered endpoint for every pool. Do not select a different epoch per pool.

The implementation protocol must preregister a training-only convergence rule
before launch, without validation masks or metrics. Until that rule is defined,
pass 10 remains the only decision endpoint and the study may claim performance
at equal repetition, not convergence. If the rule fails, label optimization
status unresolved; do not extend training after inspecting validation results.

The ten-pass count resembles the epoch count in the comparable SW0092 Slot
report, but it is not a matched optimizer budget. SW0092 trained Slot at batch32
for 21,880 updates per seed under its own optimizer and decay protocol; see
`collaborative_test/SW_0092_cross_dataset_training/README.md` and
`results/slot_our70000/training_protocol_seed{0,1,2}.json`.

The two curves answer different questions. Curve A measures sample efficiency
under equal compute. Curve B tests whether additional unique data helps when
each scene receives the same opportunity to train the jointly learned model.
Neither curve may substitute for the other.

## Evaluation and decisions

Evaluate every checkpoint on only validation IDs 1320--1639 with the unchanged
batch-8, T1024/settle512, positive-product hard QCC, threshold `.50`, minimum
group size 2, largest-component background, and fixed 16x16 metrics. Freeze
predictions before loading GT. Report every seed, pool, curve, checkpoint,
training loss, actual presentations, updates, and all three mask metrics.

For uncertainty, resample the same 320 validation image indices jointly across
all three seeds and all compared conditions, then average seed-level paired
differences. These intervals capture image uncertainty conditional on three
seeds; also report the three seed deltas rather than treating images as model
replicates.

Evidence that unique-data scaling improved the candidate requires the Curve-B
N70,000 pass-10 three-seed mean to exceed N2,500 on FG-ARI, foreground IoU, and
matched-object IoU, with at least two positive seed deltas per metric and a
positive common-image paired 95% interval for FG-ARI. N10,000 supplies the shape
of the curve; do not discard it if it is non-monotone. Curve A must be reported
regardless of direction and may explain an optimization-versus-diversity
tradeoff, but it cannot overturn a failed matched-repetition curve.

The final 70k candidate must also strictly exceed the comparable 70k Slot
three-seed means on all three fixed metrics (`.774933/.203589/.206937`) on the
registered validation set. This is model selection evidence, not final
confirmation. After the recipe, checkpoint rule, and Slot comparison are
frozen, run exactly one independent audit on reserved IDs 90640--90959 for both
models and all three seeds. Those reserved images remain unread throughout
pilot selection, scaling, checkpoint selection, and failure analysis. Claim the
goal only if the frozen three-seed candidate exceeds comparable Slot on all
three reserved-set means as well.

## Interpretation limits

A successful Curve B supports a benefit from more unique data for the frozen
joint SW0133 recipe under ten-pass training. It does not establish from-scratch
scaling if shared pretraining remains, and it does not by itself prove that each
gating stage is necessary. Gating contribution requires the separately matched
interventions and retrained ablations already required by the SW0133 protocol.

````

</details>
