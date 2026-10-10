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
