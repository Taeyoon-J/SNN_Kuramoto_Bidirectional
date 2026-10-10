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
