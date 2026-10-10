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
