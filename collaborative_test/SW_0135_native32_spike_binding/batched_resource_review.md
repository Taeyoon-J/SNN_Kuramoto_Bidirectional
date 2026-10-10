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
