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
