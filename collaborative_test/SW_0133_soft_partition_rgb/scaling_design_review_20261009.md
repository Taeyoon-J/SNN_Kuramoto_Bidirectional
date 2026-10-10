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
