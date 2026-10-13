# SW0150 conditional structural hypothesis

Status: prospective design and local kernel preparation only. No server training,
GPU preflight, or evaluation is launched by this document. SW0149 continues
unchanged. Full SW0150 training is conditional on interpreting SW0149's frozen320
endpoint; this is not an automatic replacement of a successful SW0149 model.

## Evidence and reason for changing structure

SW0142's `results_archive/completion_20261012/RESULT.md` reports joint learned-P
FG-ARI .291697, compared with .749859 for unchanged QCC on the same trained core
and .754135 for source QCC. Coarse-color permutations reduced the completion
loss. Reconstruction success did not require correct instance assignment.
SW0109's recovered checkpoint diagnostic shows no FG scaling benefit across
2,500/10,000/70,000 unique images at matched updates, but only for one historical
seed with frozen encoder/graph and disclosed missing training manifests.
SW0140 establishes frozen-model sensitivity to Kuramoto, delayed gating and
dendritic history; membrane/event necessity remains unproven.

The proposed change connects actual-spike assignment to an image likelihood,
rather than repeating a coefficient sweep of graph or reconstruction proxies.
No semantic masks or changed train/validation images are used in training.

## Assignment, fixed for training and inference

Use unchanged production actual-spike Q from the current student. Assignment
does not consume RGB, gamma, or theta, and has no SA/GRU or independent alpha
head. Use production QCC to identify the current background region. If that
region is empty, use the largest predicted region, with smallest first patch
index breaking ties. Its Q-medoid is the background seed. Select up to ten
foreground seeds from non-background patches by farthest-point distance 1-Q;
ties use the smallest patch index. Only seed selection is detached. There is
no frozen source teacher or cached source labeling.

For a 32x32 grid, four-neighbor weights are Q_ij + 1e-8 and self-loop weight is
one. Row-normalize the transition. Initialize unseeded P rows uniformly over
active seeds and seeded rows one-hot. Perform 64 differentiable diffusion
steps, resetting seed rows and row-normalizing P each step. P must remain
finite, nonnegative, and have row sums within 1e-6 of one. With no foreground
seed, return exact all-background P; zero Q credit for that image is expected.
Reject a feasibility trial only if the aggregate logical batch has no Q credit.
This encourages spatial continuity but does not guarantee connected argmax
regions, correct count, or semantic instances. Small regions, merges and
all-background collapse remain possible.

Each selected seed is clamped one-hot and therefore occupies at least one
patch. This kernel cannot establish that unused foreground candidates disappear
or that object count is learned. The seed selector's exact stopping/count rule
is not implemented or admitted yet; it must be resolved before any full pilot.
The standalone kernel accepts external seed identities only. Its probability
and gradient tests are not segmentation or object-count evidence.

## Likelihood hypothesis

Pool detached gamma content with live P. Foreground uses a small shared RGB
decoder with six quadratic relative-coordinate bases; background uses an
image-specific affine RGB plane. Expand each patch's P over its 4x4 pixels.
Minimize -mean_pixel log sum_k P_k Normal(RGB; mu_k, sigma^2 I), not an MSE of
the mixture's mean RGB. Fix sigma^2 to the first64 TRAIN image/channel spatial
population variance average; do not learn variance or tune it on validation.
The exact decoder implementation and variance-zero rejection must be registered
and checked before any training. This likelihood is a hypothesis, not a proven
solution to SW0142's color-compatible merging failure.

## Planned comparison and stopping

Before training, test source Q-to-P-to-loss gradients, K1 behavior, normalization,
stable deterministic seeds, and two disposable real updates. Both arms must
share decoder initialization and a frozen-source decoder-only 32-update warmup
checkpoint including Adam state. Warmup exposures must be disclosed separately.
Then compare encoder/graph-only against joint native-core and all 12 integration
scalar values, using source97 seed1, exact source4096 order, 256 logical B16
updates, micro4, T1024/settle512/live-tail64 and the established native LRs.
Clip native and decoder gradients separately at one. Do not change the active
SW0149 runner or inherit its trained weights.

Declare new P argmax labels as primary and preserve production QCC as secondary.
Freeze both arms' fixed320 prediction archives before GT. Register the
untrained source P reference as well, to separate readout changes from training.
Promote only a joint FG improvement of at least .01 over source QCC and control,
positive paired-image confidence bounds for both differences, and both IoU
margins above comparable Slot. Analyze the predeclared first64 TRAIN color-bin
P permutation intervention descriptively; better permuted likelihood counts as
objective-alignment failure. Do not tune seed count, variance or decoder on GT.

A failed fixed recipe closes without blind sweeps. A successful seed1 pilot
still requires seeds0/2, fresh native32 70,000-image training and matched-update
data-scaling tests, plus causal gate comparisons. Neither this document nor
CPU fixtures prove any of those requirements or a Slot Attention victory.
