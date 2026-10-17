# SW0191: production-partition error feedback

SW0188's extra QCE adaptation did not improve the exact SW0186 incumbent.
SW0190's two oracle-count spectral readouts are worse in all three TRAIN seeds.
Retain the successful production QCC/geodesic partition and test feedback from
its actual mistakes, rather than replacing the readout or repeating SW0185's
local-neighbor margin loss.

Initialize all three source seeds from the exact completed SW0186 weights,
including RGB CNN/context/graph/core/integration, with explicitly empty five-group
Adam. Keep TRAIN512 twice/64 proposals, B16/micro4, all learning rates,
T1024/settle512/live-tail64, projections and clip1. Inference is unchanged.
This is supervised extra adaptation, not fresh native32 joint70k training.

Before each proposal, use a same-RNG no-gradient native rollout and the actual
production QCC/foreground-induced merge to mine predicted labels on that TRAIN
batch. Freeze those labels for the gradient forward and the post-proposal
full-forward loss. Never re-mine targets during the acceptance comparison.
This avoids changing the objective when the discrete partition changes.
Record the frozen context SHA. All64 gradient losses must match their same-RNG
pre-proposal full-forward composite losses within the existing tolerance.

Add coefficient1 feedback to unchanged product-QCE temperature.1:
same-GT/different-predicted-label pairs receive mean(1-signed_component_rho)^2
over four components, class-balanced over affected GT classes. Different-GT/
same-predicted-label pairs receive production Q^2, balanced over unordered GT
class pairs. Use each unordered patch pair once, excluding self pairs. Average
the positive/negative terms that exist; return a connected zero when no error
pairs exist. Background label0 participates as a cluster. Accumulate large group
sums in float64 for stable strict-loss checks. This is a stop-gradient mined
contrastive surrogate, not a differentiable ARI or a structured-inference bound.

Actual source RGB/normalization/graph paths and all four gating-to-spike routes
remain intact. First two GPU proposals must prove finite-positive total spike
gradient, all five family gradient/proposed delta credit, full-forward parity,
and separately measured feedback-only emitted-S and trainable-parameter gradient
RSS. Report its per-family parameter credit without requiring every auxiliary
family to have nonzero credit. This prevents a gradient into detached-prefix
spike outputs alone from being called effective model learning. A dead auxiliary
is a scientific rejection, not evidence that the new loss worked. Learning may
help fragmentation/merging; this alone does not prove gate causality or solve
fully silent foreground or automatic object counting.

Exclusive GPU ownership and yielding remain mandatory. To prevent a yielded
checkpoint waiting indefinitely for its old physical index, SW0191 stores a
compatibility signature excluding only CUDA_VISIBLE_DEVICES. All software,
hardware name/capability/memory, logical device index and math flags still match
strictly before restoration. Model/Adam/RNG/history restoration is unchanged;
every actual physical GPU/UUID/worker attempt is recorded in an append-only
ledger copied into the final artifact. Require homogeneous server GPU hardware
in CPU admission. Cross-physical-GPU bitwise trajectory equivalence is not claimed.
No existing checkpoint or common SW0148 implementation is modified.

Freeze all three fixed320/native32 predictions and both endpoints before new GT
scoring. Primary gate is standalone FG mean above SW0186 with positive paired-image
95%CI lower bound, plus all three means above comparable Slot70k. Report SW0188,
both IoUs, all seeds, conditional output and FG.87 separately. Independently
recompute actual masks/GT/means/intervals and verify trained CNN/Adam artifacts.
Raw imagewise scores remain private; publish code, cohort summaries and audit.
If the gate fails, close this recipe without a small margin/rate/epoch sweep.
