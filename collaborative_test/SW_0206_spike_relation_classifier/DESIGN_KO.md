# SW0206 learned combination of emitted-spike relationships

205 seed2 TRAIN8 adaptation reduced splitting but increased false object merging.
Do not repeat closed185 local margins,191 partition feedback or small loss/LR
sweeps. Test a learned instance-affinity classifier while keeping the core intact.

Freeze the original independent197 seed2 baseline (not the negative205 candidate).
Extract four signed component spike correlations from8 registered TRAIN images.
Per image sample4096 same-class and4096 different-class unordered patch relations,
class/class-pair balanced, excluding self pairs. Include TRAIN background class0.
Train a CPU4->16->1 Tanh classifier (97 parameters), full-batch Adam1e-3 for200
steps, seed206002. Inputs are only symmetrized four emitted-spike correlations;
no RGB, gamma, patch position or GT counts enter the classifier at inference.
The underlying encoder/core stays frozen; no joint-core learning is claimed.

Fixed16 development offsets0,2,...30 and original193 RNGs. Replay every original
197 mask exactly, preserve its foreground support, replace only the grouping Q
by the learned probability. Keep signed pair energy/geodesic graph/normalizer.
Freeze candidate masks and classifier before validation GT scoring. Primary
extension gate: FG gain>=.01 with paired bootstrap lower bound>0 and nonnegative
object IoU. Foreground IoU must be identical. A mini is not full320 or three-seed
.90; this is mask-supervised pair classification, not unsupervised training or
proof that every gating is causally necessary. No classifier-width/LR rescue
sweep after failure. Prefix feature caching permits foreign-owner yielding.
