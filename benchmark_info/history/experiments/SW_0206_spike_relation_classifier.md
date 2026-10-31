# SW_0206_spike_relation_classifier

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `9835beca9a5e85d279c605fb5eb0fb23e86f433e271c6b29d6068c22133e8c42`

````markdown
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

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `af570ec7185f1c206f250d359fdf86401ea8e408f266617f0a96249ee9e26bc6`

````markdown
Native64 learned spike-relation classifier; seed2 screening before replication. Pending preflight, no score.

Ten local and ten Linux tests, all source AST checks, queue CLI validation and
actual CPU model/TRAIN relation-plan admission passed before registration.
Supervisor2101577 scans all physical GPUs0..3 using shared exclusive leases.
Actual GPU1 worker2102905 subsequently started. NVIDIA PID and all8 frozen TRAIN
correlation caches verified; source model is original independent197 seed2,
b2c1a56f..., not the negative205 candidate. Classifier is97 parameters, only
signed emitted-spike correlations enter it. No core gradient/optimizer updates.
Actual four-component gather ordering was separately verified on CPU.
No validation score or full320/.90 result is claimed at start.

Completed and independently audited on fixed16 development images, seed2:

| Arm | FG-ARI | Foreground IoU | Object IoU |
| --- | ---: | ---: | ---: |
| original product-Q grouping | .8276711245 | .4448827161 | .6892385263 |
| learned relation classifier | .8292020333 | .4448827161 | .6812687164 |

FG difference+.0015309087, CI[-.0003676247,.0043684974]; object IoU declines.
Gate failed, recipe closed; no width/LR/threshold micro-rescue. The97-parameter
classifier completed200 Adam steps on65536 pairs from8 TRAIN images. TRAIN pair
accuracy89.38% with the symmetrized component product and89.75% with the MLP is
not an image segmentation result. CPU refit exactly reproduced the saved head
parameters (maximum difference0), and actual Adam/moments plus32 frozen old/new
masks independently verified. Core optimizer steps0; foreground support stays
identical. No full320 or three-seed .90/gate-causality claim follows. This does
not rule out other learned readouts or joint core/classifier training.

````

</details>
