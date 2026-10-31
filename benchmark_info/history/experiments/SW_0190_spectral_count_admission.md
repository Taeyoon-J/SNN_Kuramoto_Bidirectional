# SW_0190_spectral_count_admission

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `a56a1295a51b74844327f6a3fd45f371b4e40bc811a4a71c33d5645e06fd34c1`

````markdown
# SW0190: TRAIN-only oracle-count global partition admission

SW0189 observed both fragmentation and inter-object merging in the incumbent.
Before training a count predictor, test whether supplying the correct TRAIN
object count can make a different global assignment useful at all.

Use the completed matching-seed SW0186 model, all three seeds0/1/2, and the same
16 TRAIN512 positions0,32,...,480. Native32 RGB→all four gates→actual component
spikes→unchanged production Q, T1024/settle512, B4, live-tail0. No optimizer
updates, new encoder/head, validation RGB/masks, or model tensor changes.
The baseline is the actual QCC plus foreground-induced merge production path.

Two fixed alternatives use the same Q: normalized symmetric spectral embedding
with topK eigenvectors, row normalization, deterministic farthest initialization
and50 Lloyd steps, no selected restart or parameter sweep. The first keeps the
baseline foreground support and partitions those nodes into K groups. The second
partitions all patches into K+1 groups and chooses background by maximum overlap
with the model's existing background, then cluster size and ID. Only the scalar
TRAIN modal visible-object count K is oracle information used by the readout;
no target pixel membership, GT background geometry or object correspondence
enters the assignment. Counts are not predicted yet.

Record paired TRAIN metrics for baseline and both alternatives, every source
seed and all16 images, plus exact unchanged model SHA and Q/z digests.
Do not call the oracle metric a deployable model result or update a leaderboard.
The small TRAIN cohort is an admission test, not a validation improvement.

Admission to a learned count/readout experiment requires at least one fixed
alternative's mean TRAIN FG improvement of .01 or more, no mean Foreground IoU
decrease, and positive mean FG difference in at least two of three seeds.
Report all alternatives and ties. Failure closes this spectral/count recipe;
do not perform a small clustering/threshold sweep. A pass only justifies training
a predictor without GT counts at inference and evaluating full fixed320 later.

CPU tests and all three actual frozen-model admissions precede GPU execution.
Reuse the exclusive-owner/yield queue, never share a foreign GPU. The pilot may
use a free GPU while SW0188 continues; it never stops or changes SW0188.
Detailed TRAIN imagewise reports remain private; share code and cohort aggregates.
Unsupervised, native32 joint70k and gate causality are not established here.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `08d757fe4f5f22b74ee78d21a7752ec62f7447d9b2a1109df2e29cf9338bf90d`

````markdown
# SW0190 — oracle-count global spike-affinity partition pilot

Preparation: TRAIN16, all three frozen SW0186 seeds, two global spectral
readouts versus unchanged production masks. This tests an oracle-count upper
bound before investing in a learned count/assignment model. No optimizer steps,
validation masks or leaderboard result. See [registered design](DESIGN_KO.md).

Raw TRAIN imagewise reports stay private; only code and aggregate findings are
published. SW0188 continues independently.

Registered: eight tests passed locally/server, actual CLI and all three exact frozen incumbent CPU model loads passed. Three supervisor PIDs1688076/1688085/1688094 were independently verified alive and queued. No GPU worker has started yet; current free-device opportunity closed before admission and foreign owners occupy the other devices. SW0188 continues on its existing worker. Peer patch_v2 was checked this cycle and remains cab6c676. Metadata: results_archive/cpu_v1 and start_v1.

Completed: both oracle-count spectral variants are worse in all three seeds on TRAIN16 and fail admission. Baseline FG .872016 versus .782037/.771040. These are TRAIN diagnostic values, not validation scores or an automatic count classifier. Close these recipes without a sweep. See [full findings](RESULT_KO.md); raw TRAIN reports remain private.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `9a0bc3ee441d31e813675bffef780dcb4b2d9b194de4108c99945b07d4cb0d29`

````markdown
# SW0190: both oracle-count spectral recipes fail admission

Three frozen SW0186 source models completed the same16 TRAIN images each.
Both global spectral alternatives perform worse in every source seed despite
receiving the scalar true visible-object count. Neither is admitted to a
learned-count expansion. Close these two specified recipes without a sweep.

| TRAIN16 method, three-seed mean | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| Existing production baseline | .872016 | .561423 | .751356 |
| Spectral, existing foreground support | .782037 | .561423 | .581457 |
| Spectral, all patches and model-chosen BG | .771040 | .551537 | .533890 |

| Seed | Baseline FG | Existing-support spectral FG | All-patch spectral FG |
|---|---:|---:|---:|
| 0 | .880259 | .758035 | .800179 |
| 1 | .843351 | .782438 | .740958 |
| 2 | .892438 | .805639 | .771983 |

FG differences are -.089979 and -.100976; positive seed count is0 for both.
The fixed-support alternative preserves Foreground IoU exactly, so its drop
cannot be explained by changed foreground membership. This points against this
particular global partition of Q; it does not prove all count-aware or global
assignment models are unhelpful. The original graph/geodesic production structure
should be retained in the next training-loss experiment.

These are TRAIN diagnostic numbers, not fixed320 validation results. The .8720
baseline is not a new .87 benchmark achievement. Count was supplied, not learned;
no count-classifier accuracy, automatic object counting, unsupervised improvement
or gate causality is claimed. Oracle-count feasibility here is not a mathematical
upper bound on every possible model or partition algorithm.

Eight tests and the actual CLI passed locally/server. All three actual frozen
CPU models matched the pinned SW0186 incumbent. GPU workers preserved their
entire model tensor hashes and made zero optimizer updates. TRAIN masks supplied
only scalar counts to the alternative readouts; pixel memberships were used only
for scoring. No validation RGB/masks were read. Source hashes, per-seed report
hashes, metric-vector arithmetic and the preregistered admission rule were checked.
No independent saved-mask replay was performed for this small pilot; do not
describe the arithmetic check as an independent metric recomputation.

[Public cohort summary](results_archive/summary_v1/summary.json).
Raw imagewise TRAIN reports remain private and ignored by Git. No incumbent
changes: SW0186 supervised FG and SW0165 unsupervised anchors remain separate.

````

</details>

## results_archive/cpu_v1/registered_README.md

<details>
<summary>registered_README.md</summary>

Source SHA256: `f20d9257884a028af76a3b613437e97eb5bbc77f65af94032d44707359158e57`

````markdown
# SW0190 — oracle-count global spike-affinity partition pilot

Preparation: TRAIN16, all three frozen SW0186 seeds, two global spectral
readouts versus unchanged production masks. This tests an oracle-count upper
bound before investing in a learned count/assignment model. No optimizer steps,
validation masks or leaderboard result. See [registered design](DESIGN_KO.md).

Raw TRAIN imagewise reports stay private; only code and aggregate findings are
published. SW0188 continues independently.

````

</details>
