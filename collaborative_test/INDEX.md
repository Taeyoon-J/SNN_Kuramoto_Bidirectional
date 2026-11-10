# Experiment index

| id | status | change | fg_ari | fg_iou | obj_iou |
| --- | --- | --- | --- | --- | --- |
| PV2_0001 | completed | baseline under contract v1 | 0.4457 | **0.4115** | 0.3438 |
| PV2_0002a | failed | constrain the firing rate | 0.2316 | 0.3388 | 0.2306 |
| PV2_0002b | completed | geodesic coupling graph | **0.5537** | 0.3696 | **0.3809** |
| PV2_0003 | completed | per-component spikes and pulse coupling, screened at one seed | | | |
| PV2_0004 | completed | per-component spikes on the geodesic graph | 0.5105 | **0.4967** | **0.3689** |
| PV2_0005 | completed | ceiling re-measured; spatial prior 0.35 helps, geodesic 5 hurts | | | |
| PV2_0006 | completed | `--readout plv` added; the phase readout beats the spike readout by 0.174 | | | |
| PV2_0007 | failed | distil phase synchrony into the spikes by MSE; foreground IoU collapses to 0.37 | | | |
| PV2_0008 | completed | **spike synchrony in the objective, three seeds** | 0.6075 | **0.6755** | **0.4298** |
| PV2_0009 | completed | objective re-balance screened on seed 0; bimodality 3 lifts all three | | | |
| PV2_0010 | completed | bimodality 3 at three seeds: compresses spread, mean +0.022 | 0.6337 | 0.6643 | 0.4173 |
| PV2_0011 | refuted | fg_ari does not prefer a lower spike weight; 5 is still best | | | |
| PV2_0012 | completed | **half of a six-seed draw fails outright** | | | |
| PV2_0013 | completed | **bimodality 6 lifts all three at all three seeds** | **0.6756** | 0.6878 | **0.4728** |
| PV2_0014 | failed | align to the phase contrast; gap closes by the phase side collapsing | | | |
| PV2_0015 | completed | frozen phase path; gradients arrive, the alignment term will not move | | | |
| PV2_0016-18 | failed | per-region transduction; the shared map is the prior, not the bottleneck | | | |
| PV2_0019-20 | completed | **the leak is a sampling limit: the gap falls 0.149 -> 0.023 with window length alone** | 0.6887 | 0.6828 | 0.4710 |
| PV2_0021-22 | failed | training at the readout's window collapses; bimodality 6 survives re-selection | | | |
| PV2_0023-25 | completed | **membrane threshold bug: a dead seed revives from 0.000 to 0.635 foreground IoU** | | | |
| PV2_0026-27 | completed | **ported the peer's spatial kernel: fg_ari 0.7059, obj_iou 0.4903** | **0.7059** | 0.6778 | 0.4903 |
| PV2_0028 | completed | matched-threshold spike vs phase: gap is 0.0196, not the 0.0044 first reported | | | |
| PV2_0029-30 | completed | **spectral readout ports the control: geometry alone 0.390, spikes add 0.217** | | | |
| PV2_0031 | failed to transfer | peer's low-LR rescue: spread halves, mean falls 0.040, dead seed stays dead | 0.6657 | | |
| PV2_0032 | completed | **ORACLE ceiling: true count + phase readout reaches 0.7338, below the 0.75 goal** | | | |
| PV2_0033 | running | train the feature encoder alongside the core (`--encoder-lr`) | | | |
| PV2_0034 | completed | **ORACLE features: this architecture reaches fg_ari 0.975; the encoder costs 0.27** | | | |
| PV2_0035-36 | completed | **within-object consistency is worth +0.170 and alone reaches 0.8935; the margin adds +0.090** | | | |
| PV2_0037 | failed | DINO worse, encoder training a no-op, label-free clustering lowers fg_ari | | | |
| PV2_0038 | failed | self-bootstrap from the model's own groups: +0.006, circular | | | |
| PV2_0039-40 | completed | **end-to-end works once reconstruction holds the features: 0.6779 against frozen 0.6639** | | | |
| PV2_0041 | failed | capacity even with reconstruction: depth 2 diverges, depth 3 gives 0.3910 | | | |
| PV2_0043 | completed | **protocol check: spike timing carries the affinity, r=0.982 against binary-only** | | | |
| PV2_0042 | exhausted | slot-term tuning: the defaults were already optimal (weight 2 gives 0.392) | | | |
| PV2_0044 | completed | **ported the peer's graph freeze: best on all three, 0.7250 / 0.7188 / 0.4953** | **0.7250** | **0.7188** | **0.4953** |
| PV2_0045 | failed | end-to-end under a frozen graph collapses to 0.011; the graph is image-conditioned | | | |
| PV2_0047 | superseded | matched Slot Attention at two seeds and unseeded inference: reported 0.7307. Wrong -- see PV2_0049 | | | |
| PV2_0048 | exhausted | more training images at matched compute: 0.6779 -> 0.5983 -> 0.4790, monotonically worse | | | |
| PV2_0049 | completed | **matched Slot at three seeds and seeded draws: 0.6635 vs our 0.7250 -- we win ALL THREE** | | | |
| PV2_0051 | exhausted | data scale closed on the mechanism too: w/b improved only 7% while fg_ari fell; 27-hour run cancelled | | | |
| PV2_0052 | exhausted | integration window saturated: +0.0005 from 1024 to 4096, inside the seed spread | | | |
| PV2_0053 | completed | **38.8% of images hold an object pair the graph's cosine cannot separate; position takes it to 18.1%** | | | |
| PV2_0054 | failed | widening gamma killed the spike path: joint standardisation rescaled the appearance channels | | | |
| PV2_0055 | completed | **position in the graph: loss lower, score 0.168 worse. The objective is satisfiable by geometry** | | | |
| PV2_0056 | exhausted | our partition overlaps the geometric template by only ARI 0.1164, so penalising that does nothing | | | |
| PV2_0057 | completed | **within-object spread is mostly shading: chromaticity cuts w/b 46%, 0.2030 -> 0.1093** | | | |
| PV2_0058 | failed | chromaticity never converges; but shows fg_ari follows predicted foreground extent | | | |
| PV2_0059 | completed | the fg_ari/IoU dial: spectral k from 6 to 20 spans fg_ari .579 to .772 and fg_iou .54 to .16 | | | |
| PV2_0060 | completed | **k=14 at three seeds clears 0.75 on validation: .7569 / .3449 / .6480** | | | |
| PV2_0061 | completed | **TEST, read once: .7533 / .3358 / .6549. Beats Slot on all three; 0.75 target met** | **0.7533** | **0.3358** | **0.6549** |
| PV2_0062 | exhausted | features improved at inference: best foreground IoU yet (.7626) but no fg_ari gain | | | |
| PV2_0063 | exhausted | no spectral k on the smoothed gamma dominates the settled configuration | | | |
| PV2_0064 | exhausted | **whitening enlarges the cosine margin 2.7x without labels and fg_ari still does not move** | | | |
| PV2_0065 | failed | gamma from the matched Slot encoder: cos_between .0316, fg_ari .0604; whitened at best .4840 | | | |
| PV2_0066 | completed | **the GRAPH carries 94% of the oracle's gain (.9671 at pred_fg .112); PV2_0005's +0.06 measured something else** | | | |
| PV2_0067 | exhausted | transforms confined to the graph: whitening still breaks calibration, smoothing still gains no fg_ari | | | |
| PV2_0068 | exhausted | reconstruction training the graph with features frozen: best calibrated .7175, -0.0064 | | | |
| PV2_0069 | exhausted | the graph fed its own groups as one-hot: .7239 in, .7199 out -- a fixed point, no amplification | | | |
| PV2_0070 | exhausted | **cross-view consistency: invariance achieved (term 2-3x down), fg_ari -0.0482** | | | |
| PV2_0071 | exhausted | the generator's own knobs: best is +0.0062 for -0.0401 fg_iou; contrast and decay are learned, so those rows were vacuous | | | |
| PV2_0072 | completed | **the oracle number measures transcription, not discovery; the pipeline repairs random damage (+0.15 at heavy corruption)** | | | |
| PV2_0073 | exhausted | iterating the model's own grouping converges to 0.7274: its errors are spatially coherent, which the repair cannot touch | | | |
| PV2_0074 | exhausted | the learned geodesic contrast is already best; 0.5-32 spans .7243 to .6406 and groups split before walls dominate | | | |
| PV2_0075 | completed | **why: the barrier is two thin walls (5.27x) with a background corridor 6x cheaper than a within-object step** | | | |
| PV2_0076 | completed | **superlinear geodesic step cost: TEST .7603 / .3373 / .6598, all three up, no calibration cost** | **0.7603** | **0.3373** | **0.6598** |
| PV2_0077 | failed | training under the superlinear cost gives .4268 and .5248: its gain is an inference-time correction only | | | |
| PV2_0078 | exhausted | lowering the per-step toll: base 1.0 is already best when calibrated; the spectral 'gain' is the extent dial | | | |
| PV2_0079 | exhausted | **the dynamics are not the limit: osc_dim 4 and k 256 already best; osc_dim 8 reached a LOWER loss for 0.5670** | | | |
| PV2_0080 | exhausted | **less capacity fails too (osc_dim 3: loss 16.21, fg_ari .6783), so the pattern is loss/metric decoupling, not capacity** | | | |
| PV2_0081 | exhausted | **no freeze helps; 1.9% of parameters match full training; PV2_0044 was seed-variance reduction, not a per-seed gain** | | | |
| PV2_0082 | completed | **contract v1 admits 0.9 (a perfect segmenter scores .9756-1.0000 under any rule): the gap is the model's, not the metric's** | | | |
| PV2_0083 | completed | **70.7% of the error is on single-object patches: the merges are between patches, not inside them, so input resolution is not the limit** | | | |
| PV2_0084 | completed | MOVi-A prepared; motion's margin is 1.4x standardised appearance, not 13x; channel concatenation hits the dead-membrane wall | | | |
| PV2_0085 | completed | **motion as a PLV target: calibrated control .3042, motion .0823. Negative -- but the pilot's control is too weak to be the platform** | | | |
| PV2_0086 | completed | **the motion term helps on a working platform: control .4723 at pred_fg .1006, weight 10 .4826/.5613/.3735 at .1198. First positive in ~31 routes, one seed** | | | |
| PV2_0087 | completed | **complexity: ours is O(N^3 + T N^2) in patches, Slot Attention O(K L P k^2 C^2) in pixels. 17.6x cheaper at 16x16, parity at 32x32, 13x with a sparse rollout** | | | |
| PV2_0088 | completed | **the 32x32 enablers: grid rule verified IDENTICAL to v1 at grid 16; geodesic 68.7 GB -> 4.72 GB bit-identical; gamma recipe reconstructed** | | | |
| PV2_0089 | failed | **32x32: the phase side is healthy (plv_mean .7101 vs grid16's .753) but the spike path dies at epoch 17, spike_plv_collapse = log(10^4). Transduction hyperparameters do not transfer** | 0.0000 | 0.0000 | 0.0000 |
| PV2_0092 | failed | **signed coupling: every arm widens predicted foreground past the gate and sells fg_iou. obj_iou rises in all three (+.0337/+.0284/+.0778). CONTROL IS NOT THE PUBLISHED MODEL -- .5027 vs .7239** | 0.5962 | 0.4708 | 0.4240 |
| PV2_0093 | completed | **32x32 revived: population threshold falls 4.72 -> 2.58 (alive), control 4.26 -> 5.69 (dying), spatial decay 4.32 -> 8.94 (dead). The transduction did not transfer, not the graph** | | | |
| PV2_0094 | completed | **32x32 trains healthy with the population threshold (collapse 1.86 vs grid 16's 1.90). At MATCHED extent (pred_fg .1268 vs .1290) the finer grid buys obj_iou +.0889 and sells fg_ari -.1195: better object shape, worse grouping coherence** | 0.6044 | 0.6661 | **0.5840** |
| PV2_0095 | completed | **the 32x32 failure measured: 19.5 groups per image against 6.49 true objects. sigma 4.0 recovers fg_iou to within .016 of grid 16 and holds obj_iou +.0934; min-group-size made it worse, deleting objects not noise** | 0.6293 | 0.7091 | **0.5885** |
| PV2_0096 | failed | **graph sparsity is not it: top_k 32 -> 256 (8x denser) moves fg_ari +.0019 and groups 16.76 -> 16.14. Three readout knobs swept, fg_ari stuck at .63 -- PV2_0091's verdict arriving at the other grid** | | | |
| PV2_0097 | completed | **group-count term: 2 of 3 gate conditions at weight 5 (groups 15.84 -> 13.68, fg_ari +.0104, fg_iou +.0276, obj_iou -.0103). Weight 20 reads best on fg_ari alone and fails calibration at pred_fg .41 -- the fourth time that would have misled** | 0.7803 | 0.7807 | 0.6914 |
| PV2_0101 | partly withdrawn | **the 40-epoch budget IS wrong -- 8 epochs beats it by +.1406/+.0440/+.1132 at seed 0 and seed 2 also clears it. But the "beats 16x16 on all three" claim was seed 0 only and does not survive (`PV2_0102`)** | 0.7699 | 0.7531 | 0.7017 |
| PV2_0102 | failed | **the 8-epoch result does NOT survive 3 seeds: .5894 +- .2599 (control) and .5846 +- .3230 (w5) against 16x16's .7239. Seed 1 fails in both arms, two different ways. PV2_0012's known fragility, and PV2_0044's variance fix was never applied at 1024 nodes** | 0.5894 | 0.4876 | 0.5007 |
| PV2_0103 | completed | **seed 1 rescued from .2916 by BOTH fixes: graph freeze .6777/.6895/.6357 in 54 min, 16 epochs .6867/.6008/.4887 in 341 min. The script picked the 16-epoch arm on fg_ari alone -- the fifth such wrong verdict; the freeze wins on both IoUs at a sixth the cost** | 0.6777 | 0.6895 | 0.6357 |
| PV2_0104 | completed | **the freeze closes the spread at 1024 nodes: fg_ari std .2599 -> .0512. 3-seed .7110/.7130/**.6588** vs 16x16's .7239/.7251/.4951 -- both losses inside one std, obj_iou +33% at 4.4 std. A different operating point, not a better model** | 0.7110 | 0.7130 | **0.6588** |
| PV2_0105 | completed | **ORACLE: a hierarchical readout's ceiling is .8242 fg_ari (+.1133) with obj_iou .7217 -- above aligned Slot's .8102 while holding both IoUs. Grid 16's same ceiling is only .7696. But perfect merging still does not reach 0.9** | 0.8242 | 0.7130 | 0.7217 |
| PV2_0106 | completed | **ORACLE: merging AND coverage together reach .9329 -- ABOVE 0.9, superadditive (+.1133 and +.1001 giving +.2220). Both are readout stages on spikes the model already makes, so a route above the goal exists with the core untouched** | **0.9329** | 0.8108 | **0.8390** |
| PV2_0098 | queued | training images 1000 vs 6000 on the published line: settles both `PV2_0092`'s broken control and the scope `PV2_0048` left | | | |
| PV2_0099 | completed | **the crop makes it HARDER, not easier (-0.087): smaller cells surface small objects, 6.32 -> 6.49 per image. And grid 16->32 costs Slot +0.007 but costs US -0.095 -- a fixed slot count is resolution-robust, an emergent group count is not** | | | |
| PV2_0100 | completed | **mask downsampling is worth +0.003, not the missing piece. AND the normalised comparison FLIPS: PV2_0090 moved our Slot reference .6656 -> .8102, so we are at .94x our baseline and peer at 1.00x theirs** | | | |
| PV2_0091 | completed | **the readout dial cannot reach 0.9: at matched extent (pred_fg .808 vs .853) we score .7718 against aligned Slot's .8102, and .7718 is the dial's maximum anywhere -- .13 below the bar. An architectural change is necessary, not merely desirable** | 0.7718 | | |
| PV2_0090 | completed | **the Slot Attention baseline was misaligned IN OUR FAVOUR: its centre crop made the prediction and target grids describe different regions. Aligned, Slot fg_ari 0.6656 -> 0.8102 and LEADS our 0.7603. Both IoU wins stand** | | | |
| PV2_0125 | completed | **BENCHMARK, pixel protocol, 1 seed, validation. Tetrominoes pixel FG-ARI .9280 -- above every published number in the synchrony family (SynCx .89, CtCAE .84) and only .0129 from grid 32's own ceiling of .9409. But its fg-IoU is .2561 UNDER that ceiling at pred_fg .3004, so FG-ARI alone would be a selected number. Multi-dSprites .2080: grid 32's ceiling there is .5619, measured, so a win was impossible before the model ran -- the earlier 'effectively 1.0' was inferred from patch purity, and purity does not predict a pair-counting metric** | 0.9280 | 0.7041 | 0.7889 |
| PV2_0126 | completed | **infrastructure: the evaluation batch size was a literal 25, which OOMs at grid 64 because the coupling matrix grows with the SQUARE of the node count (16x) while the per-node tensors grow 4x. Now --eval-batch, default 25, verified inert: .9362/.6914/.7817/.3004 reproduced exactly. FINDING: population thresholding reduces mean and std over the whole tensor INCLUDING the batch axis, so a score is only reproducible together with its batch size and image i's mask depends on which images shared its forward pass. Size of the effect unmeasured** | | | |
| PV2_0127 | completed | **BENCHMARK ROWS. CLEVR pixel, 3 seeds, 1000 TEST images: .6668 +- .0506, 79.33% of a .8406 ceiling. This REPLACES BENCHMARK_PLAN's .7142 (100 validation images, seed count unstated) and is .0474 LOWER -- so we are no longer distinguishable from Rotating Features' .65 (margin +.0168 inside one std); SynCx .59 and below still cleared. Tetrominoes: 1 of 3 independent seeds DIED in training (pred_fg .0032, spike rate .0154, collapse 4.90 vs 1.62/1.82), so there is no 3-seed mean -- .6135 describes instability and .9203 selects the arms that worked** | 0.6668 | 0.6922 | 0.5788 |
| PV2_0128 | completed | **Multi-dSprites holds 28% of its ceiling where Tetrominoes holds ~100% and CLEVR 79%, same core. SIX candidates eliminated by measurement: the grid (64 lifted the ceiling .5619->1.0000 but the score only .2080->.2817, fraction FELL to 28.2%), readout settings, sprite overlap, contrast, within-image uniformity, dataset variant. The failure is fragmentation only -- 4.24 predicted groups per true object, never merging. ALSO: FG-ARI has no partial credit on single-object images (86 of 300 here), which is why the grid-32 ceiling read .5619 against a >=2-object .7877 -- contract raised to v2. CORRECTED: the peer has NO unsupervised number; both their rows are mask-supervised at source** | 0.2817 | 0.5480 | 0.5327 |
| PV2_0046 | completed | contract diagnostic: a looser rule *lowers* fg_ari 0.724 to 0.533; cross-branch scores are not poolable | | | |

Three seeds on the test split, scored from spike masks. Peer experiments from
`patch_v2_sw` use the `SW_` prefix and are reviewed under `peer_updates/`.

## Reference

| | fg_ari | fg_iou | obj_iou |
| --- | --- | --- | --- |
| Slot Attention, ckpt-500, single published checkpoint | 0.6195 | 0.1241 | 0.0920 |

Not a 3-seed training mean, and trained on a different render; see
`baselines/slot_attention.md`.

## Where the goal stands

Three parts: exceed **0.75** `patch_fg_ari`; review and apply the peer's results; and
if 0.75 is unreachable, find a route to **0.9+** keeping SNN + Kuramoto bidirectional.

**Parts 1 and 2 are met. Part 3 is not.**

### Part 1 — met on the test split, three seeds (`PV2_0061`)

Test IDs 7000-7999, all 1000 images, scored from spike-classifier masks, spectral
k=14 (a fixed group count, never the true object count):

| metric | ours | matched Slot | absolute | relative |
| --- | --- | --- | --- | --- |
| `patch_fg_ari` | **0.7603** +- 0.0003 | 0.6663 +- 0.1108 | +0.0940 | +14.1% |
| `patch_foreground_iou` | **0.3373** +- 0.0207 | 0.1177 +- 0.0101 | +0.2196 | +186.6% |
| `patch_matched_object_iou` | **0.6598** +- 0.0061 | 0.0871 +- 0.0040 | +0.5727 | +657.6% |

Per seed: 0.7600 / 0.7605 / 0.7604. This is `PV2_0076`, the superlinear geodesic step
cost, on the same checkpoints and readout; `PV2_0061` at power 1.0 gave
0.7533 / 0.3358 / 0.6549 and the change is +0.0070 / +0.0015 / +0.0049. The previous
test read before this session (`PV2_0008`) was 0.6075, so fg_ari is **+0.1528**.

The same checkpoints with the connected-component readout give
**0.7258 / 0.7124 / 0.5029** on the same test images — also beating Slot on all three,
with foreground IoU more than twice as high but fg_ari below 0.75. **Both are reported
because k=14 costs -0.3767 foreground IoU**, and it predicts 0.556 foreground against
a target of 0.118. They are two points on one dial (`PV2_0059`, `PV2_0060`), and k was
selected on validation.

### The Slot Attention comparison is settled, and we win it

`PV2_0049` trained the official architecture on our own train split at three seeds
and evaluated each under three seeded inference draws:

| metric | ours | matched SA | absolute | relative |
| --- | --- | --- | --- | --- |
| `patch_fg_ari` | **0.7250** | 0.6635 | +0.0615 | +9.3% |
| `patch_foreground_iou` | **0.7188** | 0.1184 | +0.6004 | +507% |
| `patch_matched_object_iou` | **0.4953** | 0.0869 | +0.4084 | +470% |

Ahead on all three. Read with its budget caveat: 320 epochs against the official
457, so this is a lower bound on Slot Attention. `PV2_0047` reported the opposite
from two seeds and unseeded inference and is superseded.

### What the ceiling is, and where the loss sits

`PV2_0034` put oracle features through this pipeline unchanged and reached fg_ari
**0.9754**. The architecture is not the limit; the encoder costs 0.27.
`PV2_0035/0036` split that: within-object spread 0.465 -> 0.000 is worth +0.170
(reaching 0.8935), and the between-object margin carries the rest.

### Closed

Feature routes: DINOv2, label-free clustering, self-bootstrapping (circular),
encoder capacity (diverges), slot-term defaults (already optimal), end-to-end alone
(0.6779), frozen core (collapses). The graph's *quality* is spent — a ground-truth
coupling graph is worth +0.06 (`PV2_0005`). Data scale is closed on the score
(`PV2_0048`) and on the mechanism (`PV2_0051`, 7% ratio gain). The integration
window is saturated (`PV2_0052`, +0.0005 from 1024 to 4096).

### Open

`PV2_0057` found the first thing in this cycle that points forward. `PV2_0036` had
measured that taking within-object spread from 0.465 to zero is worth **+0.170** and
reaches 0.8935 -- enough on its own -- and all nine closed routes tried to *learn*
better features. None asked what that spread is made of. In CLEVR an object's albedo
is one of eight colours while its pixels vary with shading, and chromaticity is
invariant to a multiplicative intensity factor, so:

| input | w/b |
| --- | --- |
| current gamma, behind 0.7250 | 0.2030 |
| raw RGB patch means | 0.2144 |
| **chromaticity** | **0.1093** |
| oracle codes, fg_ari 0.9754 | 0.0000 |

A 46% cut by arithmetic. Two further things fall out: unit RGB reaches the same 0.1095
and log-domain centring does not (0.2085), so it is specifically dividing by intensity
that works; and **raw RGB patch means (0.2144) nearly match the trained encoder
(0.2030)**, so the encoder adds almost no separation over patch colour.

`PV2_0058` is training on it. The ratio is not linear in fg_ari and chromaticity
halves rather than zeroes the spread, so the expected gain is a fraction of +0.170.

`PV2_0056` closed the objective-side route from `PV2_0055`: our partition overlaps the
fixed geometric template by only ARI 0.1164, so a term penalising that agreement would
push down something already near zero.

## Diagnostics kept out of the headline

Oracle results, per the contract. Never compared against Slot Attention.

| | fg_ari | fg_iou | obj_iou |
| --- | --- | --- | --- |
| trained on a ground-truth coupling graph | 0.5370 | 0.6096 | 0.3884 |
| phase/PLV readouts, every variant | — | — | — |

## PV2_0131 — the GPU that is fast at what this job does not do

Infrastructure, no metric. I argued the coupling is bandwidth-bound and rented an H200 NVL
at 3875 GB/s for an estimated 11.8 h / $43. It ran at **14.286 s/update — 24.8 h, $90**, off
by 2.10x, because the premise came from a single bmm in isolation and the loop is 1,024
sequential steps at microbatch 1.

| | s/update | hours | one seed |
| --- | ---: | ---: | ---: |
| TITAN RTX, frontier, four processes on the card | 67.41 | 117.0 | — |
| H200 NVL | 14.29 | 24.8 | $89.98 |
| **RTX 4090** | **10.71** | **18.6** | **$8.07** |

The 4090 is **1.33x faster and 11.2x cheaper**. The counters say why: the H200 idled its
memory at **6%** while running its clock at 1785 MHz against the 4090's 2790 MHz, and
2790/1785 = 1.563x nearly accounts for the whole gap. **Sustained clock buys time here;
bandwidth, TFLOPS, DLPerf and VRAM do not.** An A100 at 1410 MHz would likely be slower than
either.

The frontier row shares its card with the peer's jobs and is not a clean hardware number.

## PV2_0132 — 253 at grid 64, 10,000 x 10 on CLEVR

**0.7794 / 0.7308 / 0.5807** on the 320-image fixed320 set, unsupervised, one seed, against
the peer's reported **0.4281 / 0.2778 / 0.2407** at 3,500 x 10 on the same grid. All three
rose together.

Scored again on the 320 **validation** images, which this benchmark had never scored:
**0.7862 / 0.7513 / 0.5808** — the two disjoint sets agree to 0.02 at worst and to 0.0001
on object IoU, so **validation can carry configuration choices from here on**.

Masks come from `trace['spikes']` via `spike_synchrony_components`, not from theta or PLV
(`evaluation.py:21-26`). `mask_opened_during_training: false`.

Three limits, all load-bearing: **one seed**; the 0.4281 baseline is the peer's reported
figure and **has not been reproduced here**; and unique images and gradient steps both rose
2.86x together, so this does not say which one mattered. The separating arm is 3,500 x 29.

---

## PV2_0134 — what the head is worth, and what a new dataset costs

`PV2_0134_head_selection_and_transfer/` — completed, seed 0, patch contract at grid 64.

Three findings, two of them negative.

**Change C fails.** Training the head to agree with itself across rollout seeds has a trivial
minimum — one candidate for every node makes both seeds agree perfectly — and the first sweep
hit it exactly, every arm reaching the identical validation FG-ARI whatever its rgb and tv
weights (0.4954 on CLEVR against 253's 0.8501). With entropy and size-distribution guards
added, **no arm on any dataset beat its own epoch 0**.

**It did expose a real defect in change B**, independent of the above because the offending
head was C's epoch 0 — before any update. Selecting on validation FG-ARI alone accepted a head at
**+0.0010 FG-ARI and −0.1141 Foreground IoU** on Tetrominoes. Rule **B′** adds a Foreground
IoU floor (within 0.01 of 253's, on validation) and makes epoch 0 a candidate so the rule can
decline to move. It changes nothing on CLEVR and moves `local_rgb` on Tetrominoes from epoch 0
(0.6717 FG-IoU) to epoch 14 (0.7858).

**The head does not transfer; the core does.** One head fitted on CLEVR, loaded unchanged,
loses on all three metrics on both other datasets (Tetrominoes −0.0321 / −0.1068 / −0.0349).
That is the architecture the generalisation claim already assumed, but it has to be said the
right way round.

Under B′, on the same 320 images with the same evaluation function:

| | fg_ari | fg_iou | obj_iou |
| --- | ---: | ---: | ---: |
| Slot Attention (one published checkpoint, best of four background rules) | 0.3747 | 0.2219 | 0.1165 |
| 253 core alone | 0.8421 | 0.7545 | 0.6410 |
| 253 core + head | **0.8507** | **0.7830** | **0.6709** |

A new dataset costs 256 unlabelled images and about a minute — **plus a labelled validation
split to stop at the right epoch**. Reusing CLEVR's recipe with no labels at all reproduces
the refit result on CLEVR to four decimals and improves all three on Multi-dSprites, but loses
0.0565 FG-ARI on Tetrominoes, whose own choice is epoch 2 rather than 11. A label-free
stopping rule is the next experiment.

Slot Attention's low score is **not** a background-rule artifact: all four rules give the
identical FG-ARI, because they only touch patches the truth calls background and FG-ARI
excludes those. Its checkpoint uses 6.99 of 7 slots on every image with near-equal shares —
it is tiling, not segmenting, and the same pattern appears in the older centre-crop run, so it
is not caused by our frame choice. It was trained on the original CLEVR render, not
`clevr_with_masks`; **retraining it on our render is the largest remaining fairness gap.**


---

## PV2_0135 — the rollout is deterministic, so PV2_0134's change C is withdrawn

`PV2_0135_rollout_is_deterministic/` — completed. A correction, not a new result.

One image rolled out three times at seeds 0, 777003 and 12345 from the same checkpoint in eval
mode gives **bitwise identical** `gamma`, `q`, `adjacency`, `trace['spikes']` and
`trace['component_spikes']` — max difference 0.000e+00 on every tensor, over a full
`(1, 4096, 1024)` spike trace of 290,599 spikes. `FreshModel.forward` seeds both generators but
nothing downstream draws from them.

So PV2_0134's change C could not have worked: `features_b` and `gamma_b` in the cache are
identical to the first pass, and `regions_b` is `regions_all` because `cache_seed2.py` passes
`np.ones(...)` as the merge support. C's symmetric KL compared a tensor with itself.

Withdrawn: C's result and the claim that cross-seed agreement does not help. Unaffected:
change A's isolation, rule B′ and its motivation (the −0.1141 Foreground IoU head was C's
**epoch 0**, before any training), the transfer result, the recipe result, the Slot Attention
rule sweep, and the three-dataset table — none of them read that cache.

Kept as a positive: determinism makes every cache, diagnostic and score in this project exactly
reproducible from checkpoint plus image, with no seed to record and no averaging over draws —
the opposite of Slot Attention, which resamples its slots on every forward pass.


---

## PV2_0137 — change C, finished: mirror invariance is harmful, not just unhelpful

`PV2_0137_change_C_mirror/` — completed. **Closes the B → A → C sequence.**

PV2_0135 withdrew the first change C because the rollout is deterministic in `inference_seed`.
A horizontal mirror is an invariance that does exist: the encoder is a learned CNN, the Kuramoto
coupling carries a learned direction term over ordered node pairs, and the spike clustering then
runs on different activity.

The premise was verified before the experiment was believed, which is the lesson of PV2_0135.
Un-mirroring the mirrored rollout and comparing: features differ by **0.89 relative**, gamma by
0.23–0.26 — genuinely different runs. But the region counts warn what is coming: **3 coherent
regions become 273**, and foreground agreement after un-mirroring is only 0.91 / 0.63 / 0.46 of
patches on three sample images. Under mirroring the core's synchrony structure fragments.

Six arms, 21 epochs, collapse guards retained, rule B′ selecting:

| dataset | arms selecting epoch 0 | epochs passing the Foreground IoU floor |
| --- | ---: | ---: |
| CLEVR | 6 of 6 | 1 of 21 |
| Tetrominoes | 6 of 6 | 0 of 21 |
| Multi-dSprites | 6 of 6 | 1 of 21 |

Not one trained epoch under any weighting survived the floor on Tetrominoes. So the result is
stronger than "does not help": **mirror-agreement training destroys figure-ground separation
within one epoch**, which is what the 3 → 273 fragmentation predicts.

Two things this settles. The head direction is open but **invariance is not the route through
it** — what works is change A, rule B′ and the fixed label-free recipe. And the core's grouping
is **strongly orientation-dependent**, which is a concrete thing to try fixing with mirror
augmentation if the core is ever retrained.


---

## PV2_0141 — how much of "all nine cells improve" is actually established

`PV2_0141_how_much_is_noise/` — completed. Analysis only; no GPU, no server.

Every number we hold is seed 0, so none has an error bar, and Tetrominoes' gain was **+0.0006**
FG-ARI. The `fit5` runs recorded the full per-epoch validation curve for every arm, which gives
a lower bound on the noise: how much the score moves when nothing changes except where training
stopped, or which arm was picked.

**4 of 9 reported cells clear twice that wobble.**

| dataset | fg_ari | fg_iou | obj_iou |
| --- | :-: | :-: | :-: |
| CLEVR | clears | clears | clears |
| Tetrominoes | inside | inside | inside |
| Multi-dSprites | clears | inside | inside |

Tetrominoes clears none: its arm spread on FG-ARI is 0.0453, seventy-five times the +0.0006 we
reported. And this understates the true noise — real seed variance also redraws the encoder,
which we know matters (the three CLEVR seeds' feature std is 0.2478 / 0.1337 / 0.3533).

So **"all nine cells improve" is true in sign and misleading as a claim.** What seed 0 supports
is CLEVR on all three metrics and Multi-dSprites' FG-ARI. That is still a real generalisation
result — a CLEVR-trained core plus a label-free head improves grouping on a dataset it never
trained on — but it rests on four cells, not nine.

It also reframes the seeds: they are not a formality for the contract. **Five of our nine
headline cells are currently indistinguishable from noise**, and seeds are the only thing that
can settle them.

---

## PV2_0150 — AKOrN trained on CLEVR, tested on all three, on our exact budget

`PV2_0150_akorn_generalisation/` — completed, seed 0 on both sides.

AKOrN is the comparison that matters: same mechanism family (Kuramoto), and published 3-seed
means on exactly our three datasets that are **above ours** — Tetrominoes 88.6±1.7, dSprites
78.3±1.3, CLEVR 91.0±0.5. `PV2_0149` had only trained it on the target dataset, which does not
test our claim, because ours is a generalisation claim. So it was trained on the **same 10,000
CLEVR images**, on the same 100,000 exposures, and evaluated on all three reserved splits the
way our frozen core is.

Both sides seed 0:

| dataset | ours (CLEVR-trained) | AKOrN (CLEVR-trained) |
| --- | --- | --- |
| CLEVR | **0.8468 / 0.7574 / 0.6456** | 0.6027 / 0.3327 / 0.3750 |
| Tetrominoes, zero-shot | **0.8580 / 0.7885 / 0.8173** | 0.5676 / 0.7383 / 0.4687 |
| Multi-dSprites, zero-shot | **0.6974 / 0.8177 / 0.7628** | 0.4343 / 0.6019 / 0.5751 |

**Nine of nine cells go to us**, narrowest Tetrominoes FG-IoU (+0.0502), widest CLEVR FG-IoU
(+0.4247, +128%). AKOrN trained on Multi-dSprites *itself* reaches only 0.4562 / 0.6830 / 0.6084
there — so the margin is not an out-of-domain artefact, and **which dataset AKOrN trained on
barely matters** (0.4343 vs 0.4562, inside its own seed spread).

A second AKOrN seed came free (its default 1234, from runs launched before `--seed` was passed)
and bounds its noise at this budget: FG-ARI spread **0.1339 / 0.0535 / 0.0094** on CLEVR /
Tetrominoes / Multi-dSprites. **Every margin above exceeds that spread**, by 1.8× on CLEVR and
5× and 28× on the others. It is two AKOrN seeds against one of ours, not a 3-seed comparison.

**The limit is the budget, and it is large.** AKOrN's recipe is 300 epochs over ~70,000 CLEVR
images, about **21,000,000 exposures — 210× what we gave it**, the same shape of caveat
`PV2_0149` carries for Slot Attention at 1/320. This licenses a **data-efficiency** claim, now
against a much stronger baseline, and nothing about who wins at 21M exposures. Protocol B for
AKOrN has not been run.

Its own `eval_obj.py` FG-ARI and ours agree to ≤0.0010 on every cell, which is the check that
the headline table is not an artefact of our scorer. Two new helpers make that possible and are
now in `evaluation_howto/`: `to_grid64.py` (patch majority vote, for a prediction *finer* than
the grid) and `bg_any.py` (`sa_bg.py` generalised to any dataset and any hard partition).

---

## PV2_0151 — the whole baseline field at our budget, nine of nine against the best per cell

`PV2_0151_baseline_field/` — completed, seed 0 everywhere.

PV2_0150 beat one baseline, and one baseline can be the wrong one. ItrSA and AKOrN (conv) live in
the same repo, so adding them costs nothing, and the bar becomes the **best baseline in each cell
separately** — a composite no single model achieves.

All CLEVR-trained, 100,000 exposures, our evaluation, the same reserved 320 ids. Our rows are the
**core alone, no head**.

| model | CLEVR | Tetrominoes (zero-shot) | Multi-dSprites (zero-shot) |
| --- | --- | --- | --- |
| **ours, 253 core** | **0.8421 / 0.7545 / 0.6410** | **0.8580 / 0.7885 / 0.8173** | **0.6974 / 0.8177 / 0.7628** |
| AKOrN (attn) | 0.6027 / 0.3327 / 0.3750 | 0.5676 / 0.7383 / 0.4687 | 0.4343 / 0.6019 / 0.5751 |
| AKOrN (conv) | 0.6046 / 0.4219 / 0.2769 | 0.7202 / 0.7540 / 0.5913 | 0.4055 / 0.6834 / 0.5434 |
| ItrSA | 0.5794 / 0.2787 / 0.3853 | 0.6319 / 0.7137 / 0.5687 | 0.4817 / 0.2489 / 0.6093 |
| Slot Attention | 0.2872 / 0.2220 / 0.0801 | 0.1739 / 0.2192 / 0.1071 | 0.1930 / 0.0876 / 0.0680 |
| **best baseline per cell** | 0.6046 / 0.4219 / 0.3853 | 0.7202 / 0.7540 / 0.5913 | 0.4817 / 0.6834 / 0.6093 |
| **our margin** | +0.2375 / +0.3326 / +0.2557 | +0.1378 / **+0.0345** / +0.2260 | +0.2157 / +0.1343 / +0.1535 |

Slot Attention is **last in all nine**, and its zero-shot rows are *worse* than its CLEVR one
(0.1739 and 0.1930 against 0.2872): it calls 84% and 85% of the image foreground against truths
of 24% and 9%. Its checkpoint needed no retraining — the PV2_0149 CLEVR one, run as-is, which is
the condition our own transfer rows are in.

**Which baseline is second depends on the dataset** — conv wins Tetrominoes, ItrSA wins
Multi-dSprites, attn wins nothing outright. That is what models look like when they have not
learned anything transferable. Ours is first in all nine, and the only row whose predicted
foreground tracks the truth on CLEVR (0.2001 vs 0.2115; every baseline over-claims by 2–3.4×).

**The one cell to distrust:** Tetrominoes Foreground IoU, **+0.0345**, with no seed spread
measured for AKOrN (conv).

**And one place the budget excuse runs out.** AKOrN (conv) at 1/210 of its budget **already
matches its published CLEVR number** (60.5 vs 59.0±4.3) and nearly its Tetrominoes one (72.1 vs
76.4±0.8). So beating it on Tetrominoes is not a data-efficiency artefact — it is the first cell
in this line of work where "give it its full budget" does not answer. ItrSA (58.1 vs 82.5±1.7)
and AKOrN attn (60.4 vs 91.0±0.5) are still starved, and there the caveat stands.

---

## PV2_0152 — cont5000 beats 253 on FG-ARI on both datasets, and costs CLEVR object IoU

`PV2_0152_cont5000_tetrominoes/` — completed. One core seed each side; ± is head
re-initialisation variance only.

The head direction is closed on Tetrominoes (`PV2_0146`), so the instruction was to beat 253
through the **core**. `cont5000` continues the finished CLEVR core for **5,000 × 10 = 50,000 more
exposures on Tetrominoes**, nothing else changed.

**Tetrominoes test 320** — all three up:

| metric | base core | base +head | cont core | cont +head | Δ core |
| --- | --- | --- | --- | --- | --- |
| fg_ari | 0.8580 | 0.8560±0.0033 | **0.8675** | **0.8665±0.0014** | **+0.0095** |
| foreground_iou | 0.7885 | 0.7888±0.0005 | **0.8193** | **0.8206±0.0008** | **+0.0308** |
| matched_object_iou | 0.8173 | 0.8165±0.0016 | **0.8213** | **0.8235±0.0010** | +0.0041 |

**CLEVR fixed320** — the surprise, and the cost:

| metric | base core | base +head | cont core | cont +head | Δ core |
| --- | --- | --- | --- | --- | --- |
| fg_ari | 0.8421 | 0.8468±0.0004 | **0.8505** | **0.8526±0.0008** | **+0.0085** |
| foreground_iou | 0.7545 | 0.7575±0.0003 | 0.7452 | 0.7468±0.0022 | −0.0093 |
| matched_object_iou | 0.6410 | 0.6460±0.0005 | 0.6207 | 0.6252±0.0024 | **−0.0203** |

**It beats 253 on FG-ARI on both datasets at once** (+0.0095, +0.0085), which is what was asked,
through the core, with the head closed. The accepted trade is CLEVR FG-IoU −0.0093. **The loss
not covered by that trade is CLEVR matched-object IoU, −0.0203** — the largest movement in either
table, 4× the FG-IoU loss.

**Mechanism: it merges.** Mean regions per image fall on *both* datasets — Tetrominoes 21.1 →
17.9, CLEVR 8.0 → 6.6, against a truth of about 3. On Tetrominoes that removes spurious
splinters and both grouping and coverage improve. On CLEVR it has begun joining objects to each
other: FG-ARI (a pairwise partition measure) rises, while matched-object IoU — the one metric
that punishes two true objects landing in one region — falls. Same cause, opposite sign.

The Tetrominoes FG-IoU gain is the most robust number here: +0.0308 on test, independently
+0.0328 on validation 160 before the evaluation split was touched.

---

## PV2_0153 — continuing on Multi-dSprites beats 253 on all six cells, and refutes PV2_0152

`PV2_0153_cont5000_multi_dsprites/` — completed. One core seed each side; ± is head
re-initialisation variance only.

The identical procedure as `PV2_0152` — the finished CLEVR core, +5,000 × 10 = 50,000 more
exposures — but on **Multi-dSprites**. PV2_0152 paid for its gains with CLEVR matched-object IoU
(−0.0203) and explained that by merging. This branch pays nothing.

| | Multi-dSprites test 320 | CLEVR fixed320 |
| --- | --- | --- |
| base core | 0.6974 / 0.8177 / 0.7628 | 0.8421 / 0.7545 / 0.6410 |
| **cont core** | **0.7032 / 0.8296 / 0.7758** | **0.8509 / 0.7592 / 0.6671** |
| Δ core | **+0.0058 / +0.0119 / +0.0130** | **+0.0088 / +0.0047 / +0.0261** |
| base + head | 0.7037±0.0011 / 0.8234±0.0027 / 0.7687±0.0023 | 0.8468±0.0004 / 0.7575±0.0003 / 0.6460±0.0005 |
| **cont + head** | **0.7063±0.0009 / 0.8306±0.0015 / 0.7776±0.0008** | **0.8543±0.0003 / 0.7609±0.0004 / 0.6723±0.0007** |

**Six of six up, and the largest gain is the cell PV2_0152 lost** — CLEVR matched-object IoU
+0.0261 here against −0.0203 there. All three CLEVR head deltas clear twice the head-seed std.
**`cont5000(Multi-dSprites)` + head is the best configuration in this project: CLEVR
0.8543 / 0.7609 / 0.6723.**

**PV2_0152's mechanism is withdrawn as a general claim.** Mean regions on CLEVR: base 8.0,
Tetrominoes branch **6.6** (object IoU fell), Multi-dSprites branch **9.9** (object IoU rose).
Continued training does not reduce fragmentation as such — it did on that branch. What the two
share is the FG-ARI direction (+0.0095, +0.0088 on CLEVR); everything else depends on the
continuation dataset. Multi-dSprites has 2–5 regions per image against CLEVR's 3–11; Tetrominoes
has exactly 4 every time, and it is the branch that regressed.

**The head direction, closed on Tetrominoes (`PV2_0146`), is open on this core**: it clears its
own noise in five of six cells here against three of six on the base core, and in all three on
CLEVR.

Still missing: **this core on Tetrominoes, and the Tetrominoes core on Multi-dSprites.** Each
continued core has been scored on its own dataset and on CLEVR only, so neither can enter the
three-dataset headline table yet.

---

## PV2_0156 — both continued cores on all three datasets: neither replaces 253

`PV2_0156_cont5000_three_datasets/` — completed, one core seed.

Each branch had been scored on its own dataset and CLEVR only. The two missing cells, filled:

| core | CLEVR Δ | Tetrominoes Δ | Multi-dSprites Δ |
| --- | --- | --- | --- |
| cont5000(Multi-dSprites) | +0.0088/+0.0047/+0.0261 | **−0.0129**/+0.0114/**−0.0066** | +0.0058/+0.0119/+0.0130 |
| cont5000(Tetrominoes) | +0.0084/**−0.0093**/**−0.0203** | +0.0095/+0.0308/+0.0040 | **+0.0168**/+0.0012/**−0.0067** |

**Neither wins all nine**, so under the fixed criterion neither replaces 253. `PV2_0153`'s
"best configuration this project has" is **withdrawn**: with Tetrominoes measured, that core loses
two cells.

**On the stated priority the better core is the other one.** FG-ARI first, Foreground IoU may give
a little: `cont5000(Tetrominoes)` raises FG-ARI on **all three** datasets (+0.0084 / +0.0095 /
+0.0168) at the cost of CLEVR Foreground IoU −0.0093 and matched-object IoU −0.0203. `PV2_0152`
had treated it as disqualified; that is withdrawn too.

**And one result neither branch predicted:** a core continued on **Tetrominoes** improves
**Multi-dSprites** more (+0.0168) than a core continued on Multi-dSprites does (+0.0058) — the
largest FG-ARI gain in the test, and off-diagonal. That is evidence against reading `cont5000` as
domain adaptation.

---

## PV2_0157 — object-count accuracy: the head halves it, and Tetrominoes never gets it right

`PV2_0157_object_count/` — completed. The measurement no contract metric provides.

True objects per image: CLEVR **6.36**, Tetrominoes **3.00**, Multi-dSprites **2.49**.
(`PV2_0152` and `PV2_0153` both gave CLEVR as "~3.0" — Tetrominoes' figure, carried across by
mistake. Both corrected.)

| core | dataset | truth | pred | exact | MAE |
| --- | --- | --- | --- | --- | --- |
| base 253 | CLEVR | 6.36 | 8.04 | 0.175 | 2.88 |
| base 253 | **Tetrominoes** | 3.00 | **21.07** | **0.000** | 18.07 |
| base 253 | Multi-dSprites | 2.49 | 6.62 | 0.281 | 4.56 |
| cont5000(Tetro) | CLEVR | 6.36 | **6.65** | 0.184 | **1.99** |
| cont5000(mdS) | CLEVR | 6.36 | 9.90 | 0.144 | 4.20 |
| base 253 **+head** | CLEVR | 6.36 | **12.90** | **0.035** | 6.59 |

**Three findings.** (a) **The head buys its metric gains by splitting**: it raises the predicted
count on every row without exception and cuts `exact` three- to five-fold. Its gains on the three
contract metrics are +0.003 to +0.026, and none of those metrics can see this cost. (b)
**Tetrominoes is a total failure at counting** — our best FG-ARI cell (0.8580) predicts 21 objects
against a truth of exactly 3, `exact` precisely 0.000 across all 320 images. "The model groups the
patches correctly" is true there; "the model finds the objects" is not. (c) **Count accuracy and
matched-object IoU are inverted** on CLEVR (6.65/1.99 → worst obj-IoU; 9.90/4.20 → best), which
settles `PV2_0152`'s withdrawn mechanism directly: the branch that loses obj-IoU has the *most
accurate* count, so it cannot be merging objects away.

**Why this is expected.** `model.py` and `loss.py` contain **no decoder and no reconstruction
term** (zero matches for decoder/reconstruct/deconv/upsample); the objective is
`primary + 5.*spike` over PLV structure and spike synchrony, and for 253 `self.head` is `None`.
Every baseline we compare against reconstructs the image, and a reconstruction loss penalises
splitting one object in two. Ours contains no such pressure, so nothing stops one tetromino
becoming seven mutually-synchronous fragments. The count errors are a consequence of the design.
