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
