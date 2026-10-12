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
| `patch_fg_ari` | **0.7533** +- 0.0015 | 0.6663 +- 0.1108 | +0.0870 | +13.1% |
| `patch_foreground_iou` | **0.3358** +- 0.0215 | 0.1177 +- 0.0101 | +0.2180 | +185.2% |
| `patch_matched_object_iou` | **0.6549** +- 0.0042 | 0.0871 +- 0.0040 | +0.5678 | +651.6% |

Per seed: 0.7547 / 0.7536 / 0.7517. Previous test read (`PV2_0008`) was 0.6075, so
fg_ari is **+0.1458**.

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
