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
| PV2_0054 | running | position channels in the code the graph compares, w in {0.5, 1, 2}, seed 0 | | | |
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

The active goal has three parts: exceed **0.75** `patch_fg_ari`; review and apply
the peer's results; and if 0.75 is unreachable in this architecture, find a route
to **0.9+** that keeps SNN + Kuramoto bidirectional.

**Best on validation**, three seeds, window 1024/512, spatial kernel sigma 1.0,
scored from classifier masks on the model's own spikes (`PV2_0044`):

| metric | value | std |
| --- | --- | --- |
| `patch_fg_ari` | **0.7250** | 0.0022 |
| `patch_foreground_iou` | **0.7188** | |
| `patch_matched_object_iou` | **0.4953** | |

Last confirmed **test** result is still `PV2_0008`: fg_ari 0.6075, foreground IoU
0.6755, matched-object IoU 0.4298. Everything since is validation only.

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

`PV2_0053` found a loss never measured before: **38.8% of validation images hold an
object pair whose centroids sit closer together than the objects are internally
spread** — pairs the graph's feature cosine cannot keep apart, because
`graph_generator.py:123` builds every edge from eight appearance channels and
**position never enters that code**. Appending coordinates takes that rate to 18.1%
at w=1 and 4.3% at w=4, against 0% for the oracle code. `PV2_0054` tests whether it
converts into fg_ari.

## Diagnostics kept out of the headline

Oracle results, per the contract. Never compared against Slot Attention.

| | fg_ari | fg_iou | obj_iou |
| --- | --- | --- | --- |
| trained on a ground-truth coupling graph | 0.5370 | 0.6096 | 0.3884 |
| phase/PLV readouts, every variant | — | — | — |
