# STATUS

Updated 2026-10-12 01:30. Branch `patch_v2`. Contract v1, split v1.

## Where things stand

Active goal, three parts: exceed **0.75** `patch_fg_ari`; review and apply the peer's
results; and if 0.75 is unreachable, find a route to **0.9+** keeping SNN + Kuramoto
bidirectional.

**Parts 1 and 2 are met. Part 3 is not.**

| goal condition | state |
| --- | --- |
| fg_ari 0.75 | **met** — 0.7533 on test, 3 seeds (`PV2_0061`) |
| beat Slot Attention on all three | **met** on test: +0.0870 / +0.2180 / +0.5678 |
| apply the peer's positive results | **done** — graph freeze ported (`PV2_0044`), 0.7059 -> 0.7250 |
| a demonstrated route to 0.9 | **not met**; ten feature routes closed |

### The test result (`PV2_0061`), read once

Test IDs 7000-7999, **all 1000 images**, from classifier masks on the model's own
spikes, spectral k=14 with one background cluster.

| metric | ours k=14 | matched Slot | absolute | relative |
| --- | --- | --- | --- | --- |
| `patch_fg_ari` | **0.7533** +- 0.0015 | 0.6663 +- 0.1108 | +0.0870 | +13.1% |
| `patch_foreground_iou` | **0.3358** +- 0.0215 | 0.1177 +- 0.0101 | +0.2180 | +185.2% |
| `patch_matched_object_iou` | **0.6549** +- 0.0042 | 0.0871 +- 0.0040 | +0.5678 | +651.6% |

Per seed, ours: 0.7547 / 0.7536 / 0.7517. Slot: 0.7265 / 0.7340 / 0.5385.

The same checkpoints read through connected components give **0.7258 / 0.7124 /
0.5029** on the same images — also beating Slot on all three, foreground IoU twice as
high, fg_ari below 0.75.

**What k=14 costs:** -0.3767 foreground IoU against that configuration, and predicted
foreground of 0.556 against a target of ~0.118. fg_ari is computed over true-foreground
patches only, so a wider predicted foreground can only help it; `PV2_0059` and
`PV2_0060` map the whole dial (k=12 fg_ari 0.7381 at fg_iou 0.4413; k=20 fg_ari 0.7718
at fg_iou 0.1680). k was chosen on validation, which the contract permits.

**Evidence the score comes from spikes:** the spectral readout clusters
`_spike_affinity(spikes, components)`, `--readout` stays at its default `spikes`, and
`PV2_0043` measured binary spike timing carrying that affinity at r = 0.9822
(0.9808 / 0.9807 on these checkpoints). `--spectral-k 14` is a fixed group count;
`--spectral-oracle-k` is the oracle variant and is not used.

**Limitations:** the three seeds share one graph through `PV2_0044`'s freeze, so they
are not three fully independent runs; the Slot baseline ran 320 epochs against the
official 457 and is a lower bound, with its weak seed 2 (0.5385) dragging the mean,
so a better-trained Slot could close the fg_ari gap while leaving both IoU gaps intact.

## The Slot Attention comparison

`PV2_0049`, the official architecture trained on our own train split, three seeds,
each evaluated under three seeded inference draws:

| metric | ours (3-seed) | matched SA (3-seed) | absolute | relative |
| --- | --- | --- | --- | --- |
| `patch_fg_ari` | **0.7250** | 0.6635 +- 0.1090 | +0.0615 | +9.3% |
| `patch_foreground_iou` | **0.7188** | 0.1184 +- 0.0093 | +0.6004 | +507% |
| `patch_matched_object_iou` | **0.4953** | 0.0869 +- 0.0037 | +0.4084 | +470% |

Ahead on all three. Three caveats travel with it: the baseline ran 320 epochs
against the official 457, so it is a **lower bound** on Slot Attention; our three
seeds **share one graph** through `PV2_0044`'s freeze, so the variance the graph
contributes is removed by construction; and this is validation, not test.

Inference noise is ~0.005 per checkpoint against 0.1090 across checkpoints, so the
weak seed is the checkpoint and not the draw. `PV2_0047` claimed the opposite from
two seeds with unseeded inference and is superseded.

## What is closed

**The architecture is not the limit.** `PV2_0034` ran oracle features through this
pipeline unchanged and reached **0.9754**; the encoder costs 0.27. `PV2_0035/0036`
split that loss: within-object spread 0.465 -> 0.000 is worth +0.170, the
between-object margin carries the rest.

**Graph quality is spent** — a ground-truth coupling graph is worth +0.06
(`PV2_0005`).

**Seven feature routes**: DINOv2 worse, label-free clustering worse,
self-bootstrapping circular (+0.006), encoder capacity diverges, slot-term defaults
already optimal, end-to-end alone 0.6779, frozen core collapses.

**Data scale**, both ways. `PV2_0048`: more images at matched compute gives 0.6779
-> 0.5983 -> 0.4790. `PV2_0051`: the scale-invariant separation ratio improved only
7% relative, so the encoder did not get better either, and the 27-hour
full-convergence run was cancelled rather than run.

**The integration window.** `PV2_0052`: +0.0005 going from 1024 to 4096, inside the
seed spread. It also confirms 0.7250 is not an artifact of the window.

## What is open — part 3, the route to 0.9

**Not demonstrated.** `PV2_0034` reached fg_ari 0.9754 by putting oracle codes through
this pipeline unchanged and unretrained, so the architecture is not the limit and the
encoder costs 0.27. Eleven routes to better features are closed: DINOv2, label-free
clustering, self-bootstrapping, encoder capacity, slot-term defaults, end-to-end
training, the frozen core, data scale, position in the graph, chromaticity, and
inference-time smoothing.

### The sharpest statement of what is missing

`PV2_0062` closed the last one but located the gap better than anything before it.
Oracle codes have within-object spread **0.0000 and between-object distance 3.9462**,
which is **2.4x** the baseline's 1.6592. Edge-preserving smoothing lowers within (to
0.0564 at its strongest, nearly the oracle's zero) but lowers **between** with it
(1.6592 -> 1.5180). So:

- every label-free method tried reduces within-object spread and **shrinks** the
  between-object margin;
- what makes the oracle code special is the **enlarged** margin, and increasing it
  without labels is close to knowing which object is which.

Two experiments now show the within/between ratio is **not** the quantity limiting
fg_ari. `PV2_0058`: both the Euclidean ratio and a cosine margin rank the chromaticity
tensors above the 8-channel gamma (which has the lowest effective rank of the three,
1.68) and the scores go the other way. `PV2_0062`: a 41% ratio improvement moves
fg_ari by -0.005 in the calibrated regime. The measures bound the problem; they do not
point at a solution, and `PV2_0051`'s use of the ratio to close data scale is weaker
than written there, though its conclusion survives on the score.

`PV2_0056` closed the objective-side route separately: our partition overlaps the
fixed geometric template by only ARI 0.1164, so penalising that agreement would push
down something already near zero.

### The one by-product worth keeping

`PV2_0062`'s smoothed gamma under the connected-component readout gives **foreground
IoU 0.7626**, the highest this project has measured (previous best 0.7251), at a
predicted foreground of **0.1166 against a target of 0.118** — better calibrated than
the baseline's 0.129. Its fg_ari is 0.7187, below 0.75, so it is not the published
configuration, but it is the best foreground segmentation available and is on disk at
`/work/USERS/tkim1/gamma_sequences/bilat_s1.0_r0.5_i3_grid16.pt`.

### Untried

A training objective geometry cannot satisfy, which would have to act on the
between-object margin rather than within-object spread. Reconstruction is the only
such signal tested (0.6779, below frozen features). The peer's cross-view spike
consistency (their SW0139) is another; at the last two fetches their branch head was
unchanged at `94de778` and that arm still reported "cross-view calibration remains
running, its training queued", so there is no result to port.

## Where the peer stands

Their branch moved to 32x32 = 1024 patches under its own contract. Three-seed
native32: **0.7819 / 0.6054 / 0.5970** against their matched Slot32 **0.7817 /
0.2044 / 0.2032** -- ahead on all three, but the FG-ARI margin is **+0.0003** while
their Slot32 seed spread is **0.1353**, so FG-ARI is a tie and the win is the IoUs.
Scores are not poolable with ours (128x128 vs 240x320, 4px vs 15x20 patches, 70,000
vs 6,000 training IDs). Of 25 new records none carries a positive, reproduced,
promotable result; their incumbent is unchanged at SW0097.

One mechanism does transfer: Slot Attention's foreground IoU is 0.2044 on their
contract and 0.1184 on ours. Across two resolutions, two label rules and two
budgets it still over-predicts foreground, and that is where both branches' real
advantage comes from.

## Resume point

Parts 1 and 2 are done and on the test split. Part 3 is the open work: a training
signal that geometry cannot satisfy, with the peer's SW0139 result the thing to read
first.

Verified and reusable: `--spectral-k` on the graph-frozen checkpoints is the fg_ari /
foreground-IoU dial (`PV2_0059`, `PV2_0060`); `graph_extra_channels` routes trailing
gamma channels to the coupling graph alone and is a no-op at 0; the evaluator reads
both gamma widths from the checkpoint and takes `--save-predictions` on both readout
branches; `run_slot_attention.py` takes a `--seed` and is bitwise reproducible.

**Every experiment script opens with a regression row** re-scoring an existing
checkpoint, and nothing below it is read unless that reproduces its recorded value.
`PV2_0055` is why: a patch broke the graph-selection chain so the learned graph was
replaced by the static one at `graph_extra_channels = 0`, scoring 0.0033 instead of
0.7239 — a bug that fired only in the configuration every existing result uses, while
the new-path smoke test passed. `PV2_0061` ran that gate on validation **before** the
test split was touched.

Compute: GPUs 2 and 3 only, 0 and 1 are the collaborator's — but check actual
occupancy before dispatch, since other users do appear on 2 and 3; an evaluation was
lost to OOM against another user's 22.5 GiB on GPU 3. Heavy runs queue behind a gate
that waits for a clear GPU.
