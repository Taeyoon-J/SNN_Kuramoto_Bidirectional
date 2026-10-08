# STATUS

Updated 2026-10-08 02:40. Branch `patch_v2`. Contract v1, split v1.

## Where things stand

Active goal, three parts: exceed **0.75** `patch_fg_ari`; review and apply the
peer's results; and if 0.75 is unreachable in this architecture, find a route to
**0.9+** that keeps SNN + Kuramoto bidirectional.

Best on validation (300 images, 3 seeds, readout window 1024/512, spatial kernel
sigma 1.0, from classifier masks on the model's own spikes — `PV2_0044`):

| metric | value |
| --- | --- |
| `patch_fg_ari` | **0.7250** |
| `patch_foreground_iou` | **0.7188** |
| `patch_matched_object_iou` | **0.4953** |

Last confirmed **test** result is still `PV2_0008`: foreground IoU 0.6755 +-
0.0660, fg_ari 0.6075, matched-object IoU 0.4298. Everything since is validation
only; test has been read once and is not touched again until a setting is settled.

| goal condition | state |
| --- | --- |
| fg_ari 0.75 | **not met**, 0.7250, short by 0.0250 |
| beat Slot Attention on all three | **met** on validation — see below, with its budget caveat |
| apply the peer's positive results | **done**: graph freeze ported (`PV2_0044`), 0.7059 -> 0.7250 |
| a demonstrated route to 0.9 | **not yet**; one candidate open (`PV2_0054`) |

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

## What is open

`PV2_0053` measured a loss this project had never looked at.
`graph_generator.py:123` builds every graph edge from the cosine of eight
appearance channels, and **position never enters that code** — it arrives only as a
distance prior on the logits and as the readout's Gaussian kernel. Counting images
where some object pair's centroids sit closer together than the objects are
internally spread, pairs that cosine cannot keep apart:

| code | unseparable images | min pair distance |
| --- | --- | --- |
| gamma as used (8 channels) | **38.8%** | 0.7024 |
| + position w=0.5 | 28.4% | 0.7920 |
| + position w=1 | **18.1%** | 0.9466 |
| + position w=4 | 4.3% | 1.7802 |
| oracle code (fg_ari 0.9754) | **0%** | 2.4765 |

This is a different kind of information from the seven closed routes, all of which
tried to improve the **appearance** code. Slot Attention, the baseline we beat, has
always added position to its CNN features through `SoftPositionEmbed`; we never
did. SNN and Kuramoto bidirectional stay untouched — only the input encoding widens
from 8 to 10 channels.

`PV2_0054` is queued on it: weights 0.5, 1, 2 at seed 0 against the 8-channel bar
of 0.7239. Whatever wins gets seeds 0, 1, 2 and the graph freeze before any claim.

## Resume point

Read `PV2_0054`'s result in `/work/USERS/tkim1/runs/posgamma_results.txt`. If a
weight beats 0.7239 at seed 0, retrain it at seeds 0, 1, 2, then apply the graph
freeze as in `PV2_0044` and re-evaluate all three metrics.

Verified and reusable: position-augmented gamma tensors at
`/work/USERS/tkim1/gamma_sequences/posgamma_w{0.5,1,2}_grid16.pt`, appearance
channels bitwise equal to the source; `evaluate_model.py` now reads its gamma
channel count from the checkpoint, and the 8-channel path still reproduces
0.7239 / 0.7251 / 0.4951 to four decimals; `run_slot_attention.py` takes a
`--seed` and is bitwise reproducible under it.

Compute: GPUs 2 and 3 only, 0 and 1 are the collaborator's. Heavy runs are queued
behind a gate that waits for one of 2 or 3 to be clear of other users.
