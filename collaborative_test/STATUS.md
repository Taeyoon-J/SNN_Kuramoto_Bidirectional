# STATUS

Updated 2026-10-11 22:30. Branch `patch_v2`. Contract v1, split v1.

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
| a demonstrated route to 0.9 | **not yet**; the last candidate closed (`PV2_0055`) |

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

Nothing with a measured candidate. `PV2_0055` tested the position route that
`PV2_0053` opened and it **cost 0.168 at best** (0.5556 against the 0.7239 bar;
0.4703 at w=0.5 and 0.4575 at w=2).

The informative part is *how* it failed. The trained losses came out **lower** than
the baseline's -- 13.91 at w=0.5 and 13.75 at w=2 against 17.86 -- so the model
satisfied its objective better while scoring worse. The PLV terms ask for groups
that are internally coherent and roughly the right size, and spatially compact blobs
satisfy that without tracking objects at all. `PV2_0043` measured geometry alone at
**0.3902** under this readout, and PG_w2 landed at 0.4575 with 9.05 groups and 0.31
predicted foreground, close to that level.

**So the training objective, not only the encoder, limits the features.** That is a
reason the other seven feature routes may share. `PV2_0034` reached 0.9754 by handing
oracle features to the existing, *unretrained* core, so the readout can exploit good
features -- but no objective tried so far creates them, and this one can be satisfied
without them.

A route to 0.9 needs a training signal that geometry cannot satisfy. Reconstruction
is the only such signal tested (`PV2_0039`, 0.6779, below frozen features). The peer
is running cross-view spike consistency (their SW0139): two views of one image must
agree on grouping, which needs no masks and which compact blobs do not satisfy. If
their arm beats its own-loss control it is the first peer result in this cycle worth
porting -- see `peer_updates/REVIEW_0004_20261011.md`.

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

No experiment is running. The next move is a training signal geometry cannot
satisfy; the peer's SW0139 cross-view consistency result is the thing to read first.

Verified and reusable: `graph_extra_channels` routes trailing gamma channels to the
coupling graph alone (`hyperparameter.py`, `s2net_cls.py`,
`training/train_s2net_core.py`, `collaborative_test/evaluate_model.py`), at 0 it is
the original path, and three checkpoints reproduce 0.7239 / 0.7251 / 0.4951,
0.7235 / 0.7164 / 0.4935 and 0.7275 / 0.7148 / 0.4974 to four decimals. The
evaluator reads both gamma widths from the checkpoint. `run_slot_attention.py` takes
a `--seed` and is bitwise reproducible under it. Position-augmented gamma tensors
are at `/work/USERS/tkim1/gamma_sequences/posgamma_w{0.5,1,2}_grid16.pt`.

**Every experiment script starts with a regression row** that re-scores an existing
8-channel checkpoint, and nothing below it is read unless that reproduces its
recorded value. `PV2_0055` is why: the first version of its patch broke the graph
selection chain so that the learned graph was replaced by the static one at
`graph_extra_channels = 0`, scoring 0.0033 instead of 0.7239 -- a bug that fired
only in the configuration every existing result uses, while the new-path smoke test
passed.

Compute: GPUs 2 and 3 only, 0 and 1 are the collaborator's -- but check actual
occupancy before dispatch, since other users do appear on 2 and 3; an evaluation was
lost to OOM against another user's 22.5 GiB on GPU 3. Heavy runs queue behind a gate
that waits for a clear GPU.
