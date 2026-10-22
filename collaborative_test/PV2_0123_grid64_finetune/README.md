# PV2_0123 — the fine-tune does not repair the split; the falsifier fires

Status: **completed — FAILS the registered gate.** Grid 64, seed 0, 30 validation images,
contract v3. Parent `GC_w0` at grid 32; warm start from `PV2_0122`.

## Result

| grid 64, 30 images | fg_ari | fg_iou | obj_iou | groups | pred_fg |
| --- | ---: | ---: | ---: | ---: | ---: |
| replicated, no fine-tuning | 0.7066 | **0.7633** | 0.7188 | 19.70 | 0.1257 |
| **fine-tuned, 32 steps** | 0.7108 | **0.7552** | 0.7242 | **20.73** | 0.1270 |
| *parent, grid 32* | *0.7588* | *0.7629* | *0.7179* | *15.27* | *0.1281* |

Against the gate — both IoUs at or above 0.7633 / 0.7188 **and** fg_ari above 0.7066:

- fg_ari 0.7108 clears 0.7066, by **+0.0042**, which is not a movement
- matched-object IoU 0.7242 clears 0.7188 ✓
- **foreground IoU 0.7552 falls from 0.7633** ✗

**The registered falsifier was "fg_ari flat or an IoU falls → the fine-tune is not
repairing the split". Both halves of it happened.**

## The mechanism is refuted, not just the gate

`PV2_0122` identified what a fine-tune had to do: pull the group count back from 19.70
toward the parent's 15.27. Replication gives each parent node's 2×2 block four children
with identical parameters and different γ, phases diverge inside the block, and one object
splits across groups.

**The group count went the wrong way — 19.70 to 20.73.** Thirty-two steps do not reconcile
the block; they divide it further. The gap to the parent's 0.7588 is still 0.048.

## Why 32 steps, and what that does and does not settle

The first attempt asked for 4 epochs over 1000 images at batch 4 — 1000 steps — and spent
**ten hours** emitting `expandable_segments: memory mapping failed with OOM` without
finishing an epoch, because batch 4 needs 17.52 GB and GPU 2 was shared with Tier 1. It
loaded the warm start correctly and then thrashed.

Resized from the peer rather than from habit: `SW_0192` trains native64 on **TRAIN64 with
16 proposals**, so 64 images at batch 2 is 32 steps — the same order as theirs, at the
8.90 GB that was measured.

**But their 16 proposals are not comparable to our 32 steps.** `SW_0227`'s native64 sits on
their declared **mask-supervised** leaderboard; they fine-tune against a signal our contract
does not admit. Expecting the same step count to do the same work was the wrong inference,
and it is the third time this session I sized a run from habit instead of from the record.

So this closes "32 unsupervised steps repair the replication split". It does **not** close
"a longer unsupervised fine-tune repairs it" — that is untested, and at roughly 2.3 minutes
a step it is affordable to test properly.

## What stands from `PV2_0122`

The warm start itself still holds: replication alone carried the parent's foreground IoU
(0.7633 against 0.7629) and matched-object IoU (0.7188 against 0.7179) with zero gradient
steps. **Grid 64 runs and segments.** What is unproven is that fine-tuning recovers the
FG-ARI that replication costs.

## Next

A longer fine-tune, sized by measurement: steps enough to move the group count, run on a
GPU not shared with Tier 1 so the memory is not contended. The quantity to watch is the
group count, not the loss — if it does not fall below 19.70, more steps are not the answer
and route C closes on the mechanism rather than on the budget.
