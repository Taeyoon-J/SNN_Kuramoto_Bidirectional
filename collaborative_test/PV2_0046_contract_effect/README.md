# PV2_0046 — the peer gap is the contract, but not in the direction assumed

**completed. DIAGNOSTIC ONLY; the contract is unchanged.** Parent `PV2_0044`.
Graph-freeze seeds, validation 300, window 1024, kernel sigma 1.0, sync 0.05.

## Why this was measured

The peer reached a 3-seed FG-ARI of 0.7814 where this branch reached 0.7250, and
the obvious question is whether that is the model or the measurement.

The decisive fact is that **the same official Slot Attention checkpoint scores
FG-ARI 0.6195 under this branch's contract and 0.8946 under theirs** -- 0.275 for
an identical model. Their foreground prevalence is 0.199-0.217 of patches against
this branch's 0.1146, about 1.7x denser.

Normalised to each branch's own Slot reference: this branch is at **1.17x** of
Slot, the peer at **0.87x**. So in the only comparison that holds, this branch is
ahead and theirs has not yet matched its reference. Absolute cross-branch numbers
are not comparable in either direction.

## Rebuilding the labels at several coverage thresholds

This branch's rule is strict: majority vote over each 15x20 pixel patch with ties
going to the smallest ID, so background wins any tie. Rebuilt from the same
tfrecord with a coverage threshold instead:

| rule | foreground fraction |
| --- | --- |
| **contract v1** (majority, ties to background) | **0.1146** |
| min_frac 0.50 | 0.1125 |
| min_frac 0.25 | 0.1558 |
| min_frac 0.10 | 0.1880 (peer-like) |

## The hypothesis was wrong

The expectation was that denser foreground makes grouping easier, so a looser rule
would raise this branch's score and explain the peer's lead. It does the opposite:

| target (foreground) | fg_ari | foreground_iou | matched_object_iou |
| --- | --- | --- | --- |
| contract v1 (0.118) | **0.7235 / 0.7239** | 0.7164 / 0.7251 | 0.4935 / 0.4951 |
| min_frac 0.25 (0.159) | **0.5327 / 0.5355** | 0.7221 / 0.7315 | 0.4587 / 0.4625 |

**fg_ari falls 0.724 to 0.533** under the looser rule while foreground IoU barely
moves, 0.716 to 0.722.

The pattern explains itself: this model predicts about 0.129 of patches as
foreground, matched to the strict target's 0.118. The patches a looser rule adds
are **partial-coverage patches at object boundaries**, which is exactly where the
grouping is wrong -- so foreground extent stays right while the partition over
foreground gets worse. **This model is specifically tuned to the strict
contract.**

So the Slot reference gap is not explained by foreground density in this branch's
favour. The remaining candidates are the ones the peer notes themselves -- their
Slot audit evaluated a seven-slot-trained checkpoint **with 11 slots** -- plus
render and split differences.

## What this settles, and what it does not

Settled: **cross-branch absolute scores cannot be pooled**, which the peer also
states. The usable comparison is each branch against its own Slot reference.

Not settled: which labelling rule is the right one. Majority-with-ties-to-
background and a coverage threshold are both defensible. That is a decision for
the two of us together, not something to change unilaterally -- and
`evaluation_contract.md` is v1, so changing it would raise the version and require
re-evaluating both models.

**No contract change is made here**, and no score against these diagnostic
targets is a goal claim. `targets_v1.pt` remains the only scoring target.

## A resource mistake

Four of nine evaluations died with CUDA OOM. They were sent to the same GPU as
this branch's own end-to-end training, on a machine where another user holds
12-15 GiB on every card. The `min_frac 0.10` rows are therefore missing; the
`min_frac 0.25` rows already establish the direction, and they will be re-run when
a card is free rather than by displacing the training.
