# PV2_0009 — re-balance the objective, because optimising it better scores worse

**completed (round 1).** Parent: `PV2_0008`. Contract v1, split v1. Seed 0,
validation 300 images. Screening: nothing here is a score.

## Why the objective and not the optimiser

`PV2_0008` reached foreground IoU 0.6755 on test, short of 0.700, and seed 0 was
the drag -- fg_ari 0.481 against 0.643 and 0.698. The first reading of that was
instability, and it was wrong twice over.

`--grad-clip-norm` defaults to **1.0**, so gradient clipping was already active in
every run in this line, weight 3's divergence included. And the loss traces say
the opposite of undertraining:

| seed | final loss | foreground IoU | fg_ari |
| --- | --- | --- | --- |
| 0 | **15.31** (lowest) | **0.603** (worst) | **0.481** (worst) |
| 1 | 16.67 (highest) | **0.732** (best) | 0.643 |
| 2 | 15.47 | 0.692 | 0.698 |

**Lower training loss goes with worse metrics.** Seed 0 minimised the objective
best and scored worst, so the objective is misaligned with the metric rather than
badly optimised, and making it easier to minimise is the wrong direction.

## Change

The four PLV term weights, one at a time, screened on seed 0 because seed 0 is
what holds the mean down.

## Result

Seed 0, validation 300, sync 0.50 (`fg_ari` prefers higher thresholds than
`foreground_iou` does):

| arm | fg_ari | foreground_iou | matched_object_iou | predicted_fg |
| --- | --- | --- | --- | --- |
| baseline | 0.4982 | 0.5605 | 0.3740 | 0.1442 |
| balance 10 -> 2 | 0.6359 | 0.5551 | **0.4308** | 0.1988 |
| **bimodality 1 -> 3** | **0.6390** | **0.5900** | 0.4186 | 0.1606 |
| coherence 0.5 -> 2 | 0.3130 | 0.4021 | 0.2299 | 0.0658 |

Raising bimodality lifts **all three** metrics, fg_ari by +0.141. Raising balance
lifts fg_ari about as much but gives back a little foreground IoU. Raising
coherence collapses everything.

Losses do not compare across these arms -- changing a weight changes what the
number means -- so only the metrics are compared. `balance 2` has the lowest loss
of the three and is not the best arm.

## What is not yet known

This is **one seed**, and seed 0 is specifically the seed that was anomalously
low. Two readings fit the same data and they differ in whether the goal is
reached:

- it repaired something peculiar to seed 0, so seeds 1 and 2 barely move and the
  fg_ari mean goes 0.611 -> about 0.667, short of 0.700
- the setting is genuinely better, all three seeds rise, and the mean lands near
  0.75

Nothing in this experiment separates them. `PV2_0010` trains bimodality 3 at
seeds 1 and 2, which is the only thing that can.

## Also running

Three further seeds of the unchanged setting, to judge the loss-metric
anti-correlation on six points rather than three. Seed 3 already finished at loss
42.2 against 15.3-16.7 for seeds 0-2, so it is a strong test: if the pattern
holds, that run should score well despite the much higher loss.
