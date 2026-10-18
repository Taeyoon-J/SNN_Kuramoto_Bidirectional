# PV2_0102 — the 8-epoch 32x32 result does not survive three seeds

**Status:** completed (negative). **Parent:** `PV2_0101`. **Contract:** v2.
**Hypothesis:** `PV2_0101`'s seed-0 result — 32x32 at 8 epochs beating the published
16x16 on all three metrics inside calibration — holds at seeds 1 and 2.
**It does not.**

## Result, validation 300, sigma 4.0, min 2

| arm | fg_ari | fg_iou | obj_iou | groups | pred_fg | collapse |
| --- | --- | --- | --- | --- | --- | --- |
| w0 seed 0 | 0.7699 | 0.7531 | 0.7017 | 15.84 | 0.1512 | 2.58 |
| **w0 seed 1** | **0.2916** | **0.0947** | 0.2326 | 11.44 | **0.3101** | 3.70 |
| w0 seed 2 | 0.7068 | 0.6151 | 0.5677 | 11.09 | 0.1140 | 3.08 |
| w5 seed 0 | 0.7803 | 0.7807 | 0.6914 | 13.68 | 0.1444 | 2.54 |
| **w5 seed 1** | **0.2118** | 0.1978 | 0.1466 | **1.54** | 0.0319 | 2.99 |
| w5 seed 2 | 0.7617 | 0.6997 | 0.6315 | 10.07 | 0.1190 | — |

| 3-seed mean | fg_ari | fg_iou | obj_iou | pred_fg |
| --- | --- | --- | --- | --- |
| control (w0) | 0.5894 +- **0.2599** | 0.4876 +- 0.3472 | 0.5007 +- 0.2416 | 0.1918 |
| group-count w5 | 0.5846 +- **0.3230** | 0.5594 +- 0.3158 | 0.4898 +- 0.2988 | 0.0984 |
| 16x16 published | **0.7239** | **0.7251** | 0.4951 | 0.1290 |

Neither arm beats the published configuration on all three. The standard deviations —
0.26 and 0.32 on fg_ari — are larger than every effect this project has chased.

## Seed 1 fails in two different ways

Not one failure mode but two, which rules out a single numerical cause:

- **control:** predicted foreground **0.3101**, outside the calibration gate, with 11.44
  groups. Over-prediction.
- **group-count w5:** **1.54 groups per image** against 6.49 true objects, predicted
  foreground 0.0319. Near-total synchronisation — everything in one group.

## A prediction of mine that was wrong, again

I suggested in conversation that `spike_plv_collapse` at 8 epochs might predict which
seeds fail, because seed 1's control ended at 3.70 against seed 0's 2.58. It does not.
**w5 seed 1 ends at 2.99, below w0 seed 2's 3.08, and scores 0.2118 against 0.7068.**
Collapse separates nothing here.

## This is a failure mode the project has already recorded

`PV2_0012` at grid 16: **"half of a six-seed draw fails outright."** The response was
`PV2_0044`'s graph freeze, which `PV2_0081` then characterised precisely: it **"reduced
seed variance; it did not improve a seed"** — at a fixed seed, freezing the graph gives
0.7235 against 0.7239.

**That freeze has never been applied at 32x32.** It is not a score intervention, it is a
variance intervention, and variance is exactly what is wrong here.

## What survives from `PV2_0101`

The **overtraining** finding, which is seed-independent as far as it has been tested:
8 epochs beats 40 at seed 0 by +0.1406 / +0.0440 / +0.1132, and seed 2's 0.7068 also
beats the 40-epoch model's 0.6293. The 40-epoch budget is wrong. What does not survive is
that any fixed short budget is right — two of three seeds land at 0.71-0.78 and one at
0.29.

## Next

`PV2_0103`: 32x32 at 8 epochs with `--freeze-components graph`, three seeds, against this
record as the control. If the freeze does at 1024 nodes what it did at 256, the spread
collapses and the mean moves to where seeds 0 and 2 already sit. If it does not, the
8-epoch point is not usable and the 40-epoch model at 0.6293 is what 32x32 actually
offers.
