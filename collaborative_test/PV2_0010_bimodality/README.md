# PV2_0010 — bimodality 3 at three seeds: mostly variance, not level

**completed.** Parent `PV2_0009`. Validation 300, seeds 0/1/2, sync 0.20.

| seed | baseline fg_ari | bimodality 3 | delta |
| --- | --- | --- | --- |
| 0 | 0.4715 | 0.5972 | **+0.126** |
| 1 | 0.6590 | 0.6853 | +0.026 |
| 2 | **0.7035** | 0.6186 | **-0.085** |
| mean | 0.6113 | **0.6337** | +0.022 |

foreground IoU 0.6706 -> 0.6643 (-0.006), matched-object IoU 0.4285 -> 0.4173
(-0.011).

It lifted the weak seed and cost the strong one. What it actually did was
compress the seed range, 0.232 -> 0.088, for a mean gain of only +0.022 and small
losses on the other two metrics. Of the two readings offered in `PV2_0009`, the
first was right.
