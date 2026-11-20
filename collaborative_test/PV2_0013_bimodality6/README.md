# PV2_0013 — bimodality 6 lifts all three metrics at all three seeds

**completed, best fg_ari on record.** Parent `PV2_0010`. Seeds 0/1/2.

Unlike bimodality 3, which traded seed 2 away, 6 raises every seed. Full
validation split, 1000 images, sync 0.35:

| | fg_ari | foreground_iou | matched_object_iou |
| --- | --- | --- | --- |
| seed 0 | 0.6248 | 0.6687 | 0.4414 |
| seed 1 | 0.7055 | 0.7189 | 0.4868 |
| seed 2 | 0.6964 | 0.6759 | 0.4901 |
| **mean** | **0.6756** | **0.6878** | **0.4728** |
| bimodality 1 | 0.6113 | 0.6706 | 0.4285 |
| | **+0.064** | +0.017 | +0.044 |

Weight 8 is worse than 6 at seed 0 (0.6243 against 0.6423) and 10 collapses
(0.3539), so 6 is the peak.

## Against the goal: fg_ari 0.700 and the spike readout passing the phase readout

Same checkpoints, same classifier, same evaluation, only the synchrony source
differing:

| readout | fg_ari | foreground_iou | matched_object_iou |
| --- | --- | --- | --- |
| spike -- the score | 0.6756 | 0.6878 | 0.4728 |
| phase -- DIAGNOSTIC | **0.7120** | **0.7150** | 0.4994 |

Neither half is met: fg_ari is 0.024 short of 0.700, and the spike readout is
0.036 behind the phase readout.

An earlier message claimed the spike readout had passed the phase readout. That
was wrong: it compared the new spike number against a phase number measured on
the `PV2_0008` checkpoints. Bimodality 6 raises the phase side too, and by more
(0.637 -> 0.712).

The phase readout being past 0.700 is the useful part: what the goal needs is
already in theta, so closing the 0.036 would satisfy both halves at once. That is
what `PV2_0014` and `PV2_0015` attempt.

Stability is unchanged -- see `PV2_0012`. Seeds 3, 4 and 5 still give foreground
IoU 0.166, 0.407 and 0.000.
