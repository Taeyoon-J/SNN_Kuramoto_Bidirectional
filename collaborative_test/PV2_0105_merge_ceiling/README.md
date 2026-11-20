# PV2_0105 — a hierarchical readout has a ceiling of 0.8242, above aligned Slot Attention, and below 0.9

**Status:** completed. **ORACLE DIAGNOSTIC — uses ground truth, never a score, reported
separately per the contract.** **Parent:** `PV2_0104`.
**Question:** `PV2_0104`'s frozen 32x32 model produces 16.5 groups per image against 6.49
true objects while reaching matched-object IoU 0.6588 — it splits objects into
well-shaped pieces. Every readout sweep so far moved a threshold or a kernel width
(`PV2_0091`, `PV2_0095`, `PV2_0096`). A **hierarchical** readout — form groups, then merge
groups — has never been tried. What is its ceiling?

## Two oracle powers, separated

A first version conflated two things and **its own check caught it**: foreground IoU
changed, which merging alone cannot do. The cause was that a predicted group lying
entirely on background was relabelled to 0 and therefore deleted — a foreground
correction, not a merge. Separated:

- **merge only** — every predicted group keeps its foreground status; groups landing on
  the same true object get one label; a group on background keeps its own label. Foreground
  IoU is then unchanged by construction, which the output verifies.
- **merge + oracle drop** — the earlier behaviour, which also deletes background groups.

## Result, three frozen 32x32 seeds, validation 300

| | fg_ari | fg_iou | obj_iou | groups |
| --- | --- | --- | --- | --- |
| as is | 0.7109 | 0.7130 | 0.6588 | 16.50 |
| **merge only** | **0.8242** | 0.7130 | **0.7217** | 11.94 |
| merge + oracle drop | 0.8242 | 0.8150 | 0.7217 | 6.17 |

Per seed the merge gain is +0.0969, +0.1236, +0.1193 — consistent, not one seed's luck.

**Merging alone is worth +0.1133 fg_ari and the foreground drop adds +0.0000 to it.** The
suspicion that the first version's ceiling was contaminated was wrong: `patch_fg_ari` is
computed over **true-foreground** patches, so deleting predicted foreground that sits on
true background cannot move it. It moves only foreground IoU, by +0.1020. Separating them
was still right — it is how that became a fact rather than an assumption.

## What this means

| | fg_ari | fg_iou | obj_iou |
| --- | --- | --- | --- |
| 16x16 published | 0.7239 | 0.7251 | 0.4951 |
| aligned Slot Attention (`PV2_0090`) | **0.8102** | 0.1151 | 0.1025 |
| **32x32 + perfect merging (ceiling)** | **0.8242** | 0.7130 | **0.7217** |

A perfect label-free merging step would **beat aligned Slot Attention on fg_ari** while
holding foreground IoU 6x and per-object IoU 7x above it. That is above every number
measured on this data.

The same ceiling at grid 16 is **0.7696, a gain of only +0.0457**. So 32x32 has **2.5x
the headroom from merging** that 16x16 has — "there is something to merge at 32 and not at
16" is now a measurement rather than a reading of figures.

## And it bounds the route

**Perfect merging reaches 0.8242, not 0.9.** With a perfect foreground correction on top
it is 0.8242 / 0.8150 / 0.7217. So this line has a hard ceiling below the goal: reaching
0.9 needs the **base groups** to improve, not only their merging.

That is the first quantitative bound this project has on a route it is actively pursuing,
rather than on a route already closed.

## Next

Build a label-free merging step and see how much of the 0.1133 it collects. The
information it would need is already in the model — the groups come from spike
correlation, so a second stage can agglomerate groups by their mutual spike correlation
without any new signal. `PV2_0069`'s fixed point is the risk: the graph did not amplify
its own estimate there, and a readout-side merger may inherit that.

## Limitations

Oracle throughout. Validation 300, contract v2, three seeds that share one graph.
`merge only` leaves 11.94 groups rather than 6.17 because background-only groups are kept
to preserve foreground IoU; a real merger would face both problems at once.
