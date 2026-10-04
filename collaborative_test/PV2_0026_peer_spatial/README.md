# PV2_0026-0027 — ported from the peer branch: the spatial kernel, and the control

**completed, best fg_ari on record.** Parent `PV2_0020`. Source:
`origin/patch_v2_sw`, SW_0028 and SW_0033. Validation 300, readout window 1024.

The peer's absolute scores are not comparable here -- different render, split and
foreground prevalence, which they note as well -- so only mechanisms and controls
transfer, never numbers.

## SW_0028's mechanism: multiply the affinity by a spatial kernel

`exp(-d^2 / 2 sigma^2)` on the patch grid, applied to the spike synchrony matrix.

| sigma | fg_ari | foreground_iou | matched_object_iou | groups |
| --- | --- | --- | --- | --- |
| none | 0.6887 | 0.6828 | 0.4710 | 5.49 |
| 2.0 | 0.6980 | 0.6778 | 0.4817 | 5.82 |
| 1.5 | 0.7008 | 0.6761 | 0.4853 | 5.83 |
| 1.25 | 0.7037 | 0.6755 | 0.4881 | 5.87 |
| **1.0** | **0.7043** | 0.6740 | **0.4914** | **5.89** |

**fg_ari crosses 0.700** and matched-object IoU rises on all three seeds.
Foreground IoU gives back 0.009. Predicted objects per image move 5.49 -> 5.89
against a true 6.20.

## It moves the two readouts in opposite directions

| kernel | spike (the score) | phase (DIAGNOSTIC) | gap |
| --- | --- | --- | --- |
| none | 0.6887 | 0.7110 | 0.0223 |
| sigma 1.5 | 0.7008 | 0.7052 | **0.0044** |

The kernel helps the spikes and slightly hurts the phases, so the gap falls from
0.022 to 0.004. The phase readout has not yet been measured at sigma 1.0, where
the spikes are best, and that matched-kernel comparison is what decides whether
the spike readout has passed -- comparing across different sigmas would repeat a
mistake already made here.

## Stacking with the membrane fix fails

Population threshold plus the kernel gives 0.6864, below the kernel alone at
0.7008. Both act on seed 0's weakness, so they substitute rather than add -- the
same pattern as the earlier hybrid-background and spike-loss pair. The dead-seed
revival does survive stacking (s5 0.5721), so that combination is still the one to
use when correctness matters more than the mean.

## SW_0033's control does not transfer, and the question stays open

The peer's most important result is a warning: their FG-ARI lead came from this
kernel, and scoring the **kernel alone** gave 0.4913 against 0.4946 for
membrane-times-kernel, with a random permutation of the membrane affinity at
0.4814. Nearly all of it was the spatial prior, not learned binding.

That matters here because `PV2_0005` gained by tightening `graph_spatial_decay`
from 0.55 to 0.35 and everything since sits on it.

Run here, the control degenerates: the kernel alone scores 0.0033 with **zero
groups** at every sigma. The readout is connected components with the largest
component as background, so a smooth kernel makes the whole grid one component,
that component becomes background, and no foreground group survives. The peer
uses fixed-k spectral clustering, which emits k groups whatever the affinity
contains.

So only one direction is established: **this readout cannot produce groups from
spatial structure alone.** How much of 0.7043 is grid structure is still unknown,
and answering it needs the peer's spectral-k readout, which is also their actual
mechanism. That port is outstanding.
