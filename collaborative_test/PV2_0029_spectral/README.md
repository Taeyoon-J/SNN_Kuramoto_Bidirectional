# PV2_0029-0030 — the peer's spectral readout, and the control that was owed

**completed. The open interpretation question is answered.** Parent `PV2_0027`.
Source: `origin/patch_v2_sw`, SW_0024/0027/0033 and the newer SW0047/SW0050/0052.
Validation 300, seeds 0/1/2.

## The control, run faithfully at last

`PV2_0026` ported SW_0028's spatial kernel for a real gain, but could not run
SW_0033's control: under a connected-component readout with the largest component
as background, a smooth kernel makes the grid one component, that component
becomes background, and zero foreground groups survive. Fixed-k spectral
clustering emits k groups whatever the affinity holds, so the control becomes
possible.

Same readout, same k, only the affinity's source differing:

| affinity | fg_ari | predicted_fg |
| --- | --- | --- |
| **spatial kernel alone** -- no model information | **0.3902** | 0.871 |
| **spike synchrony** | **0.6068** | 0.354 |
| difference | **+0.217** | |

**This branch's result is not mostly grid structure.** Geometry alone reaches
0.390 and the spikes add 0.217 on top. What the peer found -- their spatial-only
0.4913 against 0.4946 for membrane-times-spatial, nearly all of it the prior --
does not reproduce here.

## SW0047's permutation control: the kernel is purely positional

Keeping the kernel's values and permuting which patch each row belongs to:

| seed 0 | fg_ari |
| --- | --- |
| correctly aligned kernel | 0.6639 |
| **permuted kernel** | **0.0033** |
| no kernel at all | 0.6362 |

Seed 1 comes out at **-0.0050**. Scrambling the correspondence does not merely
remove the kernel's benefit, it drives grouping to chance and below -- far worse
than not using a kernel. A drop this complete (the peer saw 0.5993 to 0.3192)
says the kernel acts entirely through grid geometry, and that it has to be
correctly registered to the patches to act at all.

## The spectral readout itself: better objects, worse everything else

| seed 0 | fg_ari | foreground_iou | matched_object_iou |
| --- | --- | --- | --- |
| connected components | **0.6672** | **0.6192** | 0.4807 |
| spectral k=10, 1 background cluster | 0.6068 | 0.3779 | **0.5240** |
| spectral k=10, 2 background clusters | 0.5881 | 0.5052 | 0.5045 |

Matched-object IoU -- the weakest metric on this branch -- rises 0.4807 to
0.5240, which is the largest single move it has made. Foreground ARI and
foreground IoU both fall.

Keeping one cluster as background leaves k-1 as foreground, so at k=10 predicted
foreground is 0.354 against roughly 0.126 true; the peer hit the same wall at
0.659 against 0.217. Raising the background count fixes the extent -- 0.068 at
four clusters -- and collapses all three metrics together (0.3526 / 0.3378 /
0.3776). k=8 is worse still across seeds (0.2606 / 0.1879 / 0.1812).

**Not adopted**, since the goal metric is fg_ari and connected components hold
0.7059 against roughly 0.61 here. Retained as the object-IoU track, and as the
only readout under which the spatial control can be run.
