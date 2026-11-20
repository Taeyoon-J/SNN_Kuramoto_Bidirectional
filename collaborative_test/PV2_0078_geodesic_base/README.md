# PV2_0078 — lowering the per-step toll does not help, and my arithmetic mispredicted again

**completed. Twenty-fourth route closed.** Parent `PV2_0076`. Inference only,
`BIM6_s1`, validation 300, contract v1.

## What was tried, and the prediction

`PV2_0075` named a "corridor toll" as a remaining candidate. Implementing it showed the
toll is already in the expression — the leading `1` in
`euclidean * (1 + k * dissim^power)` — and the hypothesis became that it is too **large**:
at 1 the path cost is dominated by step count rather than by what the route crosses,
diluting the only signal that separates a merge.

The arithmetic looked strong. With power 2 and contrast 8, reaching an object six patches
away costs 2(1.24) + 4(1.00) = 6.48 against 6(1.009) = 6.05 for crossing a six-patch-wide
object — a ratio of 1.07. At base 0.1 the same routes cost 1.09 and 0.654, **ratio 1.66**.

## Result: it does not transfer

| readout | base | fg_ari | fg_iou | obj_iou | pred_fg |
| --- | --- | --- | --- | --- | --- |
| cc | **1.0 (original)** | **0.7327** | 0.7255 | **0.5033** | 0.1291 |
| cc | 0.5 | 0.7312 | 0.7288 | 0.4927 | 0.1284 |
| cc | 0.2 | 0.7273 | 0.7303 | 0.4866 | 0.1283 |
| cc | 0.1 | 0.7266 | 0.7306 | 0.4853 | 0.1285 |
| cc | 0.02 | 0.7241 | 0.7313 | 0.4844 | 0.1284 |
| cc, power 1 | 0.5 / 0.2 / 0.1 | 0.7271 / 0.7224 / 0.7230 | | | ~0.128 |

**In the calibrated regime the original base 1.0 is the best of them.** Lowering the toll
trades a little fg_ari for a little foreground IoU, monotonically, and never gains.

## The script's verdict was wrong and is corrected here

It printed "THE TOLL MATTERS" from the spectral rows:

| readout | base | fg_ari | fg_iou | pred_fg |
| --- | --- | --- | --- | --- |
| k=14 | 1.0 (published) | 0.7569 | **0.3449** | **0.5423** |
| k=14 | 0.1 | **0.7709** | 0.2653 | **0.6522** |

fg_ari rises 0.0140 **and predicted foreground rises from 0.5423 to 0.6522 while
foreground IoU falls 0.0796** — the extent dial of `PV2_0059`, not a better grouping. My
calibration filter only covered the connected-component rows, because the spectral
readout always sits above 0.20, so the automated check could not catch it. **The verdict
is retracted**; by the criterion this project has used since `PV2_0064` there is no gain.

## Why the arithmetic mispredicted

This is the **fourth** time a geometric or arithmetic quantity improved without the score
following: `PV2_0058` (chromaticity, both separation measures better), `PV2_0062` (41%
ratio gain, fg_ari -0.005), `PV2_0064` (cosine margin 2.7x, nothing calibrated), and now
a path-cost ratio from 1.07 to 1.66 for -0.006.

The common thread is that these quantities are computed on **ground-truth object pairs**,
while the readout acts on whatever the dynamics produce. A barrier that separates two
objects in the cost matrix still has to survive the Kuramoto rollout, the gate, the
membrane threshold and the component extraction, and a path-cost ratio says nothing about
whether it does. `PV2_0076` remains the only change of this family that transferred, and
`PV2_0077` showed even that one works only when kept out of the objective.

## Where part 3 stands

Unmet, at test 0.7603 against 0.9. The two candidates `PV2_0075` named are now both
tested: the superlinear power transferred (+0.0070, `PV2_0076`) and the toll did not. The
remaining one — making the cost's **shape** learnable rather than its scale — would have
to be trained, and `PV2_0077` is evidence against that working, since training under the
sharper cost reached a lower loss and a far worse score.
