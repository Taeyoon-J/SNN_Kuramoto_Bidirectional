# PV2_0070 — cross-view consistency: invariance achieved, score worse

**completed. Seventeenth route closed, and the last one I had named as untested.**
Parent `PV2_0066`. Seed 0, validation 300, contract v1.

## Why this was the route left

`PV2_0066` showed the coupling graph carries 94% of the oracle's gain, reaching
0.9671 / 0.9353 / 0.7584 at predicted foreground 0.1121 when its input separates
objects. Sixteen routes failed to produce such an input, and two of them said what was
missing:

- `PV2_0064`: the between-object margin **can** be manufactured (cosine 0.2483 ->
  0.7693) but the score does not follow — the geometry is reproducible, the information
  is not.
- `PV2_0069`: one-hot codes from the model's own 0.7239-accurate groups give **0.7199**.
  The graph does not amplify its own estimate, so nothing iterative climbs.

A consistency objective is the one kind of signal that does not admit that fixed point:
two views have to **agree**, and agreeing on a wrong grouping is not free when the views
differ in the nuisance that caused the error.

## Design, from the measurements rather than from a guess

`PV2_0057` identified the nuisance: within-object spread is mostly **shading**, an
object's albedo being near-constant while its pixels vary with illumination. So view two
is view one under per-image, per-channel gain and offset jitter plus elementwise
Gaussian noise — illumination and sensor nuisance in gamma space. The partition is
unchanged by construction, so requiring the two views' phase-locking matrices to agree
asks the dynamics to be **invariant to shading rather than to encode it**. No masks, no
object counts, no second render.

The term was added **beside** the existing objective, not in place of it. The code
already records two earlier attempts falling into that trap: both switched `plv_source`
and so removed every phase term at the same time, leaving a loss that could not be told
from the absence of a gain.

Verified before the runs: one epoch gives loss 24.168 against the control's 23.524 with
`view_consistency = 0.004486` live and nonzero.

## Result

| arm | fg_ari | fg_iou | obj_iou | pred_fg | final `view_consistency` |
| --- | --- | --- | --- | --- | --- |
| **bar, no term** | **0.7239** | **0.7251** | **0.4951** | 0.1290 | — |
| weight 50 | 0.6514 | 0.6675 | 0.4620 | 0.1339 | — |
| **weight 200** | **0.6757** | 0.7101 | 0.4767 | **0.1197** | 0.001360 |
| weight 1000 | 0.3989 | 0.3591 | 0.2489 | 0.1153 | — |
| weight 200, from the trained core | 0.6751 | 0.6911 | 0.4715 | 0.1236 | 0.002252 |
| oracle features to the graph (diagnostic) | 0.9671 | 0.9353 | 0.7584 | 0.1121 | — |

**The term worked and the score fell.** `view_consistency` drops from 0.004486 at
initialisation to 0.001360-0.002252, so the dynamics really did become 2-3x more
invariant to the jitter, and every arm stayed in the calibrated regime
(0.115-0.134) — the failure is not foreground over-prediction this time. The best is
**0.6757, -0.0482 on the bar**, and weight 1000 collapses to 0.3989.

## What it says

Shading invariance is not the missing ingredient, and the direction of the result
suggests something sharper: **part of what the model currently uses is
shading-correlated**, so forcing invariance to it removes usable signal. That is
consistent with `PV2_0058`, where chromaticity — which removes the same factor
analytically — also scored below the baseline.

So the pattern across seventeen routes holds. Anything that makes the features more
regular (lower within-object spread, bigger margin, invariance to a nuisance) either
leaves fg_ari where it was or lowers it, while the oracle's gain comes from its codes
**being** instance identity. No label-free procedure tried here produces that, and the
one structural reason is in the data: these are single static renders, so nothing in the
input distinguishes two same-coloured objects except their spatial separation, which the
pipeline already uses.

## Where part 3 of the goal stands

**Feasibility established, method not found.** The architecture reaches
0.9671 / 0.9353 / 0.7584 in the calibrated regime, SNN + Kuramoto bidirectional
unmodified, when the graph's input separates objects. Seventeen routes have failed to
produce that input. This is where I stopped; the remaining ideas would need either a
dataset with motion or multiple views of one scene, or supervision the contract does not
allow.
