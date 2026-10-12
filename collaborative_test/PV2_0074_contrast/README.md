# PV2_0074 / PV2_0075 — the geodesic barrier is two thin walls with a free corridor

**completed. Twentieth and twenty-first routes closed, and together they give the
mechanical reason the one mechanism designed for our failure cannot fix it.** Parent
`PV2_0073`. Inference only, `BIM6_s1`, validation 300, contract v1.

## Why the contrast was worth overriding

`PV2_0072` and `PV2_0073` pinned the failure: the pipeline repairs spatially
**incoherent** damage (+0.152 at heavy random corruption) but its own errors are
spatially **coherent** object merges, and iterating its own grouping converges at
0.7274.

The generator already has a mechanism for exactly that, and `_geodesic_distance`'s
docstring states it: same-coloured objects "look alike to the feature term and sit a few
patches apart, so they stay linked. [...] Walking between them crosses background, so
the route is what tells them apart." Each step costs
`euclidean * (1 + softplus(contrast) * (1 - similarity))`.

`PV2_0071` could not vary that contrast, because `graph_generator.py:97` makes it an
`nn.Parameter` and `load_state_dict` restores the trained 2.0523. And training had no
reason to choose the merge-separating value: the PLV objective rewards bimodal, balanced,
right-sized groups and never rewards separating a merged pair. Writing the scalar
directly into the checkpoint is the only way to ask.

## `PV2_0074` — the learned value is already the best of them

| effective contrast | fg_ari | fg_iou | obj_iou | groups | pred_fg |
| --- | --- | --- | --- | --- | --- |
| 0.5 | 0.7243 | 0.7272 | 0.4927 | 5.38 | 0.1292 |
| 1 | 0.7208 | 0.7249 | 0.4906 | 5.38 | 0.1296 |
| **2.17 (learned)** | **0.7239** | **0.7251** | 0.4951 | 5.46 | 0.1290 |
| 4 | **0.7246** | 0.7223 | 0.5005 | 5.60 | 0.1285 |
| 8 | 0.7183 | 0.7184 | 0.5015 | 5.79 | 0.1275 |
| 16 | 0.6945 | 0.6990 | 0.4958 | 6.16 | 0.1257 |
| 32 | 0.6406 | 0.6675 | 0.4764 | 6.59 | 0.1224 |

Best is +0.0007, noise. Groups per image climb 5.46 -> 6.59 as the contrast rises, so
objects **do** start splitting — and fg_ari falls, so the splits are not along object
boundaries.

## `PV2_0075` — the step costs, and why

The barrier depends on adjacent-pair dissimilarity, which is all the geodesic ever sees.
`PV2_0057`'s within 0.459 against between 1.78 compares object **centroids**, a different
quantity. Measured per adjacent pair, as `1 - cosine`:

| pair type | mean | median | p90 | count |
| --- | --- | --- | --- | --- |
| within one object | 0.0331 | 0.0152 | 0.0880 | 9,782 |
| **background to background** | **0.0054** | 0.0003 | 0.0096 | 119,087 |
| object to background | **0.1743** | 0.1491 | 0.3431 | 13,742 |
| between two objects | 0.1587 | 0.1169 | 0.3539 | 1,389 |

**The wall is real** — crossing into background costs **5.27x** a shading step inside an
object, so my first guess, that the costs do not separate, was wrong.

**The corridor is the problem.** Background-to-background costs **0.0054, six times less
than a within-object step**. So the route between two same-coloured objects is two thin
walls with a nearly frictionless corridor between them, and with the learned contrast the
arithmetic is close to a tie: at 2.17 a wall crossing costs 1.378 per step, a background
step 1.012, a within-object step 1.072. A path from one object to another six patches
away is about 2(1.378) + 4(1.012) = 6.8, while crossing a six-patch-wide object costs
6(1.072) = 6.4. **Nearly identical, which is the merge.**

And the contrast cannot fix it, because it multiplies `(1 - similarity)` **uniformly**.
Raising it scales walls and shading steps together: at 32 a wall is 6.6 but a
within-object step is 2.06 against a background step of 1.17, so objects with shading
gradients split before the walls start to dominate. That is exactly the pattern
`PV2_0074` measured — more groups, lower fg_ari.

## What would be needed

A step cost that is **not uniform in dissimilarity**: superlinear in `(1 - similarity)`,
or a fixed per-step toll so a long corridor is expensive regardless of how similar its
patches are. Both are core changes to `_geodesic_distance` rather than parameter choices,
and the margin they have to work in is tight — within-object 0.0331 against wall 0.1743,
a factor of 5.27, with the window for a selective threshold between roughly 6 and 30.
`PV2_0074` already sampled that window at 8 and got 0.7183.

Part 3 of the goal remains unmet, now with an arithmetic account of why the one
mechanism built for the failure cannot reach it.
