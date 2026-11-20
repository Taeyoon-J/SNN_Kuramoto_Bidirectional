# PV2_0062 / PV2_0063 — features improved at inference: best foreground IoU yet, no fg_ari gain

**completed. Eleventh feature route closed for the 0.9 goal, with one real
by-product.** Parent `PV2_0034`. Inference only: no training, no weights changed.
Validation 300, contract v1, three graph-frozen checkpoints.

## The idea, and why it was not one of the ten

`PV2_0034` put oracle codes through this pipeline **unchanged and unretrained** and
reached fg_ari 0.9754, so the core already exploits better features. All ten closed
routes tried to produce them by **training**. This produces them at **inference**,
where no optimiser, objective or operating point can shift underneath the attempt.

Edge-preserving averaging on the 16x16 patch grid: patches of one object are both
neighbours and similar so they average together, while an object boundary is a feature
discontinuity the similarity weight suppresses. Weights come **only from the raw
features' own pairwise similarity** — not from the model's groups, spikes, PLV or
graph, which is what made `PV2_0037`'s self-bootstrap circular (+0.006) — and not from
masks.

## Separation: every variant improved

| variant | within | between | **w/b** |
| --- | --- | --- | --- |
| baseline gamma | 0.3369 | 1.6592 | **0.2030** |
| s1.0_r0.5_i3 | 0.1808 | 1.5180 | **0.1191** |
| s1.0_r1.0_i3 | 0.1270 | 0.9034 | 0.1406 |
| s2.0_r1.0_i3 | 0.0745 | 0.4612 | 0.1615 |
| s2.0_r2.0_i3 | 0.0564 | 0.3064 | 0.1841 |
| oracle codes (0.9754) | 0.0000 | 3.9462 | **0.0000** |

All twelve improved the ratio, the best by 41%, and the strongest smoothing takes
within-object spread to 0.0564 — close to the oracle's zero.

## Score: it splits by readout

| readout | gamma | fg_ari | fg_iou | obj_iou | pred_fg |
| --- | --- | --- | --- | --- | --- |
| connected components | baseline | 0.7239 | 0.7251 | 0.4951 | 0.129 |
| **connected components** | **smoothed** | 0.7187 | **0.7626** | 0.4673 | **0.1166** |
| spectral k=14 | baseline | 0.7569 | 0.3449 | 0.6480 | 0.542 |
| spectral k=14 | smoothed | 0.7876 | 0.1370 | 0.5969 | 0.8485 |

**The by-product is real: foreground IoU 0.7626 is the highest this project has
measured** (previous best 0.7251, +0.0375), at a predicted foreground of **0.1166
against a target of 0.118** — better calibrated than the 0.129 baseline. Strong
smoothing destroys it (s1.0_r1.0_i3 collapses to 0.4310), so it works only while the
range sigma is narrow enough to preserve boundaries, which matches the mechanism.

**The fg_ari gain is not real.** Under k=14 the smoothed gamma reaches 0.7876 but
predicts 0.8485 foreground — Slot Attention's regime. Plain gamma at k=20 already
reaches 0.7718 at pred_fg 0.8084, so at matched over-prediction smoothing is worth
about +0.016, not +0.0307. **The automated verdict in the script read fg_ari alone and
printed "TRANSFERS"; that is wrong and is corrected here.**

`PV2_0063` then swept k on the smoothed gamma to look for a point with both:

| k | fg_ari | fg_iou | obj_iou | pred_fg |
| --- | --- | --- | --- | --- |
| 8 | 0.7439 | 0.3349 | 0.5484 | 0.517 |
| 10 | 0.7617 | 0.2109 | 0.5691 | 0.685 |
| 12 | 0.7784 | 0.1618 | 0.5853 | 0.789 |
| 14 | 0.7876 | 0.1370 | 0.5969 | 0.849 |
| **settled, unsmoothed k=14** | **0.7569** | **0.3449** | **0.6480** | 0.542 |

**No point dominates.** At matched fg_ari — smoothed k=10 at 0.7617 against the
settled 0.7569 — the unsmoothed configuration is better on foreground IoU
(0.3449 vs 0.2109) and matched-object IoU (0.6480 vs 0.5691). **The published test
configuration stands unchanged.**

## What this says about the 0.9 goal

This is the **second** demonstration that the within/between ratio is not the quantity
limiting fg_ari. `PV2_0058` was the first: both the Euclidean ratio and a cosine
margin ranked the chromaticity tensors above the 8-channel gamma and the scores went
the other way. Here a 41% ratio improvement moves fg_ari by -0.005 in the calibrated
regime.

The sharper reading is in the oracle's two numbers. Oracle codes have within **0.0000
and between 3.9462** — 2.4x the baseline's 1.6592. Smoothing lowers within but lowers
between with it (1.6592 -> 1.5180). **What makes the oracle special is the enlarged
between-object margin**, and `PV2_0036` priced that term at +0.090 while within was
+0.170 *in combination with* it. No label-free method tried in eleven routes increases
between-object separation, which is close to saying it knows which object is which.

That is where a route to 0.9 would have to act, and this experiment does not provide
one.
