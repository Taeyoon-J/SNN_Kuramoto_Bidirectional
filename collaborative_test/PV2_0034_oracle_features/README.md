# PV2_0034 — with perfect features this architecture reaches fg_ari 0.975

**completed. ORACLE diagnostic, reported separately, never a score. It answers
where the ceiling is.** Parent `PV2_0032`. Validation 300, seeds 0/1/2, window
1024.

## The question

`PV2_0032` showed 0.75 is unreachable by readout work: with the true object count
*and* the phase readout, fg_ari reaches only 0.7338. Every tuning axis was
exhausted. The one input never varied was the features, and the encoder feeding
everything was trained as a reconstruction autoencoder, never asked to separate
objects -- per-channel IoU against ground truth 0.079, two channels exactly
0.000.

So: if the features were perfect, would these dynamics and this readout reach
0.9?

## The oracle input

Gamma rebuilt from the ground-truth masks, each patch given the code of the
object it belongs to: **within-object feature std exactly 0.0**, between-object
cosine **-0.072**. A strict upper bound on any encoder, not a plausible one.
Nothing marks which id is background, so the readout still has to find it.

## Result

| readout | fg_ari | foreground_iou | matched_object_iou |
| --- | --- | --- | --- |
| **spike -- the scoring path** | **0.9754** | 0.7748 | 0.7533 |
| spike, no spatial kernel | 0.9633 | -- | -- |
| phase -- diagnostic | 0.9834 | 0.9370 | 0.7714 |
| **real features, for comparison** | **0.7059** | 0.6778 | 0.4903 |

Per seed on the spike path: 0.9595 / 0.9837 / 0.9831.

## Conclusion

**The dynamics and the readout are adequate; the encoder is the whole problem.**

Real features give 0.7059 and perfect features 0.9754, so **the features cost
0.27 fg_ari**. The goal of 0.9 is reachable inside SNN + Kuramoto bidirectional
without touching the oscillators, the coupling, the spiking layers, the
objective or the readout -- measured, not argued. Without the spatial kernel it
still reaches 0.9633, so even the ported spatial prior is unnecessary once the
features separate objects.

This also explains why every axis tried on this branch ran out. The graph
(+0.06 even with ground-truth coupling), classifier knobs (0), synchrony
threshold (flat), readout window (+0.18 then saturating), training window
(refuted), spike synchrony weight, bimodality weight, five attempts at
phase-to-spike imitation, the spectral readout, learning rate and epoch count --
all of them were competing under a feature ceiling.

Seed 0's foreground IoU is low here (0.4809, predicted foreground 0.327) because
of the membrane threshold saturation recorded in `PV2_0023`; its fg_ari is 0.9595
regardless, so that bug does not affect this conclusion.

## Next

The route to 0.9 is the encoder, and the open question is how good it has to be.
`PV2_0035` degrades the oracle features along a noise sweep to measure the
feature-quality-to-score transfer curve, which is what decides whether
pretrained features (DINO or similar) suffice or whether the encoder has to be
trained for object separation directly.
