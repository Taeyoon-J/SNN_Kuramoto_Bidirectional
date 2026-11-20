# PV2_0064 — the margin can be enlarged without labels, and it still does not help

**completed. Twelfth route closed, and it refutes the diagnostic framework that
steered the previous eleven.** Parent `PV2_0062`. Inference only: no training, no
weights changed. Validation 300, contract v1, three graph-frozen checkpoints.

## The gap `PV2_0062` identified

Oracle codes have within-object spread 0.0000 and between-object distance 3.9462,
2.4x the baseline's 1.6592. Every label-free method across eleven routes reduced
within while **shrinking** between (smoothing: 1.6592 -> 1.5180). So the conclusion
was that the missing ingredient is an **enlarged between-object margin**, and that
enlarging it without labels is close to knowing which object is which.

## First, the measurement was in the wrong space

All of that was Euclidean. `graph_generator.py:123` compares
`F.normalize(self.projection(gamma))` by **cosine**, which is invariant to the uniform
rescaling Euclidean distance is not. Re-measured where the graph works:

| variant | cos_within | cos_between | ratio |
| --- | --- | --- | --- |
| baseline gamma | 0.0142 | 0.2483 | 0.0571 |
| smoothed (`PV2_0062`) | 0.0070 | **0.2113** | 0.0332 |
| whitened eps=0.01 | 0.0685 | **0.7693** | 0.0890 |
| whitened eps=0.1 | 0.0298 | 0.6900 | 0.0432 |
| smoothed + whitened eps=0.01 | 0.0188 | 0.7434 | 0.0253 |
| **smoothed + whitened eps=0.1** | **0.0127** | **0.6688** | **0.0189** |
| ORACLE codes (fg_ari 0.9754) | 0.0000 | **1.0000** | 0.0000 |

**ZCA whitening is the first transform in twelve routes to enlarge the between-object
margin without labels**: 0.2483 -> up to 0.7693 against the oracle's 1.0000, and
combined with smoothing it keeps within *below* the baseline (0.0127 vs 0.0142).
It works because it is anisotropic — it rescales each eigendirection of the channel
covariance, so unlike a uniform rescaling it changes cosine similarity. Covariance
comes from train rows only; no mask enters the transform.

## Then the score

| readout | variant | fg_ari | fg_iou | obj_iou | pred_fg |
| --- | --- | --- | --- | --- | --- |
| **cc, baseline** | — | **0.7239** | **0.7251** | 0.4951 | **0.129** |
| cc | smoothed+whitened 0.1 | 0.7670 | 0.1862 | 0.4718 | 0.5904 |
| cc | smoothed+whitened 0.01 | 0.7489 | 0.1843 | 0.4957 | 0.5973 |
| cc | whitened 0.1 | 0.6508 | 0.1886 | 0.4431 | 0.5426 |
| cc | whitened 0.01 | 0.4745 | 0.1739 | 0.3931 | 0.5134 |
| k14 | smoothed+whitened 0.1 | 0.6784 | 0.1617 | 0.4017 | 0.6182 |
| k14 | whitened 0.01 | 0.4728 | 0.1501 | 0.2105 | 0.5601 |

**No variant stayed in the calibrated regime.** Whitening itself pushes predicted
foreground from 0.129 to 0.51-0.62, and foreground IoU collapses to ~0.18 in every
one. The highest fg_ari, 0.7670, comes at predicted foreground 0.5904 — the extent
dial of `PV2_0059`, not better features, since plain gamma already reaches 0.7718 at
pred_fg 0.8084.

The criterion was fixed before the run — a gain counts only below predicted foreground
0.20 — and it is what makes this readable. Reported on fg_ari alone this would have
looked like "+0.0431".

## What this settles about the 0.9 goal

**Three transforms have now improved a separation measure without improving fg_ari in
the calibrated regime**: chromaticity (`PV2_0058`, both Euclidean ratio and cosine
margin better, score worse), smoothing (`PV2_0062`, Euclidean ratio 41% better, fg_ari
-0.005), and whitening (here, cosine margin 2.7x, nothing in the calibrated regime).

So **separation statistics are not sufficient statistics for fg_ari in this pipeline**,
and the framework I used to choose eleven of the twelve routes cannot identify the
missing ingredient. The oracle's 0.9754 does not come from the geometry of its feature
cloud — it comes from the codes **being the answer**. Enlarging a margin produces the
geometry without the information.

That is a real constraint on where a route to 0.9 can come from. It has to supply
object-level information, not feature-cloud geometry, and label-free sources of that
are: reconstruction with slot competition (tested, 0.6779, below frozen features);
motion or multiple views of one scene (**not available** — CLEVR frames here are
single static renders); and cross-view consistency under augmentation, which the peer
has queued as SW0139 with no result at the last two fetches.

**No route to 0.9 has been demonstrated, and this experiment shows the method I was
using to look for one was measuring the wrong thing.**
