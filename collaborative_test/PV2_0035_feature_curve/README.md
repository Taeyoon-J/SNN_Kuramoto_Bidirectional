# PV2_0035-0036 — what the encoder has to fix, and how much each part is worth

**completed. ORACLE-guided diagnostics, reported separately, never scores.**
Parent `PV2_0034`. Seed 1, validation 300, window 1024, kernel sigma 1.0.

`PV2_0034` showed the architecture reaches fg_ari 0.9754 on perfect features
against 0.7059 on real ones, so the encoder costs 0.27 and 0.9 is reachable inside
SNN + Kuramoto bidirectional. These two experiments say what about the features
has to change.

## The transfer curve is a cliff, and the ratio is the wrong measure

Degrading the oracle features with Gaussian noise:

| features | within | between | ratio | fg_ari |
| --- | --- | --- | --- | --- |
| oracle | 0.000 | 3.947 | 0.000 | **0.9837** |
| noise 0.25 | 0.602 | 3.966 | 0.152 | **0.8813** |
| **real encoder** | 0.465 | 1.780 | **0.261** | **0.7239** |
| noise 0.5 | 1.203 | 4.045 | 0.297 | **0.1983** |
| noise 1.0 | 2.441 | 4.314 | 0.566 | -0.0058 |

The real encoder beats synthetic noise of a similar ratio by **3.6x**, 0.7239
against 0.1983. Its errors are structured in a way iid noise is not -- real
features vary smoothly across neighbouring patches, while white noise breaks the
coupling graph's top-k selection -- so the within/between ratio cannot say what to
fix. An earlier reading that placed the real encoder "between noise 0.25 and 0.5"
by ratio was wrong about what that implies.

## Splitting the two deficits

Holding between-object distance at the real encoder's **1.7803** exactly and
sweeping within-object spread to zero, by pulling each patch toward its own
object's centroid in the real feature space:

| within-object spread | fg_ari | foreground_iou | matched_object_iou |
| --- | --- | --- | --- |
| 0.465 (real) | 0.7239 | 0.7251 | 0.4951 |
| 0.349 | 0.8063 | 0.8361 | 0.5730 |
| 0.233 | 0.8478 | 0.9080 | 0.6243 |
| 0.116 | 0.8778 | 0.9300 | 0.6613 |
| **0.000** | **0.8935** | **0.9340** | 0.6721 |
| oracle (0.000 within, 3.947 between) | 0.9837 | 0.9098 | 0.7715 |

| fixing | worth | share of the gap |
| --- | --- | --- |
| **within-object consistency** | **+0.170** | 65% |
| **between-object margin** | +0.090 | 35% |

Within-object consistency alone reaches **0.8935**, 0.0065 short of the 0.9
target, and its foreground IoU, 0.9340, is higher than the full oracle's 0.9098.
Both together reach 0.9837.

## The constraint this puts on the design

Both sweeps use ground-truth masks to define objects, so they are oracle-guided.
**Training the encoder on those labels would make the model supervised** and void
the comparison against a fully unsupervised Slot Attention. The design therefore
has to obtain within-object consistency *without labels*.

That property -- patch embeddings that agree within an object, without
supervision -- is what self-supervised ViT features are known for. It is also the
cheapest route by far, since nothing in the oscillators, coupling, spiking layers,
objective or readout needs to change: `PV2_0034` reached 0.9633 with no spatial
kernel at all once the features separated objects.

Ranked by measured value per unit of work:

1. **Replace the encoder with self-supervised patch features** (DINO-style).
   Targets the +0.170 directly and needs no labels.
2. **Add a label-free consistency objective to the existing encoder** -- appearance
   and smoothness agreement between neighbouring patches. Approximates the same
   property on CLEVR, where objects are uniform in colour. Cheaper, weaker.
3. **Widen the between-object margin** for the remaining +0.090, once 1 or 2 lands.

What is explicitly *not* worth more work: every axis exhausted earlier on this
branch, all of which was competing under this feature ceiling.
