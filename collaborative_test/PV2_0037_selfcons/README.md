# PV2_0037 — three routes to better features, all closed, and what that leaves

**completed. Two negative results and one no-op.** Parent `PV2_0036`. Seed 1,
validation 300, window 1024, kernel sigma 1.0.

`PV2_0034` showed the architecture reaches fg_ari 0.9754 on perfect features
against 0.7059 on real ones, and `PV2_0035/0036` that within-object consistency is
worth +0.170 of that. This tried the three cheapest ways to get it.

## 1. DINOv2 features: worse, and already measured

| features | within | between | ratio |
| --- | --- | --- | --- |
| CNN encoder (current) | 0.465 | 1.780 | **0.261** |
| DINOv2 grid16 dim8 | 0.970 | 2.579 | 0.376 |
| DINOv2 grid16 dim384 | 8.035 | 13.969 | 0.575 |

And prior runs on this machine had already scored them end to end:

| clustered directly | FG-ARI |
| --- | --- |
| **CNN 8-channel** | **0.4657** |
| DINOv2 384-dim | 0.3482 |
| DINOv2 PCA 8-dim | 0.0464 |

through the full pipeline, DINO gives FG-ARI 0.0897-0.1072.

The reasoning that self-supervised ViT features would supply within-object
consistency was a generalisation, and it is wrong here: CLEVR objects are flat
single colours, which a colour-sensitive CNN represents with little within-object
variation, while DINO also encodes shading and texture that vary *across* an
object. The prior measurements existed on this machine and should have been read
before proposing it.

## 2. Training the encoder: not wired in this data path

`--encoder-lr` at 1e-6, 1e-5 and 1e-4 produced checkpoints **byte-identical** to
the baseline (md5 `732fe3112419` for all four). Training consumes precomputed
gamma via `--gamma-seq-path`, so `gamma_generator` is None and there is no encoder
in the loop. This is a no-op, not a negative result, and an earlier description of
it as "the last lever running" was wrong -- the lever was never connected.

Improving the features therefore needs either gamma regenerated offline from a
better encoder, or the image-to-gamma path inside the training loop, which needs
paired RGB batches. The peer hit the same constraint ("our gamma-only training
batch has no aligned RGB images").

## 3. Label-free feature clustering: lowers the spread, lowers the score

Clustering each image's features with no labels and replacing each patch by its
cluster centroid:

| | within | fg_ari | foreground_iou | matched_object_iou |
| --- | --- | --- | --- | --- |
| real features | 0.458 | **0.7239** | 0.7251 | 0.4951 |
| k=8 | 0.336 | 0.6965 | **0.7441** | 0.4986 |
| k=12 | 0.390 | 0.7034 | 0.7190 | **0.5015** |
| k=16 | 0.413 | 0.7052 | 0.7139 | 0.4973 |
| k=24 | 0.434 | 0.7132 | 0.7111 | 0.4951 |

Every arm is below the real features on fg_ari, monotonically: the less collapse,
the closer to the original.

**This refines the PV2_0035 conclusion in an important way.** Measured
within-object spread improved from 0.458 to 0.336 while the score fell. So
"+0.170 from within-object consistency" is available only when the consistency is
with respect to *true* objects. Reducing the spread by any label-free clustering
does not transfer, because discovered clusters that straddle object boundaries
blur them. The +0.170 is a target, not a method, and treating it as a prescription
was too quick.

## What that leaves

One measured fact points the way:

| source of object signal | fg_ari |
| --- | --- |
| the features' own clustering | 0.4657 |
| **the model's grouping** | **0.7239** |

The model's own output is a far better object estimate than its features are. So
collapsing features within the *model's* predicted groups -- rather than within
arbitrary feature clusters -- aligns the collapse with a 0.72-quality object
estimate instead of a 0.47-quality one. That needs no labels and no training; it
is iterative refinement at inference, and recurrent refinement of binding fits
the research question rather than straining it.

That is `PV2_0038`. If it fails too, the remaining route is the image-to-gamma
path in the training loop with paired RGB batches, which is the most fundamental
and the most expensive.
