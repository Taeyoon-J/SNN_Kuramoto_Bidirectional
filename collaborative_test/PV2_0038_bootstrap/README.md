# PV2_0038 — self-bootstrapping is circular, and that closes the label-free routes

**failed, informatively.** Parent `PV2_0037`. Seed 1, validation 300, window 1024,
kernel sigma 1.0.

## Why it looked promising

`PV2_0037` closed three routes and left one measured asymmetry:

| source of object signal | fg_ari |
| --- | --- |
| the features' own clustering | 0.4657 |
| **the model's grouping** | **0.7239** |

The model's output is a far better object estimate than its features, so
collapsing features within the model's *predicted* groups should align the
collapse with a 0.72-quality estimate instead of the 0.47-quality one that failed.

It reduces the spread convincingly, and in the right places -- 32.9 patches
collapsed per image against roughly 32 foreground patches, so foreground groups
are merged and background is left alone:

| | within | between |
| --- | --- | --- |
| real | 0.4579 | 1.7881 |
| k-means k=8 (failed, scored 0.6965) | 0.3361 | 1.6925 |
| **the model's groups** | **0.2398** | 1.7032 |

## Result: +0.006

| | fg_ari | foreground_iou | matched_object_iou |
| --- | --- | --- | --- |
| real features | 0.7239 | 0.7251 | 0.4951 |
| bootstrap, sync 0.02 | 0.7278 | 0.7277 | 0.4953 |
| bootstrap, sync 0.25 | **0.7299** | 0.7191 | **0.5020** |
| oracle-assigned collapse at the same spread | **0.8478** | | |

Halving the measured within-object spread buys **+0.006**, where the
oracle-assigned collapse at the same spread buys +0.124.

## Why: it is circular

The model's groups are the same information the readout already consumes.
Collapsing the features within them re-injects the model's own estimate for the
model to re-derive, and adds **no new information about objects**. The oracle
collapse worked because it injected true object identity -- information the model
does not have.

This is the sharpest statement of the limit on this branch: **the +0.17 requires
object information the model lacks and cannot manufacture from its own output.**

## The label-free routes, all closed

| route | outcome |
| --- | --- |
| DINOv2 features | worse; already measured here at 0.0897-0.1072 end to end |
| training the existing encoder | a no-op -- not wired into this data path |
| label-free feature clustering | 0.7239 -> 0.6965 |
| self-bootstrap from the model's groups | +0.006, circular |

## What is left, and it is the only thing left

New object information can only enter from **the images**. That means the
image-to-gamma path inside the training loop, so the binding objective reaches the
features instead of stopping at the oscillators. Today the features are frozen,
precomputed, and were never asked to separate objects; the PLV losses shape theta,
the graph and the spiking layers, and nothing upstream of them.

Concretely required:

- a dataloader yielding paired RGB and gamma, which this branch does not have --
  the peer hit the same wall ("our gamma-only training batch has no aligned RGB
  images") and established in their SW_0015 that the gamma and HDF5 indices do
  align, so the pairing is feasible
- `GammaGenerator` constructed and optimised in the loop, with its own small
  learning rate: `--encoder-lr` already exists for this and the code already
  warns that at the core's rate "the phase readout fell 23% on both seeds tried"

And the targets are measured, not guessed:

| feature quality | fg_ari |
| --- | --- |
| within 0.458, between 1.788 (today) | 0.7239 |
| within 0.233, between 1.788 | 0.8478 |
| within 0.000, between 1.788 | 0.8935 |
| within 0.000, between 3.947 | 0.9754 |

So reaching 0.9 needs within-object spread at or below about 0.1 with the present
between-object margin; reaching 0.97 needs the margin roughly doubled as well.
**Nothing in the oscillators, the coupling, the spiking layers, the objective or
the readout needs to change** -- `PV2_0034` reached 0.9633 with no spatial kernel
at all once the features separated objects.
