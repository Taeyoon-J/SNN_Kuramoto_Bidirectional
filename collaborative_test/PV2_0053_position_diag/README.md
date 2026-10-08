# PV2_0053 — 38.8% of images contain a pair the graph cannot separate

**completed. Diagnostic that opens a route none of the closed seven touched.**
Parent `PV2_0034`. Free: no training, no GPU, reuses the gamma behind our best
0.7250.

## The hole

`graph_generator.py:123` builds every edge from
`z = F.normalize(self.projection(gamma), dim=-1)` followed by a cosine similarity
and top-k. And gamma is **[B, 8, 256]** — eight appearance channels per patch from
one 3x3 convolution. **Position never enters that code.** It arrives only
afterwards, as a monotone distance prior on the logits (`spatial_decay`) and as the
readout's Gaussian kernel.

Two earlier scripts had already written down the consequence without following it
up. `colour_split2.py`: the features "are essentially colour". `colour_probe.py`:
"its edges are chosen by feature cosine, so two objects of the same colour look
alike to it and may be wired together; with coupling this strong that would pull
them into one group".

## What was measured

Normalised patch coordinates were appended to the existing gamma, scaled by a
weight, and the separation statistics recomputed. Nothing retrained, so no score.

The sharp statistic is the **minimum** between-object distance per image, not the
mean: a pair that merges is a pair whose centroids coincide, and a healthy mean
hides it. "Unseparable" counts images where some pair sits closer together than the
objects are internally spread.

| code | within | between | min pair | **unseparable images** |
| --- | --- | --- | --- | --- |
| gamma as used (8ch) | 0.4590 | 1.7812 | 0.7024 | **38.8%** |
| + position w=0.25 | 0.4716 | 1.7910 | 0.7304 | 37.5% |
| + position w=0.5 | 0.4989 | 1.8185 | 0.7920 | **28.4%** |
| + position w=1 | 0.5755 | 1.9157 | 0.9466 | **18.1%** |
| + position w=2 | 0.7687 | 2.2278 | 1.2526 | 7.7% |
| + position w=4 | 1.2166 | 3.1021 | 1.7802 | **4.3%** |
| ORACLE codes (fg_ari 0.9754) | 0.0000 | 3.9462 | 2.4765 | **0%** |

**38.8% of validation images contain an object pair the graph's cosine cannot keep
apart** — a loss this project had never measured. Position removes most of it.

The cost is that position varies within an object, so within-object spread rises
too. The scale-invariant trade, min-pair over within, peaks near w=1: 1.53 at 8ch,
1.59 at w=0.5, **1.645 at w=1**, 1.63 at w=2, 1.46 at w=4.

## Why this is not one of the seven closed routes

DINOv2, label-free clustering, self-bootstrapping, encoder capacity, slot-term
defaults, end-to-end training and the frozen core all tried to make the
**appearance** code better. This adds a different kind of information that was
absent from it. Slot Attention — the baseline we beat 0.7250 to 0.6635 — has always
added position to its CNN features through `SoftPositionEmbed` before any
grouping; we never did.

SNN and Kuramoto bidirectional are untouched. Only the input encoding widens from 8
to 10 channels.

## Limitation

Objects come from the ground-truth masks, so every number here is an oracle-guided
measurement of feature geometry — a diagnostic, never a score. Whether it converts
into fg_ari is `PV2_0054`.
