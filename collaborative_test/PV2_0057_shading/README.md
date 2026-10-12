# PV2_0057 — within-object spread is mostly shading, and arithmetic halves it

**completed. The first positive diagnostic in this cycle, and the motivation for
`PV2_0058`.** Parent `PV2_0036`. Free: no training, no GPU.

## Why it was asked

`PV2_0034` put oracle codes through this pipeline **unchanged** and reached fg_ari
**0.9754** against 0.7250 on real features, so the architecture is not the limit —
the encoder costs 0.27. `PV2_0035/0036` split that loss, and the larger term is
**within-object spread**: taking it from 0.465 to 0.000 is worth **+0.170** and
reaches 0.8935, which clears the 0.75 target on its own.

Nine routes to better features are closed — DINOv2, label-free clustering,
self-bootstrapping, encoder capacity, slot-term defaults, end-to-end training, the
frozen core, data scale, position in the graph — and **every one of them tried to
learn the features.** None asked what within-object spread is made of.

In CLEVR an object's albedo is one of eight colours and close to constant over its
surface, while its pixels vary with shading, specular highlights and the floor's
gradient. If the spread is mostly illumination it is removable by arithmetic, because
chromaticity is invariant to a multiplicative intensity factor: scaling (r,g,b) by s
leaves r/(r+g+b) unchanged.

## Result

Patch statistics on validation 300, scored the way `PV2_0035` defines separation.
Scale is removed per image by `s2net_cls.py:413`, so the comparable quantity is the
ratio.

| input | dim | within | between | **w/b** |
| --- | --- | --- | --- | --- |
| **current gamma** (behind 0.7250) | 8 | 0.3369 | 1.6592 | **0.2030** |
| raw RGB patch means | 3 | 0.0465 | 0.2169 | 0.2144 |
| **chromaticity r/(r+g+b)** | 3 | 0.0197 | 0.1805 | **0.1093** |
| unit RGB (L2 direction) | 3 | 0.0323 | 0.2953 | 0.1095 |
| chromaticity + luma | 4 | 0.0324 | 0.2027 | 0.1598 |
| log RGB, per-image centred | 3 | 0.1644 | 0.7885 | 0.2085 |
| **oracle codes** (fg_ari 0.9754) | — | 0.0000 | 3.9462 | **0.0000** |

**Chromaticity cuts the ratio by 46%**, from 0.2030 to 0.1093. Dividing by intensity
is what does it: unit RGB, the L2 version of the same invariance, lands at 0.1095,
while subtracting a log-domain mean — which removes a global factor but not a spatial
gradient — barely moves it at 0.2085.

**The learned encoder is not adding separation over patch colour means.** Raw RGB
patch means sit at 0.2144 against the trained 8-channel gamma's 0.2030, a 5%
difference for an entire encoder. `colour_split2.py` had already written that the
features "are essentially colour"; this says how little else is in them.

## Built as tensors

Two tensors, in the same row order as `targets_v1`, standardised per channel on
**train rows only** so no validation or test image influences the transform, and
scaled to the existing gamma's standard deviation because
`graph_generator.projection` sees gamma directly.

| tensor | dim | within | between | w/b |
| --- | --- | --- | --- | --- |
| `chroma3` | 3 | 0.5951 | 5.4583 | **0.1090** |
| `chroma11` (8 existing + 3) | 11 | 0.7113 | 5.7397 | 0.1239 |

The gain survives the transform, and `chroma11`'s first eight channels are **bitwise
equal** to the existing gamma.

## Limitations

- Objects come from contract v1's patch labels, so every number here is an
  **oracle-guided measurement of feature geometry** — a diagnostic, never a score.
- The ratio is not linear in fg_ari. `PV2_0036` measured +0.170 for taking within to
  **zero**; chromaticity halves the ratio, it does not zero it, so the expected gain
  is a fraction of that. `PV2_0058` measures the actual score.
