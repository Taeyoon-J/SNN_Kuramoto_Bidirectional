# Route map to 0.9, v4 — what the pixel protocol settles, and what route C cannot do

Supersedes `ROUTE_MAP_0.9_v3.md`. Written from `PV2_0125`'s measurements; no new runs.
Everything here is arithmetic over numbers already in the records, so it needed no server.

## The new axis: fraction of the grid's own ceiling

`PV2_0125` measured, per dataset and grid, what a **perfect** patch segmenter scores under
the pixel protocol. That turns every result into two separable quantities — what the grid
costs, and what the model costs — and the second one is comparable across datasets for the
first time.

| dataset | grid | ours | ceiling | **fraction of ceiling** |
| --- | ---: | ---: | ---: | ---: |
| Tetrominoes | 32 | 0.9280 | 0.9409 | **98.63%** |
| CLEVR | 16 | 0.6125 | 0.6973 | 87.84% |
| CLEVR | 32 | 0.7142 | 0.8355 | **85.48%** |
| Multi-dSprites | 32 | 0.2080 | 0.5619 | 37.02% |

CLEVR rows from `BENCHMARK_PLAN.md` (100 validation images, seed count unstated — being
redone at 3 seeds on the 1000 test images). Tetrominoes and Multi-dSprites from `PV2_0125`
(300 validation images, seed 0).

## Result 1 — route C cannot reach 0.9 on CLEVR, by arithmetic

Route C is "raise the grid." Its ceiling at grid 16 is 0.6973 and at grid 32 is 0.8355, and
grid 16 → 32 did buy **+0.1017** of real pixel score. So the direction is real. The size is
not:

> At the fraction measured at grid 32, **85.48%**, reaching 0.90 would require a ceiling of
> **1.0529**. A ceiling cannot exceed 1.0.
>
> Even a **lossless** grid — the cell equal to the pixel, ceiling exactly 1.0, which
> Multi-dSprites at grid 64 confirms is attainable — yields only **0.8548**.

And the fraction is not conserved as the grid refines. It **fell**:

| | fraction |
| --- | ---: |
| CLEVR grid 16 | 87.84% |
| CLEVR grid 32 | 85.48% |
| change | **−2.36 points** |

So the honest extrapolation to grid 64 is a fraction at or below 85%, not above it. Raising
the grid alone does not reach 0.9 on CLEVR. **It was never going to**, and the arithmetic
that shows this was available as soon as the ceiling was measured.

## Result 2 — route C's stated justification used the inference `PV2_0125` refuted

`PV2_0120` argued for grid 64 this way:

> "Patch purity rises with the grid exactly as route C's argument needs — 16: 0.9572,
> 32: 0.9779, 64: 0.9890"

`PV2_0125` measured what purity is worth as a predictor of the pixel ceiling:

| | grid-32 purity | measured pixel ceiling |
| --- | ---: | ---: |
| Tetrominoes | 0.9941 | 0.9409 |
| Multi-dSprites | 0.9878 | **0.5619** |

Two datasets within 0.006 of each other in purity, and their ceilings differ by 0.379.
Purity is a per-cell label-agreement rate; FG-ARI counts **pairs**, so a boundary error on a
small object costs far more than its cell count, and two objects separated by one cell merge
under the quantisation whatever the model does. **Purity does not bound the ceiling.**

Consequence: **the CLEVR pixel ceiling at grid 64 is unmeasured.** The 0.9890 purity in
`PV2_0120` is not a proxy for it. Any grid-64 CLEVR training is unjustified until that
number exists — and it costs no GPU: `ceil.py` on `pixel_truth_test1000.pt`, which the
interrupted `clevr_pix.sh` builds. **This is the cheapest decisive measurement left.**

## Result 3 — the core is not the limit; CLEVR is

The same architecture, same losses, same readout, holds **98.63%** of its grid's ceiling on
Tetrominoes. A 98.63% fraction, applied to a lossless grid, is 0.9863.

So the deficit on CLEVR is not an intrinsic ceiling of the SNN + Kuramoto bidirectional
core. It is specific to CLEVR, and the records already say what is different there:
`PV2_0075` measured the dissimilarity the geodesic actually sees —

| pair type | mean `1 − cosine` |
| --- | ---: |
| within one object | 0.0331 |
| background to background | 0.0054 |
| **object to background** | **0.1743** |

The object/background wall is 5.27x a within-object shading step, which is why figure/ground
works. Nothing here measures an **object-to-object** wall, and CLEVR's 10 shaded, occluding,
same-material objects are exactly where that matters, while Tetrominoes' three separated
shapes on black are exactly where it does not.

## Where that leaves the goal's third part

**The question was: can the core, modified, reach 0.9+?** The answer now has a shape it did
not have before:

1. **Not by the grid.** Route C's maximum at the measured fraction is 0.8548, and the
   fraction is falling, not rising. This is the first route closed by arithmetic rather than
   by a run — and it closes for CLEVR only.
2. **The architecture already exceeds 0.9 where the grid permits it.** Tetrominoes, pixel
   FG-ARI 0.9280. One seed and validation, so not yet a 3-seed claim, but it is positive
   evidence the core can reach that range at all — which nine closed routes on CLEVR had
   made doubtful.
3. **For CLEVR, the target must be the fraction, not the ceiling.** 0.90 needs ≥ 90% of
   ceiling; we hold 85.48% and hold 98.63% elsewhere. The gap to attack is 4.5 points of
   fraction, and the one measured thing that moves it is `PV2_0106`'s oracle — merging plus
   coverage together reaching 0.9329, superadditive, both of them readout stages over spikes
   the model already produces. That is an **oracle on the patch contract**: it states a
   ceiling for the readout, not a method. `PV2_0107b` and the peer's `SW_0234` both showed
   that merging without the oracle loses obj-IoU.
4. **The object-to-object boundary is the unmeasured quantity** that the CLEVR/Tetrominoes
   split points at, and `PV2_0075`'s table never covered it.

## Ordered next measurements, cheapest first

| # | measurement | cost | what it decides |
| --- | --- | --- | --- |
| 1 | CLEVR pixel ceiling at grid 64 | CPU only, `ceil.py` | bounds route C definitively; closes or reopens it |
| 2 | object-to-object dissimilarity, the `PV2_0075` table's missing row | CPU on saved features | whether the graph can separate adjacent objects at all |
| 3 | Multi-dSprites at grid 64 (running) | one evaluation | how much of a 37.02% fraction was the grid |
| 4 | CLEVR 3 seeds, test 1000, pixel (running) | three evaluations | replaces the 100-image CLEVR row |
| 5 | Tetrominoes seeds 1–2 (running) | ~20 GPU-hours | turns 0.9280 into a 3-seed mean |

## What this does not claim

The fraction-of-ceiling framing assumes the fraction is a meaningful property of the model
rather than an artefact of each dataset's object statistics, and four points across three
datasets is thin support for that. The CLEVR rows are 100 validation images with an
unstated seed count. The Tetrominoes row is one seed on validation. The arithmetic in
Result 1 is sound given its inputs, but its input — 85.48% — carries all of those
limitations, and a CLEVR fraction measured at 3 seeds on test could move it.
