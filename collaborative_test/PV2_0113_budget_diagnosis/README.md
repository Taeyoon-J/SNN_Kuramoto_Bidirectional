# PV2_0113 — where the coverage gap actually is, and a hypothesis that was refuted on the way

Status: **completed (diagnostic)**. Oracle throughout — reads the truth, never a score.
Parent: `PV2_0112`. Seed 0 / `GC_w0`, 60 validation images, contract v2, grid 32,
coverage measured from the production readout's saved predictions.

## The hypothesis this started with, and why it is wrong

`PV2_0112` showed the missing foreground is concentrated rather than spread, and
`PV2_0106`'s per-object coverage (0.8726) sitting below the per-patch figure (0.9456)
pointed at small objects. The loss arithmetic seemed to confirm it.

**That arithmetic is correct and is worth keeping.** `plv_group_balance_loss` targets the
mean off-diagonal PLV at 0.867. A mean over *pairs* weights a coherent group of n nodes by
**n²**, so with 1024 nodes and ~900 of them background:

| group | n | share of the 0.867 target |
| --- | ---: | ---: |
| background | 900 | **0.7726** |
| object | 40 | 0.001526 |
| object | 9 | 0.000077 |
| object | 3 | **0.000009** |
| six objects together | — | 0.002476 |

The background alone supplies 77% of the target; a 3-patch object is worth **1/90,019** of
it. Every heavily weighted PLV term is pair-quadratic the same way — bimodality (6.0),
balance (10.0), collapse (1.0) and the participation ratio all weight a group by n² — and
the one local, size-insensitive term, `plv_spatial_coherence_loss`, carries the **smallest**
weight, 0.5. So the training signal genuinely cannot see a small object.

**But the predicted consequence is mostly absent.** Coverage against object size:

| size | objects | mean coverage | fully lost | mean activity |
| --- | ---: | ---: | ---: | ---: |
| 1–2 patches | 3 | 83.3% | 0.0% | 0.2020 |
| 3–5 | 65 | 88.2% | 1.5% | 0.1910 |
| 6–10 | 102 | 90.7% | 0.0% | 0.1895 |
| 11–20 | 66 | 91.7% | 1.5% | 0.1970 |
| 21–40 | 95 | **96.5%** | 0.0% | 0.1966 |
| 41+ | 42 | 95.6% | 0.0% | 0.1996 |

There is a size gradient and it is **mild** — 88% against 96% — not the collapse the loss
arithmetic would predict. **0.5% of objects are claimed by nothing.** Loss blindness to small
objects is real and is not what is costing the coverage.

This is the seventh prediction in this project to be refuted by its own check, and the second
where correct arithmetic led to a wrong consequence.

## What the gap actually is

Per-object mean coverage **0.9243**, per-patch **0.9456**. The deficit is not whole objects
missing; it is **5–8% of nearly every object**, which is its boundary.

Set that beside the budget: **predicted foreground 0.1369 against a target of 0.121**. The
model spends **13% more foreground than exists** and still misses the edge of almost every
object. The foreground is not scarce, it is **in the wrong places** — which is exactly what
`PV2_0106`'s closing note said and what this now measures directly.

## A number that must not be conflated

`PV2_0106` reports coverage 0.8726 and lost 0.0214; this reports 0.9243 and 0.005. They are
not the same measurement. This one is seed 0 alone — the seed with the highest `pred_fg`,
0.1512 — on 60 images, and counts an object patch as covered if **any** group claims it;
`PV2_0106` is three seeds on 300 images and may count only the matched group's share. The two
figures are **not interchangeable** and neither supersedes the other until the definitions are
reconciled.

## Where this leaves the route to 0.9

Three things are now closed or reframed:

1. Admitting unclaimed patches by threshold — closed by `PV2_0112` (0.73% base rate).
2. Widening existing groups — closed by the peer's `SW_0207`.
3. "Make the model fire on small objects" — **refuted here**: it already does, at 88–96%.

What is left is narrower and better posed: the model has the foreground budget it needs and
allocates it wrongly, over-spending by 13% while under-covering every object's boundary. The
next hypothesis has to be about **boundary patches** — where patch-level ground truth is
itself ambiguous, since a boundary patch is part object and part background and reaches the
target by majority vote — not about object count or object size.

## Next

Before proposing a loss, measure whether the missed patches are boundary patches: for each
uncovered object patch, its distance to the object's edge, and the fraction of its pixels
that are actually the object. If the misses are overwhelmingly 50–60%-covered boundary
patches, part of the remaining gap is **patch-grid aliasing and not a model failure at all**,
and the honest ceiling at grid 32 is below `PV2_0106`'s oracle.
