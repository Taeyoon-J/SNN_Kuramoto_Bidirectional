# PV2_0066 / PV2_0067 — the graph carries 94% of the oracle's gain, and `PV2_0005` was wrong

**completed. This corrects a measurement that was load-bearing for most of this
session's reasoning.** Parent `PV2_0034`. Inference only; the core is unchanged apart
from a diagnostic hook that is inert unless set. Validation 300, contract v1, three
graph-frozen checkpoints.

## The contradiction that prompted it

Two numbers on record could not both be right about where features matter:

- `PV2_0005`: an **oracle coupling graph** is worth only **+0.06**.
- `PV2_0034`: **oracle features** are worth **+0.25** (0.7059 -> 0.9754).

Features reach the dynamics by exactly two paths — `self.graph_generator(gamma_seq)`
builds the coupling graph, and `_to_phase(_static_drive(gamma_seq))` initialises theta.
If the graph route were capped near +0.06, nearly all of the +0.25 would have to arrive
through theta initialisation, a 32-parameter `Linear(8 -> 4)` plus one shared gain.
That had never been decomposed. `--phase-gamma-path` now feeds the phase path a
different tensor from the graph, so the two routes separate.

## Result: it is the graph, overwhelmingly

| graph | phase | fg_ari | fg_iou | obj_iou | pred_fg |
| --- | --- | --- | --- | --- | --- |
| real | real | 0.7250 | 0.7188 | 0.4953 | 0.1299 |
| **oracle** | **real** | **0.9671** | **0.9353** | **0.7584** | **0.1121** |
| real | oracle | 0.8172 | 0.7410 | 0.6051 | 0.1292 |
| oracle | oracle | 0.9826 | 0.9190 | 0.7706 | 0.1186 |

Of the total +0.2577, the graph route alone carries **+0.2421 (94%)** and theta
initialisation **+0.0922 (36%)**.

**And the graph route does it in the calibrated regime** — predicted foreground 0.1121
against a target of 0.118, with foreground IoU **0.9353**. Every previous fg_ari gain
in this project came with foreground over-prediction; this one does not.

## Why `PV2_0005` disagreed

It forced the **final coupling matrix** `core.forced_graph`, bypassing everything the
generator does with its input: projection, cosine, top-k, geodesic relaxation, spatial
decay. Here the generator receives oracle **features** and its own machinery amplifies
them. The gap between +0.06 and +0.24 is that machinery.

**So "the graph is spent", which I repeated throughout this session, rested on a
measurement of something else.** The graph is where nearly all the headroom is.

## `PV2_0067`: the transforms, confined to the graph

`PV2_0064`'s whitening had failed with predicted foreground at 0.51-0.62, and the
natural reading was that it had wrecked the phase path. The hook lets the transform go
to the graph alone, with the phase path on real gamma:

| graph gamma | cos_between | fg_ari | fg_iou | obj_iou | pred_fg |
| --- | --- | --- | --- | --- | --- |
| real (baseline) | 0.2483 | 0.7250 | **0.7188** | 0.4953 | **0.1299** |
| whitened eps=0.1 | 0.6900 | 0.7602 | 0.2256 | 0.4535 | 0.4861 |
| smoothed+whitened eps=0.01 | 0.7434 | 0.7454 | 0.2692 | 0.4486 | 0.4464 |
| whitened eps=0.01 | 0.7693 | 0.6579 | 0.2448 | 0.4395 | 0.4453 |
| **smoothed** | 0.2113 | 0.7196 | **0.7618** | 0.4741 | **0.1178** |
| SA whitened | 0.2110 | 0.6122 | 0.2304 | 0.2857 | 0.5059 |
| SA features | 0.0316 | 0.3247 | 0.2084 | 0.1605 | 0.4681 |
| **oracle (diagnostic)** | 1.0000 | **0.9671** | **0.9353** | **0.7584** | **0.1121** |

**Only smoothing stays calibrated, and its fg_ari is -0.0054.** Every whitened variant
blows the calibration **even confined to the graph** (0.45-0.49), so the phase path was
not the problem — enlarging the graph's cosine margin by whitening makes the model call
more patches foreground by itself. Fourteenth route closed.

The script's own verdict printed "THE LEVER WORKS" because it counted the oracle row
among the calibrated transforms. The oracle is the diagnostic bound, not a usable
transform, and the verdict is corrected here.

## What is now established, and what is not

**Established:** the architecture reaches **0.9671 / 0.9353 / 0.7584 at predicted
foreground 0.1121** when the graph's input separates objects. The 0.9 target is
reachable by this SNN + Kuramoto pipeline, unmodified, and the single thing standing in
the way is a gamma whose graph-side between-object cosine distance approaches 1.0
without inflating predicted foreground.

**Not established:** any label-free way to produce that gamma. Fourteen routes have
failed, and the two levers available to a fixed transform pull against each other — the
margin can be enlarged (whitening, cos_between 0.2483 -> 0.7693) but only by widening
predicted foreground, and the foreground can be kept calibrated (smoothing, pred_fg
0.1178) but only without a margin gain (cos_between 0.2113).

The target is now exact, which it was not before: **cos_between near 1.0 at predicted
foreground near 0.12.** Oracle codes are one-hot per object, so they are orthogonal
between objects by construction; no fixed transform of appearance features reproduces
that, and a trained one would need object identity.
