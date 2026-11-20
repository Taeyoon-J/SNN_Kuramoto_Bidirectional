# PV2_0056 — our partition is not the geometric template, so that route is closed

**completed. Negative result, and it closes the objective-side route I had proposed
one experiment earlier.** Parent `PV2_0055`. Diagnostic: no training, no weights
changed, and no ground truth in the comparison.

## Why it was asked

`PV2_0055` found the PLV objective is partly satisfiable by geometry — feeding
position into the graph's code reached a **lower** loss (13.91 and 13.75 against
17.86) with a 0.17-0.27 **worse** score. I proposed a loss term penalising agreement
with the fixed geometric template. That only works if the current model agrees with
that template to begin with, which had never been measured: `PV2_0029` measured what
geometry alone can *score* (0.3902), not how much of our own partition it accounts
for.

## What was measured

Both readouts already exist (`--readout spikes` / `spatial_only`) and had only ever
been scored against ground truth, never against each other. Both were run through
**spectral k=10 with one background cluster**, the setting behind `PV2_0029`'s
0.3902 / 0.6068 pair, and the two partitions compared by a symmetric adjusted Rand
index over all 256 patches.

The geometry-only partition is **identical on every image** — a pure spatial kernel
does not look at the image — with 10.00 groups per image. It reproduced 0.3902
against ground truth exactly, which binds this run to the recorded one.

| spike partition | ARI vs the geometric template | per-image std |
| --- | --- | --- |
| BIM6_s1 | 0.1286 | 0.1695 |
| FZG_s0 | 0.1129 | 0.1617 |
| FZG_s2 | 0.1078 | 0.1586 |
| **mean** | **0.1164** | |

**0.1164.** The spike partition is almost unrelated to the template. The geometric
shortcut is something the model takes only when geometry is handed to the graph's
input; the baseline does not partition geometrically, so a term penalising agreement
would be pushing down a quantity that is already 0.12. **The route I proposed is
closed, for 20 minutes of evaluation and no training.**

It also says something positive: geometry alone scores 0.3902 from a template our
model does not use, so our 0.7126 under that readout is nearly orthogonal to the
positional prior.

## Two failures on the way, both mine

**The first run was invalid and its conclusion was discarded.** I ran the
geometry-only control through the **connected-component** readout, which `PV2_0029`
had already recorded as impossible — "with connected components the spatial kernel
alone makes one giant component, so everything becomes background, and zero
foreground groups survive". It returned an empty partition, 0.00 groups, identical on
every image, and the comparison came back `nan`. The script printed "LOW: the next
move is elsewhere" from that `nan`. The right answer happened to be LOW, but the
first run did not show it.

**`evaluate_patch_masks` was also the wrong instrument.** Its `fg_ari` is restricted
to the *target's* foreground patches, so passing a partition as the target makes the
score depend on that partition's own foreground — and returns `nan` when it has none.
Replaced with a symmetric adjusted Rand over all patches.

**Then the predictions were not written.** `--save-predictions` had been added only
where the connected-component readout builds its labels; the spectral readout returns
from an earlier branch, so four evaluations ran and wrote no file. Fixed, and the
third attempt was smoke-tested for the file before the full run.
