# PV2_0114 — the figure/ground error is over-prediction, by 4.7x

Status: **completed (diagnostic)**. Oracle throughout — reads the truth, never a score.
Parent: `PV2_0112`, `PV2_0113`. Seed 0 / `GC_w0`, 60 validation images, contract v2, grid 32.
Measured directly from the production readout's saved predictions. No modelling.

## The measurement

| per image, 1024 patches | |
| --- | ---: |
| true foreground | 118.0 |
| predicted foreground | 143.4 |
| true positives | 111.6 |
| **false positives** | **31.8 — 22.2% of what we predict** |
| false negatives | 6.4 — 5.4% of what exists |

| foreground IoU | | |
| --- | ---: | ---: |
| as it is | 0.7450 | |
| every missed patch recovered | 0.7879 | **+0.0428** |
| every false positive removed | **0.9456** | **+0.2006** |
| both | 1.0000 | |

**Inventing foreground costs 4.7x what missing it does.**

## What this says about everything tried so far

Every stage attempted on either branch has pushed foreground **up**:

| | direction | outcome |
| --- | --- | --- |
| `PV2_0112` (ours) | add unclaimed patches | fg_iou −0.0755 to −0.3267, gate failed |
| `SW_0207` (peer) | widen existing groups | fg_iou +0.171, FG-ARI −0.054, closed |

Both were aimed at the 0.0428 half of the problem while the 0.2006 half went untouched.
`PV2_0106` framed the second oracle as *coverage*, and that framing — which this project and
the peer branch both adopted — pointed at the smaller error. The arithmetic above is the
correction.

## The asymmetry that makes the other direction worth trying

`PV2_0112` failed on a base rate: among unclaimed patches only **0.73%** were foreground, so
no threshold could admit patches and be right. The reverse problem has the opposite base
rate — **22.2%** of predicted foreground is wrong, **30x more favourable**. What was
impossible in one direction is an ordinary classification problem in the other.

## Boundary ambiguity: real, but a tail

At grid 32 a boundary cell is part object and part background and its label comes from a
majority vote; `patch_purity` records the winning share. The patches the readout misses are
systematically the ambiguous ones:

| | n | mean purity | p50 | share below 0.6 |
| --- | ---: | ---: | ---: | ---: |
| covered foreground | 6697 | 0.9170 | 1.0000 | 5.4% |
| **missed foreground** | 385 | **0.7678** | 0.7500 | **23.1%** |
| background | 54358 | 0.9877 | 1.0000 | — |

Missed patches are 4.3x more likely to be narrow-majority cells. But counting only cells that
are close to a coin flip, **14.3%** of the miss is grid ambiguity (purity < 0.55), rising to
40.5% at purity < 0.70. So aliasing lowers the honest ceiling at grid 32 a little and does
**not** explain the gap: 60–77% of the misses are unambiguous object cells the model did not
claim. Ties are negligible — 0.5% of missed cells against 0.1% of covered.

## Limitations

- Seed 0 alone, 60 images. Seed 0 has the highest `pred_fg` of the three (0.1512), so it is
  the most over-predicting seed and the 22.2% is likely its upper end.
- "Covered" means claimed by any group. `PV2_0106`'s coverage figure may be stricter; the two
  are still not reconciled (see `PV2_0113`).
- Removing every false positive is an oracle operation. Whether any label-free signal can
  find them is a separate question and is measured before anything is built — the lesson
  `PV2_0112` paid for.

## Next

`PV2_0115`, gated on a signal check that runs first: per predicted group, measure internal
spike coherence, mean activity, size and border contact against the fraction of that group
that is truly foreground. If none of them separates spurious groups from real ones, a
suppression stage is not written. If one does, it becomes the dial, selected on validation,
with the same three-metric gate.
