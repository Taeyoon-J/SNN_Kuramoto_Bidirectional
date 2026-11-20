# PV2_0051 — more data did not improve the features either

**completed. Closes data scale on the mechanism, not only on the score, at zero
GPU cost.** Parent `PV2_0048`. Diagnostic; no score is reported here.

## Why it was needed

`PV2_0048` held updates constant — 6k x 40, 20k x 12, 40k x 6, all ~240k
presentations — and fg_ari fell monotonically: **0.6779, 0.5983, 0.4790**. But
holding updates constant necessarily cut exposures per image from 40 to 6, so that
result could not separate "more distinct images does not help" from "each image
was seen too few times". The training loss pointed at the second reading: 15.55 at
12 epochs against 16.01 at 6, with the 40k run still descending when it stopped
(18.52 -> 16.01).

The expensive way to settle it was 40,000 images at the full 40 epochs — 1.6M
presentations, about 27 hours. That run was queued and then **cancelled**, because
the question can be answered from what `PV2_0048` already wrote to disk.

## What was measured

`PV2_0035/0036` reduced "better features" to two numbers: cutting within-object
spread from 0.465 to 0.000 moves fg_ari 0.7239 -> 0.8935, and closing the
between-object margin from 1.780 to the oracle's 3.947 carries it to 0.9754. So the
encoders from `PV2_0048` can be judged directly, with no training and no GPU.

| training data | within | between | **w/b** | fg_ari |
| --- | --- | --- | --- | --- |
| 6,000 images, 40 epochs (bar) | 0.4579 | 1.7881 | **0.2561** | 0.6779 |
| 20,000 images, 12 epochs | 0.0566 | 0.2377 | **0.2380** | 0.5983 |
| 40,000 images, 6 epochs | 0.0583 | 0.2354 | **0.2476** | 0.4790 |
| ORACLE codes | 0.0000 | 3.9447 | **0.0000** | 0.9754 |

The absolute scale fell about 7.5x, but `s2net_cls.py:413-416` standardises gamma
**per image** before the phase mapping, so scale is removed and the comparable
quantity is the scale-invariant ratio. That ratio improved by **7% relative**
(0.2561 -> 0.2380) while the oracle needs 0.0000 and the measured value of taking
within-object spread to zero is +0.170 fg_ari.

**More data does not meaningfully improve the encoder's separation.** Data scale is
closed on both readings, and the 27-hour run is not worth starting.

## A measurement error caught on the way

The first pass reported 0.4651 / 1.7803 **identically** for all three — impossible
for three different encoders. `gen_gamma.py` keeps the manifest aligned by
rewriting only rows 6000-6999 of the base tensor and leaving the rest untouched, so
reading rows 0-999 compared three copies of the same unchanged block. Re-run on the
validation rows, after checking that those rows really do differ between files
(`validation rows equal: False`).

## Limitation

Within/between come from the ground-truth masks, so these are oracle-guided
measurements of feature quality — diagnostics, never scores.
