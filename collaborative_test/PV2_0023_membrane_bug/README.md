# PV2_0023-0025 — the membrane threshold is absolute while its scale is free

**completed. A real bug, and the single cause of the dead seeds.** Parent
`PV2_0013`.

## The bug

The membrane's steady state is `mem* ~ R_m * h_wave`, so its level follows the
dendritic drive, and nothing constrains that drive's scale. `v_th` is the
constant 0.06. Four seeds of one configuration, measured on a fixed batch:

| seed | membrane mean | above v_th | spike rate | |
| --- | --- | --- | --- | --- |
| s1 | +1.14 | 91.7% | 0.300 | healthy |
| s0 | +2.64 | **100.0%** | 0.468 | threshold inert |
| s3 | +0.26 | 72.5% | 0.284 | broken readout |
| s5 | **-0.65**, max **-0.046** | **0.0%** | **0.000** | **dead** |

Everything above threshold means every unit fires and the spikes stop carrying
membrane information -- seed 0, the chronically weak seed, which also
over-predicted foreground. Everything below means nothing fires and every mask is
empty by construction -- seed 5, whose membrane still has structure (std 0.128,
theta std 81.9).

## Confirmed by revival

Re-thresholding at the batch's mean membrane, **with no weight changed**:

| seed 5 | fg_ari | foreground_iou | matched_object_iou | rate |
| --- | --- | --- | --- | --- |
| absolute v_th | 0.0033 | **0.0000** | 0.0000 | 0.000 |
| population | **0.5916** | **0.6353** | **0.4122** | 0.137 |

The synchrony was there the whole time; an absolute threshold discarded all of
it.

The shift is one scalar for the batch, so ordering between units is preserved
exactly. Not a per-unit threshold: a per-unit median was tried earlier here and
destroyed the signal by forcing every unit to the same rate.

## What it does not explain

It explains seed 5. It does **not** explain seeds 3 and 4, whose membranes sit in
a healthy range yet call 0.50-0.58 of patches foreground; the population
threshold leaves them at foreground IoU 0.169 and 0.208. An earlier claim that
this one bug accounted for both failing seeds was too broad. Those are a separate
open failure.

## Two bugs found in fixing it

**The reset used a different threshold from the comparison.** The first version
thresholded on the population level while still resetting by the constant
`self.v_th`, so a unit could fire from a membrane near +2.6 and be reset by 0.06.
Trained that way the result collapsed: 3-seed fg_ari 0.6887 -> 0.3029, groups per
image 3.40 / 1.05 / 2.34 against a true 6.20. Both now come from one
`_threshold()`, and a regression check confirms absolute mode is unchanged to four
decimals (0.6362 / 0.7100 / 0.7199).

**`--help` crashed.** `"83% of scenes"` in a help string parses as `% o`, an octal
conversion, so argparse raised `%o format: an integer is required, not dict` for
any `--help`. It had been broken for a while; an earlier empty `--help | grep` in
this work was read as line wrapping, which was wrong. Escaped, and all six flags
added this session now display.

## Inference-only result, and why training it is separate

| | fg_ari (3 seeds) |
| --- | --- |
| absolute | 0.6887 |
| population, corrected reset | 0.6906 |

Neutral on the protocol seeds and it revives the dead one, which is the right
trade for correctness at no metric cost. Trained with it rather than only read
with it, seed 0's saturation should not arise at all -- that run is separate.
