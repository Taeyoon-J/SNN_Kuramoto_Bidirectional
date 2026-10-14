# PV2_0091 — the readout dial cannot reach 0.9, and that is now measurable

**Status:** completed from recorded values; the clean sweep is still owed (see Limitations).
**Parent:** `PV2_0059`, `PV2_0060`, `PV2_0061`, `PV2_0076`, reframed by `PV2_0090`.
**Contract:** v1, unchanged.
**Question:** `patch_fg_ari` is computed over true-foreground patches only, so predicted
extent moves it for free and comparing two models at different extents says nothing. Is
there **any** point on our extent dial that reaches 0.9, or even matches aligned Slot
Attention, while predicting no more foreground than Slot does?

## Why this is answerable now and was not before

Until `PV2_0090` no model had been measured near 0.8 on this data under this contract, so
"0.9 is out of reach here" rested on our own ceiling arguments. Now there is an external
point on the same images, the same targets and the same scoring function: **aligned Slot
Attention, 0.8102 fg_ari at a predicted foreground of 0.8531.**

## Our dial, from values already recorded

Spectral `k` is the control for predicted extent (`PV2_0059`, `PV2_0060`). Collected from
the records rather than rerun:

| configuration | fg_ari | predicted foreground | source |
| --- | --- | --- | --- |
| connected components | 0.7300 | 0.129 | `PV2_0076` |
| spectral k=12 | 0.7381 | — (foreground IoU 0.4413) | `PV2_0060` |
| spectral k=14 | **0.7603** | 0.556 | `PV2_0061` + `PV2_0076`, the published result |
| spectral k=20 | **0.7718** | 0.808 | `PV2_0060` |
| chroma3 | 0.7380 | 0.509 | `PV2_0058` |
| whitening | 0.7670 | 0.590 | `PV2_0064` |
| **aligned Slot Attention** | **0.8102** | 0.8531 | `PV2_0090` |
| a perfect segmenter | 0.9756 - 1.0000 | — | `PV2_0082` |

## What the dial says

**1. Extent cannot close the gap.** At `k=20` we predict 0.808 foreground — within 0.045
of Slot's 0.8531 — and score **0.7718 against its 0.8102**. Buying the full width Slot
buys does not buy its score; at essentially matched extent we are **about 0.04 behind**.

**2. The dial's own maximum is far below the bar.** The highest fg_ari recorded anywhere
on it is **0.7718**, which is **0.13 below 0.9** and below even the 0.7756 midpoint
between our calibrated 0.7300 and Slot's 0.8102. The curve is also flattening: from k=14
to k=20 predicted foreground rises 0.556 -> 0.808, a 45% widening, for +0.0115 of fg_ari.

**3. So the gap is in what the spikes encode, not in how they are thresholded.** This is
the measurement the route map was missing. Every previous statement that the architecture
was exhausted rested on enumerating closed routes; this one is a property of the readout
itself, read off its own operating curve.

## What follows for 0.9

`PV2_0082` established that contract v1 admits 0.9 comfortably — a perfect segmenter
scores 0.9756 even under a far stronger rule change than a tie-break. So the bar is real
and the obstacle is the model.

Putting the three together:

- the contract admits 0.9 (`PV2_0082`),
- the data admits 0.81 and nothing measured on it has exceeded that (`PV2_0090`),
- our readout saturates at 0.772 regardless of extent (here).

**An architectural change is necessary, not merely desirable.** That is the premise
`PV2_0092` was built on, and this record is what makes it a measured premise rather than
an inference from thirty-one failures. It does not say signed coupling will work; it says
no setting of the existing readout can, which is a different and firmer claim.

It also sets the size of the task honestly. Signed coupling passing its gate would be a
step, not an arrival: the nearest external evidence of what a well-functioning model does
on these images is 0.8102, and 0.9 is above everything anyone has measured here.

## Limitations

These values come from **different runs and configurations** gathered from the records,
not from one checkpoint swept in one session, so the curve is indicative rather than a
clean monotone measurement. The sweep that would fix that — one checkpoint, spectral k
from 6 to 32, `--limit 150 --num-time-steps 512` so it is cheap — was launched, produced
its first row (components, 0.7239 / 0.7251 / 0.4951 at 0.1290, reproducing exactly), and
was stopped at 25 minutes per row under GPU contention with the 32x32 training. It is
still owed, and it would sharpen point 1 by giving k=20 and k=24 on the same checkpoint
as k=14.

The conclusion is robust to that, because the two numbers it rests on — k=20 at 0.7718 and
aligned Slot at 0.8102 — were each measured directly, and the gap between them is 0.034
while the gap to 0.9 is 0.13.

## Next

The clean sweep, then whatever `PV2_0092` returns. Neither changes this record's claim,
which is about the readout rather than about any particular training run.
