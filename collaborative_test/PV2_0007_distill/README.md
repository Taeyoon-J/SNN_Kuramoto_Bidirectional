# PV2_0007 — distil the phase synchrony into the spikes (failed)

**failed.** Parent: `PV2_0006`. Contract v1, split v1. Validation, 100 images, seed 0.

## Why it was tried

`PV2_0006` added a measurement that had been missing: `--readout {spikes,plv}`,
which runs the same classifier, background rule and evaluation and changes only
where the synchrony comes from. On one checkpoint:

| GEOPC | fg_ari | foreground_iou | matched_object_iou |
| --- | --- | --- | --- |
| spike readout (the score) | 0.5993 | 0.4858 | 0.4078 |
| phase readout (DIAGNOSTIC) | 0.6506 | 0.6596 | 0.4070 |

The spiking path loses **0.174 foreground IoU** of what theta already carries --
larger than the coupling graph is worth with ground-truth coupling (+0.06) and
than every classifier knob together (0). The information the goal needs is in
theta; the spiking path destroys it.

`--spike-plv-weight` does not recover it. It applies the same PLV criterion to
the spike matrix separately, which sharpens that matrix on its own terms without
tying it to the phases, and at weight 5 it trades: spike readout 0.486 -> 0.586
while the phase readout falls 0.660 -> 0.580.

## Change

`--spike-distill-weight`: an MSE between the per-component spike synchrony matrix
and the phase PLV matrix, theta detached so the gradient only enters the spiking
layers. Tried alone, with `--spike-plv-weight` at 0.

## Result: it made things much worse

| run | fg_ari | foreground_iou | matched_object_iou | predicted_fg |
| --- | --- | --- | --- | --- |
| `DIST_w2` best | 0.6128 | **0.3661** | 0.4424 | 0.3564 |
| `DIST_w10` best | 0.5749 | 0.3681 | 0.3647 | 0.2890 |
| `DIST_w50` best | 0.5129 | 0.3348 | 0.2740 | 0.1150 |
| for comparison, `SPLV_w5` | 0.6075 | **0.6755** | 0.4298 | 0.1190 |

Foreground IoU collapses from about 0.65 to 0.37. The phase side falls too, to
0.494-0.654 from 0.660.

## Why it failed

The optimisation worked -- the distillation term fell from 0.24 to 0.088 -- so the
objective itself was wrong.

Matching the full affinity matrix under MSE drives the spike synchrony towards
the phase PLV's **values**, and the phase PLV is dense and high. Every pair
becomes similar, the graph fuses into one enormous component, and predicted
foreground balloons to 0.29-0.36 against a target of 0.126. What the readout
needs from the phases is the **contrast** between within-object and
between-object pairs, and an MSE on magnitudes does not preserve it: a matrix can
match the teacher closely in mean-squared terms while carrying none of its
structure.

The 0.174 gap is real and remains the largest lever. This was the wrong way to
close it. A loss on the *ranking* or the *margin* between within- and
between-object pairs, rather than on the values, is the version of this idea that
is still open.

## Status of the lead that did work

The gain that stands came from elsewhere: `--spike-plv-weight 5` at seeds 1 and 2,
recorded as `PV2_0008`.
