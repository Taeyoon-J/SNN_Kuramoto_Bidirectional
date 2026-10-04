# PV2_0004 — per-component spikes on the geodesic graph

**completed, best result so far.** Parent: `PV2_0002b`. Contract v1, split v1.

## Hypothesis

Foreground IoU was capped at 0.412 and a ground-truth coupling graph gave 0.610,
so what limited it was the spike synchrony matrix rather than the graph. The
dendrite collapses the osc_dim components into one signal per unit in its first
linear map, and the phase side showed that collapse is expensive -- measuring
synchrony per component instead was worth 0.202 there. Putting the components
back should restore separation the spikes were losing.

## Change

`--spike-per-component` on top of `PV2_0002b`'s geodesic graph. No architecture
change: the component axis is folded into the batch axis so the spiking layers
run once per component with every weight shared.

## Result

Test split, 300 images, three seeds. Threshold 0.010, selected on validation
across all three seeds.

| metric | seed 0 | seed 1 | seed 2 | mean | std | PV2_0001 | delta | Slot Attention | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `patch_fg_ari` | 0.5287 | 0.4255 | 0.5772 | **0.5105** | 0.0775 | 0.4457 | +0.0648 (+14.5%) | 0.6195 | below |
| `patch_foreground_iou` | 0.4909 | 0.5866 | 0.4125 | **0.4967** | 0.0872 | 0.4115 | +0.0852 (+20.7%) | 0.1241 | **beats** |
| `patch_matched_object_iou` | 0.3394 | 0.3647 | 0.4027 | **0.3689** | 0.0319 | 0.3438 | +0.0251 (+7.3%) | 0.0920 | **beats** |

All three above the baseline, and matched-object IoU's spread falls from 0.092 to
0.032. Two of three still beat the Slot Attention reference; foreground ARI is
short by 0.109.

## The correction that made this visible

Per-component synchrony is a **product over four components**, so its values are
two orders of magnitude smaller than a single correlation. Scored at the
threshold inherited from the single-correlation readout, 0.40, it filtered almost
nothing and predicted 0.335 of patches as foreground against a target of 0.126,
which is why the first measurement of per-component spikes read 0.343 foreground
IoU and was written up as no help. Swept properly it peaks at 0.486 near 0.002
and sits above baseline on all three metrics near 0.010.

That earlier conclusion -- "carrying the components through the SNN does not help"
-- was an artefact of a threshold carried over from a different measure.

## Diagnostics

| | seed 0 | seed 1 | seed 2 |
| --- | --- | --- | --- |
| spike rate | 0.488 | 0.508 | 0.423 |
| groups per image | 5.21 | 6.71 | 7.97 |
| predicted foreground | 0.182 | 0.161 | 0.292 |

Target foreground is 0.126 of patches, so foreground is still over-predicted by
about 1.5x, and the seed that over-predicts most has the lowest foreground IoU.

## Next

Foreground IoU is at 0.497 against a goal of 0.700. The ceiling that motivated
this round -- 0.610 from a ground-truth graph -- was measured with the old
threshold and the old readout, so it needs re-measuring before it is quoted
again. The threshold sensitivity is also still the largest single source of
variance.
