# PV2_0002b — geodesic coupling graph

**completed, partial improvement.** Parent: `PV2_0001`. Contract v1, split v1.

## Hypothesis

Foreground IoU is the goal and the graph is the one lever whose ceiling has been
measured: training on a ground-truth graph gives 0.610 against 0.421.

What the learned graph gets wrong is specific. Its edges come from feature cosine
under a Euclidean distance prior, so two objects of the same colour -- alike to
the features, a few patches apart -- stay linked at 0.418 mean edge weight against
0.749 inside an object, and 83% of CLEVR scenes contain a repeated colour. The
route between them crosses background, which a straight line cannot see.

## Change

`geodesic_steps=3`, `geodesic_contrast=2.0`. Each step costs Euclidean distance
scaled by dissimilarity, non-neighbours are barred, and three rounds of softened
min-plus relaxation find the shortest route. Verified before training: two
identical patches three apart get 0.123 under the geodesic against 1.141 under
the Euclidean prior, with an immediate neighbour still at 0.458.

## Result

Test split, 300 images, three seeds, synchrony threshold 0.50 chosen on
validation.

| metric | seed 0 | seed 1 | seed 2 | mean | std | PV2_0001 | delta | Slot Attention |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `patch_fg_ari` | 0.5644 | 0.4568 | 0.6397 | **0.5537** | 0.0919 | 0.4457 | +0.1080 | 0.6195 |
| `patch_foreground_iou` | 0.5487 | 0.2211 | 0.3390 | **0.3696** | 0.1659 | 0.4115 | -0.0419 | 0.1241 |
| `patch_matched_object_iou` | 0.3659 | 0.2955 | 0.4815 | **0.3809** | 0.0940 | 0.3438 | +0.0371 | 0.0920 |

Foreground ARI improves by 0.108 and matched-object IoU by 0.037. Foreground IoU
falls by 0.042, and that is a threshold artefact rather than the model: the
predicted foreground lands at 0.178, 0.372 and 0.354 of patches across the three
seeds at one fixed threshold, and foreground IoU tracks it inversely -- 0.549,
0.221, 0.339.

## What did not fix the threshold sensitivity

**Per-image calibration.** Binary-searching the threshold so the groups cover a
target fraction. Dropped: it costs the oracle graph 0.610 to 0.482, and on the
three seeds it gives 0.397, 0.233, 0.394 at target 0.10 and 0.445, 0.229, 0.428
at 0.16. Equalising how much foreground is predicted does not make it the right
foreground.

**A hybrid background rule.** Ranked by mean spike synchrony, the quietest
patches recover the true foreground at IoU 0.400 against 0.062 for chance -- the
best spike-domain signal there is, against 0.168 for firing rate and 0.134 for
mean membrane. Requiring a unit to pass both that ranking and the component rule
peaks at 0.409 at quantile 0.40, level with the baseline. Raising the quantile
trades foreground IoU for foreground ARI, which reaches 0.537 at 0.75.

## Insight, and what it means for the goal

**The goal sits above a measured ceiling.** Nothing tried beats 0.412 foreground
IoU, and a ground-truth coupling graph gives 0.610. So neither the graph nor the
classifier rule is what caps it.

What caps it is the spike synchrony matrix: with perfect coupling it still does
not separate background from objects cleanly. The best spike signal ranks the
foreground at 0.400 where the phase-side equivalent reaches 0.525, so the spiking
representation is losing separation the phases already have. Raising the ceiling
means putting that back, not tuning what reads it.

## Next

`PV2_0003`, on top of the geodesic graph: per-component spikes, which stop the
dendrite collapsing osc_dim into one signal per unit, and pulse coupling, which
closes the loop from spikes back onto the phases and is the half of the
architecture that is currently switched off.
