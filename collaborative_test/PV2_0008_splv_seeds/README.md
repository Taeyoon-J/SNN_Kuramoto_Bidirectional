# PV2_0008 — spike synchrony in the objective, three seeds

**completed, best result so far. The goal is not yet met.** Parent: `PV2_0006`.
Contract v1, split v1.

## Setting

`PV2_0004`'s model plus two changes, both already screened: the spatial prior
tightened to 0.35 (`PV2_0005`) and `--spike-plv-weight 5`, which puts
per-component spike synchrony into the objective alongside the phase terms.
Before this, the dendritic and membrane time constants were drawn from
sigmoid(U(-4,0)) and had never been trained towards keeping synchrony, because
the objective read only the phases.

Threshold **0.20**, selected on the **full validation split (1000 images)** from
the 3-seed mean, before test was read. Test was then read once. The score comes
from the classifier's masks on the model's own spikes; no phase readout and no
oracle.

## Result — test, 300 images, seeds 0/1/2

| metric | seed 0 | seed 1 | seed 2 | **mean** | std | `PV2_0004` | delta | rel | Slot Attention | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `patch_fg_ari` | 0.4812 | 0.6432 | 0.6982 | **0.6075** | 0.1128 | 0.5105 | +0.0970 | +19.0% | 0.6195 | short 0.0120 |
| `patch_foreground_iou` | 0.6028 | **0.7317** | 0.6921 | **0.6755** | 0.0660 | 0.4967 | **+0.1788** | **+36.0%** | 0.1241 | **beats** |
| `patch_matched_object_iou` | 0.3513 | 0.4542 | 0.4840 | **0.4298** | 0.0696 | 0.3689 | +0.0609 | +16.5% | 0.0920 | **beats** |

Secondary, all 1000 test images: fg_ari 0.6140, foreground IoU 0.6727,
matched-object IoU 0.4421 -- consistent with the 300-image rows. The 300-image
column is primary only because the Slot Attention reference and `PV2_0001`/`0004`
were all scored on it.

Slot Attention is one published checkpoint, not a 3-seed training mean.

## Against the goal

The goal is foreground IoU at or above 0.700 as a 3-seed mean. **0.6755 is 0.0245
short.** Seed 1 alone reaches 0.7317. Foreground ARI is also 0.0120 below the
Slot Attention reference, so the protocol's all-three condition is not met either.

Validation predicted 0.6730 and test gave 0.6755, so nothing here is overfitting
to the selection split.

## The seed that is holding it back

Seed 0 is worst on all three and by a wide margin on grouping: fg_ari 0.481
against 0.643 and 0.698. Predicted foreground is not the explanation -- 0.124,
0.127, 0.144 across the seeds, all near the 0.126 target.

If seed 0 behaved like the other two, the mean would land near 0.71.

It is not instability, and the first version of this record said it was. Two
corrections. `--grad-clip-norm` defaults to **1.0**, so gradient clipping was
already active in every run here and in weight 3's divergence -- the claim that
none was used was simply wrong. And the loss traces say the opposite of
undertraining:

| seed | final loss | foreground IoU |
| --- | --- | --- |
| 0 | **15.31** (lowest) | **0.603** (worst) |
| 1 | 16.67 (highest) | **0.732** (best) |
| 2 | 15.47 | 0.692 |

**Lower training loss goes with a worse metric.** Seed 0 minimised the objective
best and scored worst, so the objective is misaligned with the metric rather than
badly optimised. Three points is weak evidence for the anti-correlation, but it is
the opposite of what an undertrained seed predicts, which is enough to retire the
stability hypothesis.

## A reporting error this experiment exposed

Weight 5 was reported at seed 0 as "foreground IoU 0.586". Seed 0 turned out to be
the worst of the three on every metric, so that number understated the setting by
about 0.09. Exploration at one seed is allowed, but the earlier message calling
0.586 the number for this setting was wrong, and this is the third time a
single-seed reading has had to be corrected.

## Next

`PV2_0009`: re-balance the four PLV terms, screening on seed 0 because it is the
drag on the mean, with three further seeds of the unchanged setting alongside to
test the anti-correlation on six points instead of three. Optimiser settings are
not the lever -- making the objective easier to minimise is, if anything, the wrong
direction.
