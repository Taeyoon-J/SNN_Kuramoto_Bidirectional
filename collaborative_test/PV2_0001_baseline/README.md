# PV2_0001 — baseline

**completed.** Parent: none. Contract v1, split manifest v1.

## Hypothesis

Establish what the current configuration scores under the contract, from masks a
classifier builds out of spikes. Every headline number this repository had came
from spectral clustering of PLV at the true cluster count, which the contract
makes a diagnostic, so there was no legitimate baseline to compare anything to.

## Result

Test split, first 300 images, three seeds. Thresholds were chosen on validation
and test was read once.

| metric | seed 0 | seed 1 | seed 2 | mean | std | Slot Attention | |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `patch_fg_ari` | 0.3816 | 0.4343 | 0.5213 | **0.4457** | 0.0706 | 0.6195 | below |
| `patch_foreground_iou` | 0.3128 | 0.5087 | 0.4129 | **0.4115** | 0.0980 | 0.1241 | beats |
| `patch_matched_object_iou` | 0.2407 | 0.3717 | 0.4191 | **0.3438** | 0.0924 | 0.0920 | beats |

Two of three beaten, `fg_ari` short by 0.174. The goal is all three, so this is
not met.

The Slot Attention column is one published checkpoint, not a 3-seed training
mean, and it was trained on a different render; see `baselines/slot_attention.md`.

## Diagnostics

| | seed 0 | seed 1 | seed 2 |
| --- | --- | --- | --- |
| spike rate | 0.043 | 0.224 | 0.324 |
| groups per image | 4.55 | 6.81 | 8.22 |
| predicted foreground fraction | 0.228 | 0.170 | 0.280 |

Target foreground is 0.126 of patches.

## Insight

**The standard deviations, 0.07 to 0.10, are wider than most differences this
project has been measuring.** Several comparisons made before this baseline
existed were single-seed and sit inside that spread.

The likely source is in the diagnostics: **the spike rate varies sevenfold across
seeds**, 0.043 to 0.324, and the seed with the lowest rate is last on all three
metrics. Nothing in the objective constrains the firing rate -- `spike_rate_weight`
is 0 -- so the seed decides it.

## Next

`PV2_0002a`: constrain the firing rate and see whether the variance falls. Cheap,
and every later comparison depends on being able to tell an effect from a seed.
The alternative, `PV2_0002b`, is the coupling graph, which is the one lever whose
ceiling has been measured -- training on a ground-truth graph, a diagnostic, gives
foreground IoU 0.610 against 0.421 -- but running it at std 0.1 would risk
reading noise again.

## Limitations

300 of the 1000 test images. Thresholds (synchrony 0.40, min group size 2) were
selected on validation for the seed-2 checkpoint and applied to all three; a
per-seed selection would be closer to the contract but would start fitting
validation.
