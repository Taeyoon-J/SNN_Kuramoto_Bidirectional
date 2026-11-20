# PV2_0090 — the Slot Attention baseline was misaligned, in our favour

**Status:** completed. **Parent:** `PV2_0049`, `PV2_0061`. **Contract:** v1, unchanged.
**Hypothesis:** the published Slot Attention column is depressed by a preprocessing
mismatch rather than by the model.
**Success criterion:** a measurable change in the baseline's score from geometry alone.
**It is confirmed, and it overturns a standing claim.**

## The defect

`run_slot_attention.py` keeps the published preprocessing: centre-crop to the short
side, resize to 128x128. On a 240x320 frame that square is the centre **240x240**, so
the 16x16 map the baseline produces describes the centre of the image, while the 16x16
target map describes **the whole frame**.

Column `c` of the prediction is original x in `[40+15c, 40+15c+15)`.
Column `c` of the target is original x in `[20c, 20c+20)`.

Up to 40 px apart — **two whole target columns**. Every Slot Attention number this
project has reported was produced through that mismatch. The baseline record stated the
preprocessing and did not notice what it implied for the grid.

## Measurement 1: remove the crop

Same checkpoints, same images, same scoring; the crop replaced by a whole-frame resize,
which is what our own gamma encoder does to these images. Validation, 300 images, three
seeds, seeded inference.

| | fg_ari | fg_iou | obj_iou | pred_fg |
| --- | --- | --- | --- | --- |
| crop (as published) | 0.6682 | 0.1154 | 0.0878 | 0.8512 |
| whole-frame resize | **0.8065** | 0.1157 | 0.1017 | 0.8521 |
| difference | **+0.1383** | +0.0003 | +0.0139 | +0.0009 |

## Measurement 2: the clean one, no distribution shift

Measurement 1 puts the checkpoints off-distribution, so a *gain* there is informative
but a loss would not be. This arm moves nothing but the target: the **same crop
predictions**, scored against targets rebuilt on the **same centre 240x240 crop** under
the same modal rule.

| scored against | seed 0 | seed 1 | seed 2 | mean |
| --- | --- | --- | --- | --- |
| whole frame (as published) | 0.7286 | 0.7289 | 0.5470 | 0.6682 |
| its own crop | 0.7974 | 0.7903 | 0.5707 | **0.7195** |

**+0.0513 from the misalignment alone.** The remaining ~0.09 of measurement 1 is the
model genuinely doing better when it can see the whole scene instead of a crop that cuts
objects off at the left and right edges. Both measurements point the same way.

## Test re-score, 1000 images, three seeds x three seeded draws

| | fg_ari | fg_iou | obj_iou | pred_fg |
| --- | --- | --- | --- | --- |
| Slot, crop | 0.6656 +- 0.1089 | 0.1151 +- 0.0005 | 0.0874 +- 0.0042 | 0.8530 |
| Slot, aligned | **0.8102 +- 0.1250** | 0.1151 +- 0.0002 | 0.1025 +- 0.0080 | 0.8531 |
| ours, spectral k=14 | 0.7603 +- 0.0003 | 0.3373 | 0.6598 | 0.5560 |
| ours, components | 0.7300 +- 0.0017 | 0.7132 | 0.5050 | 0.1290 |

The crop row reproduces the published 0.6663 / 0.1177 / 0.0871 to within draw noise, so
the harness is the same one; only the preprocessing flag is new.

## What this does to the goal

**The claim "we beat Slot Attention on all three metrics" does not survive.**

| metric | ours (best) | aligned Slot | verdict |
| --- | --- | --- | --- |
| `patch_fg_ari` | 0.7603 | **0.8102** | **Slot leads by 0.0499** |
| `patch_foreground_iou` | **0.7132** | 0.1151 | ours, +0.5981 |
| `patch_matched_object_iou` | **0.6598** | 0.1025 | ours, +0.5573 |

Our own absolute number is untouched: 0.7603 still exceeds the 0.75 bar, and the two IoU
margins are not merely intact but larger in relative terms. What is gone is the fg_ari
lead.

**The obvious defence is available and I do not think it is sufficient.** fg_ari is
computed over true-foreground patches only, so a wider predicted foreground raises it for
free; aligned Slot reaches 0.8102 at a predicted foreground of 0.8531 against a target of
0.118, which is the behaviour `ROUTE_MAP_0.9.md` names as the calibration gate. But our
own 0.7603 comes at predicted foreground 0.5560, which is also far outside the calibrated
regime, and our calibrated configuration scores 0.7300. Applying the gate honestly leaves
**neither** number as a clean fg_ari result. What stands on its own is the figure/ground
and per-object evidence, where Slot Attention predicts 85% foreground against an 11.8%
target under every preprocessing tried.

## Limitations

The aligned run uses checkpoints trained with some preprocessing we no longer have the
script for, so the 0.8102 is Slot Attention evaluated off whatever distribution it was
trained on; a baseline retrained on whole frames could be better still, which would widen
the fg_ari gap further, not narrow it. Three seeds, spread 0.1250, driven by seed 2
(0.6658); the spread is the baseline's own, documented in `PV2_0049`. Validation and test
agree (+0.1383 and +0.1446), so this is not a split artifact.

## Reproduction

```
python run_slot_attention.py --checkpoint-dir runs/SA_s$SEED \
  --model-dir baselines/slot_attention --image-dir clevr/with_masks/images10k \
  --manifest data/split_manifest.json --split test --limit 1000 --grid 16 \
  --preprocess resize --seed $DRAW --out sa_test_s${SEED}_resize_d${DRAW}.npz
python crop_targets.py --tfrecord ... --split validation --limit 300 --grid 16 \
  --whole-frame-targets targets_v1.pt --predictions sa_align_s{0,1,2}_crop.npz
```

`--preprocess` defaults to `crop`, so every earlier command still reproduces its own
numbers.

## Next

The baseline record and `STATUS.md` are corrected in this commit. The 32x32 work
(`PV2_0089`) inherits the fix: any Slot32 baseline must use `--preprocess resize`.
