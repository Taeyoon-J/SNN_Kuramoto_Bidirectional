# Slot Attention scored on our images, our grid, our evaluation function

2026-11-04. This records the exact procedure, so the comparison can be reproduced or
disputed on its details rather than on its conclusion.

## Why this exists

Comparisons so far placed Slot Attention's published **98.8** FG-ARI beside our patch score.
Those two numbers come from different renders, different object counts and different
scoring protocols, so the pair never meant anything. The benchmark contract already says
what to do instead: *"Slot Attention masks go through the same patch conversion and the same
evaluation function."* This is that run.

## Result

CLEVR, the 320 evaluation images (`fixed320`, ids 1320–1639), grid 64,
`evaluate_patch_masks` against `dataset.labels(ids, 'fixed320')`.

| | fg_ari | fg_iou | obj_iou | predicted foreground |
| --- | ---: | ---: | ---: | ---: |
| **253, unsupervised, one seed** | **0.8421** | **0.7545** | **0.6410** | 0.2001 |
| Slot Attention, grid 64 | 0.3808 | 0.2022 | 0.1053 | **0.8510** |
| Slot Attention, grid 32 | 0.3912 | 0.2028 | 0.1070 | 0.8450 |
| ground-truth foreground | | | | 0.2115 |

**The grid is not the constraint.** 32 and 64 agree to 0.01. Slot Attention calls **85% of
the image foreground against a 21% truth**, and that one failure moves all three metrics.
An earlier baseline of ours measured the same 85.3% on a different split.

## Procedure, step by step

**1. The same images.** `fixed320` is ids 1320–1639 of `clevr_10-full.hdf5`, which the
engine uses as its evaluation split. The matching PNGs are
`CLEVR_masks_001320.png` … `CLEVR_masks_001639.png` in `clevr/with_masks/images40k`. A
manifest is built by reading `Dataset('clevr', …).test` and naming those files, with an
assert that every one exists — so the image set is taken from the engine, not retyped.

**2. The published checkpoint, unmodified.** `ckpt-500` from
`gs://gresearch/slot-attention/object-discovery`, loaded into the authors' own `model.py`
with `model_type="object_discovery"`, `resolution=(128,128)`, `num_slots=7`,
`num_iterations=3`. The restore prints optimizer-state warnings; those are the optimizer
slots, not weights, and are expected when restoring for inference.

**3. Aligned preprocessing, `--preprocess resize`.** The published centre crop takes the
middle 240×240 of a 240×320 frame, which makes the prediction grid and the target grid
describe different regions — `PV2_0090` measured that mismatch at **+0.0513** against Slot
Attention and corrected it. `resize` scales the whole frame to 128×128 and to [-1, 1].
**This is the setting that favours Slot Attention**, and it is the one used here.

**4. Masks to the grid by the same rule as the targets.** The per-pixel slot assignment is
`argmax` over the slot masks at 128×128. With `--grid 64`, each 2×2 pixel block votes by
majority — `clevr_mask_patch`'s rule, the same one that builds the targets. The slot
covering the most patches over the whole split is remapped to id 0, since slot ids are
arbitrary and one of them has to be background without consulting the truth.

**5. The same scoring call.** `evaluate_patch_masks(labels, dataset.labels(ids,'fixed320'))`
— the identical function and targets the engine uses for `historical_fixed64_patch`. No
re-implementation; `metric = c.benchmark_module('evaluation')` loads the immutable module.

```bash
python -u run_slot_attention.py \
  --checkpoint-dir  <slot_attention>  --model-dir <slot_attention> \
  --image-dir       clevr/with_masks/images40k \
  --manifest        sa_fixed320_manifest.json \
  --split fixed320 --grid 64 --preprocess resize \
  --out             sa_g64_fixed320.npz

python -u score_sa.py --labels sa_g64_fixed320.npz \
  --config configs/frontier_staged.json --run <253 output root>
```

## What this comparison does not establish

- **The render is not Slot Attention's.** This split is `clevr_with_masks`; the authors
  trained on the original CLEVR render, and their README says results on the two are not
  directly comparable. It is also CLEVR10, not the CLEVR6 the 98.8 was measured on.
- **One checkpoint, not three seeds.** The published `ckpt-500` is a single checkpoint; its
  column is a single-checkpoint reference and must never be written as a 3-seed mean.
- **We are on home ground.** 253 was trained on this split. The fair sentence is *"on
  identical data and identical scoring we lead, and this data is not the render Slot
  Attention was trained on"* — not *"we beat Slot Attention"* unqualified.
- **253 is one seed too.** Seeds 1 and 2 are not run.

## The ceiling, which decides what any of this can claim

A grid-`g` prediction cannot resolve anything finer than a cell, so quantising the pixel
truth to the grid and scoring it against itself is a hard upper bound — a property of the
data, not of a model.

| dataset | native pixels | grid 32 | grid 64 | their published pixel FG-ARI |
| --- | --- | ---: | ---: | ---: |
| CLEVR | 128×128 | 0.8823 | **0.8988** | 0.988 (CLEVR6) |
| Tetrominoes | **35×35** | **0.9352** | impossible | 0.995 |
| Multi-dSprites | **64×64** | 0.6017 | **1.0000** | 0.913 |

Against the **published pixel numbers**, CLEVR and Tetrominoes cannot be won at any grid we
can run: Tetrominoes has 35 pixels a side, and CLEVR would need grid 128, whose
`direction_learner` is 16,384² = 268M parameters — sixteen times grid 64's memory.
**Multi-dSprites is the exception**: ceiling 1.0000 at grid 64, and their weakest published
result. That is the dataset where a claim on their own terms is possible, and it is also
where a CLEVR-only core of ours already reaches 0.6893 on the pixel protocol having never
seen the dataset.

## The trained native-64 core, for the peer

`/work/USERS/tkim1/share/native64_core_253/` on frontier, world-readable
(`/work/USERS/tkim1` gives group `acm` `r-x`, and the peer is in `acm`):

| file | |
| --- | --- |
| `resume.pt` | 253 at grid 64, CLEVR, 10,000 unique × 10 epochs, 6,250 updates, seed 0 |
| `initial.pt` | initial weights and the TRAIN-fitted normalization |
| `binding.json` | the 10,000 TRAIN ids, validation and evaluation ids |
| `training_report.json` | gradient coverage, parameter changes, `mask_opened_during_training: false` |

This is an actually trained native-64 core, not a resolution conversion: all 4,096 positions
have independent `omega`, `kappa`, `tau` and a 4,096×4,096 `direction_learner` learned from
scratch, against `SW0261`'s 16×16 core replicated 4×4 into 4,096 tied positions.
