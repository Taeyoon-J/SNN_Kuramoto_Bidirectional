# PV2_0101 — the 32x32 model was overtrained, and at seed 0 the shorter budget beat grid 16

> **The headline claim is WITHDRAWN as a three-seed result (`PV2_0102`).** At seeds 1 and
> 2 the 8-epoch model gives 0.2916 and 0.7068 against seed 0's 0.7699, for a three-seed
> mean of **0.5894 +- 0.2599** — far below the published 16x16's 0.7239. Seed 1 fails
> outright in both arms, by over-predicting foreground in the control (pred_fg 0.3101)
> and by merging almost everything into one group with the group-count term (1.54 groups
> per image). **8 epochs is not a stable operating point.**
>
> What survives: the **overtraining** finding. 8 epochs beats 40 epochs at seed 0 by
> +0.1406 / +0.0440 / +0.1132 on the same 300 images, and seed 2 (0.7068) also beats the
> 40-epoch model's 0.6293. The 40-epoch budget was wrong. What does not survive is that
> any fixed short budget is right.
>
> Everything below is the seed-0 record as written, kept for provenance.


**Status:** completed, seed 0. **Parent:** `PV2_0094`, `PV2_0097`. **Contract:** v2.
**This is the first configuration in this project to beat the published one on all three
metrics inside the calibrated regime.** It came out of a control arm, not out of the
intervention the experiment was built to test.

## How it surfaced

`PV2_0097` ran a group-count term at 8 epochs with a weight-0 control. The control scored
**0.7734 / 0.7571 / 0.7047** on validation 200, against `PV2_0094`'s 40-epoch model at
**0.6293 / 0.7091 / 0.5885** on validation 300. Better on all three with a fifth of the
training — but on a different image count, so not yet a result.

This re-scores both checkpoints on the **same 300 images** with the **same arguments**.

## Result

| | fg_ari | fg_iou | obj_iou | groups | pred_fg |
| --- | --- | --- | --- | --- | --- |
| 32x32, **40 epochs** (`PV2_0094`) | 0.6293 | 0.7091 | 0.5885 | 16.76 | 0.1276 |
| 32x32, **8 epochs** | **0.7699** | **0.7531** | **0.7017** | 15.84 | 0.1512 |
| 32x32, **8 epochs + group-count w5** | **0.7803** | **0.7807** | 0.6914 | **13.68** | 0.1444 |
| 16x16, published | 0.7239 | 0.7251 | 0.4951 | — | 0.1290 |

**8 epochs minus 40 epochs: +0.1406 fg_ari, +0.0440 fg_iou, +0.1132 obj_iou.** The
200-image rows reproduce (0.7734 / 0.6318 against 0.7699 / 0.6293), so the image count was
not the effect.

**Against the published 16x16 configuration**, at predicted foreground 0.1512 against
0.1290 — wider, but well inside the 0.20 calibration gate:

| metric | 16x16 | 32x32 at 8 epochs | change |
| --- | --- | --- | --- |
| `patch_fg_ari` | 0.7239 | **0.7699** | **+0.0460** |
| `patch_foreground_iou` | 0.7251 | **0.7531** | **+0.0280** |
| `patch_matched_object_iou` | 0.4951 | **0.7017** | **+0.2066 (+42%)** |

## The contract is not handing out the points

This is a cross-contract comparison, v1 against v2, which would normally be enough to
discount it. There is a control. `PV2_0099` measured the same grid change on **Slot
Attention's** predictions: 16 -> 32 is worth **+0.0069** to it. If the finer grid were
free points, Slot Attention would have collected them. About **0.039 of our +0.0460** is
not explained by the contract change.

## What this invalidates

**Every 32x32 number this project has reported was read off an overtrained model.**
`PV2_0094`, `PV2_0095`'s sweep, `PV2_0096`'s top_k sweep and `PV2_0099`'s comparison all
used `G32_POP40`. The conclusions they drew about the *readout* stand — those sweeps were
internally controlled — but their absolute numbers, and in particular `PV2_0095`'s
"groups 19.5 against 6.49 true objects", describe a model trained five times too long.
At 8 epochs the control already sits at 15.84 groups and the group-count term brings it to
13.68.

`PV2_0094` chose 40 epochs because that is what contract v1 uses. Nothing checked whether
it transfers, and `PV2_0093` had already shown that one hyperparameter tuned at 256
oscillators does not.

## Honest status

- **Seed 0 only.** `PV2_0102` runs seeds 1 and 2 of both arms.
- **8 epochs was not selected.** It is `PV2_0093`'s screening budget, inherited. That
  means it is not tuned on validation, which is good, and that it is probably not the
  peak, which needs an epoch sweep.
- **Validation only.** Contract v2's test split is unread.
- **No Slot Attention baseline exists on contract v2**, so this is not yet a claim against
  anything external. Building one requires `--preprocess resize` per `PV2_0090`.

## Next

Seeds (`PV2_0102`), then an epoch sweep to locate the peak, then a contract-v2 Slot
baseline, and only then one test read.
