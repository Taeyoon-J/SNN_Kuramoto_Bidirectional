# PV2_0086 — motion helps, on a platform strong enough to say so

**Status:** completed. **Parent:** `PV2_0085`. **Contract:** movi-1, not comparable with v1.
**Hypothesis:** `PV2_0085`'s negative verdict on motion was not earned, because its
appearance-only control scored 0.3042 against contract v1's 0.7239. A term cannot be shown
to add nothing to a model that is not yet working.
**Success criterion:** a control that trains properly, and then a motion arm that beats it
**inside the calibrated regime** (predicted foreground below 0.20 against a target of 0.1061).

## What changed

Only the input representation. `PV2_0085` drove the model with `gamma_app_*`: three raw
RGB channels. This uses `gamma8s_*`, an 8-channel encoder trained on MOVi-A and
standardised — std 0.9858 against contract v1's 0.9866, between/within separation ratio
0.0646 against v1's 0.0571. Same arms, same motion-affinity term, same data, same seed 0,
same 40 epochs.

The contract-v1 regression row passed first: 0.7239 / 0.7251 / 0.4951, exactly.

## The platform is fixed

| | first epoch -> last | calibrated control fg_ari |
| --- | --- | --- |
| `PV2_0085`, 3 raw RGB channels | 28.25 -> 23.48 | 0.3042 |
| here, 8 trained channels | **21.14 -> 14.95** | **0.4723** |

The control now predicts foreground at **0.1006** against a target of 0.1061 without any
threshold tuning, so the arms are comparable where it matters.

## Result

Validation, 250 images, seed 0. Default readout.

| weight | fg_ari | fg_iou | obj_iou | pred_fg | in the gate? |
| --- | --- | --- | --- | --- | --- |
| 0 (appearance only) | 0.4723 | 0.5609 | 0.3531 | 0.1006 | yes |
| **10** | **0.4826** | **0.5613** | **0.3735** | 0.1198 | yes |
| 100 | 0.4910 | 0.4140 | 0.3784 | 0.1989 | borderline |

At weight 10: **+0.0103 fg_ari, +0.0004 fg_iou, +0.0204 obj_iou**, with predicted
foreground moving 0.1006 -> 0.1198. The motion-affinity term itself fell 0.2033 -> 0.1974
-> 0.1718 across the arms, so it is being optimised.

Weight 100's larger +0.0187 comes with predicted foreground nearly doubled and foreground
IoU down 0.1469. That is the calibration-gate failure mode three automated verdicts in
this project have already been wrong about, and it is not counted.

## Reading it honestly

This is the **first positive signal in roughly thirty-one routes**, and it is small. Part
of the +0.0103 is bought by the 0.019 widening of predicted foreground, which fg_ari
rewards for free; `obj_iou` +0.0204 is the harder number to explain away, since it is not
helped by over-prediction.

It does not establish anything yet. One seed, one dataset, 250 validation images, and no
Slot baseline exists on this contract. `PV2_0085`'s conclusion stands corrected in one
respect only: **its control was too weak to carry a verdict, and on a working platform the
motion term does not hurt and may help.**

## Next

1. Seeds 0, 1, 2 at weight 10, and a weight sweep between 10 and 100 to find where the
   gain stops being calibration.
2. Re-run the comparison at **matched predicted foreground**, so the fg_ari difference
   cannot be attributed to extent at all.
3. Only then a Slot Attention baseline on movi-1 — built with `--preprocess resize`,
   per `PV2_0090`.

## Limitations

MOVi-A is not the project's dataset and nothing here transfers to contract v1 numerically.
The test range of `manifest_movi.json` is deliberately empty; this is a signal test, not a
result. The 8-channel encoder was trained on MOVi-A with a reconstruction objective and no
masks, so it uses no label information, but it is a second thing that differs from
contract v1 besides the dataset.
