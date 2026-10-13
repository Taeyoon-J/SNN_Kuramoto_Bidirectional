# Slot Attention reference

Checkpoint: `gs://gresearch/slot-attention/object-discovery`, `ckpt-500`, the
object-discovery model from Locatello et al. 2020. **One published checkpoint,
not three seeds.** Its column is a single-checkpoint reference and is never
reported as a 3-seed training mean.

Run with `collaborative_test/run_slot_attention.py` under TensorFlow 2.15 in
`/work/USERS/tkim1/envs/tfenv`; scored with `score_slot_attention.py`, which
calls the same `evaluation.py` functions our model is scored with.

| | fg_ari | fg_iou | obj_iou |
| --- | --- | --- | --- |
| test split, 300 images | 0.6195 | 0.1241 | 0.0920 |

## Conditions that disadvantage it, stated rather than buried

It was trained on the **original CLEVR render**; this split comes from
`clevr_with_masks`, which is a different render with different object positions,
sizes and counts. The authors' own README says results on the two are not
directly comparable.

Its preprocessing is kept as published: centre crop to the short side, resize to
128x128, scale to [-1, 1]. Slot masks are mapped to the 16x16 grid by per-patch
majority vote, the same rule `clevr_mask_patch` applies to the targets.

> **CORRECTED by `PV2_0090`.** That crop is the centre **240x240** of a 240x320
> frame, so the prediction grid and the target grid describe **different regions**
> -- column `c` is up to 40 px, two target columns, apart. Every number below was
> produced through that mismatch and **understates Slot Attention**. Scored with
> `--preprocess resize` instead, on test, three seeds x three seeded draws:
>
> | | fg_ari | fg_iou | obj_iou | pred_fg |
> | --- | --- | --- | --- | --- |
> | crop (what the numbers below use) | 0.6656 | 0.1151 | 0.0874 | 0.8530 |
> | aligned | **0.8102** | 0.1151 | 0.1025 | 0.8531 |
>
> Scoring the *same* crop predictions against targets rebuilt on the *same* crop
> isolates the geometry at **+0.0513**; the rest is the model doing better when it
> can see the whole scene instead of one cut off at the left and right edges.
> **Aligned Slot Attention leads our 0.7603 on fg_ari by 0.0499.** Our two IoU
> margins are unaffected -- it predicts 85% foreground against an 11.8% target
> under either preprocessing. Use `--preprocess resize` for any new baseline.

Slot ids are arbitrary, so one slot has to be called background without using the
ground truth. Two rules were tried -- the slot covering the most patches, and the
slot covering the most border patches -- and they select the same slot, so the
numbers above hold either way.

**It assigns only 15.1% of patches to that background slot where the target is
87.5% background**, which is to say it splits the background across several
slots. That is what costs it foreground IoU and matched-object IoU, and it is
the behaviour expected of a model run on a render it was not trained on. Its
fg_ari stays high because that metric reads only target-foreground patches and
is insensitive to over-predicting foreground.
