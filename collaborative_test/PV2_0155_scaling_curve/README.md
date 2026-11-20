# PV2_0155 — scaling curve, 10,000 → 18,000 CLEVR images in steps of 1,000

**Status** completed
**Parent** `PV2_0137` (253 core at 10,000 images) · **Comparison** the same core, continued
**Data** clevr_with_masks, split manifest v1 · **Evaluation contract** v2 · **Base commit** `4b111cf`
**Seed** 0 only (core seeds 1 and 2 declined by the user, 2026-11-09)

## Hypothesis

More unseen in-domain CLEVR images keep improving the three metrics, and the point where each
stops improving is the same point. Changed variable: the number of unique training images.
Success criterion: a visible turn, or nine points with none.

## Result — the three metrics do not saturate together

320 reserved CLEVR images, core-only `regions_253`, no minimum-region-size rule.

| unique images | updates | fg_ari | fg_iou | obj_iou |
|---|---|---|---|---|
| 10,000 | 6,250 | .8421 | .7545 | .6410 |
| 11,000 | 6,880 | .8426 | .7539 | .6427 |
| 12,000 | 7,510 | .8449 | .7607 | .6461 |
| 13,000 | 8,140 | .8460 | .7637 | .6470 |
| 14,000 | 8,770 | .8469 | .7652 | .6524 |
| 15,000 | 9,400 | .8459 | .7702 | .6534 |
| 16,000 | 10,030 | .8456 | .7730 | .6538 |
| **17,000** | 10,660 | **.8470** | .7747 | .6572 |
| 18,000 | 11,290 | .8461 | **.7755** | **.6579** |

**FG-ARI gains +.0028 over the first two steps and then sits inside a band of .0021 for the
remaining six, with no trend.** Foreground IoU rises monotonically (+.0210 over the range) and
matched-object IoU likewise (+.0169); neither has turned at 18,000.

Mean predicted regions hold at 7.7–8.0 throughout against 6.36 true objects, so the extra data
sharpens boundaries without changing how much the readout splits.

## This withdraws the reading in BENCHMARK_PLAN_v6 §5

v6 was written at six points and said *"FG-ARI peaks at 14,000 and turns down 0.0010"*. With
nine points that is wrong: **17,000 scores .8470, above the 14,000 value**. There is no turn —
the curve is flat inside one seed's noise from 12,000 on. The *conclusion* v6 drew from it —
the three metrics saturate at different rates — survives and is in fact strengthened, because
the two IoU curves are still climbing at 18,000 while FG-ARI has been flat for six steps.

## Commands

```bash
# frontier, GPU 1
/work/USERS/tkim1/scale_curve2.sh            # 8 stages, each +630 updates / +1,000 unique images
# per-stage scoring
$PY score_cache.py --cache ev_scale/clevr_fixed320_u<updates> --dataset clevr --split fixed320
```

Each stage runs in its own run directory: `engine.py:90-104` refuses to change `train_ids`
inside one run without a `runtime_migration.json`, and `engine.py:104` returns immediately when
`complete.json` exists, which silently no-opped stages 2–8 on the first attempt (`PV2_0155`
first run, discarded). A guard now aborts unless updates advance by exactly 630.

## Interpretation and limitation

More in-domain data stops improving **how patches are grouped** at around 12,000 images while
it is still improving **where the boundaries fall**. A benchmark reporting "scaling" must say
which metric it means.

One seed. The FG-ARI band is read as flat, not as converged; the two IoU curves have not been
run far enough to find their own ceiling. Nothing here was selected on the reserved split.

## Next

If anyone extends this, the question is where Foreground IoU turns, not FG-ARI — and the answer
costs about 7 h per additional 1,000 images on one GPU.
