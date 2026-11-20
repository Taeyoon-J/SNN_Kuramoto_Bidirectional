# PV2_0112 — the coverage stage fails, and the diagnostic closes the route

Status: **completed — FAILED its pre-registered gate**
Parent: `PV2_0106` (oracle decomposition), `REVIEW_0007`. Replaces withdrawn `PV2_0111`.
Control: `PV2_0104`'s three checkpoints `GC_w0`, `RSC_freeze`, `FRZ_s2`, coverage off.
Contract v2, grid 32, validation 300. Regression row passed first: `0.7239 0.7251 0.4951`.

## Result

| three-seed mean | fg_ari | fg_iou | obj_iou | pred_fg | gate |
| --- | ---: | ---: | ---: | ---: | --- |
| control | 0.7110 ± 0.0512 | 0.7130 ± 0.0349 | 0.6588 ± 0.0372 | 0.1369 | — |
| q 0.95 | 0.7109 ± 0.0509 | **0.6375** | 0.6607 ± 0.0364 | 0.1540 | FAIL |
| q 0.90 | 0.7129 ± 0.0496 | **0.5412** | 0.6624 ± 0.0352 | 0.1853 | FAIL |
| q 0.85 | 0.7143 ± 0.0484 | **0.4561** | 0.6636 ± 0.0339 | 0.2249 | FAIL |
| q 0.80 | 0.7154 ± 0.0476 | **0.3863** | 0.6643 ± 0.0333 | 0.2691 | FAIL |

fg_ari moves +0.0044 at best. obj_iou moves +0.0055 at best. **Foreground IoU collapses**,
−0.0755 to −0.3267, and `pred_fg` breaches the 0.20 calibration gate from q 0.85 down.

**The registered falsifier fired, in the opposite direction to the one anticipated.** It said
*fg_iou up while fg_ari or obj_iou falls = `SW_0207` reproduced*. What happened is the
reverse: fg_iou **falls** while the other two barely move. This is not the peer's failure
mode; it is a different one.

## Why — the patches it added were not foreground

Backing the intersection out of the two IoUs at each quantile:

| q | patches added per image | of which true foreground | hit rate |
| ---: | ---: | ---: | ---: |
| 0.95 | 17.5 | −0.3 | −1.7% |
| 0.90 | 49.6 | 0.2 | 0.4% |
| 0.85 | 90.1 | 1.0 | 1.1% |
| 0.80 | 135.4 | 1.4 | 1.0% |

Essentially **none**. The stage was adding background, which is why foreground IoU tracked
`pred_fg` downward almost exactly.

## The diagnostic that closes the route

Oracle diagnostic on seed 0, 60 images, using the production readout's own saved predictions:

- The readout already claims **94.6%** of all true-foreground patches.
- What is left unclaimed: **385** true-foreground patches against **52,451** background ones.
- **The base rate of foreground among unclaimed patches is 0.73%.**
- Activity does carry signal — the top 5% by activity is 1.97% foreground, **2.7× the base
  rate** — but 2.7× on a 0.73% base is still 2% precision.

**No admission threshold on any scalar signal can work at that base rate.** To add 18 patches
an image and have most of them be right would need roughly a hundredfold lift, not threefold.
Swapping activity for synchrony or membrane does not change the arithmetic unless the
replacement is two orders of magnitude more discriminative, which nothing here suggests.

## Reconciling this with `PV2_0106`, which said coverage was worth +0.1001

Both are right; they are different averages, and the difference is the finding.

- `PV2_0106` measures coverage **per object**: 0.8726, i.e. the average object has 12.7% of
  itself called background, and 2.1% of objects are claimed by nothing.
- This diagnostic measures it **per patch**, pooled: 94.6% claimed.

Large objects dominate the patch count and are covered well; small objects are covered badly
and drag the per-object mean down. So the missing foreground is **not spread thinly over the
image, it is concentrated in small objects and in object edges** — which is exactly the
population an image-wide activity threshold cannot isolate, because those patches sit in a
sea of background that looks identical to them by activity.

## What this closes, and what is left

Both readout-side routes to coverage now have evidence against them:

1. **Widening existing groups** — `SW_0207` on the peer branch: foreground IoU +0.171, FG-ARI
   −0.054, instances fragment, closed.
2. **Admitting unclaimed patches by threshold** — this test: 0.73% base rate, ~2% precision at
   the top, closed.

Taken with `PV2_0107b` and `SW_0234` — where even a **GT-prescribed** merge loses
matched-object IoU — the readout-only route to 0.9 is in serious doubt. `PV2_0106`'s oracle
ceiling of 0.9329 stands as a ceiling; what this test establishes is that **the information
needed to reach it is not recoverable from the spikes by a readout stage.** If small objects
are to be covered, the model has to fire on them, which is a core-and-training change, not a
readout one.

## Limitations

- Validation only, 300 images; the diagnostic is 60 images on seed 0 alone. The 94.6% figure
  is seed 0's, the seed with the highest `pred_fg` (0.1512), so the other two seeds likely
  claim less. The base-rate argument does not depend on the exact figure.
- `--coverage-synchrony` was implemented but not swept. It constrains *linking*, not
  *admission*, so it cannot fix a 0.73% base rate; a sweep would be for completeness, not for
  rescue.
- The diagnostic reads the ground truth and is an oracle result throughout.

## Next

The honest next question is not another readout stage. It is whether the model can be made to
fire on small objects at all — which is a loss or a core change, and is the first thing in
this project that the readout cannot reach.
