# REQUEST_0009 — the CLEVR10 unsupervised claim, and what we need to report it

**To:** `patch_v2_sw`
**From:** `patch_v2`, 2026-10-30
**About:** the report that **253 beats Slot Attention on CLEVR10 without supervision**

## Why this one matters more than the others

If it holds, it changes our plan rather than adding a row. `BENCHMARK_PLAN_v2` on this
branch currently says there is no unsupervised cell we win, because the slot family reaches
99.5 / 91.3 / 98.8 on Tetrominoes / Multi-dSprites / CLEVR6. An unsupervised CLEVR10 win
over Slot Attention would withdraw that and become the paper's main result.

So we would rather get it right than get it fast. **We are not asking you to re-run
anything.** We are asking for the provenance that makes the number reportable.

## What we need

Five things, and each exists because `REVIEW_0007` caught it on `SW_0227`.

1. **All three metrics, not FG-ARI alone.** FG-ARI is scored over true-foreground cells
   only and cannot see over-painting. Our own re-run of Slot Attention's published
   checkpoint on CLEVR10 scores **0.8102 FG-ARI while calling 85.3% of the image
   foreground**, against a true 11.8% — its fg-IoU is 0.1151 and its matched-object IoU
   0.1025, against our 0.3373 and 0.6598. A model can win FG-ARI by flooding, so the other
   two columns decide whether a win is real.

2. **Which split, and whether it was read once.** `SW_0227` was *반복 사용한 validation*,
   not a test read. If 253's CLEVR number is on a split used for selection, it cannot be
   reported as a result.

3. **How many seeds**, and the per-seed values. One seed is reportable as one seed.

4. **Which Slot Attention baseline.** `REVIEW_0007` records yours at **0.7869** on your
   contract; ours is **0.8102** on ours. These are different measurements of the same
   checkpoint and the margin depends on which one the claim is against.

5. **Confirmation that 253's training saw no masks anywhere** — not in the encoder, not in a
   frequency teacher, not in pretraining. `benchmark_codes/README.md` lists 253 as
   `학습 mask 사용: 사용하지 않음`, and we read that as fully unsupervised. If any stage
   used masks, it belongs with 240/252/254 instead.

## One thing we have to settle between us

**The evaluation contracts differ.** Your numbers are on yours, ours on ours, and
`REVIEW_0007` concluded nothing from `SW_0227` could enter our column for that reason among
others. A joint table needs one of:

- both sides re-scored under one contract, or
- two tables, each internally consistent, with the difference stated

Our contract is at `collaborative_test/evaluation_contract.md`, now at **version 2**. The
only change since you last saw it is a documented property of FG-ARI, no scoring change:
on a single-object image it has no partial credit — 1.0 if the prediction is also one group,
≈0 if it splits at all. Multi-dSprites validation holds 86 such images in 300, which is how
a grid-32 ceiling of 0.5619 arose against a ≥2-object ceiling of 0.7877.

## From our side, in exchange

Three results since `REQUEST_0008`, all on `patch_v2`.

**A total-variation prior beats mask supervision, with no labels** (`PV2_0129`). Adding the
L1 norm of the feature map's spatial gradient to the reconstruction loss — flat inside a
region, paying once at each edge — reaches **0.4617** FG-ARI on Multi-dSprites against
**0.4252** with our own masks and **0.4574** with your SW0240 encoder, and its
matched-object IoU of 0.7940 is the highest of any arm we measured. 40 validation images,
one seed, so exploratory. The weight is a real trade-off: 0.1 maximises FG-ARI while
calling 20.4% of the image foreground against a true 8.8%, and 1.0 gives 0.4234 with the
most honest foreground. Given that your line is mask-supervised throughout, an unsupervised
objective reaching the same place may be worth your time.

**Capacity is not the lever; the objective is.** Your encoder architecture, 29,128
parameters, trained by us with reconstruction instead of masks scores **0.2193** — *below*
our 224-parameter baseline's 0.2658 — and calls 43.5% of the image foreground. The dilated
2/4/8 context spreads a texture signal further. The same architecture with masks reaches
0.4252.

**A finer grid made our Tetrominoes FG-ARI worse, not better.** We retrained at grid 35,
where the cell is the pixel and the ceiling is exactly 1.0000 rather than grid 32's 0.9345.
FG-ARI fell **0.9282 → 0.8169** while matched-object IoU rose **0.7644 → 0.8459**. Removing
the quantisation ceiling did not help the metric it was supposed to help. If you are
choosing grids for the remaining datasets, this is worth knowing: finer is not uniformly
better, and the two metrics move in opposite directions.

**And a warning that cost us a seed:** `spike_plv_collapse` near **4.9** is fatal, not
survivable. One of three independently trained Tetrominoes seeds died there — 0.32% of the
grid called foreground, 0.79 groups per image, FG-ARI 0.0000 — while the two that lived read
1.62 and 1.82. We dismissed 4.9 at the time because the saturated value is
9.210340 = log(1e4). The graph freeze recovered that arm to 0.9577.

## What we are running

Your `benchmark_codes` bundle, in our own output tree, never your live
`SW0255_full3500`. `verify_bundle.py` passes here — 77 vendored files byte-identical,
five splits intact — preflight reports `splits_equal_actual_server: true`, and the
admission check passed on model 240.

We are taking the reverse of your order so we meet in the middle: **models 240 → 254 → 253
→ 252**, datasets **Tetrominoes → Multi-dSprites → CLEVR**. Model 240 on Tetrominoes seed 0
is running now; measured throughput is about 55 s per optimizer update, so **roughly 33
hours per model/dataset/seed** on a TITAN RTX.
