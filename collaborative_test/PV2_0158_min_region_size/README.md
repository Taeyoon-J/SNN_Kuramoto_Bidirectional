# PV2_0158 — single-patch regions inflate the object count; a scale-free size rule removes them

**Status** completed
**Parent** `PV2_0137` (253 core) · **Comparison** the same core and the same masks, post-processed
**Data** clevr_with_masks / tetrominoes / multi_dsprites, split manifest v1 · **Contract** v2
**Base commit** `4b111cf` · **Seed** 0 · **No training; cached `regions_253` only, no GPU**

## Origin

The user looked at the predicted mask for CLEVR image 1355 and observed that several regions
were a single patch, and proposed a loss that refuses to count a one-patch region as an object.
The observation was right and is quantified below. The proposed *mechanism* is not available:
the partition is `Q ≥ 0.5` → connected components → component floor → greedy merge, all
non-differentiable, so "a one-patch region is not an object" has no gradient path back to the
spikes. It is a readout rule, and the contract already permits thresholds fixed on validation.

## 1. The observation, measured

Predicted regions on each dataset's validation split, core-only, before any rule:

| | 1-patch regions | median region size | regions/image |
|---|---|---|---|
| CLEVR | **22.8%** (552 / 2,418) | 46 patches | 7.56 |
| Tetrominoes | **27.0%** (937 / 3,474) | 6 patches | 21.71 |
| Multi-dSprites | **56.2%** (664 / 1,181) | **1 patch** | 7.38 |

On Multi-dSprites the *median* predicted region is one patch. They arise after the synchrony
grouping: `foreground_induced_merge` splits a spatially scattered synchronous component into
spatial pieces, and the small pieces survive because the 32-patch floor is applied before that
split, not after.

## 2. The rule

Discard any predicted region smaller than a fraction `f` of **that image's predicted
foreground**, reassigning it to background. Scale-free (adapts to object count and object size),
reads no ground truth, one scalar.

`f` is fixed on validation as the largest value whose FG-ARI loss is ≤ 0.005 and whose
matched-object IoU loss is ≤ 0.01. Stated before the reserved split was touched.

Two settings are reported because they are different claims:
* **(A)** `f` chosen on **CLEVR validation only** and transferred unchanged → `f = 0.01`
* **(B)** `f` chosen on **each dataset's own** validation → 0.01 / 0.10 / 0.05

## 3. Reserved split, 320 images per dataset

| | fg_ari | fg_iou | obj_iou | count (truth) | exact | MAE |
|---|---|---|---|---|---|---|
| CLEVR, no rule | .8421 | .7545 | .6410 | 8.04 (6.36) | .175 | 2.88 |
| CLEVR (A)=(B) f=.01 | .8408 | .7528 | .6387 | **5.15** | **.250** | **1.33** |
| Tetrominoes, no rule | .8580 | .7885 | .8173 | 21.07 (3.00) | .000 | 18.07 |
| Tetrominoes (A) f=.01 | .8568 | **.7950** | .8170 | 5.28 | .094 | 2.29 |
| Tetrominoes (B) f=.10 | .8550 | **.8066** | .8167 | **3.02** | **.869** | **0.14** |
| Multi-dSprites, no rule | .6974 | .8177 | .7628 | 6.62 (2.49) | .281 | 4.56 |
| Multi-dSprites (A) f=.01 | .6966 | **.8190** | .7625 | 2.64 | .531 | 0.74 |
| Multi-dSprites (B) f=.05 | .6960 | .8176 | .7599 | **2.17** | **.647** | 0.46 |

**(A) costs at most 0.0024 on any metric cell and improves Foreground IoU on both transfer
datasets**, while cutting count MAE by 1.6×, 7.9× and 6.2×. (B) is better everywhere but reads
target-domain labels, so (A) is the headline and (B) the upper bound.

The narrowest cell in the whole comparison — Tetrominoes Foreground IoU — widens from **+0.0345
to +0.0410** under (A) and to +0.0526 under (B).

## 4. Fairness check: the same rule offered to every baseline

Required before claiming any margin. Reserved split, `f` swept 0 → 0.10.

* **Tetrominoes**: nothing to gain. AKOrN (conv) .7540 → .7544 at best; AKOrN (attn) flat;
  both lose count accuracy as `f` grows.
* **CLEVR**: one real gain — **AKOrN (conv) Foreground IoU .4219 → .4267** at `f = 0.01`
  (.4376 at 0.02, with FG-ARI and obj-IoU falling). The best-baseline-per-cell row uses the
  better of the two. CtCAE and CAE++ are flat and their counts get *worse*.
* **Multi-dSprites**: all three strongest baselines move by ≤ 0.0003 at `f = 0.01`.

The baselines have nothing to clean up because fixed-`k` KMeans cannot produce a one-patch
region in the first place. **The rule is not a private advantage**, and the composite row is
unchanged except the one CLEVR cell.

## 5. A benchmark-design finding that came out of the fairness check

Every fixed-`k` baseline scores **100% exact object count on Tetrominoes** — CtCAE included, at
FG-ARI 0.0799. The reason is that Tetrominoes contains exactly three objects in every image, so
"always answer three" is exactly right without grouping anything. **Object-count accuracy is
degenerate wherever the true count is constant.** On Multi-dSprites, where it varies, those same
baselines score 0% exact against our 53.1%. `BENCHMARK_PLAN_v6 §1` made count a required metric
without this caveat; v7 fixes it.

## Commands

```bash
# frontier, CPU only
$PY minsize.py      --cache ev/<dataset>_validation --dataset <ds> --split validation
$PY minsize_rel.py  --cache ev/<dataset>_validation --dataset <ds> --split validation   # maxrel/fgrel sweep
$PY final_minsize.py --cache ev/<dataset>_<split>  --dataset <ds> --split <split> --fa 0.01 --fb <f>
$PY fair_minsize.py --dataset <ds> --split <split> --ids-from <npz> --models "<name>=<npz>,..."
```
Scripts in `evaluation_howto/`.

## Interpretation, limitation, next

The count error was mostly fragments, not genuine over-segmentation: removing regions that
carry ≤1% of the foreground recovers most of the count without touching the partition. The
`fgrel` form generalises; the absolute-`k` form needs a different constant per dataset (4 / 48 /
8) and was rejected for that reason. A `maxrel` form (fraction of the largest region) was also
tried and generalises worse.

Limitations: one seed; (A)'s `f` was chosen on CLEVR validation, which is in-domain for CLEVR
and therefore the weakest of the three cases; the fairness sweep covered the strongest baselines
per cell, not all eight on all three datasets.

Next: the cause is the component floor being applied before the spatial split. Applying it
after would remove the fragments at source and make the rule unnecessary — a core change,
analysed but not made.
