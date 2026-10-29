# Multi-dSprites, the peer gap, and whether supervision is a way out

Five linked questions, answered from measurement. Where a question is not settled, the
section says so rather than filling the gap with a story. **Nine of my own hypotheses were
refuted during this work**; the refutations are kept because they are most of what was
learned.

---

## Q1. Why is Multi-dSprites low when Tetrominoes and CLEVR are high?

Same core, same losses, same readout, same readout settings:

| dataset | grid | ceiling | ours | **fraction of ceiling** |
| --- | ---: | ---: | ---: | ---: |
| Tetrominoes | 32 | 0.9409 | 0.9478 | **~100%** |
| CLEVR | 32 | 0.8406 | 0.6668 | **79%** |
| Multi-dSprites | 64 | 1.0000 | 0.2817 | **28%** |

### The failure is fragmentation, and only fragmentation

| dataset | true objects | **predicted groups per true object** | true objects per predicted group |
| --- | ---: | ---: | ---: |
| Tetrominoes | 3.00 | **1.08** | 1.05 |
| CLEVR | 6.61 | **1.72** | 1.10 |
| Multi-dSprites | 2.37 | **4.24** | 1.02 |

The reverse error is absent everywhere — we never merge two objects into one. We only
split, and the split order matches the fraction-of-ceiling order exactly. At grid 64 it
gets worse in step with the grid: **65.4 groups per image for 2.37 objects**.

It is not object size: Multi-dSprites objects are 35.9 cells, nearly twice CLEVR's 18.9.

### Nine candidates eliminated by measurement

| candidate | how it died |
| --- | --- |
| the patch grid | grid 32 → 64 lifted the ceiling 0.5619 → 1.0000; the score moved 0.2080 → 0.2817 and the **fraction fell** 37.0% → 28.2% |
| readout settings | 12 combinations of threshold × min-group-size. **The default was the best of all 12.** Peer's own setting (0.50 / 32) came **last** at 0.1821 |
| sprite overlap mixing cells | Multi-dSprites has **fewer** cells holding two foreground labels than CLEVR, 3.05% vs 6.26% |
| weak object/background contrast | in γ it is **13.0×** within-object against CLEVR's 6.2× — stronger. Per-image correlation with the failure: **−0.039** |
| within-image uniformity | orders the three datasets correctly but explains nothing **inside** Multi-dSprites (r = −0.081); every image fragments 4.1–4.4 regardless of its own properties |
| the dataset variant | the peer used the same `colored_on_colored` through our own files |
| copying the peer's readout settings | that is the 0.50 / 32 row above, the worst of the twelve |
| porting their merge stage | ran, preserved foreground support as promised, and produced **151 groups instead of 65** |
| encoder capacity, unsupervised | the identical 29,128-parameter architecture trained by reconstruction scored **0.2193**, below our 224-parameter baseline's 0.2658 |

### What survives, and what it rests on

Two things, and the second is better supported than the first.

**A metric cliff accounts for part of it.** 86 of the 300 validation images hold exactly one
object. FG-ARI on a single true cluster has no partial credit: 1.0 if the prediction is also
one group, ≈0 if it splits at all. A model that fragments scores zero on 29% of the split by
definition. This is why the grid-32 ceiling read 0.5619 while the ≥2-object ceiling is
0.7877 — and why we first mistook a metric property for the grid being too coarse.

**The γ representation is the lever.** Within-object γ dissimilarity orders exactly like
fragmentation across datasets — 0.0086 (Tetro) → 0.0523 (CLEVR) → **0.0712** (Multi-dSprites)
against 1.08 → 1.72 → 4.24 — and swapping the encoder moves the score by **+0.19**, which no
readout change could do. It does **not** predict anything within Multi-dSprites, which is
consistent with it being a property of how the encoder represents that dataset rather than
of particular images.

**Honest limit:** three points across three datasets is thin, and "the encoder is the lever"
is supported by the swap, not by a mechanism we can point at in the equations.

---

## Q2. Why is the peer's Multi-dSprites so much higher than ours?

| Multi-dSprites, pixel FG-ARI | |
| --- | ---: |
| ours, grid 64 | 0.2817 |
| peer zero-shot, TEST1000 | **0.6833** |
| peer adapted, TEST1000 | 0.7701 |

### Roughly half of it is the encoder

Their encoder, weights and all, dropped into **our** core and **our** readout, on 40 images:

| encoder | fg_ari | fg_iou | obj_iou | pred_fg | groups |
| --- | ---: | ---: | ---: | ---: | ---: |
| ours, 224 params | 0.2658 | 0.5618 | 0.5442 | 0.1273 | 63.7 |
| **theirs, 29,128 params** | **0.4574** | 0.5381 | **0.7601** | 0.1572 | **41.6** |

So: 0.2658 → 0.4574 from the encoder alone, and 0.4574 → 0.6833 from everything else of
theirs — their trained core, their top-k 512 graph, their merge stage. Splits differ (our 40
validation against their 1000 test), so this is a decomposition in rough proportions, not to
three decimals.

### And the encoder's advantage is the objective, not the architecture

The same 29,128-parameter architecture, trained by us with **reconstruction** instead of
masks, scored **0.2193** — worse than our 224-parameter baseline. It called **43.5%** of the
image foreground against a true 8.8%, and fg-IoU collapsed to 0.1446.

Reconstruction learns what reproduces pixels: texture and brightness. The dilated 2/4/8
receptive field then spreads that signal further, which is why the bigger architecture made
it worse rather than better. **The capacity was never the point. The supervision was.**

---

## Q3. If supervision is the reason, does supervising ours help?

The peer-encoder swap already answers it partly: our pipeline, with a mask-supervised
encoder in front, goes **0.2658 → 0.4574**. That is our core and our readout.

Measured directly: the same context architecture trained by us on **Multi-dSprites masks**,
pulling same-object patch pairs together and pushing different-object pairs apart, TRAIN ids
[0, 6000) only. 40-image probe, our core, our readout:

| encoder | objective | fg_ari | fg-IoU | obj-IoU | pred_fg | groups |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ours, 224 | reconstruction | 0.2658 | **0.5618** | 0.5442 | 0.1273 | 63.7 |
| ctx 29k | reconstruction | 0.2193 | 0.1446 | 0.3752 | 0.4352 | 150.9 |
| **ctx 29k** | **our masks** | **0.4252** | **0.2069** | 0.7067 | **0.3719** | 92.5 |
| peer 29k | their masks | 0.4574 | 0.5381 | 0.7601 | 0.1572 | 41.6 |

**Supervision transfers: +0.1594 of FG-ARI, close to the peer's +0.1916.** The information
is usable with our own data and our own pipeline.

**But read the other two columns.** Predicted foreground is **0.3719** against a true 0.0880
— 37% of the image called foreground — and fg-IoU collapses from 0.5618 to **0.2069**.
FG-ARI is scored over true-foreground cells only and **cannot see over-painting**, so a model
that floods the image is rewarded. This is not a better segmentation; it is the metric being
exploited.

The fault is in my objective, not in supervision. I pulled same-object pairs together and
pushed different-object pairs apart **restricted to foreground pairs**, so nothing in the
loss had any reason to keep background distinct from foreground. The peer's teacher does
hold that line: the same supervision signal, with fg-IoU preserved at 0.5381 and 41.6 groups
against our 92.5.

**So the answer is two-part.** Mask supervision does transfer to our pipeline, and it is
worth about the same +0.16–0.19 either way. How it is applied matters as much as whether it
is applied, and our crude version buys FG-ARI by wrecking the other two metrics. A fair
supervised comparison would need the objective rebuilt with a figure/ground term.

---

## Q4. If supervised scores are high, can they go in the benchmark?

**No — not in the table we are building.** This is the part that decides whether Q3 is worth
acting on at all.

| comparison family | their supervision | can a mask-supervised model sit there? |
| --- | --- | --- |
| CAE, CAE++, CtCAE | unsupervised | **no** |
| Rotating Features, SynCx | unsupervised | **no** |
| Slot Attention | unsupervised, reconstruction | **no** |
| supervised segmentation (Mask R-CNN, SAM, …) | supervised | yes — and we lose badly |

Every number in our table comes from unsupervised object discovery. Putting a mask-supervised
model beside them is not a caveat to disclose, it is a different task. And the supervised
league has baselines far stronger than anything here.

The peer reached the same conclusion about their own rows, unprompted:

> *"neither our zero-shot row nor our mask-supervised adaptation row belongs in a
> like-for-like unsupervised table alongside CAE, CAE++, CtCAE, Rotating Features, and
> SynCx."*

Their "기존 checkpoint" row is **also** supervised — it is a CLEVR mask-supervised source
evaluated zero-shot. Zero-shot means no training on the target dataset, not that the source
saw no masks.

---

## Q5. So what actually changes in the benchmark?

**Nothing, if we stay unsupervised — and that is the recommendation.**

All three rows are now on the TEST split:

| dataset | split | seeds | grid | ceiling | ours, unsupervised |
| --- | --- | ---: | ---: | ---: | ---: |
| Tetrominoes | TEST 1000 | 3 | 32 | 0.9409 | **0.9282 ± 0.0170** |
| CLEVR | TEST 1000 | 3 | 32 | 0.8406 | 0.6668 ± 0.0506 |
| Multi-dSprites | TEST 300 | 1 | 64 | 1.0000 | 0.3061 |

### The comparison family was the wrong one

`BENCHMARK_PLAN.md` chose to compare against the synchrony-binding line on the grounds that
it is what this architecture competes with. A literature check says that choice does not
survive review:

| | Tetrominoes | Multi-dSprites | CLEVR6 |
| --- | ---: | ---: | ---: |
| **Slot Attention** | **99.5** | **91.3** | **98.8** |
| IODINE | 99.2 | 76.7 | 98.8 |
| MONet | — | 90.4 | 96.2 |
| SynCx | 89 | 82 | 59 |
| CtCAE | 84 | 56 | 54 |
| Rotating Features | 42 | 84 | 65 |
| **ours** | **92.8** | 30.6 | 66.7 |

**All of these are unsupervised.** We beat the synchrony family on Tetrominoes and lose to
the slot family by a wide margin, on a dataset where the conditions match exactly — 35×35,
three objects.

CLEVR is not comparable as listed: the published figures are **CLEVR6**, at most six objects
at 128×128, while ours is full CLEVR, ten objects at 240×320. Our own re-run of Slot
Attention under our conditions scored 0.8102, not 0.988, which is the size of that gap.

Three honest ways forward, and this is a decision rather than a measurement:

1. put the slot family in the table and report the loss
2. restrict the claim explicitly to synchrony/phase-based binding and justify the restriction
3. move to a setting where the slot family is weak — harder CLEVR, CLEVRTex

What is not available is quietly keeping the current family. A reviewer asks about Slot
Attention first.

Going supervised would cost all of that: the Tetrominoes win is only meaningful against
unsupervised baselines, and we would be entering a league where our numbers are weak.

### What is worth doing instead

1. **Publish the unsupervised table as it stands**, Multi-dSprites loss included. Excluding a
   dataset we lose on is cherry-picking, and the loss is explained rather than hidden.
2. **Report the supervised result as an ablation**, clearly separated, answering "what does
   this architecture do when the features are supervised?" That is a legitimate and
   interesting row — it just is not a comparison.
3. **The remaining unsupervised route is the objective — and it is open.** Reconstruction
   learns texture; masks learn boundaries. `PV2_0129` added a total-variation penalty, which
   says "flat inside a region, free to jump at an edge" with no labels, and it reached
   **0.4617** on Multi-dSprites against our mask-supervised 0.4252 and the peer's
   mask-supervised 0.4574. 40 images and one seed, so exploratory — but the premise that the
   unsupervised route was exhausted is withdrawn.

---

## Appendix: measurements this rests on

| measurement | where |
| --- | --- |
| per-dataset ceilings, per grid | `benchmark/ceil.py`, `PV2_0125` |
| fragmentation decomposition | `PV2_0128` |
| 12-combination readout sweep | `sweep2_readout.sh` |
| γ pair dissimilarity, four pair types | `PV2_0128` §2b |
| merge stage port and its 5/5 verification | `ported/`, `PV2_0128` §9 |
| affinity Q distribution, ours vs peer encoder | `qdist.py` |
| peer encoder swapped into our pipeline | diagnostic B |
| reconstruction-trained context encoder | `train_ctx.py` |
| mask-trained context encoder | `train_sup.py` |
