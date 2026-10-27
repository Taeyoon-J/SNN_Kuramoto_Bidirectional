# PV2_0128 — why Multi-dSprites fails, and three hypotheses the data refuted

- **status** diagnosis complete; the remaining candidate is under test
- **parent** `PV2_0125`, `PV2_0127`
- **type** measurement only. No training, no model change. CPU except the last sweep.
- **question** Tetrominoes holds ~100% of its grid's ceiling and CLEVR 79%, both with the
  same core, losses and readout. Multi-dSprites holds **37%**. Why?

## 0. The metric cliff that was mistaken for a grid problem

`PV2_0125` recorded Multi-dSprites' grid-32 ceiling as 0.5619 and concluded the grid was
too coarse. That conclusion was wrong. Binning the per-image ceiling by the image's true
object count:

| objects in image | images | ceiling FG-ARI |
| ---: | ---: | ---: |
| 1 | **86** | **0.0000** |
| 2 | 77 | 0.7390 |
| 3 | 73 | 0.8019 |
| 4 | 64 | 0.8302 |
| all | 300 | 0.5619 |
| **≥2 only** | 214 | **0.7877** |

`(86 × 0 + 214 × 0.7877) / 300 = 0.5619` exactly. **29% of the split holds one object, and
FG-ARI on a single true cluster is all-or-nothing**: the score is 1 if the prediction is
also exactly one cluster and ~0 if it splits at all, because a chance-corrected pairwise
index has no partial credit when the truth has no pairs to disagree about. Grid 32's modal
vote pushes a few boundary cells of a lone sprite to background, the foreground pixels then
carry two labels, and the image scores 0.

So the grid costs **0.21** on multi-object images (1 − 0.7877), comparable to CLEVR's
0.8406. It was never the coarse cell. CLEVR has no single-object images at all, and
Tetrominoes always has three, which is why neither shows this.

**The decision to use grid 64 still stands, for a different reason.** Published
Multi-dSprites numbers are at pixel resolution, where the identity conversion scores 1.0 on
single-object images. Only a lossless grid escapes the cliff, so grid 64 is what makes our
number comparable — not because 2×2 cells lose too much.

**This belongs in `evaluation_contract.md`.** Of the three metrics only FG-ARI has this
cliff, and it has been reading every number in this project for months.

## 1. The failure is fragmentation, and only fragmentation

From the saved predictions, grid 32:

| dataset | true objects | predicted groups | groups in fg | **groups per object** | objects per group | fg cells/object |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tetrominoes | 3.00 | 10.72 | 3.12 | **1.08** | 1.05 | 80.7 |
| CLEVR | 6.61 | 16.17 | 10.27 | **1.72** | 1.10 | 18.9 |
| Multi-dSprites | 2.37 | 18.57 | 9.66 | **4.24** | 1.02 | 35.9 |

One true object is covered by **4.24 predicted groups** on Multi-dSprites. The reverse
error is absent everywhere: 1.02–1.10 true objects per predicted group, so **the model
never merges two objects.** It only splits, and the split order matches the
fraction-of-ceiling order exactly: 1.08 → ~100%, 1.72 → 79%, 4.24 → 37%.

Not a size problem: Multi-dSprites objects are 35.9 cells, nearly **twice** CLEVR's 18.9.
Larger objects, fragmented worse.

## 2. Three hypotheses, all refuted by measurement

### 2a. "Sprites overlap, so cells mix two foreground labels" — refuted

| | cells/object | cells holding ≥2 fg labels | disagreeing adjacent pairs that are fg–fg |
| --- | ---: | ---: | ---: |
| CLEVR g32 | 19.4 | **6.26%** | 7.0% |
| Multi-dSprites g32 | 37.8 | **3.05%** | 9.0% |
| Tetrominoes g32 | 83.6 | 0.09% | 1.0% |

Multi-dSprites has **more** cells per object and **fewer** mixed cells than CLEVR. Every
quantisation measure favours it.

### 2b. "colored_on_colored gives weak object/background contrast" — refuted

Measured in the γ the model actually consumes, as mean `1 − cosine`:

| dataset | within-object | bg–bg | object–bg | **object–OTHER object** | obj-bg / within |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tetrominoes | 0.0086 | 0.3212 | 1.3543 | 0.6686 | 157× |
| CLEVR | 0.0523 | 0.0321 | 0.3257 | 0.3209 | 6.2× |
| Multi-dSprites | 0.0712 | 0.0197 | 0.9253 | 0.9486 | **13.0×** |

Multi-dSprites' object/background separation is **13.0×**, stronger than CLEVR's 6.2× where
the model does fine. Per image it predicts nothing: r = −0.039, and the weakest-contrast
third fragments 4.23 against the strongest third's 4.14.

### 2c. "Objects are less internally uniform" — holds across datasets, fails within them

Within-object dissimilarity orders exactly like fragmentation — 0.0086 → 0.0523 → 0.0712
against 1.08 → 1.72 → 4.24. But that is three points, and inside each dataset:

| dataset | corr(object–bg, frag) | **corr(within-object, frag)** | frag, lowest vs highest within-object third |
| --- | ---: | ---: | --- |
| Tetrominoes | −0.143 | 0.183 | 1.04 → 1.16 |
| CLEVR | 0.030 | **0.405** | 1.57 → 1.89 |
| Multi-dSprites | −0.039 | **−0.081** | 4.35 → 4.07 |

It explains why CLEVR is worse than Tetrominoes (r = 0.405). It explains **nothing** inside
Multi-dSprites, where every image fragments 4.1–4.4 regardless of its own properties.

**That constancy is the finding.** A failure independent of every image property points at a
dataset-wide constant, not an image characteristic.

## 3. A result that matters beyond this test

`PROPOSAL_0.9_v3` named the missing row of `PV2_0075`'s table — object to a **different**
object — as the measurement that decides whether its de-phasing proposal is worth a run.
It is measured here. On CLEVR two different objects sit at **0.3209**, against
object-to-background **0.3257**: effectively identical. **The signal needed to separate two
objects exists in the graph's input, as strong as the figure/ground signal that already
works.** The proposal's falsifier passes.

## 4. What is being tested now

The readout settings were all fixed on CLEVR and carried here unchanged:
`--synchrony-threshold 0.05 --spatial-sigma 4.0 --min-group-size 2`. **`--spatial-sigma` is
in grid units**, and a grid-32 cell is 7.5×10 px on CLEVR but 2×2 px on Multi-dSprites, so
the same 4.0 is a 30×40 px neighbourhood there and **8×8 px** here. Nothing says the CLEVR
value transfers, and it is a dataset-wide constant — the shape §2c points at.

Sweeping `sigma ∈ {4, 8, 12}` against `threshold ∈ {0.05, 0.02, 0.01}` on 150 validation
images, no training. Both directions reduce fragmentation if this is the cause. If neither
moves 4.24, the readout is not it and the 224-parameter input encoder or the training is.

**Disclosure:** this tunes on validation, which the contract permits, but it ends the claim
in `BENCHMARK_EVALUATION.md` that nothing was tuned per dataset. Both the carried setting
and any tuned setting will be reported, labelled.

## 5. Corrections to earlier records

1. **`PV2_0125`, `PV2_0127`, `benchmark.md`, `BENCHMARK_EVALUATION.md`** all say the
   Multi-dSprites grid-32 ceiling of 0.5619 means the cells are too coarse. It means 29% of
   the images score zero by the definition of ARI. The grid choice is unchanged; the reason
   is.
2. **I wrote that the model "under-paints foreground at 0.0927".** The true foreground on
   that dataset is **0.0881**. It is slightly over, and nearly exact. I had compared it to
   Tetrominoes' 0.3004 instead of to its own ground truth.
3. **Three hypotheses of mine were refuted by measurement in this test alone** — overlap,
   contrast, and within-image uniformity. Recorded because the pattern matters: each was
   plausible and each was wrong, and the cheap measurement settled all three in an hour of
   CPU.

## 6. The readout sweep: not the cause

Nine combinations, 150 validation images, same checkpoint, no training:

| | FG-ARI | groups/image | pred_fg |
| --- | ---: | ---: | ---: |
| range over all 9 | 0.2292 – 0.2361 | 18.15 – 18.79 | 0.0903 – 0.0927 |

Tripling `spatial-sigma` (4 → 12) and dropping `synchrony-threshold` fivefold
(0.05 → 0.01) moves the group count by **0.4**. Both changes admit more edges and
should reduce fragmentation if the readout were binding. It is not.

That narrows the diagnosis usefully: the fragments are not produced by the thresholding,
they are already present in the spike synchrony structure the readout reads.

## 7. Grid 64: the grid was not the cause either

The replicated checkpoint at grid 64, where the conversion is the identity and the ceiling
is exactly 1.0000:

| | grid 32 | grid 64 | change |
| --- | ---: | ---: | ---: |
| ceiling | 0.5619 | **1.0000** | +0.4381 |
| FG-ARI | 0.2080 | 0.2817 | +0.0737 |
| fg-IoU | 0.5334 | 0.5480 | +0.0146 |
| obj-IoU | 0.4180 | 0.5327 | +0.1147 |
| **fraction of ceiling** | 37.0% | **28.2%** | **−8.8 points** |
| groups per image | 18.57 | **65.37** | ×3.5 |
| predicted foreground | 0.0927 | 0.1293 | true is 0.0880 |

**The ceiling rose by 0.44 and the score by 0.07.** The fraction of ceiling *fell*. Freeing
the grid did not free the model; it made the limit plainer, and the fragmentation followed
the grid — 4× the cells, 3.5× the groups.

So five candidates are now eliminated by measurement: the grid, the readout settings, sprite
overlap, object/background contrast, and within-image uniformity.

## 8. Where that leaves it

| dataset | grid | ceiling | ours | fraction of ceiling |
| --- | ---: | ---: | ---: | ---: |
| Tetrominoes | 32 | 0.9409 | 0.9478 | ~100% |
| CLEVR | 32 | 0.8406 | 0.6668 | 79% |
| Multi-dSprites | 64 | 1.0000 | 0.2817 | **28%** |

Same core, same losses, same readout. The remaining dataset-wide constants are the
224-parameter input encoder and the training run itself (seed 0 only, 737 min, collapse
never checked).

### CORRECTION — there is no peer unsupervised number

This section first read: *"the peer reports 0.6979 on this dataset, unsupervised, at grid
64 — 0.4162 above ours. With the grid eliminated, that is a model gap."* **That was wrong**,
and it was pushed in `6f4dd7d` before their answer arrived.

Their reply to `REQUEST_0008` states it plainly: the row labelled "기존 checkpoint" is a
**CLEVR mask-supervised SW0240 checkpoint evaluated zero-shot** on Multi-dSprites. Zero-shot
means no training on the target dataset; it does not mean the source model was trained
without masks. **Neither of their two rows is unsupervised.** The comparison I drew does not
exist, and our 0.2817 is the only unsupervised Multi-dSprites number either side holds.

| Multi-dSprites, pixel FG-ARI | supervision | |
| --- | --- | ---: |
| **ours**, grid 64, validation 300 | **unsupervised** | **0.2817** |
| peer zero-shot, TEST1000 | CLEVR mask-supervised source | 0.6833 |
| peer zero-shot, TEST360 | same | 0.6979 |
| peer adapted, TEST1000 | + target-mask supervision | 0.7701 |
| CAE / CAE++ / CtCAE / RF / SynCx | unsupervised | .05 / .51 / .56 / .84 / .82 |

Against the unsupervised column our 0.2817 clears only CAE's 0.05. The same correction
applies to Tetrominoes: their 0.6164 is also supervised-source transfer, so our 0.9478 beats
it but the two rows are not like for like and the table must say so.

### What their answer did settle

- **The dataset variant is not the cause.** They used the same `colored_on_colored` through
  our own files. Eliminated.
- **Their readout is a different algorithm**, not a different setting of ours:

  | | ours | peer |
  | --- | ---: | ---: |
  | `synchrony_threshold` | 0.05 | **0.50** |
  | `min_group_size` | 2 | **32** |
  | graph top-k | 32 | **512** |
  | merge stage | none | `foreground_induced_merge`: signed pair energy, positive-gain agglomeration inside four-connected foreground regions |
  | encoder | 224 params, reconstruction | **29,128** params, dilated 2/4/8 context CNN, mask-supervised on CLEVR |

- **My sweep ran the wrong way.** I varied the threshold 0.05 → 0.01, looser, expecting
  fragments to merge. Theirs is 0.50 — ten times my highest value — with `min_group_size` 32
  against my fixed 2. The regime I swept was never the one that could have helped.
- **They fragment too**: 14.44 groups per image on TEST1000 against our 65.37, and their
  seed 2 alone produces 29.7. Not the 2–3 a clean segmentation would give.
- **They also carry 86 single-object images** in TEST360 and include them under the
  unchanged metric, confirming the cliff is not avoided by grid 64 alone — only by not
  fragmenting.

A corrected sweep is running: `threshold {0.05, 0.10, 0.30, 0.50} × min_group_size {2, 8,
32}` at grid 64. Their merge stage is **excluded** — it is a code port, and `PV2_0107b` lost
matched-object IoU with a different merge rule, so it needs its own test rather than being
folded into a configuration sweep.

## 9. Infrastructure fixed to get here

`_spike_affinity` built the per-channel affinity as a list of D `[B, N, N]` tensors and then
a (D+1)th via `torch.stack`, reduced immediately by `prod(dim=0)`. At grid 64 that asked the
**host** allocator for 80.53 GB with 161 GB needed in total against 102 GB free — a CPU
failure, not a GPU one, since the eval loop moves spikes to CPU. The product now accumulates
in place and `--affinity-chunk` bounds the image axis. Verified inert: Tetrominoes grid 32
returned 0.9362 / 0.6914 / 0.7817 / 0.3004, identical to four decimals.

At grid 32 the same code was already using 33.6 GB on CLEVR's 1000 images. It never failed,
so it was never noticed.
