# PV2_0138 — 261 trained natively on Tetrominoes and Multi-dSprites, 10,000 × 10

**Status** running (launched 2026-11-05)
**Parent** PV2_0134 (transfer rows), PV2_0136 (the label-free recipe)
**Seeds** 0 only, by decision. One run per dataset, the two in parallel.

## What "261" means here

Not the peer's 261. The peer's is a **native16 core state lifted to grid 64** by `parent_index`
(`axis = arange(64)//4`, with omega/kappa/tau tied in 4×4 blocks and `direction_learner` built
by `index_select` on both axes of a 256×256). That is a resolution conversion of an existing
checkpoint.

Here 261 is defined as **the core trained from scratch, natively at grid 64, by 253's
unsupervised recipe — no masks, no teacher — plus the SW0260 readout.** No lifting.

There is no `--model 261`: the trainable variants are `{252, 253, 254, 240}`, because 261 is not
a core variant at all, it is 253's core plus a readout. So the core is trained with
`--model 253` and the 261 readout goes on afterwards.

## Why this run exists

Every Tetrominoes and Multi-dSprites number in PV2_0133–PV2_0137 comes from **one
CLEVR-trained core reused by transfer**, with only the head refit. That answers "how well does
CLEVR transfer". It does not answer "how good is the method when the core sees the target
domain", which is the stronger claim and the one this run measures.

It is also distinct from the three-seed per-dataset cores already on disk at
`/work/USERS/tkim1/sw0255/run/jobs/model253/*/seed{0,1,2}`, which are **3,500 unique × 10
epochs** (2,190 updates). This is the **matched 10,000 × 10** budget, a 2.86× increase, so for
the first time a per-dataset core is directly comparable with 253-on-CLEVR.

## Budget, verified rather than assumed

`SW_TRAIN_PRESET` is deliberately unset, which selects the default preset. Read back from each
run's own `binding.json`:

| | unique_TRAIN | len(train_ids) | subset_seed | parent pool |
|---|---|---|---|---|
| clevr seed 0 (reference) | 10,000 | 10,000 | 2553500 | 35,000 |
| **tetrominoes (this run)** | **10,000** | **10,000** | **2553500** | 35,000 |
| **multi_dsprites (this run)** | **10,000** | **10,000** | **2553500** | 35,000 |
| tetrominoes, older core | 3,500 | 3,500 | 2553500 | 35,000 |

Identical selection procedure and identical count to 253-on-CLEVR. Batch 16 × 10 epochs →
**6,250 updates**, matching seed 0 exactly.

## Commands

```bash
B=/work/USERS/tkim1/sw0255b
PY=/work/USERS/tkim1/envs/miniforge3/envs/snn/bin/python
unset SW_TRAIN_PRESET
CUDA_VISIBLE_DEVICES=2 $PY $B/benchmark_codes/run.py \
  --config $B/benchmark_codes/configs/frontier.json \
  --output $B/run_261_tetrominoes    train --dataset tetrominoes    --model 253 --seed 0
CUDA_VISIBLE_DEVICES=3 $PY $B/benchmark_codes/run.py \
  --config $B/benchmark_codes/configs/frontier.json \
  --output $B/run_261_multi_dsprites train --dataset multi_dsprites --model 253 --seed 0
```

Logs `core261_tetrominoes.log`, `core261_multi_dsprites.log`. Measured **29.5 s/update
(51.1 h)** and **30.5 s/update (52.9 h)**; in parallel on GPUs 2 and 3, about **2.2 days**.

## The head is per-dataset too

Every part of a `261 on Tetrominoes` row comes from Tetrominoes. The core trains on
Tetrominoes from random init with nothing carried over from CLEVR; the head is then fitted on
**Tetrominoes' own unlabelled training images**, selected on Tetrominoes' validation split, and
scored on Tetrominoes' test split. Same for Multi-dSprites. Nothing crosses between datasets.

That is also true of the rows this is compared against: the PV2_0134 transfer rows are a
**CLEVR core with the head refit on the target dataset**. So in both columns the head is local,
and the single difference between them is what the **core** trained on — which is what makes
the comparison readable.

The head has to be refit per core in any case: it reads the core's activity, so a different
core needs its own head. A head fitted on one core cannot be moved to another.

What uses labels and what does not:

| | labels from the evaluated dataset |
|---|---|
| core training | none — unsupervised |
| head training loss | none — rgb, tv, spike, anchor |
| head arm/epoch under rule B′ | **yes**, that dataset's validation FG-ARI |
| head under the fixed recipe | none |

So the `recipe` row is label-free end to end, and the `B′` row costs a labelled validation
split. Both are reported.

## The blocker that was predicted, found, and fixed

`cache_any.py:34` read

```python
job = Path(a.run)/'jobs/model253/clevr/seed0'
```

hardcoded to **clevr/seed0**, exactly as `PV2_0138`'s precondition warned. `--run` was doing two
unrelated jobs at once: the data root for `connect_inputs` and the checkpoint location. It now
takes a separate `--job`, defaulting to the old path so existing commands are unchanged, and it
exits with a clear message if no `resume.pt` is there.

Verified on the same image (id 7000) rather than assumed: caching from the Tetrominoes core
gives features differing by **0.86 relative** from the CLEVR core's, different gamma, and
**117 regions against 15**. The flag genuinely loads the per-dataset core.

## Steps after the cores finish

1. Cache activity (`features`, `gamma`, `regions_253`, `regions_all`) with `cache_any.py`
   at `--min-group 8`, on train / validation / test.
2. Fit the 261 readout — `(prior + 2·tanh((sim − mean)/std))` masked to the locality radius,
   `prior_scale` 4.0 and `residual_scale` frozen, `content_encoder` frozen. This is what
   `fit5.Readout` implements, verified against SW0260's form and against `native_routing`.
3. Score under the patch contract at grid 64 on the **same 320 evaluation images** the existing
   transfer rows use, so the two columns are directly comparable. Note these datasets have
   **1,000** validation and 1,000 evaluation ids available; extending both columns to the full
   1,000 is a follow-up, and must be done for *both* or neither.

> ## CORRECTION — the first version of this comparison was confounded
>
> The target-trained rows were first taken from the **engine's own** saved predictions, which
> use `min_group_size = 8 if grid==32 else 32` (`vendor/full_benchmark/evaluation.py:25`), while
> the CLEVR-core rows came from `cache_any.py --min-group 8`. Six other readout parameters
> matched (`foreground_threshold` 0.5, `synchrony_threshold` 0.5, `settle` 512, `background`
> largest_component, `affinity_mode` spike, `spatial_grid_size` 64) but **`min_group_size` did
> not**, and that parameter matters — an earlier sweep moved all three CLEVR metrics with it.
>
> So "the only difference is what the core trained on" was **false**: there were two differences.
> The rows below are the corrected ones, re-cached at `min_group 8` on both sides, with the 320
> image ids verified identical. **The gap widened** rather than closed.

## RESULT — both datasets, core only, confounds removed and verified

Grid 64, 10,000 × 10 on every row, the same 320 evaluation images, the same
`evaluate_patch_masks`. The only difference between a pair of rows is **what the core trained
on**.

| dataset | core trained on | fg_ari | foreground_iou | matched_object_iou |
|---|---|---|---|---|
| **Tetrominoes** | **CLEVR**, frozen, never saw it | **0.8580** | **0.7885** | **0.8173** |
| Tetrominoes | itself | **0.5448** | **0.6217** | **0.4434** |
| | **difference** | **+0.3132** | +0.1668 | +0.3739 |

Tetrominoes regions per image: **22.1** from the CLEVR core against **168.5** from the
target-trained one, truth **3.0 objects**. Matching `min_group` to 8 barely moved this row
(0.5298 → 0.5448); on Multi-dSprites it moved a lot, and in both cases the conclusion held.
| **Multi-dSprites** | **CLEVR**, frozen, never saw it | **0.6974** | **0.8177** | **0.7628** |
| Multi-dSprites | itself | **0.1201** | **0.0597** | **0.0050** |
| | **difference** | **+0.5773** | **+0.7580** | **+0.7578** |

Multi-dSprites regions per image: **7.6** from the CLEVR core against **413.6** from the
target-trained one, on a truth of **2.5 objects**. At `min_group 32` the target-trained core
read 0.1758 / 0.0309 / 0.0019 with 282.8 regions; matching the parameter to 8 moved it to
0.1201 / 0.0597 / 0.0050 with 413.6 regions, so the correction **widened** the gap.

### What is controlled, and the one thing that is not

| | matched |
|---|---|
| core architecture and variant (253) | yes |
| budget (10,000 × 10, 6,250 updates) | yes |
| grid (64) | yes |
| all seven readout parameters, `min_group` included | yes |
| the 320 evaluation image ids | yes, verified equal |
| evaluation function | yes |
| **training data** | **no — the variable under test** |
| random initialisation (each run draws its own encoder) | **no** |

Initialisation is the remaining uncontrolled difference. The measured seed standard deviation on
Multi-dSprites is **0.0826** (PV2_0144); the gap is **0.5773**, seven times that. On Tetrominoes
the std is 0.0396 against a gap of 0.33, eight times. Initialisation does not account for either.

### Corroborated by records written before this run

These numbers are not a surprise; they reproduce what the project already held.
`BENCHMARK_PLAN_v3.md` had recorded Multi-dSprites as the standing weak point at **0.2817** in
our own pipeline and **0.2789 ± 0.0813** over three seeds in the peer engine at grid 32 *trained
on Multi-dSprites*, while **a CLEVR-only grid-64 core reached 0.6893** on the same pixel protocol
over 1,000 TEST images. Our new 0.6974 (320 images) and 0.1201 sit in those two families. PLAN_v3
item 3 was literally "train the core on Multi-dSprites at grid 64 … this separates grid from
training data" — and the answer is that **the grid was never the problem; training on
Multi-dSprites is**.

For Multi-dSprites, `Dataset.labels(ids,'test')` and `Dataset.pixels(ids)` are **identical**
(both 64×64), verified — so for this dataset the patch protocol *is* the pixel protocol and the
ceiling of 1.0 is real, making these rows directly comparable with published pixel numbers.

### The peer's higher Multi-dSprites numbers are supervised

`PV2_0128` already carries a correction on this, issued after the peer clarified: none of their
Multi-dSprites rows is unsupervised.

| Multi-dSprites, pixel FG-ARI | supervision | |
|---|---|---|
| **ours, CLEVR-trained, zero-shot** | **unsupervised** | **0.6974** |
| peer zero-shot, TEST1000 | CLEVR **mask-supervised** source | 0.6833 |
| peer zero-shot, TEST360 | same | 0.6979 |
| peer adapted, TEST1000 | **+ target-mask supervision** | 0.7701 |

So our unsupervised CLEVR core lands on the peer's **mask-supervised** zero-shot numbers. That is
the comparison worth making, and it is only available because the protocol coincides here.

The Multi-dSprites core trained on its own data essentially failed — matched-object IoU
**0.0019**, with **282.8 regions per image** against a truth of **2.5 objects**. And
Multi-dSprites is **exactly 1:1** at grid 64 (64×64 pixels, 4,096 nodes), so oversampling cannot
excuse it.

On both datasets, more data at a finer grid made the target-trained core **worse**:

| | grid 32, 3,500 × 10 | grid 64, 10,000 × 10 |
|---|---|---|
| Tetrominoes | 0.7730 ± 0.0396 | **0.5298** |
| Multi-dSprites | 0.2964 ± 0.0826 | **0.1758** |

### With the head on — and the head matters here, unlike on a good core

The head was refit on the target-trained core's **own** activity (it cannot be moved between
cores; a head reads the core's features, and the two differ by 0.86 relative on the same image).

| Tetrominoes, target-trained core | fg_ari | foreground_iou | matched_object_iou | patches changed |
|---|---|---|---|---|
| core alone | 0.5448 | 0.6217 | 0.4434 | |
| **+ head, rule B′** (`local_rgb`, epoch 0) | **0.5578** | **0.6995** | **0.6184** | 16.9% |
| | +0.0130 | **+0.0778** | **+0.1750** | |
| + head, fixed label-free recipe | 0.5293 | **0.7668** | 0.5878 | 8.9% |
| | **−0.0155** | **+0.1451** | +0.1443 | |

This is the opposite of what the head does on the CLEVR core, where it changes 0.3–2% of patches
and its gains sit inside the noise. On a **weak** core it changes **16.9%** and lifts
matched-object IoU by **+0.1750** — it is partially repairing what the core fragmented.

It also exercised rule B′ properly. The label-free recipe trades **−0.0155 FG-ARI for +0.1451
Foreground IoU** — the opposite direction from the stated preference — so B′ declined it and took
`local_rgb` at epoch 0, which raises all three. That is the guard doing its job.

**But the head cannot close the gap.** Best head on the target-trained core, against the CLEVR
core alone:

| Tetrominoes | fg_ari | foreground_iou | matched_object_iou |
|---|---|---|---|
| CLEVR core, **no head** | **0.8580** | **0.7885** | **0.8173** |
| target-trained core + best head | 0.5578 | 0.6995 | 0.6184 |
| | **+0.3002** | +0.0890 | +0.1989 |

A head recovers some of a broken core's partition and nowhere near all of it. The
generalisation lives in the core, and this is the sharpest evidence for that yet.

### Multi-dSprites with the head — and the head fit was impossible until it was chunked

The head fit **crashed with CUDA out-of-memory**, and the cause is the fragmentation itself:

| core | candidates/image, mean | max | decoder tensor at the max |
|---|---|---|---|
| CLEVR core, same images | **20.0** | 38 | 0.02 GiB |
| target-trained Multi-dSprites | **415.0** | **1,834** | 0.95 GiB, and backward keeps it |
| target-trained Tetrominoes | 172.4 | 258 | 0.13 GiB (fit, barely) |

`fit5` materialises `[candidates, 4096, 34]` in the decoder. At 1,834 candidates that plus the
retained activations overflows 23 GiB. **The core is so fragmented that the readout is
infeasible at standard settings** — which is a result in its own right, not a bug.
`code/head_chunk.py` recovers a number by evaluating the decoder in gradient-checkpointed chunks
over the candidate axis; the arithmetic is identical, only peak memory differs.

| Multi-dSprites, target-trained core | fg_ari | foreground_iou | matched_object_iou | patches changed |
|---|---|---|---|---|
| core alone | 0.1201 | 0.0597 | 0.0050 | |
| + head, fixed label-free recipe | **0.1631** | **0.2002** | **0.1255** | 1.88% |
| | **+0.0429** | **+0.1404** | **+0.1205** | |

### The complete table — core and head, both datasets

| dataset | configuration | fg_ari | foreground_iou | matched_object_iou |
|---|---|---|---|---|
| **Tetrominoes** | **CLEVR core, no head** | **0.8580** | **0.7885** | **0.8173** |
| | target-trained core | 0.5448 | 0.6217 | 0.4434 |
| | target-trained core + head | 0.5578 | 0.6995 | 0.6184 |
| **Multi-dSprites** | **CLEVR core, no head** | **0.6974** | **0.8177** | **0.7628** |
| | target-trained core | 0.1201 | 0.0597 | 0.0050 |
| | target-trained core + head | 0.1631 | 0.2002 | 0.1255 |

The head repairs a lot of a broken core — on Multi-dSprites +0.1404 Foreground IoU and +0.1205
matched-object IoU — and still finishes **0.5343 FG-ARI behind the CLEVR core carrying no head at
all**. On Tetrominoes the shortfall is 0.3002. **A head cannot substitute for a core that learned
to group.**

### `positive_spike` descent predicts everything, now properly controlled

PV2_0143 claimed this and it was withdrawn for mixing grids. All three cores below are grid 64
at 10,000 × 10, each scored on its own dataset under the patch protocol, so the comparison is
finally like-for-like:

| core trained on | `positive_spike` | score on its own data |
|---|---|---|
| CLEVR | **−83.2%** | **0.8421** |
| Tetrominoes | −33.3% | 0.5298 |
| Multi-dSprites | **−0.1%** | 0.1758 |

Strictly monotone. **How far the spike term descends decides how well the core works** — and it
is the term whose gradient path reaches the spikes the score is read from. `phase_primary`, which
is flat on both non-CLEVR datasets, predicts nothing.

## Tetrominoes detail

Grid 64, 10,000 × 10 on both sides, same 320 evaluation images, same `evaluate_patch_masks`.
The only difference is what the core trained on.

| dataset | core trained on | fg_ari | foreground_iou | matched_object_iou |
|---|---|---|---|---|
| Tetrominoes | **CLEVR**, frozen, never saw this dataset | **0.8580** | **0.7885** | **0.8173** |
| Tetrominoes | **itself**, 10,000 × 10 | 0.5298 | 0.6179 | 0.4417 |
| | difference | **+0.3282** | **+0.1706** | **+0.3756** |

**The CLEVR-trained core wins by 0.33 FG-ARI** — more than **8×** the measured seed standard
deviation on this dataset (0.0396, PV2_0144), so this is not seed noise.

And more data at a finer grid made the target-trained core **worse**, not better:

| comparison | CLEVR core | target-trained | gap |
|---|---|---|---|
| grid 32, 3,500 × 10 (PV2_0144) | 0.8580 | 0.7730 ± 0.0396 | +0.085 |
| **grid 64, 10,000 × 10 (this run)** | **0.8580** | **0.5298** | **+0.328** |

**The symptom is over-segmentation.** The target-trained core produces **135.8 regions per image
on average** (median 129, max 336) on scenes with roughly three objects; the CLEVR core gives
**15** on the same image. That matches PV2_0142 exactly: `phase_primary` never descended on this
dataset, and a core that never organised its phase structure fragments instead of grouping.

Scored from the engine's own saved predictions with `score_any.py`, so no caching was needed for
this row. Both sides are **seed 0**, and this is **core only** — the head rows follow when
`chain261.sh` finishes.

## Why the target-trained core is worse — training works, it just does not get as far

The obvious reading of PV2_0142 was that the loss gives no signal on this dataset, because
`phase_primary` never descended. **Scoring the untrained core refutes that.**

| Tetrominoes core | fg_ari | foreground_iou | matched_object_iou | regions/image |
|---|---|---|---|---|
| **untrained** (`initial.pt`) | **0.0100** | 0.0909 | 0.0048 | 361.5 |
| trained, grid 64, 10,000 × 10 | **0.5298** | 0.6179 | 0.4417 | 135.8 |
| trained, grid 32, 3,500 × 10 | 0.7730 ± 0.0396 | 0.7510 | 0.7659 | 6–25 |
| CLEVR core, frozen, never saw this data | **0.8580** | 0.7885 | 0.8173 | ~15 |

**Training lifts FG-ARI from 0.0100 to 0.5298 — a factor of 53** — and cuts regions per image
from 361.5 to 135.8. The loss is doing a great deal of work. It simply does not reach where
CLEVR training reaches. The learning came through `positive_spike`, which fell 33% on this
dataset; only `phase_primary` stayed flat.

So the diagnosis is **not** a broken or uninformative loss. It is that **CLEVR supplies this loss
with a far richer signal than Tetrominoes does**:

| | objects/image | native pixels | `phase_primary` start | reached |
|---|---|---|---|---|
| CLEVR | 6.4 | 128×128 | 4.64, far above its floor | **0.8580** |
| Tetrominoes | 3.0 | 35×35 | 2.62, near where CLEVR *ends* (2.46) | 0.5298 / 0.7730 |

Two things stack on Tetrominoes: less structure for the loss to exploit, and at grid 64 a
**3.34× oversampling** of a 35×35 image by 4,096 nodes, which is what drives fragmentation
(136 regions at grid 64 against 6–25 at grid 32, and the score moves with it). Multi-dSprites is
exactly 1:1 at grid 64, so oversampling cannot be the general explanation — only the
Tetrominoes-specific amplifier.

**This is the mechanism behind the generalisation result.** The CLEVR-trained core is not better
because CLEVR is "closer" to these datasets. It is better because CLEVR is the dataset rich
enough for this objective to learn grouping well, and what it learns then carries.

## The comparison this produces

| dataset | core trained on | fg_ari | fg_iou | obj_iou |
|---|---|---|---|---|
| Tetrominoes | CLEVR, transferred (PV2_0134) | 0.8580 | 0.7885 | 0.8173 |
| Tetrominoes | **itself, 10,000 × 10 (this run)** | pending | pending | pending |
| Multi-dSprites | CLEVR, transferred (PV2_0134) | 0.6974 | 0.8177 | 0.7628 |
| Multi-dSprites | **itself, 10,000 × 10 (this run)** | pending | pending | pending |

Both rows will carry the 261 readout fitted the same way, so the only difference between them is
what the core trained on.

## Note

CLEVR seeds 1 and 2 of the 253 core were stopped to free the GPUs for this, by decision. They
reached 20 and 25 updates of epoch 1; `run_s1/` and `run_s2/` are kept so they resume rather
than restart. The 3-seed requirement is therefore still open.
