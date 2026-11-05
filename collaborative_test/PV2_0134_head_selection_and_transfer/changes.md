# PV2_0134 — changes

All files live on the server at `/work/USERS/tkim1/sw0255b/benchmark_codes/`, which is not a
git checkout; `artifacts.json` records their hashes and locations. Line numbers are at the
state recorded there.

## New: `fit4.py` — change C, cross-seed agreement

**Reason.** PV2_0133 found that across CLEVR's five arms, validation RGB MSE ranked exactly
inverse to validation FG-ARI. Reconstruction points away from grouping, so the head needed an
unsupervised signal that does not. The rollout is stochastic in its inference seed, so the
same image under two seeds gives two activity traces of one scene; a grouping that tracks
objects should survive the change, one that tracks rollout noise should not.

**Computation.** `Readout.logits_from(features, gamma, masks, prior, allowed)` was factored
out of `logits` so the same head can be evaluated on the second seed's activity
(`features_b`, `gamma_b`) with the **same** candidate set, which keeps the two index spaces
aligned. The new term is a symmetric KL between the two seeds' soft assignments at
temperature 0.5.

**Expected effect.** Higher validation FG-ARI than the rgb/tv arms.

**Observed effect.** Collapse. Every arm reached the identical validation FG-ARI regardless of
its rgb and tv weights — 0.4954 on CLEVR against a 253 baseline of 0.8501, 0.6817 on
Tetrominoes against 0.8745. Assigning every node to one candidate makes both seeds agree
perfectly and drives the KL to zero.

**Follow-up in the same file.** Two collapse guards were added over the marginal
`p.mean(1)`, plus a `size_prior` per row (`masks.sum(1)/masks.sum()`):

- `ent`: `+ ent * sum(m log m)` — maximise the marginal's entropy.
- `size`: `+ size * KL(size253 || m)` — hold the marginal near 253's own size distribution.

and `--val-every` defaulted to 1 with a validation point recorded at **epoch 0**, before any
update, so selection can decline to move. With the guards no arm on any dataset beat its own
epoch 0. Verified by `diagnostics/fit4/*.json`.

## New: `fit5.py` — rule B′, derived from `fit3.py`

**Reason.** Change B selects on validation FG-ARI alone, which has no opinion about how much
of the image is called foreground. On Tetrominoes change C's epoch-0 head scored +0.0010
FG-ARI and **−0.1141 Foreground IoU** and B accepted it.

**Previous computation** (`fit3.py`, the selection block after the arm loop):

```python
if v['fg_ari'] > best[0]:
    best = (v['fg_ari'], e+1, {...})
```

a running maximum over validation FG-ARI only, starting after the first epoch.

**New computation** (`fit5.py`): the full per-epoch validation triple is recorded in `curve`,
epoch 0 is included, and selection is

```python
floor = base_val['foreground_iou'] - a.iou_floor          # a.iou_floor default 0.01
allowed = [c_ for c_ in curve if c_['foreground_iou'] >= floor]
bprime = max(allowed or curve, key=lambda c_: c_['fg_ari'])
bonly  = max(curve, key=lambda c_: c_['fg_ari'])          # recorded, not used
```

Both choices are stored per arm, so the correction's cost is visible in the log and in
`diagnostics/fit5/*.json`.

**Also in `fit5.py`, structural only, no change to any computation:** module-level argparse
and the runtime connection moved into `parse()` / `connect(a)` / `main(a, output)`, `load()`
took its cache key as an argument instead of reading a module global, and
`NeutralRegionHead` / `candidates` became local imports inside `Readout.__init__` and
`load()` — the package is only on `sys.path` after `connect()`. This exists so `xfer.py` and
`recipe.py` can reuse the identical model rather than copy it. Verified by importing `fit5`
and checking it has no side effects, and by CLEVR reproducing PV2_0133's A+B result
(0.8507/0.7830/0.6709 against 0.8503/0.7837/0.6706 — the difference is the epoch the new
every-epoch validation schedule can now reach).

**Expected effect.** Unchanged where Foreground IoU was never at risk, and a refusal where it
was.

**Observed effect.** CLEVR and Multi-dSprites select as before; on Tetrominoes `local_rgb`
moves from epoch 0 (0.8754 / 0.6717) to epoch 14 (0.7732 / 0.7858), and 12 of 21 epochs fail
the floor.

## New: `xfer.py` — does one head transfer?

**Reason.** `fit5.py` fits a head per dataset, which tests whether a head helps at all, not
whether the head generalises. The project's claim is that the core trained on CLEVR carries
to other datasets with the head as the cheap swappable part; that needs the head held fixed.

**Computation.** Loads a `fit5` checkpoint and calls `predict` on another dataset's evaluation
cache. No refitting, no per-dataset arm or epoch choice. Reuses `fit5.Readout` and `fit5.load`
directly, so there is no second copy of the model to drift.

**Observed effect.** The CLEVR head loses on all three metrics on both datasets:
Tetrominoes −0.0321 / −0.1068 / −0.0349, Multi-dSprites −0.0377 / −0.0124 / −0.0097.

## New: `recipe.py` — what a new dataset costs without labels

**Reason.** `fit5.py` trains with unsupervised losses but selects the arm and epoch on
validation FG-ARI, which needs labels. Reporting the refit result as label-free would be
wrong, so the recipe and the weights are separated.

**Computation.** Trains one fixed arm for a fixed number of epochs on the new dataset's
unlabelled images, with no validation call anywhere in the file. `--arm local_rgb_tv
--epochs 11` is CLEVR's B′ choice, reused verbatim.

**Observed effect.** Reproduces the refit result on CLEVR to four decimals
(0.8507/0.7831/0.6710) and improves all three on Multi-dSprites; loses 0.0565 FG-ARI on
Tetrominoes, whose own B′ choice is `local_rgb_trust` at epoch **2**.

## New: `sa_dump.py` + `sa_infer.py` — Slot Attention, split across two environments

**Reason.** `sa_raw.py` was written as one file and could not run: the `tfgpu` environment has
no `torch` and `full_benchmark.data.Dataset` needs it, so the data half and the TensorFlow
half cannot share a process. `sa_raw.py` was deleted.

**Computation.** `sa_dump.py` (torch env) writes the split's frames and ids;
`sa_infer.py` (tf env) runs the checkpoint and saves the **raw** per-patch slot map before any
background remapping, plus the slot-share diagnostic. Frames come from `dataset.rgb`, already
128×128 — the resolution Slot Attention was published at and the same frames the core is
given — so there is no crop and no resize and grid cell (r, c) means the identical pixels in
prediction and target. Ran on CPU (`CUDA_VISIBLE_DEVICES=""`) because both GPUs were in use.

## New: `sa_bg.py` — choosing the background rule on validation

**Reason.** Slot ids are arbitrary, so mapping one to the contract's background id 0 is a free
choice, and `run_slot_attention.py:114` fixes it to "the slot covering the most patches
overall" with no justification. Under that rule Slot Attention calls 85.1% of the image
foreground where the truth is 21.2%, which is most of why its Foreground IoU reads 0.2022. A
wrong choice there would be our error, not the model's.

**Computation.** Four rules fixed before looking at any number — `global_largest`,
`image_largest`, `border_largest`, `border_union` — scored on the **validation** split and
selected there, with every rule also reported on fixed320 as a labelled diagnostic.

**Observed effect.** All four give the identical FG-ARI (0.4061 validation, 0.3747 fixed320),
because the rules only relabel or merge border slots and those patches are background in the
truth, which FG-ARI excludes. The selection was therefore amended to break ties on Foreground
IoU — the choice most generous to the baseline — selecting `border_largest`, 0.3747 / 0.2219
/ 0.1165. The background rule is worth at most +0.017 Foreground IoU and nothing on FG-ARI.

## Related test IDs

PV2_0133 (changes A and B, and the five-arm RGB-MSE-versus-FG-ARI finding that motivated C);
`BENCHMARK_PLAN_v3.md` (the patch protocol these numbers are scored under, ceiling 1.0).
