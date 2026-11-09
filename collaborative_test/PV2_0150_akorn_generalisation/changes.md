# PV2_0150 — changes

Base commit `17d0fde931ff4da908bfa904bcf91aac52ea9651`, branch `patch_v2`. **No change to the
core, the losses, the heads or the evaluation function.** Everything here is baseline plumbing
and two scorer helpers.

## New, in the repo

### `collaborative_test/evaluation_howto/bg_any.py` (new file, 114 lines)

`sa_bg.py` generalised. `sa_bg.py` hardcodes `Dataset('clevr', output)` at line 37 and reads a
`slots` array with `num_slots` stored in the npz, so it only ever scored Slot Attention on
CLEVR. `bg_any.py` takes `--dataset`, reads `preds` (`--key` to override), infers
`K = max(pred)+1` instead of requiring a stored slot count, and takes ids either from the
prediction npz or from the source data npz via `--val-ids` / `--eval-ids`.

- **unchanged, deliberately:** the four rules (`border_mask`, `relabel`, `apply_rule`), the
  selection (`max` on validation `fg_ari` rounded to 6 places, tie-break on `foreground_iou`),
  and the diagnostic print of every rule on `fixed320`. Copied verbatim so a baseline's
  background designation is picked the same way for every baseline.
- **one cosmetic change:** in `sa_bg.py:59-73` the `global_largest` branch sets
  `bg = [{k}] * len(slots)` and then still enters the per-image loop, where no branch matches,
  so it appends nothing and the result is correct — just a wasted pass over the split.
  `bg_any.py` returns from that branch directly. **No number changes.**
- `score()` additionally records `valid_count` per metric when the evaluation function reports
  it, because the contract says a mean is only quotable when `valid_count == N`.

**Reason** a baseline whose readout is a hard partition with arbitrary ids cannot be scored
without designating a background, and that designation is worth more than most model changes
(on CLEVR it moves AKOrN's Foreground IoU from 0.2154 to 0.3061). **Expected effect** none on
any of our own numbers; it only makes a baseline's row reproducible and selected on validation.

### `collaborative_test/evaluation_howto/to_grid64.py` (new file, 32 lines)

Reduces a partition finer than the grid to the grid by per-patch majority vote, ties to the
smaller label id (`counts.argmax(-1)` on a `[N, G, G, K]` count tensor built with `np.add.at`).
A factor of 1 is a pass-through.

**Reason** AKOrN emits its partition at image resolution 128, and the contract forbids expanding
a prediction to pixels; the contract's existing `np.repeat` lift only covers predictions coarser
than the grid. **Expected effect** this is the only step between AKOrN's raw output and our
evaluation function, so it is the step most able to silently cost AKOrN points — it is a majority
vote, which is the contract's own patch rule, and the verification below bounds its cost.

**Verification** AKOrN's own FG-ARI (its `eval_obj.py`, its `calc_fgari_score`, at 128) against
ours (after this reduction, at grid 64), on the same images:

| cell (seed 0) | AKOrN's own | ours after reduction | diff |
|---|---|---|---|
| CLEVR | 0.6035 | 0.6027 | −0.0008 |
| Tetrominoes | 0.5680 | 0.5676 | −0.0004 |
| Multi-dSprites | 0.4353 | 0.4343 | −0.0010 |
| Multi-dSprites, dSprites-trained (already grid 64, factor 1) | 0.4562 | 0.4562 | 0.0000 |

The factor-1 row is exact, as it must be. The three factor-2 rows cost AKOrN ≤0.0010 FG-ARI,
two orders of magnitude below the gaps in the headline table. The seed-1234 runs give the same
picture (≤0.0011).

### `collaborative_test/PV2_0150_akorn_generalisation/akorn_{clevr_data,data128,val_data}.py`

Build our splits in AKOrN's `NumpyDataset` format. The one decision worth recording: `labels` is
**`Dataset.labels(ids, split)` at grid 64, `np.repeat`-ed ×2 to 128** — not `Dataset.pixels`.

`vendor/full_benchmark/data.py:86-92`: for CLEVR, `pixels(ids)` opens the canonical h5 and takes
`stream['mask'][ids, ..., 0]`, indexing the file **globally**, while `labels(ids, split)` at
`:81-84` goes through `r.truth(dataset, split, ids)` and the split manifest, and `data.py:32`
keeps a separate `source_ids` mapping for exactly that reason. The two need not agree on row
order, and nothing in the repo asserts that they do. Using `labels` makes the tensor AKOrN scores
itself against the identical tensor `evaluate_patch_masks` uses, so the question cannot arise.
The ×2 lift is exact, so AKOrN loses no information.

**Expected effect** AKOrN's own reported FG-ARI becomes directly comparable to ours (verified
above, ≤0.0011), at the cost of its mBO no longer being comparable to its published mBO, which
is on true pixel masks. Its mBO is recorded as a diagnostic only and no claim rests on it.

## New, on frontier only (outside git)

### `/work/USERS/tkim1/baselines/_clone/akorn/eval_dump.py`

A copy of `eval_obj.py` (commit `eabbe27`, untouched) with two additions:

1. `get_loader`, the generic `else` branch, `shuffle=True` → `shuffle=False`.
   **Reason** the upstream eval loader shuffles even at evaluation, which does not change
   AKOrN's own means — each image is clustered and scored independently — but destroys the
   alignment between the prediction order and the id order, so predictions cannot be joined to
   our truth. **Verification** the reported numbers are unchanged by the flag: 0.4514 (shuffled),
   0.4561 (shuffled), 0.4524 (unshuffled) on the dSprites row across reruns, which is the
   agglomerative clustering's own run-to-run spread and brackets both settings.
2. Nine lines appended after `print_stats(scores)` that concatenate `preds` and write them to
   `$DUMP`. No model, metric or data code is touched.

`train_obj.py`, `eval_obj.py` and everything under `source/` are unmodified; `git status` in the
clone shows only the new `eval_dump.py`, the new `data/` files and `runs/`.

### Environment

`scikit-learn 1.7.2` and `tensorboard` installed into `/work/USERS/tkim1/envs/miniforge3/envs/snn`.
`source/evals/objs/fgari.py:2` imports `sklearn.metrics.adjusted_rand_score`; the env did not
have it. Neither package is imported by any of our own code paths.

## A correction, after the record was first pushed

The first AKOrN runs were launched without `--seed`. `train_obj.py:75` defaults to
`--seed 1234`, so the numbers in the first version of this record were that seed, not seed 0,
even though they were labelled seed 0. Both AKOrN runs were retrained with `--seed=0` and
rescored, and the record now carries seed 0 as the headline with the 1234 runs kept in
`metrics.csv` as a second seed.

It changed the numbers materially — CLEVR FG-ARI 0.4688 → 0.6027 — so it is worth stating what
it did and did not change: **the conclusion is unchanged** (9 of 9 cells still ours), and the
two seeds now bound AKOrN's own noise at this budget, which is the more useful thing to have.
Our margin exceeds that spread on every dataset.

## Related test IDs

| id | relation |
|---|---|
| `PV2_0149` | protocol A, the same exposure budget and the same four background rules; Slot Attention's row |
| `PV2_0137` | our CLEVR-trained core's zero-shot rows, the three numbers AKOrN is compared against |
| `PV2_0138` | our target-trained Multi-dSprites core, 0.1201, the fourth row of the second table |
| `PV2_0147` | why the target-trained cores are broken, which is what the 0.1201 vs 0.6974 gap means |
