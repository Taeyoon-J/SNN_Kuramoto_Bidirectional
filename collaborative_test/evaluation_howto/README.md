# How every number on `patch_v2` was scored

Written for whoever wants to score their own predictions the same way. Copy `score_any.py`,
point it at your predictions, and you get the same three numbers by the same code path. Two
reference values at the end let you confirm your setup before you trust it.

## The whole thing is one function call

```python
from mod_benchmark import common as c
metric = c.benchmark_module('evaluation')        # -> vendor/benchmark/evaluation.py
target = torch.from_numpy(Dataset(ds, output).labels(ids, split))
s = metric.evaluate_patch_masks(torch.from_numpy(pred), target)   # pred, target: [N, 64, 64]
fg_ari  = float(s['mean']['fg_ari'])
fg_iou  = float(s['mean']['foreground_iou'])
obj_iou = float(s['mean']['matched_object_iou'])
```

`evaluate_patch_masks(prediction, target, background_id=0)` is at
`vendor/benchmark/evaluation.py:194`. Its own docstring is the contract:

> Evaluate three metrics on aligned `[B, grid_h, grid_w]` patch labels. Return `per_image`,
> `mean` (scalar image-wise average), and `valid_count`. Undefined scores are excluded per
> metric. **No combined weighted score is formed.** Each patch must have exactly one ID:
> resolve overlapping masks before calling, **without using ground truth**.

## Run it

```bash
python score_any.py \
  --code  <path>/benchmark_codes \
  --config <path>/benchmark_codes/configs/frontier.json \
  --run   <path>/run \
  --pred  mypreds.npz --dataset clevr --split fixed320
```

`mypreds.npz` holds exactly two arrays:

| array | dtype | shape | meaning |
|---|---|---|---|
| `pred` | int64 | `[N, G, G]` | patch labels, **background = 0**, one id per patch |
| `ids` | int64 | `[N]` | image ids, **same order as `pred`** |

## The protocol, in four points

**1. Both sides live on the patch grid.** `Dataset.labels` returns the ground truth already
quantised to the grid, and the prediction is compared there. A prediction is **never expanded to
pixels**. That is why the ceiling is **1.0** — and why these numbers are **not comparable with
pixel-protocol numbers from papers**, whose ceiling is below 1.0 because a grid prediction can
never recover pixel boundaries.

**2. The grid is 64.** A coarser prediction is lifted by an exact `np.repeat` (64/G must be a
whole number), which is what `score_any.py` does and what Slot Attention was given at grid 16
and 32. No interpolation, so no new information is invented.

**3. Background is id 0, and it matters a lot.** The metrics are foreground-only in the
*truth's* sense — a patch the truth calls background is excluded from `fg_ari`. So which of your
labels you map to 0 moves the score. Decide it by a rule fixed **in advance**, on
**validation**, never on the split you report. For Slot Attention we fixed four rules up front
and chose on validation; all four gave the identical `fg_ari` (0.3747), because they only touch
patches the truth calls background.

**4. The three metrics are reported together and never combined.** No single score.

## Three traps, each of which cost us something

**A readout that designates no background.** If nothing in your prediction is 0, then
`prediction != 0` is the whole image and `foreground_iou` collapses to the truth's foreground
fraction — a **constant**, identical for every model and every checkpoint. Our variant 252 ran
to completion reading `foreground_iou = 0.2021225` at all five checkpoints, bitwise identical,
and it was read as a plateau rather than as the bug it was. **If a metric is constant to seven
decimals across checkpoints, stop the run.**

**`mean` is a nanmean.** Images where a metric is undefined are dropped **per metric**, so the
three means can be over different image counts. `score_any.py` flags it; always confirm
`valid_count == N` before quoting a mean.

**Label ids are arbitrary.** Only the partition and which id is 0 carry meaning. Do not expect
your ids to match the truth's, and do not renumber using the truth.

## Confirm your setup with these two

Both are reproduced by `score_any.py` as shipped, on 320 images:

| what | command inputs | expected |
|---|---|---|
| our 253 core's own partition, CLEVR | `--dataset clevr --split fixed320`, grid 64 | **0.8421 / 0.7545 / 0.6410** |
| a grid-32 core on Tetrominoes, lifted | `--dataset tetrominoes --split test`, grid 32 | **0.7812 / 0.7015 / 0.7431** |

If you get those two, your path matches ours exactly. The first exercises the native-grid path,
the second the lift.

To regenerate the first input from our cache:

```python
import numpy as np, torch, pathlib
C = pathlib.Path('<path>/ev/clevr_fixed320')
pred, ids = [], []
for p in sorted(C.glob('[0-9]*.pt')):
    d = torch.load(p, map_location='cpu', weights_only=True)
    pred.append(d['regions_253'].numpy().astype(np.int64)); ids.append(int(d['image_id']))
np.savez('core253_clevr_fixed320.npz', pred=np.stack(pred), ids=np.array(ids, dtype=np.int64))
```

## Two helpers for scoring a baseline (`PV2_0150`)

`score_any.py` assumes your prediction is already on the grid and already uses background id 0.
A clustering baseline usually satisfies neither, so two scripts here close the gap, and both are
part of the contract rather than conveniences:

| script | what it does |
|---|---|
| `to_grid64.py` | reduces a partition **finer** than the grid to the grid by the contract's **patch majority vote**, ties to the smaller label id. The contract's `np.repeat` lift handles predictions coarser than the grid; this is the same rule in the other direction. Nothing is ever expanded to pixels. |
| `bg_any.py` | picks which cluster id means background, out of the **same four rules** `sa_bg.py` fixed in advance (`global_largest`, `image_largest`, `border_largest`, `border_union`), **on validation**, FG-ARI first and Foreground IoU as the tie-break. It then reports the selected rule on `fixed320` and every rule as a labelled diagnostic. |

A third script, `score_cache.py`, scores **our own** core with no head at all: it reads a
`cache_any.py` directory, takes `regions_253` and calls `evaluate_patch_masks` on it. That is the
code path the core-only reference rows came from, and it reproduces them exactly — CLEVR
0.8421 / 0.7545 / 0.6410 and Tetrominoes 0.8580 / 0.7885 / 0.8173 — so a new core can be put
beside them without fitting a head first.

```bash
python score_cache.py --cache <cache dir> --dataset tetrominoes --split test \
  --code <path>/benchmark_codes --config <path>/benchmark_codes/configs/frontier.json \
  --run <path>/run
```

`bg_any.py` is `sa_bg.py` generalised: any model whose readout is a hard partition with
arbitrary ids, any dataset, predictions in an npz under `preds` and ids either in the same npz
or in the source data npz via `--val-ids` / `--eval-ids`.

```bash
python to_grid64.py --inp preds128.npz --out preds_g64.npz
python bg_any.py --val val_g64.npz --eval eval320_g64.npz \
  --val-ids val_data.npz --eval-ids eval320_data.npz --dataset multi_dsprites \
  --code <path>/benchmark_codes --config <path>/benchmark_codes/configs/frontier.json \
  --run <path>/run --out bg.json
```

All four rules give the **identical FG-ARI** on every dataset we have tried, because they only
relabel patches the truth calls background and FG-ARI excludes those. So the rule choice moves
Foreground IoU and matched-object IoU only — which is exactly why it must not be made on the
split you report.

## Where things are (frontier)

```
scorer         /work/USERS/tkim1/sw0255b/benchmark_codes/score_any.py
the function   /work/USERS/tkim1/sw0255b/benchmark_codes/vendor/benchmark/evaluation.py:194
our core       /work/USERS/tkim1/share/native64_core_253/{resume.pt, initial.pt}
               sha256 e56b1a4219ab606b35952dab99ce0623...  (6,250 updates, 10,000x10, grid 64)
our heads      /work/USERS/tkim1/sw0255b/fit5/*.pt
cached activity /work/USERS/tkim1/sw0255b/ev/<dataset>_<split>/
```

All of it is readable by anyone in group `acm`.

**`initial.pt` is required alongside `resume.pt`** — the feature `normalization` lives there and
the model cannot be built without it.

**Do not load** `/work/USERS/tkim1/sw0255b/run/jobs/model253/clevr/seed0/resume.pt`. It is an
aborted run at **8 updates**, and its file size is **identical** (336,219,420 B) to the real
checkpoint's, so size does not distinguish them. Check the sha256.

## Splits

`Dataset` fixes them; do not redraw. CLEVR validation is ids 1000–1319 and the reserved
evaluation split `fixed320` is 1320–1639. Tetrominoes and Multi-dSprites have 1,000 validation
and 1,000 test ids; our published rows use the **first 320** of test, so match that if you want
to compare against them directly.
