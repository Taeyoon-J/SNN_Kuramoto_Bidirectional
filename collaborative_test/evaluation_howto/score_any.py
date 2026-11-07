#!/usr/bin/env python3
"""Score any patch-grid predictions exactly the way every number in patch_v2 was scored.

Point it at an .npz holding your predictions and the image ids, say which dataset and split,
and it prints the three metrics. Nothing model-specific: it only calls the benchmark's own
evaluation function on your labels.

  python score_any.py --code <benchmark_codes> --config <configs/frontier.json> --run <run> \
      --pred mypreds.npz --dataset clevr --split fixed320

The .npz needs two arrays:
  pred  int64 [N, G, G]  patch labels, background = 0, every patch exactly one id
  ids   int64 [N]        the image ids, in the same order as pred

THE PROTOCOL, in four points
  1. Both sides live on the patch grid. The ground truth is quantised to the grid by
     `Dataset.labels`, and the prediction is compared there. A prediction is NEVER expanded to
     pixels. This is why the ceiling is 1.0 and why these numbers are not comparable with
     pixel-protocol numbers from papers.
  2. The grid is 64. A coarser prediction is lifted by an exact `np.repeat` (64/G must be a
     whole number) -- the same lift Slot Attention was given at grid 16 and 32.
  3. Background is id 0, and the metrics are foreground-only in the truth's sense: a patch the
     truth calls background is excluded from fg_ari. So which of your labels is "background"
     changes the score a lot; decide it by a rule fixed in advance, on validation, never on the
     split you report.
  4. The three metrics are reported together and never combined into one score.

WHAT TO WATCH FOR
  - Predicted label ids are arbitrary. Only the partition and which id is 0 matter.
  - If your readout designates no background at all, foreground IoU collapses to the truth's
     foreground fraction and is constant for every model. That is a bug, not a plateau; it cost
     this project a whole experiment to notice.
  - `mean` is a nanmean over images, and images with an undefined metric are dropped per metric.
     Always check `valid_count` equals N before quoting a mean.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--code', required=True, help='the benchmark_codes directory')
ap.add_argument('--config', required=True); ap.add_argument('--run', required=True)
ap.add_argument('--pred', required=True, help="npz with 'pred' [N,G,G] and 'ids' [N]")
ap.add_argument('--dataset', required=True)
ap.add_argument('--split', required=True, help='train | validation | test | fixed320')
ap.add_argument('--grid', type=int, default=64)
ap.add_argument('--out', default=None)
a = ap.parse_args()

CODE = Path(a.code); sys.path.insert(0, str(CODE/'vendor')); sys.path.insert(0, str(CODE))
import portable_runtime as pr
config = pr.configure(a.config); output = pr.connect_inputs(config, a.run)
from full_benchmark.data import Dataset
from mod_benchmark import common as c
metric = c.benchmark_module('evaluation')      # vendor/benchmark/evaluation.py
print('evaluation module:', metric.__file__)

blob = np.load(a.pred)
pred, ids = blob['pred'].astype(np.int64), blob['ids'].astype(np.int64)
assert pred.ndim == 3 and len(pred) == len(ids), (pred.shape, ids.shape)
G = pred.shape[-1]
if G != a.grid:
    assert a.grid % G == 0, f'grid {a.grid} is not a whole multiple of {G}'
    r = a.grid // G
    pred = np.repeat(np.repeat(pred, r, axis=1), r, axis=2)   # exact lift, no interpolation
    print(f'lifted predictions {G} -> {a.grid} by exact repeat x{r}')

dataset = Dataset(a.dataset, output)
target = torch.from_numpy(dataset.labels(ids, a.split))
assert target.shape == pred.shape, f'target {tuple(target.shape)} vs prediction {tuple(pred.shape)}'
s = metric.evaluate_patch_masks(torch.from_numpy(pred), target)

KEYS = ('fg_ari', 'foreground_iou', 'matched_object_iou')
row = {k: float(s['mean'][k]) for k in KEYS}
valid = {k: int(s['valid_count'][k]) for k in KEYS}
print(f'\n{a.dataset} / {a.split} / {len(ids)} images / grid {a.grid}')
for k in KEYS:
    flag = '' if valid[k] == len(ids) else f'   <-- only {valid[k]}/{len(ids)} images defined'
    print(f'  patch_{k:20s} {row[k]:.4f}{flag}')
print(f'\n  predicted foreground {float((pred > 0).mean()):.4f}   '
      f'truth foreground {float((target.numpy() > 0).mean()):.4f}')
if a.out:
    Path(a.out).write_text(json.dumps(dict(dataset=a.dataset, split=a.split, count=len(ids),
        grid=a.grid, metrics=row, valid_count=valid), indent=1)+'\n')
    print('wrote', a.out)
