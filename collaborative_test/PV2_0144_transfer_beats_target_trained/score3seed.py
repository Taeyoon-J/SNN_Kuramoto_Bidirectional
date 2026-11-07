#!/usr/bin/env python3
"""Score the finished per-dataset cores under OUR patch protocol, from their saved predictions.

The six 3,500 x 10 per-dataset cores were evaluated under the pixel protocol, which is not
comparable to anything else in this project. But their per-image predictions were saved as
32x32 label maps, and the patch contract scores a grid prediction against truth quantised to
the grid -- so the same lift Slot Attention got at grid 16 and 32 (np.repeat to 64, which is
exact for an integer factor) puts them on our protocol with no GPU and no retraining.

What this makes comparable, and what it does not:

  comparable   the evaluation function, the targets, the image ids
  NOT equal    grid (32 lifted to 64, against our native 64) and budget (3,500 x 10 against
               10,000 x 10)

So this is not the CLEVR-transfer comparison. It is the project's first three-seed table under
the contract's own protocol, which is what the reporting format has never been tested against.
"""
import argparse, json, statistics, sys
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--code', required=True); ap.add_argument('--config', required=True)
ap.add_argument('--run', required=True)
ap.add_argument('--old-root', default='/work/USERS/tkim1/sw0255/run/jobs/model253')
ap.add_argument('--limit', type=int, default=None,
                help='score only the first N evaluation ids, to match our 320-image rows')
ap.add_argument('--out', required=True)
a = ap.parse_args()
CODE = Path(a.code); sys.path.insert(0, str(CODE/'vendor')); sys.path.insert(0, str(CODE))
import portable_runtime as pr
config = pr.configure(a.config); output = pr.connect_inputs(config, a.run)
from full_benchmark.data import Dataset
from mod_benchmark import common as c
metric = c.benchmark_module('evaluation')
KEYS = ('fg_ari', 'foreground_iou', 'matched_object_iou')

report = {}
for ds in ('tetrominoes', 'multi_dsprites'):
    dataset = Dataset(ds, output)
    per = {k: [] for k in KEYS}
    rows = []
    for s in (0, 1, 2):
        E = Path(a.old_root)/ds/f'seed{s}/evaluation'
        fz = json.load(open(E/'frozen_predictions.json'))
        ids = np.asarray(fz['ids'], dtype=np.int64)
        if a.limit: ids = ids[:a.limit]
        preds = []
        for i in ids:
            m = np.load(E/f'{i:06d}.npy')
            assert m.shape == (32, 32), m.shape
            preds.append(np.repeat(np.repeat(m, 2, axis=0), 2, axis=1))   # exact 32 -> 64 lift
        pred = np.stack(preds).astype(np.int64)
        tgt = torch.from_numpy(dataset.labels(ids, 'test'))
        assert pred.shape[1:] == tuple(tgt.shape[1:]), (pred.shape, tgt.shape)
        sc = metric.evaluate_patch_masks(torch.from_numpy(pred), tgt)
        r = {k: float(sc['mean'][k]) for k in KEYS}
        rows.append(dict(seed=s, n=len(ids), **r))
        for k in KEYS: per[k].append(r[k])
    print(f'\n{ds}: 253-style core trained on {ds}, 3,500x10, grid 32 lifted to 64, '
          f'patch protocol, {rows[0]["n"]} images')
    print(f"  {'seed':>5s} " + ' '.join(f'{k:>22s}' for k in KEYS))
    for r in rows:
        print(f"  {r['seed']:>5d} " + ' '.join(f'{r[k]:>22.4f}' for k in KEYS))
    summ = {}
    for k in KEYS:
        v = per[k]
        summ[k] = dict(per_seed=v, mean=statistics.fmean(v), std=statistics.stdev(v))
    print(f"  {'mean':>5s} " + ' '.join(f'{summ[k]["mean"]:>22.4f}' for k in KEYS))
    print(f"  {'std':>5s} " + ' '.join(f'{summ[k]["std"]:>22.4f}' for k in KEYS))
    report[ds] = dict(rows=rows, metrics=summ,
                      note='grid 32 lifted to 64 by exact np.repeat; 3,500x10 budget; '
                           'patch protocol, same evaluate_patch_masks and targets as our rows')
Path(a.out).write_text(json.dumps(report, indent=1)+'\n')
print('\nwrote', a.out)
