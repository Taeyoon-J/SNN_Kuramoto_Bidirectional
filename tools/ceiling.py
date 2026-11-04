#!/usr/bin/env python3
"""The quantisation ceiling: the best FG-ARI a grid-64 prediction can reach under the pixel
protocol, measured per dataset rather than assumed.

Slot Attention reports pixel-level FG-ARI (Tetrominoes 99.5, Multi-dSprites 91.3, CLEVR6
98.8). We predict one label per grid cell. Expanding a 64x64 prediction back to 128x128
pixels cannot resolve anything finer than a cell, so there is a hard upper bound: take the
pixel truth, quantise it to the grid by majority vote, expand it back, and score it against
itself. A perfect model scores exactly that. If the bound sits below Slot Attention's
number, no model change can win and the grid has to change instead.

No model is run here -- this is a property of the data and the grid.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--code', required=True); ap.add_argument('--config', required=True)
ap.add_argument('--run', required=True); ap.add_argument('--count', type=int, default=320)
ap.add_argument('--out', required=True)
a = ap.parse_args()
CODE = Path(a.code); sys.path.insert(0, str(CODE/'vendor')); sys.path.insert(0, str(CODE))
import portable_runtime as pr
config = pr.configure(a.config); output = pr.connect_inputs(config, a.run)
from full_benchmark.data import Dataset
from full_benchmark.evaluation import metrics
from mod_benchmark import common as c

cells = c.benchmark_module('make_targets_grid')
KEYS = ('fg_ari','foreground_iou','matched_object_iou')
table = {}
for ds in ('clevr','tetrominoes','multi_dsprites'):
    dataset = Dataset(ds, output)
    split = 'fixed320' if ds == 'clevr' else 'test'
    ids = dataset.test[:a.count]
    pixels = dataset.pixels(ids)                      # the pixel-level truth, 128x128
    row = {}
    for grid in (32, 64):
        if min(pixels.shape[1:]) < grid:
            print(f"{ds:15s} grid {grid}: impossible -- native pixels are {pixels.shape[1]}x{pixels.shape[2]}", flush=True)
            row[f"grid{grid}"] = None
            continue
        # majority label per grid cell, then expanded back to pixels: a perfect grid-`grid`
        # prediction reproduces exactly this, so it is the ceiling.
        quant = cells.mask_cells(torch.from_numpy(pixels.astype(np.int64)), grid)[0]
        r, _ = metrics(quant.numpy().astype(np.int64), dataset, ids)
        row[f'grid{grid}'] = {k: float(r['pixel'][k]) for k in KEYS}
        print(f"{ds:15s} grid {grid}: " + "  ".join(f"{k} {row[f'grid{grid}'][k]:.4f}" for k in KEYS), flush=True)
    row['truth_foreground_fraction'] = float((pixels > 0).mean())
    row['count'] = len(ids)
    table[ds] = row
Path(a.out).write_text(json.dumps(table, indent=1)+'\n')
print("CEILING DONE ->", a.out, flush=True)
