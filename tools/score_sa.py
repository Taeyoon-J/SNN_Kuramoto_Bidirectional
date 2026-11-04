#!/usr/bin/env python3
"""Score Slot Attention with the same function and targets our model is scored with.

`evaluate_patch_masks` against `dataset.labels(ids,'fixed320')` -- the identical call the
engine makes for `historical_fixed64_patch`, on the identical 320 images. Nothing about the
comparison is inferred: same images, same grid, same metric, same targets.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--labels', required=True); ap.add_argument('--code', required=True)
ap.add_argument('--config', required=True); ap.add_argument('--run', required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()
CODE = Path(a.code); sys.path.insert(0, str(CODE/'vendor')); sys.path.insert(0, str(CODE))
import portable_runtime as pr
config = pr.configure(a.config); output = pr.connect_inputs(config, a.run)
from full_benchmark.data import Dataset
from mod_benchmark import common as c

blob = np.load(a.labels, allow_pickle=True)
labels = blob['patch_labels'].astype(np.int64)
grid = labels.shape[-1]
dataset = Dataset('clevr', output); ids = dataset.test
assert len(labels) == len(ids), (len(labels), len(ids))
metric = c.benchmark_module('evaluation')
target64 = torch.from_numpy(dataset.labels(ids, 'fixed320'))
if grid != 64:
    r = 64 // grid
    labels = np.repeat(np.repeat(labels, r, axis=1), r, axis=2)
s = metric.evaluate_patch_masks(torch.from_numpy(labels), target64)
KEYS = ('fg_ari','foreground_iou','matched_object_iou')
row = {k: float(s['mean'][k]) for k in KEYS}
row.update(grid=int(grid), count=len(ids), predicted_foreground=float((labels > 0).mean()),
           truth_foreground=float((target64.numpy() > 0).mean()),
           note='same images (fixed320 ids 1320-1639), same evaluate_patch_masks, same targets as 253')
print(f"Slot Attention grid {grid}: " + "  ".join(f"{k} {row[k]:.4f}" for k in KEYS))
print(f"  predicted foreground {row['predicted_foreground']:.4f}   truth {row['truth_foreground']:.4f}")
Path(a.out).write_text(json.dumps(row, indent=1)+'\n')
