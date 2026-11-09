#!/usr/bin/env python3
"""Score a cached core's own partition, with no head, straight out of a cache directory.

The core-only rows every transfer claim is compared against (PV2_0137: CLEVR 0.8421,
Tetrominoes 0.8580, Multi-dSprites 0.6974) are `regions_253` evaluated by
`evaluate_patch_masks` -- no readout, no fitting, no background rule to choose. This reads a
cache written by `cache_any.py` and reproduces exactly that, so a continued-training core can
be put beside them without a train cache or a head.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--cache', required=True, help='a cache_any.py output directory')
ap.add_argument('--dataset', required=True)
ap.add_argument('--split', required=True, help='train | validation | test | fixed320')
ap.add_argument('--code', required=True); ap.add_argument('--config', required=True)
ap.add_argument('--run', required=True); ap.add_argument('--out', default=None)
ap.add_argument('--key', default='regions_253', choices=['regions_253', 'regions_all'])
a = ap.parse_args()

CODE = Path(a.code); sys.path.insert(0, str(CODE/'vendor')); sys.path.insert(0, str(CODE))
import portable_runtime as pr
config = pr.configure(a.config); output = pr.connect_inputs(config, a.run)
from full_benchmark.data import Dataset
from mod_benchmark import common as c
dataset = Dataset(a.dataset, output); metric = c.benchmark_module('evaluation')
KEYS = ('fg_ari', 'foreground_iou', 'matched_object_iou')

files = sorted(Path(a.cache).glob('*.pt'))
pred, ids, regions = [], [], []
for f in files:
    d = torch.load(f, map_location='cpu', weights_only=False)
    if 'image_id' not in d:            # the manifest, not a frame
        continue
    r = d[a.key].numpy().astype(np.int64)
    pred.append(r); ids.append(int(d['image_id'])); regions.append(int(r.max()))
pred = np.stack(pred); ids = np.asarray(ids, dtype=np.int64)
order = np.argsort(ids)                # the contract compares aligned rows; sort both the same way
pred, ids = pred[order], ids[order]
print(f'{a.cache}: {len(ids)} frames, ids {ids[0]}..{ids[-1]}, grid {pred.shape[-1]}, '
      f'key {a.key}, mean regions {np.mean(regions):.1f}')

target = torch.from_numpy(dataset.labels(ids, a.split))
s = metric.evaluate_patch_masks(torch.from_numpy(pred), target)
row = {k: float(s['mean'][k]) for k in KEYS}
if 'valid_count' in s:
    row['valid_count'] = {k: int(s['valid_count'][k]) for k in KEYS}
row.update(n=len(ids), mean_regions=float(np.mean(regions)),
           predicted_foreground=float((pred > 0).mean()),
           truth_foreground=float((target.numpy() > 0).mean()))
print(f"  fg_ari {row['fg_ari']:.4f}  fg_iou {row['foreground_iou']:.4f}  "
      f"obj_iou {row['matched_object_iou']:.4f}  "
      f"pred_fg {row['predicted_foreground']:.4f}  truth_fg {row['truth_foreground']:.4f}")
if row.get('valid_count'):
    print('  valid_count', row['valid_count'], '(must equal', len(ids), 'to quote the mean)')
if a.out:
    Path(a.out).write_text(json.dumps(dict(cache=a.cache, dataset=a.dataset, split=a.split,
                                           key=a.key, **row), indent=1)+'\n')
    print('->', a.out)
