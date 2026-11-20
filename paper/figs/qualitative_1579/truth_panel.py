#!/usr/bin/env python3
"""Render the ground-truth patch labels of one image, on the same grid and scale."""
import argparse, sys
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument('--code', required=True); ap.add_argument('--config', required=True)
ap.add_argument('--run', required=True); ap.add_argument('--dataset', default='clevr')
ap.add_argument('--split', default='fixed320'); ap.add_argument('--image-id', type=int, required=True)
ap.add_argument('--scale', type=int, default=8); ap.add_argument('--out', required=True)
a = ap.parse_args()
CODE = Path(a.code); sys.path.insert(0, str(CODE/'vendor')); sys.path.insert(0, str(CODE))
import portable_runtime as pr
config = pr.configure(a.config); output = pr.connect_inputs(config, a.run)
from full_benchmark.data import Dataset
ds = Dataset(a.dataset, output)
truth = ds.labels(np.array([a.image_id]), a.split)[0].astype(np.int64)
PALETTE = np.array([[68,119,170],[238,102,119],[34,136,51],[204,187,68],[170,51,119],
                    [102,204,238],[187,187,187],[238,119,51],[0,153,136],[136,34,85]], dtype=np.uint8)
ids = [int(v) for v in np.unique(truth) if v > 0]
ids.sort(key=lambda v: -int((truth == v).sum()))
canvas = np.zeros(truth.shape + (3,), dtype=np.uint8)
for i, v in enumerate(ids): canvas[truth == v] = PALETTE[i % len(PALETTE)]
big = np.repeat(np.repeat(canvas, a.scale, axis=0), a.scale, axis=1)
from PIL import Image
Image.fromarray(big).save(a.out)
print(f'{a.out}: {len(ids)} true objects, sizes {[int((truth==v).sum()) for v in ids]}, {big.shape}')
