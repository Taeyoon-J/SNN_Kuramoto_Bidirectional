#!/usr/bin/env python3
"""Dump the frames and ids of any dataset's split, so Slot Attention can be run in its own env.

`sa_dump.py` is CLEVR-only and is left untouched so PV2_0149 stays reproducible. This copy adds
`--dataset` and takes the evaluation split the way our transfer rows do -- the **first 320** of
test, since Tetrominoes and Multi-dSprites have 1,000 test ids and our published rows use 320.

`dataset.rgb(ids)` is called exactly as `sa_dump.py` calls it, with no split argument: for every
dataset but CLEVR the split is ignored and the id indexes `rgb128.npy` directly, so this is
identical behaviour, and for CLEVR it is the identical call that produced the PV2_0149 row.
"""
import argparse, sys
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--code', required=True); ap.add_argument('--config', required=True)
ap.add_argument('--run', required=True); ap.add_argument('--dataset', required=True)
ap.add_argument('--split', required=True, help='fixed320 | validation')
ap.add_argument('--out', required=True)
a = ap.parse_args()
CODE = Path(a.code); sys.path.insert(0, str(CODE/'vendor')); sys.path.insert(0, str(CODE))
import portable_runtime as pr
config = pr.configure(a.config); output = pr.connect_inputs(config, a.run)
from full_benchmark.data import Dataset
dataset = Dataset(a.dataset, output)
if a.split == 'fixed320':
    ids = np.asarray(dataset.test)[:320]
else:
    ids = np.asarray(dataset.validation)
rgb = np.asarray(dataset.rgb(ids))
assert rgb.shape[1:] == (128, 128, 3) and rgb.dtype == np.uint8, (rgb.shape, rgb.dtype)
np.savez(a.out, rgb=rgb, ids=ids, split=a.split)
print(f'{a.dataset} {a.split}: {len(ids)} frames {rgb.shape} ids {ids[0]}..{ids[-1]} -> {a.out}',
      flush=True)
