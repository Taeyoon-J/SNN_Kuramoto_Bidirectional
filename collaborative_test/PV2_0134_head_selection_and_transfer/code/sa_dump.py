#!/usr/bin/env python3
"""Dump the exact frames and ids of a split, so Slot Attention can be run in its own env.

The TensorFlow environment here has no torch, and our Dataset needs it, so the two halves
cannot share a process. This half writes the frames; `sa_infer.py` reads them.
"""
import argparse, sys
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--code', required=True); ap.add_argument('--config', required=True)
ap.add_argument('--run', required=True); ap.add_argument('--split', required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()
CODE = Path(a.code); sys.path.insert(0, str(CODE/'vendor')); sys.path.insert(0, str(CODE))
import portable_runtime as pr
config = pr.configure(a.config); output = pr.connect_inputs(config, a.run)
from full_benchmark.data import Dataset
dataset = Dataset('clevr', output)
ids = np.asarray(dataset.test if a.split == 'fixed320' else dataset.validation)
rgb = np.asarray(dataset.rgb(ids))
assert rgb.shape[1:] == (128, 128, 3) and rgb.dtype == np.uint8, (rgb.shape, rgb.dtype)
np.savez(a.out, rgb=rgb, ids=ids, split=a.split)
print(f'{a.split}: {len(ids)} frames {rgb.shape} -> {a.out}', flush=True)
