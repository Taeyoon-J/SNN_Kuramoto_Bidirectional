#!/usr/bin/env python3
"""Dump the TRAIN frames Slot Attention will be retrained on -- the identical ids our core uses.

The published checkpoint was trained on the original CLEVR render, not clevr_with_masks, and on
our render it tiles rather than segments: 6.99 of 7 slots active on every frame with near-equal
shares. Beating a model that is failing to transfer is not beating the model, so the reference
has to be retrained on our images.

"Our images" is meant exactly. This writes the same 10,000 TRAIN ids the 253 core trains on,
drawn by the same `RandomState(2553500)` subset of the same 35,000 parent pool, as the same
128x128 frames from `dataset.rgb`. Slot Attention then sees neither more nor less data than we
do, and the only thing that differs between the two columns is the model.

The torch environment has to do this because `Dataset` needs torch and the TensorFlow
environment has none; `sa_train.py` reads the npy this writes.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--code', required=True); ap.add_argument('--config', required=True)
ap.add_argument('--run', required=True); ap.add_argument('--dataset', default='clevr')
ap.add_argument('--split', default='train', choices=['train', 'validation', 'test'])
ap.add_argument('--out', required=True)
a = ap.parse_args()
CODE = Path(a.code); sys.path.insert(0, str(CODE/'vendor')); sys.path.insert(0, str(CODE))
import portable_runtime as pr
config = pr.configure(a.config); output = pr.connect_inputs(config, a.run)
from full_benchmark.data import Dataset

dataset = Dataset(a.dataset, output)
ids = np.asarray({'train': dataset.train, 'validation': dataset.validation,
                  'test': dataset.test}[a.split])
print(f'{a.dataset} {a.split}: {len(ids)} ids', flush=True)
# written incrementally: 10,000 frames at 128x128x3 is 491 MB, which is fine on disk but
# should not be assembled twice in memory
out = Path(a.out)
frames = np.lib.format.open_memmap(out, mode='w+', dtype=np.uint8,
                                   shape=(len(ids), 128, 128, 3))
step = 500
for i in range(0, len(ids), step):
    chunk = np.asarray(dataset.rgb(ids[i:i+step], a.split))
    assert chunk.shape[1:] == (128, 128, 3) and chunk.dtype == np.uint8, (chunk.shape, chunk.dtype)
    frames[i:i+len(chunk)] = chunk
    if (i//step) % 4 == 0: print(f'  {min(i+step, len(ids))}/{len(ids)}', flush=True)
frames.flush(); del frames
out.with_suffix('.ids.json').write_text(json.dumps(
    dict(dataset=a.dataset, split=a.split, count=len(ids), ids=ids.tolist(),
         note='the identical ids the 253 core trains on; same subset_seed 2553500, '
              'same 35,000 parent pool'), indent=1)+'\n')
print(f'wrote {out} and {out.with_suffix(".ids.json")}', flush=True)
