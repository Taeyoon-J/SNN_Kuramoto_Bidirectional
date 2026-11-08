"""Tetrominoes / Multi-dSprites in AKOrN's npz format with the truth lifted to 128.

The CLEVR-trained AKOrN runs at the resolution it was trained at (128, psize 8), so the
cross-dataset cells need the evaluation frames at 128 too. `labels` is our grid-64 truth
`np.repeat`-ed x2, exactly as in the CLEVR npz, so AKOrN's own metric and our evaluation
function see the same tensor and no id mapping is involved.
"""
import numpy as np, sys, pathlib
B = pathlib.Path('/work/USERS/tkim1/sw0255b')
sys.path.insert(0, str(B/'benchmark_codes/vendor')); sys.path.insert(0, str(B/'benchmark_codes'))
import portable_runtime as pr
cfg = pr.configure(str(B/'benchmark_codes/configs/frontier.json'))
root = pr.connect_inputs(cfg, str(B/'run'))
from full_benchmark.data import Dataset

OUT = pathlib.Path('/work/USERS/tkim1/baselines/_clone/akorn/data/ours')
for ds, name in (('multi_dsprites', 'dsprites'), ('tetrominoes', 'tetrominoes')):
    d = Dataset(ds, root)
    for split, ids, s in (('val128', np.asarray(d.validation), 'validation'),
                          ('eval320_128', np.asarray(d.test)[:320], 'test')):
        rgb = np.asarray(d.rgb(ids, s))
        lab = d.labels(ids, s)
        lab128 = np.repeat(np.repeat(lab, 2, axis=1), 2, axis=2)
        assert lab128.shape[1:] == rgb.shape[1:3], (lab128.shape, rgb.shape)
        images = (rgb.astype(np.float32)/255.).transpose(0, 3, 1, 2)
        f = OUT/f'{name}_{split}.npz'
        np.savez_compressed(f, images=images, labels=lab128.astype(np.int32),
                            ids=ids.astype(np.int64))
        print(f'{name}_{split}: {images.shape} labels {lab128.shape} '
              f'ids {ids[0]}..{ids[-1]} -> {f}', flush=True)
