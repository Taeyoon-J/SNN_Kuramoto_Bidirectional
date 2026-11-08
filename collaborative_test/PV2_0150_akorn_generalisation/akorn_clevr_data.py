"""Build AKOrN's npz format for CLEVR from our own splits.

AKOrN's CLEVR loader is the same NumpyDataset as dsprites/tetrominoes, reading
`./data/clevr_with_masks/clevr_{split}.npz` with `images` (N,C,H,W) float in [0,1] and
`labels` pixelwise instance ids.

Two deliberate choices, both so that AKOrN's row and ours differ only by the model:

  * train ids are the identical 10,000 our 253 core trained on, and the evaluation ids are
    the identical reserved 320 our rows are scored on.
  * `labels` is OUR grid-64 truth, `np.repeat`-ed x2 to 128. It is not the raw h5 pixel mask.
    That makes the tensor AKOrN scores itself against the very tensor our evaluation function
    uses, with no id-mapping question between the h5 rows and our split manifest, and it costs
    AKOrN nothing: the grid is the contract's truth and the x2 lift is exact.
"""
import numpy as np, sys, pathlib
B = pathlib.Path('/work/USERS/tkim1/sw0255b')
sys.path.insert(0, str(B/'benchmark_codes/vendor')); sys.path.insert(0, str(B/'benchmark_codes'))
import portable_runtime as pr
cfg = pr.configure(str(B/'benchmark_codes/configs/frontier.json'))
root = pr.connect_inputs(cfg, str(B/'run'))
from full_benchmark.data import Dataset

OUT = pathlib.Path('/work/USERS/tkim1/baselines/_clone/akorn/data/clevr_with_masks')
OUT.mkdir(parents=True, exist_ok=True)
d = Dataset('clevr', root)
first = np.sort(np.random.RandomState(2553500).choice(d.train, 10000, replace=False))
SPLITS = (('train', first, 'train'),
          ('val', np.asarray(d.validation), 'validation'),
          ('eval320', np.asarray(d.test)[:320], 'test'))
for name, ids, split in SPLITS:
    rgb = np.asarray(d.rgb(ids, split))                       # (N,128,128,3) uint8
    lab = d.labels(ids, split)                                # (N,64,64) grid truth
    lab128 = np.repeat(np.repeat(lab, 2, axis=1), 2, axis=2)   # exact x2 lift
    assert lab128.shape[1:] == rgb.shape[1:3], (lab128.shape, rgb.shape)
    images = (rgb.astype(np.float32)/255.).transpose(0, 3, 1, 2)
    f = OUT/f'clevr_{name}.npz'
    np.savez_compressed(f, images=images, labels=lab128.astype(np.int32),
                        ids=ids.astype(np.int64))
    print(f'clevr_{name}: images {images.shape} labels {lab128.shape} '
          f'ids {ids[0]}..{ids[-1]} n={len(ids)} -> {f.stat().st_size/2**20:.0f} MB', flush=True)
