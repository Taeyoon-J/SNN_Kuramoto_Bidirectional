"""Write our CLEVR splits into the EMORL h5 layout that CtCAE / CAE++ / CAE read.

`ctcae/data.py` documents the layout it expects:

    {train|val|test}/
        imgs/   [N, H, W, C]
        masks/  [N, num_objects, H, W, C]      masks[0] is background

and `__getitem__` takes `masks[...,0]`, resizes each object plane to the dataset resolution,
then `argmax`es over the object axis to get its labels. So the truth it scores itself against is
whatever we put in `masks`.

Two deliberate choices, both to keep its pipeline from resampling anything that matters:

  * `imgs` are our full 128x128 uint8 frames. Its own loader downsamples them to the model
    resolution with its own filter, exactly as it does for its own data -- we do not pre-shrink
    the input and so do not choose that filter for it.
  * `masks` are written at **64x64**, built from `Dataset.labels` -- our grid. The loader's
    resize is then an identity (64 -> 64), so the truth reaches the model and its metric
    bit-exact, with no interpolation of label planes. This also makes its predictions come out
    at 64x64, which is our grid, so nothing is resampled on the way back either.

Only CLEVR is written, because only CLEVR is trained on. Zero-shot evaluation on Tetrominoes
and Multi-dSprites goes through our own `Dataset` in the eval script and needs no h5.
"""
import numpy as np, sys, pathlib, h5py
B = pathlib.Path('/work/USERS/tkim1/sw0255b')
sys.path.insert(0, str(B/'benchmark_codes/vendor')); sys.path.insert(0, str(B/'benchmark_codes'))
import portable_runtime as pr
cfg = pr.configure(str(B/'benchmark_codes/configs/frontier.json'))
root = pr.connect_inputs(cfg, str(B/'run'))
from full_benchmark.data import Dataset

OUT = pathlib.Path('/work/USERS/tkim1/baselines/_clone/ctcae/data')
OUT.mkdir(parents=True, exist_ok=True)
K = 11                      # CLEVR labels run 0..10, so 11 planes with background first

d = Dataset('clevr', root)
train_ids = np.sort(np.random.RandomState(2553500).choice(d.train, 10000, replace=False))
SPLITS = (('clevr6_with_masks_train.h5', 'train', train_ids, 'train'),
          ('clevr6_with_masks_and_factors_test.h5', 'test', np.asarray(d.test)[:320], 'test'),
          ('ours_clevr_val.h5', 'test', np.asarray(d.validation), 'validation'))

for fname, group, ids, split in SPLITS:
    rgb = np.asarray(d.rgb(ids, split)).astype(np.uint8)         # (N,128,128,3)
    lab = d.labels(ids, split)                                   # (N,64,64) our grid
    assert lab.max() < K, (lab.max(), K)
    n, g = lab.shape[0], lab.shape[1]
    # one-hot over the object axis, background first, 0/255 so their `masks[1:]/255` is 0/1
    masks = np.zeros((n, K, g, g, 1), np.uint8)
    for k in range(K):
        masks[:, k, :, :, 0] = (lab == k).astype(np.uint8) * 255
    f = OUT/fname
    with h5py.File(f, 'w') as h:
        gp = h.create_group(group)
        gp.create_dataset('imgs', data=rgb, compression='gzip', compression_opts=1)
        gp.create_dataset('masks', data=masks, compression='gzip', compression_opts=1)
        gp.create_dataset('ids', data=ids.astype(np.int64))
    # the argmax their loader will compute must equal our labels exactly
    back = masks[:, :, :, :, 0].argmax(axis=1)
    assert (back == lab).all(), 'round-trip through the one-hot changed the labels'
    print(f'{fname} [{group}]: imgs {rgb.shape} masks {masks.shape} ids {ids[0]}..{ids[-1]} '
          f'-> {f.stat().st_size/2**20:.0f} MB  argmax round-trip exact', flush=True)
