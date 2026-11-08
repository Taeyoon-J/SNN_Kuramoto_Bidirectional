#!/usr/bin/env python3
"""Reduce a finer-than-grid hard partition to the contract's grid by per-patch majority vote.

The evaluation contract scores on the patch grid and never expands a prediction to pixels. A
prediction coarser than the grid is lifted by an exact `np.repeat`; a prediction finer than the
grid -- which is what AKOrN gives at image resolution 128 -- is reduced the other way, by the
contract's own patch majority vote, ties going to the smaller label id.
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--inp', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--grid', type=int, default=64); ap.add_argument('--key', default='preds')
a = ap.parse_args()

p = np.load(a.inp)[a.key].astype(np.int64)
n, h, w = p.shape
assert h == w and h % a.grid == 0, (p.shape, a.grid)
f = h // a.grid
if f == 1:
    out = p
else:
    K = int(p.max()) + 1
    # [n, G, f, G, f] -> count each label inside every patch, argmax takes the smallest id on a tie
    blocks = p.reshape(n, a.grid, f, a.grid, f).transpose(0, 1, 3, 2, 4).reshape(n, a.grid, a.grid, f*f)
    counts = np.zeros((n, a.grid, a.grid, K), np.int32)
    np.add.at(counts, (np.arange(n)[:, None, None, None], np.arange(a.grid)[None, :, None, None],
                       np.arange(a.grid)[None, None, :, None], blocks), 1)
    out = counts.argmax(-1).astype(np.int64)
np.savez_compressed(a.out, preds=out.astype(np.int32))
print(f'{a.inp} {p.shape} -> {a.out} {out.shape} factor {f} labels 0..{out.max()}')
