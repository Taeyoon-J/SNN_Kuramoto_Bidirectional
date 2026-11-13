#!/usr/bin/env python3
"""Re-render the object-mask panel for the paper's image at a given fgrel f, plus the before.

Same image, same scored mask, same colour assignment -- only the minimum-region-size rule
differs, so the two panels are directly comparable side by side.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ap = argparse.ArgumentParser()
ap.add_argument('--npz', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--fs', default='0,0.01'); ap.add_argument('--dpi', type=int, default=300)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
d = np.load(a.npz)
reg0, truth = d['regions_cached'], d['truth']
n_truth = len(np.unique(truth[truth > 0]))

def fgrel(x, f):
    if f <= 0: return x.copy()
    y = x.copy()
    lab, cnt = np.unique(y, return_counts=True)
    keep = lab > 0; lab, cnt = lab[keep], cnt[keep]
    thr = f * cnt.sum()
    for l, n in zip(lab, cnt):
        if n < thr: y[y == l] = 0
    return y

# One colour per region of the UNFILTERED mask, so a region keeps its colour after filtering
# and the reader can see exactly which pieces were dropped.
labs0 = [int(l) for l in np.unique(reg0) if l > 0]
base = plt.get_cmap('tab10').colors
colour = {l: base[i % 10] for i, l in enumerate(labs0)}

print(f"truth objects {n_truth}; unfiltered regions {len(labs0)}")
for f in [float(x) for x in a.fs.split(',')]:
    r = fgrel(reg0, f)
    present = [l for l in labs0 if (r == l).any()]
    cols = [(0.88, 0.88, 0.90)] + [colour[l] for l in present]
    remap = np.zeros_like(r)
    for i, l in enumerate(present): remap[r == l] = i + 1
    fig, ax = plt.subplots(figsize=(1.6, 1.6))
    ax.imshow(remap, cmap=ListedColormap(cols), interpolation='nearest', vmin=0, vmax=len(present))
    ax.set_axis_off()
    name = 'p_masks.png' if f <= 0 else f'p_masks_f{f:g}.png'
    fig.savefig(out/name, dpi=a.dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    sizes = sorted((int((r == l).sum()) for l in present), reverse=True)
    print(f"  f={f:<5g} regions {len(present):>3}  sizes {sizes}  -> {name}")
