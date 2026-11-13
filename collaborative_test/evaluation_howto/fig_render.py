#!/usr/bin/env python3
"""Render the four real-data panels of the architecture figure from fig_panels.py's npz.

Every panel is borderless and axis-free: the TikZ figure draws its own frames and labels, so
anything matplotlib adds would be drawn twice.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ap = argparse.ArgumentParser()
ap.add_argument('--npz', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--dpi', type=int, default=300)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
d = np.load(a.npz)
rgb, reg, truth = d['rgb'], d['regions_cached'], d['truth']
q, spikes, settle = d['q'], d['spikes'][0], int(d['settle'])
print(f"image_id {int(d['image_id'])}  fg_ari {float(d['fg_ari']):.4f}  "
      f"truth objects {int(d['n_truth_objects'])}  predicted regions {reg.max()}")

def save(fig, name):
    p = out/name
    fig.savefig(p, dpi=a.dpi, bbox_inches='tight', pad_inches=0, transparent=False)
    plt.close(fig); print(f'  wrote {p}')

flat = reg.reshape(-1)
labs = [l for l in np.unique(reg) if l > 0]
base = plt.get_cmap('tab10').colors

# ---------------------------------------------------------------- 1. the input image
fig, ax = plt.subplots(figsize=(1.6, 1.6))
ax.imshow(rgb if rgb.dtype == np.uint8 else np.clip(rgb, 0, 1)); ax.set_axis_off()
save(fig, 'p_image.png')

# ---------------------------------------------------------------- 2. predicted object masks
cols = [(0.88, 0.88, 0.90)] + [base[i % 10] for i in range(len(labs))]
remap = np.zeros_like(reg)
for i, l in enumerate(labs): remap[reg == l] = i + 1
fig, ax = plt.subplots(figsize=(1.6, 1.6))
ax.imshow(remap, cmap=ListedColormap(cols), interpolation='nearest', vmin=0, vmax=len(labs))
ax.set_axis_off()
save(fig, 'p_masks.png')

# ---------------------------------------------------------------- 3. spike correlation Q
# Q is 4096x4096. Show the foreground patches only, ordered by predicted object, so the
# block structure the merge step reads is what the panel shows.
fgidx = np.where(flat > 0)[0]
fgidx = fgidx[np.argsort(flat[fgidx], kind='stable')]
sel = fgidx[::max(1, len(fgidx)//360)]
fig, ax = plt.subplots(figsize=(1.6, 1.6))
ax.imshow(q[np.ix_(sel, sel)], cmap='GnBu', interpolation='nearest', vmin=0, vmax=1)
for b in np.where(np.diff(flat[sel]) != 0)[0] + 0.5:
    ax.axhline(b, color='white', lw=0.5); ax.axvline(b, color='white', lw=0.5)
ax.set_axis_off()
save(fig, 'p_q.png')

# ---------------------------------------------------------------- 4. spiking dynamics
# NOT a binary raster. s_ia(t) = g_i(t) e_ia(t) is continuous in [0,1], and the firing
# indicator e is 1 in 99.99% of settled steps, so a dot-per-spike plot would be a solid
# block and would misrepresent the trace. This shows s itself: rows are patches grouped by
# predicted object, columns are settled steps. Patches of one object share a band pattern;
# different objects are offset. That is the structure the correlation reads.
rng = np.random.default_rng(0)
rows, sep, per = [], [], 11
for l in labs[:4]:
    idx = np.where(flat == l)[0]
    rows.extend(np.sort(rng.choice(idx, size=min(per, len(idx)), replace=False)).tolist())
    sep.append(len(rows))
win = slice(settle, settle + 200)
fig, ax = plt.subplots(figsize=(2.6, 1.6))
ax.imshow(spikes[np.array(rows)][:, win], aspect="auto", cmap="GnBu",
          vmin=0, vmax=1, interpolation="nearest")
for b in sep[:-1]:
    ax.axhline(b - 0.5, color="white", lw=0.9)
ax.set_axis_off()
save(fig, 'p_spikes.png')
