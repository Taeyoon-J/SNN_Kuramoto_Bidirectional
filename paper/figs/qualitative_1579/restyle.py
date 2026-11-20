#!/usr/bin/env python3
"""Re-render the per-object panels so the selected object reads regardless of its colour.

The first version tinted each object with a palette colour, which disappears when the tint
and the object are the same hue (a green tint on the green cylinder). Here the panel keeps
the selected object at full colour and dims everything else, with a coloured outline on the
mask boundary.

Masks are recovered from 02_masks_only.png, whose colours are exactly the palette, so no
recomputation and no model run is needed.
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--dir', required=True)
ap.add_argument('--dim', type=float, default=0.38, help='brightness kept outside the mask')
ap.add_argument('--desaturate', type=float, default=0.75, help='grey mixed in outside the mask')
ap.add_argument('--outline', type=int, default=3)
a = ap.parse_args()

D = Path(a.dir)
rgb = np.asarray(Image.open(D/'01_original.png').convert('RGB')).astype(np.float32)
masks_png = np.asarray(Image.open(D/'02_masks_only.png').convert('RGB'))

colours = [c for c in np.unique(masks_png.reshape(-1, 3), axis=0) if c.any()]
masks = [(masks_png == c).all(axis=-1) for c in colours]
order = np.argsort([-m.sum() for m in masks])            # largest object first, as before
colours = [colours[i] for i in order]
masks = [masks[i] for i in order]
print(f'{len(masks)} objects, sizes {[int(m.sum()) for m in masks]}')

def boundary(mask, width):
    pad = np.pad(mask, width, mode='edge')
    eroded = np.ones_like(mask)
    for dy in range(-width, width + 1):
        for dx in range(-width, width + 1):
            eroded &= pad[width + dy:width + dy + mask.shape[0],
                          width + dx:width + dx + mask.shape[1]]
    return mask & ~eroded

def panel(selected_masks, selected_colours):
    keep = np.zeros(rgb.shape[:2], dtype=bool)
    for m in selected_masks: keep |= m
    grey = rgb.mean(axis=-1, keepdims=True).repeat(3, axis=-1)
    out = (a.dim * ((1 - a.desaturate) * rgb + a.desaturate * grey))
    out[keep] = rgb[keep]
    for m, c in zip(selected_masks, selected_colours):
        out[boundary(m, a.outline)] = np.asarray(c, dtype=np.float32)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

panel(masks, colours).save(D/'03_overlay_all.png')
print('rewrote 03_overlay_all.png')
for n, (m, c) in enumerate(zip(masks, colours), start=1):
    panel([m], [c]).save(D/f'04_overlay_{n:02d}.png')
    print(f'rewrote 04_overlay_{n:02d}.png')
