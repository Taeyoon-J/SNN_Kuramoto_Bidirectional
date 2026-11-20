#!/usr/bin/env python3
"""Compose the qualitative panels of one image into a single row.

    original | masks only | all masks over the image | one mask at a time (one cell each)

Reads the PNGs written by qualitative.py and the metrics in selection.json, and writes
05_row_all.png beside them.
"""
import argparse, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument('--dir', required=True)
ap.add_argument('--cell', type=int, default=512, help='panel size in the output')
ap.add_argument('--gap', type=int, default=18)
ap.add_argument('--out', default='05_row_all.png')
a = ap.parse_args()

D = Path(a.dir)
meta = json.loads((D/'selection.json').read_text())
singles = sorted(D.glob('04_overlay_*.png'))

panels = [(D/'01_original.png', 'Input image'),
          (D/'02_masks_only.png', 'Predicted masks'),
          (D/'03_overlay_all.png', 'All masks on the image')]
panels += [(p, f'Object {i}') for i, p in enumerate(singles, start=1)]

def font(size):
    for path in ('/System/Library/Fonts/Supplemental/Arial.ttf',
                 '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        try: return ImageFont.truetype(path, size)
        except OSError: continue
    return ImageFont.load_default()

label_font, title_font = font(30), font(32)
label_h, title_h, pad = 48, 54, 20
cell, gap = a.cell, a.gap
width = len(panels) * cell + (len(panels) - 1) * gap + 2 * pad
height = title_h + cell + label_h + 2 * pad

sheet = Image.new('RGB', (width, height), 'white')
draw = ImageDraw.Draw(sheet)

title = (f"KAIROS, CLEVR image {meta['image_id']} "
         f"({meta['split']}) — FG-ARI {meta['fg_ari']:.4f} · "
         f"FG-IoU {meta['foreground_iou']:.4f} · Obj-IoU {meta['matched_object_iou']:.4f} · "
         f"{meta['predicted_objects']} predicted / {meta['true_objects']} true objects")
draw.text((pad, pad), title, fill='black', font=title_font)

for n, (path, label) in enumerate(panels):
    x = pad + n * (cell + gap)
    y = pad + title_h
    image = Image.open(path).convert('RGB')
    if image.size != (cell, cell):
        image = image.resize((cell, cell), Image.NEAREST)
    sheet.paste(image, (x, y))
    draw.rectangle([x, y, x + cell - 1, y + cell - 1], outline=(210, 210, 210), width=1)
    w = draw.textlength(label, font=label_font)
    draw.text((x + (cell - w) / 2, y + cell + 10), label, fill='black', font=label_font)

sheet.save(D/a.out)
print(f'wrote {D/a.out}  {sheet.size[0]}x{sheet.size[1]}  ({len(panels)} panels)')
