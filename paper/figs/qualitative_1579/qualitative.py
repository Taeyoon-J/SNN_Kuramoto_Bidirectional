#!/usr/bin/env python3
"""Qualitative masks from the 253 (KAIROS) core on CLEVR.

Reads the cached regions_253 for a split, applies the published readout rule
(fgrel, f = 0.01, on top of the 8-patch component floor already baked into the cache),
scores every image with the protected evaluator, picks the best one by FG-ARI, and writes:

    01_original.png          the 128x128 RGB
    02_masks_only.png        the predicted partition alone, one colour per object
    03_overlay_all.png       RGB with every mask overlaid
    04_overlay_<k>.png       RGB with exactly one mask overlaid, one file per object

No ground truth enters the masks. Ground truth is used only to score, and therefore only
to choose which image to show -- that selection makes this a hand-picked example.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument('--code', required=True); ap.add_argument('--config', required=True)
ap.add_argument('--run', required=True); ap.add_argument('--cache', required=True)
ap.add_argument('--dataset', default='clevr'); ap.add_argument('--split', default='fixed320')
ap.add_argument('--f', type=float, default=0.01)
ap.add_argument('--out', required=True)
ap.add_argument('--rank', type=int, default=0, help='0 = best image, 1 = second best, ...')
ap.add_argument('--select', default='obj_iou', choices=['obj_iou', 'fg_ari'],
                help='obj_iou ranks by per-object mask overlap, which is what a mask figure shows; '
                     'fg_ari can read 1.0 on an image where a whole object was dropped, because '
                     'predicted background counts as one more cluster')
ap.add_argument('--match-count', action='store_true',
                help='only consider images where the predicted object count equals the true count')
ap.add_argument('--scale', type=int, default=4, help='nearest-neighbour display scale of the saved PNGs')
a = ap.parse_args()

CODE = Path(a.code); sys.path.insert(0, str(CODE/'vendor')); sys.path.insert(0, str(CODE))
import portable_runtime as pr
config = pr.configure(a.config); output = pr.connect_inputs(config, a.run)
from full_benchmark.data import Dataset
from mod_benchmark import common as c
metric = c.benchmark_module('evaluation')

# ---------------------------------------------------------------- cached predictions
rows = []
for f in sorted(Path(a.cache).glob('[0-9]*.pt')):
    d = torch.load(f, map_location='cpu', weights_only=False)
    if 'image_id' not in d: continue
    rows.append((int(d['image_id']), d['regions_253'].numpy().astype(np.int64)))
rows.sort(key=lambda t: t[0])
ids = np.array([r[0] for r in rows], dtype=np.int64)
pred = np.stack([r[1] for r in rows])
print(f'{len(ids)} cached predictions, grid {pred.shape[1]}x{pred.shape[2]}')

# ------------------------------------------- the published rule: fgrel at f = 0.01
def fgrel(p, f):
    if f <= 0: return p
    out = p.copy()
    for i in range(len(out)):
        x = out[i]
        lab, cnt = np.unique(x, return_counts=True)
        keep = lab > 0
        lab, cnt = lab[keep], cnt[keep]
        if len(cnt) == 0: continue
        thr = f * cnt.sum()
        for l, n in zip(lab, cnt):
            if n < thr: x[x == l] = 0
    return out

pred = fgrel(pred, a.f)

# ------------------------------------------------------------------------ scoring
ds = Dataset(a.dataset, output)
target = torch.from_numpy(ds.labels(ids, a.split))
scores = metric.evaluate_patch_masks(torch.from_numpy(pred), target)
per = {k: np.asarray(v, dtype=np.float64) for k, v in scores['per_image'].items()}
print('means:', {k: round(float(np.nanmean(v)), 4) for k, v in per.items()})

p_cnt = np.array([len(np.unique(x[x > 0])) for x in pred])
t_cnt = np.array([len(np.unique(x[x > 0])) for x in target.numpy()])
primary = 'matched_object_iou' if a.select == 'obj_iou' else 'fg_ari'
secondary = 'fg_ari' if a.select == 'obj_iou' else 'matched_object_iou'
key = np.nan_to_num(per[primary], nan=-1).copy()
if a.match_count: key[p_cnt != t_cnt] = -1
order = np.lexsort((np.nan_to_num(per[secondary], nan=-1), key))[::-1]
print(f'\ntop candidates by {primary}' + (' (count-matched only)' if a.match_count else ''))
print(f'  {"rank":>4} {"image":>6} {"fg_ari":>8} {"fg_iou":>8} {"obj_iou":>8} {"pred":>5} {"true":>5}')
for r in range(min(8, len(order))):
    j = int(order[r])
    print(f'  {r:>4} {int(ids[j]):>6} {per["fg_ari"][j]:>8.4f} {per["foreground_iou"][j]:>8.4f} '
          f'{per["matched_object_iou"][j]:>8.4f} {p_cnt[j]:>5} {t_cnt[j]:>5}')
pick = int(order[a.rank])
image_id = int(ids[pick])
labels = pred[pick]
truth = target.numpy()[pick]
print(f'rank {a.rank}: image_id {image_id}  '
      f'fg_ari {per["fg_ari"][pick]:.4f}  fg_iou {per["foreground_iou"][pick]:.4f}  '
      f'obj_iou {per["matched_object_iou"][pick]:.4f}  '
      f'predicted objects {len(np.unique(labels[labels > 0]))}  '
      f'true objects {len(np.unique(truth[truth > 0]))}')

# ------------------------------------------------------------------------ rendering
rgb = ds.rgb(np.array([image_id]), a.split)[0]
if rgb.dtype != np.uint8: rgb = np.clip(rgb, 0, 255).astype(np.uint8)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

# a fixed, colour-blind-safe qualitative palette; background stays black / unchanged
PALETTE = np.array([
    [ 68, 119, 170], [238, 102, 119], [ 34, 136,  51], [204, 187,  68],
    [170,  51, 119], [102, 204, 238], [187, 187, 187], [238, 119,  51],
    [  0, 153, 136], [136,  34,  85], [ 51,  34, 136], [221, 204, 119],
], dtype=np.uint8)

def upsample(x, k):
    return np.repeat(np.repeat(x, k, axis=0), k, axis=1)

GRID_TO_PIXEL = rgb.shape[0] // labels.shape[0]      # 128 / 64 = 2, an exact repeat

def save(path, array):
    from PIL import Image
    Image.fromarray(array).save(path)
    print('wrote', path)

object_ids = [int(v) for v in np.unique(labels) if v > 0]
object_ids.sort(key=lambda v: -int((labels == v).sum()))      # largest object first
colour_of = {v: PALETTE[i % len(PALETTE)] for i, v in enumerate(object_ids)}

# 1. original
save(out/'01_original.png', upsample(rgb, a.scale))

# 2. masks only, on black
canvas = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
for v in object_ids: canvas[labels == v] = colour_of[v]
save(out/'02_masks_only.png', upsample(canvas, GRID_TO_PIXEL * a.scale))

# 3. every mask over the image
def overlay(base, selected, alpha=0.55):
    mask = upsample(np.isin(labels, selected).astype(np.uint8), GRID_TO_PIXEL).astype(bool)
    tint = np.zeros_like(base)
    for v in selected:
        tint[upsample((labels == v).astype(np.uint8), GRID_TO_PIXEL).astype(bool)] = colour_of[v]
    blended = base.astype(np.float32).copy()
    blended[mask] = (1 - alpha) * blended[mask] + alpha * tint[mask].astype(np.float32)
    return upsample(np.clip(blended, 0, 255).astype(np.uint8), a.scale)

save(out/'03_overlay_all.png', overlay(rgb, object_ids))

# 4. one mask at a time
for n, v in enumerate(object_ids, start=1):
    save(out/f'04_overlay_{n:02d}.png', overlay(rgb, [v]))

(out/'selection.json').write_text(json.dumps({
    'dataset': a.dataset, 'split': a.split, 'image_id': image_id, 'rank': a.rank,
    'min_region_size_f': a.f, 'component_floor_min_group': 8,
    'fg_ari': float(per['fg_ari'][pick]),
    'foreground_iou': float(per['foreground_iou'][pick]),
    'matched_object_iou': float(per['matched_object_iou'][pick]),
    'predicted_objects': len(object_ids),
    'true_objects': int(len(np.unique(truth[truth > 0]))),
    'split_means': {k: float(np.nanmean(v)) for k, v in per.items()},
    'note': 'masks use no ground truth; ground truth scored the images and thus chose this one',
}, indent=2) + '\n')
print('wrote', out/'selection.json')
