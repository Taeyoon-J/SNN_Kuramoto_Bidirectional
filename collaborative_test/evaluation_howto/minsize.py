#!/usr/bin/env python3
"""How much of the object-count error is single-patch regions, and what does dropping them cost?

Operates on cached regions_253 only -- no GPU, no rollout. For each k, every predicted region
with fewer than k patches is reassigned to background, and the three contract metrics plus
object-count accuracy are recomputed on exactly the same masks otherwise.
"""
import argparse, sys
from pathlib import Path
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument('--code', required=True); ap.add_argument('--config', required=True)
ap.add_argument('--run', required=True); ap.add_argument('--cache', required=True)
ap.add_argument('--dataset', required=True); ap.add_argument('--split', required=True)
ap.add_argument('--ks', default='1,2,3,4,6,8,12,16,24,32,48,64')
a = ap.parse_args()
CODE = Path(a.code); sys.path.insert(0, str(CODE/'vendor')); sys.path.insert(0, str(CODE))
import portable_runtime as pr
config = pr.configure(a.config); output = pr.connect_inputs(config, a.run)
from full_benchmark.data import Dataset
from mod_benchmark import common as c
metric = c.benchmark_module('evaluation')

ds = Dataset(a.dataset, output)
rows = []
for f in sorted(Path(a.cache).glob('*.pt')):
    d = torch.load(f, map_location='cpu', weights_only=False)
    if 'image_id' not in d: continue
    rows.append((int(d['image_id']), d['regions_253'].numpy().astype(np.int64)))
rows.sort(key=lambda t: t[0])
ids = np.array([r[0] for r in rows], dtype=np.int64)
pred = np.stack([r[1] for r in rows])
tgt = torch.from_numpy(ds.labels(ids, a.split))
tn = tgt.numpy()

# --- the observation: how big are the predicted regions?
sizes = []
for p in pred:
    for l in np.unique(p):
        if l > 0: sizes.append(int((p == l).sum()))
sizes = np.array(sizes)
print(f'{a.dataset}/{a.split}: {len(ids)} images, {len(sizes)} predicted regions total')
print(f'  regions per image      : {len(sizes)/len(ids):.2f}')
for thr in (1, 2, 4, 8, 16, 32):
    n = int((sizes < thr).sum()) if thr > 1 else int((sizes == 1).sum())
    lab = f'== 1 patch' if thr == 1 else f'<  {thr} patches'
    print(f'  regions {lab:14s}: {n:5d}  ({100.0*n/len(sizes):5.1f}% of regions)')
print(f'  median region size     : {np.median(sizes):.0f} patches')

def objcount(p):
    return np.array([len(np.unique(x[x > 0])) for x in p])
t_cnt = objcount(tn)

def drop(p, k):
    if k <= 1: return p
    out = p.copy()
    for i in range(len(out)):
        x = out[i]
        lab, cnt = np.unique(x, return_counts=True)
        for l, n in zip(lab, cnt):
            if l > 0 and n < k: x[x == l] = 0
    return out

print(f'\n  truth objects/image {t_cnt.mean():.2f}')
print(f'  {"k":>4} {"fg_ari":>8} {"fg_iou":>8} {"obj_iou":>8} {"pred_cnt":>9} {"exact":>7} {"MAE":>6}')
for k in [int(x) for x in a.ks.split(',')]:
    p = drop(pred, k)
    s = metric.evaluate_patch_masks(torch.from_numpy(p), tgt)['mean']
    pc = objcount(p)
    print(f'  {k:>4} {float(s["fg_ari"]):>8.4f} {float(s["foreground_iou"]):>8.4f} '
          f'{float(s["matched_object_iou"]):>8.4f} {pc.mean():>9.2f} '
          f'{float((pc==t_cnt).mean()):>7.3f} {float(np.abs(pc-t_cnt).mean()):>6.2f}')
