#!/usr/bin/env python3
"""Scale-free versions of the minimum-region-size rule.

The absolute rule needs a different k per dataset (CLEVR ~4, Tetrominoes ~48, dSprites ~8),
which is three hyperparameters and weakens a generality claim. These two rules are one
hyperparameter each and adapt to the image:

    maxrel : drop a region smaller than f * (largest predicted region in that image)
    fgrel  : drop a region smaller than f * (total predicted foreground in that image)

Cached regions_253 only -- no GPU, no rollout, no ground truth is used to set anything.
"""
import argparse, sys
from pathlib import Path
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument('--code', required=True); ap.add_argument('--config', required=True)
ap.add_argument('--run', required=True); ap.add_argument('--cache', required=True)
ap.add_argument('--dataset', required=True); ap.add_argument('--split', required=True)
ap.add_argument('--rule', default='both', choices=['maxrel', 'fgrel', 'both'])
ap.add_argument('--fs', default='0,0.01,0.02,0.03,0.05,0.08,0.10,0.15,0.20,0.30')
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
tgt = torch.from_numpy(ds.labels(ids, a.split)); tn = tgt.numpy()
t_cnt = np.array([len(np.unique(x[x > 0])) for x in tn])

def apply_rule(p, rule, f):
    if f <= 0: return p
    out = p.copy()
    for i in range(len(out)):
        x = out[i]
        lab, cnt = np.unique(x, return_counts=True)
        keep = lab > 0
        lab, cnt = lab[keep], cnt[keep]
        if len(cnt) == 0: continue
        thr = f * (cnt.max() if rule == 'maxrel' else cnt.sum())
        for l, n in zip(lab, cnt):
            if n < thr: x[x == l] = 0
    return out

print(f'{a.dataset}/{a.split}: {len(ids)} images, truth objects/image {t_cnt.mean():.2f}')
for rule in (['maxrel', 'fgrel'] if a.rule == 'both' else [a.rule]):
    print(f'\n  rule = {rule}')
    print(f'  {"f":>6} {"fg_ari":>8} {"fg_iou":>8} {"obj_iou":>8} {"pred_cnt":>9} {"exact":>7} {"MAE":>6}')
    for f in [float(x) for x in a.fs.split(',')]:
        p = apply_rule(pred, rule, f)
        s = metric.evaluate_patch_masks(torch.from_numpy(p), tgt)['mean']
        pc = np.array([len(np.unique(x[x > 0])) for x in p])
        print(f'  {f:>6.2f} {float(s["fg_ari"]):>8.4f} {float(s["foreground_iou"]):>8.4f} '
              f'{float(s["matched_object_iou"]):>8.4f} {pc.mean():>9.2f} '
              f'{float((pc==t_cnt).mean()):>7.3f} {float(np.abs(pc-t_cnt).mean()):>6.2f}')
