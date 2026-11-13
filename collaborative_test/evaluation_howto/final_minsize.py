#!/usr/bin/env python3
"""Apply the fgrel minimum-region-size rule to the reserved split, once, at f fixed on validation.

Selection rule, stated before looking at the reserved split: on each dataset's VALIDATION
split, take the largest f whose FG-ARI loss is <= 0.005 and whose matched-object IoU loss is
<= 0.01 against f = 0. That gives
    (A) f chosen on CLEVR validation only and reused unchanged  -> 0.01 everywhere
    (B) f chosen on each dataset's own validation               -> 0.01 / 0.10 / 0.05
(A) uses no target-domain labels at all; (B) uses each target's validation labels.
No ground truth is read per image by either, and the core is untouched.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument('--code', required=True); ap.add_argument('--config', required=True)
ap.add_argument('--run', required=True); ap.add_argument('--cache', required=True)
ap.add_argument('--dataset', required=True); ap.add_argument('--split', required=True)
ap.add_argument('--fa', type=float, required=True, help='(A) f transferred from CLEVR')
ap.add_argument('--fb', type=float, required=True, help='(B) f chosen on this dataset')
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

def fgrel(p, f):
    if f <= 0: return p
    out = p.copy()
    for i in range(len(out)):
        x = out[i]
        lab, cnt = np.unique(x, return_counts=True)
        keep = lab > 0; lab, cnt = lab[keep], cnt[keep]
        if len(cnt) == 0: continue
        thr = f * cnt.sum()
        for l, n in zip(lab, cnt):
            if n < thr: x[x == l] = 0
    return out

def row(p):
    s = metric.evaluate_patch_masks(torch.from_numpy(p), tgt)['mean']
    pc = np.array([len(np.unique(x[x > 0])) for x in p])
    return dict(fg_ari=float(s['fg_ari']), fg_iou=float(s['foreground_iou']),
                obj_iou=float(s['matched_object_iou']), count=float(pc.mean()),
                exact=float((pc == t_cnt).mean()), mae=float(np.abs(pc - t_cnt).mean()))

base = row(pred); A = row(fgrel(pred, a.fa)); B = row(fgrel(pred, a.fb))
print(f'{a.dataset}/{a.split}  n={len(ids)}  truth objects/image {t_cnt.mean():.2f}')
hdr = f'  {"":<22} {"fg_ari":>8} {"fg_iou":>8} {"obj_iou":>8} {"count":>7} {"exact":>7} {"MAE":>6}'
print(hdr)
for nm, r in (('reported (no rule)', base), (f'(A) f={a.fa:g} from CLEVR', A),
              (f'(B) f={a.fb:g} own valid.', B)):
    print(f'  {nm:<22} {r["fg_ari"]:>8.4f} {r["fg_iou"]:>8.4f} {r["obj_iou"]:>8.4f} '
          f'{r["count"]:>7.2f} {r["exact"]:>7.3f} {r["mae"]:>6.2f}')
for nm, r in (('(A)', A), ('(B)', B)):
    print(f'  delta {nm:<16} {r["fg_ari"]-base["fg_ari"]:>+8.4f} {r["fg_iou"]-base["fg_iou"]:>+8.4f} '
          f'{r["obj_iou"]-base["obj_iou"]:>+8.4f} {r["count"]-base["count"]:>+7.2f} '
          f'{r["exact"]-base["exact"]:>+7.3f} {r["mae"]-base["mae"]:>+6.2f}')
print()
