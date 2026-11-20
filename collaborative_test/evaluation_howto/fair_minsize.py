#!/usr/bin/env python3
"""Fairness check: give the baselines the same minimum-region-size rule we would adopt.

If a post-processing rule helps us, it must be offered to every baseline before any margin
is claimed. Background rule is selected per baseline on validation FG-ARI exactly as
bg_any.py does (tie-break Foreground IoU), then the size rule is swept on the same split.
Ground truth is never used to set either. Everything here is validation.
"""
import argparse, sys
from pathlib import Path
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument('--code', required=True); ap.add_argument('--config', required=True)
ap.add_argument('--run', required=True); ap.add_argument('--dataset', required=True)
ap.add_argument('--split', default='validation')
ap.add_argument('--ids-from', required=True, help='npz carrying the id order for this split')
ap.add_argument('--models', required=True, help='name=path,name=path,...')
ap.add_argument('--fs', default='0,0.01,0.02,0.03,0.05,0.08,0.10')
ap.add_argument('--key', default='preds')
a = ap.parse_args()
CODE = Path(a.code); sys.path.insert(0, str(CODE/'vendor')); sys.path.insert(0, str(CODE))
import portable_runtime as pr
config = pr.configure(a.config); output = pr.connect_inputs(config, a.run)
from full_benchmark.data import Dataset
from mod_benchmark import common as c
metric = c.benchmark_module('evaluation')
dataset = Dataset(a.dataset, output)
KEYS = ('fg_ari', 'foreground_iou', 'matched_object_iou')

def border_mask(g):
    m = np.zeros((g, g), bool); m[0] = m[-1] = True; m[:, 0] = m[:, -1] = True
    return m

def relabel(slots, background_ids, K):
    out = np.zeros_like(slots)
    for i, (s, bg) in enumerate(zip(slots, background_ids)):
        nxt = 1
        for k in range(K):
            if k in bg: continue
            hit = s == k
            if hit.any(): out[i][hit] = nxt; nxt += 1
    return out

def apply_rule(slots, K, rule):
    g = slots.shape[-1]; b = border_mask(g); bg = []
    if rule == 'global_largest':
        k = np.bincount(slots.reshape(-1), minlength=K).argmax()
        return relabel(slots, [{int(k)}]*len(slots), K)
    for s in slots:
        size = np.bincount(s.reshape(-1), minlength=K)
        if rule == 'image_largest':   bg.append({int(size.argmax())})
        elif rule == 'border_largest': bg.append({int(np.bincount(s[b], minlength=K).argmax())})
        elif rule == 'border_union':
            on = np.bincount(s[b], minlength=K)
            pick = {int(k) for k in range(K) if size[k] and on[k] > .5*size[k]}
            bg.append(pick or {int(size.argmax())})
    return relabel(slots, bg, K)

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

ids = np.load(a.ids_from)['ids'].astype(np.int64)
tgt = torch.from_numpy(dataset.labels(ids, a.split)); tn = tgt.numpy()
t_cnt = np.array([len(np.unique(x[x > 0])) for x in tn])
def sc(p):
    s = metric.evaluate_patch_masks(torch.from_numpy(p.astype(np.int64)), tgt)['mean']
    return tuple(float(s[k]) for k in KEYS)

print(f'{a.dataset}/{a.split}: {len(ids)} images, truth objects/image {t_cnt.mean():.2f}\n')
for spec in a.models.split(','):
    name, path = spec.split('=')
    pr_ = np.load(path)[a.key].astype(np.int64)
    # Baseline dumps differ in length (AKOrN cached 1000 validation images, the CAE family
    # 320). They were all written in dataset.validation order, so truncating to the shortest
    # compares the same images. A model with FEWER than len(ids) cannot be aligned and is
    # skipped rather than silently mismatched.
    if len(pr_) < len(ids):
        print(f"{name}: only {len(pr_)} images, need {len(ids)} -- SKIPPED\n"); continue
    pr_ = pr_[:len(ids)]
    K = int(pr_.max()) + 1
    best = max((('x',), ), default=None)
    bestrule, bestv = None, None
    for rule in ('global_largest', 'image_largest', 'border_largest', 'border_union'):
        v = sc(apply_rule(pr_, K, rule))
        if bestv is None or (v[0], v[1]) > (bestv[0], bestv[1]): bestrule, bestv = rule, v
    lab = apply_rule(pr_, K, bestrule)
    print(f'{name}  (K={K}, background rule chosen on validation: {bestrule})')
    print(f'  {"f":>6} {"fg_ari":>8} {"fg_iou":>8} {"obj_iou":>8} {"pred_cnt":>9} {"exact":>7} {"MAE":>6}')
    for f in [float(x) for x in a.fs.split(',')]:
        p = fgrel(lab, f); v = sc(p)
        pc = np.array([len(np.unique(x[x > 0])) for x in p])
        print(f'  {f:>6.2f} {v[0]:>8.4f} {v[1]:>8.4f} {v[2]:>8.4f} {pc.mean():>9.2f} '
              f'{float((pc==t_cnt).mean()):>7.3f} {float(np.abs(pc-t_cnt).mean()):>6.2f}')
    print()
