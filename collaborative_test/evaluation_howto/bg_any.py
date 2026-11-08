#!/usr/bin/env python3
"""Choose any clustering baseline's background rule on validation, then report it on fixed320.

Same four rules, the same selection procedure and the same evaluation function as sa_bg.py, but
for any model whose readout is a hard partition with arbitrary ids: pass two npz files holding
`preds` [N,G,G], with the ids either inside the npz as `ids` or taken from the source data npz
via --val-ids/--eval-ids. The rule is picked on validation FG-ARI with ties broken on
Foreground IoU, exactly as the Slot Attention row was picked, so a baseline's background
designation is never tuned on the reserved split.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--val', required=True); ap.add_argument('--eval', required=True)
ap.add_argument('--val-ids', default=None); ap.add_argument('--eval-ids', default=None)
ap.add_argument('--dataset', required=True)
ap.add_argument('--code', required=True); ap.add_argument('--config', required=True)
ap.add_argument('--run', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--key', default='preds')
a = ap.parse_args()
CODE = Path(a.code); sys.path.insert(0, str(CODE/'vendor')); sys.path.insert(0, str(CODE))
import portable_runtime as pr
config = pr.configure(a.config); output = pr.connect_inputs(config, a.run)
from full_benchmark.data import Dataset
from mod_benchmark import common as c
dataset = Dataset(a.dataset, output); metric = c.benchmark_module('evaluation')
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
        if rule == 'image_largest':
            bg.append({int(size.argmax())})
        elif rule == 'border_largest':
            bg.append({int(np.bincount(s[b], minlength=K).argmax())})
        elif rule == 'border_union':
            on = np.bincount(s[b], minlength=K)
            pick = {int(k) for k in range(K) if size[k] and on[k] > .5*size[k]}
            bg.append(pick or {int(size.argmax())})
    return relabel(slots, bg, K)


def score(pred, ids, split):
    s = metric.evaluate_patch_masks(torch.from_numpy(pred.astype(np.int64)),
                                    torch.from_numpy(dataset.labels(ids, split)))
    row = {k: float(s['mean'][k]) for k in KEYS}
    if 'valid_count' in s:
        row['valid_count'] = {k: int(s['valid_count'][k]) for k in KEYS}
    row['predicted_foreground'] = float((pred > 0).mean())
    return row


def ids_of(blob, src):
    if src: return np.load(src)['ids'].astype(np.int64)
    return blob['ids'].astype(np.int64)


RULES = ('global_largest', 'image_largest', 'border_largest', 'border_union')
val, ev = np.load(a.val), np.load(a.eval)
vp, ep = val[a.key].astype(np.int64), ev[a.key].astype(np.int64)
vi, ei = ids_of(val, a.val_ids), ids_of(ev, a.eval_ids)
assert len(vp) == len(vi) and len(ep) == len(ei), (vp.shape, vi.shape, ep.shape, ei.shape)
K = int(max(vp.max(), ep.max())) + 1
print(f'dataset {a.dataset}  K {K}  val {vp.shape} ids {vi[0]}..{vi[-1]}  '
      f'eval {ep.shape} ids {ei[0]}..{ei[-1]}')
tv = float((dataset.labels(vi, 'validation') > 0).mean())
te = float((dataset.labels(ei, 'fixed320') > 0).mean())
print(f'truth foreground: validation {tv:.4f}   fixed320 {te:.4f}\n')
report = {}
for rule in RULES:
    v = score(apply_rule(vp, K, rule), vi, 'validation')
    report[rule] = dict(validation=v)
    print(f"  {rule:15s} val fg_ari {v['fg_ari']:.4f}  fg_iou {v['foreground_iou']:.4f}  "
          f"obj_iou {v['matched_object_iou']:.4f}  predicted fg {v['predicted_foreground']:.4f}")
pick = max(RULES, key=lambda r: (round(report[r]['validation']['fg_ari'], 6),
                                 report[r]['validation']['foreground_iou']))
e = score(apply_rule(ep, K, pick), ei, 'fixed320')
report[pick]['fixed320'] = e
print(f'\nselected on validation FG-ARI, ties broken on Foreground IoU: {pick}')
print(f"fixed320: fg_ari {e['fg_ari']:.4f}  fg_iou {e['foreground_iou']:.4f}  "
      f"obj_iou {e['matched_object_iou']:.4f}  predicted fg {e['predicted_foreground']:.4f}")
print('\ndiagnostic -- every rule on fixed320 (not used for selection):')
for rule in RULES:
    d = score(apply_rule(ep, K, rule), ei, 'fixed320')
    report[rule]['fixed320_diagnostic'] = d
    print(f"  {rule:15s} {d['fg_ari']:.4f}  {d['foreground_iou']:.4f}  {d['matched_object_iou']:.4f}  "
          f"predicted fg {d['predicted_foreground']:.4f}")
Path(a.out).write_text(json.dumps(dict(dataset=a.dataset, selected=pick, rules=report, K=K,
    truth_foreground=dict(validation=tv, fixed320=te)), indent=1)+'\n')
print('\nBG_ANY DONE ->', a.out, flush=True)
