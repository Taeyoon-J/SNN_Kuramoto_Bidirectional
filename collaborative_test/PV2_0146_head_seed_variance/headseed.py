#!/usr/bin/env python3
"""Is the head's gain stable under head re-initialisation?

Three-seed core means need rented GPUs (51-64 h per seed on frontier against 18.5 h on a
4090). This is the part of the error bar that is free: the core is held fixed and only the
head's initialisation changes, which asks a narrower but real question -- is the reported head
gain reproducible at all, or is it a draw?

This is NOT a substitute for 3-seed core means. It isolates head-training variance and says
nothing about core variance.
"""
import argparse, json, statistics, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit5

ap = argparse.ArgumentParser()
ap.add_argument('--code', required=True); ap.add_argument('--config', required=True)
ap.add_argument('--run', required=True); ap.add_argument('--dataset', required=True)
ap.add_argument('--train-cache', required=True); ap.add_argument('--eval-cache', required=True)
ap.add_argument('--eval-split', required=True); ap.add_argument('--arm', default='local_rgb_trust')
ap.add_argument('--epochs', type=int, default=2); ap.add_argument('--seeds', type=int, default=5)
ap.add_argument('--out', required=True)
a = ap.parse_args()
output = fit5.connect(a)
from full_benchmark.data import Dataset
from mod_benchmark import common as c

train, ev = fit5.load(a.train_cache, 'regions_253'), fit5.load(a.eval_cache, 'regions_253')
dataset = Dataset(a.dataset, output); metric = c.benchmark_module('evaluation')
KEYS = ('fg_ari', 'foreground_iou', 'matched_object_iou')
ids = np.array([r['image_id'] for r in ev])
target = torch.from_numpy(dataset.labels(ids, a.eval_split))
def score(preds):
    s = metric.evaluate_patch_masks(torch.from_numpy(np.stack(preds).astype(np.int64)), target)
    return {k: float(s['mean'][k]) for k in KEYS}

base = score([r['regions'] for r in ev])
print(f"{a.dataset}: core alone {base['fg_ari']:.4f}/{base['foreground_iou']:.4f}/"
      f"{base['matched_object_iou']:.4f}   arm {a.arm}, {a.epochs} epochs, "
      f"{a.seeds} head seeds", flush=True)

rows = []
for s in range(a.seeds):
    torch.manual_seed(1000 + 7919*s)
    m = fit5.Readout(s, a.arm).cuda()
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=.003)
    t0 = time.time()
    for e in range(a.epochs):
        m.train()
        for r in train:
            loss, _ = m(r); opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.)
            opt.step()
    m.eval()
    h = score([m.predict(r) for r in ev])
    rows.append(h)
    print(f"  head seed {s}: " + ' '.join(f'{h[k]:.4f}' for k in KEYS)
          + '   delta ' + ' '.join(f'{h[k]-base[k]:+.4f}' for k in KEYS)
          + f'   ({time.time()-t0:.0f}s)', flush=True)

print(f"\n  {'':12s} " + ' '.join(f'{k:>22s}' for k in KEYS))
summ = {}
for lab, f in (('head mean', lambda v: statistics.fmean(v)),
               ('head std', lambda v: statistics.stdev(v)),
               ('delta mean', lambda v: statistics.fmean(v))):
    vals = []
    for k in KEYS:
        v = [r[k] for r in rows]
        vals.append(f([x-base[k] for x in v]) if lab == 'delta mean' else f(v))
    print(f'  {lab:12s} ' + ' '.join(f'{x:>22.4f}' for x in vals))
for k in KEYS:
    v = [r[k] for r in rows]
    summ[k] = dict(per_head_seed=v, mean=statistics.fmean(v), std=statistics.stdev(v),
                   delta_mean=statistics.fmean([x-base[k] for x in v]),
                   core=base[k])
    sd = summ[k]['std']; dm = abs(summ[k]['delta_mean'])
    print(f'  {k:22s} delta {summ[k]["delta_mean"]:+.4f}  head-seed std {sd:.4f}  '
          f'-> {"clears 2x" if dm >= 2*sd else "INSIDE head-seed noise"}')
Path(a.out).write_text(json.dumps(dict(dataset=a.dataset, arm=a.arm, epochs=a.epochs,
    head_seeds=a.seeds, core=base, metrics=summ,
    note='head-training variance only, core held fixed; NOT a 3-seed core mean'), indent=1)+'\n')
print('HEADSEED DONE ->', a.out, flush=True)
