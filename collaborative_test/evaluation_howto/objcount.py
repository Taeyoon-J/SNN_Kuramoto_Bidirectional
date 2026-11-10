#!/usr/bin/env python3
"""Object-count accuracy: does the readout predict the right NUMBER of objects?

Every metric in this project scores the partition. None of them answers "how many objects did
it think there were", and that turned out to matter: on CLEVR the branch whose predicted count
is nearly exact (PV2_0152, 6.6 against a truth of 6.36) has the *worse* matched-object IoU, and
the branch that over-segments most (PV2_0153, 9.9) has the better one. So the count is not a
proxy for the partition and has to be measured on its own.

Counts are read the same way the contract reads a prediction: the number of distinct non-zero
labels in the grid. Background is id 0 on both sides and is not an object.

Reported per dataset:
  exact      fraction of images with predicted count == true count
  within 1   fraction within +-1, which is the tolerance a human would call "right"
  MAE        mean absolute error
  bias       mean signed error; positive means over-segmenting
  truth/pred mean counts, and the confusion between them

A head can be included: it is fitted exactly as `headseed.py` fits it (same fixed label-free
recipe, same seeds) and its `predict` output is counted instead of the core's.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--cache', required=True, help='eval cache (cache_any.py output)')
ap.add_argument('--train-cache', default=None, help='train cache; given, the head is included')
ap.add_argument('--dataset', required=True); ap.add_argument('--split', required=True)
ap.add_argument('--code', required=True); ap.add_argument('--config', required=True)
ap.add_argument('--run', required=True); ap.add_argument('--out', default=None)
ap.add_argument('--label', default=''); ap.add_argument('--seeds', type=int, default=5)
ap.add_argument('--arm', default='local_rgb_trust'); ap.add_argument('--epochs', type=int, default=2)
a = ap.parse_args()

CODE = Path(a.code); sys.path.insert(0, str(CODE/'vendor')); sys.path.insert(0, str(CODE))
import torch
import portable_runtime as pr
config = pr.configure(a.config); output = pr.connect_inputs(config, a.run)
from full_benchmark.data import Dataset
import fit5

ev = fit5.load(a.cache, 'regions_253')
ids = np.array([r['image_id'] for r in ev])
truth = Dataset(a.dataset, output).labels(ids, a.split)
t_cnt = np.array([len(np.unique(l[l > 0])) for l in truth])


def counts(preds):
    return np.array([len(np.unique(p[p > 0])) for p in preds])


def report(name, p_cnt):
    err = p_cnt - t_cnt
    row = dict(label=name, n=int(len(t_cnt)),
               exact=float((err == 0).mean()), within1=float((np.abs(err) <= 1).mean()),
               mae=float(np.abs(err).mean()), bias=float(err.mean()),
               pred_mean=float(p_cnt.mean()), truth_mean=float(t_cnt.mean()),
               pred_min=int(p_cnt.min()), pred_max=int(p_cnt.max()))
    print(f"  {name:28s} exact {row['exact']:.3f}  within1 {row['within1']:.3f}  "
          f"MAE {row['mae']:.2f}  bias {row['bias']:+.2f}  "
          f"pred {row['pred_mean']:.2f} vs truth {row['truth_mean']:.2f}", flush=True)
    return row


print(f"{a.dataset} {a.split}: {len(ids)} images, truth objects/img mean {t_cnt.mean():.2f} "
      f"(min {t_cnt.min()}, max {t_cnt.max()})", flush=True)
out = {'dataset': a.dataset, 'split': a.split, 'cache': a.cache,
       'truth_mean': float(t_cnt.mean()), 'rows': []}
out['rows'].append(report((a.label + ' core alone').strip(), counts([r['regions'] for r in ev])))

if a.train_cache:
    import time
    train = fit5.load(a.train_cache, 'regions_253')
    per_seed = []
    for s in range(a.seeds):
        torch.manual_seed(1000 + 7919*s)
        m = fit5.Readout(s, a.arm).cuda()
        opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=.003)
        for _ in range(a.epochs):
            m.train()
            for r in train:
                loss, _ = m(r); opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.)
                opt.step()
        m.eval()
        per_seed.append(counts([m.predict(r) for r in ev]))
    # the head is a per-image relabelling, so average its metrics over seeds rather than its counts
    rows = [report(f'{a.label} +head seed {s}'.strip(), c) for s, c in enumerate(per_seed)]
    agg = {k: float(np.mean([r[k] for r in rows])) for k in ('exact', 'within1', 'mae', 'bias', 'pred_mean')}
    agg_std = {k: float(np.std([r[k] for r in rows], ddof=1)) for k in ('exact', 'within1', 'mae', 'bias')}
    print(f"  {'+head, mean of ' + str(a.seeds) + ' seeds':28s} exact {agg['exact']:.3f}"
          f"+-{agg_std['exact']:.3f}  within1 {agg['within1']:.3f}+-{agg_std['within1']:.3f}  "
          f"MAE {agg['mae']:.2f}+-{agg_std['mae']:.2f}  bias {agg['bias']:+.2f}  "
          f"pred {agg['pred_mean']:.2f}", flush=True)
    out['head_mean'] = agg; out['head_std'] = agg_std; out['head_per_seed'] = rows

if a.out:
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    print('->', a.out)
print('OBJCOUNT_DONE', flush=True)
