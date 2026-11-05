#!/usr/bin/env python3
"""Can the head's epoch be chosen without labels?

Rule B' picks the arm and epoch on validation FG-ARI, which needs labels, so a new dataset
costs a labelled validation split. Reusing CLEVR's epoch count instead costs no labels and
works on CLEVR and Multi-dSprites, but loses 0.0565 FG-ARI on Tetrominoes, whose own choice is
epoch 2 rather than 11. The epoch is the whole problem: the arm barely matters.

Two label-free criteria are computed per epoch on the *unlabelled training* images.

  stability   the fraction of patches whose predicted label changed since the previous epoch.
              Low means the partition has stopped moving.

A second criterion, the adjusted Rand index between the predictions under two rollout seeds,
was written and then removed: PV2_0135 found the 253 rollout is deterministic in
`inference_seed` -- three seeds give bitwise identical gamma, q, adjacency and spikes -- so
there is no second seed to compare against and the score read exactly 1.000 at every epoch.
The `features_b` / `gamma_b` / `regions_b` fields in the caches are junk for the same reason.

The validation triple is recorded alongside, marked as a diagnostic: it is what the criteria
are judged against, and it is read on CLEVR only. Whichever criterion is chosen there is then
applied to Tetrominoes and Multi-dSprites with nothing from their labels.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit5

ap = argparse.ArgumentParser()
ap.add_argument('--code', required=True); ap.add_argument('--config', required=True)
ap.add_argument('--run', required=True); ap.add_argument('--dataset', required=True)
ap.add_argument('--train-cache', required=True); ap.add_argument('--val-cache', required=True)
ap.add_argument('--eval-cache', required=True); ap.add_argument('--eval-split', required=True)
ap.add_argument('--regions', default='253'); ap.add_argument('--epochs', type=int, default=20)
ap.add_argument('--out', required=True)
a = ap.parse_args()
output = fit5.connect(a)
from full_benchmark.data import Dataset
from mod_benchmark import common as c


key = 'regions_253' if a.regions == '253' else 'regions_all'
train, val, ev = (fit5.load(a.train_cache, key), fit5.load(a.val_cache, key),
                  fit5.load(a.eval_cache, key))

print(f"{a.dataset}: train {len(train)} val {len(val)} eval {len(ev)}")

dataset = Dataset(a.dataset, output); metric = c.benchmark_module('evaluation')
KEYS = ('fg_ari','foreground_iou','matched_object_iou')
def score(rows, preds, split):
    ids = np.array([r['image_id'] for r in rows])
    s = metric.evaluate_patch_masks(torch.from_numpy(np.stack(preds).astype(np.int64)),
                                    torch.from_numpy(dataset.labels(ids, split)))
    return {k: float(s['mean'][k]) for k in KEYS}

base_val = score(val, [r['regions'] for r in val], 'validation')
base_ev = score(ev, [r['regions'] for r in ev], a.eval_split)
print(f"253   val {base_val['fg_ari']:.4f}/{base_val['foreground_iou']:.4f}/{base_val['matched_object_iou']:.4f}"
      f"   eval {base_ev['fg_ari']:.4f}/{base_ev['foreground_iou']:.4f}/{base_ev['matched_object_iou']:.4f}",
      flush=True)


report = {}
for arm in fit5.ARMS:
    torch.manual_seed(1000); m = fit5.Readout(0, arm).cuda()
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=.003)
    curve = []; prev = None; t1 = time.time()
    for e in range(a.epochs+1):
        if e:
            m.train()
            for r in train:
                loss, _ = m(r); opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.)
                opt.step()
        m.eval()
        # label-free, on the unlabelled training images only
        now = [m.predict(r) for r in train]
        stability = None if prev is None else float(np.mean([(x != y).mean() for x, y in zip(now, prev)]))
        prev = now
        v = score(val, [m.predict(r) for r in val], 'validation')   # diagnostic only
        curve.append(dict(epoch=e, stability=stability, validation=v))
    report[arm] = curve
    best = max(curve, key=lambda c_: c_['validation']['fg_ari'])
    rates = "  ".join(f"{c_['stability']:.4f}" for c_ in curve[1:9])
    print(f"  {arm:18s} best validation fg_ari at epoch {best['epoch']:2d}"
          f"   stability over epochs 1-8: {rates}   ({time.time()-t1:.0f}s)", flush=True)

Path(a.out).write_text(json.dumps(dict(dataset=a.dataset, arms=report,
    baseline_253_validation=base_val, baseline_253_eval=base_ev,
    note='stability is label-free; validation is a diagnostic, read on CLEVR only'), indent=1)+'\n')
print('STOP DONE ->', a.out, flush=True)
