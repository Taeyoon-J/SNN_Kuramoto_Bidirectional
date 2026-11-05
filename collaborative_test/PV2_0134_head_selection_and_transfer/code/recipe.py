#!/usr/bin/env python3
"""Fit a head on a new dataset with no labels from that dataset at all.

The CLEVR head's weights do not transfer: on Tetrominoes it loses 0.0321 FG-ARI and 0.1068
Foreground IoU, on Multi-dSprites 0.0377 and 0.0124. So the head is per-dataset. The
question that leaves is what a new dataset actually costs.

fit5 refits it, but fit5 chooses the arm and the epoch on that dataset's validation FG-ARI,
and FG-ARI needs labels. The training losses are unsupervised; the selection is not, and
calling the result label-free would be wrong.

This separates the two. The recipe -- which arm, how many epochs -- is fixed once on CLEVR
and reused verbatim. Only the unlabelled images of the new dataset are used, through the
same unsupervised losses. Nothing from the new dataset's labels enters at any point, so if
the numbers hold, a new dataset costs 256 unlabelled images and about a minute.

Selection cannot be replaced by the training loss, which was checked: across CLEVR's five
arms, validation RGB MSE ranked exactly inverse to FG-ARI. Fixing the recipe is the way
around that, not minimising the loss.
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
ap.add_argument('--train-cache', required=True); ap.add_argument('--eval-cache', required=True)
ap.add_argument('--eval-split', required=True); ap.add_argument('--regions', default='253')
ap.add_argument('--arm', required=True); ap.add_argument('--epochs', type=int, required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()
output = fit5.connect(a)
from full_benchmark.data import Dataset
from mod_benchmark import common as c

key = 'regions_253' if a.regions == '253' else 'regions_all'
train, ev = fit5.load(a.train_cache, key), fit5.load(a.eval_cache, key)
print(f"{a.dataset}: {len(train)} unlabelled training images, {len(ev)} evaluation images")
print(f"recipe fixed on CLEVR: arm {a.arm}, {a.epochs} epochs, no validation use", flush=True)

torch.manual_seed(1000)
m = fit5.Readout(0, a.arm).cuda()
opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=.003)
t0 = time.time()
for e in range(a.epochs):
    m.train()
    for r in train:
        loss, _ = m(r); opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.); opt.step()
m.eval()
print(f"trained in {time.time()-t0:.0f}s", flush=True)

dataset = Dataset(a.dataset, output); metric = c.benchmark_module('evaluation')
KEYS = ('fg_ari','foreground_iou','matched_object_iou')
ids = np.array([r['image_id'] for r in ev])
target = torch.from_numpy(dataset.labels(ids, a.eval_split))
def score(preds):
    s = metric.evaluate_patch_masks(torch.from_numpy(np.stack(preds).astype(np.int64)), target)
    return {k: float(s['mean'][k]) for k in KEYS}
base = score([r['regions'] for r in ev])
preds = [m.predict(r) for r in ev]
head = score(preds)
moved = float(np.mean([(p != r['regions']).mean() for p, r in zip(preds, ev)]))
print(f"\n{a.dataset} ({len(ev)} images, split {a.eval_split})")
print(f"{'':18s} {'fg_ari':>9s} {'fg_iou':>9s} {'obj_iou':>9s}")
print(f"{'253':18s} {base['fg_ari']:9.4f} {base['foreground_iou']:9.4f} {base['matched_object_iou']:9.4f}")
print(f"{'head (recipe)':18s} {head['fg_ari']:9.4f} {head['foreground_iou']:9.4f} {head['matched_object_iou']:9.4f}")
print(f"{'delta':18s} {head['fg_ari']-base['fg_ari']:+9.4f} "
      f"{head['foreground_iou']-base['foreground_iou']:+9.4f} "
      f"{head['matched_object_iou']-base['matched_object_iou']:+9.4f}   changed {moved:.4f}", flush=True)
torch.save(dict(model=m.state_dict(), arm=a.arm,
                selected=dict(epoch=a.epochs, rule='recipe fixed on CLEVR, no validation use')),
           Path(a.out).with_suffix('.pt'))
Path(a.out).write_text(json.dumps(dict(dataset=a.dataset, arm=a.arm, epochs=a.epochs,
    labels_from_this_dataset='none', baseline_253=base, head_recipe=head,
    patches_changed=moved, eval_count=len(ev)), indent=1)+'\n')
print('RECIPE DONE ->', a.out, flush=True)
