#!/usr/bin/env python3
"""Does ONE head, fitted on CLEVR, carry to the other datasets unchanged?

The claim this project wants to make is that the core trained on CLEVR still finds objects
elsewhere, with the head as the cheap per-dataset part. Fitting a head per dataset, which is
what fit5 does, does not test that claim -- it tests whether a head helps at all.

This does test it. The CLEVR head is loaded as it was selected, with no refitting, no
per-dataset arm choice and no per-dataset epoch choice, and run on Tetrominoes and
Multi-dSprites. Three columns are then comparable on the same images:

  253             the core's own partition, no head
  head (own)      a head fitted on that dataset        -- fit5, the per-dataset cost
  head (CLEVR)    the CLEVR head, unchanged            -- this file, zero per-dataset cost

If the third column holds up, the head is not a per-dataset component at all and the
generalisation claim is about the core. If it collapses, the head is doing dataset-specific
work and that has to be said plainly.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit5

ap = argparse.ArgumentParser()
ap.add_argument('--code', required=True); ap.add_argument('--config', required=True)
ap.add_argument('--run', required=True)
ap.add_argument('--checkpoint', required=True, help='a head saved by fit5')
ap.add_argument('--dataset', required=True); ap.add_argument('--eval-cache', required=True)
ap.add_argument('--eval-split', required=True); ap.add_argument('--regions', default='253')
ap.add_argument('--out', required=True)
a = ap.parse_args()
output = fit5.connect(a)
from full_benchmark.data import Dataset
from mod_benchmark import common as c

blob = torch.load(a.checkpoint, map_location='cuda:0', weights_only=True)
arm = blob['arm']
m = fit5.Readout(0, arm).cuda().eval(); m.load_state_dict(blob['model'])
print(f"head from {Path(a.checkpoint).name}: arm {arm}, epoch {blob['selected']['epoch']}, "
      f"selected on its own dataset's validation split", flush=True)

key = 'regions_253' if a.regions == '253' else 'regions_all'
ev = fit5.load(a.eval_cache, key)
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
print(f"{'':16s} {'fg_ari':>9s} {'fg_iou':>9s} {'obj_iou':>9s}")
print(f"{'253':16s} {base['fg_ari']:9.4f} {base['foreground_iou']:9.4f} {base['matched_object_iou']:9.4f}")
print(f"{'head (CLEVR)':16s} {head['fg_ari']:9.4f} {head['foreground_iou']:9.4f} {head['matched_object_iou']:9.4f}")
print(f"{'delta':16s} {head['fg_ari']-base['fg_ari']:+9.4f} "
      f"{head['foreground_iou']-base['foreground_iou']:+9.4f} "
      f"{head['matched_object_iou']-base['matched_object_iou']:+9.4f}   changed {moved:.4f}", flush=True)
Path(a.out).write_text(json.dumps(dict(dataset=a.dataset, source_checkpoint=str(a.checkpoint),
    arm=arm, source_epoch=blob['selected']['epoch'], refitted=False,
    baseline_253=base, head_clevr=head, patches_changed=moved, eval_count=len(ev)), indent=1)+'\n')
print('XFER DONE ->', a.out, flush=True)
