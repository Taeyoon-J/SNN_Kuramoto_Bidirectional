#!/usr/bin/env python3
"""Choose Slot Attention's background rule on validation, then report it on fixed320.

Slot ids are arbitrary, so mapping one of them to the contract's background id 0 is a free
choice, and the choice is worth a lot: under "the slot covering the most patches overall",
Slot Attention calls 85.1% of the image foreground where the truth is 21.2%, which is most
of why its foreground IoU reads 0.2022.

Four rules are fixed before looking at any number, and the winner is picked by validation
FG-ARI -- the same rule, on the same split, that picks our own head's arm and epoch. The
fixed320 column is then the selected rule's score and nothing is tuned on it.

  global_largest   the slot with the most patches across the whole split (what we used)
  image_largest    per image, the slot with the most patches
  border_largest   per image, the slot owning the most patches on the frame border
  border_union     per image, every slot with over half its patches on the border

`border_union` is the only one that can merge a split background, which is the failure the
others cannot fix: with 7 slots on an unfamiliar render, one object-slot set can carve the
floor into pieces, and then no single-slot rule can recover a background at all.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--val', required=True); ap.add_argument('--eval', required=True)
ap.add_argument('--code', required=True); ap.add_argument('--config', required=True)
ap.add_argument('--run', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args()
CODE = Path(a.code); sys.path.insert(0, str(CODE/'vendor')); sys.path.insert(0, str(CODE))
import portable_runtime as pr
config = pr.configure(a.config); output = pr.connect_inputs(config, a.run)
from full_benchmark.data import Dataset
from mod_benchmark import common as c
dataset = Dataset('clevr', output); metric = c.benchmark_module('evaluation')
KEYS = ('fg_ari','foreground_iou','matched_object_iou')


def border_mask(g):
    m = np.zeros((g, g), bool); m[0] = m[-1] = True; m[:, 0] = m[:, -1] = True
    return m


def relabel(slots, background_ids, K):
    """Send every slot in background_ids to 0 and compact the rest."""
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
        bg = [{int(k)}] * len(slots)
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
    row['predicted_foreground'] = float((pred > 0).mean())
    return row


RULES = ('global_largest','image_largest','border_largest','border_union')
val, ev = np.load(a.val), np.load(a.eval)
K = int(val['num_slots'])
truth_val = float((dataset.labels(val['ids'], 'validation') > 0).mean())
truth_ev = float((dataset.labels(ev['ids'], 'fixed320') > 0).mean())
print(f"truth foreground: validation {truth_val:.4f}   fixed320 {truth_ev:.4f}\n")
report = {}
for rule in RULES:
    v = score(apply_rule(val['slots'], K, rule), val['ids'], 'validation')
    report[rule] = dict(validation=v)
    print(f"  {rule:15s} val fg_ari {v['fg_ari']:.4f}  fg_iou {v['foreground_iou']:.4f}  "
          f"obj_iou {v['matched_object_iou']:.4f}  predicted fg {v['predicted_foreground']:.4f}")
# Every rule gives the identical FG-ARI: the rules only relabel or merge slots that lie on
# the border, and those patches are background in the truth, which FG-ARI excludes. So FG-ARI
# cannot separate them, and the tie breaks on Foreground IoU -- the choice most favourable to
# the baseline, still decided on validation alone.
pick = max(RULES, key=lambda r: (round(report[r]['validation']['fg_ari'], 6),
                                 report[r]['validation']['foreground_iou']))
e = score(apply_rule(ev['slots'], K, pick), ev['ids'], 'fixed320')
report[pick]['fixed320'] = e
print(f"\nselected on validation FG-ARI, ties broken on Foreground IoU: {pick}")
print(f"fixed320: fg_ari {e['fg_ari']:.4f}  fg_iou {e['foreground_iou']:.4f}  "
      f"obj_iou {e['matched_object_iou']:.4f}  predicted fg {e['predicted_foreground']:.4f}")
# also report every rule on fixed320, labelled as diagnostic, so the spread is visible
print('\ndiagnostic -- every rule on fixed320 (not used for selection):')
for rule in RULES:
    d = score(apply_rule(ev['slots'], K, rule), ev['ids'], 'fixed320')
    report[rule]['fixed320_diagnostic'] = d
    print(f"  {rule:15s} {d['fg_ari']:.4f}  {d['foreground_iou']:.4f}  {d['matched_object_iou']:.4f}  "
          f"predicted fg {d['predicted_foreground']:.4f}")
Path(a.out).write_text(json.dumps(dict(selected=pick, rules=report, num_slots=K,
    truth_foreground=dict(validation=truth_val, fixed320=truth_ev)), indent=1)+'\n')
print('\nSA_BG DONE ->', a.out, flush=True)
