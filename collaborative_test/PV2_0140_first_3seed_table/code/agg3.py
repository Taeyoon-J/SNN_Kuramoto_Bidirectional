#!/usr/bin/env python3
"""Aggregate per-seed scores into the table the contract requires.

Per-seed values, mean and standard deviation, for all three metrics, never combined into one
number. Refuses to emit a mean from fewer than the seeds it was told to expect, because the
failure this guards against is reporting one checkpoint evaluated three times as a 3-seed
training mean.
"""
import argparse, json, statistics
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--results', nargs='+', required=True,
                help='fit5/recipe json files, one per seed')
ap.add_argument('--label', required=True)
ap.add_argument('--expect-seeds', type=int, default=3)
ap.add_argument('--key', default='head', choices=['head', 'head_recipe', 'baseline_253'])
ap.add_argument('--out', required=True)
a = ap.parse_args()

KEYS = ('fg_ari', 'foreground_iou', 'matched_object_iou')
rows = []
for p in a.results:
    d = json.load(open(p))
    blob = d.get(a.key) or d.get('head') or d.get('head_recipe')
    if blob is None:
        raise SystemExit(f'{p}: no "{a.key}" block; keys are {sorted(d)}')
    rows.append(dict(file=str(p), dataset=d.get('dataset'), arm=d.get('arm'),
                     epoch=d.get('epoch'), **{k: float(blob[k]) for k in KEYS}))

seeds = len(rows)
if seeds != a.expect_seeds:
    raise SystemExit(f'expected {a.expect_seeds} seed results, got {seeds}. '
                     'A mean over fewer is not the 3-seed mean the contract asks for, and '
                     'one checkpoint scored three times is never a 3-seed training mean.')
datasets = {r['dataset'] for r in rows}
if len(datasets) != 1:
    raise SystemExit(f'results span more than one dataset: {datasets}')

summary = {}
print(f'\n{a.label} -- {rows[0]["dataset"]}, {seeds} training seeds')
print(f"  {'seed':>4s} {'arm':>18s} {'ep':>3s} " + ' '.join(f'{k:>20s}' for k in KEYS))
for i, r in enumerate(rows):
    print(f"  {i:>4d} {str(r['arm']):>18s} {str(r['epoch']):>3s} "
          + ' '.join(f'{r[k]:>20.4f}' for k in KEYS))
print(f"  {'mean':>4s} {'':>18s} {'':>3s} ", end='')
for k in KEYS:
    v = [r[k] for r in rows]
    summary[k] = dict(per_seed=v, mean=statistics.fmean(v),
                      std=statistics.stdev(v) if len(v) > 1 else 0.0)
    print(f'{summary[k]["mean"]:>20.4f}', end='')
print(f"\n  {'std':>4s} {'':>18s} {'':>3s} ", end='')
for k in KEYS:
    print(f'{summary[k]["std"]:>20.4f}', end='')
print()

Path(a.out).write_text(json.dumps(dict(
    label=a.label, dataset=rows[0]['dataset'], seeds=seeds, per_seed_rows=rows,
    metrics=summary,
    note='per-seed values, mean and standard deviation; the three metrics are never combined'),
    indent=1)+'\n')
print(f'wrote {a.out}')
