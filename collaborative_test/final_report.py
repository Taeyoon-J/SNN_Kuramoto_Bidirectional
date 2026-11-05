#!/usr/bin/env python3
"""Generate the final side-by-side report from result files, with the honesty rules compiled in.

The report the contract asks for is both models' three metrics as per-seed values, means and
standard deviations, on the same images. It does not exist yet because the runs are not
finished. This builds it from whatever result files are present and, more importantly, refuses
to produce a misleading version of it.

Four rules are enforced rather than remembered, because each is a mistake this project has
already come close to making:

  1. A mean is labelled an N-seed mean only if N independent TRAINING runs produced it. One
     checkpoint scored N times is rejected outright (PV2_0140).
  2. A cell whose delta is inside the measurable wobble is marked, not reported as a gain
     (PV2_0141: only 4 of 9 cleared it at seed 0).
  3. Slot Attention appears in two separate rows -- the published checkpoint as a TRANSFER
     result, and any retrained version with its step count attached. A short run is never
     reported simply as "Slot Attention" (PV2_0139).
  4. The three metrics are never combined into one score, and any row drawn from a core that
     trained on the dataset it is scored on is kept out of the generalisation column
     (PV2_0138, PV2_0140).
"""
import argparse, json, statistics
from pathlib import Path

KEYS = ('fg_ari', 'foreground_iou', 'matched_object_iou')
NICE = {'fg_ari': 'patch_fg_ari', 'foreground_iou': 'patch_foreground_iou',
        'matched_object_iou': 'patch_matched_object_iou'}


def load(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return None


def cell(values, noise=None):
    """One table cell: mean +- std over seeds, flagged if it sits inside the noise."""
    m = statistics.fmean(values)
    s = statistics.stdev(values) if len(values) > 1 else None
    txt = f'{m:.4f}' + (f' ± {s:.4f}' if s is not None else '')
    if noise is not None and abs(m) < 2*noise:
        txt += ' ⚠'
    return txt, m, s


def seed_rows(paths, key):
    """Collect one metric triple per seed, refusing anything that is not an independent run."""
    rows = []
    for p in paths:
        d = load(p)
        if d is None: continue
        blob = d.get(key) or d.get('head') or d.get('head_recipe')
        if blob is None: continue
        ident = (d.get('core_checkpoint') or d.get('source_checkpoint')
                 or d.get('dataset'), d.get('seed'))
        rows.append((ident, {k: float(blob[k]) for k in KEYS}))
    idents = [r[0] for r in rows]
    if len(rows) > 1 and len(set(idents)) == 1:
        raise SystemExit('REFUSED: every result came from the same run. One checkpoint scored '
                         'N times is not an N-seed training mean.')
    return [r[1] for r in rows]


ap = argparse.ArgumentParser()
ap.add_argument('--manifest', help='json describing which result files make up each row; '
                                   'see --example')
ap.add_argument('--noise', default=None, help='PV2_0141 noise.json, to flag unestablished cells')
ap.add_argument('--out')
ap.add_argument('--example', action='store_true',
                help='print a manifest skeleton and exit; --manifest and --out not needed')
a = ap.parse_args()
if not a.example and not (a.manifest and a.out):
    ap.error('--manifest and --out are required unless --example is given')

if a.example:
    print(json.dumps({'dataset': 'clevr', 'eval_split': 'fixed320', 'n_images': 320,
        'rows': [
          {'label': 'Slot Attention, published checkpoint (TRANSFER, trained on the original '
                    'CLEVR render, single checkpoint)', 'files': ['sa/bg_rules.json'],
           'key': 'fixed320', 'kind': 'reference_transfer'},
          {'label': 'Slot Attention, retrained on clevr_with_masks, N steps',
           'files': [], 'key': 'head', 'kind': 'reference_matched'},
          {'label': '253 core alone', 'files': ['fit5/clevr.json'], 'key': 'baseline_253',
           'kind': 'ours'},
          {'label': '253 core + 261 readout (label-free recipe)',
           'files': ['fit5/recipe_clevr.json'], 'key': 'head_recipe', 'kind': 'ours'}]},
        indent=1))
    raise SystemExit

man = json.loads(Path(a.manifest).read_text())
noise = load(a.noise) if a.noise else None
nz = {}
if noise and man['dataset'] in noise:
    n = noise[man['dataset']]
    nz = {k: max(n['late_epoch_std'][k], n['arm_std'][k]) for k in KEYS}

out = [f"# Final report — {man['dataset']}, {man['eval_split']}, {man['n_images']} images",
       '', 'Same image ids in the same order for every row, scored by the same '
       '`evaluate_patch_masks` on the patch grid at 64, where the ceiling is 1.0.', '',
       '| row | seeds | ' + ' | '.join(NICE[k] for k in KEYS) + ' |',
       '|---|---|' + '---|'*len(KEYS)]
flagged = 0
for r in man['rows']:
    vals = seed_rows(r.get('files', []), r.get('key', 'head'))
    if not vals:
        out.append(f"| {r['label']} | — | " + ' | '.join('**not produced**' for _ in KEYS) + ' |')
        continue
    cells = []
    for k in KEYS:
        txt, m, s = cell([v[k] for v in vals], nz.get(k) if r['kind'] == 'ours' else None)
        if '⚠' in txt: flagged += 1
        cells.append(txt)
    n = len(vals)
    label = r['label'] + ('' if n > 1 else '  *(one seed — a candidate, not a result)*')
    out.append(f'| {label} | {n} | ' + ' | '.join(cells) + ' |')

out += ['', '⚠ marks a cell inside twice the measurable wobble (PV2_0141), which is a lower '
        'bound on the noise: real seed variance also redraws the encoder.', '']
if flagged:
    out.append(f'**{flagged} cell(s) are not established.** They are shown for completeness and '
               'must not be described as gains.')
if any(not r.get('files') for r in man['rows']):
    out.append('**At least one row was not produced.** The report is incomplete and the '
               'comparison it implies cannot be claimed.')
Path(a.out).write_text('\n'.join(out) + '\n')
print('\n'.join(out))
