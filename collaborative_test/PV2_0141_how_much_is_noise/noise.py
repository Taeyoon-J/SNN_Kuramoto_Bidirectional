#!/usr/bin/env python3
"""How big are our reported deltas next to the run-to-run wobble we can already measure?

Every number in PV2_0133-0138 is seed 0, so none of them has an error bar, and some of the
reported gains are very small -- Tetrominoes improved by +0.0006 FG-ARI. Without a spread, that
is unreadable: it could be a real effect or it could be nothing.

Seed spread needs seeds we do not have. But the fit5 runs recorded the full per-epoch
validation curve for every arm, and that is a different and still useful quantity: how much the
score moves when nothing about the model changes except where training happened to stop. It is
a lower bound on the noise -- genuine seed variance adds the random encoder draw on top -- so a
delta that is already small against THIS is certainly not established.

Two numbers per dataset:

  late-epoch spread   std of validation FG-ARI over epochs 10-20 of the selected arm, after the
                      curve has settled. Movement with no modelling content.
  arm spread          std across the five arms of their selected validation FG-ARI. How much
                      the answer depends on which loss mixture was picked.
"""
import json, statistics
from pathlib import Path

D = Path(__file__).resolve().parent.parent/'PV2_0134_head_selection_and_transfer/diagnostics/fit5'
KEYS = ('fg_ari', 'foreground_iou', 'matched_object_iou')
# the eval-split deltas we have reported, head minus 253, from metrics.csv
REPORTED = {
    'clevr':          dict(label='B′ refit',  fg_ari=+0.0086, foreground_iou=+0.0285, matched_object_iou=+0.0298),
    'tetrominoes':    dict(label='B′ refit',  fg_ari=+0.0006, foreground_iou=+0.0010, matched_object_iou=+0.0015),
    'multi_dsprites': dict(label='B′ refit',  fg_ari=+0.0067, foreground_iou=+0.0036, matched_object_iou=+0.0032),
}
RECIPE = {
    'clevr':          dict(fg_ari=+0.0047, foreground_iou=+0.0029, matched_object_iou=+0.0046),
    'tetrominoes':    dict(fg_ari=+0.0006, foreground_iou=+0.0010, matched_object_iou=+0.0015),
    'multi_dsprites': dict(fg_ari=+0.0067, foreground_iou=+0.0036, matched_object_iou=+0.0032),
}

out = {}
for ds in ('clevr', 'tetrominoes', 'multi_dsprites'):
    blob = json.load(open(D/f'{ds}.json'))
    arms, chosen = blob['arms'], blob['arm']
    curve = arms[chosen]['curve']
    late = [c for c in curve if c['epoch'] >= 10]
    late_std = {k: statistics.stdev([c[k] for c in late]) for k in KEYS}
    arm_std = {k: statistics.stdev([arms[a]['selected'][k] for a in arms]) for k in KEYS}
    out[ds] = dict(selected_arm=chosen, selected_epoch=blob['epoch'],
                   late_epoch_std=late_std, arm_std=arm_std,
                   reported_delta=REPORTED[ds], recipe_delta=RECIPE[ds])

    print(f'\n=== {ds}   (selected {chosen}, epoch {blob["epoch"]})')
    print(f"  {'metric':<22s} {'reported':>10s} {'late-ep std':>12s} {'arm std':>10s} {'verdict':>22s}")
    for k in KEYS:
        d = abs(REPORTED[ds][k]); noise = max(late_std[k], arm_std[k])
        verdict = ('inside the wobble' if d < noise else
                   'comparable to wobble' if d < 2*noise else 'clears it')
        print(f'  {k:<22s} {REPORTED[ds][k]:>+10.4f} {late_std[k]:>12.4f} '
              f'{arm_std[k]:>10.4f} {verdict:>22s}')

print('\n' + '='*78)
survive = sum(1 for ds in out for k in KEYS
              if abs(out[ds]['reported_delta'][k]) >= 2*max(out[ds]['late_epoch_std'][k],
                                                            out[ds]['arm_std'][k]))
print(f'{survive} of 9 reported cells clear twice the measurable wobble.')
print('This is a LOWER bound on the noise: real seed variance adds the random encoder draw,')
print('which the normalization provenance shows differs per seed (feature std 0.2478 vs')
print('0.1337 vs 0.3533 on the three CLEVR seeds). Cells that do not clear it here are not')
print('established by a seed-0 run and must not be reported as gains without seeds.')
(Path(__file__).resolve().parent/'noise.json').write_text(json.dumps(out, indent=1)+'\n')
