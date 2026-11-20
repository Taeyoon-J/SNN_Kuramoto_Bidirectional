import json, pathlib, statistics
OLD = pathlib.Path('/work/USERS/tkim1/sw0255/run/jobs/model253')
K = ('phase_primary', 'positive_spike')
print('The older per-dataset cores: 3,500 unique x 10 epochs = 2,190 updates, finished.')
print('If these also failed to move phase_primary, the behaviour belongs to the datasets,')
print('not to the current run.\n')
for ds in ('tetrominoes', 'multi_dsprites'):
    for s in (0, 1, 2):
        p = OLD/ds/f'seed{s}/training.jsonl'
        if not p.exists(): print(f'{ds} seed{s}: missing'); continue
        rows = [json.loads(l) for l in open(p) if l.strip()]
        a = {k: statistics.fmean([r['parts'][k] for r in rows[:50]]) for k in K}
        b = {k: statistics.fmean([r['parts'][k] for r in rows[-50:]]) for k in K}
        gn = [r['gradient_norm'] for r in rows]
        txt = '   '.join(f'{k.split("_")[0]} {a[k]:6.3f}->{b[k]:6.3f} '
                         f'({(b[k]-a[k])/abs(a[k])*100:+6.1f}%)' for k in K)
        print(f'{ds:15s} seed{s}  {len(rows)} upd   {txt}   grad median {statistics.median(gn):.2f}')
