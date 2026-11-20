import json, pathlib, statistics
B = pathlib.Path('/work/USERS/tkim1/sw0255b')
runs = {'253 CLEVR (finished)': B/'recovered/run/jobs/model253/clevr/seed0/training.jsonl',
        '261 tetrominoes': B/'run_261_tetrominoes/jobs/model253/tetrominoes/seed0/training.jsonl',
        '261 multi_dsprites': B/'run_261_multi_dsprites/jobs/model253/multi_dsprites/seed0/training.jsonl'}
K = ('phase_primary', 'positive_spike')
print('Like-for-like: each run measured at the same fraction of the 6,250-update schedule.')
print('The question is whether 253-on-CLEVR had already moved by update ~4,200, or whether')
print('its gains came in the final third. If the former, the new runs are behind.\n')
for name, p in runs.items():
    rows = [json.loads(l) for l in open(p) if l.strip()]
    print(f'{name}: {len(rows)} updates')
    base = {k: statistics.fmean([r['parts'][k] for r in rows[:50]]) for k in K}
    for frac in (0.25, 0.50, 0.66, 0.69, 1.00):
        i = int(frac*6250)
        if i > len(rows): continue
        w = rows[max(0, i-50):i] or rows[:50]
        cur = {k: statistics.fmean([r['parts'][k] for r in w]) for k in K}
        s = '   '.join(f'{k.split("_")[0]} {cur[k]:6.3f} ({(cur[k]-base[k])/abs(base[k])*100:+6.1f}%)'
                       for k in K)
        print(f'   at {frac*100:5.1f}% (update {i:5d}):  {s}')
    print()
