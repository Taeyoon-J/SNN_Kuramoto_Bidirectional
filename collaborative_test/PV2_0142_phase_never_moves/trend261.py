import json, pathlib, statistics, time
B = pathlib.Path('/work/USERS/tkim1/sw0255b')
REF = B/'recovered/run/jobs/model253/clevr/seed0/training.jsonl'
def parts(rows, lo, hi):
    out = {}
    for k in ('phase_primary', 'positive_spike'):
        v = [r['parts'][k] for r in rows[lo:hi] if k in r['parts']]
        if v: out[k] = statistics.fmean(v)
    return out
def show(name, p, target=6250):
    rows = [json.loads(l) for l in open(p) if l.strip()]
    a, b = parts(rows, 0, 50), parts(rows, -50, None)
    print(f'\n{name}: {len(rows)} updates of {target}  ({100*len(rows)/target:.0f}%)')
    for k in a:
        d = (b[k]-a[k])/abs(a[k])*100
        print(f'   {k:18s} {a[k]:8.3f} -> {b[k]:8.3f}  {d:+7.1f}%')
    gn = [r['gradient_norm'] for r in rows]
    print(f'   {"gradient_norm":18s} {statistics.fmean(gn[:50]):8.2f} -> '
          f'{statistics.fmean(gn[-50:]):8.2f}   median {statistics.median(gn):.2f}')
    # rate from the most recent contiguous session only (seconds restarts on resume)
    recent = [r for r in rows[-40:]]
    secs = [r['seconds'] for r in recent]
    if len(recent) > 5 and secs[-1] > secs[0]:
        dt = (secs[-1]-secs[0])/(recent[-1]['update']-recent[0]['update'])
        left = (target-len(rows))*dt
        print(f'   rate {dt:.1f} s/update -> {left/3600:.1f} h remaining '
              f'({target-len(rows)} updates)')
    return rows
show('253 core on CLEVR (reference, finished)', REF)
for d in ('tetrominoes', 'multi_dsprites'):
    show(f'261 core on {d} (running)',
         B/f'run_261_{d}/jobs/model253/{d}/seed0/training.jsonl')
