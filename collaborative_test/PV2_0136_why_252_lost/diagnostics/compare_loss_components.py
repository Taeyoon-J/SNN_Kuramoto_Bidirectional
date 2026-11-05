import json, pathlib, statistics
B = pathlib.Path('/work/USERS/tkim1/sw0255b')
paths = {'252': B/'recovered252/run252/jobs/model252/clevr/seed0/training.jsonl',
         '253': B/'recovered/run/jobs/model253/clevr/seed0/training.jsonl'}
for name, p in paths.items():
    rows = [json.loads(l) for l in open(p) if l.strip()]
    first = [r['parts'] for r in rows[:50]]
    last = [r['parts'] for r in rows[-50:]]
    keys = sorted(set(first[0]) | set(last[0]))
    print(f"\n=== {name}: {len(rows)} updates, loss components (mean of first 50 vs last 50)")
    print(f"   {'component':24s} {'first50':>12s} {'last50':>12s} {'change':>9s}")
    for k in keys:
        a = statistics.fmean(d[k] for d in first if k in d)
        b = statistics.fmean(d[k] for d in last if k in d)
        rel = (b-a)/abs(a)*100 if a else float('nan')
        print(f"   {k:24s} {a:12.5f} {b:12.5f} {rel:+8.1f}%")
    gn_a = statistics.fmean(r['gradient_norm'] for r in rows[:50])
    gn_b = statistics.fmean(r['gradient_norm'] for r in rows[-50:])
    print(f"   {'gradient_norm':24s} {gn_a:12.2f} {gn_b:12.2f} {(gn_b-gn_a)/gn_a*100:+8.1f}%")
