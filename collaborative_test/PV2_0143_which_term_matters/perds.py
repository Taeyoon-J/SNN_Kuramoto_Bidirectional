import json, pathlib, statistics
OLD = pathlib.Path('/work/USERS/tkim1/sw0255/run/jobs/model253')
K = ('fg_ari', 'foreground_iou', 'matched_object_iou')
print('Per-dataset cores, 3,500 unique x 10 epochs, FINISHED, evaluated on 1,000 images')
print('of their own dataset. These are the cores whose phase_primary never moved.\n')
for ds in ('tetrominoes', 'multi_dsprites'):
    per = {k: [] for k in K}
    for s in (0, 1, 2):
        p = OLD/ds/f'seed{s}/evaluation/result.json'
        if not p.exists(): print(f'{ds} seed{s}: no result.json'); continue
        d = json.load(open(p))
        # find the metric block whatever it is called
        flat = {}
        def walk(o, pre=''):
            if isinstance(o, dict):
                for k, v in o.items(): walk(v, pre+'/'+k)
            elif isinstance(o, (int, float)): flat[pre] = o
        walk(d)
        hits = {k: [(kk, vv) for kk, vv in flat.items() if kk.endswith('/'+k)] for k in K}
        if s == 0:
            print(f'{ds}: metric keys found -> '
                  + ', '.join(f'{k}:{len(v)}' for k, v in hits.items()))
            for k in K:
                for kk, vv in hits[k][:4]: print(f'    {kk} = {vv:.4f}')
        for k in K:
            if hits[k]: per[k].append(hits[k][0][1])
    if all(per[k] for k in K):
        print(f'  {ds}: per-seed and 3-seed mean')
        for k in K:
            v = per[k]
            m = statistics.fmean(v)
            sd = statistics.stdev(v) if len(v) > 1 else 0.0
            print(f'    {k:22s} {["%.4f" % x for x in v]}  mean {m:.4f} ± {sd:.4f}')
    print()
