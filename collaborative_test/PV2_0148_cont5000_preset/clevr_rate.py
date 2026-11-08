import json, pathlib
for name, p in (('253 CLEVR seed0', '/work/USERS/tkim1/sw0255b/recovered/run/jobs/model253/clevr/seed0'),
                ('261 tetrominoes', '/work/USERS/tkim1/sw0255b/run_261_tetrominoes/jobs/model253/tetrominoes/seed0'),
                ('261 multi_dsprites', '/work/USERS/tkim1/sw0255b/run_261_multi_dsprites/jobs/model253/multi_dsprites/seed0')):
    d = pathlib.Path(p)
    r = json.load(open(d/'training_report.json')) if (d/'training_report.json').exists() else {}
    rows = [json.loads(l) for l in open(d/'training.jsonl') if l.strip()]
    tot = r.get('seconds')
    print(f'{name:20s} updates {r.get("updates")}  report seconds {tot}'
          + (f'  -> {tot/r["updates"]:.2f} s/update WALL' if tot and r.get('updates') else ''))
    # the jsonl 'seconds' restarts each session, so the max within a session is the best proxy
    spans = []
    cur = [rows[0]]
    for a, b in zip(rows, rows[1:]):
        if b['seconds'] >= a['seconds']: cur.append(b)
        else: spans.append(cur); cur = [b]
    spans.append(cur)
    longest = max(spans, key=len)
    dt = (longest[-1]['seconds']-longest[0]['seconds'])/(longest[-1]['update']-longest[0]['update'])
    print(f'{"":20s} {len(spans)} session(s); longest {len(longest)} updates at {dt:.2f} s/update')
