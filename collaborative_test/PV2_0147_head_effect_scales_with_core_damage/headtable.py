import json, pathlib
B = pathlib.Path('/work/USERS/tkim1/sw0255b')
K = ('fg_ari', 'foreground_iou', 'matched_object_iou')
rows = []
# target-trained cores (ev261 caches, grid 64, 10,000x10, min_group 8)
for ds, files in (('tetrominoes', [('fit261/tetrominoes.json', "B'", 'head'),
                                   ('fit261/recipe_tetrominoes.json', 'recipe', 'head_recipe')]),
                  ('multi_dsprites', [('fit261/multi_dsprites.json', "B'", 'head'),
                                      ('fit261/recipe_multi_dsprites.json', 'recipe', 'head_recipe')])):
    for f, rule, key in files:
        p = B/f
        if not p.exists(): print(f'MISSING {f}'); continue
        d = json.load(open(p))
        core = d['baseline_253']; head = d.get(key) or d.get('head') or d.get('head_recipe')
        rows.append(dict(core_on=ds, rule=rule, arm=d.get('arm'),
                         epoch=d.get('epoch', d.get('epochs')),
                         moved=d.get('patches_changed'),
                         core={k: core[k] for k in K}, head={k: head[k] for k in K}))
print(f"{'core trained on':16s} {'rule':8s} {'arm':18s} {'ep':>3s} {'moved':>7s}  "
      + '  '.join(f'{k[:11]:>26s}' for k in K))
for r in rows:
    cells = []
    for k in K:
        c, h = r['core'][k], r['head'][k]
        rel = (h-c)/c*100 if c else float('inf')
        cells.append(f'{c:.4f}->{h:.4f} {h-c:+.4f} {rel:+7.1f}%')
    print(f"{r['core_on']:16s} {r['rule']:8s} {str(r['arm']):18s} {str(r['epoch']):>3s} "
          f"{r['moved']:7.4f}  " + '  '.join(cells))
json.dump(rows, open('/tmp/headtable.json','w'), indent=1)
