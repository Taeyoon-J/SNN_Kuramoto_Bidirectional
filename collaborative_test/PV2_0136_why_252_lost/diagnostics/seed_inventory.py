import json, pathlib
root = pathlib.Path('/work/USERS/tkim1/sw0255/run/jobs/model253')
for d in sorted(root.glob('*')):
    for s in sorted(d.glob('seed*')):
        rp = s/'training_report.json'
        if not rp.exists():
            print(f'{d.name:16s} {s.name}  no report'); continue
        r = json.load(open(rp))
        print(f'{d.name:16s} {s.name}  updates={r.get("updates")}  images_seen={r.get("images_seen")}  '
              f'unique_TRAIN={r.get("unique_TRAIN")}  epochs={r.get("epochs_completed")}  '
              f'ckpt={"resume.pt" if (s/"resume.pt").exists() else "MISSING"}')
