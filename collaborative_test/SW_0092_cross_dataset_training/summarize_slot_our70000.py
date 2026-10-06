import json, statistics, sys
from pathlib import Path
root=Path(sys.argv[1]); out=Path(sys.argv[2]); keys=("fg_ari","foreground_iou","matched_object_iou")
epochs={}
for epoch in (1,3,10):
    seeds={}
    for seed in range(3):
        r=json.loads((root/f"seed{seed}_epoch{epoch}/evaluation_summary.json").read_text())
        seeds[str(seed)]={k:r["scores"]["mean"][k] for k in keys}
    epochs[str(epoch)]={"seeds":seeds,"mean":{k:statistics.mean(seeds[str(s)][k] for s in range(3)) for k in keys}}
report={"experiment":"Slot Attention trained on our 70000 unique scenes", "evaluation":"fixed HDF5 IDs1320-1639", "epochs":epochs}
out.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))

