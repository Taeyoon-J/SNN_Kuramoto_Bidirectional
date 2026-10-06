import json, statistics, sys
from pathlib import Path
root=Path(sys.argv[1]); out=Path(sys.argv[2]); keys=("fg_ari","foreground_iou","matched_object_iou")
epochs={}
for epoch in (1,3,10):
    seeds={}
    for seed in range(3):
        r=json.loads((root/f"seed{seed}_epoch{epoch}.json").read_text()); row=r["sweep"][0]
        seeds[str(seed)]=row["scored_targets"]["our_hdf5"]["metrics"]
    epochs[str(epoch)]={"seeds":seeds,"mean":{k:statistics.mean(seeds[str(s)][k] for s in range(3)) for k in keys}}
report={"experiment":"our model trained on official released Slot CLEVR6", "training_images":34766,
        "evaluation":"our fixed HDF5 IDs1320-1639", "epochs":epochs}
out.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))

