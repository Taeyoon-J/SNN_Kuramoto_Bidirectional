#!/usr/bin/env python3
import argparse,json
from pathlib import Path
M=("fg_ari","foreground_iou","matched_object_iou")
B={"fg_ari":.6316519272381114,"foreground_iou":.3852088450351232,"matched_object_iou":.46168352926256573}
p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();rows={}
for tag in ("anchor0","anchor100"):
 d=json.loads((a.root/f"SW0069_frozen_s1_{tag}/eval_n32.json").read_text()); r=[x for x in d["sweep"] if x["affinity_mode"]=="spike" and x["synchrony_threshold"]==.35][0]; s=r["scored_targets"]["our_hdf5"]
 m={k:float(s["metrics"][k]) for k in M}; manifest=json.loads((a.root/f"SW0069_frozen_s1_{tag}/model/manifest.json").read_text()); assert manifest["core_max_parameter_change"]==0
 rows[tag]={"metrics":m,"delta":{k:m[k]-B[k] for k in M},"count_mae":s["object_count"]["mae"],"gamma_rms_drift":manifest["final_gamma_rms_drift_n32"]}
out={"experiment":"SW0069 frozen-core feature tuning","baseline":B,"rows":rows,"advancing_arms":[t for t,r in rows.items() if all(r["delta"][k]>0 for k in M)]}
if a.output.exists(): raise FileExistsError(a.output)
a.output.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
