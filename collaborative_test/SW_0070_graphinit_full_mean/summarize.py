#!/usr/bin/env python3
import argparse,json,statistics
from pathlib import Path
M=("fg_ari","foreground_iou","matched_object_iou")
p=argparse.ArgumentParser();p.add_argument("--sw0055-summary",type=Path,required=True);p.add_argument("--candidate-dir",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
old=json.loads(a.sw0055_summary.read_text()); oldagg=old["aggregates"]["long"]["0.5"]
values={k:[float(oldagg[k]["seed_values"]["0"])] for k in M}
rows={"0":{k:values[k][0] for k in M}}
for seed in (1,2):
 d=json.loads((a.candidate_dir/f"seed{seed}_long.json").read_text()); assert d["ids"]==[1320,1639] and d["ground_truth_used_for_prediction"] is False
 r=[x for x in d["sweep"] if x["affinity_mode"]=="spike" and x["synchrony_threshold"]==.5]; assert len(r)==1
 m=r[0]["scored_targets"]["our_hdf5"]["metrics"]; rows[str(seed)]={k:float(m[k]) for k in M}
 for k in M: values[k].append(float(m[k]))
new={k:{"seed_values":values[k],"mean":statistics.mean(values[k]),"std_population":statistics.pstdev(values[k]),"old_mean":float(oldagg[k]["mean"]),"delta":statistics.mean(values[k])-float(oldagg[k]["mean"])} for k in M}
out={"experiment":"SW0070 graph-init0 full three-seed mean","contract":{"ids":[1320,1639],"steps":1024,"settle":512,"threshold":.5},"rows":rows,"metrics":new,"improves_all_three":all(new[k]["delta"]>0 for k in M)}
if a.output.exists(): raise FileExistsError(a.output)
a.output.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
