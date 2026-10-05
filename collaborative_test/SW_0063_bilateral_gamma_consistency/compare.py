#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path

METRICS=("fg_ari","foreground_iou","matched_object_iou")
p=argparse.ArgumentParser(); p.add_argument("--report",type=Path,required=True); p.add_argument("--baseline",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
r=json.loads(a.report.read_text()); b=json.loads(a.baseline.read_text())
assert r["ids"]==[1320,1351] and r["images"]==32 and r["ground_truth_used_for_prediction"] is False
rows=[x for x in r["sweep"] if x["affinity_mode"]=="spike" and x["synchrony_threshold"]==0.35]; assert len(rows)==1
c=rows[0]["scored_targets"]["our_hdf5"]["metrics"]; n=b["normal_reference"]["fixed_readouts"]["spike_cc"]["metrics"]
m={k:{"native":float(n[k]),"bilateral":float(c[k]),"delta":float(c[k])-float(n[k])} for k in METRICS}
assert all(math.isfinite(v) for x in m.values() for v in x.values())
o={"experiment":"SW0063 bilateral gamma consistency pilot","metrics":m,"all_three_improved":all(x["delta"]>0 for x in m.values()),"prediction_ground_truth_independent":True}
if a.output.exists(): raise FileExistsError(a.output)
a.output.write_text(json.dumps(o,indent=2)+"\n"); print(json.dumps(o,indent=2))
