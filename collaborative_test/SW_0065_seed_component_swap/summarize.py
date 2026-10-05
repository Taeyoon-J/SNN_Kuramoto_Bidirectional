#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path

METRICS=("fg_ari","foreground_iou","matched_object_iou")
VARIANTS=("base0","base2","drive0","graph0","kuramoto0")
p=argparse.ArgumentParser(); p.add_argument("--result-dir",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
rows={}
for variant in VARIANTS:
 r=json.loads((a.result_dir/f"{variant}_n32.json").read_text())
 assert r["ids"]==[1320,1351] and r["images"]==32 and r["ground_truth_used_for_prediction"] is False
 x=[q for q in r["sweep"] if q["affinity_mode"]=="spike" and q["synchrony_threshold"]==0.35]; assert len(x)==1
 m=x[0]["scored_targets"]["our_hdf5"]["metrics"]; rows[variant]={k:float(m[k]) for k in METRICS}
 assert all(math.isfinite(v) for v in rows[variant].values())
out={"experiment":"SW0065 causal seed component swaps","ids":[1320,1351],"rows":rows,"delta_from_base2":{v:{k:rows[v][k]-rows["base2"][k] for k in METRICS} for v in VARIANTS if v not in ("base0","base2")},"interpretation":"validation-only component swap diagnostic; no training claim"}
if a.output.exists(): raise FileExistsError(a.output)
a.output.write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2))
