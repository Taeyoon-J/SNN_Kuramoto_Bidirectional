#!/usr/bin/env python3
import argparse,json,math
from pathlib import Path
M=("fg_ari","foreground_iou","matched_object_iou")
p=argparse.ArgumentParser();p.add_argument("--result-dir",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args(); rows={}
for seed in (1,2):
 rows[str(seed)]={}
 for kind in ("baseline","candidate"):
  r=json.loads((a.result_dir/f"{kind}_seed{seed}_n32.json").read_text()); assert r["ids"]==[1320,1351] and r["ground_truth_used_for_prediction"] is False
  x=[q for q in r["sweep"] if q["affinity_mode"]=="spike" and q["synchrony_threshold"]==.35];assert len(x)==1
  rows[str(seed)][kind]={k:float(x[0]["scored_targets"]["our_hdf5"]["metrics"][k]) for k in M}
 rows[str(seed)]["delta"]={k:rows[str(seed)]["candidate"][k]-rows[str(seed)]["baseline"][k] for k in M}
out={"experiment":"SW0066 graph-init0 stage1","rows":rows,"advance":all(rows[str(s)]["delta"][k]>0 for s in (1,2) for k in M),"gate":"both seeds must improve all three fixed metrics"}
assert all(math.isfinite(v) for s in rows.values() for kind in s.values() for v in kind.values())
if a.output.exists():raise FileExistsError(a.output)
a.output.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
