#!/usr/bin/env python3
import argparse,json,math
from pathlib import Path
M=("fg_ari","foreground_iou","matched_object_iou"); BASE={"fg_ari":.7087568718226954,"foreground_iou":.46810477545343715,"matched_object_iou":.5098805140855838}
p=argparse.ArgumentParser();p.add_argument("--result-dir",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();rows={}
for tag in ("p1k","p10k"):
 d=json.loads((a.result_dir/f"{tag}_seed1_n32.json").read_text());assert d["ids"]==[1320,1351] and d["ground_truth_used_for_prediction"] is False
 q=[x for x in d["sweep"] if x["affinity_mode"]=="spike" and x["synchrony_threshold"]==.35];assert len(q)==1;s=q[0]["scored_targets"]["our_hdf5"];m={k:float(s["metrics"][k]) for k in M};rows[tag]={"metrics":m,"delta":{k:m[k]-BASE[k] for k in M},"count_mae":float(s["object_count"]["mae"])}
assert all(math.isfinite(v) for r in rows.values() for v in (*r["metrics"].values(),r["count_mae"]))
out={"experiment":"SW0075 activity-profile anchored features","baseline_sw0072_seed1":BASE,"rows":rows,"advancing_arms":[t for t,r in rows.items() if all(r["delta"][k]>0 for k in M)],"gate":"one arm must improve all three fixed seed1 metrics over SW0072"}
if a.output.exists():raise FileExistsError(a.output)
a.output.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
