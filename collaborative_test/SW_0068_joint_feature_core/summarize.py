#!/usr/bin/env python3
import argparse,json,math
from pathlib import Path

METRICS=("fg_ari","foreground_iou","matched_object_iou")
BASE={"fg_ari":.6316519272381114,"foreground_iou":.3852088450351232,
      "matched_object_iou":.46168352926256573}

def main():
 p=argparse.ArgumentParser(); p.add_argument("--result-dir",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
 rows={}
 for tag in ("lr3e6","lr3e5"):
  d=json.loads((a.result_dir/f"{tag}_seed1_n32.json").read_text()); assert d["ids"]==[1320,1351] and d["ground_truth_used_for_prediction"] is False
  found=[x for x in d["sweep"] if x["affinity_mode"]=="spike" and x["synchrony_threshold"]==.35 and x.get("target_foreground") is None]; assert len(found)==1
  s=found[0]["scored_targets"]["our_hdf5"]
  metrics={k:float(s["metrics"][k]) for k in METRICS}
  rows[tag]={"metrics":metrics,"delta":{k:metrics[k]-BASE[k] for k in METRICS},
             "count_mae":float(s["object_count"]["mae"]),
             "predicted_count_mean":float(s["object_count"]["predicted_mean"]),
             "predicted_foreground_fraction":float(found[0]["predicted_foreground_fraction"])}
 out={"experiment":"SW0068 joint feature/core stage1","baseline":BASE,"rows":rows,
      "advancing_arms":[tag for tag,r in rows.items() if all(r["delta"][k]>0 for k in METRICS)],
      "gate":"one arm must improve all three fixed seed1 metrics"}
 assert all(math.isfinite(v) for r in rows.values() for v in [*r["metrics"].values(),r["count_mae"]])
 if a.output.exists(): raise FileExistsError(a.output)
 a.output.write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2))

if __name__=="__main__": main()
