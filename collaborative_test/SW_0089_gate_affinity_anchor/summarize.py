#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path

METRICS=("fg_ari","foreground_iou","matched_object_iou")

def load_metrics(path):
    x=json.loads(Path(path).read_text())
    if x.get("ids") != [1320,1351] or x.get("images") != 32: raise ValueError("split mismatch")
    c=x["inference"]
    if c["steps"] != 256 or c["settle"] != 64 or c["synchrony_thresholds"] != [.35]: raise ValueError("readout mismatch")
    rows=[r for r in x["sweep"] if r.get("affinity_mode")=="spike" and r.get("synchrony_threshold")==.35]
    if len(rows)!=1: raise ValueError("fixed row missing")
    m=rows[0]["scored_targets"]["our_hdf5"]["metrics"]
    out={k:float(m[k]) for k in METRICS}
    if not all(math.isfinite(v) for v in out.values()): raise ValueError("nonfinite metric")
    return out

def summarize(rows):
    if set(rows)!={"baseline","H","I","J"}: raise ValueError("need baseline and H/I/J")
    baseline=rows["baseline"]; arms={}
    for arm in "HIJ":
        delta={k:rows[arm][k]-baseline[k] for k in METRICS}
        arms[arm]={"metrics":rows[arm],"delta_vs_sw0072_seed1":delta,
                   "all_three_strictly_improved":all(v>0 for v in delta.values())}
    return {"experiment":"SW0089 classifier-aligned gate-affinity anchor",
            "baseline_sw0072_seed1":baseline,"arms":arms,
            "readout":"IDs1320-1351 T256/settle64 vth.06 spike-CC .35",
            "ground_truth_used_for_training":False}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--results-dir",required=True); p.add_argument("--output",required=True); a=p.parse_args()
    root=Path(a.results_dir); out=Path(a.output)
    if out.exists(): raise FileExistsError(out)
    result=summarize({n:load_metrics(root/f"{n}_seed1_n32.json") for n in ("baseline","H","I","J")})
    out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
