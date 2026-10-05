#!/usr/bin/env python3
import argparse, json
from pathlib import Path

M = ("fg_ari", "foreground_iou", "matched_object_iou")
p = argparse.ArgumentParser(); p.add_argument("--result-dir", type=Path, required=True); p.add_argument("--output", type=Path, required=True); a = p.parse_args()
rows = {}
for tag in ("w0p3", "w1"):
    rows[tag] = {}
    for mode, suffix in (("joint", ""), ("feature_only", "_feature_only"), ("core_only", "_core_only")):
        d = json.loads((a.result_dir / f"{tag}{suffix}_seed1_n32.json").read_text())
        found = [x for x in d["sweep"] if x["affinity_mode"] == "spike" and x["synchrony_threshold"] == .35]
        assert d["ids"] == [1320, 1351] and d["ground_truth_used_for_prediction"] is False and len(found) == 1
        metrics = found[0]["scored_targets"]["our_hdf5"]["metrics"]
        rows[tag][mode] = {key: float(metrics[key]) for key in M}
out = {"experiment": "SW0073 component swap diagnosis", "rows": rows,
       "interpretation": "feature_only isolates learned RGB features in SW0072 core; core_only isolates the jointly trained core on native gamma"}
if a.output.exists(): raise FileExistsError(a.output)
a.output.write_text(json.dumps(out, indent=2) + "\n"); print(json.dumps(out, indent=2))
