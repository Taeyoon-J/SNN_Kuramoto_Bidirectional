#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path

M = ("fg_ari", "foreground_iou", "matched_object_iou")
BASE = {"fg_ari": .7087568718226954, "foreground_iou": .46810477545343715,
        "matched_object_iou": .5098805140855838}
p = argparse.ArgumentParser(); p.add_argument("--result-dir", type=Path, required=True); p.add_argument("--output", type=Path, required=True); a = p.parse_args()
rows = {}
for tag in ("w0p3", "w1"):
    d = json.loads((a.result_dir / f"{tag}_seed1_n32.json").read_text())
    assert d["ids"] == [1320, 1351] and d["ground_truth_used_for_prediction"] is False
    found = [x for x in d["sweep"] if x["affinity_mode"] == "spike" and x["synchrony_threshold"] == .35]
    assert len(found) == 1
    scored = found[0]["scored_targets"]["our_hdf5"]
    metrics = {key: float(scored["metrics"][key]) for key in M}
    rows[tag] = {"metrics": metrics, "delta": {key: metrics[key] - BASE[key] for key in M},
                 "count_mae": float(scored["object_count"]["mae"])}
assert all(math.isfinite(v) for row in rows.values() for v in (*row["metrics"].values(), row["count_mae"]))
out = {"experiment": "SW0073 reconstruction-held frozen-graph stage1", "baseline_sw0072_seed1": BASE,
       "rows": rows, "advancing_arms": [tag for tag, row in rows.items() if all(row["delta"][key] > 0 for key in M)],
       "gate": "one arm must improve all three fixed seed1 metrics over SW0072"}
if a.output.exists(): raise FileExistsError(a.output)
a.output.write_text(json.dumps(out, indent=2) + "\n"); print(json.dumps(out, indent=2))
