#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")
BASELINE = {
    "fg_ari": 0.7087568718226954,
    "foreground_iou": 0.46810477545343715,
    "matched_object_iou": 0.5098805140855838,
}

parser = argparse.ArgumentParser()
parser.add_argument("--result-dir", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
rows = {}
for tag in ("g1k", "g10k"):
    data = json.loads((args.result_dir / f"{tag}_seed1_n32.json").read_text())
    assert data["ids"] == [1320, 1351]
    assert data["ground_truth_used_for_prediction"] is False
    matches = [
        row for row in data["sweep"]
        if row["affinity_mode"] == "spike" and row["synchrony_threshold"] == 0.35
    ]
    assert len(matches) == 1
    scored = matches[0]["scored_targets"]["our_hdf5"]
    metrics = {key: float(scored["metrics"][key]) for key in METRICS}
    rows[tag] = {
        "metrics": metrics,
        "delta": {key: metrics[key] - BASELINE[key] for key in METRICS},
        "count_mae": float(scored["object_count"]["mae"]),
    }
assert all(
    math.isfinite(value)
    for row in rows.values()
    for value in (*row["metrics"].values(), row["count_mae"])
)
output = {
    "experiment": "SW0076 graph-output anchored features",
    "baseline_sw0072_seed1": BASELINE,
    "rows": rows,
    "advancing_arms": [
        tag for tag, row in rows.items()
        if all(row["delta"][key] > 0 for key in METRICS)
    ],
    "gate": "one arm must improve all three fixed seed1 metrics over SW0072",
}
if args.output.exists():
    raise FileExistsError(args.output)
args.output.write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps(output, indent=2))
