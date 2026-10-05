#!/usr/bin/env python3
import argparse, json, math, statistics
from pathlib import Path

M = ("fg_ari", "foreground_iou", "matched_object_iou")
p = argparse.ArgumentParser()
p.add_argument("--candidate-dir", type=Path, required=True)
p.add_argument("--baseline", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
old = json.loads(a.baseline.read_text())
baseline = {key: float(old["metrics"][key]["mean"]) for key in M}
rows = {"0": {key: float(old["rows"]["0"][key]) for key in M}}
for seed in (1, 2):
    report = json.loads((a.candidate_dir / f"seed{seed}_long.json").read_text())
    assert report["ids"] == [1320, 1639] and report["ground_truth_used_for_prediction"] is False
    found = [row for row in report["sweep"] if row["affinity_mode"] == "spike" and row["synchrony_threshold"] == .5]
    assert len(found) == 1
    metrics = found[0]["scored_targets"]["our_hdf5"]["metrics"]
    rows[str(seed)] = {key: float(metrics[key]) for key in M}
means = {key: statistics.mean(rows[str(seed)][key] for seed in range(3)) for key in M}
deltas = {key: means[key] - baseline[key] for key in M}
assert all(math.isfinite(value) for value in means.values())
out = {"experiment": "SW0072 frozen trained graph full mean", "rows": rows,
       "means": means, "baseline_sw0070": baseline, "delta": deltas,
       "improves_all_three": all(value > 0 for value in deltas.values())}
if a.output.exists():
    raise FileExistsError(a.output)
a.output.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
