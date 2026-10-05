#!/usr/bin/env python3
"""Apply the preregistered all-three seed1 pilot gate without threshold selection."""
import argparse
import json
import math
from pathlib import Path

METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")


def read_row(path):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("ids") != [1320, 1351] or report.get("images") != 32:
        raise ValueError(f"not the fixed SW0080 pilot slice: {path}")
    matches = [row for row in report.get("sweep", []) if row.get("synchrony_threshold") == 0.35]
    if len(matches) != 1:
        raise ValueError(f"expected one fixed .35 readout row: {path}")
    metrics = matches[0]["scored_targets"]["our_hdf5"]["metrics"]
    if any(not math.isfinite(float(metrics[key])) for key in METRICS):
        raise ValueError(f"nonfinite metric in {path}")
    return {key: float(metrics[key]) for key in METRICS}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--r0p3", required=True)
    p.add_argument("--r1p0", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    output = Path(a.output)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    baseline = read_row(a.baseline)
    candidates = {"r0p3": read_row(a.r0p3), "r1p0": read_row(a.r1p0)}
    rows = {}
    for name, values in candidates.items():
        delta = {key: values[key] - baseline[key] for key in METRICS}
        rows[name] = {"metrics": values, "delta_vs_sw0072_seed1": delta,
                      "advance_gate_all_three_strictly_positive": all(x > 0 for x in delta.values())}
    report = {"baseline": baseline, "candidates": rows,
              "gate": "advance only an arm with positive deltas on all three fixed metrics",
              "threshold_or_parameter_selected_from_scores": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
