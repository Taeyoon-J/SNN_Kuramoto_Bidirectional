#!/usr/bin/env python3
"""Fixed, no-threshold-selection gate against the SW0072 seed1 pilot."""
import argparse
import json
import math
from pathlib import Path

METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")


def row(path):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("ids") != [1320, 1351] or report.get("images") != 32:
        raise ValueError(f"expected fixed aligned IDs1320-1351 in {path}")
    matches = [item for item in report.get("sweep", [])
               if item.get("synchrony_threshold") == 0.35]
    if len(matches) != 1:
        raise ValueError(f"expected one fixed threshold .35 in {path}")
    values = matches[0]["scored_targets"]["our_hdf5"]["metrics"]
    result = {name: float(values[name]) for name in METRICS}
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError(f"nonfinite metrics in {path}")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--w0p1", required=True)
    p.add_argument("--w1x", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    base = row(a.baseline)
    arms = {"w0p1": row(a.w0p1), "w1x": row(a.w1x)}
    candidates = {}
    for key, values in arms.items():
        delta = {metric: values[metric] - base[metric] for metric in METRICS}
        candidates[key] = {"metrics": values, "delta_vs_sw0072": delta,
                           "all_three_strictly_improved": all(v > 0 for v in delta.values())}
    result = {"pilot_slice": [1320, 1351], "readout": "spike CC synchrony threshold .35",
              "baseline": base, "candidates": candidates,
              "gate": "advance only if all three metrics strictly improve",
              "validation_labels_used_for_training_or_weight_choice": False}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
