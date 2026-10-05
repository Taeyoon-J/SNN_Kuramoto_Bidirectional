#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--raw-summary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    raw = json.loads(args.raw_summary.read_text(encoding="utf-8"))
    if report.get("ids") != [1320, 1351] or report.get("images") != 32:
        raise ValueError("expected exact 32-image fixed pilot")
    if report.get("ground_truth_used_for_prediction") is not False:
        raise ValueError("prediction must be ground-truth independent")
    rows = [x for x in report["sweep"] if x["affinity_mode"] == "spike" and x["synchrony_threshold"] == 0.35]
    if len(rows) != 1:
        raise ValueError("expected one fixed spike row")
    candidate = rows[0]["scored_targets"]["our_hdf5"]["metrics"]
    baseline = raw["normal_reference"]["fixed_readouts"]["spike_cc"]["metrics"]
    metrics = {}
    for name in METRICS:
        before, after = float(baseline[name]), float(candidate[name])
        if not math.isfinite(before) or not math.isfinite(after):
            raise ValueError(f"non-finite {name}")
        metrics[name] = {"autoencoder_gamma": before, "rgb_patch_gamma": after, "delta": after - before}
    result = {
        "experiment": "SW0061 label-free RGB patch feature pilot",
        "comparison": "same checkpoint and fixed IDs1320-1351; T256/settle64; threshold .35",
        "prediction_ground_truth_independent": True,
        "metrics": metrics,
        "all_three_improved": all(x["delta"] > 0 for x in metrics.values()),
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
