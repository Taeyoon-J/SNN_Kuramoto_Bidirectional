#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--feature-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    manifest = json.loads(args.feature_manifest.read_text(encoding="utf-8"))
    if report.get("ids") != [1320, 1351] or report.get("images") != 32:
        raise ValueError("expected fixed IDs1320-1351/count32")
    if report.get("ground_truth_used_for_prediction") is not False:
        raise ValueError("prediction must be ground-truth independent")
    if manifest.get("pca_fit_ids") != [0, 999] or manifest.get("uses_masks_counts_or_labels") is not False:
        raise ValueError("DINO PCA must be training-only and label-free")
    rows = [x for x in report["sweep"] if x["affinity_mode"] == "spike" and x["synchrony_threshold"] == 0.35]
    if len(rows) != 1:
        raise ValueError("expected one fixed spike threshold row")
    candidate = rows[0]["scored_targets"]["our_hdf5"]["metrics"]
    native = baseline["normal_reference"]["fixed_readouts"]["spike_cc"]["metrics"]
    metrics = {}
    for name in METRICS:
        before, after = float(native[name]), float(candidate[name])
        if not math.isfinite(before) or not math.isfinite(after):
            raise ValueError(f"non-finite {name}")
        metrics[name] = {"native_epoch25": before, "dino_epoch5": after, "delta": after - before}
    result = {
        "experiment": "SW0062 DINOv2 core-adaptation direction pilot",
        "comparison": "seed0; IDs1320-1351; T256/settle64; spike-CC threshold .35",
        "candidate_epochs": 5,
        "baseline_epochs": 25,
        "prediction_ground_truth_independent": True,
        "metrics": metrics,
        "all_three_above_native_epoch25": all(x["delta"] > 0 for x in metrics.values()),
        "interpretation": "Direction pilot only; unequal epoch count prevents a final controlled claim.",
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
