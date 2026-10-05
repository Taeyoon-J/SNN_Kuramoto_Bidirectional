#!/usr/bin/env python3
"""Fixed seed1 SW0084 pilot comparisons at saved epochs 5 and 10."""
import argparse
import json
import math
from pathlib import Path

METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")
ARMS = ("A", "B", "C", "D")
EPOCHS = ("05", "10")


def load_metrics(path):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("ids") != [1320, 1351] or report.get("images") != 32:
        raise ValueError(f"wrong fixed pilot IDs/count: {path}")
    config = report.get("inference", {})
    if (config.get("steps") != 256 or config.get("settle") != 64
            or float(config.get("membrane_vth", -1)) != .06
            or config.get("synchrony_thresholds") != [.35]):
        raise ValueError(f"readout mismatch: {path}")
    rows = [row for row in report.get("sweep", [])
            if row.get("affinity_mode") == "spike" and row.get("synchrony_threshold") == .35]
    if len(rows) != 1:
        raise ValueError(f"expected one spike CC row: {path}")
    metrics = rows[0]["scored_targets"]["our_hdf5"]["metrics"]
    result = {key: float(metrics[key]) for key in METRICS}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError(f"nonfinite metric: {path}")
    return result


def summarize(reports):
    required = {"baseline_epoch10"} | {f"{arm}_epoch{epoch}" for arm in ARMS for epoch in EPOCHS}
    if set(reports) != required:
        raise ValueError(f"need baseline and A-D at epochs 05/10; missing {sorted(required-set(reports))}")
    baseline = reports["baseline_epoch10"]
    rows = {}
    for arm in ARMS:
        rows[arm] = {}
        for epoch in EPOCHS:
            metrics = reports[f"{arm}_epoch{epoch}"]
            delta = {key: metrics[key] - baseline[key] for key in METRICS}
            rows[arm][epoch] = {"metrics": metrics, "delta_vs_sw0072_seed1": delta,
                                "all_three_strictly_improved": all(v > 0 for v in delta.values())}
    return {"experiment": "SW0084 jointly trainable RGB feature/core with phase-slot reconstruction",
            "baseline_sw0072_seed1": baseline, "arms": rows,
            "readout": "aligned IDs1320-1351, T256/settle64, vth .06, spike CC threshold .35",
            "selection": "report fixed epoch checkpoints; no per-image or threshold selection",
            "ground_truth_used_for_training": False}


def markdown(result):
    lines = ["# SW0084 fixed seed1 pilot", "",
             "| Arm | Epoch | FG-ARI | Δ | FG IoU | Δ | Object IoU | Δ | All three improve |",
             "|:---:|---:|---:|---:|---:|---:|---:|---:|:---:|"]
    for arm, epochs in result["arms"].items():
        for epoch, row in epochs.items():
            m, d = row["metrics"], row["delta_vs_sw0072_seed1"]
            values = [arm, epoch, f"{m['fg_ari']:.6f}", f"{d['fg_ari']:+.6f}",
                      f"{m['foreground_iou']:.6f}", f"{d['foreground_iou']:+.6f}",
                      f"{m['matched_object_iou']:.6f}", f"{d['matched_object_iou']:+.6f}",
                      "yes" if row["all_three_strictly_improved"] else "no"]
            lines.append("| " + " | ".join(values) + " |")
    lines += ["", result["readout"], result["selection"],
              "Interpret joint-loss and initialization controls together; this is one seed's pilot.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()
    root = Path(args.results_dir)
    names = ["baseline_epoch10"] + [f"{arm}_epoch{epoch}" for arm in ARMS for epoch in EPOCHS]
    for name in names:
        if not (root / f"{name}_seed1_n32.json").is_file():
            raise FileNotFoundError(root / f"{name}_seed1_n32.json")
    out_json, out_md = Path(args.output_json), Path(args.output_md)
    if out_json.exists() or out_md.exists():
        raise FileExistsError("refusing to overwrite summary")
    result = summarize({name: load_metrics(root / f"{name}_seed1_n32.json") for name in names})
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
