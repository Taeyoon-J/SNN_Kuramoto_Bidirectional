#!/usr/bin/env python3
"""Compare centered-gate pilot with the same-window raw SW0054 pilot."""
import argparse
import json
import math
from pathlib import Path

METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--centered-report", type=Path, required=True)
    p.add_argument("--raw-report", type=Path,
                   help="Optional same-evaluator SW0059 raw report; otherwise use validated SW0054 summary.")
    p.add_argument("--sw0054-summary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    centered = json.loads(args.centered_report.read_text(encoding="utf-8"))
    if centered.get("mode") != "centered_raw" or centered.get("ids") != [1320, 1351]:
        raise ValueError("centered result must be validated mode centered_raw on fixed IDs1320-1351")
    if args.raw_report:
        raw = json.loads(args.raw_report.read_text(encoding="utf-8"))
        if raw.get("mode") != "raw" or raw.get("checkpoint_sha256") != centered.get("checkpoint_sha256"):
            raise ValueError("raw and centered reports must use the same checkpoint")
        raw_metrics = raw["metrics"]
        raw_source = str(args.raw_report)
    else:
        summary = json.loads(args.sw0054_summary.read_text(encoding="utf-8"))
        if summary.get("count") != 32 or summary.get("ids") != [1320, 1351]:
            raise ValueError("SW0054 source must be the validated count-32 fixed subset")
        if summary.get("checkpoint_sha256") != centered.get("checkpoint_sha256"):
            raise ValueError("SW0054 raw reference checkpoint differs from centered evaluation")
        raw_metrics = summary["normal_reference"]["fixed_readouts"]["spike_cc"]["metrics"]
        raw_source = str(args.sw0054_summary)
    metrics = {}
    for name in METRICS:
        a, b = float(centered["metrics"][name]), float(raw_metrics[name])
        if not math.isfinite(a) or not math.isfinite(b):
            raise ValueError(f"non-finite {name}")
        metrics[name] = {"raw": b, "centered_raw": a, "delta": a - b}
    result = {"experiment": "SW0059 centered delayed-gate validation pilot",
              "comparison": "fixed seed0 checkpoint; IDs1320-1351; T256/settle64; spike-CC threshold .35",
              "prediction_ground_truth_independent": True,
              "centered_report": str(args.centered_report), "raw_reference": raw_source,
              "checkpoint_sha256": centered["checkpoint_sha256"], "metrics": metrics,
              "interpretation": "32-image validation pilot only; no retraining or general claim."}
    md = args.output.with_suffix(".md")
    if args.output.exists() or md.exists():
        raise FileExistsError(f"refusing overwrite: {args.output} or {md}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write("\n")
    with md.open("x", encoding="utf-8") as f:
        f.write("# SW0059 centered delayed-gate pilot\n\n")
        f.write("Fixed seed0 checkpoint, IDs1320-1351, T256/settle64, spike-CC threshold .35.\n\n")
        f.write("| Metric | Raw | Centered delayed | Delta |\n|---|---:|---:|---:|\n")
        for name, row in metrics.items():
            f.write(f"| {name} | {row['raw']:.6f} | {row['centered_raw']:.6f} | {row['delta']:.6f} |\n")
        f.write("\n32-image validation pilot only; no retraining or general claim.\n")
    print(f"wrote {args.output} and {md}")


if __name__ == "__main__":
    main()
