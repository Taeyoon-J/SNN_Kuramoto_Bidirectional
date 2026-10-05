#!/usr/bin/env python3
"""Summarize fixed membrane-threshold inference sweep with event diagnostics."""
import argparse
import json
import math
from pathlib import Path

METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")
EVENTS = ("binary_event_rate", "binary_always_on_fraction",
          "binary_constant_history_fraction", "binary_temporal_std_mean")
EXPECTED = {"v006": .06, "v05": .5, "v10": 1.0, "v20": 2.0}


def read_report(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract(report, expected_vth):
    if report.get("ids") != [1320, 1351] or report.get("images") != 32:
        raise ValueError("all rows must score aligned IDs1320-1351 (32 images)")
    inference = report.get("inference", {})
    if (inference.get("steps") != 256 or inference.get("settle") != 64
            or not math.isclose(float(inference.get("membrane_vth", -1)), expected_vth)):
        raise ValueError("inference protocol or membrane threshold mismatch")
    if inference.get("synchrony_thresholds") != [.35] or inference.get("affinity_modes") != ["spike"]:
        raise ValueError("expected fixed spike affinity and synchrony threshold .35")
    rows = [row for row in report.get("sweep", [])
            if row.get("affinity_mode") == "spike" and row.get("synchrony_threshold") == .35]
    if len(rows) != 1:
        raise ValueError("expected one fixed spike readout row at threshold .35")
    metrics = rows[0]["scored_targets"]["our_hdf5"]["metrics"]
    event = report.get("event_diagnostics")
    if not event:
        raise ValueError("missing GT-free event saturation diagnostics")
    if event.get("shape_after_settle") != [32, 4, 256, 192] or event.get("binary_values_only") is not True:
        raise ValueError("event diagnostics must be binary [32,4,256,192] component crossings")
    result = {key: float(metrics[key]) for key in METRICS}
    result.update({key: float(event[key]) for key in EVENTS})
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("nonfinite metric or event diagnostic")
    return result


def summarize_reports(reports):
    if set(reports) != set(EXPECTED):
        raise ValueError("all four fixed thresholds are required")
    rows = {tag: extract(reports[tag], vth) for tag, vth in EXPECTED.items()}
    base = rows["v006"]
    for tag, values in rows.items():
        values["metric_delta_vs_v006"] = {metric: values[metric] - base[metric] for metric in METRICS}
        values["all_three_metrics_improve_vs_v006"] = all(
            value > 0 for value in values["metric_delta_vs_v006"].values())
    return {
        "experiment": "SW0083 inference-only membrane threshold recalibration",
        "fixed_ids": [1320, 1351], "steps": 256, "settle": 64,
        "readout": "spike connected components; synchrony threshold .35",
        "thresholds": {tag: {"membrane_vth": EXPECTED[tag], **values}
                       for tag, values in rows.items()},
        "prediction_used_validation_labels": False,
        "interpretation": "thresholds are fixed globally per run; no per-image adaptive thresholding",
    }


def markdown(result):
    cols = ("binary_event_rate", "binary_always_on_fraction",
            "binary_constant_history_fraction", "fg_ari", "foreground_iou", "matched_object_iou")
    lines = ["# SW0083 membrane-threshold recalibration", "",
             "| vth | Event rate | Always-on | Constant histories | FG-ARI | FG IoU | Object IoU | All metrics improve |",
             "|---:|---:|---:|---:|---:|---:|---:|:---:|"]
    for tag, row in result["thresholds"].items():
        lines.append("| " + " | ".join([
            f"{row['membrane_vth']:g}", *(f"{row[c]:.6f}" for c in cols),
            "yes" if row["all_three_metrics_improve_vs_v006"] else "no",
        ]) + " |")
    lines += ["", "Fixed aligned 32-image pilot; no per-image threshold adaptation.",
              "Event diagnostics are computed from binary component spike histories after settle, before scoring.", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=True)
    a = p.parse_args()
    root = Path(a.results_dir)
    reports = {tag: read_report(root / f"{tag}_seed1_n32.json") for tag in EXPECTED}
    out_json, out_md = Path(a.output_json), Path(a.output_md)
    if out_json.exists() or out_md.exists():
        raise FileExistsError("refusing to overwrite summary files")
    result = summarize_reports(reports)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
