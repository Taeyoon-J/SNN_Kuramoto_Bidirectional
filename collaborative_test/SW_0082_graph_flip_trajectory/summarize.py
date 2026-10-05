#!/usr/bin/env python3
"""Summarize fixed SW0082 epoch readouts against its SW0072 baseline."""
import argparse
import json
import math
from pathlib import Path

METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")


def read_metrics(path):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("ids") != [1320, 1351] or report.get("images") != 32:
        raise ValueError(f"unexpected pilot IDs/count in {path}")
    matching = [row for row in report.get("sweep", [])
                if row.get("synchrony_threshold") == 0.35]
    if len(matching) != 1:
        raise ValueError(f"expected one fixed threshold .35 in {path}")
    values = matching[0]["scored_targets"]["our_hdf5"]["metrics"]
    result = {key: float(values[key]) for key in METRICS}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError(f"nonfinite metric in {path}")
    return result


def summarize(baseline_path, epoch_paths, sw0081_path):
    if set(epoch_paths) != {1, 2, 3, 4, 5}:
        raise ValueError("exactly epochs 1 through 5 are required")
    baseline = read_metrics(baseline_path)
    sw0081 = read_metrics(sw0081_path)
    rows = {}
    for epoch in range(1, 6):
        metrics = read_metrics(epoch_paths[epoch])
        delta = {key: metrics[key] - baseline[key] for key in METRICS}
        rows[str(epoch)] = {"metrics": metrics, "delta_vs_sw0072": delta,
                            "advance": all(value > 0 for value in delta.values())}
    final_delta = {key: rows["5"]["metrics"][key] - sw0081[key] for key in METRICS}
    if any(abs(value) > 1e-5 for value in final_delta.values()):
        raise ValueError(f"epoch5 does not reproduce SW0081 fixed-readout metrics: {final_delta}")
    return {"baseline": baseline, "sw0081_reference": sw0081,
            "epoch5_delta_vs_sw0081": final_delta, "epochs": rows,
            "readout": "aligned IDs1320-1351, T256/settle64, spike CC threshold .35",
            "advance_rule": "an epoch advances only when all three deltas are strictly positive",
            "validation_threshold_or_epoch_postselected_for_training": False}


def markdown(report):
    lines = ["# SW0082 fixed pilot trajectory", "",
             "| Epoch | FG-ARI | Δ | Foreground IoU | Δ | Matched-object IoU | Δ | Advance |",
             "|---:|---:|---:|---:|---:|---:|---:|:---:|"]
    for epoch, row in report["epochs"].items():
        m, d = row["metrics"], row["delta_vs_sw0072"]
        lines.append(f"| {epoch} | {m['fg_ari']:.6f} | {d['fg_ari']:+.6f} | "
                     f"{m['foreground_iou']:.6f} | {d['foreground_iou']:+.6f} | "
                     f"{m['matched_object_iou']:.6f} | {d['matched_object_iou']:+.6f} | "
                     f"{'yes' if row['advance'] else 'no'} |")
    lines += ["", report["readout"], report["advance_rule"],
              "No threshold or epoch was selected using training labels.", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--sw0081-reference", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=True)
    a = p.parse_args()
    directory = Path(a.results_dir)
    paths = {epoch: directory / f"epoch{epoch}_seed1_n32.json" for epoch in range(1, 6)}
    baseline = directory / "baseline_seed1_n32.json"
    for path in [baseline, *paths.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)
    json_out, md_out = Path(a.output_json), Path(a.output_md)
    if json_out.exists() or md_out.exists():
        raise FileExistsError("refusing to overwrite summary output")
    result = summarize(baseline, paths, a.sw0081_reference)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md_out.write_text(markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
