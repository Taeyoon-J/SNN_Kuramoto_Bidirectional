"""Summarize the two pre-registered SW0057 readouts across seeds 0/1/2."""
import argparse
import json
import math
import statistics
from pathlib import Path

SEEDS = (0, 1, 2)
METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")
READOUTS = ("spike_cc_threshold_0p50", "membrane_spatial_sigma1p5_k10")


def summarize(model_root):
    reports = {}
    recipe = None
    common_assets = None
    for seed in SEEDS:
        path = model_root / f"SW0055_unique2500_s{seed}_e10_lr0p0003" / "sw0057_fixed_multireadout_long.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing seed{seed} report: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("schema_version") != 1 or report.get("experiment") != "SW0057 fixed multi-readout":
            raise ValueError(f"invalid SW0057 schema in {path}")
        if report.get("seed") != seed or report.get("ids") != [1320, 1639] or report.get("images") != 320:
            raise ValueError(f"seed/split/count mismatch in {path}")
        if report.get("split") != "fixed_hdf5_aligned_validation" or report.get("readout_contract", {}).get("ground_truth_used_for_prediction") is not False:
            raise ValueError(f"target provenance or prediction contract mismatch in {path}")
        if report.get("inference", {}).get("steps") != 1024 or report["inference"].get("settle") != 512:
            raise ValueError(f"window mismatch in {path}")
        for blob_name in ("checkpoint", "evaluator"):
            digest = report.get(blob_name, {}).get("sha256", "")
            if len(digest) != 64:
                raise ValueError(f"missing {blob_name} SHA256 in {path}")
        evaluator_code = report["evaluator"].get("dependency_code_sha256", "")
        gamma_sum = report.get("gamma", {}).get("sha256", "")
        manifest_sum = report.get("gamma", {}).get("manifest_sha256", "")
        if any(len(value) != 64 for value in (evaluator_code, gamma_sum, manifest_sum)):
            raise ValueError(f"missing evaluator/gamma/manifest provenance hash in {path}")
        current_assets = (evaluator_code, gamma_sum, manifest_sum)
        if common_assets is None:
            common_assets = current_assets
        elif current_assets != common_assets:
            raise ValueError("evaluator or validation assets differ across seeds")
        if recipe is None:
            recipe = report["inference"]
        elif report["inference"] != recipe:
            raise ValueError("inference recipe differs across seeds")
        rows = {row.get("readout"): row for row in report.get("rows", [])}
        if set(rows) != set(READOUTS):
            raise ValueError(f"readout set mismatch in {path}")
        for row in rows.values():
            for metric in METRICS:
                value = float(row["metrics"][metric])
                if not math.isfinite(value):
                    raise ValueError(f"non-finite {metric} in {path}")
        reports[seed] = rows

    result = {"schema_version": 1, "experiment": "SW0057 fixed multi-readout three-seed summary",
              "split": "fixed_hdf5_aligned_validation", "ids": [1320, 1639], "images": 320,
              "window": {"steps": 1024, "settle": 512}, "seeds": list(SEEDS), "readouts": {}}
    for readout in READOUTS:
        result["readouts"][readout] = {}
        for metric in METRICS:
            values = [float(reports[seed][readout]["metrics"][metric]) for seed in SEEDS]
            result["readouts"][readout][metric] = {
                "per_seed": values, "mean": statistics.mean(values), "std_sample": statistics.stdev(values)}
        for metric in ("predicted_object_count_mean", "predicted_foreground_fraction"):
            values = [float(reports[s][readout][metric]) for s in SEEDS]
            result["readouts"][readout][metric] = {
                "per_seed": values, "mean": statistics.mean(values), "std_sample": statistics.stdev(values)}
    return result


def markdown(result):
    lines = ["# SW0057 fixed multi-readout summary", "", "Fixed validation IDs 1320-1639; long T1024/settle512; seeds 0/1/2.",
             "Ground truth is used only for scoring. Mean and sample standard deviation are across seeds.", "",
             "| Readout | Metric | Mean | Sample SD | Seed 0 | Seed 1 | Seed 2 |", "|---|---|---:|---:|---:|---:|---:|"]
    for name, metrics in result["readouts"].items():
        for metric, values in metrics.items():
            if metric not in METRICS:
                continue
            lines.append(f"| {name} | {metric} | {values['mean']:.6f} | {values['std_sample']:.6f} | " +
                         " | ".join(f"{x:.6f}" for x in values["per_seed"]) + " |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    md_path = args.output.with_suffix(".md")
    if args.output.exists() or md_path.exists():
        raise FileExistsError(f"refusing to overwrite {args.output} or {md_path}")
    result = summarize(args.model_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    md_path.write_text(markdown(result), encoding="utf-8")
    print(f"wrote {args.output} and {md_path}")


if __name__ == "__main__":
    main()
