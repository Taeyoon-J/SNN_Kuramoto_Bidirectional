"""Summarize common SW0044 validation configurations across seeds 0/1/2."""
import argparse
import json
import math
import statistics
from pathlib import Path


SEEDS = (0, 1, 2)
WINDOWS = ("short", "long")
METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")


def _canonical_sigma(value):
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"inf", "infinity"}:
        return "inf"
    numeric = float(value)
    return "inf" if math.isinf(numeric) else round(numeric, 8)


def _row_key(row):
    # Earlier SW0042 base evaluations predate these fields and represent S-only.
    mode = row.get("affinity_mode", "spike")
    if mode not in {"spike", "spike_spatial", "spatial_only"}:
        raise ValueError(f"Unknown affinity_mode in validation row: {mode!r}")
    return (mode, _canonical_sigma(row.get("spatial_sigma")),
            round(float(row["synchrony_threshold"]), 8))


def _mean_std(values):
    return {"mean": statistics.mean(values), "std_sample": statistics.stdev(values),
            "per_seed": values}


def summarize_window(model_root, window):
    loaded = {}
    base_loaded = {}
    base_names = {
        "short": "validation_short_T256_settle64.json",
        "long": "validation_long_T1024_settle512.json",
    }
    for seed in SEEDS:
        seed_root = model_root / f"SW_0042_HDF5_aligned_BIM6_s{seed}"
        path = seed_root / f"spatial_affinity_{window}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing required seed{seed} {window} validation JSON: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("split") != "validation" or report.get("ground_truth_used_for_prediction") is not False:
            raise ValueError(f"Not a prediction-first validation report: {path}")
        if report.get("target_sources", {}).get("our_hdf5") is None:
            raise ValueError(f"Expected HDF5 target metrics in validation report: {path}")
        rows = {}
        for row in report.get("sweep", []):
            key = _row_key(row)
            if key in rows:
                raise ValueError(f"Duplicate configuration {key} in {path}")
            scored = row.get("scored_targets", {}).get("our_hdf5")
            if scored is None:
                raise ValueError(f"Missing our_hdf5 scores for {key} in {path}")
            rows[key] = {"row": row, "scored": scored}
        loaded[seed] = {"path": str(path), "rows": rows}

        base_path = seed_root / base_names[window]
        if not base_path.is_file():
            raise FileNotFoundError(
                f"Missing required seed{seed} SW0042 {window} baseline validation JSON: {base_path}")
        base_report = json.loads(base_path.read_text(encoding="utf-8"))
        if base_report.get("split") != "validation" or base_report.get("ground_truth_used_for_prediction") is not False:
            raise ValueError(f"Not a prediction-first SW0042 validation report: {base_path}")
        if base_report.get("target_sources", {}).get("our_hdf5") is None:
            raise ValueError(f"Expected HDF5 target metrics in SW0042 report: {base_path}")
        base_rows = {}
        for row in base_report.get("sweep", []):
            key = _row_key(row)
            scored = row.get("scored_targets", {}).get("our_hdf5")
            if key[0] == "spike" and key[1] is None and scored is not None:
                base_rows[key] = {"row": row, "scored": scored}
        base_loaded[seed] = {"path": str(base_path), "rows": base_rows}

    common = set.intersection(*(set(loaded[seed]["rows"]) for seed in SEEDS))
    if not common:
        raise ValueError(f"No common (affinity_mode, sigma, threshold) configurations across seeds for {window}.")
    table = []
    for mode, sigma, threshold in sorted(common, key=lambda key: (key[0], str(key[1]), key[2])):
        seed_rows = [loaded[seed]["rows"][(mode, sigma, threshold)] for seed in SEEDS]
        metric_summary = {}
        for metric in METRICS:
            values = [float(entry["scored"]["metrics"][metric]) for entry in seed_rows]
            metric_summary[metric] = _mean_std(values)
        diag_values = {
            "groups_per_image_mean": [float(entry["row"]["predicted_object_count"]["mean"])
                                       for entry in seed_rows],
            "predicted_foreground_fraction": [float(entry["row"]["predicted_foreground_fraction"])
                                               for entry in seed_rows],
            "empty_image_count": [int(entry["row"]["predicted_object_count"]["empty_image_count"])
                                  for entry in seed_rows],
        }
        table.append({
            "affinity_mode": mode,
            "spatial_sigma": sigma,
            "synchrony_threshold": threshold,
            "metrics": metric_summary,
            "diagnostics": {key: _mean_std(values) for key, values in diag_values.items()},
        })
    base_common = set.intersection(*(set(base_loaded[seed]["rows"]) for seed in SEEDS))
    base_table = []
    for mode, sigma, threshold in sorted(base_common, key=lambda key: key[2]):
        seed_rows = [base_loaded[seed]["rows"][(mode, sigma, threshold)] for seed in SEEDS]
        metrics = {
            metric: _mean_std([float(entry["scored"]["metrics"][metric]) for entry in seed_rows])
            for metric in METRICS
        }
        base_table.append({
            "affinity_mode": mode,
            "spatial_sigma": sigma,
            "synchrony_threshold": threshold,
            "metrics": metrics,
        })
    per_seed_keys = {seed: len(loaded[seed]["rows"]) for seed in SEEDS}
    return {
        "window": window,
        "validation_only": True,
        "selection_performed": False,
        "target": "our_hdf5",
        "seeds": list(SEEDS),
        "input_files": {seed: loaded[seed]["path"] for seed in SEEDS},
        "sw0042_baseline_input_files": {seed: base_loaded[seed]["path"] for seed in SEEDS},
        "configurations_per_seed": per_seed_keys,
        "common_configuration_count": len(table),
        "summary_rows": table,
        "sw0042_spike_control_common_count": len(base_table),
        "sw0042_spike_control_rows": base_table,
    }


def summarize(model_root):
    root = Path(model_root)
    return {
        "model_root": str(root),
        "source": "SW0044 readouts and SW0042 spike controls from HDF5-aligned seed checkpoints on validation IDs 1320-1639",
        "aggregation": "per common configuration; mean and sample standard deviation across seeds 0/1/2",
        "ground_truth_used_for_prediction": False,
        "selection_performed": False,
        "windows": {window: summarize_window(root, window) for window in WINDOWS},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True,
                        help="Directory containing SW_0042_HDF5_aligned_BIM6_s{0,1,2} folders.")
    parser.add_argument("--output", required=True, help="Output summary JSON path.")
    args = parser.parse_args()
    report = summarize(Path(args.model_root))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "short_rows": report["windows"]["short"]["common_configuration_count"],
                      "long_rows": report["windows"]["long"]["common_configuration_count"]}, indent=2))


if __name__ == "__main__":
    main()
