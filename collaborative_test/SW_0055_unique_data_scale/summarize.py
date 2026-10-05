#!/usr/bin/env python3
"""Validate six SW0055 seed/window reports and summarize prespecified thresholds."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")
THRESHOLDS = (0.50, 0.35)
WINDOWS = {"short": (256, 64), "long": (1024, 512)}
BASELINES = {
    0.50: {"fg_ari": 0.562902, "foreground_iou": 0.381328, "matched_object_iou": 0.356857},
    0.35: {"fg_ari": 0.560105, "foreground_iou": 0.381852, "matched_object_iou": 0.349261},
}


def valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_tree(obj: Any, path: str = "root") -> None:
    if isinstance(obj, float) and not math.isfinite(obj):
        raise ValueError(f"non-finite value at {path}")
    if isinstance(obj, dict):
        for key, value in obj.items():
            finite_tree(value, f"{path}.{key}")
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            finite_tree(value, f"{path}[{i}]")


def validate_report(report: dict[str, Any], seed: int, window: str, checkpoint: Path) -> None:
    steps, settle = WINDOWS[window]
    if report.get("split") != "validation" or report.get("ids") != [1320, 1639] or report.get("images") != 320:
        raise ValueError(f"seed{seed}/{window}: expected validation IDs 1320-1639, count 320")
    report_checkpoint = str(report.get("checkpoint", "")).replace("\\", "/")
    expected_suffix = f"SW0055_unique2500_s{seed}_e10_lr0p0003/core.pt"
    if not report_checkpoint.endswith(expected_suffix):
        raise ValueError(f"seed{seed}/{window}: checkpoint path mismatch")
    if report.get("ground_truth_used_for_prediction") is not False:
        raise ValueError(f"seed{seed}/{window}: prediction protocol missing/violated")
    infer = report.get("inference", {})
    if (infer.get("steps"), infer.get("settle")) != (steps, settle):
        raise ValueError(f"seed{seed}/{window}: expected T{steps}/settle{settle}")
    gamma = report.get("gamma_source", {})
    if gamma.get("global_start") != 1320 or gamma.get("manifest", {}).get("image_ids") != [1320, 1639]:
        raise ValueError(f"seed{seed}/{window}: aligned gamma manifest mismatch")
    gamma_sha = gamma.get("manifest", {}).get("gamma_sha256")
    if not valid_sha256(gamma_sha):
        raise ValueError(f"seed{seed}/{window}: aligned validation gamma hash missing")
    targets = report.get("target_sources", {})
    if targets.get("our_hdf5", {}).get("ids") != [1320, 1639]:
        raise ValueError(f"seed{seed}/{window}: HDF5 target provenance mismatch")
    sweep = report.get("sweep")
    if not isinstance(sweep, list):
        raise ValueError(f"seed{seed}/{window}: missing sweep")
    for threshold in THRESHOLDS:
        row = next((x for x in sweep if math.isclose(float(x.get("synchrony_threshold", -1)), threshold, abs_tol=1e-9)), None)
        if row is None:
            raise ValueError(f"seed{seed}/{window}: missing fixed threshold {threshold}")
        metrics = row.get("scored_targets", {}).get("our_hdf5", {}).get("metrics", {})
        if not set(METRICS).issubset(metrics):
            raise ValueError(f"seed{seed}/{window}: threshold {threshold} metrics missing")
    finite_tree(report)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint}")


def summarize(model_root: Path) -> dict[str, Any]:
    observations: dict[str, dict[str, dict[float, dict[str, float]]]] = {w: {} for w in WINDOWS}
    checkpoint_hashes: dict[str, str] = {}
    code_hashes, gamma_hashes, validation_gamma_hashes = set(), set(), set()
    expected_train_ids = list(range(1000)) + list(range(1640, 3140))
    for seed in range(3):
        out = model_root / f"SW0055_unique2500_s{seed}_e10_lr0p0003"
        manifest_path = out / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing seed manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("seed") != seed or manifest.get("train_ids") != expected_train_ids:
            raise ValueError(f"seed{seed}: manifest identity/data range mismatch")
        if manifest.get("validation_ids") != list(range(1320, 1640)):
            raise ValueError(f"seed{seed}: manifest validation range mismatch")
        if (manifest.get("exposures"), manifest.get("optimizer_updates")) != (25000, 1570):
            raise ValueError(f"seed{seed}: matched-exposure recipe mismatch")
        for field, dest in (("code_sha256", code_hashes), ("gamma_sha256", gamma_hashes)):
            digest = manifest.get(field)
            if not valid_sha256(digest):
                raise ValueError(f"seed{seed}: invalid {field}")
            dest.add(digest)
        checkpoint = out / "core.pt"
        checkpoint_hashes[str(seed)] = file_sha256(checkpoint)
        for window, (steps, settle) in WINDOWS.items():
            path = out / f"validation_{window}_T{steps}_settle{settle}.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            validate_report(report, seed, window, checkpoint)
            validation_gamma_hashes.add(report["gamma_source"]["manifest"]["gamma_sha256"])
            observations[window][str(seed)] = {}
            for threshold in THRESHOLDS:
                row = next(x for x in report["sweep"] if math.isclose(float(x["synchrony_threshold"]), threshold, abs_tol=1e-9))
                metrics = row["scored_targets"]["our_hdf5"]["metrics"]
                observations[window][str(seed)][threshold] = {m: float(metrics[m]) for m in METRICS}
    if len(code_hashes) != 1 or len(gamma_hashes) != 1 or len(validation_gamma_hashes) != 1:
        raise ValueError("seed manifests/reports disagree on code or gamma SHA-256")

    aggregates: dict[str, Any] = {}
    for window in WINDOWS:
        aggregates[window] = {}
        for threshold in THRESHOLDS:
            values = {metric: [observations[window][str(seed)][threshold][metric] for seed in range(3)] for metric in METRICS}
            stats = {}
            for metric, rows in values.items():
                mean = statistics.fmean(rows)
                baseline = BASELINES[threshold][metric] if window == "long" else None
                stats[metric] = {
                    "seed_values": {str(seed): values[metric][seed] for seed in range(3)},
                    "mean": mean,
                    "std_population": statistics.pstdev(rows),
                    "baseline": baseline,
                    "delta_vs_SW0053_baseline": None if baseline is None else mean - baseline,
                }
            aggregates[window][str(threshold)] = stats
    return {
        "experiment": "SW0055 unique training-data scale",
        "split": {"ids": [1320, 1639], "images": 320},
        "manifest_code_sha256": next(iter(code_hashes)),
        "training_gamma_sha256": next(iter(gamma_hashes)),
        "validation_gamma_sha256": next(iter(validation_gamma_hashes)),
        "checkpoint_sha256_by_seed": checkpoint_hashes,
        "endpoint_policy": {"primary": {"window": "long", "steps": 1024, "settle": 512, "threshold": 0.50},
                            "secondary": {"window": "long", "steps": 1024, "settle": 512, "threshold": 0.35},
                            "no_posthoc_threshold_selection": True},
        "SW0053_baselines": {str(k): v for k, v in BASELINES.items()},
        "aggregates": aggregates,
        "source_values_by_window_seed_threshold": observations,
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = ["# SW0055 three-seed summary", "",
             "Validation IDs 1320-1639 (320 images). Primary endpoint: long T1024/settle512 at fixed threshold .50; .35 is secondary. No posthoc threshold selection.", "",
             "| Window | Threshold | Metric | Seed 0 | Seed 1 | Seed 2 | Mean | Population SD | SW0053 baseline | Delta vs baseline |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for window in ("long", "short"):
        for threshold in THRESHOLDS:
            for metric in METRICS:
                row = summary["aggregates"][window][str(threshold)][metric]
                seeds = row["seed_values"]
                baseline = "n/a" if row["baseline"] is None else f"{row['baseline']:.6f}"
                delta = "n/a" if row["delta_vs_SW0053_baseline"] is None else f"{row['delta_vs_SW0053_baseline']:+.6f}"
                lines.append(f"| {window} | {threshold:.2f} | {metric} | {seeds['0']:.6f} | {seeds['1']:.6f} | {seeds['2']:.6f} | {row['mean']:.6f} | {row['std_population']:.6f} | {baseline} | {delta} |")
    lines += ["", f"Code SHA-256: `{summary['manifest_code_sha256']}`  ",
              f"Training gamma SHA-256: `{summary['training_gamma_sha256']}`", ""]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root", type=Path, required=True, help="directory containing SW0055 seed output folders")
    p.add_argument("--output", type=Path, required=True, help="JSON output path; Markdown uses same stem")
    p.add_argument("--overwrite", action="store_true", help="allow replacing existing output pair")
    args = p.parse_args()
    result = summarize(args.model_root)
    md = args.output.with_suffix(".md")
    if not args.overwrite and (args.output.exists() or md.exists()):
        raise FileExistsError(f"refusing to overwrite {args.output} or {md}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    md.write_text(markdown(result), encoding="utf-8")


if __name__ == "__main__":
    main()
