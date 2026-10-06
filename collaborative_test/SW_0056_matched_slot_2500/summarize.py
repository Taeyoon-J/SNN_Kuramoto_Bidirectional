"""Summarize fixed SW0056 aligned Slot Attention scores across three seeds."""
import argparse
import json
import math
import statistics
from pathlib import Path

SEEDS = (0, 1, 2)
IDS = [1320, 1639]
METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")


def _family_value(value):
    if isinstance(value, dict):
        return {k: _family_value(v) for k, v in sorted(value.items())
                if k not in {"seed", "checkpoint_prefix", "checkpoint_sha256", "checkpoint_files_sha256",
                             "checkpoint_dir", "training_protocol_path", "trainer_sha256",
                             "run_script_sha256", "execution_backend"}}
    if isinstance(value, list):
        return [_family_value(v) for v in value]
    return value


def summarize_reports(reports):
    if set(reports) != set(SEEDS):
        raise ValueError("Exactly seed reports 0, 1, and 2 are required")
    common_family = None
    values = {metric: [] for metric in METRICS}
    provenance = []
    for seed in SEEDS:
        report = reports[seed]
        protocol = report.get("protocol", {})
        if report.get("ground_truth_used_for_prediction") is not False:
            raise ValueError(f"seed{seed} protocol does not certify GT-free prediction")
        if protocol.get("seed") != seed or protocol.get("image_ids") != IDS or protocol.get("count") != 320:
            raise ValueError(f"seed{seed} protocol seed/split/count mismatch")
        if protocol.get("inference_seed") != 0 or protocol.get("resolution") != [128, 128] or protocol.get("num_slots") != 11 or protocol.get("iterations") != 3:
            raise ValueError(f"seed{seed} inference recipe mismatch")
        if protocol.get("training_protocol", {}).get("validation_ids_inclusive") != IDS:
            raise ValueError(f"seed{seed} training protocol validation range mismatch")
        for metric in METRICS:
            raw = report.get("scores", {}).get("mean", {}).get(metric)
            if raw is None or not math.isfinite(float(raw)):
                raise ValueError(f"seed{seed} metric is missing/non-finite: {metric}")
            values[metric].append(float(raw))
        family = _family_value(protocol)
        if common_family is None:
            common_family = family
        elif family != common_family:
            raise ValueError("seed evaluation protocol families do not match")
        provenance.append({"seed": seed, "checkpoint_prefix": protocol.get("checkpoint_prefix"),
                           "checkpoint_sha256": protocol.get("checkpoint_sha256"),
                           "model_sha256": protocol.get("model_sha256"),
                           "training_protocol": protocol.get("training_protocol")})
    result = {"schema_version": 1, "experiment": "SW0056 matched-data Slot Attention three-seed summary",
              "split": "HDF5 aligned validation", "image_ids_inclusive": IDS, "count": 320,
              "seeds": list(SEEDS), "patch_size": 8, "patch_grid": [16, 16],
              "ground_truth_used_for_prediction": False, "protocol_family": common_family,
              "provenance": provenance, "metrics": {}}
    for metric, per_seed in values.items():
        result["metrics"][metric] = {"per_seed": per_seed, "mean": statistics.mean(per_seed),
                                     "std_sample": statistics.stdev(per_seed)}
    return result


def markdown(result):
    lines = ["# SW0056 matched-data Slot Attention three-seed summary", "",
             "Validation IDs 1320-1639, patch size 8, seeds 0/1/2. Mean and sample standard deviation are across seeds.",
             "", "| Metric | Mean | Sample SD | Seed 0 | Seed 1 | Seed 2 |", "|---|---:|---:|---:|---:|---:|"]
    for metric, row in result["metrics"].items():
        lines.append(f"| {metric} | {row['mean']:.6f} | {row['std_sample']:.6f} | " +
                     " | ".join(f"{v:.6f}" for v in row["per_seed"]) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    md_path = args.output.with_suffix(".md")
    if args.output.exists() or md_path.exists():
        raise FileExistsError("Refusing to overwrite SW0056 summary artifacts")
    reports = {}
    for seed in SEEDS:
        path = args.model_root / f"SW0056_matched_slot_2500_seed{seed}" / "validation1320_1639" / "evaluation_summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing seed{seed} summary: {path}")
        reports[seed] = json.loads(path.read_text(encoding="utf-8"))
    result = summarize_reports(reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with md_path.open("x", encoding="utf-8") as stream:
        stream.write(markdown(result))
    print(f"Wrote {args.output} and {md_path}")


if __name__ == "__main__":
    main()
