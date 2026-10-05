#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")


def row(path, target=None):
    report = json.loads(Path(path).read_text())
    assert report["ids"] == [1320, 1351]
    assert report["ground_truth_used_for_prediction"] is False
    matches = [x for x in report["sweep"]
               if x["affinity_mode"] == "spike"
               and x["synchrony_threshold"] == .35]
    if target is not None:
        matches = [x for x in matches if x.get("target_foreground") == target]
    assert len(matches) == 1
    scored = matches[0]["scored_targets"]["our_hdf5"]
    return {
        "metrics": {key: float(scored["metrics"][key]) for key in METRICS},
        "count_mae": float(scored["object_count"]["mae"]),
        "predicted_count_mean": float(scored["object_count"]["predicted_mean"]),
        "predicted_foreground_fraction": float(matches[0]["predicted_foreground_fraction"]),
    }


def mean(rows, key, nested=None):
    values = [item[key] if nested is None else item[key][nested] for item in rows]
    return sum(values) / len(values)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-dir", type=Path, required=True)
    p.add_argument("--fixed-seed0", type=Path, required=True)
    p.add_argument("--fixed-seed1", type=Path, required=True)
    p.add_argument("--fixed-seed2", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    fixed_paths = [args.fixed_seed0, args.fixed_seed1, args.fixed_seed2]
    fixed = [row(path) for path in fixed_paths]
    adaptive = [row(args.target_dir / f"target022_seed{s}_n32.json", .22)
                for s in range(3)]
    fixed_mean = {key: mean(fixed, "metrics", key) for key in METRICS}
    adaptive_mean = {key: mean(adaptive, "metrics", key) for key in METRICS}
    delta = {key: adaptive_mean[key] - fixed_mean[key] for key in METRICS}
    fixed_count_mae = mean(fixed, "count_mae")
    adaptive_count_mae = mean(adaptive, "count_mae")
    output = {
        "experiment": "SW0067 adaptive foreground classifier stage1",
        "ids": [1320, 1351],
        "target_foreground": .22,
        "fixed": {str(s): fixed[s] for s in range(3)},
        "adaptive": {str(s): adaptive[s] for s in range(3)},
        "three_seed_mean": {
            "fixed_metrics": fixed_mean,
            "adaptive_metrics": adaptive_mean,
            "delta": delta,
            "fixed_count_mae": fixed_count_mae,
            "adaptive_count_mae": adaptive_count_mae,
        },
        "advance": (all(delta[key] > 0 for key in METRICS)
                    and adaptive_count_mae <= fixed_count_mae),
        "gate": "three-seed mean improves all mask metrics and count MAE does not rise",
    }
    assert all(math.isfinite(v) for v in [*fixed_mean.values(),
                                           *adaptive_mean.values(),
                                           fixed_count_mae, adaptive_count_mae])
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
