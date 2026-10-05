#!/usr/bin/env python3
"""Select one SW0084 checkpoint for causal diagnosis and optional promotion."""
import argparse
import json
import math
from pathlib import Path

METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")


def select(summary):
    baseline = summary["baseline_sw0072_seed1"]
    rows = []
    for arm, epochs in summary["arms"].items():
        for epoch, row in epochs.items():
            metrics = row["metrics"]
            relative = {key: (metrics[key] - baseline[key]) / abs(baseline[key])
                        for key in METRICS}
            if not all(math.isfinite(value) for value in relative.values()):
                raise ValueError(f"nonfinite candidate {arm}/{epoch}")
            rows.append({
                "arm": arm,
                "epoch": epoch,
                "metrics": metrics,
                "delta": {key: metrics[key] - baseline[key] for key in METRICS},
                "relative_delta": relative,
                "all_three_strictly_improved": all(value > 0 for value in relative.values()),
                "minimum_relative_delta": min(relative.values()),
                "mean_relative_delta": sum(relative.values()) / len(relative),
            })
    if len(rows) != 8:
        raise ValueError(f"expected eight SW0084 candidates, got {len(rows)}")
    advancing = [row for row in rows if row["all_three_strictly_improved"]]
    pool = advancing or rows
    chosen = max(pool, key=lambda row: (row["minimum_relative_delta"],
                                        row["mean_relative_delta"],
                                        row["arm"], row["epoch"]))
    return {
        "experiment": "SW0085 automatic SW0084 follow-up",
        "selection_rule": (
            "among all-three pilot advances, maximize the worst relative metric delta; "
            "if none advances, apply the same maximin rule to all candidates for diagnosis"
        ),
        "baseline": baseline,
        "chosen": chosen,
        "full320_promotion": chosen["all_three_strictly_improved"],
        "candidate_count": len(rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = select(json.loads(args.summary.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

