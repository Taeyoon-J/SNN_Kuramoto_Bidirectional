#!/usr/bin/env python3
"""Aggregate the fixed-contract SW0071 readouts across seeds."""

import argparse
import json
import math
import statistics
from pathlib import Path


METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_report = json.loads(args.baseline.read_text(encoding="utf-8"))
    baseline = {key: float(baseline_report["metrics"][key]["mean"]) for key in METRICS}
    by_mode: dict[str, dict[str, list[float]]] = {}
    per_seed: dict[str, dict[str, dict[str, float]]] = {}

    for seed in range(3):
        report = json.loads((args.input_dir / f"seed{seed}_long.json").read_text(encoding="utf-8"))
        assert report["ids"] == [1320, 1639]
        assert report["core"]["steps"] == 1024 and report["core"]["settle"] == 512
        assert report["readout"]["frozen_spike_cc_threshold"] == 0.5
        rows = {row["mode"]: row for row in report["rows"]}
        cc = rows["spike_cc_baseline"]
        per_seed[str(seed)] = {}
        for mode, row in rows.items():
            assert row["foreground_mask_exactly_matches_spike_cc"] is True
            assert math.isclose(row["foreground_iou"], cc["foreground_iou"], abs_tol=1e-12)
            values = {key: float(row[key]) for key in METRICS}
            assert all(math.isfinite(value) for value in values.values())
            per_seed[str(seed)][mode] = values
            bucket = by_mode.setdefault(mode, {key: [] for key in METRICS})
            for key, value in values.items():
                bucket[key].append(value)

    internal_baseline = {
        key: statistics.mean(by_mode["spike_cc_baseline"][key]) for key in METRICS
    }
    rows = []
    for mode, values in by_mode.items():
        means = {key: statistics.mean(values[key]) for key in METRICS}
        deltas = {key: means[key] - internal_baseline[key] for key in METRICS}
        rows.append(
            {
                "mode": mode,
                **means,
                "delta_vs_internal_spike_cc": deltas,
                "delta_vs_sw0070_report": {
                    key: means[key] - baseline[key] for key in METRICS
                },
                "foreground_preserved": math.isclose(
                    means["foreground_iou"], internal_baseline["foreground_iou"], abs_tol=1e-12
                ),
                "accepted": (
                    deltas["fg_ari"] > 0
                    and deltas["matched_object_iou"] >= 0
                    and math.isclose(deltas["foreground_iou"], 0.0, abs_tol=1e-12)
                ),
            }
        )
    rows.sort(key=lambda row: (row["accepted"], row["fg_ari"], row["matched_object_iou"]), reverse=True)
    output = {
        "experiment": "SW0071 stabilized hybrid readout",
        "contract": {"ids": [1320, 1639], "steps": 1024, "settle": 512, "spike_threshold": 0.5},
        "baseline_sw0070_report": baseline,
        "baseline_internal_spike_cc": internal_baseline,
        "internal_vs_sw0070_evaluation_drift": {
            key: internal_baseline[key] - baseline[key] for key in METRICS
        },
        "per_seed": per_seed,
        "rows": rows,
        "accepted_modes": [row["mode"] for row in rows if row["accepted"]],
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
