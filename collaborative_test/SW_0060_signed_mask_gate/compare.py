#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--raw-summary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    raw = json.loads(args.raw_summary.read_text(encoding="utf-8"))
    if candidate.get("mode") != "signed_mask" or candidate.get("ids") != [1320, 1351]:
        raise ValueError("expected validated signed_mask report on IDs1320-1351")
    if raw.get("checkpoint_sha256") != candidate.get("checkpoint_sha256"):
        raise ValueError("checkpoint mismatch")
    raw_metrics = raw["normal_reference"]["fixed_readouts"]["spike_cc"]["metrics"]
    rows = {}
    for name in METRICS:
        before, after = float(raw_metrics[name]), float(candidate["metrics"][name])
        if not math.isfinite(before) or not math.isfinite(after):
            raise ValueError(f"non-finite {name}")
        rows[name] = {"raw": before, "signed_mask": after, "delta": after - before}
    result = {
        "experiment": "SW0060 signed delayed-mask carrier pilot",
        "comparison": "fixed seed0 checkpoint; IDs1320-1351; T256/settle64; spike-CC threshold .35",
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "prediction_ground_truth_independent": True,
        "metrics": rows,
        "all_three_improved": all(row["delta"] > 0 for row in rows.values()),
        "interpretation": "32-image validation pilot only; no retraining or general claim.",
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
