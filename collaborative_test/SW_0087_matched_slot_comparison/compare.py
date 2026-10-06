#!/usr/bin/env python3
"""Compare the fixed SW0072 mean with the matched-data Slot three-seed mean."""
import argparse
import json
import math
from pathlib import Path


METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")


def compare(model, slot):
    if model.get("experiment") != "SW0072 frozen trained graph full mean":
        raise ValueError("unexpected model summary")
    if slot.get("experiment") != "SW0056 matched-data Slot Attention three-seed summary":
        raise ValueError("unexpected Slot summary")
    if slot.get("image_ids_inclusive") != [1320, 1639] or slot.get("count") != 320:
        raise ValueError("Slot split mismatch")
    if slot.get("seeds") != [0, 1, 2] or slot.get("ground_truth_used_for_prediction") is not False:
        raise ValueError("Slot seed/provenance mismatch")

    rows = {}
    for metric in METRICS:
        model_value = float(model.get("means", {}).get(metric, float("nan")))
        slot_value = float(slot.get("metrics", {}).get(metric, {}).get("mean", float("nan")))
        if not math.isfinite(model_value) or not math.isfinite(slot_value):
            raise ValueError(f"missing/non-finite metric: {metric}")
        rows[metric] = {
            "model_mean": model_value,
            "slot_mean": slot_value,
            "delta_model_minus_slot": model_value - slot_value,
            "strictly_exceeds_slot": model_value > slot_value,
        }
    return {
        "schema_version": 1,
        "experiment": "SW0087 fixed-contract matched Slot goal comparison",
        "model": "SW0072 frozen trained graph",
        "slot": "SW0056 matched-data Slot Attention",
        "contract": "HDF5 validation IDs1320-1639; 16x16 patch masks; three-seed means",
        "metrics": rows,
        "goal_achieved": all(row["strictly_exceeds_slot"] for row in rows.values()),
    }


def markdown(result):
    lines = [
        "# SW0087 matched Slot goal comparison",
        "",
        result["contract"] + ".",
        "",
        "| Metric | SW0072 mean | Matched Slot mean | Delta | Strictly exceeds |",
        "|---|---:|---:|---:|:---:|",
    ]
    for metric, row in result["metrics"].items():
        lines.append(
            f"| {metric} | {row['model_mean']:.6f} | {row['slot_mean']:.6f} | "
            f"{row['delta_model_minus_slot']:+.6f} | {str(row['strictly_exceeds_slot']).lower()} |"
        )
    lines.extend(["", f"Goal achieved on all three metrics: **{str(result['goal_achieved']).lower()}**", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--slot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    md = args.output.with_suffix(".md")
    if args.output.exists() or md.exists():
        raise FileExistsError("refusing to overwrite comparison artifacts")
    result = compare(json.loads(args.model.read_text(encoding="utf-8")),
                     json.loads(args.slot.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    md.write_text(markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
