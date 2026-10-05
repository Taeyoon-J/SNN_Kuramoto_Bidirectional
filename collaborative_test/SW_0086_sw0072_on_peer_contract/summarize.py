#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

METRICS = ("patch_fg_ari", "patch_foreground_iou", "patch_matched_object_iou")
SLOT = {"patch_fg_ari": .6195, "patch_foreground_iou": .1241,
        "patch_matched_object_iou": .0920}


def summarize_reports(reports):
    if [row["seed"] for row in reports] != [0, 1, 2]:
        raise ValueError("need seeds 0,1,2")
    for row in reports:
        if row["peer_rows"] != [6000, 6299] or row["images"] != 300:
            raise ValueError("cross-contract scope mismatch")
        if row["ground_truth_used_for_prediction"] is not False:
            raise ValueError("invalid prediction provenance")
    means = {key: sum(row["metrics"][key] for row in reports) / 3 for key in METRICS}
    if not all(math.isfinite(v) for v in means.values()):
        raise ValueError("nonfinite mean")
    return {
        "experiment": "SW0086 SW0072 three-seed transfer to peer contract",
        "seeds": {str(row["seed"]): row["metrics"] for row in reports},
        "means": means,
        "peer_single_checkpoint_slot_reference": SLOT,
        "delta_vs_peer_slot": {key: means[key] - SLOT[key] for key in METRICS},
        "exceeds_peer_slot_all_three": all(means[key] > SLOT[key] for key in METRICS),
        "scope": "peer validation rows6000-6299, original SW0072 settings, no retraining/tuning",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reports = [json.loads((args.results / f"seed{seed}.json").read_text())
               for seed in range(3)]
    result = summarize_reports(reports)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

