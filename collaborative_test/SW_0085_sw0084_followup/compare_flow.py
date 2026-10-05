#!/usr/bin/env python3
"""Compare layer-resolved object margins for the selected SW0084 candidate."""
import argparse
import json
from pathlib import Path

ORDER = ("gamma", "graph", "kuramoto_plv_late", "gate_late",
         "dendritic_late", "membrane_late", "spike_late")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if baseline["ids"] != [1320, 1351] or candidate["ids"] != [1320, 1351]:
        raise ValueError("flow probes must use fixed IDs1320-1351")
    rows = []
    for stage in ORDER:
        before = float(baseline["pair_affinity"][stage]["object_margin"])
        after = float(candidate["pair_affinity"][stage]["object_margin"])
        rows.append({"stage": stage, "baseline_object_margin": before,
                     "candidate_object_margin": after, "delta": after - before})
    result = {"experiment": "SW0085 selected-candidate activation-flow diagnosis",
              "chosen": selection["chosen"], "stages": rows,
              "ground_truth_role": "diagnostic only after forward pass; never prediction input"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

