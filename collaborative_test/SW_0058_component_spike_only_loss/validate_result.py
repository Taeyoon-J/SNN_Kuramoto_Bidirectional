#!/usr/bin/env python3
"""Validate fixed SW0058 output and add asset/code hashes to spike reports."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fixed_spike_row(report, threshold=0.50):
    if report.get("ground_truth_used_for_prediction") is not False:
        raise ValueError("SW0058 predictions must be GT-free")
    count = int(report.get("images", 0))
    if count not in (4, 320) or report.get("ids") != [1320, 1320 + count - 1]:
        raise ValueError("expected aligned validation prefix count4 smoke or count320 full")
    if report.get("inference", {}).get("steps") not in (256, 1024):
        raise ValueError("unexpected inference steps")
    expected_settle = 64 if report["inference"]["steps"] == 256 else 512
    if report["inference"].get("settle") != expected_settle:
        raise ValueError("unexpected inference settle window")
    matches = [r for r in report.get("sweep", [])
               if r.get("affinity_mode") == "spike"
               and r.get("synchrony_threshold") == threshold]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one fixed spike row at threshold {threshold}")
    metrics = matches[0].get("scored_targets", {}).get("our_hdf5", {}).get("metrics", {})
    if not set(METRICS).issubset(metrics):
        raise ValueError("fixed spike row misses required metrics")
    for metric in METRICS:
        if not math.isfinite(float(metrics[metric])):
            raise ValueError(f"non-finite metric {metric}")
    return matches[0]


def evaluator_code_sha256():
    files = [HERE / "evaluate.sh", HERE / "validate_result.py",
             ROOT / "collaborative_test" / "SW_0040_peer_transfer" / "evaluate.py",
             ROOT / "collaborative_test" / "evaluate_fixed_split.py",
             ROOT / "snn_kuramoto_bidirectional" / "spike_classifier.py",
             ROOT / "snn_kuramoto_bidirectional" / "evaluation.py",
             ROOT / "snn_kuramoto_bidirectional" / "s2net_cls.py",
             ROOT / "snn_kuramoto_bidirectional" / "kuramoto_layer.py",
             ROOT / "snn_kuramoto_bidirectional" / "dendric_layer.py",
             ROOT / "snn_kuramoto_bidirectional" / "membrane_layer.py"]
    h = hashlib.sha256()
    for path in files:
        h.update(path.read_bytes())
    return h.hexdigest()


def validate_spike(path, checkpoint, seed, window, count):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    row = fixed_spike_row(report)
    checkpoint = Path(checkpoint)
    if Path(report.get("checkpoint", "")).resolve() != checkpoint.resolve():
        raise ValueError("report checkpoint path differs from requested checkpoint")
    gamma = Path(report["gamma_source"]["path"])
    manifest = Path(report["gamma_source"]["manifest_path"])
    if report["gamma_source"].get("global_start") != 1320:
        raise ValueError("unexpected gamma global start")
    inf = report["inference"]
    expected = (256, 64) if window == "short" else (1024, 512)
    if report.get("images") != int(count):
        raise ValueError("report count differs from requested smoke/full count")
    if (inf["steps"], inf["settle"]) != expected:
        raise ValueError("report window differs from requested window")
    if (inf.get("membrane_vth") != 0.06 or inf.get("dendritic_projection") != "shared"
            or inf.get("graph_spatial_decay") != 0.35 or inf.get("geodesic_steps") != 3
            or inf.get("kuramoto_backend") != "factorized"):
        raise ValueError("model configuration differs from SW0042 aligned recipe")
    provenance = {
        "schema_version": 1,
        "checkpoint_sha256": sha256(checkpoint),
        "gamma_sha256": sha256(gamma),
        "gamma_manifest_sha256": sha256(manifest),
        "evaluator_code_sha256": evaluator_code_sha256(),
        "fixed_threshold": 0.50,
        "window": {"name": window, "steps": expected[0], "settle": expected[1]},
        "count": int(count),
        "readout": "spike synchrony connected components",
        "metrics": {name: float(row["scored_targets"]["our_hdf5"]["metrics"][name])
                    for name in METRICS},
        "ground_truth_used_for_prediction": False,
    }
    # Persist the code/data/checkpoint binding beside the evaluator's report.
    provenance_path = Path(path).with_suffix(".provenance.json")
    with provenance_path.open("x", encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report, provenance


def main():
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=("spike",))
    p.add_argument("report")
    p.add_argument("checkpoint")
    p.add_argument("seed", type=int, choices=(0, 1, 2))
    p.add_argument("window", choices=("short", "long"))
    p.add_argument("count", type=int, choices=(4, 320))
    args = p.parse_args()
    report, provenance = validate_spike(args.report, args.checkpoint, args.seed, args.window, args.count)
    assert report.get("images") == args.count and provenance["ground_truth_used_for_prediction"] is False
    print(f"validated SW0058 seed{args.seed} {args.window}; provenance={Path(args.report).with_suffix('.provenance.json')}")


if __name__ == "__main__":
    main()
