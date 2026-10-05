#!/usr/bin/env python3
"""Validate a SW0059 fixed split/mode report and checkpoint identity."""
import argparse
import hashlib
import json
import math
from pathlib import Path

METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")
ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate(path, checkpoint, mode):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if Path(report.get("checkpoint", "")).resolve() != Path(checkpoint).resolve():
        raise ValueError("checkpoint path mismatch")
    if report.get("ids") != [1320, 1351] or report.get("images") != 32:
        raise ValueError("expected exact fixed short IDs1320-1351/count32")
    if report.get("ground_truth_used_for_prediction") is not False:
        raise ValueError("GT must be used only after prediction")
    inf = report.get("inference", {})
    if (inf.get("steps"), inf.get("settle"), inf.get("gate_mode")) != (256, 64, mode):
        raise ValueError("fixed short window or gate mode mismatch")
    if inf.get("membrane_vth") != 0.06 or inf.get("kuramoto_backend") != "factorized":
        raise ValueError("SW0053 seed0 dynamics config mismatch")
    rows = [row for row in report.get("sweep", [])
            if row.get("affinity_mode") == "spike" and row.get("synchrony_threshold") == 0.35]
    if len(rows) != 1:
        raise ValueError("expected exactly one spike readout row at fixed threshold .35")
    metrics = rows[0].get("scored_targets", {}).get("our_hdf5", {}).get("metrics", {})
    values = {k: float(metrics[k]) for k in METRICS}
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError("non-finite metrics")
    gamma = Path(report["gamma_source"]["path"])
    manifest = Path(report["gamma_source"]["manifest_path"])
    code_files = [Path(__file__), ROOT / "collaborative_test" / "SW_0040_peer_transfer" / "evaluate.py",
                  ROOT / "collaborative_test" / "evaluate_fixed_split.py",
                  ROOT / "snn_kuramoto_bidirectional" / "sinusoidal_gating.py",
                  ROOT / "snn_kuramoto_bidirectional" / "s2net_cls.py"]
    code_hash = hashlib.sha256(b"".join(path.read_bytes() for path in code_files)).hexdigest()
    return {"mode": mode, "checkpoint_sha256": sha(checkpoint),
            "gamma_sha256": sha(gamma), "manifest_sha256": sha(manifest),
            "evaluator_code_sha256": code_hash, "ids": report["ids"],
            "count": 32, "steps": 256, "settle": 64, "threshold": 0.35,
            "metrics": values, "ground_truth_used_for_prediction": False}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("report")
    p.add_argument("checkpoint")
    p.add_argument("mode", choices=("raw", "centered_raw", "signed_mask"))
    args = p.parse_args()
    result = validate(args.report, args.checkpoint, args.mode)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
