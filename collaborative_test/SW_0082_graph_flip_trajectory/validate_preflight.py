#!/usr/bin/env python3
"""Reject stale or malformed SW0081 calibration before SW0082 training."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(marker_path, repo_root, core, gamma, encoder, stats, sw0081_dir):
    marker = json.loads(Path(marker_path).read_text(encoding="utf-8"))
    required = {
        "source_core_sha256": core,
        "gamma_sha256": gamma,
        "encoder_sha256": encoder,
        "stats_sha256": stats,
        "trainer_sha256": Path(sw0081_dir) / "train_graph.py",
        "equivariance_impl_sha256": Path(sw0081_dir) / "equivariance.py",
    }
    for key, path in required.items():
        if marker.get(key) != sha(path):
            raise ValueError(f"stale SW0081 preflight: {key} mismatch")
    expected_dependencies = {
        "snn_kuramoto_bidirectional/s2net_cls.py",
        "snn_kuramoto_bidirectional/graph_generator.py",
        "snn_kuramoto_bidirectional/input_layer_generator.py",
        "snn_kuramoto_bidirectional/gamma_initializer.py",
        "snn_kuramoto_bidirectional/loss_function.py",
        "snn_kuramoto_bidirectional/training/train_s2net_core.py",
    }
    dependencies = marker.get("code_dependencies_sha256", {})
    if set(dependencies) != expected_dependencies:
        raise ValueError("stale or incomplete SW0081 dependency provenance")
    for relative, digest in dependencies.items():
        if sha(Path(repo_root) / relative) != digest:
            raise ValueError(f"stale SW0081 code dependency: {relative}")
    measured = marker["measured"]
    norms = (float(measured["binding_gradient_norm"]),
             float(measured["raw_equivariance_gradient_norm"]))
    mse = float(measured["initial_raw_equivariance_mse"])
    weights = marker["weights"]
    if not all(math.isfinite(x) for x in (*norms, mse)) or norms[0] < 0 or norms[1] <= 0 or mse < 0:
        raise ValueError("preflight contains invalid loss/gradient measurements")
    expected = {"weight_0p1x": 0.1 * norms[0] / norms[1],
                "weight_1x": norms[0] / norms[1]}
    for key, value in expected.items():
        if not math.isclose(float(weights[key]), value, rel_tol=1e-6, abs_tol=1e-12):
            raise ValueError(f"preflight weight formula mismatch: {key}")
    return marker


def main():
    p = argparse.ArgumentParser()
    for arg in ("marker", "repo-root", "core", "gamma", "encoder", "stats", "sw0081-dir"):
        p.add_argument("--" + arg, required=True)
    a = p.parse_args()
    result = validate(a.marker, a.repo_root, a.core, a.gamma, a.encoder, a.stats, a.sw0081_dir)
    print(json.dumps({"valid": True, "weight_0p1x": result["weights"]["weight_0p1x"],
                      "weight_1x": result["weights"]["weight_1x"]}, indent=2))


if __name__ == "__main__":
    main()
