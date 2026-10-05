#!/usr/bin/env python3
"""Run one canonical trainer batch on real SW0053 gamma and validate output."""
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", required=True)
    parser.add_argument("--sc", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"preflight output exists: {output}")
    output.mkdir(parents=True)

    gamma_path, sc_path = Path(args.gamma), Path(args.sc)
    gamma = torch.load(gamma_path, map_location="cpu", weights_only=True).float()
    sc = torch.load(sc_path, map_location="cpu", weights_only=True).float()
    if tuple(gamma.shape) != (1000, 8, 256) or tuple(sc.shape) != (256, 256):
        raise ValueError(f"unexpected gamma/sc shapes: {tuple(gamma.shape)}, {tuple(sc.shape)}")
    samples = gamma.reshape(-1, 256)
    centered = samples - samples.mean(0, keepdim=True)
    normalized = centered / torch.linalg.vector_norm(centered, dim=0, keepdim=True).clamp_min(1e-8)
    regenerated_sc = (normalized.T @ normalized).abs().clamp(0, 1)
    if not torch.allclose(regenerated_sc, sc, atol=1e-6, rtol=1e-6):
        raise ValueError("existing SC does not match the full SW0053 gamma tensor")
    subset = output / "gamma_first16.pt"
    torch.save(gamma[:16], subset)
    checkpoint = output / "core.pt"
    command = [
        sys.executable, "-u", "-m", "snn_kuramoto_bidirectional.training.train_s2net_core",
        "--gamma-seq-path", str(subset), "--sc-path", str(sc_path), "--save-path", str(checkpoint),
        "--num-regions", "256", "--num-feature-maps", "8", "--device", args.device,
        "--epochs", "1", "--batch-size", "16", "--lr", "0.0003", "--seed", "0", "--osc-dim", "4",
        "--gamma-drive-mode", "static", "--num-time-steps", "64", "--plv-settle", "32",
        "--theta-init", "gamma", "--gamma-phase-mode", "standardize_tanh", "--freq-gain", "2.0",
        "--graph-mode", "learned", "--graph-top-k", "32", "--graph-spatial-decay", "0.35",
        "--geodesic-steps", "3", "--geodesic-radius", "1.5", "--geodesic-contrast", "2.0",
        "--geodesic-temperature", "0.5", "--geodesic-cap", "16", "--kuramoto-backend", "factorized",
        "--spike-spatial-grid-size", "16", "--k", "256", "--membrane-vth", "0.06",
        "--membrane-low-m", "-4", "--membrane-high-m", "0", "--low-n", "-4", "--high-n", "0",
        "--branch", "4", "--gate-mode", "raw", "--plv-source", "phase", "--plv-combine", "mean",
        "--primary-loss-weight", "0", "--spike-plv-weight", "5.0", "--loss-signal", "sigmoid_membrane",
        "--sample-activity-diversity-weight", "0", "--spike-rate-weight", "0",
        "--spike-smooth-weight", "0", "--spike-diversity-weight", "0", "--structural-weight", "0",
        "--plv-collapse-weight", "1.0", "--plv-bimodality-weight", "6.0",
        "--plv-balance-weight", "10.0", "--plv-target-density", "0.867",
        "--plv-coherence-weight", "0.5", "--spike-per-component",
        "--dendritic-projection", "shared", "--verbose",
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run(command, cwd=ROOT, env=environment, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (output / "training.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"canonical trainer failed with rc={completed.returncode}")
    if not checkpoint.is_file() or "Epoch 0001/0001" not in completed.stdout or "trained S2NetCore:" not in completed.stdout:
        raise RuntimeError("canonical trainer did not complete its one batch")
    match = re.search(
        r"Epoch 0001/0001 \| loss=([0-9.eE+-]+).*?"
        r"primary_unscaled_total=([0-9.eE+-]+).*?"
        r"primary_weighted_total=([0-9.eE+-]+).*?"
        r"spike_weighted_total=([0-9.eE+-]+)",
        completed.stdout,
    )
    if not match or not all(math.isfinite(float(value)) for value in match.groups()):
        raise RuntimeError("finite canonical loss fields were not found")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    tensors = [value for value in state.values() if torch.is_tensor(value)]
    if not tensors or not all(torch.isfinite(tensor).all().item() for tensor in tensors):
        raise RuntimeError("checkpoint contains missing or non-finite tensors")
    report = {
        "schema_version": 2, "experiment": "SW0058 canonical one-batch preflight",
        "gamma_path": str(gamma_path.resolve()), "gamma_sha256": sha256(gamma_path),
        "sc_path": str(sc_path.resolve()), "sc_sha256": sha256(sc_path), "sc_matches_full_gamma": True,
        "ids": [0, 15], "device": args.device, "seed": 0,
        "recipe": {"primary_loss_weight": 0.0, "spike_plv_weight": 5.0,
                   "steps_per_batch": 64, "plv_settle": 32, "epochs": 1, "batches": 1},
        "loss": float(match.group(1)), "primary_total_unweighted_logged": float(match.group(2)),
        "primary_total_weighted_logged": float(match.group(3)),
        "spike_aux_total_weighted_logged": float(match.group(4)),
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint),
        "finite_checkpoint": True, "canonical_training_module": True,
    }
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
