#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "snn_kuramoto_bidirectional")]
from snn_kuramoto_bidirectional.gamma_initializer import FeaturePatchGammaInitializer
from snn_kuramoto_bidirectional.input_layer_generator import CNNFeatureEncoder


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--encoder", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    output, manifest = Path(a.output), Path(a.manifest)
    if output.exists() or manifest.exists():
        raise FileExistsError("refusing to overwrite gamma export")
    device = torch.device(a.device)
    encoder = CNNFeatureEncoder(8, 3, in_channels=3, bias=True).to(device)
    encoder.load_state_dict(torch.load(a.encoder, map_location=device,
                                       weights_only=True), strict=True)
    encoder.eval()
    stats = torch.load(a.stats, map_location=device, weights_only=True)
    mean, std = stats["mean"].float(), stats["std"].float()
    clip = float(stats.get("clip", 3.0))
    patcher = FeaturePatchGammaInitializer(grid_size=16).to(device)
    rows = []
    with h5py.File(a.dataset, "r") as ds, torch.no_grad():
        if a.start < 0 or a.count < 1 or a.start + a.count > len(ds["image"]):
            raise ValueError("invalid image range")
        for start in range(a.start, a.start + a.count, a.batch_size):
            end = min(start + a.batch_size, a.start + a.count)
            images = torch.from_numpy(ds["image"][start:end]).permute(0, 3, 1, 2).to(device)
            features = encoder(images.float().div(255.0))
            rows.append(patcher(((features - mean) / std).clamp(-clip, clip)).cpu())
    gamma = torch.cat(rows).float()
    if tuple(gamma.shape) != (a.count, 8, 256) or not torch.isfinite(gamma).all():
        raise AssertionError(f"bad gamma export {tuple(gamma.shape)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(gamma, output)
    record = {"experiment": "SW0068 learned validation gamma", "ids": [a.start, a.start+a.count-1],
              "shape": list(gamma.shape), "encoder_sha256": digest(a.encoder),
              "stats_sha256": digest(a.stats), "gamma_sha256": digest(output),
              "ground_truth_used": False}
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
