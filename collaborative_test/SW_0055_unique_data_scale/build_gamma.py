import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from snn_kuramoto_bidirectional.gamma_initializer import feature_maps_to_patch_gamma
from snn_kuramoto_bidirectional.training.train_gamma_initializer import load_input_encoder


HELD_OUT = range(1000, 1640)


def training_ids(count):
    if count < 1000:
        raise ValueError("count must retain all original 1000 training scenes")
    ids = list(range(1000)) + list(range(1640, 1640 + count - 1000))
    if set(ids).intersection(HELD_OUT):
        raise AssertionError("training IDs overlap held-out IDs 1000-1639")
    return ids


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5")
    parser.add_argument("--encoder", default="/Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002/input_encoder/input_layer_encoder.pt")
    parser.add_argument("--stats", default="/Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002/feature_preprocessing.pt")
    parser.add_argument("--reference-gamma", default="/Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002/gamma_train.pt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--count", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-reference-diff", type=float, default=1e-6)
    args = parser.parse_args()
    if args.batch_size < 1 or args.count < 1000:
        raise ValueError("positive batch size and count >= 1000 required")

    ids = training_ids(args.count)
    stats = torch.load(args.stats, map_location="cpu", weights_only=True)
    if stats.get("mode") != "standardize" or "mean" not in stats or "std" not in stats:
        raise ValueError("expected saved scalar standardization statistics")
    mean, std = stats["mean"].float(), stats["std"].float()
    clip = float(stats.get("clip", 3.0))
    if mean.numel() != 1 or std.numel() != 1 or not torch.isfinite(mean + std).all() or std <= 0:
        raise ValueError("invalid scalar preprocessing statistics")

    device = torch.device(args.device)
    encoder = load_input_encoder(args.encoder, num_kernels=8, kernel_size=3,
                                 channels=3, device=device).eval()
    rows = []
    with h5py.File(args.dataset, "r") as dataset, torch.no_grad():
        if max(ids) >= len(dataset["image"]):
            raise ValueError("requested training ID exceeds dataset")
        for start in range(0, len(ids), args.batch_size):
            batch_ids = ids[start:start + args.batch_size]
            raw = dataset["image"][batch_ids]
            images = torch.from_numpy(raw).permute(0, 3, 1, 2).float().div_(255.0).to(device)
            features = encoder(images).cpu()
            normalized = ((features - mean) / std).clamp(-clip, clip)
            gamma = feature_maps_to_patch_gamma(normalized, grid_size=16, device="cpu")
            rows.append(gamma.float())
    expanded = torch.cat(rows)
    if tuple(expanded.shape) != (args.count, 8, 256) or not torch.isfinite(expanded).all():
        raise AssertionError(f"invalid expanded gamma {tuple(expanded.shape)}")

    reference = torch.load(args.reference_gamma, map_location="cpu", weights_only=True).float()
    if tuple(reference.shape) != (1000, 8, 256):
        raise AssertionError("unexpected original gamma shape")
    max_diff = float((expanded[:1000] - reference).abs().max())
    if max_diff > args.max_reference_diff:
        raise AssertionError(f"first 1000 rows do not reproduce reference: max diff {max_diff}")

    output, manifest = Path(args.output), Path(args.manifest)
    if output.exists() or manifest.exists():
        raise FileExistsError("refusing to overwrite gamma or manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(expanded, output)
    record = {
        "contract_version": 1,
        "dataset": args.dataset,
        "training_ids": {"segments": [[0, 999], [1640, 1639 + args.count - 1000]],
                         "count": args.count},
        "held_out_ids_excluded": [1000, 1639],
        "shape": list(expanded.shape),
        "dtype": str(expanded.dtype),
        "encoder": args.encoder,
        "encoder_sha256": sha256(args.encoder),
        "preprocessing": args.stats,
        "preprocessing_sha256": sha256(args.stats),
        "reference_gamma": args.reference_gamma,
        "reference_gamma_sha256": sha256(args.reference_gamma),
        "first_1000_max_abs_diff": max_diff,
        "gamma_sha256": sha256(output),
    }
    manifest.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
