"""Fixed, GT-free multi-readout evaluation for completed SW0055 checkpoints."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "collaborative_test"), str(ROOT / "collaborative_test" / "SW_0027_component_membrane_spectral"), str(ROOT / "collaborative_test" / "SW_0028_spatial_membrane_spectral"), str(ROOT)]
from evaluate_fixed_split import _core
from evaluate import correlation, spectral_labels
from spatial_evaluate import spatial_kernel
from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch, evaluate_patch_masks, spatial_components_to_patch_labels
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_components
from gamma_contract import aligned_gamma_prefix
from target_contract import require_complete_predictions


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dependency_code_sha256():
    files = [HERE / "evaluate.py", HERE / "validate_result.py", HERE / "preflight.sh", HERE / "gamma_contract.py", HERE / "target_contract.py",
             ROOT / "collaborative_test" / "evaluate_fixed_split.py",
             ROOT / "collaborative_test" / "SW_0027_component_membrane_spectral" / "evaluate.py",
             ROOT / "collaborative_test" / "SW_0028_spatial_membrane_spectral" / "spatial_evaluate.py",
             ROOT / "snn_kuramoto_bidirectional" / "spike_classifier.py",
             ROOT / "snn_kuramoto_bidirectional" / "evaluation.py",
             ROOT / "snn_kuramoto_bidirectional" / "training" / "evaluate_binding.py"]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser()
    for key in ("checkpoint", "gamma-path", "gamma-manifest", "dataset-path", "output-path"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    p.add_argument("--global-start", type=int, default=1320)
    p.add_argument("--count", type=int, default=320)
    p.add_argument("--steps", type=int, default=1024)
    p.add_argument("--settle", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.global_start != 1320 or args.count not in (4, 320) or (args.steps, args.settle) != (1024, 512):
        raise ValueError("SW0057 accepts full IDs1320-1639 or a four-row preflight, always at T1024/settle512")
    for raw in (args.checkpoint, args.gamma_path, args.gamma_manifest, args.dataset_path):
        if not Path(raw).is_file():
            raise FileNotFoundError(raw)
    out = Path(args.output_path)
    if out.exists():
        raise FileExistsError(f"refusing overwrite: {out}")
    ids = list(range(args.global_start, args.global_start + args.count))
    gamma = torch.load(args.gamma_path, map_location="cpu", weights_only=True)
    manifest = json.loads(Path(args.gamma_manifest).read_text(encoding="utf-8"))
    gamma = aligned_gamma_prefix(gamma, manifest, sha256(args.gamma_path), args.global_start,
                                 args.count, args.steps, args.settle).float()

    model = _core(args.device, args.checkpoint, args.steps, "shared", 3, 1.5, 2.0, 0.5, 16.0, 0.35, "factorized")
    model.membrane_layer.vth = 0.06
    kernel = spatial_kernel(1.5).float()
    predictions = {"spike_cc_threshold_0p50": [], "membrane_spatial_sigma1p5_k10": []}
    with torch.no_grad():
        for offset in range(0, args.count, args.batch_size):
            batch = gamma[offset:offset + args.batch_size].to(args.device)
            _, spikes, membrane, _ = model(batch, return_core_out=True, return_theta=True)
            groups = spike_synchrony_components(
                spikes.cpu(), synchrony_threshold=0.50, min_group_size=2,
                settle=args.settle, components=model.last_component_spikes.cpu(),
                background="largest_component")
            predictions["spike_cc_threshold_0p50"].append(spatial_components_to_patch_labels(groups, 16))
            for bi in range(membrane.shape[0]):
                affinity = correlation(membrane[bi, :, args.settle:].cpu()).abs() * kernel
                predictions["membrane_spatial_sigma1p5_k10"].append(spectral_labels(affinity, 10).unsqueeze(0))

    # Targets are deliberately loaded only after every GT-free prediction exists.
    require_complete_predictions(predictions, args.count)
    with h5py.File(args.dataset_path, "r") as h5:
        truth = clevr_mask_patch(torch.from_numpy(h5["mask"][ids]), 8)["patch_labels"]

    rows = []
    for name, chunks in predictions.items():
        pred = torch.cat(chunks, dim=0).long()
        scores = evaluate_patch_masks(pred, truth)["mean"]
        per_image_counts = [int(torch.unique(image[image > 0]).numel()) for image in pred]
        row = {
            "readout": name,
            "metrics": {k: float(v) for k, v in scores.items()},
            "predicted_object_count_mean": float(np.mean(per_image_counts)),
            "predicted_foreground_fraction": float((pred != 0).float().mean()),
            "predicted_empty_image_count": int(sum(n == 0 for n in per_image_counts)),
            "per_image_object_counts": per_image_counts,
        }
        if not all(math.isfinite(v) for v in row["metrics"].values()):
            raise ValueError(f"Non-finite metric in {name}")
        rows.append(row)
    report = {
        "schema_version": 1,
        "experiment": "SW0057 fixed multi-readout",
        "seed": args.seed,
        "split": "fixed_hdf5_aligned_validation",
        "ids": [ids[0], ids[-1]], "images": args.count,
        "checkpoint": {"path": str(Path(args.checkpoint).resolve()), "sha256": sha256(args.checkpoint)},
        "evaluator": {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__),
                      "dependency_code_sha256": dependency_code_sha256()},
        "gamma": {"path": str(Path(args.gamma_path).resolve()), "sha256": sha256(args.gamma_path),
                  "manifest_path": str(Path(args.gamma_manifest).resolve()), "manifest_sha256": sha256(args.gamma_manifest),
                  "manifest": manifest, "global_start": args.global_start},
        "target": {"path": str(Path(args.dataset_path).resolve()), "ids": [ids[0], ids[-1]]},
        "inference": {"steps": args.steps, "settle": args.settle, "membrane_vth": 0.06,
                      "dendritic_projection": "shared", "graph_spatial_decay": 0.35,
                      "geodesic_steps": 3, "geodesic_radius": 1.5, "geodesic_contrast": 2.0,
                      "geodesic_temperature": 0.5, "geodesic_cap": 16.0,
                      "kuramoto_backend": "factorized", "batch_size": args.batch_size},
        "readout_contract": {"spike_cc_threshold": 0.50, "min_group_size": 2,
                              "background": "largest_component", "membrane_spectral_sigma": 1.5,
                              "membrane_spectral_k": 10, "ground_truth_used_for_prediction": False},
        "rows": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(out), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
