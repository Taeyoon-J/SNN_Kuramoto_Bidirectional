"""Fixed-data RGB-bilateral graph override pilot; labels load only for final scoring."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import h5py
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "collaborative_test"), str(ROOT)]

from evaluate_fixed_split import _core
from snn_kuramoto_bidirectional.evaluation import evaluate_patch_masks, spatial_components_to_patch_labels, clevr_mask_patch
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_components
from rgb_graph import blend_adjacencies, rgb_bilateral_graph


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_aligned_gamma(gamma, manifest, gamma_path):
    if not isinstance(manifest, dict):
        raise ValueError("gamma manifest must be a JSON object")
    if manifest.get("image_ids") != [1320, 1639] or manifest.get("shape") != [320, 8, 256]:
        raise ValueError("gamma manifest is not the registered IDs1320-1639 aligned slice")
    if manifest.get("patch_grid_size") != 16 or manifest.get("dtype") != "torch.float32":
        raise ValueError("gamma manifest has unexpected dtype or patch grid")
    if gamma.ndim != 3 or list(gamma.shape) != [320, 8, 256] or not torch.isfinite(gamma).all():
        raise ValueError("gamma tensor must be finite float [320,8,256]")
    if manifest.get("gamma_sha256") != sha256(gamma_path):
        raise ValueError("gamma checksum differs from manifest")


def graph_stats(graph):
    return {
        "shape": list(graph.shape),
        "min": float(graph.min()),
        "max": float(graph.max()),
        "mean": float(graph.mean()),
        "mean_row_sum": float(graph.sum(dim=-1).mean()),
        "max_asymmetry": float((graph - graph.transpose(1, 2)).abs().max()),
    }


def dependency_sha256():
    files = [HERE / "evaluate.py", HERE / "rgb_graph.py",
             ROOT / "snn_kuramoto_bidirectional" / "s2net_cls.py",
             ROOT / "snn_kuramoto_bidirectional" / "graph_generator.py",
             ROOT / "snn_kuramoto_bidirectional" / "spike_classifier.py",
             ROOT / "collaborative_test" / "evaluate_fixed_split.py"]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gamma-path", required=True)
    parser.add_argument("--gamma-manifest", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--start", type=int, default=1320)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--settle", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.start, args.count, args.steps, args.settle) != (1320, 32, 256, 64):
        raise ValueError("SW0078 is fixed to IDs1320-1351, count32, T256/settle64")
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        raise ValueError("threshold must lie in [0,1]")
    for path in (args.checkpoint, args.gamma_path, args.gamma_manifest, args.dataset_path):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    output = Path(args.output_path)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")

    gamma_blob = torch.load(args.gamma_path, map_location="cpu", weights_only=True).float()
    manifest = json.loads(Path(args.gamma_manifest).read_text(encoding="utf-8"))
    validate_aligned_gamma(gamma_blob, manifest, args.gamma_path)
    gamma = gamma_blob[:args.count]
    ids = list(range(args.start, args.start + args.count))
    with h5py.File(args.dataset_path, "r") as dataset:
        images = torch.from_numpy(dataset["image"][ids]).permute(0, 3, 1, 2).contiguous()

    model = _core(args.device, args.checkpoint, args.steps, "shared", 3, 1.5, 2.0,
                  0.5, 16.0, 0.35, "factorized", "raw")
    model.membrane_layer.vth = 0.06
    if model.graph_generator is None:
        raise RuntimeError("SW0078 requires the learned image-conditioned graph")
    modes = [("learned_default", None), ("rgb_only", 1.0), ("blend_alpha_0p25", 0.25),
             ("blend_alpha_0p50", 0.50)]
    predictions, graph_diagnostics = {name: [] for name, _ in modes}, {name: [] for name, _ in modes}
    for name, alpha in modes:
        with torch.no_grad():
            for offset in range(0, args.count, args.batch_size):
                end = min(offset + args.batch_size, args.count)
                gamma_batch = gamma[offset:end].to(args.device)
                image_batch = images[offset:end].to(args.device)
                learned = model.graph_generator(gamma_batch)
                rgb = rgb_bilateral_graph(image_batch, model.graph_generator)
                override = None if alpha is None else blend_adjacencies(learned, rgb, alpha)
                active_graph = learned if override is None else override
                graph_diagnostics[name].append(graph_stats(active_graph))
                _, spikes, _, _ = model(
                    gamma_batch, return_core_out=True, return_theta=True,
                    graph_override=override,
                )
                components = model.last_component_spikes
                if components is None:
                    raise RuntimeError("SW0078 requires per-component spike outputs")
                groups = spike_synchrony_components(
                    spikes.cpu(), synchrony_threshold=args.threshold,
                    min_group_size=2, settle=64, components=components.cpu(),
                    background="largest_component", affinity_mode="spike",
                )
                predictions[name].append(spatial_components_to_patch_labels(groups, 16))

    # Targets are deliberately not loaded until all four graph/readout predictions exist.
    with h5py.File(args.dataset_path, "r") as dataset:
        target = clevr_mask_patch(torch.from_numpy(dataset["mask"][ids]), 8)["patch_labels"]
    rows = []
    for name, _ in modes:
        prediction = torch.cat(predictions[name], dim=0)
        scores = evaluate_patch_masks(prediction, target)
        rows.append({
            "mode": name,
            "graph_batches": graph_diagnostics[name],
            "predicted_foreground_fraction": float((prediction != 0).float().mean()),
            "predicted_group_count_mean": float(torch.tensor([
                int(torch.unique(image[image != 0]).numel()) for image in prediction
            ], dtype=torch.float32).mean()),
            "metrics": {key: float(value) for key, value in scores["mean"].items()},
            "valid_count": {key: int(value) for key, value in scores["valid_count"].items()},
        })
    report = {
        "experiment": "SW0078 label-free RGB-bilateral graph override",
        "ids": [args.start, args.start + args.count - 1],
        "images": args.count,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "evaluator_dependencies_sha256": dependency_sha256(),
        "gamma_path": str(Path(args.gamma_path).resolve()),
        "gamma_sha256": sha256(args.gamma_path),
        "gamma_manifest": manifest,
        "gamma_global_start": 1320,
        "image_source": {"path": args.dataset_path, "ids": [args.start, args.start + args.count - 1]},
        "inference": {"steps": args.steps, "settle": args.settle, "membrane_vth": 0.06,
                      "threshold": args.threshold, "background": "largest_component",
                      "min_group_size": 2, "dendritic_projection": "shared",
                      "graph_spatial_decay": 0.35, "geodesic_steps": 3,
                      "geodesic_radius": 1.5, "geodesic_contrast": 2.0,
                      "geodesic_temperature": 0.5, "geodesic_cap": 16,
                      "kuramoto_backend": "factorized", "gate_mode": "raw"},
        "rgb_graph": {"formula": "bilateral = exp(-color_d2/(2*2.0^2)) * exp(-grid_d2/(2*2.0^2))",
                      "color_features": "16x16 RGB patch means, per-image median and 1.4826*MAD normalization, scale floor 4 intensity levels, clamp [-8,8]",
                      "top_k": int(model.graph_generator.top_k),
                      "coupling_scale": float(model.graph_generator.log_coupling_gain.detach().exp()),
                      "normalization": "top-k bilateral weights sum to learned coupling scale per row before symmetrization",
                      "symmetry": "A = 0.5*(A + A.T)"},
        "blend_alphas": {"learned_default": None, "rgb_only": 1.0,
                         "blend_alpha_0p25": 0.25, "blend_alpha_0p50": 0.50},
        "ground_truth_used_for_graph_or_prediction": False,
        "readout": "same per-component spike synchrony connected components, threshold .35, largest component background",
        "sweep": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
