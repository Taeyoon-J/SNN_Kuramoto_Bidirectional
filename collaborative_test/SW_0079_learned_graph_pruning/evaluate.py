"""Fixed seed1 pilot of top-k pruning on the frozen learned graph."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "collaborative_test"), str(ROOT)]

from evaluate_fixed_split import _core
from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch, evaluate_patch_masks, spatial_components_to_patch_labels
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_components
from pruning import adjacency_at_top_k


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def graph_stats(graph):
    return {
        "shape": list(graph.shape),
        "min": float(graph.min()),
        "max": float(graph.max()),
        "mean": float(graph.mean()),
        "mean_row_sum": float(graph.sum(dim=-1).mean()),
        "max_asymmetry": float((graph - graph.transpose(1, 2)).abs().max()),
        "nonzero_per_row_mean": float((graph > 0).sum(dim=-1).float().mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--gamma-path", required=True)
    p.add_argument("--gamma-manifest", required=True)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--output-path", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=4)
    a = p.parse_args()
    for path in (a.checkpoint, a.gamma_path, a.gamma_manifest, a.dataset_path):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    output = Path(a.output_path)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    gamma_manifest = json.loads(Path(a.gamma_manifest).read_text(encoding="utf-8"))
    if (gamma_manifest.get("image_ids") != [1320, 1639]
            or gamma_manifest.get("shape") != [320, 8, 256]
            or gamma_manifest.get("patch_grid_size") != 16
            or gamma_manifest.get("dtype") != "torch.float32"):
        raise ValueError("gamma manifest does not match the fixed aligned slice")
    if gamma_manifest.get("gamma_sha256") != sha256(a.gamma_path):
        raise ValueError("gamma checksum does not match manifest")
    gamma_full = torch.load(a.gamma_path, map_location="cpu", weights_only=True).float()
    if tuple(gamma_full.shape) != (320, 8, 256) or not torch.isfinite(gamma_full).all():
        raise ValueError("gamma tensor must be finite [320,8,256]")
    gamma = gamma_full[:32]
    ids = list(range(1320, 1352))
    with h5py.File(a.dataset_path, "r") as dataset:
        images = torch.from_numpy(dataset["image"][ids]).permute(0, 3, 1, 2).contiguous()

    model = _core(a.device, a.checkpoint, 256, "shared", 3, 1.5, 2.0, 0.5, 16.0, 0.35, "factorized", "raw")
    model.membrane_layer.vth = 0.06
    graph_generator = model.graph_generator
    if graph_generator is None or int(graph_generator.top_k) != 32:
        raise ValueError("SW0079 requires the frozen seed1 learned graph with top_k=32")
    modes = [("learned_topk32_default", None), ("learned_topk16", 16), ("learned_topk8", 8)]
    predictions = {name: [] for name, _ in modes}
    graph_diagnostics = {name: [] for name, _ in modes}
    for name, k in modes:
        with torch.no_grad():
            for offset in range(0, 32, a.batch_size):
                end = min(offset + a.batch_size, 32)
                gamma_batch = gamma[offset:end].to(a.device)
                if k is None:
                    graph = graph_generator(gamma_batch)
                    override = None
                else:
                    graph = adjacency_at_top_k(graph_generator, gamma_batch, k)
                    override = graph
                if int(graph_generator.top_k) != 32:
                    raise AssertionError("graph top_k was not restored to checkpoint value")
                graph_diagnostics[name].append(graph_stats(graph))
                _, spikes, _, _ = model(
                    gamma_batch, return_core_out=True, return_theta=True,
                    graph_override=override,
                )
                components = model.last_component_spikes
                if components is None:
                    raise RuntimeError("per-component spikes are required for the registered readout")
                groups = spike_synchrony_components(
                    spikes.cpu(), synchrony_threshold=0.35, min_group_size=2,
                    settle=64, components=components.cpu(), background="largest_component",
                    affinity_mode="spike",
                )
                predictions[name].append(spatial_components_to_patch_labels(groups, 16))

    # Targets are loaded only after every GT-free prediction is complete.
    with h5py.File(a.dataset_path, "r") as dataset:
        targets = clevr_mask_patch(torch.from_numpy(dataset["mask"][ids]), 8)["patch_labels"]
    rows = []
    for name, _ in modes:
        pred = torch.cat(predictions[name], dim=0)
        scores = evaluate_patch_masks(pred, targets)
        rows.append({
            "mode": name,
            "graph_batches": graph_diagnostics[name],
            "predicted_foreground_fraction": float((pred != 0).float().mean()),
            "predicted_group_count_mean": float(torch.tensor([
                int(torch.unique(image[image != 0]).numel()) for image in pred
            ], dtype=torch.float32).mean()),
            "metrics": {key: float(value) for key, value in scores["mean"].items()},
            "valid_count": {key: int(value) for key, value in scores["valid_count"].items()},
        })
    report = {
        "experiment": "SW0079 frozen learned graph top-k pruning",
        "ids": [1320, 1351], "images": 32,
        "checkpoint": str(Path(a.checkpoint).resolve()),
        "checkpoint_sha256": sha256(a.checkpoint),
        "gamma_path": str(Path(a.gamma_path).resolve()),
        "gamma_sha256": sha256(a.gamma_path),
        "gamma_manifest": gamma_manifest,
        "gamma_global_start": 1320,
        "inference": {"steps": 256, "settle": 64, "membrane_vth": 0.06,
                      "graph_spatial_decay": 0.35, "geodesic_steps": 3,
                      "geodesic_radius": 1.5, "geodesic_contrast": 2.0,
                      "geodesic_temperature": 0.5, "geodesic_cap": 16.0,
                      "kuramoto_backend": "factorized", "gate_mode": "raw",
                      "readout": "per-component spike CC, threshold .35, min size2, largest component background"},
        "intervention": "temporarily change only graph_generator.top_k during graph recomputation; restore to 32 before rollout",
        "preserved": ["checkpoint tensors", "graph projection/logits", "coupling gain", "symmetrization", "gamma", "Kuramoto/dendritic/membrane dynamics", "classifier"],
        "ground_truth_used_for_graph_or_prediction": False,
        "sweep": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sweep": rows}, indent=2))


if __name__ == "__main__":
    main()
