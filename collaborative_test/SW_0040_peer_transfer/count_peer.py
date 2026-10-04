"""Peer validation object-count diagnostics; never used to form predictions."""
import argparse
import importlib.util
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "collaborative_test"))
sys.path.insert(0, str(REPO_ROOT))

def count_scores(predicted_counts, target_counts):
    predicted = torch.as_tensor(predicted_counts, dtype=torch.float32)
    target = torch.as_tensor(target_counts, dtype=torch.float32)
    error = predicted - target
    return {
        "exact_accuracy": float((error == 0).float().mean()),
        "mae": float(error.abs().mean()),
        "bias": float(error.mean()),
        "within_one_accuracy": float((error.abs() <= 1).float().mean()),
        "predicted_mean": float(predicted.mean()),
        "target_mean": float(target.mean()),
    }


def build_peer_evaluator(evaluator_path, checkpoint, device, steps=256):
    """Load the peer evaluator and model classes so checkpoint keys match exactly."""
    path = Path(evaluator_path)
    spec = importlib.util.spec_from_file_location("peer_evaluate_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import peer evaluator at {path}.")
    peer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(peer)
    config = SimpleNamespace(
        checkpoint=str(checkpoint),
        dendrite_per_region=False,
        membrane_threshold_mode="absolute",
        membrane_threshold_k=0.5,
        num_regions=256,
        num_time_steps=steps,
        osc_dim=4,
        freq_gain=2.0,
        graph_top_k=32,
        graph_spatial_decay=0.35,
        gate_mode="raw",
        spike_per_component=True,
        spike_pulse_gain=0.0,
        no_center_pulse=False,
        geodesic_steps=3,
        geodesic_radius=1.5,
        geodesic_contrast=2.0,
        grid=16,
    )
    model = peer.build_core(config, device)
    return peer, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gamma-seq-path", required=True)
    parser.add_argument("--targets", default="/work/USERS/tkim1/clevr/with_masks/targets_v1.pt")
    parser.add_argument("--manifest", default="/export_home/tkim1/collaborative_test/data/split_manifest.json")
    parser.add_argument("--peer-evaluator", default="/export_home/tkim1/collaborative_test/evaluate_model.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--settle", type=int, default=64)
    parser.add_argument("--synchrony-threshold", type=float, default=0.35)
    args = parser.parse_args()
    if args.steps < 1 or not 0 <= args.settle < args.steps:
        parser.error("steps must be positive and settle must be in [0, steps)")

    manifest = json.loads(Path(args.manifest).read_text())
    start, end = manifest["ranges"]["validation"]
    if (start, end) != (6000, 7000):
        raise ValueError(f"Expected peer validation 6000-6999, got {start}-{end - 1}.")
    blob = torch.load(args.targets, map_location="cpu", weights_only=True)
    if blob.get("contract_version") != 1 or "patch_labels" not in blob or "names" not in blob:
        raise ValueError("Peer target file must be contract v1 with patch_labels and names.")
    target = blob["patch_labels"][start:end].long()
    expected_names = manifest["splits"]["validation"]
    if blob["names"][start:end] != expected_names:
        raise ValueError("Peer validation target names do not match manifest order.")
    if tuple(target.shape) != (1000, 16, 16):
        raise ValueError(f"Expected [1000,16,16] targets, got {tuple(target.shape)}.")
    gamma = torch.load(args.gamma_seq_path, map_location="cpu", weights_only=True)[start:end].float()
    if gamma.size(0) != target.size(0):
        raise ValueError("Peer gamma and targets have different validation counts.")

    # Import the peer's code path because its checkpoint schema differs from ours.
    peer, model = build_peer_evaluator(args.peer_evaluator, args.checkpoint,
                                       args.device, args.steps)
    model.membrane_layer.vth = 0.06
    spikes_batches, components_batches = [], []
    with torch.no_grad():
        for offset in range(0, len(gamma), args.batch_size):
            batch = gamma[offset:offset + args.batch_size].to(args.device)
            _, spikes, _, _ = model(batch, return_core_out=True, return_theta=True)
            if model.last_component_spikes is None:
                raise RuntimeError("Checkpoint/configuration did not return component spikes.")
            spikes_batches.append(spikes.float().cpu())
            components_batches.append(model.last_component_spikes.float().cpu())
    spikes = torch.cat(spikes_batches)
    components = torch.cat(components_batches)
    groups = peer.spike_synchrony_components(
        spikes, synchrony_threshold=args.synchrony_threshold,
        min_group_size=2, settle=args.settle,
        components=components, background="largest_component",
    )

    # GT enters only after the spike-derived groups are fixed.
    predicted_counts = [len(image_groups) for image_groups in groups]
    target_counts = [int(torch.unique(image[image != 0]).numel()) for image in target]
    report = {
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "split": "peer_validation",
        "ids": [start, end - 1],
        "images": len(target),
        "training_ids": [0, 5999],
        "inference": {"steps": args.steps, "settle": args.settle,
                      "synchrony_threshold": args.synchrony_threshold,
                      "membrane_vth": 0.06,
                      "geodesic_steps": 3, "geodesic_radius": 1.5,
                      "geodesic_contrast_initialization": 2.0,
                      "geodesic_temperature": 0.5, "geodesic_cap": 16.0,
                      "min_group_size": 2,
                      "background": "largest_component",
                      "component_combination": "product"},
        "ground_truth_used_for_prediction": False,
        "object_count": count_scores(predicted_counts, target_counts),
        "predicted_counts_per_image": predicted_counts,
        "target_counts_per_image": target_counts,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["object_count"], indent=2))


if __name__ == "__main__":
    main()
