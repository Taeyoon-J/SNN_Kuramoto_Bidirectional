"""Read-only paired 64-vs-1024-horizon diagnostic for trained SW0123 legacy arms."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0123_adaptive_temporal_assignment import run
from collaborative_test.SW_0123_adaptive_temporal_assignment.diagnose_assignment import (
    DIAGNOSTIC_BATCHES, _diagnostic_training_batch, _load_completed, _quantiles, batch_diagnostics,
)
from collaborative_test.SW_0123_adaptive_temporal_assignment.model import (
    normalized_patch_centers,
)
from collaborative_test.SW_0123_adaptive_temporal_assignment.readout import assignment_to_labels
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity

SEEDS = (0, 1, 2)
ARM = "legacy_full"
HORIZONS = (64, 1024)
SETTLES = {64: 32, 1024: 512}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_static_gamma_horizon(gamma, horizon, settle):
    if gamma.ndim != 3 or tuple(gamma.shape[1:]) != (8, 256):
        raise ValueError("registered gamma input is static [B,8,256]; time belongs to the core")
    if int(horizon) <= int(settle):
        raise ValueError("core horizon must exceed registered settle length")
    return int(horizon)


def _rollout_components(core, gamma, settle):
    horizon = _validate_static_gamma_horizon(
        gamma, getattr(core, "num_time_steps", -1), settle)
    core(gamma, return_core_out=True, return_theta=True)
    components = core.last_component_spikes
    expected = (gamma.shape[0], 4, 256, horizon)
    if components is None or tuple(components.shape) != expected or not torch.isfinite(components).all():
        raise AssertionError("legacy source rollout returned invalid component-spike traces")
    return components


def _core_at_horizon(seed, loaded, horizon, device):
    if horizon == 64:
        core = loaded["core"]
        if int(core.num_time_steps) != horizon:
            raise AssertionError("loaded training core is not the registered T64 checkpoint")
        core._detect_object_groups = lambda _out, spikes: [[] for _ in range(spikes.shape[0])]
        return core
    core = run.base.make_core(device, int(horizon))
    core.load_state_dict(torch.load(loaded["source"], map_location=device, weights_only=True), strict=True)
    core._detect_object_groups = lambda _out, spikes: [[] for _ in range(spikes.shape[0])]
    trained_core = loaded["folder"] / "core.pt"
    core.load_state_dict(torch.load(trained_core, map_location=device, weights_only=True), strict=True)
    core._detect_object_groups = lambda _out, spikes: [[] for _ in range(spikes.shape[0])]
    core.eval(); core.graph_generator.eval()
    return core


@torch.inference_mode()
def _temporal_and_affinity_diagnostics(model, components, settle):
    b, d, n, t = components.shape
    rms = model.assignment_head.spike_rms.to(device=components.device, dtype=components.dtype)
    normalized = components / rms.view(1, 4, 1, 1)
    model_input = normalized[:, :, :, settle:].permute(0, 2, 1, 3).reshape(
        b * n, 4, t - settle)
    temporal_features = model.assignment_head.temporal(model_input)
    patch_embedding = temporal_features.mean(dim=-1).reshape(b, n, -1)
    embedding_spatial_variance = patch_embedding.var(dim=1, unbiased=False).mean(dim=-1)
    # Preserve one value per flattened image-patch; the tensor is [B*N, C, T].
    temporal_variance = temporal_features.var(dim=-1, unbiased=False).mean(dim=1)

    mean_activity = components.mean(dim=1)
    affinity = spike_synchrony_affinity(mean_activity, components=components,
                                       settle=settle, affinity_mode="spike")
    offdiag = ~torch.eye(n, dtype=torch.bool, device=components.device)
    q_values = affinity[:, offdiag]
    component_correlations = []
    for index in range(4):
        component_q = spike_synchrony_affinity(
            components[:, index], settle=settle, affinity_mode="spike")
        component_correlations.append(component_q[:, offdiag])
    per_component = torch.stack(component_correlations, dim=1)
    return {
        "settled_temporal_conv_feature_variance": _quantiles(temporal_variance.cpu().numpy()),
        "settled_patch_embedding_spatial_variance": _quantiles(embedding_spatial_variance.cpu().numpy()),
        "actual_four_component_positive_product_affinity": {
            "offdiagonal_value_quantiles": _quantiles(q_values.cpu().numpy()),
            "offdiagonal_positive_fraction": float((q_values > 0).float().mean()),
            "offdiagonal_zero_fraction": float((q_values == 0).float().mean()),
            "per_component_offdiagonal_correlation_quantiles": [
                _quantiles(per_component[:, index].cpu().numpy()) for index in range(4)],
        },
    }


@torch.inference_mode()
def _one_horizon_batch(core, model, gamma, rgb, arm, settle, centers, permutation):
    components = _rollout_components(core, gamma, settle)
    probability, slot_embedding, _patch_embedding = model.assignment_head(components[..., settle:])
    mixed_patches, decoded = model.rgb_decoder(probability, slot_embedding, centers)
    out = {"assignment": probability, "slot_embedding": slot_embedding,
           "decoded_slot_patches": decoded,
           "reconstructed_image": run.image_from_patches(mixed_patches)}
    labels = assignment_to_labels(probability)
    fg_fraction = (labels > 0).float().mean(dim=(1, 2))
    groups_per_image = torch.tensor([
        torch.unique(image[image > 0]).numel() for image in labels], dtype=torch.float32)
    bg_fraction = (labels == 0).float().mean(dim=(1, 2))
    return {
        "batch_assignment_diagnostics": batch_diagnostics(out, rgb, permutation),
        "foreground_fraction_per_image": [float(x) for x in fg_fraction],
        "foreground_group_count_per_image": [int(x) for x in groups_per_image],
        "background_fraction_per_image": [float(x) for x in bg_fraction],
        "all_background_image_count": int((fg_fraction == 0).sum()),
        "temporal_and_affinity": _temporal_and_affinity_diagnostics(model, components, settle),
    }


def diagnose(seed, device="cuda"):
    if seed not in SEEDS:
        raise ValueError("SW0123 horizon diagnostic is restricted to registered seeds 0/1/2")
    loaded = _load_completed(seed, ARM, device)
    gamma_cache, gamma_manifest, rgb_cache, _rgb_meta, rgb_sha = run.load_registered_data()
    cores = {horizon: _core_at_horizon(seed, loaded, horizon, device) for horizon in HORIZONS}
    centers = normalized_patch_centers(device=device)
    permutation = torch.as_tensor(np.random.default_rng(12301).permutation(256),
                                  dtype=torch.long, device=device)
    paired_batches = []
    image_ids, cache_rows, gamma_drift = [], [], []
    with torch.inference_mode():
        for batch in range(DIAGNOSTIC_BATCHES):
            start = batch * run.BATCH
            rows, _images, rgb, gamma = _diagnostic_training_batch(
                seed, loaded["rows"], start, gamma_cache, rgb_cache,
                loaded["encoder"], loaded["patcher"], loaded["mean"],
                loaded["std"], loaded["clip"], device)
            cached_gamma = gamma_cache[torch.as_tensor(rows, dtype=torch.long)].to(device)
            gamma_drift.append(float((gamma - cached_gamma).abs().max()))
            horizon_results = {}
            for horizon in HORIZONS:
                horizon_results[str(horizon)] = _one_horizon_batch(
                    cores[horizon], loaded["model"], gamma, rgb, ARM,
                    SETTLES[horizon], centers, permutation)
            paired_batches.append(horizon_results)
            image_ids.extend(loaded["ids"][start:start + run.BATCH].tolist())
            cache_rows.extend(rows.tolist())

    manifest_path = loaded["manifest_path"]
    folder = loaded["folder"]
    checkpoint_files = {name: folder / filename for name, filename in {
        "core": "core.pt", "encoder": "encoder.pt", "assignment_head": "assignment_head.pt",
        "rgb_decoder": "rgb_decoder.pt", "history": "history.json"}.items()}
    checkpoint_hashes = {name: _sha(path) for name, path in checkpoint_files.items()}
    if any(loaded["manifest"].get(name + "_sha256") != digest
           for name, digest in checkpoint_hashes.items()):
        raise AssertionError("checkpoint changed during paired horizon diagnostic")

    per_horizon = {}
    for horizon in HORIZONS:
        key = str(horizon)
        reports = [batch[key] for batch in paired_batches]
        per_horizon[key] = {
            "settle": SETTLES[horizon],
            "foreground_fraction_first64": [value for row in reports for value in row["foreground_fraction_per_image"]],
            "foreground_group_count_first64": [value for row in reports for value in row["foreground_group_count_per_image"]],
            "background_fraction_first64": [value for row in reports for value in row["background_fraction_per_image"]],
            "all_background_images_first64": sum(row["all_background_image_count"] for row in reports),
            "batch_diagnostics": [row["batch_assignment_diagnostics"] for row in reports],
            "temporal_and_affinity": [row["temporal_and_affinity"] for row in reports],
            "mean_foreground_fraction": float(np.mean([
                value for row in reports for value in row["foreground_fraction_per_image"]])),
            "mean_foreground_group_count": float(np.mean([
                value for row in reports for value in row["foreground_group_count_per_image"]])),
            "mean_background_fraction": float(np.mean([
                value for row in reports for value in row["background_fraction_per_image"]])),
        }
    return {
        "experiment": "SW0123", "stage": "paired_read_only_horizon_diagnostic",
        "status": "complete", "seed": seed, "arm": ARM,
        "ground_truth_used": False, "optimizer_updates": 0,
        "diagnostic_contract": {"same_ordered_training_images": 64, "batch_size": 16,
                                "batches": 4, "horizons": list(HORIZONS),
                                "settles": {str(k): v for k, v in SETTLES.items()},
                                "native_gamma_shape": [16, 8, 256], "permutation_seed": 12301},
        "training_image_ids": image_ids, "gamma_cache_rows": cache_rows,
        "trained_encoder_vs_source_gamma_max_abs_per_batch": gamma_drift,
        "gamma_cache_manifest_status": gamma_manifest.get("status"),
        "rgb_cache_sha256": rgb_sha, "source_core_sha256": _sha(loaded["source"]),
        "source_manifest_sha256": _sha(loaded["source_manifest"]),
        "training_manifest_sha256": _sha(manifest_path),
        "calibration_sha256": _sha(loaded["calibration_path"]),
        "training_checkpoint_sha256": checkpoint_hashes,
        "implementation_sha256": {
            "diagnostic": _sha(Path(__file__)), "run": _sha(run.HERE / "run.py"),
            "evaluate": _sha(run.HERE / "evaluate.py"), "model": _sha(HERE / "model.py"),
            "assignment_diagnostic": _sha(HERE / "diagnose_assignment.py"),
        },
        "paired_horizon_results": per_horizon,
        "comparison": {
            "mean_foreground_fraction_delta_1024_minus_64": (
                per_horizon["1024"]["mean_foreground_fraction"]
                - per_horizon["64"]["mean_foreground_fraction"]),
            "mean_foreground_group_count_delta_1024_minus_64": (
                per_horizon["1024"]["mean_foreground_group_count"]
                - per_horizon["64"]["mean_foreground_group_count"]),
            "same_images_and_regenerated_gamma_reused_for_both_horizons": True,
        },
        "note": "Same trained legacy checkpoint, encoder, head, decoder, and 64 ordered TRAIN IDs; only core horizon and registered settle length differ. This is read-only diagnosis, not an evaluation endpoint or new training result.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or HERE / "results_archive" / f"horizon_diagnostic_seed{args.seed}_legacy_full_20261009.json"
    if output.exists():
        raise FileExistsError(f"preserving existing horizon diagnostic: {output}")
    result = diagnose(args.seed, args.device)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "seed": args.seed,
                      "mean_foreground_fraction_delta_1024_minus_64": result["comparison"]["mean_foreground_fraction_delta_1024_minus_64"],
                      "all_background_images": {key: value["all_background_images_first64"]
                                                for key, value in result["paired_horizon_results"].items()},
                      "output": str(output)}, allow_nan=False))


if __name__ == "__main__":
    main()
