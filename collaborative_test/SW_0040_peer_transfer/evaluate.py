"""Validation-only sweep of peer spike-synchrony connected-component readout.

Predictions are formed from spikes and their per-component synchrony only.
Ground-truth masks are used after prediction for the three mask scores and the
object-count diagnostics.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "collaborative_test"))
sys.path.insert(0, str(REPO_ROOT))

from evaluate_fixed_split import _core
from snn_kuramoto_bidirectional.evaluation import (
    clevr_mask_patch,
    evaluate_patch_masks,
    spatial_components_to_patch_labels,
)
from snn_kuramoto_bidirectional.loss_function import phase_locking_value
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_components


def _count_metrics(predicted_counts, target_counts):
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


def compact_sweep(rows):
    """Human-readable summary that preserves the per-target result structure."""
    compact = []
    for row in rows:
        item = {
            "affinity_mode": row.get("affinity_mode", "spike"),
            "spatial_sigma": row.get("spatial_sigma"),
            "threshold": row["synchrony_threshold"],
            "target_foreground": row.get("target_foreground"),
            "predicted_foreground_fraction": row.get("predicted_foreground_fraction"),
            "groups_per_image_mean": row.get("predicted_object_count", {}).get("mean"),
            "empty_image_count": row.get("predicted_object_count", {}).get("empty_image_count"),
            "targets": {
            name: {
                "metrics": scored["metrics"],
                "object_count": scored["object_count"],
            }
            for name, scored in row["scored_targets"].items()
            },
        }
        if "readout" in row:
            item["readout"] = row["readout"]
        if "rgb_border_distance_threshold" in row:
            item["rgb_border_distance_threshold"] = row["rgb_border_distance_threshold"]
        if "spatial_permutation_seed" in row:
            item["spatial_permutation_seed"] = row["spatial_permutation_seed"]
        compact.append(item)
    return compact


def gamma_row_indices(global_ids, global_start, row_count):
    """Map requested global scene IDs into an aligned gamma slice."""
    rows = [int(global_id) - int(global_start) for global_id in global_ids]
    if not rows or min(rows) < 0 or max(rows) >= int(row_count):
        raise ValueError("Gamma data does not contain requested validation IDs.")
    return rows


def phase_plv_components(plv, threshold, min_group_size=2):
    """Connected components of phase-PLV affinity; diagnostic readout only."""
    import numpy as np

    groups = []
    for image_affinity in plv.detach().cpu():
        adjacency = image_affinity.numpy() >= float(threshold)
        adjacency = adjacency.copy()
        np.fill_diagonal(adjacency, False)
        count, labels = connected_components(csr_matrix(adjacency), directed=False)
        components = [tuple(np.flatnonzero(labels == index).tolist())
                      for index in range(count)]
        components = sorted(
            (component for component in components
             if len(component) >= int(min_group_size)),
            key=len, reverse=True,
        )
        # Match the spike readout's largest-component background convention.
        groups.append(components[1:] if components else [])
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gamma-path", required=True)
    parser.add_argument("--gamma-global-start", type=int, default=0,
                        help="Global ID represented by gamma row 0; default 0 for full tensors.")
    parser.add_argument("--gamma-manifest", default=None,
                        help="Optional provenance manifest for a sliced/aligned gamma tensor.")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--peer-targets", default=None,
                        help="Optional peer targets_v1.pt; scores same predictions against these labels too.")
    parser.add_argument("--peer-manifest", default=None,
                        help="Peer split manifest; required with --peer-targets.")
    parser.add_argument("--peer-index-map", default="identity",
                        help=("Use 'identity' for the verified same-index pairing, or pass JSON "
                              "mapping our global IDs to peer target rows."))
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--start", type=int, default=1320)
    parser.add_argument("--count", type=int, default=320)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--settle", type=int, default=64)
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5])
    parser.add_argument("--affinity-modes", nargs="+",
                        choices=["spike", "spike_binary", "spike_spatial", "spike_spatial_permuted", "spatial_only"],
                        default=["spike"],
                        help="Spike, per-component binary, spatial, permuted-spatial, or spatial-only affinity controls.")
    parser.add_argument("--spatial-sigmas", type=float, nargs="+", default=[float("inf")],
                        help="Gaussian patch-grid sigma values; use inf for the uniform kernel.")
    parser.add_argument("--spatial-permutation-seeds", type=int, nargs="+", default=[0],
                        help="Deterministic node-permutation seeds used by spike_spatial_permuted.")
    parser.add_argument("--min-group-size", type=int, default=2)
    parser.add_argument("--membrane-vth", type=float, default=2.0,
                        help="Set the loaded membrane threshold; SW_0038 spike5 uses 2.0.")
    parser.add_argument("--background", choices=["largest_component", "activity", "hybrid"],
                        default="largest_component")
    parser.add_argument("--foreground-threshold", type=float, default=0.15)
    parser.add_argument("--synchrony-quantile", type=float, default=0.35)
    parser.add_argument(
        "--target-foreground", type=float, default=None,
        help=("Optional fixed foreground-area prior. When set, each image uses "
              "a label-free threshold search to approach this coverage; the "
              "provided synchrony threshold is retained only as provenance."),
    )
    parser.add_argument("--rgb-border-distance-thresholds", type=float, nargs="+", default=[],
                        help="Optional GT-free whole-component veto using robust RGB color distance from border patches.")
    parser.add_argument("--phase-endpoint", action="store_true",
                        help="Score phase-PLV connected components as a diagnostic alongside spikes.")
    parser.add_argument("--event-diagnostics", action="store_true",
                        help="Report GT-free component binary-event rate, always-on, constant-history and temporal-std diagnostics.")
    parser.add_argument("--dendritic-projection", choices=["shared", "per_region"], default="shared")
    parser.add_argument("--geodesic-steps", type=int, default=0)
    parser.add_argument("--geodesic-radius", type=float, default=1.5)
    parser.add_argument("--geodesic-contrast", type=float, default=2.0)
    parser.add_argument("--geodesic-temperature", type=float, default=0.5)
    parser.add_argument("--geodesic-cap", type=float, default=16.0)
    parser.add_argument("--graph-spatial-decay", type=float, default=0.55)
    parser.add_argument("--kuramoto-backend", choices=["pairwise", "factorized"], default="pairwise")
    parser.add_argument("--gate-mode", choices=["sigmoid", "raw", "phase_mean", "centered_raw", "signed_mask", "phasor_imag_raw"], default="raw",
                        help="Transduction/gating mode; centered modes retain raw mask membrane gating.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    reshape_binary_crossings = None
    if args.event_diagnostics:
        sys.path.insert(0, str(REPO_ROOT / "collaborative_test/SW_0083_threshold_recalibration"))
        from event_diagnostics import reshape_binary_crossings
    import h5py

    if args.start < 1000 and args.start + args.count > 0:
        raise ValueError("Evaluation must exclude training IDs 0-999.")
    if args.count < 1 or args.settle < 0 or args.settle >= args.steps:
        raise ValueError("Require positive count and 0 <= settle < steps.")
    if not args.thresholds or any(not 0.0 <= x <= 1.0 for x in args.thresholds):
        raise ValueError("Synchrony thresholds must lie in [0, 1].")
    if args.target_foreground is not None and not 0.0 < args.target_foreground < 1.0:
        raise ValueError("target foreground must lie strictly between 0 and 1")
    if args.target_foreground is not None and len(args.thresholds) != 1:
        raise ValueError("target foreground evaluation requires exactly one provenance threshold")
    if any(not math.isfinite(x) or x <= 0 for x in args.rgb_border_distance_thresholds):
        raise ValueError("RGB border distance thresholds must be finite and positive")
    if args.rgb_border_distance_thresholds and any(
            mode not in {"spike", "spike_binary"} for mode in args.affinity_modes):
        raise ValueError("RGB border component filtering requires spike or spike_binary affinity")
    if not args.spatial_sigmas or any(x <= 0.0 or x != x for x in args.spatial_sigmas):
        raise ValueError("Spatial sigma values must be positive (or inf).")
    if not args.spatial_permutation_seeds or any(seed < 0 for seed in args.spatial_permutation_seeds):
        raise ValueError("Spatial permutation seeds must be non-negative integers.")
    affinity_runs = []
    for mode in args.affinity_modes:
        sigmas = [None] if mode in {"spike", "spike_binary"} else args.spatial_sigmas
        for sigma in sigmas:
            seeds = args.spatial_permutation_seeds if mode == "spike_spatial_permuted" else [None]
            affinity_runs.extend((mode, sigma, seed) for seed in seeds)
    ids = list(range(args.start, args.start + args.count))
    gamma_blob = torch.load(args.gamma_path, map_location="cpu", weights_only=True)
    gamma_rows = gamma_row_indices(ids, args.gamma_global_start, len(gamma_blob))
    gamma = gamma_blob[gamma_rows].float()
    gamma_manifest = None
    if args.gamma_manifest is not None:
        manifest_path = Path(args.gamma_manifest)
        if not manifest_path.is_file():
            raise ValueError(f"Gamma manifest does not exist: {manifest_path}")
        gamma_manifest = json.loads(manifest_path.read_text())
    with h5py.File(args.dataset_path, "r") as dataset:
        rgb_images = (torch.from_numpy(dataset["image"][ids]).permute(0, 3, 1, 2).contiguous()
                      if args.rgb_border_distance_thresholds else None)
        target = clevr_mask_patch(torch.from_numpy(dataset["mask"][ids]), 8)["patch_labels"]
    targets = {"our_hdf5": target}
    target_provenance = {"our_hdf5": {"path": args.dataset_path, "ids": [ids[0], ids[-1]]}}
    if args.peer_targets is not None:
        if args.peer_manifest is None:
            raise ValueError("--peer-targets requires --peer-manifest.")
        peer_blob = torch.load(args.peer_targets, map_location="cpu", weights_only=True)
        if peer_blob.get("contract_version") != 1 or "patch_labels" not in peer_blob:
            raise ValueError("Peer targets must be contract_version 1 with patch_labels.")
        peer_labels = peer_blob["patch_labels"]
        if args.peer_index_map == "identity":
            mapped = list(ids)
        else:
            index_map = json.loads(Path(args.peer_index_map).read_text())
            mapped = []
            for our_id in ids:
                if str(our_id) not in index_map:
                    raise ValueError(f"Peer index map is missing our HDF5/gamma ID {our_id}.")
                mapped.append(int(index_map[str(our_id)]))
        if min(mapped) < 0 or max(mapped) >= len(peer_labels):
            raise ValueError("Peer index map points outside the target blob.")
        manifest = json.loads(Path(args.peer_manifest).read_text())
        names = peer_blob.get("names")
        if names is None or len(names) != len(peer_labels):
            raise ValueError("Peer targets must include names aligned with patch_labels.")
        for split, bounds in manifest["ranges"].items():
            start, end = bounds
            split_names = manifest["splits"][split]
            if end - start != len(split_names):
                raise ValueError(f"Peer manifest {split} range and name list disagree.")
            overlap = [row for row in mapped if start <= row < end]
            if any(names[row] != split_names[row - start] for row in overlap):
                raise ValueError(f"Peer target names do not match manifest {split} ordering.")
        if args.peer_index_map == "identity" and (ids[0] < 1000 or ids[-1] >= len(peer_labels)):
            raise ValueError("Identity pairing is reserved for our held-out IDs 1000+.")
        peer_target = peer_labels[mapped].long()
        if tuple(peer_target.shape[1:]) != (16, 16):
            raise ValueError("Peer patch_labels must have shape [images, 16, 16].")
        targets["peer_targets"] = peer_target
        target_provenance["peer_targets"] = {
            "path": args.peer_targets,
            "manifest": args.peer_manifest,
            "index_map": args.peer_index_map,
            "peer_ids": [mapped[0], mapped[-1]],
            "peer_id_mapping_verified": args.peer_index_map == "identity",
            "peer_rows_split": [
                split for split, bounds in manifest["ranges"].items()
                if bounds[0] <= mapped[0] < bounds[1]
            ],
            "peer_split_role": (
                "train_for_peer_model_but_unseen_by_our_checkpoint"
                if args.peer_index_map == "identity" and mapped[0] < manifest["ranges"]["validation"][0]
                else "peer_validation"
            ),
            "diagnostic_only": True,
            "our_checkpoint_train_ids": [0, 999],
            "our_checkpoint_train_range_exclusive": [0, 1000],
            "pairing_evidence": {
                "same_index_binary_iou": 0.27338,
                "same_index_object_count_correlation": 0.94091,
                "offset_plus_one_binary_iou": 0.14144,
                "offset_plus_one_object_count_correlation": 0.03439,
                "reversed_binary_iou": 0.14596,
                "reversed_object_count_correlation": 0.01306,
            } if args.peer_index_map == "identity" else None,
        }

    model = _core(
        args.device, args.checkpoint, args.steps, args.dendritic_projection,
        args.geodesic_steps, args.geodesic_radius, args.geodesic_contrast,
        args.geodesic_temperature, args.geodesic_cap, args.graph_spatial_decay,
        args.kuramoto_backend,
        args.gate_mode,
    )
    model.membrane_layer.vth = float(args.membrane_vth)
    spikes_rows, components_rows, phase_rows, event_rows = [], [], [], []
    captured_binary_steps = []
    def membrane_hook(module, inputs, output):
        captured_binary_steps.append((output[0] > module.v_th).detach().cpu().numpy())
    hook_handle = model.membrane_layer.register_forward_hook(membrane_hook) if args.event_diagnostics else None
    try:
        with torch.no_grad():
            for start in range(0, args.count, args.batch_size):
                batch = gamma[start:start + args.batch_size].to(args.device)
                captured_binary_steps.clear()
                _, spikes, _, theta = model(batch, return_core_out=True, return_theta=True)
                spikes_rows.append(spikes.float().cpu())
                component_activity = model.last_component_spikes
                components_rows.append(
                    component_activity.float().cpu() if component_activity is not None else None
                )
                if args.event_diagnostics:
                    if component_activity is None:
                        raise RuntimeError("event diagnostics require component dynamics")
                    binary = reshape_binary_crossings(
                        captured_binary_steps, len(batch), model.osc_dim,
                        model.in_dim, args.steps)
                    event_rows.append(binary)
                phase_rows.append(phase_locking_value(theta, settle=args.settle,
                                                      combine="product").cpu())
    finally:
        if hook_handle is not None:
            hook_handle.remove()
    spikes = torch.cat(spikes_rows)
    components = torch.cat(components_rows) if components_rows[0] is not None else None
    phase = torch.cat(phase_rows)
    event_diagnostics = None
    if args.event_diagnostics:
        import numpy as np
        binary_events = np.concatenate(event_rows, axis=0)[..., args.settle:]
        temporal_std = binary_events.astype(np.float32).std(axis=-1)
        event_diagnostics = {
            "binary_event_rate": float(binary_events.mean()),
            "binary_always_on_fraction": float((binary_events.mean(axis=-1) == 1.0).mean()),
            "binary_constant_history_fraction": float((temporal_std <= 1e-8).mean()),
            "binary_temporal_std_mean": float(temporal_std.mean()),
            "diagnostic_source": "membrane forward-hook (pre-threshold membrane > current v_th), after settle; no labels used",
            "shape_after_settle": list(binary_events.shape),
            "binary_values_only": bool(np.isin(binary_events, (False, True)).all()),
        }

    rows = []
    for affinity_mode, spatial_sigma, spatial_permutation_seed in affinity_runs:
      for threshold in args.thresholds:
        groups = spike_synchrony_components(
            spikes,
            synchrony_threshold=threshold,
            min_group_size=args.min_group_size,
            settle=args.settle,
            components=components,
            background=args.background,
            foreground_threshold=args.foreground_threshold,
            synchrony_quantile=args.synchrony_quantile,
            target_foreground=args.target_foreground,
            spatial_sigma=spatial_sigma,
            spatial_grid_size=16,
            affinity_mode=affinity_mode,
            spatial_permutation_seed=spatial_permutation_seed,
        )
        prediction_candidates = [{"readout": "spike_connected_components", "groups": groups}]
        if args.rgb_border_distance_thresholds:
            from SW_0077_rgb_border_component_veto.rgb_filter import border_color_component_filter
            for distance_threshold in args.rgb_border_distance_thresholds:
                filtered, diagnostics = border_color_component_filter(
                    rgb_images, groups, distance_threshold, grid_size=16
                )
                prediction_candidates.append({
                    "readout": "spike_components_rgb_border_veto",
                    "groups": filtered,
                    "rgb_border_distance_threshold": float(distance_threshold),
                    "rgb_diagnostics": diagnostics,
                })
        phase_endpoint = None
        if args.phase_endpoint:
            phase_groups = phase_plv_components(phase, threshold, args.min_group_size)
            phase_prediction = spatial_components_to_patch_labels(phase_groups, 16)
            phase_counts = [len(image_groups) for image_groups in phase_groups]
            phase_scores = {}
            for target_name, target_labels in targets.items():
                scores = evaluate_patch_masks(phase_prediction, target_labels)
                phase_target_counts = [
                    int(torch.unique(image[image != 0]).numel())
                    for image in target_labels
                ]
                phase_scores[target_name] = {
                    "metrics": {key: float(value) for key, value in scores["mean"].items()},
                    "object_count": _count_metrics(phase_counts, phase_target_counts),
                }
            phase_endpoint = {
                "diagnostic_only": True,
                "readout": "phase_plv_threshold_connected_components",
                "predicted_object_count_mean": float(
                    torch.tensor(phase_counts, dtype=torch.float32).mean()),
                "scored_targets": phase_scores,
            }
        for candidate in prediction_candidates:
            candidate_groups = candidate["groups"]
            prediction = spatial_components_to_patch_labels(candidate_groups, 16)
            predicted_counts = [len(image_groups) for image_groups in candidate_groups]
            scored_targets = {}
            for target_name, target_labels in targets.items():
                scores = evaluate_patch_masks(prediction, target_labels)
                target_counts = [int(torch.unique(image[image != 0]).numel()) for image in target_labels]
                scored_targets[target_name] = {
                    "metrics": {key: float(value) for key, value in scores["mean"].items()},
                    "valid_count": {key: int(value) for key, value in scores["valid_count"].items()},
                    "object_count": _count_metrics(predicted_counts, target_counts),
                    "target_foreground_fraction": float((target_labels != 0).float().mean()),
                    "per_image": {key: value.tolist() for key, value in scores["per_image"].items()},
                }
            row = {
                "affinity_mode": affinity_mode,
                "spatial_sigma": ("inf" if spatial_sigma is not None and math.isinf(spatial_sigma)
                                  else spatial_sigma),
                "synchrony_threshold": float(threshold),
                "target_foreground": args.target_foreground,
                "predicted_foreground_fraction": float((prediction != 0).float().mean()),
                "predicted_object_count": {
                    "mean": float(torch.tensor(predicted_counts, dtype=torch.float32).mean()),
                    "per_image": predicted_counts,
                    "empty_image_count": int(sum(count == 0 for count in predicted_counts)),
                },
                "scored_targets": scored_targets,
                "phase_endpoint": phase_endpoint if candidate["readout"] == "spike_connected_components" else None,
            }
            if candidate["readout"] != "spike_connected_components":
                row["readout"] = candidate["readout"]
            for key in ("rgb_border_distance_threshold", "rgb_diagnostics"):
                if key in candidate:
                    row[key] = candidate[key]
            if spatial_permutation_seed is not None:
                row["spatial_permutation_seed"] = int(spatial_permutation_seed)
            rows.append(row)

    offdiag = ~torch.eye(phase.size(-1), dtype=torch.bool)
    report = {
        "checkpoint": args.checkpoint,
        "split": "validation",
        "ids": [args.start, args.start + args.count - 1],
        "images": args.count,
        "gamma_source": {
            "path": args.gamma_path,
            "global_start": args.gamma_global_start,
            "manifest_path": args.gamma_manifest,
            "manifest": gamma_manifest,
        },
        "target_sources": target_provenance,
        "classifier": "peer_spike_synchrony_components",
        "component_combination": "clamped-positive per-component correlation product",
        "background": args.background,
        "inference": {
            "steps": args.steps,
            "settle": args.settle,
            "membrane_vth": args.membrane_vth,
            "min_group_size": args.min_group_size,
            "dendritic_projection": args.dendritic_projection,
            "graph_spatial_decay": args.graph_spatial_decay,
            "kuramoto_backend": args.kuramoto_backend,
            "gate_mode": args.gate_mode,
            "geodesic_steps": args.geodesic_steps,
            "geodesic_radius": args.geodesic_radius,
            "geodesic_contrast": args.geodesic_contrast,
            "geodesic_temperature": args.geodesic_temperature,
            "geodesic_cap": args.geodesic_cap,
            "synchrony_thresholds": args.thresholds,
            "target_foreground": args.target_foreground,
            "affinity_modes": args.affinity_modes,
            "spatial_sigmas": [None if mode in {"spike", "spike_binary"} else
                               ["inf" if math.isinf(sigma) else sigma
                                for sigma in args.spatial_sigmas]
                                for mode in args.affinity_modes],
            "spatial_affinity_formula": "S_ij * exp(-d_ij^2/(2*sigma^2)); spatial_only uses G, spike uses S",
            "spatial_grid_size": [16, 16],
        },
        "settle": args.settle,
        "gamma_global_start": args.gamma_global_start,
        "phase_endpoint_diagnostic_only": args.phase_endpoint,
        "phase_diagnostic": {
            "product_plv_mean": float(phase.mean()),
            "per_image_mean_offdiagonal_plv": phase[:, offdiag].mean(dim=1).tolist(),
        },
        "ground_truth_used_for_prediction": False,
        "sweep": rows,
    }
    if event_diagnostics is not None:
        report["event_diagnostics"] = event_diagnostics
    if "spike_spatial_permuted" in args.affinity_modes:
        report["spatial_permutation_seeds"] = [int(seed) for seed in args.spatial_permutation_seeds]
    if args.rgb_border_distance_thresholds:
        report["inference"]["rgb_border_distance_thresholds"] = args.rgb_border_distance_thresholds
        report["inference"]["rgb_background_model"] = (
            "per-image border-patch RGB median and 1.4826*MAD (scale floor 4/255 intensity levels)"
        )
    if "spike_binary" in args.affinity_modes:
        report["inference"]["spike_binary_definition"] = (
            "per-component crossing traces thresholded as components != 0 before the standard correlation product"
        )
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "ids": report["ids"],
        "sweep": compact_sweep(rows),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
