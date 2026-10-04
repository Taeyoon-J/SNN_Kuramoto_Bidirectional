"""Frozen-checkpoint validation diagnostics; hooks observe the unchanged core."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "collaborative_test"))
sys.path.insert(0, str(ROOT / "collaborative_test/SW_0027_component_membrane_spectral"))
sys.path.insert(0, str(ROOT / "collaborative_test/SW_0028_spatial_membrane_spectral"))
from evaluate_fixed_split import _core
from evaluate import correlation, score, spectral_labels
from spatial_evaluate import spatial_kernel
from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch
from snn_kuramoto_bidirectional.loss_function import (
    UnsupervisedS2NetLoss, phase_locking_value, signal_synchrony,
)


def auc(same, different):
    if not len(same) or not len(different):
        return None
    ranks = rankdata(np.concatenate([same, different]))
    n, m = len(same), len(different)
    return float((ranks[:n].sum() - n * (n + 1) / 2) / (n * m))


def pair_summary(same, different):
    return {"same_count": len(same), "different_count": len(different),
            "same_mean": float(np.mean(same)) if len(same) else None,
            "different_mean": float(np.mean(different)) if len(different) else None,
            "auc": auc(same, different)}


def gamma_rows_for_ids(global_ids, global_start, row_count):
    """Map scene IDs to local gamma rows for full or aligned tensors."""
    rows = [int(scene_id) - int(global_start) for scene_id in global_ids]
    if not rows or min(rows) < 0 or max(rows) >= int(row_count):
        raise ValueError("Gamma tensor does not cover the requested global IDs.")
    return rows


def main():
    parser = argparse.ArgumentParser()
    for name in ("checkpoint", "gamma-path", "dataset-path", "output-path"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--start", type=int, default=1320,
                        help="First global validation scene ID (default preserves SW0034).")
    parser.add_argument("--gamma-global-start", type=int, default=0,
                        help="Global scene ID represented by gamma row zero.")
    parser.add_argument("--gamma-manifest", default=None,
                        help="Optional gamma alignment/provenance JSON to copy into results.")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--settle", type=int, default=64)
    parser.add_argument("--membrane-vth", type=float, default=0.06)
    parser.add_argument("--dendritic-projection", choices=["shared", "per_region"], default="shared")
    parser.add_argument("--graph-spatial-decay", type=float, default=0.55)
    parser.add_argument("--geodesic-steps", type=int, default=0)
    parser.add_argument("--geodesic-radius", type=float, default=1.5)
    parser.add_argument("--geodesic-contrast", type=float, default=2.0)
    parser.add_argument("--geodesic-temperature", type=float, default=0.5)
    parser.add_argument("--geodesic-cap", type=float, default=16.0)
    parser.add_argument("--kuramoto-backend", choices=["pairwise", "factorized"], default="pairwise")
    args = parser.parse_args()
    if not 1 <= args.count <= 320 or args.batch_size < 1 or not 0 <= args.settle < args.steps:
        raise ValueError("Require 1-320 images, positive batch size, and 0 <= settle < steps")
    torch.manual_seed(0)
    ids = list(range(args.start, args.start + args.count))
    gamma_blob = torch.load(args.gamma_path, map_location="cpu", weights_only=True)
    gamma_rows = gamma_rows_for_ids(ids, args.gamma_global_start, len(gamma_blob))
    gamma = gamma_blob[gamma_rows].float()
    gamma_manifest = None
    if args.gamma_manifest is not None:
        manifest_path = Path(args.gamma_manifest)
        if not manifest_path.is_file():
            raise ValueError(f"Gamma manifest does not exist: {manifest_path}")
        gamma_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with h5py.File(args.dataset_path, "r") as dataset:
        truth = clevr_mask_patch(torch.from_numpy(dataset["mask"][ids]), 8)["patch_labels"]
    model = _core(
        args.device, args.checkpoint, args.steps, args.dendritic_projection,
        args.geodesic_steps, args.geodesic_radius, args.geodesic_contrast,
        args.geodesic_temperature, args.geodesic_cap, args.graph_spatial_decay,
        args.kuramoto_backend,
    )
    model.membrane_layer.vth = float(args.membrane_vth)
    captured = {}

    def dendrite_hook(module, inputs, output):
        # Fold order is B,D,N. Capture exact SNN inputs and branch states.
        for key, value in (("gating", inputs[0][..., 0]),
                           ("h_wave", output), ("branch_state", module.h)):
            captured.setdefault(key, []).append(value.detach().cpu().clone())

    def membrane_hook(module, inputs, output):
        binary = (output[0] > module.v_th).to(output[0].dtype)
        captured.setdefault("binary_threshold", []).append(binary.detach().cpu())

    hooks = [model.dendric_layer.register_forward_hook(dendrite_hook),
             model.membrane_layer.register_forward_hook(membrane_hook)]
    grid = torch.stack(torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij"), -1).reshape(256, 2)
    d2 = ((grid[:, None] - grid[None, :]) ** 2).sum(-1)
    upper = torch.triu(torch.ones(256, 256, dtype=torch.bool), 1)
    strata = sorted(int(x) for x in torch.unique(d2[upper & (d2 <= 9)]))
    kernel = spatial_kernel(1.5)
    pair_values = {}
    predictions = {key: [] for key in ("spatial_only", "membrane_spatial", "spike_spatial")}
    control = spectral_labels(kernel, 10)
    branch_ratios = []
    signal_statistics = {}
    try:
        with torch.no_grad():
            for start in range(0, args.count, args.batch_size):
                captured.clear()
                batch = gamma[start:start + args.batch_size].to(args.device)
                _, spikes, membrane, theta = model(batch, return_core_out=True, return_theta=True)
                b, n, t = membrane.shape
                histories = {"membrane": membrane.cpu(), "gated_spike": spikes.cpu()}
                for key in ("gating", "h_wave", "binary_threshold"):
                    folded = torch.stack(captured[key], -1).reshape(b, model.osc_dim, n, t)
                    histories[key] = folded.mean(1)
                    for component in range(model.osc_dim):
                        histories[f"{key}_component{component}"] = folded[:, component]
                for key, value in (("membrane", model.last_component_out),
                                   ("gated_spike", model.last_component_spikes)):
                    for component in range(model.osc_dim):
                        histories[f"{key}_component{component}"] = value[:, component].cpu()
                branches = torch.stack(captured["branch_state"], -1)[..., args.settle:]
                ratio = branches.sum(2).abs() / branches.abs().sum(2).clamp_min(1e-8)
                branch_ratios.extend(ratio.mean((1, 2)).tolist())
                phase = phase_locking_value(theta, settle=args.settle, combine="mean").cpu()
                for key, history in histories.items():
                    observed = history[..., args.settle:]
                    stats = signal_statistics.setdefault(key, {"nodes": 0, "constant_nodes": 0,
                                                               "std_sum": 0.0, "mean_sum": 0.0})
                    temporal_std = observed.std(-1, unbiased=False)
                    stats["nodes"] += temporal_std.numel()
                    stats["constant_nodes"] += int((temporal_std <= 1e-8).sum())
                    stats["std_sum"] += float(temporal_std.sum())
                    stats["mean_sum"] += float(observed.mean(-1).sum())
                for offset in range(b):
                    matrices = {key: correlation(value[offset, :, args.settle:]).abs()
                                for key, value in histories.items()}
                    matrices.update(phase_plv=phase[offset], spatial_only=kernel)
                    matrices["membrane_spatial"] = matrices["membrane"] * kernel
                    matrices["spike_spatial"] = matrices["gated_spike"] * kernel
                    for key in predictions:
                        predictions[key].append(control if key == "spatial_only"
                                                else spectral_labels(matrices[key], 10))
                    labels = truth[start + offset].flatten()
                    fg = (labels[:, None] != 0) & (labels[None, :] != 0)
                    equal = labels[:, None] == labels[None, :]
                    for key, matrix in matrices.items():
                        target = pair_values.setdefault(key, {s: [[], []] for s in strata})
                        for s in strata:
                            mask = upper & fg & (d2 == s)
                            target[s][0].extend(matrix[mask & equal].tolist())
                            target[s][1].extend(matrix[mask & ~equal].tolist())
    finally:
        for hook in hooks:
            hook.remove()
    diagnostics = {}
    for key, values in pair_values.items():
        pooled = [sum((entry[i] for entry in values.values()), []) for i in (0, 1)]
        rows = {str(s): pair_summary(*entry) for s, entry in values.items()}
        valid = [row["auc"] for row in rows.values() if row["auc"] is not None]
        diagnostics[key] = {"pooled": pair_summary(*pooled), "by_squared_distance": rows,
                            "macro_distance_auc": float(np.mean(valid)) if valid else None}
    for key, entry in diagnostics.items():
        increments = [row["auc"] - diagnostics["spatial_only"]["by_squared_distance"][s]["auc"]
                      for s, row in entry["by_squared_distance"].items()
                      if row["auc"] is not None]
        entry["macro_auc_increment_over_spatial"] = float(np.mean(increments)) if increments else None
    # Separate 64-step/32-settle training-loss connectivity from 256/64 readout diagnostics.
    model.num_time_steps = 64
    criterion = UnsupervisedS2NetLoss(
        spike_rate_weight=0, spike_smooth_weight=0, spike_diversity_weight=0,
        structural_weight=0, plv_collapse_weight=1, plv_bimodality_weight=1,
        plv_balance_weight=10, plv_coherence_weight=.5,
        plv_target_density=.867, patch_grid_size=16)
    gradients = {}
    for source in ("phase", "membrane", "spikes"):
        model.zero_grad(set_to_none=True)
        _, spike, mem, theta = model(gamma[:min(2, args.count)].to(args.device),
                                     return_core_out=True, return_theta=True)
        plv = (phase_locking_value(theta, settle=32, combine="mean") if source == "phase"
               else signal_synchrony(mem if source == "membrane" else spike, settle=32))
        loss, _ = criterion(plv=plv)
        loss.backward()
        gradients[source] = {"loss": float(loss.detach()), "parameters": {
            name: {"connected": p.grad is not None,
                   "norm": float(p.grad.norm()) if p.grad is not None else None,
                   "finite": bool(torch.isfinite(p.grad).all()) if p.grad is not None else None}
            for name, p in model.named_parameters()}}
    digest = hashlib.sha256()
    with open(args.checkpoint, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    checkpoint_hash = digest.hexdigest()
    result = {"checkpoint": args.checkpoint, "checkpoint_sha256": checkpoint_hash,
              "ids": ids, "diagnostic_steps": args.steps, "diagnostic_settle": args.settle,
              "gradient_steps": 64, "gradient_settle": 32, "seed": 0,
              "gamma_source": {"path": args.gamma_path,
                               "global_start": args.gamma_global_start,
                               "manifest_path": args.gamma_manifest,
                               "manifest": gamma_manifest,
                               "local_rows": [gamma_rows[0], gamma_rows[-1]]},
              "target_source": {"path": args.dataset_path, "ids": [ids[0], ids[-1]],
                                "ground_truth_used_for_prediction": False},
              "core_config": {"membrane_vth": args.membrane_vth,
                              "dendritic_projection": args.dendritic_projection,
                              "graph_spatial_decay": args.graph_spatial_decay,
                              "geodesic_steps": args.geodesic_steps,
                              "geodesic_radius": args.geodesic_radius,
                              "geodesic_contrast": args.geodesic_contrast,
                              "geodesic_temperature": args.geodesic_temperature,
                              "geodesic_cap": args.geodesic_cap,
                              "kuramoto_backend": args.kuramoto_backend},
              "pair_diagnostics": diagnostics, "gradient_connectivity": gradients,
              "signal_statistics": {key: {
                  "node_count": stats["nodes"],
                  "constant_fraction": stats["constant_nodes"] / stats["nodes"],
                  "temporal_std_mean": stats["std_sum"] / stats["nodes"],
                  "activation_mean": stats["mean_sum"] / stats["nodes"]}
                  for key, stats in signal_statistics.items()},
              "branch_abs_sum_over_sum_abs_mean": float(np.mean(branch_ratios)),
              "readout": {key: {**score(torch.stack(value), truth),
                          "foreground_fraction": float((torch.stack(value) != 0).float().mean())}
                          for key, value in predictions.items()},
              "warning": "GT diagnoses only. Component AUCs are exploratory; no GT-selected masks. Equal-distance spatial AUC is theoretically 0.5; subtract measured spatial AUC to account for floating-point ties. Gated spikes are continuous; binary_threshold is the ungated comparator. No training or checkpoint update."}
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "readout": result["readout"]}, indent=2))


if __name__ == "__main__":
    main()
