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


def main():
    parser = argparse.ArgumentParser()
    for name in ("checkpoint", "gamma-path", "dataset-path", "output-path"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 1 <= args.count <= 320 or args.batch_size < 1:
        raise ValueError("Require 1-320 validation images and positive batch size")
    torch.manual_seed(0)
    ids = list(range(1320, 1320 + args.count))
    gamma = torch.load(args.gamma_path, map_location="cpu", weights_only=True)[ids].float()
    with h5py.File(args.dataset_path, "r") as dataset:
        truth = clevr_mask_patch(torch.from_numpy(dataset["mask"][ids]), 8)["patch_labels"]
    model = _core(args.device, args.checkpoint, 256)
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
                branches = torch.stack(captured["branch_state"], -1)[..., 64:]
                ratio = branches.sum(2).abs() / branches.abs().sum(2).clamp_min(1e-8)
                branch_ratios.extend(ratio.mean((1, 2)).tolist())
                phase = phase_locking_value(theta, settle=64, combine="mean").cpu()
                for offset in range(b):
                    matrices = {key: correlation(value[offset, :, 64:]).abs()
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
              "ids": ids, "diagnostic_steps": 256, "diagnostic_settle": 64,
              "gradient_steps": 64, "gradient_settle": 32, "seed": 0,
              "pair_diagnostics": diagnostics, "gradient_connectivity": gradients,
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
