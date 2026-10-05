"""Causal interventions on one frozen SW0053 seed0 epoch-25 checkpoint."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "collaborative_test"), str(ROOT / "collaborative_test/SW_0027_component_membrane_spectral"),
                str(ROOT / "collaborative_test/SW_0028_spatial_membrane_spectral"), str(ROOT)]
from evaluate_fixed_split import _core
from evaluate import correlation, score, spectral_labels
from spatial_evaluate import spatial_kernel
from snn_kuramoto_bidirectional import s2net_cls as core_module
from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch, evaluate_patch_masks, spatial_components_to_patch_labels
from snn_kuramoto_bidirectional.loss_function import phase_locking_value
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_components
from interventions import permute_gate_regions, temporal_mean_gates


def gamma_rows(nrows, global_start, count):
    if nrows == count:
        return list(range(count))
    if nrows >= global_start + count:
        return list(range(global_start, global_start + count))
    if global_start >= 1320 and nrows >= count:
        return list(range(count))
    raise ValueError("gamma tensor does not cover requested validation rows")


def auc(same, different):
    if not same or not different:
        return None
    n, m = len(same), len(different)
    ranks = rankdata(np.concatenate((same, different)))
    return float((ranks[:n].sum() - n * (n + 1) / 2) / (n * m))


def distance_pair_values(matrix, labels, squared_distance):
    values = {}
    upper = np.triu(np.ones_like(squared_distance, dtype=bool), 1)
    labels = np.asarray(labels).reshape(-1)
    foreground = (labels[:, None] != 0) & (labels[None, :] != 0)
    same_object = labels[:, None] == labels[None, :]
    for distance in range(1, 10):
        mask = upper & foreground & (squared_distance == distance)
        values[str(distance)] = (
            matrix[mask & same_object].astype(float).tolist(),
            matrix[mask & ~same_object].astype(float).tolist(),
        )
    return values


def summarize_auc(entries):
    rows = {}
    for signal, strata in entries.items():
        per_distance = {d: auc(same, diff) for d, (same, diff) in strata.items()}
        valid = [x for x in per_distance.values() if x is not None]
        rows[signal] = {"macro_distance_auc": float(np.mean(valid)) if valid else None,
                        "by_squared_distance": per_distance}
    return rows


def run_one(model, gamma, condition, seed=None, means=None, permutation=None):
    batch_size = gamma.shape[0]
    originals = core_module.sinusoidal_gating
    gate_records, h_records, graph_records = [], [], []

    def gate_wrapper(theta_hist, t, delay, gate_mode="sigmoid"):
        drive_gate, membrane_gate = originals(theta_hist, t, delay, gate_mode=gate_mode)
        if condition == "gate_perm":
            drive_gate, membrane_gate = permute_gate_regions(drive_gate, membrane_gate, permutation)
        elif condition == "gate_mean":
            drive_gate, membrane_gate = means
        gate_records.append((drive_gate.detach().cpu(), membrane_gate.detach().cpu()))
        return drive_gate, membrane_gate

    def dendrite_hook(_module, _inputs, output):
        shape = output.shape
        if model.spike_per_component:
            output = output.reshape(batch_size, model.osc_dim, model.in_dim).mean(dim=1)
        h_records.append(output.detach().cpu())

    def graph_hook(_module, _inputs, output):
        graph_records.append(output.detach().cpu().clone())

    hooks = [model.dendric_layer.register_forward_hook(dendrite_hook)]
    if model.graph_generator is not None:
        hooks.append(model.graph_generator.register_forward_hook(graph_hook))
    old_k = model.kuramoto.K
    if condition == "K0":
        model.kuramoto.K = 0.0
    core_module.sinusoidal_gating = gate_wrapper
    try:
        with torch.no_grad():
            _, spikes, membrane, theta = model(gamma, return_core_out=True, return_theta=True)
        if condition == "K0" and model.kuramoto.K != 0.0:
            raise AssertionError("K=0 intervention was not active during rollout")
    finally:
        model.kuramoto.K = old_k
        core_module.sinusoidal_gating = originals
        for hook in hooks:
            hook.remove()
    gate_drive = torch.stack([x[0] for x in gate_records], dim=-1)  # [B,N,D,T]
    gate_membrane = torch.stack([x[1] for x in gate_records], dim=-1)  # [B,N,T]
    h_wave = torch.stack(h_records, dim=-1)
    components = model.last_component_spikes.detach().cpu().clone() if model.last_component_spikes is not None else None
    return {"theta": theta.detach().cpu(), "spikes": spikes.detach().cpu(),
            "membrane": membrane.detach().cpu(), "components": components,
            "gate_drive": gate_drive, "gate": gate_membrane, "h_wave": h_wave,
            "graph": graph_records[0] if graph_records else None,
            "K_used": 0.0 if condition == "K0" else float(old_k)}


def main():
    p = argparse.ArgumentParser()
    for name in ("checkpoint", "gamma-path", "gamma-manifest", "dataset-path", "output-path"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--count", type=int, default=320)
    p.add_argument("--start", type=int, default=1320)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--settle", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.count < 1 or args.start < 1320 or args.start + args.count > 1640:
        raise ValueError("Use only aligned validation IDs 1320-1639")
    if args.steps <= args.settle or args.settle < 0 or args.batch_size < 1:
        raise ValueError("Invalid rollout window or batch size")
    ids = list(range(args.start, args.start + args.count))
    gamma_blob = torch.load(args.gamma_path, map_location="cpu", weights_only=True)
    gamma = gamma_blob[gamma_rows(len(gamma_blob), args.start, args.count)].float()
    manifest = json.loads(Path(args.gamma_manifest).read_text(encoding="utf-8"))
    if manifest.get("image_ids") != [1320, 1639]:
        raise ValueError("Unexpected aligned gamma manifest ID range")
    with h5py.File(args.dataset_path, "r") as h5:
        truth = clevr_mask_patch(torch.from_numpy(h5["mask"][ids]), 8)["patch_labels"]

    model = _core(args.device, args.checkpoint, args.steps, "shared", 3, 1.5, 2.0, 0.5, 16.0, 0.35, "factorized")
    model.membrane_layer.vth = 0.06
    if model.kuramoto.spike_pulse_gain is not None:
        raise RuntimeError("Gate interventions require pulse feedback off to guarantee theta invariance")
    if model.graph_generator is None or model.graph_generator.uses_feedback:
        raise RuntimeError("Causal comparison assumes the fixed learned graph, with feedback off")
    if model.theta_init_noise != 0.0:
        raise RuntimeError("Theta initialization noise must be zero so every condition starts identically")

    coords = torch.stack(torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij"), -1).reshape(256, 2)
    d2 = ((coords[:, None] - coords[None, :]) ** 2).sum(-1).numpy()
    kernel = spatial_kernel(1.5)
    conditions = ["normal", "gate_perm_s0", "gate_perm_s1", "gate_perm_s2", "gate_mean", "K0"]
    auc_acc = {c: {s: {str(d): ([], []) for d in range(1, 10)} for s in ("phase", "gate", "h_wave", "membrane", "spike")} for c in conditions}
    predictions = {c: {"spike_cc": [], "membrane_spatial": []} for c in conditions}
    invariants = {c: [] for c in conditions if c.startswith("gate_")}
    count_means = {c: [] for c in conditions}
    theta_deltas = {c: [] for c in conditions}
    original_K = float(model.kuramoto.K)
    permutations = {}
    for seed in range(3):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        permutations[seed] = torch.randperm(model.in_dim, generator=gen)
    if any(torch.equal(permutations[s], torch.arange(model.in_dim)) for s in range(3)):
        raise RuntimeError("A fixed region permutation unexpectedly became identity")
    if any(torch.equal(permutations[s], torch.arange(model.in_dim)) for s in permutations):
        raise RuntimeError("A requested gate permutation unexpectedly became identity")

    with torch.no_grad():
        for start in range(0, args.count, args.batch_size):
            batch = gamma[start:start + args.batch_size].to(args.device)
            base = run_one(model, batch, "normal")
            drive_mean, membrane_mean = temporal_mean_gates(
                base["gate_drive"].permute(0, 3, 1, 2), base["gate"].permute(0, 2, 1)
            )
            condition_outputs = {"normal": base}
            for seed in range(3):
                condition_outputs[f"gate_perm_s{seed}"] = run_one(
                    model, batch, "gate_perm", seed=seed, permutation=permutations[seed]
                )
            condition_outputs["gate_mean"] = run_one(
                model, batch, "gate_mean", means=(drive_mean.to(args.device), membrane_mean.to(args.device))
            )
            condition_outputs["K0"] = run_one(model, batch, "K0")

            for cond, out in condition_outputs.items():
                theta = out["theta"]
                # Prediction channels contain no target-dependent step.
                groups = spike_synchrony_components(
                    out["spikes"], synchrony_threshold=0.35, min_group_size=2,
                    settle=args.settle, components=out["components"],
                    background="largest_component")
                spike_mask = spatial_components_to_patch_labels(groups, 16)
                predictions[cond]["spike_cc"].extend(spike_mask)
                membrane_labels = [spectral_labels(correlation(m[ :, args.settle:]).abs() * kernel, 10)
                                   for m in out["membrane"]]
                predictions[cond]["membrane_spatial"].extend(membrane_labels)
                count_means[cond].extend(len(g) for g in groups)
                histories = {"gate": out["gate"], "h_wave": out["h_wave"],
                             "membrane": out["membrane"], "spike": out["spikes"]}
                for bi in range(len(batch)):
                    phase_matrix = phase_locking_value(
                        theta[bi:bi + 1], settle=args.settle, combine="product"
                    )[0].cpu().numpy()
                    matrices = {"phase": phase_matrix}
                    matrices.update({name: correlation(value[bi, :, args.settle:]).abs().numpy()
                                     for name, value in histories.items()})
                    label = truth[start + bi].flatten().numpy()
                    entries = {name: distance_pair_values(matrix, label, d2)
                               for name, matrix in matrices.items()}
                    for signal, strata in entries.items():
                        for distance, (same, different) in strata.items():
                            auc_acc[cond][signal][distance][0].extend(same)
                            auc_acc[cond][signal][distance][1].extend(different)
                    if cond.startswith("gate_"):
                        graph_exact = torch.equal(out["graph"], base["graph"])
                        theta_delta = float((out["theta"] - base["theta"]).abs().max())
                        invariants[cond].append({"graph_bitwise_equal": graph_exact,
                                                 "theta_max_abs_delta": theta_delta,
                                                 "theta_allclose": torch.allclose(out["theta"], base["theta"], atol=1e-6, rtol=0)})
                        if not graph_exact or not torch.allclose(out["theta"], base["theta"], atol=1e-6, rtol=0):
                            raise AssertionError(f"Gate intervention changed theta/graph for {cond}")
                    elif cond == "K0":
                        if out["graph"] is None or not torch.equal(out["graph"], base["graph"]):
                            raise AssertionError("K=0 unexpectedly changed graph generator output")
                        theta_deltas[cond].append(float((out["theta"] - base["theta"]).abs().max()))

    scored = {}
    for cond in conditions:
        scored[cond] = {}
        for readout, masks in predictions[cond].items():
            pred = torch.stack(masks).long()
            metric = evaluate_patch_masks(pred, truth)["mean"]
            scored[cond][readout] = {
                "metrics": {key: float(value) for key, value in metric.items()},
                "predicted_foreground_fraction": float((pred != 0).float().mean()),
                "predicted_object_count_mean": float(np.mean(count_means[cond])) if readout == "spike_cc" else score(pred, truth)["predicted_groups_mean"],
            }
    condition_results = {}
    for cond in conditions:
        condition_results[cond] = {
            "distance_controlled_macro_auc": summarize_auc(auc_acc[cond]),
            "fixed_readouts": scored[cond],
        }
        if cond.startswith("gate_"):
            condition_results[cond]["gate_invariants"] = invariants[cond]
        if cond == "K0":
            condition_results[cond]["kuramoto_K_during_rollout"] = 0.0
            condition_results[cond]["theta_max_abs_delta_from_normal"] = max(theta_deltas[cond], default=0.0)
    result = {
        "experiment": "SW0054 causal mechanism ablation",
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "ids": [ids[0], ids[-1]], "count": args.count,
        "gamma": {"path": args.gamma_path, "manifest_path": args.gamma_manifest,
                  "manifest": manifest, "global_start": args.start},
        "target": {"path": args.dataset_path, "ids": [ids[0], ids[-1]],
                   "ground_truth_used_for_prediction": False},
        "core": {"steps": args.steps, "settle": args.settle, "vth": 0.06,
                 "projection": "shared", "graph_decay": 0.35, "geodesic_steps": 3,
                 "backend": "factorized", "K_normal": original_K,
                 "spike_pulse_gain": None},
        "interventions": {
            "gate_permutation": {str(seed): permutations[seed].tolist() for seed in range(3)},
            "gate_mean": "separate per-image/per-region time mean for gamma_wave and membrane gate, repeated each step",
            "K0": "sets inter-region Kuramoto stiffness K to exactly zero; gamma/frequency/theta init untouched",
            "prediction_ground_truth_free": True,
        },
        "conditions": condition_results,
        "warning": "Validation diagnostic only; target masks are used only after prediction for pair labels and scores.",
    }
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "conditions": list(condition_results)}, indent=2))


if __name__ == "__main__":
    main()
