"""Causal interventions on one frozen SW0053 seed0 epoch-25 checkpoint."""
import argparse
import hashlib
import json
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
from interventions import apply_carrier_mask_intervention, temporal_means


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


def run_one(model, gamma, condition, means=None, permutation=None):
    batch_size = gamma.shape[0]
    originals = core_module.sinusoidal_gating
    carrier_records, mask_records, h_records, graph_records = [], [], [], []

    def gate_wrapper(theta_hist, t, delay, gate_mode="sigmoid"):
        if gate_mode != "raw":
            raise RuntimeError("SW0054 interventions are derived for the checkpoint's raw gate mode")
        _, local_mask = originals(theta_hist, t, delay, gate_mode=gate_mode)
        local_carrier = torch.sin(theta_hist[t])
        intervention = {
            "normal": "normal", "gate_perm": "gate_perm", "carrier_perm": "carrier_perm",
            "gate_mean": "gate_mean", "carrier_mean": "carrier_mean", "K0": "normal",
        }[condition]
        mean_carrier, mean_mask = means if means is not None else (None, None)
        drive_gate, used_carrier, used_mask = apply_carrier_mask_intervention(
            local_carrier, local_mask, intervention, permutation=permutation,
            mean_mask=mean_mask, mean_carrier=mean_carrier,
        )
        carrier_records.append(used_carrier.detach().cpu())
        mask_records.append(used_mask.detach().cpu())
        membrane_gate = used_mask
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
    carriers = torch.stack(carrier_records, dim=-1)  # [B,N,D,T]
    gate_membrane = torch.stack(mask_records, dim=-1)  # [B,N,T]
    h_wave = torch.stack(h_records, dim=-1)
    components = model.last_component_spikes.detach().cpu().clone() if model.last_component_spikes is not None else None
    expected = {
        "theta": (batch_size, model.num_time_steps, model.in_dim, model.osc_dim),
        "spikes": (batch_size, model.in_dim, model.num_time_steps),
        "membrane": (batch_size, model.in_dim, model.num_time_steps),
        "carrier": (batch_size, model.in_dim, model.osc_dim, model.num_time_steps),
        "gate": (batch_size, model.in_dim, model.num_time_steps),
        "h_wave": (batch_size, model.in_dim, model.num_time_steps),
    }
    actual = {"theta": tuple(theta.shape), "spikes": tuple(spikes.shape),
              "membrane": tuple(membrane.shape), "carrier": tuple(carriers.shape),
              "gate": tuple(gate_membrane.shape), "h_wave": tuple(h_wave.shape)}
    if actual != expected:
        raise RuntimeError(f"Unexpected causal intervention output shapes: {actual} != {expected}")
    if not all(torch.isfinite(value).all() for value in
               (theta, spikes, membrane, carriers, gate_membrane, h_wave)):
        raise RuntimeError(f"Non-finite output under intervention {condition}")
    return {"theta": theta.detach().cpu(), "spikes": spikes.detach().cpu(),
            "membrane": membrane.detach().cpu(), "components": components,
            "carrier": carriers, "gate": gate_membrane, "h_wave": h_wave,
            "graph": graph_records[0] if graph_records else None,
            "K_used": 0.0 if condition == "K0" else float(old_k)}


def main():
    p = argparse.ArgumentParser()
    for name in ("checkpoint", "gamma-path", "gamma-manifest", "dataset-path", "output-path"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--count", type=int, default=320)
    p.add_argument("--condition-set", choices=("pilot", "full"), default="full")
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
    if model.gate_mode != "raw":
        raise RuntimeError(f"Expected raw gate_mode, got {model.gate_mode!r}")
    full_conditions = (["normal"] + [f"gate_perm_s{i}" for i in range(3)]
                       + [f"carrier_perm_s{i}" for i in range(3)]
                       + ["gate_mean", "carrier_mean", "K0"])
    pilot_conditions = ["normal", "gate_perm_s0", "carrier_perm_s0", "gate_mean", "carrier_mean", "K0"]
    conditions = full_conditions if args.condition_set == "full" else pilot_conditions
    auc_acc = {c: {s: {str(d): ([], []) for d in range(1, 10)} for s in ("phase", "gate", "carrier", "h_wave", "membrane", "spike")} for c in conditions}
    predictions = {c: {"spike_cc": [], "membrane_spatial": []} for c in conditions}
    invariants = {c: [] for c in conditions if c.startswith("gate_")}
    invariants.update({c: [] for c in conditions if c.startswith("carrier_")})
    count_means = {c: [] for c in conditions}
    theta_deltas = {c: [] for c in conditions}
    activity_stats = {c: {"spike_event_rate": [], "membrane_variance": [], "membrane_abs_mean": []} for c in conditions}
    original_K = float(model.kuramoto.K)
    permutations = {}
    for seed in range(3):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        permutations[seed] = torch.randperm(model.in_dim, generator=gen)
    if any(torch.equal(permutations[s], torch.arange(model.in_dim)) for s in range(3)):
        raise RuntimeError("A fixed region permutation unexpectedly became identity")
    with torch.no_grad():
        for start in range(0, args.count, args.batch_size):
            batch = gamma[start:start + args.batch_size].to(args.device)
            base = run_one(model, batch, "normal")
            mean_carrier, mean_mask = temporal_means(
                base["carrier"].permute(0, 3, 1, 2), base["gate"].permute(0, 2, 1)
            )
            condition_outputs = {"normal": base}
            for cond in conditions:
                if cond == "normal":
                    continue
                if cond.startswith("gate_perm_s"):
                    seed = int(cond.rsplit("s", 1)[1])
                    condition_outputs[cond] = run_one(model, batch, "gate_perm", permutation=permutations[seed])
                elif cond.startswith("carrier_perm_s"):
                    seed = int(cond.rsplit("s", 1)[1])
                    condition_outputs[cond] = run_one(model, batch, "carrier_perm", permutation=permutations[seed])
                elif cond == "gate_mean":
                    condition_outputs[cond] = run_one(model, batch, "gate_mean", means=(None, mean_mask.to(args.device)))
                elif cond == "carrier_mean":
                    condition_outputs[cond] = run_one(model, batch, "carrier_mean", means=(mean_carrier.to(args.device), None))
                elif cond == "K0":
                    condition_outputs[cond] = run_one(model, batch, "K0")

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
                histories = {"gate": out["gate"], "carrier": out["carrier"].mean(dim=2), "h_wave": out["h_wave"],
                             "membrane": out["membrane"], "spike": out["spikes"]}
                for bi in range(len(batch)):
                    activity_stats[cond]["spike_event_rate"].append(float(
                        (out["components"][bi] > 0).float().mean()
                        if out["components"] is not None else (out["spikes"][bi] > 0).float().mean()))
                    activity_stats[cond]["membrane_variance"].append(float(
                        out["membrane"][bi, :, args.settle:].var(unbiased=False)))
                    activity_stats[cond]["membrane_abs_mean"].append(float(
                        out["membrane"][bi, :, args.settle:].abs().mean()))
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
                    if cond.startswith("gate_") or cond.startswith("carrier_"):
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
        if cond.startswith("gate_") or cond.startswith("carrier_"):
            condition_results[cond]["gate_invariants"] = invariants[cond]
        condition_results[cond]["activity_scale"] = {
            "spike_event_rate_mean": float(np.mean(activity_stats[cond]["spike_event_rate"])),
            "membrane_temporal_variance_mean": float(np.mean(activity_stats[cond]["membrane_variance"])),
            "membrane_abs_mean": float(np.mean(activity_stats[cond]["membrane_abs_mean"])),
            "per_image": activity_stats[cond],
        }
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
            "gate_permutation_seeds": {str(seed): permutations[seed].tolist() for seed in range(3)},
            "gate_perm": "local sin(theta) carrier times permuted delayed raw mask; membrane gate uses the same permuted mask",
            "carrier_perm": "permuted sin(theta) carrier times local dynamic delayed mask; membrane gate remains local",
            "gate_mean": "local carrier times per-image/per-region time-mean delayed mask; membrane gate uses the same mean mask",
            "carrier_mean": "per-image/per-region time-mean carrier times local dynamic mask; membrane gate remains local",
            "condition_set": args.condition_set,
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
