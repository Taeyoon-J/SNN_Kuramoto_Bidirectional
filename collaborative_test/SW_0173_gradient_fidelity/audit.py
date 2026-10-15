"""Fixed SW0173 tail/full gradient and virtual-Adam fidelity audit.

Read-only diagnostic: fixed TRAIN cohort/transfer rows, two prescribed scales,
no committed optimizer updates and no validation/reserve reads.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from collaborative_test.SW_0173_gradient_fidelity import gradient_fidelity as gf
from collaborative_test.SW_0173_gradient_fidelity import resource_probe as probe

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = HERE / "results_archive" / "gradient_fidelity_audit_20261015" / "report.json"
COHORT = (0, 1, 2, 3)
TRANSFER = (256, 257, 258, 259)
HORIZONS = {"tail64": 64, "full1024": 1024}
SCALES = (1.0 / 16.0, 1.0)
TRACE_FIELDS = probe.TRACE_FIELDS
RESOURCE_REPORT_SHA256 = "83a721329114d5ed948ce5ab210e15b97d17867f14f8517afdf48be7959d7e52"


def _sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _parameter_key(group, name):
    return f"{group['name']}::{name}"


def fingerprint():
    train = probe._import(probe.TRAIN)
    dependencies = {
        "collaborative_test/SW_0173_gradient_fidelity/DESIGN_KO.md": HERE / "DESIGN_KO.md",
        "collaborative_test/SW_0173_gradient_fidelity/gradient_fidelity.py": HERE / "gradient_fidelity.py",
        "collaborative_test/SW_0173_gradient_fidelity/resource_probe.py": HERE / "resource_probe.py",
        "collaborative_test/SW_0173_gradient_fidelity/resource_queue.py": HERE / "resource_queue.py",
        "collaborative_test/SW_0173_gradient_fidelity/audit.py": HERE / "audit.py",
        "collaborative_test/SW_0173_gradient_fidelity/audit_queue.py": HERE / "audit_queue.py",
        "collaborative_test/SW_0173_gradient_fidelity/test_audit.py": HERE / "test_audit.py",
        "collaborative_test/SW_0171_contextual_instance_graph/train_prefix.py": ROOT / "collaborative_test/SW_0171_contextual_instance_graph/train_prefix.py",
        "collaborative_test/SW_0168_end_to_end_connectivity/connectivity_objective.py": ROOT / "collaborative_test/SW_0168_end_to_end_connectivity/connectivity_objective.py",
        "collaborative_test/SW_0164_foreground_preserving_partition/foreground_signed.py": ROOT / "collaborative_test/SW_0164_foreground_preserving_partition/foreground_signed.py",
        "collaborative_test/SW_0135_native32_spike_binding/evaluation_contract32.py": ROOT / "collaborative_test/SW_0135_native32_spike_binding/evaluation_contract32.py",
        "collaborative_test/SW_0136_native32_transfer/evaluate.py": ROOT / "collaborative_test/SW_0136_native32_transfer/evaluate.py",
    }
    files = {}
    for relative, path in dependencies.items():
        if not path.is_file():
            raise FileNotFoundError(f"SW0173 audit dependency missing: {relative}")
        files[relative] = _sha_file(path)
    return {"files": files, "training_fingerprint": train._training_fingerprint(),
            "resource_fingerprint": probe.fingerprint(),
            "experiment": "SW0173_native_Q_gradient_horizon_fidelity_audit",
            "cohort_positions": list(COHORT), "transfer_positions": list(TRANSFER),
            "horizons": HORIZONS, "scales": list(SCALES), "optimizer_updates": 0,
            "validation_data_read": False, "reserve_data_read": False}


def prepare_cpu():
    """Use existing SW0171 TRAIN inputs; preserve CPU-only admission semantics."""
    return probe.prepare_cpu()


def _require_resource_probe():
    path = probe.OUTPUT
    raw = path.read_bytes()
    digest = _sha_bytes(raw)
    if digest != RESOURCE_REPORT_SHA256:
        raise ValueError("resource receipt SHA differs from frozen SW0173 admission")
    report = json.loads(raw.decode("utf-8"))
    if (report.get("status") != "complete_resource_probe"
            or report.get("optimizer_updates") != 0
            or report.get("validation_data_read") is not False
            or report.get("trace_parity", {}).get("all_trace_fields_exact") is not True
            or report.get("implementation_fingerprint") != probe.fingerprint()):
        raise ValueError("resource receipt did not admit the frozen full1024 source path")
    return {"sha256": digest, "report": report, "path": str(path)}


def _modules(found):
    return {"wrapped": found.wrapped, "encoder": found.encoder}


def _state_snapshot(modules):
    return [(tensor, tensor.detach().clone()) for _, tensor in gf._state_tensors(modules)]


def _restore_state(snapshot):
    with torch.no_grad():
        for tensor, value in snapshot:
            tensor.copy_(value)


def _set_rng(state):
    gf._rng_restore(state)


def fixed_route_loss(q, plan):
    """Evaluate the SW0168 pair-weighted loss on immutable selected forest edges."""
    from collaborative_test.SW_0168_end_to_end_connectivity import connectivity_objective as obj
    terms = []
    if plan["positive_pair_mass"]:
        values, weights = obj._edge_values(q, plan["positive_edges"])
        terms.append((weights * (1.0 - values).square()).sum() / plan["positive_pair_mass"])
    if plan["negative_pair_mass"]:
        values, weights = obj._edge_values(q, plan["negative_edges"])
        terms.append((weights * values.square()).sum() / plan["negative_pair_mass"])
    return q.sum() * 0.0 if not terms else torch.stack(terms).mean()


def _loss(q, target, support, objective):
    return objective.constrained_maximin_loss(q,
        torch.as_tensor(target, device=q.device),
        torch.as_tensor(np.asarray(support) > 0, device=q.device))


def _single_train_metrics(contract, pred, gt):
    from snn_kuramoto_bidirectional.evaluation import evaluate_patch_masks
    result = evaluate_patch_masks(torch.as_tensor(np.asarray(pred, dtype=np.int64))[None],
                                 torch.as_tensor(np.asarray(gt, dtype=np.int64))[None])
    return {name: float(result["per_image"][name][0]) for name in contract.METRICS}


def _readouts(q, a, trace, arrays, position, gain, solver, contract, sw136):
    qcc = np.asarray(sw136._qcc_labels(trace)[0], dtype=np.int32)
    supports = {"source_F": np.asarray(arrays["source_support"][position], dtype=np.int32),
                "native_QCC": qcc}
    outputs, metrics, merges = {}, {}, {}
    q_np = q.detach().cpu().numpy().astype(np.float64, copy=True)
    a_np = a.detach().cpu().numpy().astype(np.float64, copy=True)
    gt = arrays["modal_gt"][position]
    for name, support in supports.items():
        result = solver.foreground_induced_merge(q_np, a_np, float(gain), support,
                                                grid_shape=(32, 32))
        labels = np.asarray(result["labels"], dtype=np.int32)
        if labels.shape != (32, 32) or not np.array_equal(labels > 0, support > 0):
            raise ValueError(f"{name} readout changed its frozen TRAIN foreground support")
        outputs[name] = labels
        metrics[name] = _single_train_metrics(contract, labels, gt)
        merges[name] = int(result["merge_count"])
    return outputs, metrics, merges, qcc


def _event_baseline(trace, threshold):
    gates = trace["component_gates"].detach() > 0
    events = trace["component_spikes"].detach() > 0
    membrane = trace["component_membrane"].detach()
    margin = membrane - threshold
    count = int(gates.sum().item())
    return {"active_gate_samples": count,
            "actual_events_on_active_gates": int((events & gates).sum().item()),
            "event_rate_on_active_gates": float((events & gates).sum().item() / count) if count else 0.0,
            "actual_membrane_quantiles_on_active_gates": torch.quantile(
                membrane[gates].float(), torch.tensor([0., .25, .5, .75, 1.], device=membrane.device)
            ).detach().cpu().tolist() if count else [],
            "margin_quantiles_on_active_gates": torch.quantile(
                margin[gates].float(), torch.tensor([0., .25, .5, .75, 1.], device=margin.device)
            ).detach().cpu().tolist() if count else []}


def _event_change(base, trial, threshold0, threshold1):
    gate0 = base["component_gates"].detach().to(trial["component_gates"].device) > 0
    gate1 = trial["component_gates"].detach() > 0
    common = gate0 & gate1
    events0 = base["component_spikes"].detach().to(trial["component_spikes"].device) > 0
    events1 = trial["component_spikes"].detach() > 0
    margin0 = base["component_membrane"].detach().to(trial["component_membrane"].device) - threshold0
    margin1 = trial["component_membrane"].detach() - threshold1
    count = int(common.sum().item())
    return {"common_active_samples": count,
            "event_flips_on_common_active": int(((events0 != events1) & common).sum().item()),
            "threshold_crossings_on_common_active": int(
                (((margin0 >= 0) != (margin1 >= 0)) & common).sum().item()),
            "source_margin_quantiles": torch.quantile(margin0[common].float(),
                torch.tensor([0., .25, .5, .75, 1.], device=margin0.device)).cpu().tolist() if count else [],
            "trial_margin_quantiles": torch.quantile(margin1[common].float(),
                torch.tensor([0., .25, .5, .75, 1.], device=margin1.device)).cpu().tolist() if count else []}


def _component_rho(trace):
    """Same B1 component loop, centering/norm epsilon/clamp as production Q."""
    parts = []
    for component in range(4):
        values = trace["component_spikes"][:, component].float()[..., 512:]
        values = values - values.mean(dim=-1, keepdim=True)
        values = values / values.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        parts.append((values @ values.transpose(-1, -2)).clamp(-1.0, 1.0))
    return torch.stack(parts, dim=1)


def clamp_blocking_diagnostics(q, trace, q_loss, source_plan):
    """Measure exact component Jacobians on source-positive route edges."""
    rho = _component_rho(trace)
    q_product = rho.clamp_min(0.0).prod(dim=1)
    q_batch = q.unsqueeze(0) if q.ndim == 2 else q
    if not torch.equal(q_product, q_batch):
        raise ValueError("same-mode component rho reconstruction differs from production Q")
    dloss_dq = torch.autograd.grad(q_loss, q, retain_graph=True)[0]
    edges = source_plan["positive_edges"]
    if not edges:
        return {"positive_route_edges": 0, "all_component_partials_zero_edges": 0,
                "blocked_edge_fraction": 0.0, "blocked_positive_pair_mass_fraction": 0.0,
                "blocked_positive_loss_gradient_mass_fraction": 0.0}
    ii = torch.as_tensor([edge[0] for edge in edges], dtype=torch.long, device=q.device)
    jj = torch.as_tensor([edge[1] for edge in edges], dtype=torch.long, device=q.device)
    node_count = int(q.shape[-1])
    edge_rho = rho[0].reshape(4, -1)[:, ii * node_count + jj]
    if edge_rho.shape != (4, len(edges)):
        raise ValueError("component affinity edge tensor has unexpected shape")
    edge_products = edge_rho.clamp_min(0.0).prod(dim=0)
    if not torch.equal(edge_products, q[ii, jj]):
        raise ValueError("edge-wise component product differs from actual Q")
    # Each indexed edge has independent four-component inputs. Autograd on the
    # summed edge products returns one exact partial vector per edge.
    partials = torch.autograd.grad(edge_products.sum(), edge_rho, retain_graph=True)[0]
    blocked_flags = (~torch.any(partials != 0, dim=0)).detach().cpu().tolist()
    total_mass = sum(int(edge[2]) for edge in edges)
    blocked_mass = 0
    blocked_grad, all_grad = 0.0, 0.0
    for index, (i, j, weight) in enumerate(edges):
        contribution = abs(float(dloss_dq[i, j].detach().double().cpu()))
        all_grad += contribution
        if blocked_flags[index]:
            blocked_mass += int(weight)
            blocked_grad += contribution
    nblocked = sum(bool(value) for value in blocked_flags)
    return {"positive_route_edges": len(edges), "all_component_partials_zero_edges": nblocked,
            "blocked_edge_fraction": nblocked / len(edges),
            "blocked_positive_pair_mass": blocked_mass, "positive_pair_mass": total_mass,
            "blocked_positive_pair_mass_fraction": blocked_mass / total_mass if total_mass else 0.0,
            "blocked_positive_loss_gradient_mass_fraction": blocked_grad / all_grad if all_grad else 0.0}


def _trace_hashes(trace):
    return {name: probe._tensor_sha(trace[name]) for name in TRACE_FIELDS}


def _forward(train, found, arrays, position, device, horizon):
    rgb = torch.from_numpy(np.ascontiguousarray(arrays["rgb"][position:position+1])).to(device)
    return train._native_forward(found, rgb, live_tail_steps=HORIZONS[horizon])


def _clone_forward(value):
    gamma, trace, a, z, q = value
    return {"gamma": gamma.detach().cpu().clone(), "a": a.detach().cpu().clone(),
            "z": z.detach().cpu().clone(), "q": q.detach().cpu().clone(),
            "trace": {name: trace[name].detach().cpu().contiguous().clone() for name in TRACE_FIELDS}}


def _virtual_delta(found, gradients):
    train = probe._import(probe.TRAIN)
    modules = _modules(found)
    before_state = _state_snapshot(modules)
    before_sha = gf.state_sha256(modules)
    rng = gf._rng_snapshot()
    optimizer, _families = train._make_optimizer(found)
    params_before = {id(p): p.detach().clone()
                     for group in optimizer.param_groups for p in group["params"]}
    try:
        for group in optimizer.param_groups:
            for name, parameter in zip(group["parameter_names"], group["params"]):
                gradient = gradients.get(_parameter_key(group, name))
                parameter.grad = None if gradient is None else gradient.detach().clone()
        torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group["params"]], 1.0)
        optimizer.step()
        found.wrapped.project_integrations_()
        delta = {}
        for group in optimizer.param_groups:
            for name, parameter in zip(group["parameter_names"], group["params"]):
                delta[_parameter_key(group, name)] = parameter.detach().clone() - params_before[id(parameter)]
        return delta, before_sha
    finally:
        optimizer.zero_grad(set_to_none=True)
        _restore_state(before_state)
        gf._rng_restore(rng)
        if gf.state_sha256(modules) != before_sha:
            raise RuntimeError("virtual Adam direction did not restore exact model/buffer state")


def _decision_checks(parity, trial_results):
    small = SCALES[0]
    transfer_comparisons = []
    for scale in SCALES:
        full_transfer = next(row for row in trial_results["full1024"]["transfer"]
                             if row["scale"] == scale)
        tail_transfer = next(row for row in trial_results["tail64"]["transfer"]
                             if row["scale"] == scale)
        transfer_comparisons.append({
            "scale": scale,
            "full_native_more_negative_than_tail": (
                full_transfer["actual_native_change"] < tail_transfer["actual_native_change"]),
            "full_fixed_more_negative_than_tail": (
                full_transfer["actual_fixed_route_change"] < tail_transfer["actual_fixed_route_change"]),
        })
    full_requirements = []
    for scale in SCALES:
        for population in ("cohort", "transfer"):
            row = next(value for value in trial_results["full1024"][population]
                       if value["scale"] == scale)
            full_requirements.append({
                "scale": scale, "population": population,
                "predicted_direction_negative": row["predicted_directional_change"] < 0,
                "native_route_loss_decreased": row["actual_native_change"] < 0,
                "fixed_source_route_loss_decreased": row["actual_fixed_route_change"] < 0})
    return {
        "same_rng_repeat_jitter_zero": all(
            all(value for value in row["repeat_full"].values()) for row in parity),
        "tail_full_exact_source_parity": all(
            row["tail_full"]["all_trace_fields_exact"]
            and row["tail_full"]["q_exact"] and row["tail_full"]["gamma_exact"]
            and row["tail_full"]["actual_A_exact"] and row["tail_full"]["projection_exact"]
            for row in parity),
        "full1024_direction_checks": full_requirements,
        "all_full1024_direction_checks_pass": all(
            row["predicted_direction_negative"] and row["native_route_loss_decreased"]
            and row["fixed_source_route_loss_decreased"] for row in full_requirements),
        "full_transfer_comparisons_by_scale": transfer_comparisons,
        "full_transfer_more_negative_than_tail_at_both_scales": all(
            row["full_native_more_negative_than_tail"] for row in transfer_comparisons),
    }


def _cross_directional_dot_matrix(optimizer, gradients, deltas, horizons=tuple(HORIZONS)):
    matrix = {}
    for grad_horizon in horizons:
        matrix[grad_horizon] = {}
        for delta_horizon in horizons:
            value = 0.0
            for group in optimizer.param_groups:
                for name, parameter in zip(group["parameter_names"], group["params"]):
                    gradient = gradients[grad_horizon]["by_id"][id(parameter)]
                    if gradient is not None:
                        value += float((gradient.double() *
                            deltas[delta_horizon][_parameter_key(group, name)].double()).sum().cpu())
            matrix[grad_horizon][delta_horizon] = value
    return matrix


def _evaluate_trial_matrix(*, train, found, arrays, device, deltas, gradients,
                           image_rng, source_plans, source_events, gain,
                           objective, solver, contract, sw136, source_baseline,
                           horizons=HORIZONS, scales=SCALES, populations=None):
    """Run every fixed horizon × scale × TRAIN population and aggregate it."""
    if populations is None:
        populations = {"cohort": COHORT, "transfer": TRANSFER}
    results = {horizon: {population: [] for population in populations}
               for horizon in horizons}
    optimizer, _families = train._make_optimizer(found)
    for horizon in horizons:
        for scale in scales:
            for population_name, positions in populations.items():
                rows = _trial_set(train, found, arrays, positions, device,
                    deltas[horizon], scale, gradients[horizon]["by_name"], image_rng,
                    source_plans, source_events, gain, objective, solver, contract, sw136)
                base_rows = source_baseline[population_name]
                native_base = float(np.mean([row["native_loss"] for row in base_rows]))
                fixed_base = float(np.mean([row["fixed_route_loss"] for row in base_rows]))
                native_trial = float(np.mean([row["native_loss_recomputed_route"] for row in rows]))
                fixed_trial = float(np.mean([row["fixed_source_route_loss"] for row in rows]))
                predicted_dot = 0.0
                for group in optimizer.param_groups:
                    for parameter_name, parameter in zip(group["parameter_names"], group["params"]):
                        grad = gradients[horizon]["by_id"][id(parameter)]
                        if grad is not None:
                            predicted_dot += float((grad.double() *
                                deltas[horizon][_parameter_key(group, parameter_name)].double()).sum().cpu())
                results[horizon][population_name].append({
                    "scale": float(scale), "images": rows,
                    "source_native_mean_loss": native_base,
                    "trial_native_mean_loss": native_trial,
                    "actual_native_change": native_trial - native_base,
                    "source_fixed_route_mean_loss": fixed_base,
                    "trial_fixed_route_mean_loss": fixed_trial,
                    "actual_fixed_route_change": fixed_trial - fixed_base,
                    "predicted_directional_change": float(scale * predicted_dot),
                    "gradient_dot_projected_adam_delta": predicted_dot})
    return results


def _trial_set(train, found, arrays, positions, device, delta, scale, gradients,
               per_image_rng, source_plans, source_events, gain, objective, solver, contract, sw136):
    modules = _modules(found)
    snapshot = _state_snapshot(modules)
    before_sha = gf.state_sha256(modules)
    rng = gf._rng_snapshot()
    try:
        optimizer, _ = train._make_optimizer(found)
        param_delta = {}
        for group in optimizer.param_groups:
            for name, parameter in zip(group["parameter_names"], group["params"]):
                param_delta[id(parameter)] = delta[_parameter_key(group, name)]
        with torch.no_grad():
            for parameter in [p for group in optimizer.param_groups for p in group["params"]]:
                parameter.add_(param_delta[id(parameter)], alpha=float(scale))
        # Do not re-project here: delta is already the projected Adam segment,
        # and interpolation s in [0,1] preserves physical bounds.
        for name in ("a_d", "a_m"):
            value = getattr(found.wrapped, name)
            if bool(torch.any(value < 0)) or bool(torch.any(value > 1)):
                raise ValueError(f"virtual projected segment left physical {name} range")
        rows = []
        for position in positions:
            gf._rng_restore(per_image_rng[position])
            gamma, trace, a, z, q = _forward(train, found, arrays, position, device, "full1024")
            q_image = q[0]
            qloss = _loss(q_image, arrays["feasible_targets"][position],
                          arrays["source_support"][position], objective)
            fixed = fixed_route_loss(q_image, source_plans[position])
            graph_gain = float(torch.exp(
                found.wrapped.core.graph_generator.log_coupling_gain.detach()).cpu())
            labels, metrics, merge_count, qcc = _readouts(q_image, a[0], trace, arrays,
                position, gain, solver, contract, sw136)
            threshold = 0.06 * torch.exp(found.wrapped.b.detach().reshape(1, 4, 1, 1))
            event = _event_change(source_events[position]["trace"], trace,
                                  source_events[position]["threshold"].to(device), threshold)
            clamp = clamp_blocking_diagnostics(q_image, trace, qloss, source_plans[position])
            rows.append({"position": int(position), "image_id": int(arrays["image_ids"][position]),
                "native_loss_recomputed_route": float(qloss.detach().cpu()),
                "fixed_source_route_loss": float(fixed.detach().cpu()),
                "actual_graph_gain": graph_gain,
                "readout_gain_bound_to_source": float(gain),
                "metrics": metrics, "merge_counts": merge_count,
                "qcc_sha256": probe._tensor_sha(torch.from_numpy(qcc)),
                "labels_sha256": {name: probe._tensor_sha(torch.from_numpy(value))
                                  for name, value in labels.items()},
                "event_diagnostics_vs_source": event, "clamp_blocking": clamp})
            del gamma, trace, a, z, q, q_image, qloss, fixed
        return rows
    finally:
        _restore_state(snapshot)
        gf._rng_restore(rng)
        if gf.state_sha256(modules) != before_sha:
            raise RuntimeError("full1024 finite-step trial failed state restoration")


def run_audit(*, prepared=None, device="cuda:0", output=OUTPUT):
    """Run fixed B4 tail/full gradients and full1024 projected-step trials."""
    started = time.perf_counter()
    prepared = prepare_cpu() if prepared is None else prepared
    resource = _require_resource_probe()
    implementation = fingerprint()
    if torch.cuda.is_initialized():
        raise RuntimeError("audit worker must perform CPU admission before CUDA initialization")
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0173 audit requires an assigned exclusive CUDA device")
    torch.cuda.set_device(dev)
    train = probe._import(probe.TRAIN)
    objective = probe._import("collaborative_test.SW_0168_end_to_end_connectivity.connectivity_objective")
    solver = probe._import("collaborative_test.SW_0164_foreground_preserving_partition.foreground_signed")
    contract = probe._import("collaborative_test.SW_0135_native32_spike_binding.evaluation_contract32")
    sw136 = probe._import("collaborative_test.SW_0136_native32_transfer.evaluate")
    arrays = prepared["admitted"]["prepared"]["arrays"]
    if prepared.get("source_proof") != resource["report"].get("source_proof"):
        raise ValueError("resource probe and audit input source proof differ")
    model = probe.build_source(prepared, device=dev)
    found, optimizer = model["found"], model["optimizer"]
    modules = _modules(found)
    optimizer_groups = optimizer.param_groups
    module_groups = gf.parameter_union(modules)
    if len({id(group.parameter) for group in module_groups}) != sum(
            len(group["params"]) for group in optimizer_groups):
        raise ValueError("alias-aware trainable parameter union differs from SW0171 optimizer families")
    state_before = gf.state_sha256(modules)
    state_snapshot = _state_snapshot(modules)
    source_digest_before = train._source_sha(found, model["base_encoder"])
    rng_initial = gf._rng_snapshot()
    gain = float(prepared["admitted"]["prepared"]["producer_bindings"]["source_gain"])
    actual_gain = float(torch.exp(found.wrapped.core.graph_generator.log_coupling_gain.detach()).cpu())
    if not math.isfinite(gain) or gain <= 0 or abs(gain - actual_gain) > 2e-6:
        raise ValueError("loaded source gain differs from immutable TRAIN cache binding")
    report = {"experiment": implementation["experiment"], "status": "running",
        "implementation_fingerprint": implementation,
        "resource_probe_sha256": resource["sha256"], "source_proof": prepared["source_proof"],
        "source_model_digest_before_sha256": source_digest_before,
        "model_state_before_sha256": state_before,
        "cohort_positions": list(COHORT), "transfer_positions": list(TRANSFER),
        "source_gain": gain, "scales": list(SCALES),
        "gradient_batch": "fixed TRAIN B4 virtual audit, microbatch 1; not B16 training",
        "optimizer_recipe": "fresh Adam, SW0171 group learning rates, global clip 1, integration projection after virtual step",
        "optimizer_updates": 0, "validation_data_read": False, "reserve_data_read": False,
        "ground_truth_scope": "canonical TRAIN rows only",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        positions = (*COHORT, *TRANSFER)
        image_rng = {position: gf._rng_snapshot() for position in positions}
        parity, source_baseline, source_plans, source_events = [], {"cohort": [], "transfer": []}, {}, {}

        # Exact same-RNG tail/full/readout and repeated-source checks. Only the
        # event tensors needed for later finite-step comparisons are retained.
        for position in positions:
            _set_rng(image_rng[position])
            tail = _forward(train, found, arrays, position, dev, "tail64")
            tail_trace = {name: tail[1][name].detach().cpu().contiguous().clone() for name in TRACE_FIELDS}
            tail_q, tail_gamma, tail_a, tail_z = (tail[4].detach().cpu().clone(),
                tail[0].detach().cpu().clone(), tail[2].detach().cpu().clone(), tail[3].detach().cpu().clone())
            del tail
            _set_rng(image_rng[position])
            full = _forward(train, found, arrays, position, dev, "full1024")
            full_trace = {name: full[1][name].detach().cpu().contiguous().clone() for name in TRACE_FIELDS}
            full_q, full_gamma, full_a, full_z = (full[4].detach().cpu().clone(),
                full[0].detach().cpu().clone(), full[2].detach().cpu().clone(), full[3].detach().cpu().clone())
            checks = {"trace_fields_exact": {name: torch.equal(tail_trace[name], full_trace[name])
                                               for name in TRACE_FIELDS},
                      "q_exact": torch.equal(tail_q, full_q),
                      "gamma_exact": torch.equal(tail_gamma, full_gamma),
                      "actual_A_exact": torch.equal(tail_a, full_a),
                      "projection_exact": torch.equal(tail_z, full_z)}
            checks["all_trace_fields_exact"] = all(checks["trace_fields_exact"].values())
            if not checks["all_trace_fields_exact"] or not all(checks[key] for key in
                    ("q_exact", "gamma_exact", "actual_A_exact", "projection_exact")):
                raise ValueError(f"tail/full parity failed at fixed TRAIN position {position}")

            q = full[4][0]
            target, support = arrays["feasible_targets"][position], arrays["source_support"][position]
            qloss = _loss(q, target, support, objective)
            plan = objective.forest_routes(q.detach(), target, np.asarray(support) > 0)
            fixed = fixed_route_loss(q, plan)
            if not torch.equal(qloss.detach(), fixed.detach()):
                raise ValueError("native and fixed route losses differ on source baseline")
            labels, metrics, merges, qcc = _readouts(q, full[2][0], full[1], arrays,
                position, gain, solver, contract, sw136)
            threshold = 0.06 * torch.exp(found.wrapped.b.detach().reshape(1, 4, 1, 1))
            clamp = clamp_blocking_diagnostics(q, full[1], qloss, plan)
            source_plans[position] = plan
            source_events[position] = {"trace": {name: full[1][name].detach().cpu().contiguous().clone()
                for name in ("component_gates", "component_spikes", "component_membrane")},
                "threshold": threshold.detach().cpu().clone()}
            set_name = "cohort" if position in COHORT else "transfer"
            source_baseline[set_name].append({"position": int(position),
                "image_id": int(arrays["image_ids"][position]),
                "native_loss": float(qloss.detach().cpu()), "fixed_route_loss": float(fixed.detach().cpu()),
                "metrics": metrics, "merge_counts": merges,
                "qcc_labels_sha256": probe._tensor_sha(torch.from_numpy(qcc)),
                "readout_labels_sha256": {name: probe._tensor_sha(torch.from_numpy(value))
                                           for name, value in labels.items()},
                "event_diagnostics": _event_baseline(full[1], threshold),
                "clamp_blocking": clamp,
                "route_positive_pair_mass": int(plan["positive_pair_mass"]),
                "route_negative_pair_mass": int(plan["negative_pair_mass"]),
                "route_positive_edge_count": len(plan["positive_edges"])})
            full_trace_sha = {name: probe._tensor_sha(torch.from_numpy(full_trace[name].numpy()))
                              for name in TRACE_FIELDS}
            del full, q, qloss, fixed, labels, metrics, merges, qcc, clamp

            # Hold only CPU snapshots while repeating the native full rollout.
            _set_rng(image_rng[position])
            repeat = _forward(train, found, arrays, position, dev, "full1024")
            repeat_trace = {name: repeat[1][name].detach().cpu().contiguous().clone()
                            for name in TRACE_FIELDS}
            jitter = {"trace_fields_exact": all(torch.equal(full_trace[name], repeat_trace[name])
                                                 for name in TRACE_FIELDS),
                      "q_exact": torch.equal(full_q, repeat[4].detach().cpu()),
                      "gamma_exact": torch.equal(full_gamma, repeat[0].detach().cpu()),
                      "actual_A_exact": torch.equal(full_a, repeat[2].detach().cpu()),
                      "projection_exact": torch.equal(full_z, repeat[3].detach().cpu())}
            del repeat
            if not all(jitter.values()):
                raise ValueError(f"same-state repeat jitter is nonzero at TRAIN position {position}")
            parity.append({"position": int(position), "image_id": int(arrays["image_ids"][position]),
                           "tail_full": checks, "repeat_full": jitter,
                           "full_trace_sha256": full_trace_sha})

        # B4 / micro1 image-mean gradients for each truncation horizon.
        gradients, family_norms, named_norms = {}, {}, {}
        for horizon, live_steps in HORIZONS.items():
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for position in COHORT:
                _set_rng(image_rng[position])
                gamma, trace, actual_a, z, q = _forward(train, found, arrays, position, dev, horizon)
                value = _loss(q[0], arrays["feasible_targets"][position],
                              arrays["source_support"][position], objective)
                if not bool(torch.isfinite(value)):
                    raise FloatingPointError(f"nonfinite {horizon} Q loss at TRAIN position {position}")
                (value / len(COHORT)).backward()
                losses.append(float(value.detach().cpu()))
                del gamma, trace, actual_a, z, q, value
            by_name, by_id, family, named = {}, {}, {}, {}
            for group in optimizer.param_groups:
                sq = 0.0
                rows = []
                for name, parameter in zip(group["parameter_names"], group["params"]):
                    grad = None if parameter.grad is None else parameter.grad.detach().clone()
                    key = _parameter_key(group, name)
                    by_name[key], by_id[id(parameter)] = grad, grad
                    norm = 0.0 if grad is None else float(grad.double().norm().cpu())
                    if not math.isfinite(norm):
                        raise FloatingPointError(f"nonfinite gradient {key}")
                    sq += norm * norm
                    rows.append({"name": name, "gradient_l2": norm,
                                 "gradient_present": grad is not None})
                family[group["name"]] = math.sqrt(sq)
                named[group["name"]] = rows
            gradients[horizon] = {"by_name": by_name, "by_id": by_id}
            family_norms[horizon], named_norms[horizon] = family, named
            optimizer.zero_grad(set_to_none=True)

        family_cosines = {}
        named_parameter_cosines = {}
        total_dot = total_tail_sq = total_full_sq = 0.0
        for group in optimizer.param_groups:
            dot = tail_sq = full_sq = 0.0
            for parameter in group["params"]:
                left = gradients["tail64"]["by_id"][id(parameter)]
                right = gradients["full1024"]["by_id"][id(parameter)]
                if left is None or right is None:
                    continue
                dot += float((left.double() * right.double()).sum().cpu())
                tail_sq += float(left.double().square().sum().cpu())
                full_sq += float(right.double().square().sum().cpu())
            family_cosines[group["name"]] = None if not tail_sq * full_sq else dot / math.sqrt(tail_sq * full_sq)
            total_dot += dot; total_tail_sq += tail_sq; total_full_sq += full_sq
            for name, parameter in zip(group["parameter_names"], group["params"]):
                left = gradients["tail64"]["by_id"][id(parameter)]
                right = gradients["full1024"]["by_id"][id(parameter)]
                if left is None or right is None:
                    named_parameter_cosines[_parameter_key(group, name)] = None
                    continue
                denom = float(left.double().norm().cpu()) * float(right.double().norm().cpu())
                named_parameter_cosines[_parameter_key(group, name)] = (
                    float((left.double() * right.double()).sum().cpu()) / denom if denom else None)
        union_cosine = None if not total_tail_sq * total_full_sq else total_dot / math.sqrt(total_tail_sq * total_full_sq)

        deltas, delta_hashes = {}, {}
        for horizon in HORIZONS:
            delta, before_sha = _virtual_delta(found, gradients[horizon]["by_name"])
            if before_sha != state_before:
                raise ValueError("virtual Adam did not start from exact original state")
            deltas[horizon] = delta
            delta_hashes[horizon] = {name: probe._tensor_sha(value) for name, value in delta.items()}
        cross_dots = _cross_directional_dot_matrix(optimizer, gradients, deltas)

        trial_results = _evaluate_trial_matrix(train=train, found=found, arrays=arrays,
            device=dev, deltas=deltas, gradients=gradients, image_rng=image_rng,
            source_plans=source_plans, source_events=source_events, gain=gain,
            objective=objective, solver=solver, contract=contract, sw136=sw136,
            source_baseline=source_baseline)

        decision = _decision_checks(parity, trial_results)

        source_digest_after = train._source_sha(found, model["base_encoder"])
        state_after = gf.state_sha256(modules)
        report.update({"status": "complete_gradient_audit",
            "source_model_digest_after_sha256": source_digest_after,
            "model_state_after_sha256": state_after,
            "source_model_unchanged": source_digest_before == source_digest_after,
            "model_state_restored": state_before == state_after,
            "horizon_family_gradient_norms": family_norms,
            "horizon_named_parameter_gradient_norms": named_norms,
            "horizon_family_gradient_cosines": family_cosines,
            "horizon_named_parameter_gradient_cosines": named_parameter_cosines,
            "gradient_union_cosine": union_cosine,
            "cross_gradient_projected_adam_delta_dot_matrix": cross_dots,
            "virtual_adam_delta_sha256": delta_hashes,
            "same_rng_parity_rows": parity, "source_baseline": source_baseline,
            "trials": trial_results, "preregistered_decision_checks": decision,
            "elapsed_seconds": time.perf_counter() - started})
        if source_digest_before != source_digest_after or state_before != state_after:
            raise RuntimeError("gradient audit changed source/model state")
    finally:
        optimizer.zero_grad(set_to_none=True)
        _restore_state(state_snapshot)
        gf._rng_restore(rng_initial)
        if gf.state_sha256(modules) != state_before:
            raise RuntimeError("audit cleanup did not restore exact source/model state")
    train.preflight._write_report(Path(output), report)
    return report

