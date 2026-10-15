"""SW0170 TRAIN8 source-bound Stage1 admission (no validation/score path)."""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from collaborative_test.SW_0169_normalized_hard_reset import calibration_v2 as calibration
from collaborative_test.SW_0169_normalized_hard_reset import hard_reset
from collaborative_test.SW_0170_causal_synaptic_readout import causal_synapse as synapse

EXPERIMENT = "SW0170_causal_synaptic_readout"
REGISTERED_R = (2.8129515647888184, 2.6795597076416016,
                2.7440667152404785, 2.076063871383667)
IDS = 8
STEPS, SETTLE, TAIL = 1024, 512, 64
ARMS = ("identity_synapse", "causal_synapse")
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_tensor(value):
    return _sha_bytes(value.detach().cpu().contiguous().numpy().tobytes())


def _import(name):
    return calibration._import_keep_cuda(name)


def implementation_fingerprint():
    paths = (
        "collaborative_test/SW_0170_causal_synaptic_readout/DESIGN_KO.md",
        "collaborative_test/SW_0170_causal_synaptic_readout/causal_synapse.py",
        "collaborative_test/SW_0170_causal_synaptic_readout/preflight.py",
        "collaborative_test/SW_0169_normalized_hard_reset/hard_reset.py",
        "collaborative_test/SW_0169_normalized_hard_reset/calibration_v2.py",
        "collaborative_test/SW_0169_normalized_hard_reset/preflight_v3.py",
        "collaborative_test/SW_0168_end_to_end_connectivity/connectivity_objective.py",
        "collaborative_test/SW_0155_supervised_spike_relations/train.py",
        "collaborative_test/SW_0155_supervised_spike_relations/objective.py",
        "collaborative_test/SW_0167_structured_relation_energy/audit_train_cache.py",
        "collaborative_test/SW_0167_structured_relation_energy/qcc_readout.py",
        "snn_kuramoto_bidirectional/spike_classifier.py",
    )
    files = {}
    for relative in paths:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"required SW0170 dependency missing: {relative}")
        files[relative] = _sha_bytes(path.read_bytes())
    return {"files": files,
            "calibration_dependencies": calibration.implementation_fingerprint(),
            "registered_r": list(REGISTERED_R),
            "decay": synapse.DECAY, "live_start": synapse.LIVE_START,
            "settle": SETTLE, "steps": STEPS}


def prepare_inputs_cpu():
    """Validate only the fixed first-eight TRAIN source/cache/RGB inputs.

    TRAIN masks are intentionally loaded later, after GPU source parity has
    completed, by the frozen SW0169 fresh-CPU target handoff.
    """
    prepared = calibration.prepare_calibration_cpu()
    if (prepared.get("selected_ids") is None or len(prepared["selected_ids"]) != IDS
            or prepared.get("positions") != list(range(IDS))):
        raise ValueError("SW0170 requires the fixed ordered first eight TRAIN rows")
    if prepared.get("ground_truth_read") is not False or prepared.get("validation_data_read") is not False:
        raise ValueError("SW0170 CPU preparation exceeded source-only scope")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU preparation unexpectedly initialized CUDA")
    prepared["source_proof"] = {
        "seed": 1, "selected_ids": list(map(int, prepared["selected_ids"])),
        "positions": list(map(int, prepared["positions"])),
        "source_checkpoint_sha256": prepared["source_checkpoint_sha256"],
        "source_manifest_sha256": prepared["source_manifest_sha256"],
        "source_cache_manifest_sha256": prepared["source_cache_manifest_sha256"],
        "source_cache_bindings_sha256": prepared["source_cache_bindings_sha256"],
        "source_cache_state_sha256": prepared["source_cache_state_sha256"],
        "context_cache_manifest_sha256": prepared["context_cache_manifest_sha256"],
        "context_cache_bindings_sha256": prepared["context_cache_bindings_sha256"],
        "context_cache_state_sha256": prepared["context_cache_state_sha256"],
        "rgb_sha256": list(prepared["rgb_sha256"]),
        "train_rgb_manifest_sha256": prepared["train_rgb_manifest_sha256"],
        "train_hdf5_path": prepared["train_hdf5_path"],
        "train_hdf5_size_bytes": int(prepared["train_hdf5_size_bytes"]),
        "train_hdf5_mtime_ns": int(prepared["train_hdf5_mtime_ns"]),
        "implementation_fingerprint": prepared["implementation_fingerprint"],
        "ground_truth_read": False, "validation_data_read": False,
    }
    return prepared


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _norm(tensors):
    values = [value for value in tensors if value is not None]
    if not values:
        return 0.0
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise FloatingPointError("nonfinite SW0170 gradient")
    return math.sqrt(sum(float(value.detach().double().square().sum().cpu()) for value in values))


def _weighted_route_means(q, plan):
    result = {}
    for kind in ("positive", "negative"):
        edges = plan[f"{kind}_edges"]
        mass = int(plan[f"{kind}_pair_mass"])
        result[f"{kind}_weighted_edges"] = [
            {"i": int(a), "j": int(b), "pair_mass": int(weight),
             "q": float(q[a, b].detach().cpu())}
            for a, b, weight in edges]
        if mass <= 0:
            result[kind] = None
        else:
            numerator = sum(float(q[a, b].detach().cpu()) * int(weight)
                            for a, b, weight in edges)
            result[kind] = numerator / mass
    if result["positive"] is None or result["negative"] is None:
        result["separation"] = None
    else:
        result["separation"] = result["positive"] - result["negative"]
    result["positive_pair_mass"] = int(plan["positive_pair_mass"])
    result["negative_pair_mass"] = int(plan["negative_pair_mass"])
    return result


def _gate_failures(rows, identity_loss, filtered_loss, identity_sep, filtered_sep,
                   filtered_foreground_images):
    failures = []
    if not math.isfinite(identity_loss) or not math.isfinite(filtered_loss):
        failures.append("nonfinite_loss")
    elif not filtered_loss < identity_loss:
        failures.append("filtered_maximin_loss_not_lower")
    eligible_identity = [i for i, row in enumerate(identity_sep) if row["separation"] is not None]
    eligible_filtered = [i for i, row in enumerate(filtered_sep) if row["separation"] is not None]
    if not eligible_identity or eligible_identity != eligible_filtered:
        failures.append("no_fixed_eligible_forest_rows")
    elif not float(np.mean([filtered_sep[i]["separation"] for i in eligible_filtered])) > float(
            np.mean([identity_sep[i]["separation"] for i in eligible_identity])):
        failures.append("filtered_forest_separation_not_higher")
    if len(filtered_foreground_images) < 4:
        failures.append("filtered_qcc_foreground_below_four_of_eight")
    for row in rows:
        if not row.get("finite") or not row.get("nonconstant_offdiagonal"):
            failures.append(f"degenerate_affinity_row_{row['index']}")
        if row.get("filtered_activity_variance", 0.0) <= 0:
            failures.append(f"degenerate_filtered_activity_row_{row['index']}")
    return failures


def _cpu_filtered_qcc(filtered_components):
    components = np.asarray(filtered_components)
    if (components.dtype != np.float32 or components.ndim != 4
            or components.shape[1:] != (4, 1024, SETTLE)
            or components.shape[0] != IDS):
        raise ValueError("SW0170 CPU QCC requires ordered TRAIN8 [8,4,1024,512] traces")
    audit = _import("collaborative_test.SW_0167_structured_relation_energy.audit_train_cache")
    qcc = _import("collaborative_test.SW_0167_structured_relation_energy.qcc_readout")
    q_rows, labels_rows = [], []
    for start in (0, 4):
        # The immutable source-cache CPU-Q implementation is defined for its
        # original B4 producer grouping; preserve that grouping explicitly.
        q_batch = audit._recompute_cpu_q(np.ascontiguousarray(components[start:start + 4]))
        if q_batch.dtype != np.float32 or q_batch.shape != (4, 1024, 1024):
            raise ValueError("production CPU Q helper changed its registered B4 output schema")
        q_rows.append(q_batch)
        label_result = qcc.qcc_labels_from_affinity(q_batch)
        if torch.is_tensor(label_result):
            label_result = label_result.detach().cpu().numpy()
        label_batch = np.asarray(label_result, dtype=np.int32)
        if label_batch.shape != (4, 32, 32):
            raise ValueError("production CPU QCC must return one native32 B4 batch")
        labels_rows.append(label_batch)
    q_cpu = np.concatenate(q_rows, axis=0)
    labels = np.concatenate(labels_rows, axis=0)
    if labels.ndim != 3 or labels.shape[1:] != (32, 32):
        raise ValueError("production CPU QCC must return [B,32,32]")
    return q_cpu, labels


def _new_initialized_source(device, r):
    found = calibration._load_source(device)
    found.encoder.eval(); found.patcher.eval(); found.wrapped.eval()
    source_sha = calibration._source_digest(found)
    if getattr(found.wrapped.core.graph_generator, "_sw0143_geodesic_arm", None) != "T":
        raise ValueError("SW0170 requires the frozen SW0143-T graph operator")
    if found.wrapped.core.kuramoto_backend != "factorized":
        raise ValueError("SW0170 requires the registered factorized backend")
    init = hard_reset.initialize_slow_membrane_(found.wrapped)
    return found, source_sha, init


def _measure_initial(prepared, calibration_report, training, device):
    """Evaluate both readouts on the same eight actual source-mode candidate traces."""
    classifier = _import("snn_kuramoto_bidirectional.spike_classifier")
    objective = _import("collaborative_test.SW_0168_end_to_end_connectivity.connectivity_objective")
    r = torch.tensor(calibration_report["r"], device=device, dtype=torch.float32)
    if tuple(float(x) for x in calibration_report["r"]) != REGISTERED_R:
        raise ValueError("recomputed source calibration differs from the pinned SW0169 values")
    found, pre_hash, initialization = _new_initialized_source(device, r)
    if pre_hash != prepared["source_cache_state_sha256"]:
        raise ValueError("fresh initialized source was not loaded from the registered source cache")
    initial_hash = calibration._source_digest(found)
    per_row, identity_losses, filtered_losses = [], [], []
    identity_routes, filtered_routes = [], []
    filtered_fg_images = []
    raw_chunks, filtered_chunks = [], []
    start_time = time.perf_counter()

    for start in (0, 4):
        rgb = torch.from_numpy(np.ascontiguousarray(training["rgb8"][start:start + 4])).to(device)
        gamma = found.encode(rgb)
        trace = hard_reset.rollout(found.wrapped, gamma, r, total_steps=STEPS,
                                   live_tail_steps=TAIL, detach_history=False)
        raw = trace["component_spikes"]
        identity = synapse.causal_synapse(raw, identity=True)
        filtered = synapse.causal_synapse(raw, live_start=synapse.LIVE_START)
        if identity is not raw or not torch.equal(identity, raw):
            raise ValueError("identity readout must alias the actual emitted S")
        q_raw = classifier.spike_synchrony_affinity(
            trace["spikes"], raw, settle=SETTLE, affinity_mode="spike")
        q_identity = synapse.production_affinity(identity, settle=SETTLE)
        q_filtered = synapse.production_affinity(filtered, settle=SETTLE)
        if not torch.equal(q_raw, q_identity):
            raise ValueError("identity synapse changed the native point-event Q")
        if not torch.isfinite(q_filtered).all() or not torch.isfinite(filtered).all():
            raise FloatingPointError("filtered synaptic trace/Q is nonfinite")
        raw_chunks.append(raw.detach().cpu().contiguous().numpy())
        filtered_chunks.append(filtered.detach().cpu().contiguous().numpy())

        for offset in range(4):
            index = start + offset
            target = training["targets"][index]
            support = training["supports"][index]
            loss0 = objective.constrained_maximin_loss(q_identity[offset], target, support > 0)
            raw_loss = objective.constrained_maximin_loss(q_raw[offset], target, support > 0)
            loss1 = objective.constrained_maximin_loss(q_filtered[offset], target, support > 0)
            if not torch.equal(loss0, raw_loss):
                raise ValueError("identity readout changed the native point-event objective")
            plan0 = objective.forest_routes(q_identity[offset], target, support > 0)
            plan1 = objective.forest_routes(q_filtered[offset], target, support > 0)
            route0, route1 = _weighted_route_means(q_identity[offset], plan0), _weighted_route_means(
                q_filtered[offset], plan1)
            identity_losses.append(float(loss0.detach().cpu()))
            filtered_losses.append(float(loss1.detach().cpu()))
            identity_routes.append(route0)
            filtered_routes.append(route1)
            offdiag = ~torch.eye(q_filtered.shape[-1], dtype=torch.bool, device=q_filtered.device)
            z_row = filtered[offset, :, :, SETTLE:]
            row = {"index": index, "image_id": int(training["selected_ids"][index]),
                   "raw_s_sha256": _sha_tensor(raw[offset]),
                   "identity_q_sha256": _sha_tensor(q_identity[offset]),
                   "filtered_q_sha256": _sha_tensor(q_filtered[offset]),
                   "identity_offdiagonal_q_std": float(q_identity[offset][offdiag].std(unbiased=False).cpu()),
                   "identity_loss": identity_losses[-1], "filtered_loss": filtered_losses[-1],
                   "identity_routes": route0, "filtered_routes": route1,
                   "raw_s_nonzero_fraction": float((raw[offset, ..., SETTLE:] != 0).float().mean().cpu()),
                   "raw_event_nonzero_fraction": float((trace["raw_events"][offset, ..., SETTLE:] != 0).float().mean().cpu()),
                   "filtered_mean": float(z_row.mean().cpu()),
                   "filtered_activity_variance": float(z_row.var(unbiased=False).cpu()),
                   "filtered_offdiagonal_q_std": float(q_filtered[offset][offdiag].std(unbiased=False).cpu()),
                   "finite": bool(torch.isfinite(q_filtered[offset]).all()),
                   "nonconstant_offdiagonal": bool(q_filtered[offset][offdiag].std(unbiased=False) > 0)}
            per_row.append(row)
        del rgb, gamma, trace, raw, identity, filtered, q_raw, q_identity, q_filtered

    # CPU QCC support for the standalone readout is recomputed in the same
    # registered B4 grouping from the filtered actual S-derived traces.
    settled_filtered = np.concatenate(filtered_chunks, axis=0)[..., SETTLE:]
    cpu_q, cpu_labels = _cpu_filtered_qcc(settled_filtered)
    identity_cpu_q, identity_labels = _cpu_filtered_qcc(
        np.concatenate(raw_chunks, axis=0)[..., SETTLE:])
    for index, labels in enumerate(cpu_labels):
        fg_count = int(np.count_nonzero(labels > 0))
        identity_row_labels = identity_labels[index]
        identity_fg_count = int(np.count_nonzero(identity_row_labels > 0))
        per_row[index]["filtered_qcc_foreground_patches"] = fg_count
        per_row[index]["filtered_qcc_group_sizes"] = sorted(
            [int(np.count_nonzero(labels == group)) for group in np.unique(labels) if group > 0],
            reverse=True)
        per_row[index]["identity_qcc_foreground_patches"] = identity_fg_count
        per_row[index]["identity_qcc_group_sizes"] = sorted(
            [int(np.count_nonzero(identity_row_labels == group))
             for group in np.unique(identity_row_labels) if group > 0], reverse=True)
        if fg_count > 0:
            filtered_fg_images.append(index)
    post_hash = calibration._source_digest(found)
    if post_hash != initial_hash:
        raise ValueError("source parameters changed during optimizer-zero readout measurements")
    report = {"source_state_sha256_before_init": pre_hash,
              "source_state_sha256_after_init": initial_hash,
              "source_state_sha256_after_measurement": post_hash,
              "initialization": initialization,
              "source_state_unchanged_during_measurement": True,
              "rows": per_row,
              "mean_identity_loss": float(np.mean(identity_losses)),
              "mean_filtered_loss": float(np.mean(filtered_losses)),
              "mean_identity_forest_separation": (
                  float(np.mean([x["separation"] for x in identity_routes
                                 if x["separation"] is not None]))
                  if any(x["separation"] is not None for x in identity_routes) else None),
              "mean_filtered_forest_separation": (
                  float(np.mean([x["separation"] for x in filtered_routes
                                 if x["separation"] is not None]))
                  if any(x["separation"] is not None for x in filtered_routes) else None),
              "eligible_image_indices": [i for i, route in enumerate(identity_routes)
                                         if route["separation"] is not None],
              "filtered_qcc_foreground_image_indices": filtered_fg_images,
              "filtered_cpu_q_sha256": [_sha_bytes(np.ascontiguousarray(row).tobytes())
                                         for row in cpu_q],
              "identity_cpu_q_sha256": [_sha_bytes(np.ascontiguousarray(row).tobytes())
                                         for row in identity_cpu_q],
              "elapsed_seconds": time.perf_counter() - start_time,
              "source_ids": list(map(int, training["selected_ids"]))}
    return found, r, report


def _family_gradients(families):
    return {name: _norm([parameter.grad for parameter in parameters])
            for name, parameters in families.items()}


def _direct_synapse_carry_gradient(captured):
    """Isolate the actual causal-filter carry partial at a live timestep."""
    if captured is None or captured["next_state"].grad is None:
        raise ValueError("captured causal Z state did not receive objective gradient")
    previous = captured["previous_state"].detach().clone().requires_grad_(True)
    current = captured["input_spike"].detach()
    replay = synapse.DECAY * previous + (1.0 - synapse.DECAY) * current
    if not torch.equal(replay.detach(), captured["next_state"].detach()):
        raise ValueError("isolated causal-state partial replay changed the actual recurrence output")
    gradient, = torch.autograd.grad(
        replay, (previous,), grad_outputs=captured["next_state"].grad.detach())
    return {"step": int(captured["step"]), "previous_state_gradient_norm": _norm([gradient]),
            "scope": "local causal Z carry partial; actual S_t held fixed"}


def _named_parameter_records(found, before_named):
    records = {}
    for name, parameter in found.wrapped.core.named_parameters():
        old = before_named[name]
        records[name] = {
            "raw_gradient_norm": _norm([parameter.grad]),
            "parameter_delta": float(torch.linalg.vector_norm(
                parameter.detach().cpu() - old)),
        }
    return records


def _disposable_updates(training, device, r):
    """Two same-objective disposable updates for identity and causal readouts."""
    objective = _import("collaborative_test.SW_0168_end_to_end_connectivity.connectivity_objective")
    classifier = _import("snn_kuramoto_bidirectional.spike_classifier")
    parameter_helper = _import("collaborative_test.SW_0169_normalized_hard_reset.preflight_v3")
    arm_reports = {}
    common_initial = None
    for arm in ARMS:
        found, _source_sha, initialization = _new_initialized_source(device, r)
        families = parameter_helper._parameter_families(found)
        initial_model = calibration._source_digest(found)
        if common_initial is None:
            common_initial = initial_model
        elif initial_model != common_initial:
            raise ValueError("identity and causal arms do not share exact fresh initialization")
        groups = [dict(params=params, lr=3e-5, name=name)
                  for name, params in families.items() if name != "integration"]
        groups.append(dict(params=families["integration"], lr=1e-3, name="integration"))
        optimizer = torch.optim.Adam(groups)
        updates = []
        for update in (1, 2):
            before = {name: [p.detach().cpu().clone() for p in params]
                      for name, params in families.items()}
            named_before = {name: p.detach().cpu().clone()
                            for name, p in found.wrapped.core.named_parameters()}
            integration_before = {name: parameter.detach().cpu().clone()
                                  for name, parameter in zip(
                                      ("a_d", "a_m", "b"), families["integration"])}
            optimizer.zero_grad(set_to_none=True)
            q_norm, z_norm, z_live_norm, s_norm, s_live_norm, losses = [], [], [], [], [], []
            direct_z_carry = None
            for start in (0, 4):
                rgb = torch.from_numpy(np.ascontiguousarray(training["rgb8"][start:start + 4])).to(device)
                gamma = found.encode(rgb)
                trace = hard_reset.rollout(found.wrapped, gamma, r, total_steps=STEPS,
                                           live_tail_steps=TAIL, detach_history=False)
                s = trace["component_spikes"]
                if not s.requires_grad:
                    raise ValueError("actual emitted S is not connected to source parameters")
                s.retain_grad()
                if arm == "causal_synapse" and update == 1 and start == 0:
                    z, captured_z = synapse.causal_synapse(
                        s, live_start=synapse.LIVE_START, capture_step=1000)
                else:
                    z = synapse.causal_synapse(
                        s, identity=(arm == "identity_synapse"),
                        live_start=synapse.LIVE_START)
                    captured_z = None
                if z.requires_grad:
                    z.retain_grad()
                q = synapse.production_affinity(z, settle=SETTLE)
                if not q.requires_grad:
                    raise ValueError("production affinity is disconnected from synaptic trace")
                q.retain_grad()
                per_image = [objective.constrained_maximin_loss(
                    q[i], training["targets"][start + i],
                    training["supports"][start + i] > 0) for i in range(4)]
                loss = torch.stack(per_image).sum() / IDS
                loss.backward()
                if captured_z is not None:
                    direct_z_carry = _direct_synapse_carry_gradient(captured_z)
                losses.extend(float(value.detach().cpu()) for value in per_image)
                q_norm.append(_norm([q.grad]))
                z_norm.append(_norm([z.grad]) if z.grad is not None else 0.0)
                z_live_norm.append(_norm([z.grad[..., synapse.LIVE_START:]])
                                   if z.grad is not None else 0.0)
                s_norm.append(_norm([s.grad]) if s.grad is not None else 0.0)
                s_live_norm.append(_norm([s.grad[..., synapse.LIVE_START:]])
                                   if s.grad is not None else 0.0)
                del rgb, gamma, trace, s, z, q, per_image, loss
            gradients = _family_gradients(families)
            named_core_gradients = {name: _norm([p.grad for pname, p in
                                    found.wrapped.core.named_parameters() if name in pname])
                                    for name in ("kuramoto", "dendric_layer", "membrane_layer")}
            tau_m_gradient = _norm([found.wrapped.core.membrane_layer.tau_m.grad])
            b_gradient = _norm([found.wrapped.b.grad])
            if not all(math.isfinite(value) for value in gradients.values()):
                raise FloatingPointError("nonfinite family gradient in disposable update")
            # Preserve truthful unused families, but source-path credit must be
            # live through encoder, graph, and native core.
            critical_gradient = (all(gradients[name] > 0
                                     for name in ("encoder", "graph", "native_core"))
                                 and tau_m_gradient > 0 and b_gradient > 0)
            if (not critical_gradient or max(q_norm) <= 0 or max(z_live_norm) <= 0
                    or max(s_live_norm) <= 0):
                raise ValueError("causal readout has no actionable Q->Z->S/source gradient")
            if arm == "causal_synapse" and update == 1 and (
                    direct_z_carry is None or direct_z_carry["previous_state_gradient_norm"] <= 0):
                raise ValueError("actual causal Z recurrence carry received no direct gradient")
            tau_before = found.wrapped.core.membrane_layer.tau_m.detach().cpu().clone()
            b_before = found.wrapped.b.detach().cpu().clone()
            params = [parameter for values in families.values() for parameter in values]
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            found.wrapped.project_integrations_()
            deltas = {name: math.sqrt(sum(float((old.double() - p.detach().cpu().double()).square().sum())
                                             for old, p in zip(before[name], families[name])))
                      for name in families}
            tau_m_delta = float(torch.linalg.vector_norm(
                found.wrapped.core.membrane_layer.tau_m.detach().cpu() - tau_before))
            b_delta = float(torch.linalg.vector_norm(found.wrapped.b.detach().cpu() - b_before))
            critical_changes = (all(deltas[name] > 0
                                    for name in ("encoder", "graph", "native_core"))
                                and tau_m_delta > 0 and b_delta > 0)
            critical = critical_gradient and critical_changes
            updates.append({"update": update, "mean_loss": float(np.mean(losses)),
                            "q_gradient_norm": float(np.mean(q_norm)),
                            "z_gradient_norm": float(np.mean(z_norm)),
                            "live_tail_z_gradient_norm": float(np.mean(z_live_norm)),
                            "actual_s_gradient_norm": float(np.mean(s_norm)),
                            "live_tail_s_gradient_norm": float(np.mean(s_live_norm)),
                            "direct_z_carry_partial": direct_z_carry,
                            "family_gradient_norms": gradients,
                            "named_core_gradient_norms": named_core_gradients,
                            "named_core_parameter_credit": _named_parameter_records(
                                found, named_before),
                            "tau_m_gradient_norm": tau_m_gradient,
                            "b_gradient_norm": b_gradient,
                            "tau_m_parameter_delta": tau_m_delta,
                            "b_parameter_delta": b_delta,
                            "integration_parameter_deltas": {
                                name: float(torch.linalg.vector_norm(
                                    parameter.detach().cpu() - integration_before[name]))
                                for name, parameter in zip(
                                    ("a_d", "a_m", "b"), families["integration"])},
                            "integration_parameter_gradient_norms": {
                                name: _norm([parameter.grad])
                                for name, parameter in zip(
                                    ("a_d", "a_m", "b"), families["integration"])},
                            "family_parameter_deltas": deltas,
                            "critical_gradient_credit": critical_gradient,
                            "critical_parameter_changes": critical_changes,
                            "critical_credit": critical})
        arm_reports[arm] = {"initial_state_sha256": initial_model,
                            "initialization": initialization, "updates": updates}
        del optimizer, families, found
    return arm_reports


def run_preflight(*, device="cuda:0", output_path, prepared=None):
    """Run source parity, TRAIN8 optimizer-zero gates, then disposable updates.

    Scientific gate failures are returned as ``training_admission=false``
    receipts. Operational failures raise and should be preserved by the caller.
    """
    if prepared is None:
        prepared = prepare_inputs_cpu()
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0170 preflight requires CPU admission before CUDA initialization")
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise RuntimeError("SW0170 disposable admission requires an owned CUDA device")
    dev = torch.device(device)
    torch.cuda.set_device(dev)
    torch.cuda.reset_peak_memory_stats(dev)

    source_calibration = calibration.run_source_calibration_gpu(prepared, device=str(dev))
    if tuple(float(value) for value in source_calibration["r"]) != REGISTERED_R:
        raise ValueError("live source calibration differs from frozen SW0169 r4")
    source_handoff = {**source_calibration,
                      "source_cache_manifest_sha256": prepared["source_cache_manifest_sha256"],
                      "context_cache_manifest_sha256": prepared["context_cache_manifest_sha256"],
                      "source_cache_state_sha256": prepared["source_cache_state_sha256"],
                      "context_cache_state_sha256": prepared["context_cache_state_sha256"]}
    target_prep = _import("collaborative_test.SW_0169_normalized_hard_reset.preflight_v3")
    training = target_prep._prepare_targets_fresh_cpu(source_handoff)
    if training["selected_ids"] != prepared["selected_ids"]:
        raise ValueError("fresh TRAIN-target worker changed the registered first-eight IDs")
    found, r, metrics = _measure_initial(prepared, source_calibration, training, dev)

    identity_routes = [row["identity_routes"] for row in metrics["rows"]]
    filtered_routes = [row["filtered_routes"] for row in metrics["rows"]]
    failures = _gate_failures(metrics["rows"], metrics["mean_identity_loss"],
                              metrics["mean_filtered_loss"], identity_routes,
                              filtered_routes, metrics["filtered_qcc_foreground_image_indices"])
    base_report = {"experiment": EXPERIMENT, "status": "scientific_rejection" if failures else "stage1_passed",
                   "training_admission": False, "validation_data_read": False,
                   "ground_truth_scope": "eight modal32 TRAIN masks only",
                   "source": {"selected_ids": list(map(int, prepared["selected_ids"])),
                              "positions": list(range(IDS)),
                              "source_cache_manifest_sha256": prepared["source_cache_manifest_sha256"],
                              "context_cache_manifest_sha256": prepared["context_cache_manifest_sha256"],
                              "source_checkpoint_sha256": prepared["source_checkpoint_sha256"],
                              "r": list(REGISTERED_R),
                              "source_proof": prepared.get("source_proof")},
                   "train_targets": {
                       "selected_ids": list(map(int, training["selected_ids"])),
                       "target_receipts": training["target_receipts"],
                       "hdf5_path": training["hdf5_path"],
                       "hdf5_size_bytes": int(training["hdf5_size_bytes"]),
                       "hdf5_mtime_ns": int(training["hdf5_mtime_ns"]),
                       "ground_truth_scope": training["ground_truth_scope"],
                       "validation_data_read": False},
                   "source_calibration": source_calibration,
                   "optimizer_zero": metrics,
                   "disposable_logical_batch_size": 8,
                   "disposable_microbatch_size": 4,
                   "stage1_gate_failures": failures,
                   "implementation_fingerprint": implementation_fingerprint(),
                   "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(dev))}
    if failures:
        _atomic_json(output_path, base_report)
        return base_report

    del found
    arm_reports = _disposable_updates(training, dev, r)
    passed = all(len(arm_reports[arm]["updates"]) == 2
                 and all(row["critical_credit"] for row in arm_reports[arm]["updates"])
                 for arm in ARMS)
    base_report.update({"status": "complete_admitted" if passed else "scientific_rejection",
                        "training_admission": bool(passed), "arms": arm_reports,
                        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(dev))})
    if not passed:
        base_report["stage1_gate_failures"].append("disposable_update_credit_failed")
    _atomic_json(output_path, base_report)
    return base_report
