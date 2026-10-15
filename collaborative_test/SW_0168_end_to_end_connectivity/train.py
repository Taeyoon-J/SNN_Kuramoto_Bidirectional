"""Fixed two-pass TRAIN pilot for the SW0168 native connectivity losses."""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import random
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from collaborative_test.SW_0168_end_to_end_connectivity import connectivity_objective as objective
from collaborative_test.SW_0168_end_to_end_connectivity import preflight_v3 as preflight

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARMS = preflight.ARMS
COUNT, BATCH, MICRO = 512, 16, 4
UPDATES, EPOCHS = 64, 2
SHUFFLE_SEED = 168001
MODEL_SEED = 168001
BASE_LR, INTEGRATION_LR, CLIP = 3e-5, 1e-3, 1.0
TRAIN_ROOT = HERE / "results_archive" / "training_20261015"
PREFLIGHT_STATE = HERE / "results_archive" / "source_admission_v4_20261015" / "state.json"
PREFLIGHT_REPORT = HERE / "results_archive" / "source_admission_v4_20261015" / "preflight.json"


def _module(name):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        return importlib.import_module(name)
    finally:
        if visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible


def _sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _json_sha(value):
    return _sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode("utf-8"))


def _epoch_orders():
    rng = random.Random(SHUFFLE_SEED)
    return [rng.sample(range(COUNT), COUNT) for _ in range(EPOCHS)]


def implementation_fingerprint():
    rels = (
        "collaborative_test/SW_0168_end_to_end_connectivity/DESIGN_KO.md",
        "collaborative_test/SW_0168_end_to_end_connectivity/connectivity_objective.py",
        "collaborative_test/SW_0168_end_to_end_connectivity/preflight_v3.py",
        "collaborative_test/SW_0168_end_to_end_connectivity/preflight_queue_v4.py",
        "collaborative_test/SW_0168_end_to_end_connectivity/train.py",
        "collaborative_test/SW_0167_structured_relation_energy/audit_train_cache.py",
        "collaborative_test/SW_0136_native32_transfer/evaluate.py",
        "collaborative_test/SW_0161_supervised_spike_partition/cache.py",
        "collaborative_test/SW_0155_supervised_spike_relations/train.py",
        "collaborative_test/SW_0155_supervised_spike_relations/objective.py",
        "collaborative_test/SW_0130_phase_state_integration/run.py",
        "collaborative_test/SW_0134_native_spike_binding/rollout.py",
        "collaborative_test/SW_0135_native32_spike_binding/foundation.py",
        "collaborative_test/SW_0143_native32_selective_geodesic/geodesic_operator.py",
        "collaborative_test/SW_0148_checkpointable_boundary_q/checkpoint.py",
        "collaborative_test/SW_0162_connected_signed_partition/preflight.py",
    )
    out = {}
    for rel in rels:
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(f"SW0168 training dependency missing: {rel}")
        out[rel] = _sha_file(path)
    return out


def _load_admission():
    pq = _module("collaborative_test.SW_0168_end_to_end_connectivity.preflight_queue_v4")
    prep = _module("collaborative_test.SW_0168_end_to_end_connectivity.preflight_v3")
    if not PREFLIGHT_STATE.is_file() or not PREFLIGHT_REPORT.is_file():
        raise FileNotFoundError("SW0168 source8 preflight receipt is missing")
    state = json.loads(PREFLIGHT_STATE.read_text(encoding="utf-8"))
    if state.get("experiment") != pq.EXPERIMENT or state.get("status") != "complete_admitted":
        raise ValueError("SW0168 source8 preflight owner state is not complete_admitted")
    report_sha = _sha_file(PREFLIGHT_REPORT)
    if report_sha != state.get("report_sha256"):
        raise ValueError("SW0168 source8 preflight report hash differs from terminal owner state")
    _prep, _prepared, proof = pq._source_admission()
    queue_fp = pq.fingerprint_sha256()
    report = json.loads(PREFLIGHT_REPORT.read_text(encoding="utf-8"))
    pq.validate_report(report, source_proof=proof, fingerprint=queue_fp)
    # Bind the exact source cache/context manifests used by the report.
    if (report.get("source_cache_manifest_sha256") != proof["source_cache_manifest_sha256"]
            or report.get("context_cache_manifest_sha256") != proof["context_cache_manifest_sha256"]):
        raise ValueError("SW0168 preflight no longer matches current cache manifests")
    return prep, proof, report_sha


def _open_full_caches(proof):
    audit = _module("collaborative_test.SW_0167_structured_relation_energy.audit_train_cache")
    source_cache, context_cache = audit.source_cache, audit.context_cache
    source_manifest_path = source_cache.CACHE_ROOT / "manifest.json"
    context_manifest_path = context_cache.CACHE_ROOT / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    context_manifest = json.loads(context_manifest_path.read_text(encoding="utf-8"))
    source_cache.validate_final_cache(source_cache.CACHE_ROOT,
                                      expected_bindings=source_manifest["bindings"])
    context_cache.validate_final_cache(context_cache.CACHE_ROOT,
                                       expected_bindings=context_manifest["bindings"])
    if source_manifest.get("count") != COUNT or context_manifest.get("count") != COUNT:
        raise ValueError("SW0168 training requires both complete 512-row source caches")
    checkpoint, source_manifest_file, _manifest, pool_rows, ordered_ids, source_sha = (
        audit.sw130.source_contract(1))
    source_rgb_sha = source_manifest["bindings"].get("image_rgb_sha256")
    if (source_sha != proof["source_checkpoint_sha256"]
            or _sha_file(source_manifest_path) != proof["source_cache_manifest_sha256"]
            or _sha_file(context_manifest_path) != proof["context_cache_manifest_sha256"]
            or [int(v) for v in ordered_ids[:COUNT]] != source_manifest["bindings"].get("image_ids")
            or [int(v) for v in ordered_ids[:COUNT]] != context_manifest["bindings"].get("ordered_image_ids")
            or [int(v) for v in ordered_ids[:8]] != proof["selected_ids"]):
        raise ValueError("full cache checkpoint/order bindings differ from admitted source")
    source = source_cache._AtomicTraceCache(source_cache.CACHE_ROOT,
        source_manifest["bindings"], ordered_ids[:COUNT], source_rgb_sha)
    context = context_cache.AtomicContextCache(context_cache.CACHE_ROOT,
        context_manifest["bindings"])
    if len(source.records) != COUNT or len(context.records) != COUNT:
        raise ValueError("cache loaders do not expose exactly 512 committed rows")
    gain, gain_receipt = audit._load_registered_gain(source_sha, _sha_file(source_manifest_file))
    return audit, {
        "source": source, "context": context,
        "source_manifest": source_manifest, "context_manifest": context_manifest,
        "source_manifest_sha256": _sha_file(source_manifest_path),
        "context_manifest_sha256": _sha_file(context_manifest_path),
        "source_ledger_sha256": _sha_file(source_cache.CACHE_ROOT / "ledger.json"),
        "context_ledger_sha256": _sha_file(context_cache.CACHE_ROOT / "ledger.json"),
        "source_file_sha256": _sha_file(source_manifest_file),
        "source_checkpoint_sha256": source_sha,
        "source_checkpoint_path": str(checkpoint),
        "source_ids": [int(v) for v in ordered_ids[:COUNT]],
        "pool_rows": np.asarray(pool_rows[:COUNT], dtype=np.int64),
        "gain": float(gain), "gain_receipt_sha256": gain_receipt,
        "source_state_sha256": source_manifest["source_model_state_before_sha256"],
    }


def _prepare_supports_targets(audit, caches):
    qcc = _module("collaborative_test.SW_0167_structured_relation_energy.qcc_readout")
    structured = _module("collaborative_test.SW_0167_structured_relation_energy.structured_energy")
    sw155 = _module("collaborative_test.SW_0155_supervised_spike_relations.train")
    objective155 = _module("collaborative_test.SW_0155_supervised_spike_relations.objective")
    sw130 = audit.sw130
    rgb_memmap = np.load(sw130.TRAIN_RGB, mmap_mode="r", allow_pickle=False)
    rgb = np.asarray(rgb_memmap[caches["pool_rows"]]).copy()
    if rgb.dtype != np.uint8 or rgb.shape != (COUNT, 128, 128, 3):
        raise ValueError("SW0168 aligned TRAIN RGB cache schema changed")
    ids = caches["source_ids"]
    aligned_rgb, masks = sw155.load_aligned_train_rows(
        sw155.HDF5_PATH, ids, rgb, 0, COUNT)
    if (aligned_rgb.dtype != torch.uint8 or aligned_rgb.shape != (COUNT, 128, 128, 3)
            or not np.array_equal(aligned_rgb.cpu().numpy(), rgb)):
        raise ValueError("canonical HDF5 image rows do not match registered TRAIN RGB rows")
    targets32 = objective155.modal_train_targets32(masks).cpu().numpy()
    if targets32.shape != (COUNT, 1024):
        raise ValueError("TRAIN modal32 target builder changed output schema")
    supports, targets = [], []
    for start in range(0, COUNT, 4):
        components = np.stack([caches["source"].load_row(i)[0]
                               for i in range(start, start + 4)])
        q_b4 = audit._recompute_cpu_q(components)
        for offset in range(4):
            index = start + offset
            support = qcc.qcc_labels_from_affinity(q_b4[offset:offset+1])[0].numpy().astype(np.int32)
            target = structured.feasible_target(support, targets32[index].reshape(32, 32))
            supports.append(support)
            targets.append(target.astype(np.int32, copy=False))
    if len(supports) != COUNT or len(targets) != COUNT:
        raise ValueError("SW0168 support/target admission did not cover all 512 TRAIN rows")
    hdf_stat = Path(sw155.HDF5_PATH).stat()
    return {"rgb": rgb, "supports": np.stack(supports), "targets": np.stack(targets),
            "hdf5_path": str(Path(sw155.HDF5_PATH).resolve()),
            "hdf5_size_bytes": int(hdf_stat.st_size), "hdf5_mtime_ns": int(hdf_stat.st_mtime_ns)}


def prepare_training_inputs_cpu():
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0168 training inputs must be validated before CUDA initialization")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    _prep, proof, admission_sha = _load_admission()
    audit, caches = _open_full_caches(proof)
    data = _prepare_supports_targets(audit, caches)
    bindings = {
        "experiment": "SW0168_actual_spike_connectivity_training", "seed": 1,
        "arms": list(ARMS), "preflight_report_sha256": admission_sha,
        "preflight_source": proof,
        "source_checkpoint_sha256": caches["source_checkpoint_sha256"],
        "source_manifest_sha256": caches["source_file_sha256"],
        "source_cache_manifest_sha256": caches["source_manifest_sha256"],
        "context_cache_manifest_sha256": caches["context_manifest_sha256"],
        "source_cache_ledger_sha256": caches["source_ledger_sha256"],
        "context_cache_ledger_sha256": caches["context_ledger_sha256"],
        "source_state_sha256": caches["source_state_sha256"],
        "ordered_train_ids": caches["source_ids"],
        "ordered_train_ids_sha256": _sha_bytes(np.asarray(caches["source_ids"], dtype="<i8").tobytes()),
        "source_pool_rows_sha256": _sha_bytes(caches["pool_rows"].astype("<i8").tobytes()),
        "source_gain": caches["gain"], "gain_receipt_sha256": caches["gain_receipt_sha256"],
        "source_support_sha256": _sha_bytes(np.ascontiguousarray(data["supports"], dtype="<i4").tobytes()),
        "feasible_targets_sha256": _sha_bytes(np.ascontiguousarray(data["targets"], dtype="<i4").tobytes()),
        "train_rgb_sha256": [_sha_bytes(row.tobytes()) for row in data["rgb"]],
        "train_rgb_manifest_sha256": caches["source_manifest"]["bindings"]["source_rgb_asset_validation"]["train_manifest_sha256"],
        "hdf5_path": data["hdf5_path"], "hdf5_size_bytes": data["hdf5_size_bytes"],
        "hdf5_mtime_ns": data["hdf5_mtime_ns"],
        "updates": UPDATES, "epochs": EPOCHS, "logical_batch": BATCH, "microbatch": MICRO,
        "shuffle_seed": SHUFFLE_SEED, "model_seed": MODEL_SEED,
        "base_learning_rate": BASE_LR, "integration_learning_rate": INTEGRATION_LR,
        "optimizer": "Adam", "gradient_clip_norm": CLIP,
        "objectives": {"constrained_maximin": "pair-count weighted connected maximin positive/negative forest",
                       "matched_local_edge": "class-balanced local four-neighbor edge loss"},
        "ground_truth_scope": "512 canonical TRAIN modal32 masks only",
    }
    if os.environ.get("CUDA_VISIBLE_DEVICES") != visible or torch.cuda.is_initialized():
        raise RuntimeError("CPU training input preparation altered CUDA visibility/state")
    caches["data"] = data
    caches["bindings"] = bindings
    caches["admission_sha256"] = admission_sha
    caches["preflight_source"] = proof
    return caches


def _model_families(found):
    graph = found.wrapped.core.graph_generator
    graph.geodesic_contrast.requires_grad_(False)
    groups = {
        "encoder": list(found.encoder.parameters()),
        "graph": [p for name, p in graph.named_parameters() if name != "geodesic_contrast"],
        "native_core": [p for name, p in found.wrapped.core.named_parameters()
                        if not name.startswith("graph_generator.")],
        "integration": [found.wrapped.a_d, found.wrapped.a_m, found.wrapped.b],
    }
    params = [p for values in groups.values() for p in values]
    if not params or len({id(p) for p in params}) != len(params):
        raise ValueError("SW0168 optimizer families are empty or overlap")
    for parameter in params:
        parameter.requires_grad_(True)
    return groups


def _state_sha(found):
    cache = _module("collaborative_test.SW_0161_supervised_spike_partition.cache")
    return cache._state_digest(found)


def _training_bindings(base, arm):
    return {**base, "arm": arm}


def _update(arm, found, optimizer, indices, caches, device):
    optimizer.zero_grad(set_to_none=True)
    before = {group["name"]: [p.detach().clone() for p in group["params"]]
              for group in optimizer.param_groups}
    losses = []
    data = caches["data"]
    for micro_start in range(0, BATCH, MICRO):
        ids = indices[micro_start:micro_start + MICRO]
        if len(ids) != MICRO:
            raise ValueError("SW0168 logical update has an incomplete four-image microbatch")
        rgb = torch.from_numpy(np.ascontiguousarray(data["rgb"][ids])).to(device)
        gamma = found.encode(rgb)
        trace = found.rollout(gamma, total_steps=1024, live_tail_steps=64)
        if any(not torch.isfinite(trace[key]).all() for key in
               ("theta", "membrane", "spikes", "component_membrane", "component_spikes", "component_gates")):
            raise FloatingPointError("SW0168 training rollout contains nonfinite traces")
        q = preflight._q(trace)
        losses_micro = []
        for offset, raw_index in enumerate(ids):
            index = int(raw_index)
            target = data["targets"][index]
            support = data["supports"][index]
            foreground = np.asarray(support) > 0
            loss = (objective.constrained_maximin_loss(q[offset], target, foreground)
                    if arm == "constrained_maximin" else
                    objective.matched_local_edge_loss(q[offset], target, foreground))
            if loss.ndim or not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite SW0168 {arm} loss at TRAIN row {index}")
            losses_micro.append(loss)
            losses.append(float(loss.detach().cpu()))
        (torch.stack(losses_micro).sum() / BATCH).backward()
        del rgb, gamma, trace, q, losses_micro, loss
    family_gradients = {}
    active = [p for group in optimizer.param_groups for p in group["params"]]
    grad_sq = 0.0
    for group in optimizer.param_groups:
        family_sq = 0.0
        for parameter in group["params"]:
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError("nonfinite SW0168 trainable gradient")
                contribution = float(parameter.grad.detach().double().square().sum().cpu())
                grad_sq += contribution
                family_sq += contribution
        family_gradients[group["name"]] = math.sqrt(family_sq)
    grad_norm = math.sqrt(grad_sq)
    norm = torch.nn.utils.clip_grad_norm_(active, CLIP)
    if not torch.isfinite(norm):
        raise FloatingPointError("nonfinite global gradient norm")
    optimizer.step()
    found.wrapped.project_integrations_()
    family_deltas = {}
    named_credit = {}
    integration_scalars = {}
    for group in optimizer.param_groups:
        delta_sq = sum(float((old.detach().double() - parameter.detach().double()).square().sum().cpu())
                       for old, parameter in zip(before[group["name"]], group["params"]))
        family_deltas[group["name"]] = math.sqrt(delta_sq)
        names = group["parameter_names"]
        if group["name"] == "integration":
            for parameter_name, old, parameter in zip(names, before[group["name"]], group["params"]):
                key = parameter_name.rsplit(".", 1)[-1]
                integration_scalars[key] = {
                    "raw_gradient_l2": (float(parameter.grad.detach().double().norm().cpu())
                                        if parameter.grad is not None else 0.0),
                    "parameter_delta_l2": float((old.detach().double() - parameter.detach().double()).norm().cpu()),
                    "gradient_present": parameter.grad is not None,
                }
        for subgroup in ("kuramoto", "dendrite", "membrane", "integration_other"):
            selected = []
            for parameter_name, old, parameter in zip(names, before[group["name"]], group["params"]):
                label = parameter_name.lower()
                if subgroup == "kuramoto" and ("kuramoto" in label or "gamma" in label or "drive" in label):
                    selected.append((parameter_name, old, parameter))
                elif subgroup == "dendrite" and "dendrit" in label:
                    selected.append((parameter_name, old, parameter))
                elif subgroup == "membrane" and "membrane" in label:
                    selected.append((parameter_name, old, parameter))
                elif subgroup == "integration_other" and group["name"] == "integration":
                    selected.append((parameter_name, old, parameter))
            if selected:
                raw_sq = sum(float(p.grad.detach().double().square().sum().cpu())
                             for _, _, p in selected if p.grad is not None)
                change_sq = sum(float((old.detach().double() - p.detach().double()).square().sum().cpu())
                                for _, old, p in selected)
                named_credit[f"{group['name']}/{subgroup}"] = {
                    "parameter_names": [name for name, _, _ in selected],
                    "raw_gradient_l2": math.sqrt(raw_sq),
                    "parameter_delta_l2": math.sqrt(change_sq),
                    "gradient_present_count": sum(p.grad is not None for _, _, p in selected),
                }
    return {"mean_loss": float(np.mean(losses)), "gradient_norm": grad_norm,
            "family_gradient_norms": family_gradients,
            "family_parameter_delta_l2": family_deltas,
            "named_credit": named_credit,
            "integration_scalar_credit": integration_scalars,
            "clip_scale": min(1.0, CLIP / (float(norm.detach().cpu()) + 1e-12)),
            "image_indices": [int(i) for i in indices], "images_seen": BATCH}


def _diagnose_eight(found, caches, *, phase):
    """No-update native T readout/dynamics audit on the registered eight TRAIN rows."""
    solver = _module("collaborative_test.SW_0164_foreground_preserving_partition.foreground_signed")
    evaluator = _module("collaborative_test.SW_0136_native32_transfer.evaluate")
    data = caches["data"]
    device = next(found.encoder.parameters()).device
    was_training = (found.encoder.training, found.patcher.training, found.wrapped.training)
    found.encoder.eval(); found.patcher.eval(); found.wrapped.eval()
    rows = []
    try:
        with torch.no_grad():
            for index in range(8):
                rgb = torch.from_numpy(np.ascontiguousarray(data["rgb"][index:index+1])).to(device)
                gamma, trace, actual_a = preflight._capture_trace(found, rgb)
                q = preflight._q(trace)
                qcc_labels = np.asarray(evaluator._qcc_labels(trace)[0], dtype=np.int32)
                support_labels = np.asarray(data["supports"][index], dtype=np.int32)
                support = support_labels > 0
                target = np.asarray(data["targets"][index], dtype=np.int32)
                plan = objective.forest_routes(q[0], target, support)
                edge_errors = {}
                for field, edges, positive in (
                    ("critical_positive", plan["positive_edges"], True),
                    ("critical_negative", plan["negative_edges"], False)):
                    mass = sum(int(weight) for _, _, weight in edges)
                    if mass:
                        vals = [float(q[0, a, b].cpu()) for a, b, _ in edges]
                        weights = [int(weight) for _, _, weight in edges]
                        err = [((1.0 - value) ** 2 if positive else value ** 2) * weight
                               for value, weight in zip(vals, weights)]
                        edge_errors[field] = sum(err) / mass
                    else:
                        edge_errors[field] = None
                local = plan["local_edges"]
                local_errors = {}
                for same in (1, 0):
                    vals = [float(q[0, a, b].cpu()) for a, b, label in local if label == same]
                    if vals:
                        local_errors["same" if same else "different"] = float(np.mean(
                            [(1.0-v)**2 if same else v**2 for v in vals]))
                    else:
                        local_errors["same" if same else "different"] = None
                m = trace["component_membrane"][..., preflight.SETTLE:]
                gates = trace["component_gates"][..., preflight.SETTLE:]
                spikes = trace["component_spikes"][..., preflight.SETTLE:]
                threshold = (0.06 * torch.exp(found.wrapped.b.detach())).view(1, 4, 1, 1)
                margin = m - threshold
                events = margin > 0
                expected_spikes = events.to(spikes.dtype) * gates
                if not torch.equal(expected_spikes, spikes):
                    raise ValueError("SW0168 diagnostic trace violates S=E*g identity")
                centered = spikes.float() - spikes.float().mean(dim=-1, keepdim=True)
                centered = centered / centered.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                raw_component_corr = torch.stack([
                    centered[:, component] @ centered[:, component].transpose(-1, -2)
                    for component in range(centered.shape[1])
                ], dim=1)[0]
                clamp_active = raw_component_corr <= 0.0
                critical_edges = plan["positive_edges"] + plan["negative_edges"]
                critical_clamp = ([bool(clamp_active[:, a, b].any().cpu())
                                   for a, b, _ in critical_edges] if critical_edges else [])
                active_gate = gates > 0
                eligible_margin = margin[active_gate]
                gain = float(torch.exp(found.wrapped.core.graph_generator.log_coupling_gain.detach()).cpu())
                constrained_loss = float(objective.constrained_maximin_loss(q[0], target, support).detach().cpu())
                local_loss = float(objective.matched_local_edge_loss(q[0], target, support).detach().cpu())
                source_readout = solver.foreground_induced_merge(
                    q[0].detach().cpu().numpy().astype(np.float64, copy=True),
                    actual_a[0].detach().cpu().numpy().astype(np.float64, copy=True),
                    gain, support_labels, grid_shape=(32, 32))
                standalone_readout = solver.foreground_induced_merge(
                    q[0].detach().cpu().numpy().astype(np.float64, copy=True),
                    actual_a[0].detach().cpu().numpy().astype(np.float64, copy=True),
                    gain, qcc_labels, grid_shape=(32, 32))
                q_cpu = q[0].detach().cpu().float().numpy()
                if not np.isfinite(q_cpu).all() or not np.isfinite(actual_a.detach().cpu().numpy()).all():
                    raise FloatingPointError("SW0168 diagnostic Q/A is nonfinite")
                rows.append({
                    "index": index, "image_id": int(caches["source_ids"][index]),
                    "q_sha256": _sha_bytes(np.ascontiguousarray(q_cpu).tobytes()),
                    "actual_a_sha256": _sha_bytes(actual_a[0].detach().cpu().contiguous().numpy().tobytes()),
                    "q_mean": float(q_cpu.mean()), "q_min": float(q_cpu.min()), "q_max": float(q_cpu.max()),
                    "qcc_foreground_patches": int(np.count_nonzero(qcc_labels)),
                    "partition_foreground_patches": int(np.count_nonzero(source_readout["labels"])),
                    "partition_merge_count": int(source_readout["merge_count"]),
                    "partition_labels_sha256": _sha_bytes(np.asarray(source_readout["labels"], dtype="<i4").tobytes()),
                    "standalone_foreground_patches": int(np.count_nonzero(standalone_readout["labels"])),
                    "standalone_merge_count": int(standalone_readout["merge_count"]),
                    "standalone_labels_sha256": _sha_bytes(np.asarray(standalone_readout["labels"], dtype="<i4").tobytes()),
                    "constrained_maximin_loss": constrained_loss,
                    "matched_local_edge_loss": local_loss,
                    "gain": gain,
                    "represented_positive_pair_mass": int(plan["positive_pair_mass"]),
                    "represented_negative_pair_mass": int(plan["negative_pair_mass"]),
                    "excluded_unreachable_negative_pair_mass": int(plan["excluded_unreachable_negative_pair_mass"]),
                    "critical_edge_errors": edge_errors, "local_edge_errors": local_errors,
                    "event_on_fraction": float(events.float().mean().cpu()),
                    "event_transition_fraction": float((events[..., 1:] != events[..., :-1]).float().mean().cpu()),
                    "gate_positive_fraction": float(active_gate.float().mean().cpu()),
                    "event_on_fraction_when_gate_positive": (float(events[active_gate].float().mean().cpu())
                                                               if active_gate.any() else None),
                    "weighted_spike_mean": float(spikes.float().mean().cpu()),
                    "component_product_clamp_active_fraction": clamp_active.float().mean(dim=(1, 2)).cpu().tolist(),
                    "critical_edge_has_component_clamp_fraction": (float(np.mean(critical_clamp))
                        if critical_clamp else None),
                    "margin_quantiles": (torch.quantile(eligible_margin.float(),
                        torch.tensor([.05,.5,.95], device=device)).cpu().tolist()
                        if eligible_margin.numel() else []),
                })
                del rgb, gamma, trace, actual_a, q, m, gates, spikes, margin, events, expected_spikes
    finally:
        found.encoder.train(was_training[0]); found.patcher.train(was_training[1]); found.wrapped.train(was_training[2])
    if len(rows) != 8:
        raise RuntimeError("SW0168 fixed TRAIN diagnostic failed to cover all eight rows")
    return {"phase": phase, "selected_image_ids": [row["image_id"] for row in rows],
            "rows": rows, "validation_data_read": False, "ground_truth_scope": "registered TRAIN masks only"}


def _cache_receipt():
    audit = _module("collaborative_test.SW_0167_structured_relation_energy.audit_train_cache")
    result = {}
    for name, root in (("source", audit.source_cache.CACHE_ROOT),
                       ("context", audit.context_cache.CACHE_ROOT)):
        for relative in ("manifest.json", "ledger.json"):
            result[f"{name}/{relative}"] = _sha_file(root / relative)
        for index in range(COUNT):
            result[f"{name}/rows/{index:04d}.npz"] = _sha_file(root / "rows" / f"{index:04d}.npz")
    return result


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def train_arm(arm, *, physical_gpu, output_root=TRAIN_ROOT, max_updates=UPDATES):
    if arm not in ARMS or max_updates != UPDATES:
        raise ValueError("SW0168 arms and 64-update budget are fixed")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("SW0168 training worker is not on its assigned owner-held GPU")
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0168 CUDA initialized before full TRAIN/cache admission")
    caches = prepare_training_inputs_cpu()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu) or torch.cuda.is_initialized():
        raise RuntimeError("SW0168 CPU admission changed assigned device visibility")
    torch.set_num_threads(1)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    foundation_mod = _module("collaborative_test.SW_0135_native32_spike_binding.foundation")
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(149001)
        torch.cuda.manual_seed(149001)
        found = foundation_mod.load_native32_foundation(1, device, verify_rgb_assets=False)
    operator = _module("collaborative_test.SW_0143_native32_selective_geodesic.geodesic_operator")
    operator.install_operator(found.wrapped.core.graph_generator, "T")
    if _state_sha(found) != caches["source_state_sha256"]:
        raise ValueError("SW0168 loaded source foundation differs from admitted cache")
    families = _model_families(found)
    initial_diagnostics = _diagnose_eight(found, caches, phase="initial_source_model")
    runtime = _module("collaborative_test.SW_0148_checkpointable_boundary_q.checkpoint").runtime_signature(device)
    named_families = {
        "encoder": list(found.encoder.named_parameters()),
        "graph": [(name, p) for name, p in found.wrapped.core.graph_generator.named_parameters()
                  if name != "geodesic_contrast"],
        "native_core": [(name, p) for name, p in found.wrapped.core.named_parameters()
                        if not name.startswith("graph_generator.")],
        "integration": [(name, getattr(found.wrapped, name)) for name in ("a_d", "a_m", "b")],
    }
    optimizer_groups = []
    for name, params in families.items():
        named = named_families[name]
        if [id(p) for _, p in named] != [id(p) for p in params]:
            raise ValueError(f"SW0168 named parameter mapping changed for {name}")
        optimizer_groups.append(dict(params=params,
            parameter_names=[parameter_name for parameter_name, _ in named],
            lr=INTEGRATION_LR if name == "integration" else BASE_LR, name=name))
    optimizer = torch.optim.Adam(optimizer_groups)
    bindings = _training_bindings(caches["bindings"], arm)
    work = Path(output_root) / "work" / f"{arm}_seed1"
    output = Path(output_root) / f"{arm}_seed1"
    if output.exists():
        validate_training_artifact(arm, output, expected_bindings=bindings)
        return json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    durable = _module("collaborative_test.SW_0148_checkpointable_boundary_q.checkpoint")
    if (work / "boundary.ptx").is_file():
        loaded = durable.load_latest(work, wrapped=found.wrapped, encoder=found.encoder,
            optimizer=optimizer, expected_bindings=bindings, expected_runtime=runtime,
            device=device, max_updates=UPDATES)
        completed, history = loaded["completed_updates"], loaded["history"]
    else:
        if work.exists() and any(work.iterdir()):
            raise FileExistsError("preserving uncommitted SW0168 training artifacts")
        completed, history = 0, []
        durable.save_boundary(work, completed_updates=0, wrapped=found.wrapped,
            encoder=found.encoder, optimizer=optimizer, history=history,
            bindings=bindings, runtime=runtime, device=device)
    orders = _epoch_orders()
    if not (0 <= completed <= UPDATES and len(history) == completed):
        raise ValueError("SW0168 checkpoint cursor/history mismatch")
    if len(history) != completed:
        raise ValueError("SW0168 durable boundary has an invalid history length")
    found.encoder.train(); found.patcher.train(); found.wrapped.train()
    cache_before = _cache_receipt()
    for update in range(completed, UPDATES):
        epoch, position = divmod(update, COUNT // BATCH)
        indices = np.asarray(orders[epoch][position * BATCH:(position + 1) * BATCH], dtype=np.int64)
        torch.manual_seed(MODEL_SEED + update)
        torch.cuda.manual_seed(MODEL_SEED + update)
        row = _update(arm, found, optimizer, indices, caches, device)
        row.update({"update": update + 1, "epoch": epoch + 1,
                    "image_ids": [caches["source_ids"][int(i)] for i in indices]})
        history.append(row)
        durable.save_boundary(work, completed_updates=update + 1, wrapped=found.wrapped,
            encoder=found.encoder, optimizer=optimizer, history=history,
            bindings=bindings, runtime=runtime, device=device)
        if (update + 1) % 8 == 0:
            print(json.dumps({"arm": arm, "completed_updates": update + 1,
                              "mean_loss": row["mean_loss"]}, allow_nan=False), flush=True)
    if cache_before != _cache_receipt():
        raise RuntimeError("SW0168 immutable TRAIN source/context caches changed during training")
    final_diagnostics = _diagnose_eight(found, caches, phase="final_after_64_updates")
    final_state = _state_sha(found)
    state_dict = {"wrapped": {k: v.detach().cpu().clone() for k, v in found.wrapped.state_dict().items()},
                  "encoder": {k: v.detach().cpu().clone() for k, v in found.encoder.state_dict().items()}}
    model_state_sha = _state_sha_from_state(state_dict)
    runtime_fp = implementation_fingerprint()
    staging = output.with_name(output.name + f".stage-{uuid.uuid4().hex}")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        checkpoint_path, optimizer_path, history_path = (staging / "checkpoint.pt",
                                                          staging / "optimizer.pt",
                                                          staging / "history.json")
        initial_diagnostics_path = staging / "initial_train_diagnostics.json"
        final_diagnostics_path = staging / "final_train_diagnostics.json"
        shutil.copyfile(work / "boundary.ptx", staging / "boundary.ptx")
        with checkpoint_path.open("xb") as stream:
            torch.save({"arm": arm, "bindings": bindings, "wrapped": state_dict["wrapped"],
                        "encoder": state_dict["encoder"], "model_state_sha256": model_state_sha}, stream)
            stream.flush(); os.fsync(stream.fileno())
        with optimizer_path.open("xb") as stream:
            torch.save(optimizer.state_dict(), stream); stream.flush(); os.fsync(stream.fileno())
        _atomic_json(history_path, {"updates": history})
        _atomic_json(initial_diagnostics_path, initial_diagnostics)
        _atomic_json(final_diagnostics_path, final_diagnostics)
        summary = _optimizer_step_summary(optimizer)
        manifest = {"experiment": "SW0168_actual_spike_connectivity_training",
            "status": "training_complete", "arm": arm, "seed": 1,
            "completed_updates": UPDATES, "updates": UPDATES, "epochs": EPOCHS,
            "logical_batch": BATCH, "microbatch": MICRO, "exposures": UPDATES * BATCH,
            "train_ids": [caches["source_ids"][int(i)] for order in orders for i in order],
            "ground_truth_scope": "512 canonical TRAIN modal32 masks only",
            "bindings": bindings, "implementation_fingerprint": runtime_fp,
            "source_model_state_before_sha256": caches["source_state_sha256"],
            "final_model_state_sha256": final_state,
            "checkpoint_model_state_sha256": model_state_sha,
            "optimizer_step_summary": summary,
            "checkpoint_sha256": _sha_file(checkpoint_path),
            "optimizer_sha256": _sha_file(optimizer_path),
            "history_sha256": _sha_file(history_path),
            "initial_train_diagnostics_sha256": _sha_file(initial_diagnostics_path),
            "final_train_diagnostics_sha256": _sha_file(final_diagnostics_path),
            "boundary_sha256": _sha_file(staging / "boundary.ptx"),
            "source_cache_file_receipt_sha256": _json_sha(cache_before),
            "runtime": runtime}
        _atomic_json(staging / "manifest.json", manifest)
        validate_training_artifact(arm, staging, expected_bindings=bindings)
        os.rename(staging, output)
    except BaseException:
        # Preserve staging for explicit recovery/evidence.
        raise
    return manifest


def _state_sha_from_state(state):
    cache = _module("collaborative_test.SW_0161_supervised_spike_partition.cache")
    h = hashlib.sha256()
    for section in ("wrapped", "encoder"):
        for key, value in sorted(state[section].items()):
            h.update((section + "." + key).encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _optimizer_step_summary(optimizer):
    summary = {}
    for group in optimizer.param_groups:
        name = group.get("name")
        if not isinstance(name, str) or name in summary:
            raise ValueError("SW0168 optimizer family names must be unique")
        names = group.get("parameter_names")
        if not isinstance(names, list) or len(names) != len(group["params"]):
            raise ValueError("SW0168 optimizer parameter names are missing")
        updated = [(parameter_name, int(optimizer.state[p]["step"].item()))
                   for parameter_name, p in zip(names, group["params"]) if p in optimizer.state]
        steps = [step for _, step in updated]
        summary[name] = {"parameter_count": len(group["params"]),
                         "updated_parameter_count": len(steps),
                         "unused_parameter_count": len(group["params"]) - len(steps),
                         "updated_parameter_names": [parameter_name for parameter_name, _ in updated],
                         "unused_parameter_names": [parameter_name for parameter_name, p in zip(names, group["params"])
                                                     if p not in optimizer.state],
                         "steps": sorted(steps), "max_step": max(steps, default=0)}
    return summary


def _validate_optimizer_summary(summary):
    if not isinstance(summary, dict) or not summary:
        raise ValueError("SW0168 optimizer step summary is incomplete")
    for name in ("encoder", "graph", "native_core"):
        value = summary.get(name)
        if (not isinstance(value, dict) or value.get("max_step") != UPDATES
                or value.get("updated_parameter_count", 0) < 1):
            raise ValueError(f"SW0168 critical optimizer family {name} lacks 64-step credit")
    for name, value in summary.items():
        steps = value.get("steps") if isinstance(value, dict) else None
        counts = (value.get("parameter_count"), value.get("updated_parameter_count"),
                  value.get("unused_parameter_count")) if isinstance(value, dict) else ()
        if (not isinstance(steps, list) or any(type(step) is not int or step < 1 or step > UPDATES
                                               for step in steps)
                or any(type(count) is not int or count < 0 for count in counts)
                or value.get("max_step") != max(steps, default=0)
                or value.get("updated_parameter_count") != len(steps)
                or value.get("unused_parameter_count", -1) + len(steps) != value.get("parameter_count")):
            raise ValueError(f"SW0168 optimizer family {name} has invalid per-parameter steps")


def validate_training_artifact(arm, directory=None, *, expected_bindings=None):
    if arm not in ARMS:
        raise ValueError("unregistered SW0168 training arm")
    root = Path(directory) if directory is not None else TRAIN_ROOT / f"{arm}_seed1"
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bindings = manifest.get("bindings", {})
    if (manifest.get("experiment") != "SW0168_actual_spike_connectivity_training"
            or manifest.get("status") != "training_complete" or manifest.get("arm") != arm
            or manifest.get("seed") != 1 or manifest.get("completed_updates") != UPDATES
            or manifest.get("updates") != UPDATES or manifest.get("epochs") != EPOCHS
            or manifest.get("logical_batch") != BATCH or manifest.get("microbatch") != MICRO
            or manifest.get("exposures") != COUNT * EPOCHS
            or manifest.get("ground_truth_scope") != "512 canonical TRAIN modal32 masks only"
            or bindings.get("arm") != arm
            or (expected_bindings is not None and bindings != expected_bindings)
            or manifest.get("implementation_fingerprint") != implementation_fingerprint()):
        raise ValueError("SW0168 training manifest/provenance schema mismatch")
    for key, name in (("checkpoint_sha256", "checkpoint.pt"),
                      ("optimizer_sha256", "optimizer.pt"),
                      ("history_sha256", "history.json"),
                      ("initial_train_diagnostics_sha256", "initial_train_diagnostics.json"),
                      ("final_train_diagnostics_sha256", "final_train_diagnostics.json"),
                      ("boundary_sha256", "boundary.ptx")):
        if _sha_file(root / name) != manifest.get(key):
            raise ValueError(f"SW0168 artifact bytes mismatch: {name}")
    history = json.loads((root / "history.json").read_text(encoding="utf-8")).get("updates")
    if not isinstance(history, list) or len(history) != UPDATES:
        raise ValueError("SW0168 history must contain exactly 64 logical updates")
    if [row.get("update") for row in history] != list(range(1, UPDATES + 1)):
        raise ValueError("SW0168 history update cursor is incomplete")
    for epoch in range(EPOCHS):
        ids = manifest["train_ids"][epoch * COUNT:(epoch + 1) * COUNT]
        if len(ids) != COUNT or sorted(ids) != sorted(bindings.get("ordered_train_ids", [])):
            raise ValueError("SW0168 exposure does not cover the registered 512 TRAIN IDs exactly")
    expected_orders = _epoch_orders()
    for update, row in enumerate(history):
        epoch, position = divmod(update, COUNT // BATCH)
        expected_positions = expected_orders[epoch][position * BATCH:(position + 1) * BATCH]
        expected_ids = [bindings["ordered_train_ids"][i] for i in expected_positions]
        if (row.get("image_indices") != expected_positions
                or row.get("image_ids") != expected_ids
                or row.get("images_seen") != BATCH
                or not np.isfinite(float(row.get("gradient_norm", np.nan)))
                or float(row["gradient_norm"]) < 0):
            raise ValueError(f"SW0168 update {update + 1} has invalid TRAIN exposure/order")
    if any(not np.isfinite(float(row.get("mean_loss"))) for row in history):
        raise ValueError("SW0168 history contains nonfinite objective")
    for name, expected_phase in (("initial_train_diagnostics.json", "initial_source_model"),
                                 ("final_train_diagnostics.json", "final_after_64_updates")):
        diagnostic = json.loads((root / name).read_text(encoding="utf-8"))
        if (diagnostic.get("phase") != expected_phase
                or diagnostic.get("validation_data_read") is not False
                or diagnostic.get("selected_image_ids") != bindings.get("ordered_train_ids", [])[:8]
                or len(diagnostic.get("rows", [])) != 8
                or diagnostic.get("ground_truth_scope") != "registered TRAIN masks only"):
            raise ValueError(f"SW0168 {expected_phase} diagnostic contract is invalid")
        for row in diagnostic["rows"]:
            if (row.get("image_id") not in diagnostic["selected_image_ids"]
                    or not isinstance(row.get("q_sha256"), str) or len(row["q_sha256"]) != 64
                    or not isinstance(row.get("actual_a_sha256"), str) or len(row["actual_a_sha256"]) != 64
                    or not np.isfinite(float(row.get("event_on_fraction", np.nan)))
                    or not 0 <= float(row["event_on_fraction"]) <= 1
                    or not np.isfinite(float(row.get("event_transition_fraction", np.nan)))
                    or not 0 <= float(row["event_transition_fraction"]) <= 1
                    or not np.isfinite(float(row.get("gate_positive_fraction", np.nan)))
                    or not 0 <= float(row["gate_positive_fraction"]) <= 1
                    or len(row.get("component_product_clamp_active_fraction", [])) != 4
                    or any(not np.isfinite(float(v)) or not 0 <= float(v) <= 1
                           for v in row["component_product_clamp_active_fraction"])):
                raise ValueError(f"SW0168 {expected_phase} diagnostic row is malformed")
            for loss_name in ("constrained_maximin_loss", "matched_local_edge_loss"):
                if not np.isfinite(float(row.get(loss_name, np.nan))):
                    raise ValueError(f"SW0168 {expected_phase} diagnostic loss is nonfinite")
            if (not isinstance(row.get("represented_positive_pair_mass"), int)
                    or row["represented_positive_pair_mass"] < 0
                    or not isinstance(row.get("represented_negative_pair_mass"), int)
                    or row["represented_negative_pair_mass"] < 0
                    or not isinstance(row.get("excluded_unreachable_negative_pair_mass"), int)
                    or row["excluded_unreachable_negative_pair_mass"] < 0):
                raise ValueError(f"SW0168 {expected_phase} pair accounting is invalid")
    payload = torch.load(root / "checkpoint.pt", map_location="cpu", weights_only=False)
    optimizer_payload = torch.load(root / "optimizer.pt", map_location="cpu", weights_only=False)
    if payload.get("arm") != arm or payload.get("bindings") != bindings:
        raise ValueError("SW0168 checkpoint lineage mismatch")
    if _state_sha_from_state({"wrapped": payload["wrapped"], "encoder": payload["encoder"]}) != manifest.get("checkpoint_model_state_sha256"):
        raise ValueError("SW0168 checkpoint model digest mismatch")
    summary = manifest.get("optimizer_step_summary", {})
    actual_summary = {}
    for group in optimizer_payload["param_groups"]:
        name = group.get("name")
        if not isinstance(name, str) or name in actual_summary:
            raise ValueError("SW0168 optimizer family names are invalid")
        parameter_names = group.get("parameter_names")
        if not isinstance(parameter_names, list) or len(parameter_names) != len(group["params"]):
            raise ValueError("SW0168 optimizer group parameter names are missing")
        updated = [(parameter_name, int(optimizer_payload["state"][parameter_id]["step"].item()))
                   for parameter_name, parameter_id in zip(parameter_names, group["params"])
                   if parameter_id in optimizer_payload["state"]]
        group_steps = [step for _, step in updated]
        actual_summary[name] = {"parameter_count": len(group["params"]),
                                "updated_parameter_count": len(group_steps),
                                "unused_parameter_count": len(group["params"]) - len(group_steps),
                                "updated_parameter_names": [name for name, _ in updated],
                                "unused_parameter_names": [name for name, parameter_id in
                                    zip(parameter_names, group["params"])
                                    if parameter_id not in optimizer_payload["state"]],
                                "steps": sorted(group_steps), "max_step": max(group_steps, default=0)}
    if summary != actual_summary:
        raise ValueError("SW0168 optimizer step summary is incomplete")
    _validate_optimizer_summary(actual_summary)
    boundary_outer = torch.load(root / "boundary.ptx", map_location="cpu", weights_only=False)
    payload_bytes = boundary_outer.get("payload_bytes")
    if (not isinstance(payload_bytes, bytes)
            or _sha_bytes(payload_bytes) != boundary_outer.get("payload_sha256")):
        raise ValueError("SW0168 durable boundary container is corrupt")
    boundary = torch.load(__import__("io").BytesIO(payload_bytes),
                          map_location="cpu", weights_only=False)
    durable = _module("collaborative_test.SW_0148_checkpointable_boundary_q.checkpoint")
    if (boundary.get("completed_updates") != UPDATES
            or boundary.get("bindings") != bindings
            or boundary.get("bindings_sha256") != durable.canonical_sha(bindings)
            or boundary.get("history") != history
            or boundary.get("wrapped_state") is None or boundary.get("encoder_state") is None):
        raise ValueError("SW0168 final durable boundary lineage/cursor mismatch")
    for section, saved in (("wrapped", boundary["wrapped_state"]),
                           ("encoder", boundary["encoder_state"])):
        if set(saved) != set(payload[section]) or any(
                not torch.equal(saved[name].cpu(), payload[section][name].cpu()) for name in saved):
            raise ValueError("SW0168 final checkpoint differs from durable boundary model state")
    boundary_opt = boundary.get("optimizer_state", {})
    for key in ("param_groups", "state"):
        if key not in boundary_opt or key not in optimizer_payload:
            raise ValueError("SW0168 final durable boundary lacks optimizer state")
    if len(boundary_opt["state"]) != len(optimizer_payload["state"]):
        raise ValueError("SW0168 boundary/optimizer parameter state counts differ")
    for parameter_id, state in optimizer_payload["state"].items():
        other = boundary_opt["state"].get(parameter_id)
        if other is None or set(other) != set(state):
            raise ValueError("SW0168 boundary/optimizer state keys differ")
        for name, value in state.items():
            expected = other[name]
            if torch.is_tensor(value):
                if not torch.equal(value.cpu(), expected.cpu()):
                    raise ValueError("SW0168 final checkpoint optimizer differs from durable boundary")
            elif value != expected:
                raise ValueError("SW0168 final checkpoint optimizer differs from durable boundary")
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args()
    train_arm(args.arm, physical_gpu=args.physical_gpu)

