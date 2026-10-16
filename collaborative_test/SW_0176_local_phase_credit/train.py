"""SW0176 matched local-phase full-window training from a fresh source model."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

import numpy as np  # Initialize NumPy/OpenMP before torch on Windows.
import torch

from collaborative_test.SW_0175_critical_component_binding import train_v2 as base
from collaborative_test.SW_0176_local_phase_credit import preflight_v4
from collaborative_test.SW_0176_local_phase_credit.policy import detach_previous_theta

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENT = "SW0176_local_phase_fullwindow_training"
ARM = "local_phase"
SEED = 1
OUTPUT_ROOT = HERE / "results_archive" / "training_v4_20261016"
TRAIN_COUNT, PASSES, UPDATES = 512, 2, 64
BATCH, MICRO = 16, 4
MODEL_SEED = 171001
CONTEXT_SEED = 171001
SHUFFLE_SEED = 171001
STEPS, SETTLE = 1024, 512
LIVE_TAIL = 1024


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha(value):
    return _sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode("utf-8"))


def _json_safe_optimizer_value(value):
    """Normalize torch-save values and fresh Python values to one JSON shape."""
    if isinstance(value, tuple):
        return [_json_safe_optimizer_value(item) for item in value]
    if isinstance(value, list):
        return [_json_safe_optimizer_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe_optimizer_value(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def _source_files():
    return (
        "collaborative_test/SW_0176_local_phase_credit/DESIGN_KO.md",
        "collaborative_test/SW_0176_local_phase_credit/policy.py",
        "collaborative_test/SW_0176_local_phase_credit/preflight_v4.py",
        "collaborative_test/SW_0176_local_phase_credit/resource_queue_v4.py",
        "collaborative_test/SW_0176_local_phase_credit/train.py",
        "collaborative_test/SW_0176_local_phase_credit/training_queue.py",
        "collaborative_test/SW_0176_local_phase_credit/test_policy.py",
        "collaborative_test/SW_0176_local_phase_credit/test_preflight_v4.py",
        "collaborative_test/SW_0176_local_phase_credit/test_resource_queue_v4.py",
        "collaborative_test/SW_0176_local_phase_credit/test_train.py",
        "collaborative_test/SW_0176_local_phase_credit/test_training_queue.py",
        "collaborative_test/SW_0175_critical_component_binding/DESIGN_KO.md",
        "collaborative_test/SW_0175_critical_component_binding/train_v2.py",
        "collaborative_test/SW_0175_critical_component_binding/critical_component_objective.py",
        "collaborative_test/SW_0171_contextual_instance_graph/contextual_encoder.py",
        "collaborative_test/SW_0169_normalized_hard_reset/calibration_v2.py",
        "collaborative_test/SW_0134_native_spike_binding/rollout.py",
        "collaborative_test/SW_0135_native32_spike_binding/foundation.py",
        "collaborative_test/SW_0136_native32_transfer/evaluate.py",
        "collaborative_test/SW_0143_native32_selective_geodesic/geodesic_operator.py",
        "collaborative_test/SW_0148_checkpointable_boundary_q/checkpoint.py",
        "collaborative_test/SW_0161_supervised_spike_partition/cache.py",
        "collaborative_test/SW_0162_connected_signed_partition/preflight.py",
        "collaborative_test/SW_0164_foreground_preserving_partition/foreground_signed.py",
        "collaborative_test/SW_0168_end_to_end_connectivity/connectivity_objective.py",
        "snn_kuramoto_bidirectional/graph_generator.py",
        "snn_kuramoto_bidirectional/spike_classifier.py",
        "snn_kuramoto_bidirectional/kuramoto_layer.py",
    )


def implementation_fingerprint():
    values = {}
    for relative in _source_files():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"SW0176 trainer dependency missing: {relative}")
        values[relative] = _sha_file(path)
    return {"files": values, "experiment": EXPERIMENT, "seed": SEED,
        "source_control_experiment": base.EXPERIMENT,
        "training_images_per_pass": TRAIN_COUNT, "passes": PASSES,
        "updates": UPDATES, "logical_batch": BATCH, "microbatch": MICRO,
        "model_seed": MODEL_SEED, "context_seed": CONTEXT_SEED,
        "shuffle_seed": SHUFFLE_SEED, "steps": STEPS, "settle": SETTLE,
        "live_tail_steps": LIVE_TAIL, "optimizer": "Adam",
        "learning_rates": {"context_encoder": base.ENCODER_LR,
            "graph": base.GRAPH_CORE_LR, "native_core": base.GRAPH_CORE_LR,
            "integration": base.INTEGRATION_LR},
        "global_clip_norm": base.CLIP_NORM,
        "objective": "SW0175 critical_component_loss",
        "gradient_intervention": "detach_only_theta_prev_each_kuramoto_transition"}


def _control_artifact_receipt(manifest):
    control_root = base.OUTPUT_ROOT / f"{base.ARMS[1]}_seed1"
    base.validate_training_artifact(base.ARMS[1], control_root)
    files = ("manifest.json", "checkpoint.pt", "optimizer.pt", "history.json",
             "boundary.ptx", "initial_train8.json", "final_train8.json")
    hashes = {name: _sha_file(control_root / name) for name in files}
    if hashes["manifest.json"] != preflight_v4._sha_file(control_root / "manifest.json"):
        raise ValueError("SW0176 control manifest changed during receipt creation")
    if manifest.get("checkpoint_sha256") != hashes["checkpoint.pt"]:
        raise ValueError("SW0176 control checkpoint receipt mismatch")
    saved_optimizer = torch.load(control_root / "optimizer.pt", map_location="cpu", weights_only=False)
    group_contract = [{key: _json_safe_optimizer_value(group.get(key)) for key in
        ("name", "parameter_names", "lr", "betas", "eps", "weight_decay", "amsgrad",
         "maximize", "foreach", "capturable", "differentiable", "fused")}
        for group in saved_optimizer.get("param_groups", [])]
    if not group_contract or any(not isinstance(group.get("parameter_names"), list)
                                 or not group["parameter_names"] for group in group_contract):
        raise ValueError("SW0176 SW0175 control optimizer lacks named ordered groups")
    return {"experiment": base.EXPERIMENT, "arm": base.ARMS[1], "seed": 1,
        "manifest_sha256": hashes["manifest.json"], "artifact_file_sha256": hashes,
        "initial_model_state_sha256": manifest["bindings"]["initial_model_state_sha256"],
        "optimizer_group_contract": group_contract,
        "training_fingerprint_sha256": _json_sha(manifest["training_fingerprint"]),
        "ordered_train_ids_sha256": manifest["bindings"]["ordered_train_ids_sha256"],
        "updates": 64, "logical_batch": 16, "microbatch": 4}


def _optimizer_group_contract(optimizer):
    keys = ("name", "parameter_names", "lr", "betas", "eps", "weight_decay", "amsgrad",
            "maximize", "foreach", "capturable", "differentiable", "fused")
    result = []
    for group in optimizer.param_groups:
        names = group.get("parameter_names")
        if (not isinstance(group.get("name"), str) or not isinstance(names, list)
                or len(names) != len(group["params"]) or not all(isinstance(n, str) for n in names)):
            raise ValueError("SW0176 fresh Adam groups lack registered parameter names/order")
        row = {key: group.get(key) for key in keys}
        row = _json_safe_optimizer_value(row)
        result.append(row)
    return result


def _validated_source_preflight():
    prepared = preflight_v4.prepare_inputs_cpu()
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0176 input admission initialized CUDA before the leased worker")
    queue = __import__("collaborative_test.SW_0176_local_phase_credit.resource_queue_v4",
                       fromlist=["RUNTIME"])
    state_path = queue.RUNTIME / "state.json"
    report_path = queue.RUNTIME / "preflight.json"
    if not state_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("SW0176 requires the immutable v4 source-preflight result")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_sha = _sha_file(report_path)
    fingerprint = preflight_v4.implementation_fingerprint()
    proof = prepared["source_proof"]
    report_status = report.get("status")
    expected_queue_status = {"complete_source_preflight": "complete",
                             "scientific_rejection": "complete_rejected"}.get(report_status)
    if (expected_queue_status is None or state.get("status") != expected_queue_status
            or state.get("report_file_sha256") != report_sha
            or state.get("source_proof") != proof
            or state.get("implementation_fingerprint") != fingerprint
            or report.get("source_proof") != proof
            or report.get("implementation_fingerprint") != fingerprint
            or report.get("optimizer_updates") != 0
            or report.get("validation_data_read") is not False
            or not isinstance(report.get("forward_parity"), dict)
            or not all(report["forward_parity"].values())):
        raise ValueError("SW0176 v4 source-preflight receipt failed strict validation")
    if report_status == "scientific_rejection" and not report.get("rejection_reasons"):
        raise ValueError("SW0176 rejected preflight lacks a scientific reason")
    control_sha = _sha_file(base.OUTPUT_ROOT / f"{base.ARMS[1]}_seed1" / "manifest.json")
    if report.get("control_training_manifest_sha256") != control_sha:
        raise ValueError("SW0176 source preflight does not bind the current immutable tail control")
    control_receipt = _control_artifact_receipt(prepared["control_manifest"])
    return {"prepared": prepared, "source_proof": proof,
        "preflight_report": report, "preflight_report_sha256": report_sha,
        "preflight_state_sha256": _sha_file(state_path),
        "preflight_fingerprint": fingerprint, "control_receipt": control_receipt,
        "implementation_fingerprint": implementation_fingerprint()}


def prepare_training_inputs_cpu():
    """Public no-CUDA admission shared by the parent and leased worker."""
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0176 training CPU admission must precede CUDA initialization")
    admitted = _validated_source_preflight()
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0176 CPU source/control admission unexpectedly initialized CUDA")
    return admitted


def prepare_fresh_training_state_cpu(admitted=None):
    """Build the real fresh Source-T/context/Adam factory without CUDA or updates.

    This validates initialization, frozen-source lineage, parameter grouping,
    and optimizer settings against the completed SW0175 critical control before
    a leased worker initializes CUDA. The resulting models are discarded.
    """
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0176 fresh factory validation must precede CUDA initialization")
    admitted = prepare_training_inputs_cpu() if admitted is None else admitted
    if admitted["preflight_report"].get("status") != "complete_source_preflight":
        raise RuntimeError("SW0176 rejected source preflight cannot construct a training model")
    prepared = admitted["prepared"]
    found, base_encoder, source_sha, base_encoder_sha, init_sha, families = \
        preflight_v4._build_source(prepared, "cpu")
    optimizer, families = base._make_optimizer(found)
    initial_sha = base._state_sha(found.wrapped, found.encoder)
    _control_bindings(admitted, initial_sha, source_sha, base_encoder_sha,
                      families, optimizer)
    if initial_sha != init_sha:
        raise ValueError("SW0176 CPU fresh factory changed during optimizer construction")
    if optimizer.state:
        raise ValueError("SW0176 fresh Adam must have no state before the first update")
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0176 CPU fresh factory unexpectedly initialized CUDA")
    receipt = {"status": "complete_fresh_cpu_factory", "source_proof": admitted["source_proof"],
        "preflight_report_sha256": admitted["preflight_report_sha256"],
        "control_receipt_sha256": admitted["control_receipt"]["manifest_sha256"],
        "source_state_sha256": source_sha, "base_encoder_sha256": base_encoder_sha,
        "initial_model_state_sha256": initial_sha,
        "optimizer_group_contract": _optimizer_group_contract(optimizer),
        "optimizer_state_entries": len(optimizer.state),
        "cuda_initialized": False, "validation_data_read": False, "optimizer_updates": 0}
    admitted["fresh_cpu_factory_receipt"] = receipt
    receipt["receipt_sha256"] = _json_sha({key: value for key, value in receipt.items()
                                           if key != "receipt_sha256"})
    return receipt


def _expected_batches():
    # Use the exact registered NumPy two-permutation schedule from SW0175.
    return base._expected_batches()


def _validate_cursor(completed, history):
    base._validate_cursor(completed, history)
    if completed > UPDATES:
        raise ValueError("SW0176 committed cursor exceeds its registered 64 updates")


def _atomic_json_create(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_replace(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_rejection(work, *, completed, history, bindings, reasons):
    path = Path(work) / "credit_rejection.json"
    report = {"experiment": EXPERIMENT, "status": "scientific_rejection",
        "arm": ARM, "completed_updates": completed,
        "history_sha256": _json_sha(history), "bindings_sha256": _json_sha(bindings),
        "boundary_sha256": _sha_file(Path(work) / "boundary.ptx"),
        "source_proof": bindings["source_proof"],
        "training_fingerprint": bindings["training_fingerprint"],
        "rejection_reasons": list(reasons), "validation_data_read": False}
    _atomic_json_create(path, report)
    return report


def _validate_rejection(work, *, completed, history, bindings):
    path = Path(work) / "credit_rejection.json"
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    expected = {"experiment": EXPERIMENT, "status": "scientific_rejection",
        "arm": ARM, "completed_updates": completed,
        "history_sha256": _json_sha(history), "bindings_sha256": _json_sha(bindings),
        "boundary_sha256": _sha_file(Path(work) / "boundary.ptx"),
        "source_proof": bindings["source_proof"],
        "training_fingerprint": bindings["training_fingerprint"],
        "validation_data_read": False}
    if any(report.get(key) != value for key, value in expected.items()) or not report.get("rejection_reasons"):
        raise ValueError("SW0176 scientific rejection does not bind the committed state")
    return report


def _resume_rejection(work, completed, history, bindings):
    saved = _validate_rejection(work, completed=completed, history=history, bindings=bindings)
    if saved is not None:
        return saved
    for row in history[:2]:
        reasons = base._credit_rejection(row)
        if reasons:
            return _write_rejection(work, completed=int(row["update"]), history=history,
                                    bindings=bindings, reasons=reasons)
    return None


def _train_update(found, optimizer, indices, admitted, device, *, update_number):
    prepared = admitted["prepared"]
    arrays = prepared["arrays"]
    base_prepared = prepared["admitted"]["prepared"]
    optimizer.zero_grad(set_to_none=True)
    before = {group["name"]: [parameter.detach().clone() for parameter in group["params"]]
              for group in optimizer.param_groups}
    total_losses, q_losses, micro_records = [], [], []
    q_to_s_squared = 0.0
    total_policy_calls = 0
    for offset in range(0, BATCH, MICRO):
        ids = indices[offset:offset + MICRO]
        rgb = torch.from_numpy(np.ascontiguousarray(arrays["rgb"][ids])).to(device)
        with detach_previous_theta(found.wrapped.core.kuramoto) as policy_calls:
            _gamma, trace, adjacency, z, q, rho = base._native_forward(
                found, rgb, live_tail_steps=LIVE_TAIL)
        total_policy_calls += int(policy_calls["count"])
        if policy_calls["count"] != STEPS:
            raise RuntimeError("SW0176 policy did not intercept every transition in a full rollout")
        micro_losses = []
        for local in range(MICRO):
            index = int(ids[local])
            target = arrays["feasible_targets"][index]
            support = arrays["source_support"][index]
            q_loss = base._native_q_loss(q[local], target, support)
            total = base._loss_for_arm("critical_component", q[local], rho[local],
                                       target, support)
            if (not bool(torch.isfinite(q_loss)) or not bool(torch.isfinite(total))):
                raise FloatingPointError(f"SW0176 nonfinite objective at TRAIN position {index}")
            if update_number <= 2:
                gradient = torch.autograd.grad(total, trace["component_spikes"],
                    retain_graph=True, allow_unused=True)[0]
                if gradient is not None:
                    if not bool(torch.isfinite(gradient).all()):
                        raise FloatingPointError("SW0176 loss-to-emitted-spike gradient is nonfinite")
                    q_to_s_squared += float(gradient.detach().double().square().sum().cpu())
            micro_losses.append(total)
            total_losses.append(float(total.detach().cpu()))
            q_losses.append(float(q_loss.detach().cpu()))
        (torch.stack(micro_losses).sum() / BATCH).backward()
        micro_records.append({"indices": list(map(int, ids)),
            "image_ids": [int(base_prepared["cache_proof"]["image_ids"][int(i)]) for i in ids],
            "q_sha256": base._tensor_sha(q), "projection_sha256": base._tensor_sha(z),
            "theta_prev_detach_calls": int(policy_calls["count"])})
        del rgb, trace, adjacency, z, q, rho, micro_losses

    raw_by_parameter, family_norms = {}, {}
    all_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    for group in optimizer.param_groups:
        square = 0.0
        for name, parameter in zip(group["parameter_names"], group["params"]):
            if parameter.grad is None:
                continue
            if not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError(f"SW0176 nonfinite gradient {group['name']}/{name}")
            norm = float(parameter.grad.detach().double().norm().cpu())
            raw_by_parameter[id(parameter)] = norm
            square += norm * norm
        family_norms[group["name"]] = math.sqrt(square)
    total_norm = math.sqrt(sum(value * value for value in family_norms.values()))
    clipped = torch.nn.utils.clip_grad_norm_(all_parameters, base.CLIP_NORM)
    if not bool(torch.isfinite(clipped)):
        raise FloatingPointError("SW0176 global gradient norm is nonfinite")
    optimizer.step()
    found.wrapped.project_integrations_()

    family_deltas, named_credit = {}, {}
    for group in optimizer.param_groups:
        delta_sq = 0.0
        rows = []
        for name, old, parameter in zip(group["parameter_names"], before[group["name"]], group["params"]):
            delta = float((parameter.detach().double() - old.detach().double()).norm().cpu())
            if not math.isfinite(delta):
                raise FloatingPointError("SW0176 parameter delta is nonfinite")
            delta_sq += delta * delta
            rows.append({"name": name, "gradient_l2": raw_by_parameter.get(id(parameter), 0.0),
                "gradient_present": parameter.grad is not None, "parameter_delta_l2": delta})
        family_deltas[group["name"]] = math.sqrt(delta_sq)
        named_credit[group["name"]] = rows
    cache_ids = base_prepared["cache_proof"]["image_ids"]
    return {"update": int(update_number), "pass": (update_number - 1) // 32 + 1,
        "image_indices": list(map(int, indices)),
        "image_ids": [int(cache_ids[int(index)]) for index in indices],
        "images_seen": BATCH, "microbatch": MICRO,
        "mean_loss": float(np.mean(total_losses)),
        "mean_q_loss": float(np.mean(q_losses)),
        "mean_registered_loss": float(np.mean(total_losses)),
        "gradient_norm_before_clip": total_norm,
        "clip_scale": min(1.0, base.CLIP_NORM / (float(clipped.detach().cpu()) + 1e-12)),
        "family_gradient_norms": family_norms,
        "family_parameter_delta_l2": family_deltas,
        "named_parameter_credit": named_credit, "microbatches": micro_records,
        "policy_theta_prev_detach_calls": total_policy_calls,
        "registered_loss_to_component_spikes_gradient_l2": math.sqrt(q_to_s_squared)
            if update_number <= 2 else None}


def _control_bindings(admitted, initial_state_sha, source_sha, base_encoder_sha,
                      families, optimizer):
    control = admitted["prepared"]["control_manifest"]
    expected_initial = control["bindings"]["initial_model_state_sha256"]
    if initial_state_sha != expected_initial:
        raise ValueError("fresh SW0176 source/context initialization differs from SW0175 control")
    if control["bindings"]["source_state_sha256"] != source_sha:
        raise ValueError("SW0176 fresh source model differs from SW0175 tail-control source")
    if control["bindings"]["base_encoder_sha256"] != base_encoder_sha:
        raise ValueError("SW0176 base encoder differs from matched SW0175 control")
    control_contract = admitted["control_receipt"].get("optimizer_group_contract")
    fresh_contract = _optimizer_group_contract(optimizer)
    if fresh_contract != control_contract:
        raise ValueError("SW0176 fresh Adam group names/order/defaults differ from SW0175 control")
    family_names = {name: [n for n, _ in rows] for name, rows in families.items()}
    if {group["name"]: group["parameter_names"] for group in fresh_contract} != family_names:
        raise ValueError("SW0176 optimizer groups and fresh trainable families disagree")


def _make_bindings(admitted, initial_state_sha, source_sha, base_encoder_sha, families, optimizer):
    _control_bindings(admitted, initial_state_sha, source_sha, base_encoder_sha, families, optimizer)
    control = admitted["control_receipt"]
    prepared = admitted["prepared"]
    proof = admitted["source_proof"]
    cpu_factory = admitted.get("fresh_cpu_factory_receipt")
    if not isinstance(cpu_factory, dict) or cpu_factory.get("status") != "complete_fresh_cpu_factory":
        raise ValueError("SW0176 bindings require a completed no-CUDA fresh factory receipt")
    if cpu_factory.get("receipt_sha256") != _json_sha({key: value for key, value in cpu_factory.items()
                                                         if key != "receipt_sha256"}):
        raise ValueError("SW0176 CPU fresh factory receipt digest differs")
    if (cpu_factory.get("source_state_sha256") != source_sha
            or cpu_factory.get("base_encoder_sha256") != base_encoder_sha
            or cpu_factory.get("initial_model_state_sha256") != initial_state_sha
            or cpu_factory.get("optimizer_group_contract") != _optimizer_group_contract(optimizer)):
        raise ValueError("SW0176 GPU fresh initialization differs from the CPU factory proof")
    ids = list(map(int, prepared["arrays"]["image_ids"]))
    return {"experiment": EXPERIMENT, "arm": ARM, "seed": SEED,
        "source_proof": proof, "source_state_sha256": source_sha,
        "source_checkpoint_sha256": proof["source_checkpoint_sha256"],
        "source_manifest_sha256": proof["source_manifest_sha256"],
        "base_encoder_sha256": base_encoder_sha,
        "input_cache_manifest_sha256": proof["producer_manifest_sha256"],
        "input_cache_arrays_sha256": proof["producer_arrays_sha256"],
        "ordered_train_ids": ids,
        "ordered_train_ids_sha256": _sha_bytes(np.asarray(ids, dtype="<i8").tobytes()),
        "control_receipt": control, "control_receipt_sha256": _json_sha(control),
        "cpu_fresh_factory_receipt": cpu_factory,
        "cpu_fresh_factory_receipt_sha256": cpu_factory["receipt_sha256"],
        "control_initial_model_state_sha256": control["initial_model_state_sha256"],
        "preflight_report_sha256": admitted["preflight_report_sha256"],
        "preflight_state_sha256": admitted["preflight_state_sha256"],
        "preflight_fingerprint": admitted["preflight_fingerprint"],
        "initial_model_state_sha256": initial_state_sha,
        "optimizer_parameter_names": {name: [n for n, _ in rows] for name, rows in families.items()},
        "optimizer_group_contract": _optimizer_group_contract(optimizer),
        "training_images_per_pass": TRAIN_COUNT, "passes": PASSES, "updates": UPDATES,
        "logical_batch": BATCH, "microbatch": MICRO, "model_seed": MODEL_SEED,
        "context_seed": CONTEXT_SEED, "shuffle_seed": SHUFFLE_SEED,
        "optimizer": "Adam", "learning_rates": {"context_encoder": base.ENCODER_LR,
            "graph": base.GRAPH_CORE_LR, "native_core": base.GRAPH_CORE_LR,
            "integration": base.INTEGRATION_LR}, "global_clip_norm": base.CLIP_NORM,
        "steps": STEPS, "settle": SETTLE, "live_tail_steps": LIVE_TAIL,
        "gradient_intervention": "detach_only_theta_prev_each_kuramoto_transition",
        "objective": "SW0175 critical_component_loss",
        "training_fingerprint": implementation_fingerprint()}


def validate_training_artifact(root, *, expected_bindings=None):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("experiment") != EXPERIMENT or manifest.get("status") != "training_complete"
            or manifest.get("arm") != ARM or manifest.get("seed") != SEED
            or manifest.get("completed_updates") != UPDATES or manifest.get("updates") != UPDATES
            or manifest.get("validation_data_read") is not False):
        raise ValueError("SW0176 training artifact status/cursor/scope is invalid")
    bindings = manifest.get("bindings")
    if not isinstance(bindings, dict) or (expected_bindings is not None and bindings != expected_bindings):
        raise ValueError("SW0176 training artifact bindings differ")
    if manifest.get("training_fingerprint") != bindings.get("training_fingerprint"):
        raise ValueError("SW0176 manifest training fingerprint differs from bindings")
    history = json.loads((root / "history.json").read_text(encoding="utf-8")).get("updates")
    _validate_cursor(len(history) if isinstance(history, list) else -1, history)
    for filename in ("checkpoint.pt", "optimizer.pt", "history.json", "boundary.ptx",
                     "control_receipt.json"):
        if _sha_file(root / filename) != manifest.get("files_sha256", {}).get(filename):
            raise ValueError(f"SW0176 artifact file digest differs: {filename}")
    checkpoint = torch.load(root / "checkpoint.pt", map_location="cpu", weights_only=False)
    if (checkpoint.get("experiment") != EXPERIMENT or checkpoint.get("bindings") != bindings
            or checkpoint.get("model_state_sha256") != manifest.get("final_model_state_sha256")):
        raise ValueError("SW0176 published model checkpoint lineage/hash differs")
    control_receipt = json.loads((root / "control_receipt.json").read_text(encoding="utf-8"))
    if (control_receipt != bindings.get("control_receipt")
            or _json_sha(control_receipt) != bindings.get("control_receipt_sha256")):
        raise ValueError("SW0176 published control receipt differs from model bindings")
    boundary = __import__("collaborative_test.SW_0148_checkpointable_boundary_q.checkpoint",
                          fromlist=["inspect_committed_metadata"])
    metadata = boundary.inspect_committed_metadata(root)
    if metadata.get("completed_updates") != UPDATES or metadata.get("bindings") != bindings:
        raise ValueError("SW0176 durable final boundary differs from published manifest")
    rejection = manifest.get("rejection")
    if rejection is not None:
        raise ValueError("SW0176 complete training artifact unexpectedly carries a rejection")
    return manifest


def _publish_training(output, work, found, optimizer, history, bindings, runtime,
                      source_sha, base_encoder_sha, frozen_before, frozen_after, cache_before):
    output = Path(output)
    if output.exists():
        raise FileExistsError("preserving existing SW0176 completed training artifact")
    staging = output.with_name(output.name + f".stage-{uuid.uuid4().hex}")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        checkpoint = {"experiment": EXPERIMENT, "arm": ARM, "bindings": bindings,
            "wrapped": {k: v.detach().cpu().clone() for k, v in found.wrapped.state_dict().items()},
            "encoder": {k: v.detach().cpu().clone() for k, v in found.encoder.state_dict().items()}}
        checkpoint["model_state_sha256"] = base._state_sha(found.wrapped, found.encoder)
        with (staging / "checkpoint.pt").open("xb") as stream:
            torch.save(checkpoint, stream); stream.flush(); os.fsync(stream.fileno())
        with (staging / "optimizer.pt").open("xb") as stream:
            torch.save(optimizer.state_dict(), stream); stream.flush(); os.fsync(stream.fileno())
        (staging / "boundary.ptx").write_bytes((Path(work) / "boundary.ptx").read_bytes())
        _atomic_json_create(staging / "history.json", {"updates": history})
        control_receipt = bindings["control_receipt"]
        _atomic_json_create(staging / "control_receipt.json", control_receipt)
        files = {name: _sha_file(staging / name) for name in
                 ("checkpoint.pt", "optimizer.pt", "history.json", "boundary.ptx", "control_receipt.json")}
        manifest = {"experiment": EXPERIMENT, "status": "training_complete",
            "arm": ARM, "seed": SEED, "completed_updates": UPDATES,
            "updates": UPDATES, "passes": PASSES,
            "training_images_per_pass": TRAIN_COUNT, "logical_batch": BATCH,
            "microbatch": MICRO, "exposures": UPDATES * BATCH,
            "validation_data_read": False,
            "ground_truth_scope": "canonical TRAIN512 feasible targets only",
            "bindings": bindings, "training_fingerprint": bindings["training_fingerprint"],
            "source_state_sha256_before": source_sha,
            "source_state_sha256_after": base._source_sha(found, found.encoder.base_encoder),
            "base_encoder_sha256_before": base_encoder_sha,
            "base_encoder_sha256_after": base.preflight._module_sha(found.encoder.base_encoder),
            "frozen_source_components_before": frozen_before,
            "frozen_source_components_after": frozen_after,
            "initial_model_state_sha256": bindings["initial_model_state_sha256"],
            "final_model_state_sha256": checkpoint["model_state_sha256"],
            "source_cache_file_sha256": cache_before,
            "optimizer_step_summary": base._optimizer_summary(optimizer),
            "files_sha256": files}
        _atomic_json_create(staging / "manifest.json", manifest)
        validate_training_artifact(staging, expected_bindings=bindings)
        os.rename(staging, output)
    except BaseException:
        raise
    return json.loads((output / "manifest.json").read_text(encoding="utf-8"))


def run_training(*, physical_gpu, output_root=OUTPUT_ROOT, admitted=None):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("SW0176 worker must inherit its leased physical GPU visibility")
    admitted = prepare_training_inputs_cpu() if admitted is None else admitted
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0176 trainer initialized CUDA before source/admission checks")
    if admitted["preflight_report"].get("status") != "complete_source_preflight":
        raise RuntimeError("SW0176 rejected source preflight cannot start training")
    if admitted.get("fresh_cpu_factory_receipt") is None:
        prepare_fresh_training_state_cpu(admitted)
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0176 fresh CPU factory initialized CUDA before leased training")
    torch.set_num_threads(1)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    prepared = admitted["prepared"]
    found, base_encoder, source_sha, base_encoder_sha, init_state_sha, _ = preflight_v4._build_source(
        prepared, device)
    if base._source_sha(found, base_encoder) != source_sha:
        raise ValueError("SW0176 fresh source state differs from validated source lineage")
    frozen_before = base._frozen_source_components(found, base_encoder)
    optimizer, families = base._make_optimizer(found)
    initial_state_sha = base._state_sha(found.wrapped, found.encoder)
    bindings = _make_bindings(admitted, initial_state_sha, source_sha, base_encoder_sha, families, optimizer)
    if init_state_sha != initial_state_sha:
        raise ValueError("SW0176 fresh initialization changed while optimizer was constructed")
    runtime_module = __import__("collaborative_test.SW_0148_checkpointable_boundary_q.checkpoint",
                                fromlist=["runtime_signature"])
    runtime = runtime_module.runtime_signature(device)
    output_root = Path(output_root)
    work = output_root / "work" / f"{ARM}_seed1"
    output = output_root / f"{ARM}_seed1"
    if output.exists():
        return validate_training_artifact(output, expected_bindings=bindings)
    if work.exists() and (work / "boundary.ptx").is_file():
        loaded = runtime_module.load_latest(work, wrapped=found.wrapped, encoder=found.encoder,
            optimizer=optimizer, expected_bindings=bindings, expected_runtime=runtime,
            device=device, max_updates=UPDATES)
        completed, history = loaded["completed_updates"], loaded["history"]
    else:
        if work.exists() and any(work.iterdir()):
            raise FileExistsError("preserving incomplete SW0176 work files without a committed boundary")
        completed, history = 0, []
        _atomic_json_create(work / "run_bindings.json", bindings)
        runtime_module.save_boundary(work, completed_updates=0, wrapped=found.wrapped,
            encoder=found.encoder, optimizer=optimizer, history=history,
            bindings=bindings, runtime=runtime, device=device)
    _validate_cursor(completed, history)
    rejection = _resume_rejection(work, completed, history, bindings)
    if rejection is not None:
        return rejection
    arrays = prepared["arrays"]
    base_prepared = prepared["admitted"]["prepared"]
    cache_root = Path(base_prepared["cache_root"])
    cache_before = {"manifest": _sha_file(cache_root / "manifest.json"),
                    "arrays": _sha_file(cache_root / "arrays.npz")}
    found.encoder.train(); found.encoder.base_encoder.eval()
    found.patcher.eval(); found.wrapped.train()
    expected = _expected_batches()
    for update in range(completed, UPDATES):
        indices = np.asarray(expected[update], dtype=np.int64)
        torch.manual_seed(MODEL_SEED + update)
        torch.cuda.manual_seed(MODEL_SEED + update)
        row = _train_update(found, optimizer, indices, admitted, device, update_number=update + 1)
        history.append(row)
        runtime_module.save_boundary(work, completed_updates=update + 1,
            wrapped=found.wrapped, encoder=found.encoder, optimizer=optimizer,
            history=history, bindings=bindings, runtime=runtime, device=device)
        if update < 2:
            reasons = base._credit_rejection(row)
            if reasons:
                return _write_rejection(work, completed=update + 1, history=history,
                                        bindings=bindings, reasons=reasons)
    cache_after = {"manifest": _sha_file(cache_root / "manifest.json"),
                   "arrays": _sha_file(cache_root / "arrays.npz")}
    if cache_after != cache_before:
        raise RuntimeError("SW0176 immutable source training cache changed during updates")
    frozen_after = base._frozen_source_components(found, base_encoder)
    if frozen_after != frozen_before:
        raise ValueError("SW0176 training changed frozen source components")
    if _sha_file(cache_root / "manifest.json") != proof_sha(prepared["source_proof"], "producer_manifest_sha256"):
        raise ValueError("SW0176 source cache manifest no longer matches admitted proof")
    return _publish_training(output, work, found, optimizer, history, bindings, runtime,
        source_sha, base_encoder_sha, frozen_before, frozen_after, cache_before)


def proof_sha(proof, key):
    value = proof.get(key)
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"SW0176 source proof lacks {key}")
    return value
