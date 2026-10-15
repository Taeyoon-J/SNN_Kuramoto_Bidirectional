"""SW0175 paired 64-update critical-component training."""
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

from collaborative_test.SW_0171_contextual_instance_graph import preflight
from collaborative_test.SW_0171_contextual_instance_graph.contextual_encoder import ContextualFeatureEncoder

EXPERIMENT = "SW0175_critical_component_binding_training"
ARMS = ("native_Q", "critical_component")
TRAIN_COUNT = 512
PREFIX_IMAGES = 512
PREFIX_PASSES = 2
UPDATES = 64
BATCH, MICRO = 16, 4
SHUFFLE_SEED = 171001
MODEL_SEED = 171001
CONTEXT_SEED = 171001
ENCODER_LR, GRAPH_CORE_LR, INTEGRATION_LR = 3e-4, 3e-5, 1e-3
CLIP_NORM = 1.0
HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE / "results_archive" / "training_20261015"
ADMISSION_QUEUE = "collaborative_test.SW_0171_contextual_instance_graph.admission_queue"


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _module(name):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        return importlib.import_module(name)
    finally:
        if visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible


def _training_fingerprint() -> dict:
    paths = (
        "collaborative_test/SW_0171_contextual_instance_graph/DESIGN_KO.md",
        "collaborative_test/SW_0171_contextual_instance_graph/contextual_encoder.py",
        "collaborative_test/SW_0171_contextual_instance_graph/instance_objective.py",
        "collaborative_test/SW_0171_contextual_instance_graph/preflight.py",
        "collaborative_test/SW_0171_contextual_instance_graph/train_prefix.py",
        "collaborative_test/SW_0175_critical_component_binding/DESIGN_KO.md",
        "collaborative_test/SW_0175_critical_component_binding/critical_component_objective.py",
        "collaborative_test/SW_0175_critical_component_binding/train.py",
        "collaborative_test/SW_0171_contextual_instance_graph/admission_queue.py",
        "collaborative_test/SW_0169_normalized_hard_reset/calibration_v2.py",
        "collaborative_test/SW_0168_end_to_end_connectivity/connectivity_objective.py",
        "collaborative_test/SW_0148_checkpointable_boundary_q/checkpoint.py",
        "collaborative_test/SW_0134_native_spike_binding/rollout.py",
        "collaborative_test/SW_0135_native32_spike_binding/foundation.py",
        "collaborative_test/SW_0136_native32_transfer/evaluate.py",
        "collaborative_test/SW_0162_connected_signed_partition/preflight.py",
        "snn_kuramoto_bidirectional/graph_generator.py",
        "snn_kuramoto_bidirectional/spike_classifier.py",
    )
    result = {}
    for relative in paths:
        path = preflight.ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"SW0175 trainer dependency missing: {relative}")
        result[relative] = _sha_file(path)
    return {"files": result, "preflight_fingerprint": preflight.implementation_fingerprint(),
            "experiment": EXPERIMENT, "seed": 1, "context_seed": CONTEXT_SEED,
            "shuffle_seed": SHUFFLE_SEED, "training_images_per_pass": PREFIX_IMAGES,
            "passes": PREFIX_PASSES, "updates": UPDATES,
            "logical_batch": BATCH, "microbatch": MICRO,
            "encoder_lr": ENCODER_LR, "graph_core_lr": GRAPH_CORE_LR,
            "integration_lr": INTEGRATION_LR, "global_clip_norm": CLIP_NORM}


def _training_fingerprint_sha256() -> str:
    return _sha_bytes(json.dumps(_training_fingerprint(), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _load_admitted_inputs():
    queue = _module(ADMISSION_QUEUE)
    prep, prepared, proof = queue.source_admission()
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0175 training inputs must be admitted before CUDA initialization")
    state_path, report_path = queue.STATE, queue.REPORT
    if not state_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("SW0175 owner admission has no terminal state/report")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") != "complete_admitted":
        raise ValueError(f"SW0175 training requires complete_admitted, got {state.get('status')!r}")
    queue_fp = queue.fingerprint_sha256()
    prep_fp = prep.implementation_fingerprint()
    report_sha = _sha_file(report_path)
    if (state.get("report_file_sha256") != report_sha
            or state.get("queue_fingerprint_sha256") != queue_fp
            or state.get("implementation_fingerprint") != prep_fp):
        raise ValueError("SW0175 owner admission receipt hash/fingerprint mismatch")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    status = queue.validate_report(report, source_proof=proof,
        queue_fingerprint=queue_fp, preflight_fingerprint=prep_fp)
    if status != "complete_admitted" or report.get("training_admission") is not True:
        raise ValueError("SW0175 training is not admitted by the zero-update source preflight")
    if report.get("implementation_fingerprint") != prepared["implementation_fingerprint"]:
        raise ValueError("SW0175 admitted report and current prepared source differ")
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0175 source/input admission initialized CUDA")
    return {"queue": queue, "preflight": prep, "prepared": prepared,
            "proof": proof, "queue_fingerprint_sha256": queue_fp,
            "preflight_fingerprint": prep_fp, "admission_report_sha256": report_sha,
            "admission_report": report}


def _orders() -> list[list[int]]:
    rng = np.random.RandomState(SHUFFLE_SEED)
    return [rng.permutation(PREFIX_IMAGES).astype(int).tolist()
            for _ in range(PREFIX_PASSES)]


def _expected_batches() -> list[list[int]]:
    orders = _orders()
    batches = []
    for order in orders:
        batches.extend([order[i:i+BATCH] for i in range(0, PREFIX_IMAGES, BATCH)])
    if len(batches) != UPDATES or any(len(batch) != BATCH for batch in batches):
        raise AssertionError("registered SW0175 training does not form 64 complete B16 updates")
    return batches


def _validate_cursor(completed, history) -> None:
    expected = _expected_batches()
    if type(completed) is not int or completed != len(history) or not 0 <= completed <= UPDATES:
        raise ValueError("SW0175 boundary cursor/history mismatch")
    if any(row.get("update") != index + 1 or row.get("image_indices") != expected[index]
           for index, row in enumerate(history)):
        raise ValueError("SW0175 saved training history differs from registered exposure order")


def _backward_microbatch_loss(losses, logical_batch):
    if losses.ndim != 1 or losses.numel() < 1 or losses.numel() > logical_batch:
        raise ValueError("SW0175 microbatch loss vector has invalid geometry")
    if not bool(torch.isfinite(losses).all()):
        raise FloatingPointError("SW0175 microbatch contains nonfinite loss")
    (losses.sum() / logical_batch).backward()


def _named_families(found):
    graph = found.wrapped.core.graph_generator
    graph.geodesic_contrast.requires_grad_(False)
    # The production foundation is frozen on load. Re-enable the registered
    # trainable families before selecting optimizer parameters.
    for parameter in graph.parameters():
        parameter.requires_grad_(parameter is not graph.geodesic_contrast)
    for name, parameter in found.wrapped.core.named_parameters():
        if not name.startswith("graph_generator."):
            parameter.requires_grad_(True)
    for name in ("a_d", "a_m", "b"):
        getattr(found.wrapped, name).requires_grad_(True)
    found.encoder.base_encoder.requires_grad_(False)
    for name, parameter in found.encoder.context.named_parameters():
        parameter.requires_grad_(True)
    # The source CNN inside ContextualFeatureEncoder remains frozen. Only the
    # four RF31 context layers and zero-final map are optimized.
    families = {
        "context_encoder": [(name, p) for name, p in found.encoder.named_parameters()
                            if not name.startswith("base_encoder.") and p.requires_grad],
        "graph": [(name, p) for name, p in graph.named_parameters()
                  if name != "geodesic_contrast" and p.requires_grad],
        "native_core": [(name, p) for name, p in found.wrapped.core.named_parameters()
                        if not name.startswith("graph_generator.") and p.requires_grad],
        "integration": [(name, getattr(found.wrapped, name)) for name in ("a_d", "a_m", "b")],
    }
    flattened = [p for entries in families.values() for _, p in entries]
    if not flattened or len({id(p) for p in flattened}) != len(flattened):
        raise ValueError("SW0175 optimizer families are empty or overlap")
    for group in families.values():
        for _, parameter in group:
            parameter.requires_grad_(True)
    if any(p.requires_grad for p in found.encoder.base_encoder.parameters()):
        raise ValueError("SW0175 original source encoder must remain frozen")
    return families


def _state_sha(wrapped, encoder):
    digest = hashlib.sha256()
    for section, module in (("wrapped", wrapped), ("encoder", encoder)):
        for name, value in sorted(module.state_dict().items()):
            digest.update((section + "." + name).encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _source_sha(found, encoder_base):
    source_cal = _module("collaborative_test.SW_0169_normalized_hard_reset.calibration_v2")
    contextual = found.encoder
    found.encoder = encoder_base
    try:
        return source_cal._source_digest(found)
    finally:
        found.encoder = contextual


def _validate_source_proof_digest(proof, source_sha):
    recorded = proof.get("source_cache_state_sha256") if isinstance(proof, dict) else None
    if not isinstance(recorded, str) or recorded != source_sha:
        raise ValueError("SW0175 source digest differs from source_cache_state_sha256 admission proof")
    return recorded


def _frozen_source_components(found, base_encoder):
    """Hashes for source components that must remain frozen during the training."""
    graph = found.wrapped.core.graph_generator
    values = {
        "base_encoder": preflight._module_sha(base_encoder),
        "geodesic_contrast": _tensor_sha(graph.geodesic_contrast),
        "patcher": preflight._module_sha(found.patcher),
        "feature_mean": _tensor_sha(torch.as_tensor(found.feature_mean)),
        "feature_std": _tensor_sha(torch.as_tensor(found.feature_std)),
        "feature_clip": _sha_bytes(json.dumps(float(found.feature_clip), allow_nan=False).encode()),
    }
    if any(len(value) != 64 for value in values.values()):
        raise ValueError("SW0175 frozen-source component digest is malformed")
    return values


def _credit_rejection(row):
    """Return registered first-two-update credit failures, without inventing activity."""
    failures = []
    q_to_s = float(row.get("registered_loss_to_component_spikes_gradient_l2", math.nan))
    if not math.isfinite(q_to_s) or q_to_s <= 0:
        failures.append("Q-to-component-spikes gradient is not finite-positive")
    norms = row.get("family_gradient_norms", {})
    deltas = row.get("family_parameter_delta_l2", {})
    for family in ("graph", "native_core"):
        grad, delta = float(norms.get(family, math.nan)), float(deltas.get(family, math.nan))
        if not math.isfinite(grad) or not math.isfinite(delta):
            failures.append(f"{family} family gradient/delta is nonfinite")
    credit = row.get("named_parameter_credit", {}).get("context_encoder", [])
    if int(row.get("update", 0)) == 1:
        output_rows = [item for item in credit if item.get("name", "").startswith("context.8.")]
        if not output_rows or not any(item.get("gradient_l2", 0) > 0 and
                                      item.get("parameter_delta_l2", 0) > 0 for item in output_rows):
            failures.append("zero-final contextual output map received no first-update credit/change")
    elif int(row.get("update", 0)) == 2:
        hidden_rows = [item for item in credit if item.get("name", "").startswith("context.")
                       and not item.get("name", "").startswith("context.8.")]
        if not hidden_rows or not any(item.get("gradient_l2", 0) > 0 and
                                      item.get("parameter_delta_l2", 0) > 0 for item in hidden_rows):
            failures.append("contextual hidden layers received no second-update credit/change")
    return failures


def _history_sha(history):
    return _sha_bytes(json.dumps(history, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode("utf-8"))


def _binding_sha(bindings):
    return _sha_bytes(json.dumps(bindings, sort_keys=True, separators=(",", ":"),
                       allow_nan=False).encode("utf-8"))


def _read_credit_rejection(work, *, arm, bindings, completed, history):
    path = Path(work) / "credit_rejection.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("experiment") != EXPERIMENT or report.get("status") != "scientific_rejection"
            or report.get("arm") != arm or report.get("completed_updates") != completed
            or report.get("history_sha256") != _history_sha(history)
            or report.get("bindings_sha256") != _binding_sha(bindings)
            or report.get("boundary_sha256") != _sha_file(Path(work) / "boundary.ptx")
            or report.get("source_proof") != bindings.get("source_proof")
            or report.get("training_fingerprint") != bindings.get("training_fingerprint")
            or report.get("validation_data_read") is not False
            or not isinstance(report.get("rejection_reasons"), list)
            or not report["rejection_reasons"]):
        raise ValueError("SW0175 saved scientific rejection is not bound to the committed boundary")
    return report


def _write_credit_rejection(work, *, arm, completed, history, bindings, reasons):
    report = {"experiment": EXPERIMENT, "status": "scientific_rejection",
        "arm": arm, "completed_updates": completed,
        "history_sha256": _history_sha(history), "bindings_sha256": _binding_sha(bindings),
        "boundary_sha256": _sha_file(Path(work) / "boundary.ptx"),
        "source_proof": bindings.get("source_proof"),
        "training_fingerprint": bindings.get("training_fingerprint"),
        "rejection_reasons": list(reasons), "validation_data_read": False}
    _atomic_json_create(Path(work) / "credit_rejection.json", report)
    return report


def _recover_terminal_rejection(work, *, arm, bindings, completed, history):
    saved = _read_credit_rejection(work, arm=arm, bindings=bindings,
                                   completed=completed, history=history)
    if saved is not None:
        return saved
    for committed_row in history[:2]:
        failures = _credit_rejection(committed_row)
        if failures:
            return _write_credit_rejection(work, arm=arm,
                completed=int(committed_row["update"]), history=history,
                bindings=bindings, reasons=failures)
    return None


def _make_optimizer(found):
    families = _named_families(found)
    groups = []
    for name, entries in families.items():
        groups.append({"params": [p for _, p in entries],
                       "parameter_names": [parameter_name for parameter_name, _ in entries],
                       "name": name,
                       "lr": INTEGRATION_LR if name == "integration" else
                            ENCODER_LR if name == "context_encoder" else GRAPH_CORE_LR})
    return torch.optim.Adam(groups), families


def _loss_for_arm(arm, q, rho, target, support, objective=None):
    from collaborative_test.SW_0175_critical_component_binding import critical_component_objective as loss_api
    if arm == "native_Q":
        return loss_api.native_q_loss(q, target, support)
    if arm == "critical_component":
        return loss_api.critical_component_loss(q, rho, target, support)
    raise ValueError(f"unregistered SW0175 objective arm {arm!r}")


def _native_forward(found, rgb, *, live_tail_steps):
    from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
    from collaborative_test.SW_0162_connected_signed_partition.preflight import capture_source_coupling
    from collaborative_test.SW_0171_contextual_instance_graph.instance_objective import capture_native_projection
    critical = _module("collaborative_test.SW_0175_critical_component_binding.critical_component_objective")

    gamma = found.encode(rgb)
    graph = found.wrapped.core.graph_generator
    with capture_native_projection(graph) as projection:
        trace, adjacency = capture_source_coupling(found.wrapped.core.kuramoto,
            lambda: late_rollout(found.wrapped, gamma, total_steps=preflight.STEPS,
                                 live_tail_steps=live_tail_steps))
    if len(projection) != 1:
        raise ValueError("native graph projection must be captured exactly once")
    rho, q = critical.native_component_rho_q(trace["spikes"], trace["component_spikes"],
                                                settle=preflight.SETTLE)
    for name in ("theta", "membrane", "spikes", "component_membrane",
                 "component_spikes", "component_gates"):
        if name not in trace or not bool(torch.isfinite(trace[name]).all()):
            raise FloatingPointError(f"SW0175 native rollout has missing/nonfinite trace {name}")
    if (not bool(torch.isfinite(adjacency).all()) or not bool(torch.isfinite(projection[0]).all())
            or not bool(torch.isfinite(q).all())):
        raise FloatingPointError("SW0175 native A/z/Q contains nonfinite values")
    return gamma, trace, adjacency, projection[0], q, rho


def _train_update(arm, found, optimizer, indices, admitted, device, *, update_number):
    prepared = admitted["prepared"]
    arrays = prepared["arrays"]
    objective = _module("collaborative_test.SW_0168_end_to_end_connectivity.connectivity_objective")
    optimizer.zero_grad(set_to_none=True)
    before = {group["name"]: [p.detach().clone() for p in group["params"]]
              for group in optimizer.param_groups}
    total_losses, q_losses = [], []
    q_to_s_squared = 0.0
    micro_records = []
    for offset in range(0, BATCH, MICRO):
        ids = indices[offset:offset+MICRO]
        rgb = torch.from_numpy(np.ascontiguousarray(arrays["rgb"][ids])).to(device)
        gamma, trace, adjacency, z, q, rho = _native_forward(found, rgb, live_tail_steps=preflight.TAIL)
        micro_total = []
        for local in range(MICRO):
            index = int(ids[local])
            target = arrays["feasible_targets"][index]
            support = arrays["source_support"][index]
            q_loss = _module("collaborative_test.SW_0175_critical_component_binding.critical_component_objective").native_q_loss(
                q[local], target, support)
            total = q_loss if arm == "native_Q" else _loss_for_arm(
                arm, q[local], rho[local], target, support, objective)
            if (total.ndim != 0 or not bool(torch.isfinite(total))
                    or q_loss.ndim != 0 or not bool(torch.isfinite(q_loss))):
                raise FloatingPointError(f"nonfinite SW0175 {arm} loss at TRAIN row {index}")
            if update_number <= 2:
                q_to_s = torch.autograd.grad(total, trace["component_spikes"],
                    retain_graph=True, allow_unused=True)[0]
                if q_to_s is not None:
                    if not bool(torch.isfinite(q_to_s).all()):
                        raise FloatingPointError("nonfinite SW0175 Q-to-component-spikes gradient")
                    q_to_s_squared += float(q_to_s.detach().double().square().sum().cpu())
            micro_total.append(total)
            total_losses.append(float(total.detach().cpu()))
            q_losses.append(float(q_loss.detach().cpu()))
        _backward_microbatch_loss(torch.stack(micro_total), BATCH)
        micro_records.append({"indices": list(map(int, ids)), "q_sha256": _tensor_sha(q),
                              "z_sha256": _tensor_sha(z)})
        del rgb, gamma, trace, adjacency, z, q, rho, micro_total

    gradients, raw_by_id = {}, {}
    all_parameters = [p for group in optimizer.param_groups for p in group["params"]]
    for group in optimizer.param_groups:
        sq = 0.0
        for name, parameter in zip(group["parameter_names"], group["params"]):
            if parameter.grad is not None:
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise FloatingPointError(f"nonfinite SW0175 gradient in {group['name']}/{name}")
                norm = float(parameter.grad.detach().double().norm().cpu())
                raw_by_id[id(parameter)] = norm
                sq += norm * norm
        gradients[group["name"]] = math.sqrt(sq)
    total_grad_norm = math.sqrt(sum(value * value for value in gradients.values()))
    clipped_norm = torch.nn.utils.clip_grad_norm_(all_parameters, CLIP_NORM)
    if not bool(torch.isfinite(clipped_norm)):
        raise FloatingPointError("nonfinite SW0175 global gradient norm")
    optimizer.step()
    found.wrapped.project_integrations_()

    deltas, named_credit = {}, {}
    for group in optimizer.param_groups:
        delta_sq = 0.0
        name_rows = []
        for name, old, parameter in zip(group["parameter_names"], before[group["name"]], group["params"]):
            delta = float((parameter.detach().double() - old.detach().double()).norm().cpu())
            if not math.isfinite(delta):
                raise FloatingPointError("nonfinite SW0175 parameter delta")
            delta_sq += delta * delta
            name_rows.append({"name": name, "gradient_l2": raw_by_id.get(id(parameter), 0.0),
                              "gradient_present": parameter.grad is not None,
                              "parameter_delta_l2": delta})
        deltas[group["name"]] = math.sqrt(delta_sq)
        named_credit[group["name"]] = name_rows
    if any(not math.isfinite(v) for v in gradients.values()):
        raise FloatingPointError("nonfinite SW0175 family gradient norm")
    return {"update": int(update_number), "image_indices": list(map(int, indices)),
            "image_ids": [int(prepared["cache_proof"]["image_ids"][int(i)]) for i in indices],
            "images_seen": BATCH, "microbatch": MICRO,
            "mean_loss": float(np.mean(total_losses)), "mean_q_loss": float(np.mean(q_losses)),
            "mean_registered_loss": float(np.mean(total_losses)),
            "gradient_norm_before_clip": total_grad_norm,
            "clip_scale": min(1.0, CLIP_NORM / (float(clipped_norm.detach().cpu()) + 1e-12)),
            "family_gradient_norms": gradients, "family_parameter_delta_l2": deltas,
            "named_parameter_credit": named_credit, "microbatches": micro_records,
            "registered_loss_to_component_spikes_gradient_l2": math.sqrt(q_to_s_squared)
                if update_number <= 2 else None}


def _tensor_sha(tensor):
    return _sha_bytes(tensor.detach().cpu().contiguous().numpy().tobytes())


def _diagnose_train8(found, admitted, *, phase):
    arrays = admitted["prepared"]["arrays"]
    objective = _module("collaborative_test.SW_0168_end_to_end_connectivity.connectivity_objective")
    from collaborative_test.SW_0136_native32_transfer.evaluate import _qcc_labels
    rows = []
    device = next(found.encoder.parameters()).device
    prior_modes = (found.encoder.training, found.patcher.training, found.wrapped.training)
    found.encoder.eval(); found.patcher.eval(); found.wrapped.eval()
    try:
        with torch.no_grad():
            for start in (256, 260):
                rgb = torch.from_numpy(np.ascontiguousarray(arrays["rgb"][start:start+4])).to(device)
                _gamma, trace, _a, z, q, rho = _native_forward(found, rgb, live_tail_steps=0)
                losses = []
                q_losses = []
                for local in range(4):
                    q_losses.append(_module("collaborative_test.SW_0175_critical_component_binding.critical_component_objective").native_q_loss(
                        q[local], arrays["feasible_targets"][start+local], arrays["source_support"][start+local]))
                    losses.append(q_losses[-1] if admitted.get("diagnostic_arm", "native_Q") == "native_Q" else
                        _loss_for_arm(admitted["diagnostic_arm"], q[local], rho[local],
                            arrays["feasible_targets"][start+local], arrays["source_support"][start+local], objective))
                labels = _qcc_labels(trace)
                if torch.is_tensor(labels):
                    labels = labels.detach().cpu().numpy()
                labels = np.asarray(labels, dtype=np.int32)
                if labels.shape != (4, 32, 32):
                    raise ValueError("SW0175 native QCC diagnostic has wrong output shape")
                for local in range(4):
                    i = start + local
                    rows.append({"position": i,
                        "image_id": int(admitted["prepared"]["cache_proof"]["image_ids"][i]),
                        "native_q_loss": float(q_losses[local].detach().cpu()),
                        "registered_loss": float(losses[local].detach().cpu()),
                        "source_foreground_nonempty": bool(np.any(labels[local] > 0)),
                        "native_qcc_sha256": _sha_bytes(labels[local].tobytes()),
                        "q_sha256": _tensor_sha(q[local]), "z_sha256": _tensor_sha(z[local])})
                del rgb, trace, z, q, rho, losses, q_losses, labels
    finally:
        found.encoder.train(prior_modes[0]); found.patcher.train(prior_modes[1]); found.wrapped.train(prior_modes[2])
    return {"phase": phase, "arm": admitted.get("diagnostic_arm", "native_Q"),
            "positions": list(range(256, 264)), "rows": rows,
            "mean_native_q_loss": float(np.mean([r["native_q_loss"] for r in rows])),
            "mean_registered_loss": float(np.mean([r["registered_loss"] for r in rows]))}


def _bindings(admitted, arm, initial_state_sha, source_sha, base_encoder_sha):
    prep = admitted["prepared"]
    return {"experiment": EXPERIMENT, "arm": arm, "seed": 1,
        "admission_report_sha256": admitted["admission_report_sha256"],
        "admission_queue_fingerprint_sha256": admitted["queue_fingerprint_sha256"],
        "preflight_implementation_fingerprint": admitted["preflight_fingerprint"],
        "source_proof": admitted["proof"], "source_state_sha256": source_sha,
        "base_encoder_sha256": base_encoder_sha,
        "source_checkpoint_sha256": prep["source_checkpoint_sha256"],
        "source_manifest_sha256": prep["source_manifest_sha256"],
        "input_cache_manifest_sha256": prep["cache_manifest_sha256"],
        "input_cache_arrays_sha256": prep["arrays_sha256"],
        "ordered_train_ids_sha256": _sha_bytes(np.asarray(prep["cache_proof"]["image_ids"], dtype="<i8").tobytes()),
        "ordered_train_ids": list(map(int, prep["cache_proof"]["image_ids"])),
        "context_seed": CONTEXT_SEED,
        "shuffle_seed": SHUFFLE_SEED, "training_images_per_pass": PREFIX_IMAGES,
        "passes": PREFIX_PASSES, "updates": UPDATES,
        "logical_batch": BATCH, "microbatch": MICRO,
        "optimizer": "Adam", "clip_norm": CLIP_NORM,
        "learning_rates": {"context_encoder": ENCODER_LR,
                            "graph": GRAPH_CORE_LR, "native_core": GRAPH_CORE_LR,
                            "integration": INTEGRATION_LR},
        "initial_model_state_sha256": initial_state_sha,
        "training_fingerprint": _training_fingerprint()}


def train(arm, *, physical_gpu, output_root=OUTPUT_ROOT):
    if arm not in ARMS:
        raise ValueError("SW0175 registered training arm required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("SW0175 trainer is not running on its assigned physical GPU")
    admitted = _load_admitted_inputs()
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0175 CUDA initialized before training input/admission checks")
    torch.set_num_threads(1)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    source_cal = _module("collaborative_test.SW_0169_normalized_hard_reset.calibration_v2")
    found = source_cal._load_source(device)
    found.encoder.eval(); found.patcher.eval(); found.wrapped.eval()
    graph = found.wrapped.core.graph_generator
    if (getattr(graph, "_sw0143_geodesic_arm", None) != "T"
            or found.wrapped.core.kuramoto_backend != "factorized"):
        raise ValueError("SW0175 training requires unchanged original SW0143-T source dynamics")
    source_sha = source_cal._source_digest(found)
    _validate_source_proof_digest(admitted["proof"], source_sha)
    base_encoder = found.encoder
    base_encoder_sha = preflight._module_sha(base_encoder)
    frozen_components_before = _frozen_source_components(found, base_encoder)
    admitted_init = admitted["admission_report"].get("initialization", {})
    if (base_encoder_sha != admitted["admission_report"].get("base_encoder_sha256")
            or preflight._integration_values(found.wrapped) != admitted_init.get("integration")
            or float(found.feature_clip) != admitted_init.get("feature_clip")):
        raise ValueError("SW0175 fresh source integration/base preprocessing differs from preflight")
    found.encoder = ContextualFeatureEncoder(base_encoder, seed=CONTEXT_SEED).to(device)
    found.encoder.eval()
    context_initial_sha = preflight._module_sha(found.encoder)
    if context_initial_sha != admitted["admission_report"].get("context_zero_init_sha256"):
        raise ValueError("SW0175 fresh contextual encoder initialization differs from preflight")
    optimizer, families = _make_optimizer(found)
    initial_state_sha = _state_sha(found.wrapped, found.encoder)
    bindings = _bindings(admitted, arm, initial_state_sha, source_sha, base_encoder_sha)
    runtime_module = _module("collaborative_test.SW_0148_checkpointable_boundary_q.checkpoint")
    runtime = runtime_module.runtime_signature(device)
    work = Path(output_root) / "work" / f"{arm}_seed1"
    output = Path(output_root) / f"{arm}_seed1"
    if output.exists():
        return validate_training_artifact(arm, output, expected_bindings=bindings)
    if work.exists() and (work / "boundary.ptx").exists():
        loaded = runtime_module.load_latest(work, wrapped=found.wrapped, encoder=found.encoder,
            optimizer=optimizer, expected_bindings=bindings, expected_runtime=runtime,
            device=device, max_updates=UPDATES)
        completed, history = loaded["completed_updates"], loaded["history"]
    else:
        if work.exists() and any(work.iterdir()):
            raise FileExistsError("preserving unexpected/incomplete SW0175 training work files")
        completed, history = 0, []
        _atomic_json_create(work / "run_bindings.json", bindings)
        runtime_module.save_boundary(work, completed_updates=0, wrapped=found.wrapped,
            encoder=found.encoder, optimizer=optimizer, history=history,
            bindings=bindings, runtime=runtime, device=device)
    _validate_cursor(completed, history)
    terminal_rejection = _recover_terminal_rejection(work, arm=arm, bindings=bindings,
                                                     completed=completed, history=history)
    if terminal_rejection is not None:
        return terminal_rejection
    expected = _expected_batches()
    diag_path = work / "initial_train8.json"
    if diag_path.is_file():
        initial_diagnostics = json.loads(diag_path.read_text(encoding="utf-8"))
    elif completed == 0:
        admitted["diagnostic_arm"] = arm
        initial_diagnostics = _diagnose_train8(found, admitted, phase="initial_no_update")
        _atomic_json_create(diag_path, initial_diagnostics)
    else:
        raise FileNotFoundError("resumable SW0175 work lacks initial TRAIN8 diagnostic")

    found.encoder.train(); found.encoder.base_encoder.eval()
    found.patcher.eval(); found.wrapped.train()
    cache_root = Path(admitted["prepared"]["cache_root"])
    cache_before = {"manifest": _sha_file(cache_root / "manifest.json"),
                    "arrays": _sha_file(cache_root / "arrays.npz")}
    for update in range(completed, UPDATES):
        batch = np.asarray(expected[update], dtype=np.int64)
        torch.manual_seed(MODEL_SEED + update)
        torch.cuda.manual_seed(MODEL_SEED + update)
        row = _train_update(arm, found, optimizer, batch, admitted, device,
                            update_number=update + 1)
        row["pass"] = update // 32 + 1
        history.append(row)
        runtime_module.save_boundary(work, completed_updates=update + 1,
            wrapped=found.wrapped, encoder=found.encoder, optimizer=optimizer,
            history=history, bindings=bindings, runtime=runtime, device=device)
        if update < 2:
            failures = _credit_rejection(row)
            if failures:
                return _write_credit_rejection(work, arm=arm, completed=update + 1,
                    history=history, bindings=bindings, reasons=failures)
    if cache_before != {"manifest": _sha_file(cache_root / "manifest.json"),
                        "arrays": _sha_file(cache_root / "arrays.npz")}:
        raise RuntimeError("SW0175 immutable training-input cache changed during training updates")
    admitted["diagnostic_arm"] = arm
    final_diagnostics = _diagnose_train8(found, admitted, phase="after_update64")
    frozen_components_after = _frozen_source_components(found, base_encoder)
    if frozen_components_after != frozen_components_before:
        raise ValueError("SW0175 frozen source components changed during training")
    work_payload = runtime_module.inspect_committed_metadata(work)
    if work_payload.get("completed_updates") != UPDATES or work_payload.get("bindings") != bindings:
        raise ValueError("SW0175 final durable checkpoint is not at the exact 64-update boundary")
    return _publish_training(arm, output, work, found, optimizer, history,
                           initial_diagnostics, final_diagnostics, bindings,
                           runtime, source_sha, base_encoder_sha, cache_before,
                           frozen_components_before, frozen_components_after,
                           _source_sha(found, base_encoder))


def _atomic_json_create(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _publish_training(arm, output, work, found, optimizer, history, initial_diagnostics,
                    final_diagnostics, bindings, runtime, source_sha, base_encoder_sha,
                    cache_before, frozen_components_before, frozen_components_after,
                    trained_model_state_sha):
    if output.exists():
        raise FileExistsError("preserving existing SW0175 training artifact")
    staging = output.with_name(output.name + f".stage-{uuid.uuid4().hex}")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        import io
        checkpoint = {"experiment": EXPERIMENT, "arm": arm, "bindings": bindings,
            "wrapped": {k: v.detach().cpu().clone() for k, v in found.wrapped.state_dict().items()},
            "encoder": {k: v.detach().cpu().clone() for k, v in found.encoder.state_dict().items()}}
        checkpoint["model_state_sha256"] = _state_sha(found.wrapped, found.encoder)
        with (staging / "checkpoint.pt").open("xb") as stream:
            torch.save(checkpoint, stream); stream.flush(); os.fsync(stream.fileno())
        with (staging / "optimizer.pt").open("xb") as stream:
            torch.save(optimizer.state_dict(), stream); stream.flush(); os.fsync(stream.fileno())
        shutil.copyfile(work / "boundary.ptx", staging / "boundary.ptx")
        _atomic_json_create(staging / "history.json", {"updates": history})
        _atomic_json_create(staging / "initial_train8.json", initial_diagnostics)
        _atomic_json_create(staging / "final_train8.json", final_diagnostics)
        step_summary = _optimizer_summary(optimizer)
        manifest = {"experiment": EXPERIMENT, "status": "training_complete",
            "arm": arm, "seed": 1, "completed_updates": UPDATES, "updates": UPDATES,
            "passes": PREFIX_PASSES, "training_images_per_pass": PREFIX_IMAGES,
            "logical_batch": BATCH, "microbatch": MICRO, "exposures": UPDATES * BATCH,
            "train_image_indices": [i for batch in _expected_batches() for i in batch],
            "train_ids": [item for row in history for item in row["image_ids"]],
            "ground_truth_scope": "canonical TRAIN input cache only; fixed TRAIN8 diagnostics",
            "validation_data_read": False, "bindings": bindings, "runtime": runtime,
            "source_model_state_before_sha256": source_sha,
            "source_model_state_after_sha256": trained_model_state_sha,
            "frozen_base_encoder_sha256_before": base_encoder_sha,
            "frozen_base_encoder_sha256_after": preflight._module_sha(found.encoder.base_encoder),
            "frozen_source_components_before": frozen_components_before,
            "frozen_source_components_after": frozen_components_after,
            "initial_model_state_sha256": bindings["initial_model_state_sha256"],
            "final_model_state_sha256": checkpoint["model_state_sha256"],
            "optimizer_step_summary": step_summary,
            "source_cache_file_sha256": cache_before,
            "initial_train8_sha256": _sha_file(staging / "initial_train8.json"),
            "final_train8_sha256": _sha_file(staging / "final_train8.json"),
            "history_sha256": _sha_file(staging / "history.json"),
            "checkpoint_sha256": _sha_file(staging / "checkpoint.pt"),
            "optimizer_sha256": _sha_file(staging / "optimizer.pt"),
            "boundary_sha256": _sha_file(staging / "boundary.ptx"),
            "training_fingerprint": bindings["training_fingerprint"]}
        _atomic_json_create(staging / "manifest.json", manifest)
        validate_training_artifact(arm, staging, expected_bindings=bindings)
        os.rename(staging, output)
    except BaseException:
        # Preserve staging for explicit inspection/recovery.
        raise
    return json.loads((output / "manifest.json").read_text(encoding="utf-8"))


def _optimizer_summary(optimizer):
    summary = {}
    for group in optimizer.param_groups:
        name = group["name"]
        counts = []
        updated_names = []
        for parameter_name, parameter in zip(group["parameter_names"], group["params"]):
            if parameter in optimizer.state:
                step = int(optimizer.state[parameter]["step"].item())
                counts.append(step); updated_names.append(parameter_name)
        summary[name] = {"parameter_count": len(group["params"]),
            "updated_parameter_count": len(counts), "unused_parameter_count": len(group["params"]) - len(counts),
            "updated_parameter_names": updated_names,
            "unused_parameter_names": [n for n in group["parameter_names"] if n not in updated_names],
            "steps": sorted(counts), "max_step": max(counts, default=0)}
    return summary


def validate_training_artifact(arm, root, *, expected_bindings=None):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("experiment") != EXPERIMENT or manifest.get("status") != "training_complete"
            or manifest.get("arm") != arm or manifest.get("seed") != 1
            or manifest.get("completed_updates") != UPDATES or manifest.get("updates") != UPDATES
            or manifest.get("passes") != PREFIX_PASSES or manifest.get("training_images_per_pass") != PREFIX_IMAGES
            or manifest.get("logical_batch") != BATCH or manifest.get("microbatch") != MICRO
            or manifest.get("exposures") != UPDATES * BATCH
            or manifest.get("validation_data_read") is not False):
        raise ValueError("SW0175 training manifest has incorrect experiment/cursor/scope")
    bindings = manifest.get("bindings")
    if not isinstance(bindings, dict) or bindings.get("arm") != arm:
        raise ValueError("SW0175 training binding schema mismatch")
    if expected_bindings is not None and bindings != expected_bindings:
        raise ValueError("SW0175 training bindings differ from current source/admission")
    expected_indices = [index for batch in _expected_batches() for index in batch]
    if manifest.get("train_image_indices") != expected_indices:
        raise ValueError("SW0175 training exposures differ from registered two-pass order")
    ordered_ids = bindings.get("ordered_train_ids")
    if (not isinstance(ordered_ids, list) or len(ordered_ids) != TRAIN_COUNT
            or _sha_bytes(np.asarray(ordered_ids, dtype="<i8").tobytes())
               != bindings.get("ordered_train_ids_sha256")):
        raise ValueError("SW0175 training bindings lack the canonical ordered TRAIN IDs")
    expected_ids = [int(ordered_ids[i]) for i in expected_indices]
    if manifest.get("train_ids") != expected_ids:
        raise ValueError("SW0175 training exposure IDs do not follow the bound canonical order")
    after_source_sha = manifest.get("source_model_state_after_sha256")
    if (not isinstance(after_source_sha, str) or len(after_source_sha) != 64
            or any(c not in "0123456789abcdef" for c in after_source_sha.lower())
            or manifest.get("frozen_base_encoder_sha256_before")
               != manifest.get("frozen_base_encoder_sha256_after")):
        raise ValueError("SW0175 training source/base encoder immutability evidence differs")
    proof = bindings.get("source_proof")
    if (not isinstance(proof, dict)
            or manifest.get("source_model_state_before_sha256") != proof.get("source_cache_state_sha256")
            or manifest.get("source_model_state_before_sha256") != bindings.get("source_state_sha256")
            or manifest.get("frozen_base_encoder_sha256_before") != bindings.get("base_encoder_sha256")
            or manifest.get("frozen_source_components_before") != manifest.get("frozen_source_components_after")
            or not isinstance(manifest.get("frozen_source_components_before"), dict)
            or any(not isinstance(value, str) or len(value) != 64
                   for value in manifest["frozen_source_components_before"].values())
            or manifest.get("training_fingerprint") != bindings.get("training_fingerprint")
            or manifest.get("source_cache_file_sha256") != {
                "manifest": bindings.get("input_cache_manifest_sha256"),
                "arrays": bindings.get("input_cache_arrays_sha256")}):
        raise ValueError("SW0175 training manifest provenance differs from its bound source/cache")
    for filename, key in (("history.json", "history_sha256"),
                           ("checkpoint.pt", "checkpoint_sha256"),
                           ("optimizer.pt", "optimizer_sha256"),
                           ("boundary.ptx", "boundary_sha256"),
                           ("initial_train8.json", "initial_train8_sha256"),
                           ("final_train8.json", "final_train8_sha256")):
        if _sha_file(root / filename) != manifest.get(key):
            raise ValueError(f"SW0175 training artifact hash mismatch: {filename}")
    history = json.loads((root / "history.json").read_text(encoding="utf-8")).get("updates")
    batches = _expected_batches()
    if (not isinstance(history, list) or len(history) != UPDATES
            or [row.get("update") for row in history] != list(range(1, UPDATES + 1))
            or [row.get("image_indices") for row in history] != batches
            or [row.get("image_ids") for row in history]
               != [[int(ordered_ids[i]) for i in batch] for batch in batches]):
        raise ValueError("SW0175 training history cursor/exposure order mismatch")
    for row in history:
        for key in ("mean_loss", "mean_q_loss", "mean_registered_loss", "gradient_norm_before_clip"):
            if not math.isfinite(float(row.get(key, math.nan))):
                raise ValueError("SW0175 training history contains nonfinite training metrics")
        for field in ("family_gradient_norms", "family_parameter_delta_l2"):
            values = row.get(field)
            if (not isinstance(values, dict)
                    or set(values) != {"context_encoder", "graph", "native_core", "integration"}
                    or any(not math.isfinite(float(value)) or float(value) < 0 for value in values.values())):
                raise ValueError("SW0175 training history lacks finite named-family evidence")
        if row.get("images_seen") != BATCH or len(row.get("microbatches", [])) != BATCH // MICRO:
            raise ValueError("SW0175 logical update lacks four committed microbatches")
    for row in history[:2]:
        failures = _credit_rejection(row)
        if failures:
            raise ValueError("SW0175 first-two-update credit gate failed: " + "; ".join(failures))
    if history[0].get("arm") not in (None, arm):
        raise ValueError("SW0175 history arm mismatch")
    payload = torch.load(root / "checkpoint.pt", map_location="cpu", weights_only=False)
    if payload.get("bindings") != bindings or payload.get("arm") != arm:
        raise ValueError("SW0175 training checkpoint lineage mismatch")
    if _state_sha_from_state(payload) != manifest.get("final_model_state_sha256"):
        raise ValueError("SW0175 training checkpoint model digest mismatch")
    optimizer_payload = torch.load(root / "optimizer.pt", map_location="cpu", weights_only=False)
    _validate_optimizer_state(optimizer_payload, manifest.get("optimizer_step_summary"))
    for phase in ("initial", "final"):
        diagnostic = json.loads((root / f"{phase}_train8.json").read_text(encoding="utf-8"))
        if diagnostic.get("phase") != ("initial_no_update" if phase == "initial" else "after_update64"):
            raise ValueError("SW0175 fixed TRAIN8 diagnostic phase mismatch")
        if [row.get("position") for row in diagnostic.get("rows", [])] != list(range(256, 264)):
            raise ValueError("SW0175 fixed TRAIN8 diagnostic IDs/positions incomplete")
        if [row.get("image_id") for row in diagnostic["rows"]] != [int(ordered_ids[i]) for i in range(256, 264)]:
            raise ValueError("SW0175 fixed TRAIN8 diagnostic image IDs differ from source contract")
    return manifest


def _state_sha_from_state(payload):
    digest = hashlib.sha256()
    for section in ("wrapped", "encoder"):
        for name, value in sorted(payload[section].items()):
            digest.update((section + "." + name).encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _validate_optimizer_state(payload, expected):
    if not isinstance(expected, dict) or not isinstance(payload.get("param_groups"), list):
        raise ValueError("SW0175 optimizer state/summary is malformed")
    actual = {}
    for group in payload["param_groups"]:
        name = group.get("name")
        names = group.get("parameter_names")
        if not isinstance(name, str) or not isinstance(names, list) or len(names) != len(group["params"]):
            raise ValueError("SW0175 serialized optimizer family schema is invalid")
        steps, updated = [], []
        for pname, pid in zip(names, group["params"]):
            if pid in payload["state"]:
                step = int(payload["state"][pid]["step"].item())
                steps.append(step); updated.append(pname)
                if not 1 <= step <= UPDATES:
                    raise ValueError("SW0175 optimizer parameter step is outside training cursor")
        actual[name] = {"parameter_count": len(names), "updated_parameter_count": len(steps),
            "unused_parameter_count": len(names) - len(steps),
            "updated_parameter_names": updated,
            "unused_parameter_names": [name for name in names if name not in updated],
            "steps": sorted(steps), "max_step": max(steps, default=0)}
    if actual != expected:
        raise ValueError("SW0175 optimizer summary differs from serialized Adam state")


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(argv)
    manifest = train(args.arm, physical_gpu=args.physical_gpu, output_root=args.output_root)
    print(json.dumps({"status": manifest["status"], "arm": manifest["arm"],
                      "completed_updates": manifest["completed_updates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

