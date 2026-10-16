"""Bounded SW0176 CPU admission and one-example full-window gradient preflight."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np  # Initialize NumPy/OpenMP before torch on Windows.
import torch

from collaborative_test.SW_0175_critical_component_binding import train_v2 as train

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENT = "SW0176_local_phase_credit_preflight"
STEPS = 1024
SETTLE = 512
TRAIN_COUNT = 512
POSITION = 0
ARM = "critical_component"


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_tensor(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _fingerprint_paths():
    return tuple(ROOT / rel for rel in (
        "collaborative_test/SW_0176_local_phase_credit/DESIGN_KO.md",
        "collaborative_test/SW_0176_local_phase_credit/policy.py",
        "collaborative_test/SW_0176_local_phase_credit/preflight_v2.py",
        "collaborative_test/SW_0176_local_phase_credit/test_policy.py",
        "collaborative_test/SW_0176_local_phase_credit/test_preflight_v2.py",
        "collaborative_test/SW_0176_local_phase_credit/resource_queue_v2.py",
        "collaborative_test/SW_0176_local_phase_credit/test_resource_queue_v2.py",
        "collaborative_test/SW_0175_critical_component_binding/DESIGN_KO.md",
        "collaborative_test/SW_0175_critical_component_binding/train_v2.py",
        "collaborative_test/SW_0175_critical_component_binding/critical_component_objective.py",
        "collaborative_test/SW_0171_contextual_instance_graph/contextual_encoder.py",
        "collaborative_test/SW_0169_normalized_hard_reset/calibration_v2.py",
        "collaborative_test/SW_0134_native_spike_binding/rollout.py",
        "collaborative_test/SW_0135_native32_spike_binding/foundation.py",
        "collaborative_test/SW_0136_native32_transfer/evaluate.py",
        "collaborative_test/SW_0143_native32_selective_geodesic/geodesic_operator.py",
        "collaborative_test/SW_0161_supervised_spike_partition/cache.py",
        "collaborative_test/SW_0162_connected_signed_partition/preflight.py",
        "collaborative_test/SW_0164_foreground_preserving_partition/foreground_signed.py",
        "collaborative_test/SW_0168_end_to_end_connectivity/connectivity_objective.py",
        "snn_kuramoto_bidirectional/kuramoto_layer.py",
        "snn_kuramoto_bidirectional/graph_generator.py",
        "snn_kuramoto_bidirectional/spike_classifier.py",
    ))


def implementation_fingerprint():
    paths = _fingerprint_paths()
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"SW0176 preflight dependency missing: {missing[0]}")
    return {path.relative_to(ROOT).as_posix(): _sha_file(path) for path in paths}


def _validate_arrays(arrays):
    if not isinstance(arrays, dict):
        raise ValueError("SW0176 admitted input cache has no arrays mapping")
    expected = {
        "image_ids": ((TRAIN_COUNT,), np.dtype(np.int64)),
        "rgb": ((TRAIN_COUNT, 128, 128, 3), np.dtype(np.uint8)),
        "source_support": ((TRAIN_COUNT, 32, 32), None),
        "feasible_targets": ((TRAIN_COUNT, 32, 32), None),
        "modal_gt": ((TRAIN_COUNT, 32, 32), None),
    }
    if set(arrays) != set(expected):
        raise ValueError("SW0176 expected the frozen SW0175 TRAIN-cache array set")
    for key, (shape, dtype) in expected.items():
        value = np.asarray(arrays[key])
        if value.shape != shape or (dtype is not None and value.dtype != dtype):
            raise ValueError(f"SW0176 TRAIN cache {key} has invalid schema")
        if key in {"source_support", "feasible_targets", "modal_gt"}:
            if not np.issubdtype(value.dtype, np.integer) or np.any(value < 0):
                raise ValueError(f"SW0176 TRAIN cache {key} must contain nonnegative integer labels")
    if not np.array_equal(np.asarray(arrays["image_ids"]), np.asarray(arrays["image_ids"], dtype=np.int64)):
        raise ValueError("SW0176 TRAIN IDs are not canonical int64 values")
    return arrays


def _validate_tail_control(manifest, proof):
    if not isinstance(manifest, dict) or manifest.get("status") != "training_complete":
        raise ValueError("SW0176 tail control is not a completed SW0175 artifact")
    bindings = manifest.get("bindings")
    if not isinstance(bindings, dict) or bindings.get("arm") != ARM or bindings.get("seed") != 1:
        raise ValueError("SW0176 tail control arm/seed identity changed")
    required = {
        "source_checkpoint_sha256": proof.get("source_checkpoint_sha256"),
        "source_manifest_sha256": proof.get("source_manifest_sha256"),
        "source_state_sha256": proof.get("source_cache_state_sha256"),
        "input_cache_manifest_sha256": proof.get("producer_manifest_sha256"),
        "input_cache_arrays_sha256": proof.get("producer_arrays_sha256"),
        "training_images_per_pass": 512, "passes": 2, "updates": 64,
        "logical_batch": 16, "microbatch": 4, "optimizer": "Adam",
        "context_seed": 171001, "shuffle_seed": 171001,
    }
    for key, expected in required.items():
        if bindings.get(key) != expected:
            raise ValueError(f"SW0176 SW0175 tail-control binding changed: {key}")
    fingerprint = bindings.get("training_fingerprint")
    if not isinstance(fingerprint, dict):
        raise ValueError("SW0176 tail control lacks its frozen training fingerprint")
    # Bind the complete registered SW0175 implementation/objective and optimizer
    # contract. The only intended difference in SW0176 is the theta gradient
    # intervention, not an unrecorded change to the trained control.
    current_fingerprint = train._training_fingerprint()
    if fingerprint != current_fingerprint:
        raise ValueError("SW0176 tail control training/objective fingerprint differs from frozen SW0175")
    objective_path = "collaborative_test/SW_0175_critical_component_binding/critical_component_objective.py"
    if objective_path not in fingerprint.get("files", {}):
        raise ValueError("SW0176 tail control does not pin the registered objective source")
    expected_fp = {"encoder_lr": train.ENCODER_LR, "graph_core_lr": train.GRAPH_CORE_LR,
                   "integration_lr": train.INTEGRATION_LR, "global_clip_norm": train.CLIP_NORM,
                   "updates": train.UPDATES, "logical_batch": train.BATCH,
                   "microbatch": train.MICRO, "shuffle_seed": train.SHUFFLE_SEED,
                   "context_seed": train.CONTEXT_SEED}
    for key, expected in expected_fp.items():
        if fingerprint.get(key) != expected:
            raise ValueError(f"SW0176 SW0175 tail-control training fingerprint changed: {key}")
    return manifest


def _context_zero_init_sha(admission_report):
    """Read the actual SW0175 producer field (top-level beside initialization)."""
    value = admission_report.get("context_zero_init_sha256") if isinstance(admission_report, dict) else None
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("SW0175 admission report lacks its top-level zero-context initialization digest")
    return value


def prepare_inputs_cpu():
    """Validate the actual SW0175 source/cache/control contract without CUDA."""
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0176 CPU preparation must run before CUDA initialization")
    admitted = train._load_admitted_inputs()
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0175 source admission unexpectedly initialized CUDA")
    arrays = _validate_arrays(admitted["prepared"].get("arrays"))
    proof = admitted.get("proof")
    if not isinstance(proof, dict):
        raise ValueError("SW0176 source admission proof is missing")
    # This complete artifact is only eligible as the registered tail64 control
    # when its exact source, schedule, and optimizer lineage validate.
    control = train.validate_training_artifact(ARM, train.OUTPUT_ROOT / f"{ARM}_seed1")
    _validate_tail_control(control, proof)
    bound_ids = control["bindings"].get("ordered_train_ids")
    actual_ids = np.asarray(arrays["image_ids"], dtype=np.int64).tolist()
    ids_sha = train._sha_bytes(np.asarray(actual_ids, dtype="<i8").tobytes())
    if (not isinstance(bound_ids, list) or len(bound_ids) != TRAIN_COUNT
            or actual_ids != list(map(int, bound_ids))
            or ids_sha != control["bindings"].get("ordered_train_ids_sha256")):
        raise ValueError("SW0176 source cache first TRAIN ID differs from the bound tail-control order")
    source_proof = {key: proof[key] for key in sorted(proof)}
    fingerprint = implementation_fingerprint()
    return {"admitted": admitted, "arrays": arrays, "control_manifest": control,
            "control_bindings": control["bindings"], "source_proof": source_proof,
            "implementation_fingerprint": fingerprint,
            "selected_position": POSITION,
            "selected_image_id": int(np.asarray(arrays["image_ids"])[POSITION]),
            "validation_data_read": False, "optimizer_updates": 0}


def _module(name):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        import importlib
        return importlib.import_module(name)
    finally:
        if visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible


def _build_source(prepared, device):
    calibration = _module("collaborative_test.SW_0169_normalized_hard_reset.calibration_v2")
    operator = _module("collaborative_test.SW_0143_native32_selective_geodesic.geodesic_operator")
    found = calibration._load_source(device)
    operator.install_operator(found.wrapped.core.graph_generator, "T")
    found.encoder.eval(); found.patcher.eval(); found.wrapped.eval()
    if (getattr(found.wrapped.core.graph_generator, "_sw0143_geodesic_arm", None) != "T"
            or found.wrapped.core.kuramoto_backend != "factorized"):
        raise ValueError("SW0176 requires the frozen source's native SW0143-T factorized graph")
    source_sha = calibration._source_digest(found)
    train._validate_source_proof_digest(prepared["source_proof"], source_sha)
    base_encoder = found.encoder
    base_encoder_sha = train.preflight._module_sha(base_encoder)
    admitted_init = prepared["admitted"]["admission_report"].get("initialization", {})
    if (base_encoder_sha != prepared["admitted"]["admission_report"].get("base_encoder_sha256")
            or float(found.feature_clip) != admitted_init.get("feature_clip")
            or train.preflight._integration_values(found.wrapped) != admitted_init.get("integration")):
        raise ValueError("SW0176 fresh Source-T model differs from the admitted SW0175 initialization")
    from collaborative_test.SW_0171_contextual_instance_graph.contextual_encoder import ContextualFeatureEncoder
    found.encoder = ContextualFeatureEncoder(base_encoder, seed=train.CONTEXT_SEED).to(device)
    found.encoder.eval()
    context_sha = train.preflight._module_sha(found.encoder)
    if context_sha != _context_zero_init_sha(prepared["admitted"]["admission_report"]):
        raise ValueError("SW0176 zero-context initialization differs from SW0175 fresh source")
    train._named_families(found)
    init_state_sha = train._state_sha(found.wrapped, found.encoder)
    if init_state_sha != prepared["control_bindings"].get("initial_model_state_sha256"):
        raise ValueError("completed SW0175 tail control does not share the fresh source initialization")
    families = train._named_families(found)
    return found, base_encoder, source_sha, base_encoder_sha, init_state_sha, families


def _trace_hashes(trace, capture):
    return {name: capture._tensor_sha(trace[name]) for name in capture.TRACE_FIELDS}


def _grad_stats(named):
    rows = []
    total_sq = 0.0
    present = 0
    finite = True
    for name, parameter in named:
        grad = parameter.grad
        if grad is None:
            rows.append({"name": name, "present": False, "norm": 0.0})
            continue
        norm = float(torch.linalg.vector_norm(grad.detach().float()).cpu())
        ok = bool(torch.isfinite(grad).all()) and np.isfinite(norm)
        finite = finite and ok
        present += 1
        total_sq += norm * norm
        rows.append({"name": name, "present": True, "norm": norm, "finite": ok})
    return {"parameter_count": len(rows), "gradient_parameter_count": present,
            "gradient_l2": float(total_sq ** 0.5), "finite": finite, "parameters": rows}


def _validate_gradient_evidence(gradient_families, q_gradient_l2, emitted_gradient_l2):
    if not all(isinstance(row, dict) and row.get("finite") is True
               for row in gradient_families.values()):
        raise FloatingPointError("SW0176 full-window parameter-family gradient is nonfinite")
    if not np.isfinite(q_gradient_l2) or q_gradient_l2 <= 0:
        raise FloatingPointError("SW0176 production Q has no finite positive objective gradient")
    if not np.isfinite(emitted_gradient_l2) or emitted_gradient_l2 <= 0:
        raise FloatingPointError("SW0176 emitted component spikes have no finite positive objective gradient")
    for name in ("live_context_encoder", "graph", "kuramoto"):
        row = gradient_families.get(name)
        if not isinstance(row, dict) or not np.isfinite(row.get("gradient_l2", np.nan)) or row["gradient_l2"] <= 0:
            raise FloatingPointError(f"SW0176 {name} family has no finite positive full-window gradient")


def _readout(q, a, gain, support):
    solver = _module("collaborative_test.SW_0164_foreground_preserving_partition.foreground_signed")
    q_np = q.detach().cpu().numpy().astype(np.float64, copy=True)
    a_np = a.detach().cpu().numpy().astype(np.float64, copy=True)
    labels = np.asarray(support, dtype=np.int32)
    result = solver.foreground_induced_merge(q_np, a_np, float(gain), labels,
                                             grid_shape=(32, 32))
    output = np.asarray(result["labels"], dtype=np.int32)
    if output.shape != (32, 32) or not np.array_equal(output > 0, labels > 0):
        raise ValueError("SW0176 Source-T readout changed the foreground mask")
    return output, int(result["merge_count"])


def run_preflight(*, device="cuda:0", prepared=None, output):
    """Compare source and local-phase policy on one fixed TRAIN image; no updates."""
    prepared = prepare_inputs_cpu() if prepared is None else prepared
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0176 source setup must precede CUDA initialization")
    if torch.device(device).type != "cuda":
        raise ValueError("SW0176 bounded resource preflight requires an exclusively leased CUDA device")
    import time as _time
    torch.set_num_threads(1)
    torch.cuda.reset_peak_memory_stats(torch.device(device))
    start = _time.perf_counter()
    found = base_encoder = None
    original_rng = train.preflight._state_rng(torch.device(device))
    try:
        found, base_encoder, source_sha, base_encoder_sha, init_state_sha, families = _build_source(prepared, device)
        arrays = prepared["arrays"]
        image_id = int(np.asarray(arrays["image_ids"])[POSITION])
        if image_id != prepared["selected_image_id"]:
            raise ValueError("SW0176 selected TRAIN image identity changed")
        image = torch.as_tensor(np.asarray(arrays["rgb"][POSITION]), device=device).unsqueeze(0)
        target = torch.as_tensor(np.asarray(arrays["feasible_targets"][POSITION]),
                                 device=device, dtype=torch.long)
        source_support = np.asarray(arrays["source_support"][POSITION], dtype=np.int32).copy()
        source_state_sha = train._source_sha(found, base_encoder)
        if source_state_sha != source_sha:
            raise ValueError("SW0176 source foundation changed before readout")

        from collaborative_test.SW_0162_connected_signed_partition import preflight as capture
        from collaborative_test.SW_0176_local_phase_credit.policy import detach_previous_theta
        from collaborative_test.SW_0136_native32_transfer import evaluate as sw136
        from collaborative_test.SW_0175_critical_component_binding import critical_component_objective as objective

        baseline_rng = train.preflight._state_rng(torch.device(device))
        with torch.no_grad():
            base_gamma, base_trace, base_a, base_z, base_q, base_rho = train._native_forward(
                found, image, live_tail_steps=0)
        train.preflight._restore_rng(torch.device(device), baseline_rng)
        policy_start = _time.perf_counter()
        with detach_previous_theta(found.wrapped.core.kuramoto) as policy_calls:
            gamma, trace, actual_a, projection, q, rho = train._native_forward(
                found, image, live_tail_steps=STEPS)
        policy_elapsed = _time.perf_counter() - policy_start
        if policy_calls["count"] != STEPS:
            raise ValueError("SW0176 policy did not intercept every full-window Kuramoto transition")

        parity = {"gamma_exact": torch.equal(gamma, base_gamma),
            "six_traces_exact": all(torch.equal(trace[name], base_trace[name]) for name in capture.TRACE_FIELDS),
            "actual_A_exact": torch.equal(actual_a, base_a),
            "projection_exact": torch.equal(projection, base_z), "production_Q_exact": torch.equal(q, base_q)}
        standalone_base = np.asarray(sw136._qcc_labels(base_trace)[0], dtype=np.int32)
        standalone = np.asarray(sw136._qcc_labels(trace)[0], dtype=np.int32)
        if not np.array_equal(standalone, standalone_base):
            raise ValueError("SW0176 source and local-phase policy changed native QCC labels")
        gain = float(torch.exp(found.wrapped.core.graph_generator.log_coupling_gain.detach()).cpu())
        source_base_labels = source_support
        source_labels, source_merge_count = _readout(base_q[0], base_a[0], gain, source_base_labels)
        policy_source_labels, policy_source_merge_count = _readout(q[0], actual_a[0], gain, source_base_labels)
        standalone_labels, standalone_merge_count = _readout(base_q[0], base_a[0], gain, standalone_base)
        policy_standalone_labels, policy_standalone_merge_count = _readout(q[0], actual_a[0], gain, standalone)
        parity["conditional_mask_exact"] = np.array_equal(source_base_labels > 0, source_support > 0)
        parity["standalone_mask_exact"] = np.array_equal(standalone_base > 0, standalone > 0)
        parity["conditional_readout_exact"] = np.array_equal(source_labels, policy_source_labels)
        parity["standalone_readout_exact"] = np.array_equal(standalone_labels, policy_standalone_labels)
        if not all(parity.values()):
            raise ValueError("SW0176 local-phase forward/readout parity failed")

        trace["component_spikes"].retain_grad()
        q.retain_grad()
        loss = train._loss_for_arm(ARM, q[0], rho[0], target, source_support)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("SW0176 one-example critical objective is nonfinite")
        loss.backward()
        gradient_families = {name: _grad_stats(entries) for name, entries in families.items()}
        # Add the actual native stages and phase family explicitly; do not infer
        # family credit from a sibling activation stack.
        core = found.wrapped.core
        gradient_families["kuramoto"] = _grad_stats(list(core.kuramoto.named_parameters()))
        gradient_families["dendrite"] = _grad_stats(list(core.dendric_layer.named_parameters()))
        gradient_families["membrane"] = _grad_stats(list(core.membrane_layer.named_parameters()))
        gradient_families["live_context_encoder"] = _grad_stats(list(found.encoder.context.named_parameters()))
        if q.grad is None or not bool(torch.isfinite(q.grad).all()):
            raise FloatingPointError("SW0176 actual production Q has no finite objective gradient")
        emitted_grad = trace["component_spikes"].grad
        if emitted_grad is None or not bool(torch.isfinite(emitted_grad).all()):
            raise FloatingPointError("SW0176 actual emitted component spikes have no finite loss gradient")
        q_gradient_l2 = float(torch.linalg.vector_norm(q.grad.detach().float()).cpu())
        emitted_gradient_l2 = float(torch.linalg.vector_norm(emitted_grad.detach().float()).cpu())
        _validate_gradient_evidence(gradient_families, q_gradient_l2, emitted_gradient_l2)
        source_after_sha = train._source_sha(found, base_encoder)
        if source_after_sha != source_sha:
            raise ValueError("SW0176 read-only source preflight changed foundation state")
        report = {"experiment": EXPERIMENT, "status": "complete_source_preflight",
            "arm": ARM, "seed": 1, "source_proof": prepared["source_proof"],
            "implementation_fingerprint": prepared["implementation_fingerprint"],
            "control_training_manifest_sha256": train._sha_file(
                train.OUTPUT_ROOT / f"{ARM}_seed1" / "manifest.json"),
            "control_initial_model_state_sha256": prepared["control_bindings"]["initial_model_state_sha256"],
            "fresh_initial_model_state_sha256": init_state_sha,
            "source_state_sha256_before": source_sha, "source_state_sha256_after": source_after_sha,
            "base_encoder_sha256": base_encoder_sha, "selected_train_position": POSITION,
            "selected_train_image_id": image_id, "steps": STEPS, "settle": SETTLE,
            "live_tail_steps": STEPS, "kuramoto_theta_prev_detach_calls": policy_calls["count"],
            "forward_parity": parity, "baseline_trace_sha256": _trace_hashes(base_trace, capture),
            "policy_trace_sha256": _trace_hashes(trace, capture),
            "baseline_q_sha256": _sha_tensor(base_q), "policy_q_sha256": _sha_tensor(q),
            "baseline_actual_A_sha256": _sha_tensor(base_a), "policy_actual_A_sha256": _sha_tensor(actual_a),
            "baseline_projection_sha256": _sha_tensor(base_z), "policy_projection_sha256": _sha_tensor(projection),
            "source_support_sha256": hashlib.sha256(source_support.astype("<i4",copy=False).tobytes()).hexdigest(),
            "standalone_qcc_sha256": hashlib.sha256(standalone.astype("<i4",copy=False).tobytes()).hexdigest(),
            "readouts": {"conditional": {"foreground_exact": parity["conditional_mask_exact"],
                "merge_count": source_merge_count, "policy_merge_count": policy_source_merge_count},
                "standalone": {"foreground_exact": parity["standalone_mask_exact"],
                "merge_count": standalone_merge_count, "policy_merge_count": policy_standalone_merge_count}},
            "objective": "SW0175 critical_component_loss", "loss": float(loss.detach().cpu()),
            "gradient_families": gradient_families,
            "q_gradient_l2": q_gradient_l2,
            "emitted_component_spike_gradient_l2": emitted_gradient_l2,
            "elapsed_seconds": _time.perf_counter()-start, "policy_rollout_seconds": policy_elapsed,
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(torch.device(device))),
            "optimizer_updates": 0, "validation_data_read": False,
            "ground_truth_scope": "one registered TRAIN row feasible target/source support only",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
        output = Path(output)
        train.preflight._write_report(output, report)
        return report
    finally:
        train.preflight._restore_rng(torch.device(device), original_rng)
        if found is not None and base_encoder is not None:
            # Keep the foundation object as it was (the temporary context wrapper
            # is only an inference adapter, never a checkpoint mutation).
            found.encoder = base_encoder
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
