"""One-image resource/parity probe for native full-horizon Q gradients.

This module measures a single frozen-source initialization only. It never
steps an optimizer, reads validation data, or chooses among epsilon values.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import time

import numpy as np  # Import NumPy before Torch on Windows.
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = HERE / "results_archive" / "resource_probe_20261015" / "report.json"
TRAIN = "collaborative_test.SW_0171_contextual_instance_graph.train_prefix"
DESIGN = HERE / "DESIGN_KO.md"
TRACE_FIELDS = ("theta", "membrane", "spikes", "component_membrane",
                "component_spikes", "component_gates")


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _import(name):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        return importlib.import_module(name)
    finally:
        if visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible


def fingerprint():
    train = _import(TRAIN)
    dependencies = {
        "collaborative_test/SW_0173_gradient_fidelity/DESIGN_KO.md": DESIGN,
        "collaborative_test/SW_0173_gradient_fidelity/gradient_fidelity.py": HERE / "gradient_fidelity.py",
        "collaborative_test/SW_0173_gradient_fidelity/resource_probe.py": HERE / "resource_probe.py",
        "collaborative_test/SW_0173_gradient_fidelity/resource_queue.py": HERE / "resource_queue.py",
        "collaborative_test/SW_0173_gradient_fidelity/test_resource_queue.py": HERE / "test_resource_queue.py",
        "collaborative_test/SW_0171_contextual_instance_graph/train_prefix.py": ROOT / "collaborative_test/SW_0171_contextual_instance_graph/train_prefix.py",
    }
    hashes = {}
    for relative, path in dependencies.items():
        if not path.is_file():
            raise FileNotFoundError(f"SW0173 resource-probe dependency missing: {relative}")
        hashes[relative] = _sha_file(path)
    return {"files": hashes, "sw0171_training_fingerprint": train._training_fingerprint(),
            "experiment": "SW0173_native_Q_gradient_horizon_resource_probe",
            "position": 0, "batch": 1, "steps": 1024,
            "tail_live_steps": 64, "full_live_steps": 1024,
            "optimizer_updates": 0, "validation_data_read": False}


def prepare_cpu():
    """Load the existing admitted SW0171 TRAIN cache before CUDA initialization."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0173 input preparation must precede CUDA initialization")
    train = _import(TRAIN)
    admitted = train._load_admitted_inputs()
    if torch.cuda.is_initialized() or os.environ.get("CUDA_VISIBLE_DEVICES") != visible:
        raise RuntimeError("SW0171 input preparation changed CUDA state or visibility")
    prepared = admitted["prepared"]
    arrays = prepared["arrays"]
    expected = {"rgb": ((512, 128, 128, 3), np.uint8),
                "source_support": ((512, 32, 32), np.int32),
                "feasible_targets": ((512, 32, 32), np.int32)}
    for key, (shape, dtype) in expected.items():
        value = np.asarray(arrays.get(key))
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f"admitted SW0171 {key} cache has wrong schema")
    ids = list(map(int, admitted["proof"]["ordered_train_ids"]))
    if len(ids) != 512 or len(set(ids)) != 512:
        raise ValueError("SW0173 requires the exact ordered TRAIN512 admission")
    return {"admitted": admitted, "source_proof": admitted["proof"],
            "position": 0, "image_id": ids[0],
            "input_cache_manifest_sha256": prepared["cache_manifest_sha256"],
            "input_cache_arrays_sha256": prepared["arrays_sha256"],
            "cuda_initialized": False, "cuda_visible_devices": visible,
            "validation_data_read": False}


def validate_native_trace(trace, *, batch, steps=1024):
    """Validate exact SW0134 native tensor axes before affinity calculation."""
    if not isinstance(trace, dict) or not set(TRACE_FIELDS).issubset(trace):
        raise ValueError("native forward omitted one or more trace fields")
    expected_shapes = {
        "theta": (batch, steps, 1024, 4),
        "membrane": (batch, 1024, steps),
        "spikes": (batch, 1024, steps),
        "component_membrane": (batch, 4, 1024, steps),
        "component_spikes": (batch, 4, 1024, steps),
        "component_gates": (batch, 4, 1024, steps),
    }
    for name, expected in expected_shapes.items():
        value = trace[name]
        if not torch.is_tensor(value) or tuple(value.shape) != expected:
            got = tuple(value.shape) if torch.is_tensor(value) else type(value).__name__
            raise ValueError(f"native {name} has geometry {got}; expected {expected}")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"native {name} contains nonfinite values")
    return expected_shapes


def _trace_copy(trace):
    return {name: trace[name].detach().cpu().contiguous().clone() for name in TRACE_FIELDS}


def _tensor_sha(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _gradient_report(families):
    output = {}
    for family, entries in families.items():
        values = []
        total_sq = 0.0
        for name, parameter in entries:
            grad = parameter.grad
            norm = 0.0 if grad is None else float(grad.detach().double().norm().cpu())
            if not np.isfinite(norm):
                raise FloatingPointError(f"nonfinite gradient in {family}.{name}")
            total_sq += norm * norm
            values.append({"name": name, "gradient_l2": norm,
                           "gradient_present": grad is not None})
        family_norm = float(np.sqrt(total_sq))
        if not np.isfinite(family_norm):
            raise FloatingPointError(f"nonfinite aggregate gradient for {family}")
        output[family] = {"parameter_count": len(entries), "gradient_l2": family_norm,
                          "parameters": values}
    return output


def _same_forward(tail, full):
    fields = {name: bool(torch.equal(tail[name], full[name])) for name in TRACE_FIELDS}
    return {"trace_fields_exact": fields, "all_trace_fields_exact": all(fields.values())}


def _restore_rng(train, device, state):
    train.preflight._restore_rng(device, state)


def _is_oom(error):
    if isinstance(error, getattr(torch.cuda, "OutOfMemoryError", ())):
        return True
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def build_source(prepared, *, device="cpu"):
    """Construct the fresh, zero-residual SW0171 source model for read-only checks."""
    dev = torch.device(device)
    if dev.type == "cpu" and torch.cuda.is_initialized():
        raise RuntimeError("CPU source construction must precede CUDA initialization")
    train = _import(TRAIN)
    source_cal = _import("collaborative_test.SW_0169_normalized_hard_reset.calibration_v2")
    found = source_cal._load_source(dev)
    found.encoder.eval(); found.patcher.eval(); found.wrapped.eval()
    source_before = train._source_sha(found, found.encoder)
    train._validate_source_proof_digest(prepared["source_proof"], source_before)
    base_encoder = found.encoder
    admitted = prepared["admitted"]
    admission = admitted["admission_report"]
    init = admission.get("initialization", {})
    if (train.preflight._module_sha(base_encoder) != admission.get("base_encoder_sha256")
            or train.preflight._integration_values(found.wrapped) != init.get("integration")
            or float(found.feature_clip) != init.get("feature_clip")):
        raise ValueError("fresh SW0171 source differs from the admitted original initialization")
    from collaborative_test.SW_0171_contextual_instance_graph.contextual_encoder import ContextualFeatureEncoder
    found.encoder = ContextualFeatureEncoder(base_encoder, seed=171001).to(dev)
    found.encoder.eval()
    context_sha = train.preflight._module_sha(found.encoder)
    if context_sha != admission.get("context_zero_init_sha256"):
        raise ValueError("fresh zero-residual SW0171 encoder differs from source admission")
    optimizer, families = train._make_optimizer(found)
    return {"train": train, "source_calibration": source_cal, "found": found,
            "base_encoder": base_encoder, "source_state_sha256": source_before,
            "context_zero_init_sha256": context_sha,
            "optimizer": optimizer, "families": families}


def run_probe(*, prepared=None, device="cuda:0", output=OUTPUT):
    """Run tail/full parity and one full-horizon Q backward on TRAIN position 0."""
    started = time.perf_counter()
    prepared = prepare_cpu() if prepared is None else prepared
    if prepared.get("cuda_initialized") is not False or prepared.get("validation_data_read") is not False:
        raise ValueError("SW0173 probe input proof exceeds its CPU/TRAIN-only scope")
    fingerprint_value = fingerprint()
    admitted = prepared["admitted"]
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0173 worker must own the first CUDA initialization")
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0173 resource probe requires the assigned CUDA device")
    torch.cuda.set_device(dev)
    torch.set_num_threads(1)

    report = {"experiment": fingerprint_value["experiment"], "status": "running",
              "implementation_fingerprint": fingerprint_value,
              "source_proof": prepared["source_proof"], "position": 0,
              "image_id": prepared["image_id"], "input_cache_manifest_sha256": prepared["input_cache_manifest_sha256"],
              "input_cache_arrays_sha256": prepared["input_cache_arrays_sha256"],
              "optimizer_updates": 0, "validation_data_read": False,
              "ground_truth_scope": "TRAIN position 0 feasible target and source support only",
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "tail_live_steps": 64, "full_live_steps": 1024}
    source_before = None
    try:
        model = build_source(prepared, device=dev)
        train, source_cal = model["train"], model["source_calibration"]
        found, base_encoder = model["found"], model["base_encoder"]
        source_before = model["source_state_sha256"]
        context_sha = model["context_zero_init_sha256"]
        optimizer, families = model["optimizer"], model["families"]
        optimizer.zero_grad(set_to_none=True)

        arrays = admitted["prepared"]["arrays"]
        rgb = torch.from_numpy(np.ascontiguousarray(arrays["rgb"][:1])).to(dev)
        target = arrays["feasible_targets"][0]
        support = arrays["source_support"][0]
        rng = train.preflight._state_rng(dev)
        start_tail = time.perf_counter()
        # Both horizons use the same grad-enabled forward mode. The tail graph
        # is discarded without a backward; the full graph alone is consumed.
        with torch.enable_grad():
            _g_tail, trace_tail, _a_tail, _z_tail, q_tail = train._native_forward(
                found, rgb, live_tail_steps=64)
        validate_native_trace(trace_tail, batch=1)
        tail_cpu = _trace_copy(trace_tail)
        q_tail_cpu = q_tail.detach().cpu().clone()
        gamma_tail_cpu = _g_tail.detach().cpu().clone()
        adjacency_tail_cpu = _a_tail.detach().cpu().clone()
        z_tail_cpu = _z_tail.detach().cpu().clone()
        tail_elapsed = time.perf_counter() - start_tail
        del trace_tail, q_tail, _g_tail, _a_tail, _z_tail

        _restore_rng(train, dev, rng)
        torch.cuda.reset_peak_memory_stats(dev)
        start_full = time.perf_counter()
        _g_full, trace_full, _a_full, z_full, q_full = train._native_forward(
            found, rgb, live_tail_steps=1024)
        validate_native_trace(trace_full, batch=1)
        full_cpu = _trace_copy(trace_full)
        parity = _same_forward(tail_cpu, full_cpu)
        parity["q_exact"] = bool(torch.equal(q_tail_cpu, q_full.detach().cpu()))
        parity["gamma_exact"] = bool(torch.equal(gamma_tail_cpu, _g_full.detach().cpu()))
        parity["actual_coupling_exact"] = bool(torch.equal(adjacency_tail_cpu, _a_full.detach().cpu()))
        parity["projection_exact"] = bool(torch.equal(z_tail_cpu, z_full.detach().cpu()))
        if not parity["all_trace_fields_exact"] or not parity["q_exact"]:
            raise ValueError("tail64 and full1024 source forward parity differed at the same RNG")
        if not parity["gamma_exact"] or not parity["actual_coupling_exact"] or not parity["projection_exact"]:
            raise ValueError("tail64 and full1024 gamma/A/z parity differed at the same RNG")

        objective = _import("collaborative_test.SW_0168_end_to_end_connectivity.connectivity_objective")
        q_loss, _ = train._loss_for_arm("Q_only", q_full[0], z_full[0], target,
                                        support, None, objective)
        if not bool(torch.isfinite(q_loss.detach())):
            raise FloatingPointError("full-horizon TRAIN Q loss is nonfinite")
        q_loss.backward()
        gradient_rows = _gradient_report(families)
        full_elapsed = time.perf_counter() - start_full
        source_after = train._source_sha(found, base_encoder)
        report.update({"status": "complete_resource_probe",
            "source_state_before_sha256": source_before,
            "source_state_after_sha256": source_after,
            "source_state_unchanged": source_before == source_after,
            "base_encoder_sha256": train.preflight._module_sha(base_encoder),
            "context_zero_init_sha256": context_sha,
            "tail_forward_seconds": tail_elapsed, "full_forward_backward_seconds": full_elapsed,
            "q_loss": float(q_loss.detach().cpu()), "gradient_families": gradient_rows,
            "trace_parity": parity,
            "gamma_sha256": _tensor_sha(_g_full),
            "actual_coupling_sha256": _tensor_sha(_a_full),
            "projection_sha256": _tensor_sha(z_full),
            "tail_trace_sha256": {name: _tensor_sha(value) for name, value in tail_cpu.items()},
            "full_trace_sha256": {name: _tensor_sha(value) for name, value in full_cpu.items()},
            "q_sha256": _tensor_sha(q_full),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(dev)),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(dev)),
            "cuda_initialized": bool(torch.cuda.is_initialized()),
            "elapsed_seconds": time.perf_counter() - started})
        if source_after != source_before:
            report["status"] = "resource_rejected"
            report["rejection_reason"] = "original_source_state_changed"
    except (FloatingPointError, torch.cuda.OutOfMemoryError) as exc:
        report.update({"status": "resource_rejected", "rejection_reason": str(exc),
                       "elapsed_seconds": time.perf_counter() - started,
                       "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(dev)),
                       "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(dev))})
    except RuntimeError as exc:
        if not _is_oom(exc):
            raise
        report.update({"status": "resource_rejected", "rejection_reason": str(exc),
                       "elapsed_seconds": time.perf_counter() - started,
                       "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(dev)),
                       "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(dev))})
    finally:
        # Measurement uses gradients only; no optimizer state or model tensor is stepped.
        if "optimizer" in locals():
            optimizer.zero_grad(set_to_none=True)
        if "found" in locals() and source_before is not None:
            final_source_sha = train._source_sha(found, base_encoder)
            report["source_state_after_finally_sha256"] = final_source_sha
            if final_source_sha != source_before:
                report["status"] = "resource_rejected"
                report["source_state_unchanged"] = False
                report["rejection_reason"] = "original_source_state_changed_during_probe"
        if "rng" in locals():
            _restore_rng(train, dev, rng)
    report_module = train.preflight
    return report_module._write_report(Path(output), report)
