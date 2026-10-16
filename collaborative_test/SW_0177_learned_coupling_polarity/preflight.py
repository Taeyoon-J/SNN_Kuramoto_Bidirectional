"""Bounded SW0177 source-backed, one-TRAIN-image polarity admission."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from collaborative_test.SW_0175_critical_component_binding import train_v2 as train
from collaborative_test.SW_0177_learned_coupling_polarity.polarity_operator import (
    install_polarity_prepare, seeded_edge_multiplier,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENT = "SW0177_learned_coupling_polarity_source_preflight"
POSITION = 0
ARM = "signed"
STEPS, SETTLE, TAIL = 1024, 512, 64
SEED = 177001


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint_paths():
    return (
        "collaborative_test/SW_0177_learned_coupling_polarity/DESIGN_KO.md",
        "collaborative_test/SW_0177_learned_coupling_polarity/polarity_operator.py",
        "collaborative_test/SW_0177_learned_coupling_polarity/preflight.py",
        "collaborative_test/SW_0177_learned_coupling_polarity/resource_queue.py",
        "collaborative_test/SW_0177_learned_coupling_polarity/test_polarity_operator.py",
        "collaborative_test/SW_0177_learned_coupling_polarity/test_preflight.py",
        "collaborative_test/SW_0177_learned_coupling_polarity/test_resource_queue.py",
        "collaborative_test/SW_0175_critical_component_binding/DESIGN_KO.md",
        "collaborative_test/SW_0175_critical_component_binding/train_v2.py",
        "collaborative_test/SW_0175_critical_component_binding/critical_component_objective.py",
        "collaborative_test/SW_0171_contextual_instance_graph/contextual_encoder.py",
        "collaborative_test/SW_0171_contextual_instance_graph/preflight.py",
        "collaborative_test/SW_0169_normalized_hard_reset/calibration_v2.py",
        "collaborative_test/SW_0134_native_spike_binding/rollout.py",
        "collaborative_test/SW_0135_native32_spike_binding/foundation.py",
        "collaborative_test/SW_0136_native32_transfer/evaluate.py",
        "collaborative_test/SW_0143_native32_selective_geodesic/geodesic_operator.py",
        "collaborative_test/SW_0162_connected_signed_partition/preflight.py",
        "collaborative_test/SW_0164_foreground_preserving_partition/foreground_signed.py",
        "collaborative_test/SW_0168_end_to_end_connectivity/connectivity_objective.py",
        "snn_kuramoto_bidirectional/graph_generator.py",
        "snn_kuramoto_bidirectional/kuramoto_layer.py",
        "snn_kuramoto_bidirectional/spike_classifier.py",
    )


def implementation_fingerprint():
    missing = [rel for rel in _fingerprint_paths() if not (ROOT / rel).is_file()]
    if missing:
        raise FileNotFoundError(f"SW0177 implementation dependency missing: {missing[0]}")
    return {rel: _sha_file(ROOT / rel) for rel in _fingerprint_paths()}


def prepare_inputs_cpu():
    """Run the registered SW0175 source/cache admission before CUDA setup."""
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0177 source preparation must precede CUDA initialization")
    admitted = train._load_admitted_inputs()
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0175 source admission unexpectedly initialized CUDA")
    arrays = admitted["prepared"]["arrays"]
    required = {"image_ids", "rgb", "source_support", "feasible_targets"}
    if not isinstance(arrays, dict) or not required.issubset(arrays):
        raise ValueError("SW0177 requires the admitted SW0175 TRAIN cache arrays")
    image_ids = np.asarray(arrays["image_ids"])
    rgb = np.asarray(arrays["rgb"])
    if (image_ids.shape != (512,) or image_ids.dtype != np.int64
            or rgb.shape != (512, 128, 128, 3) or rgb.dtype != np.uint8):
        raise ValueError("SW0177 admitted TRAIN identity/RGB schema changed")
    for key in ("source_support", "feasible_targets"):
        value = np.asarray(arrays[key])
        if value.shape != (512, 32, 32) or not np.issubdtype(value.dtype, np.integer):
            raise ValueError(f"SW0177 admitted TRAIN {key} schema changed")
    prepared = {"admitted": admitted, "arrays": arrays,
        "source_proof": admitted["proof"], "selected_position": POSITION,
        "selected_image_id": int(image_ids[POSITION]),
        "implementation_fingerprint": implementation_fingerprint(),
        "optimizer_updates": 0, "validation_data_read": False}
    return prepared


def _module(name):
    import importlib
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        return importlib.import_module(name)
    finally:
        if visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible


def _context_sha(report):
    value = report.get("context_zero_init_sha256") if isinstance(report, dict) else None
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("SW0177 SW0175 admission lacks zero-context initialization SHA")
    return value


def _build_source(prepared, device):
    calibration = _module("collaborative_test.SW_0169_normalized_hard_reset.calibration_v2")
    geodesic = _module("collaborative_test.SW_0143_native32_selective_geodesic.geodesic_operator")
    found = calibration._load_source(device)
    geodesic.install_operator(found.wrapped.core.graph_generator, "T")
    found.encoder.eval(); found.patcher.eval(); found.wrapped.eval()
    if (getattr(found.wrapped.core.graph_generator, "_sw0143_geodesic_arm", None) != "T"
            or found.wrapped.core.kuramoto_backend != "factorized"):
        raise ValueError("SW0177 requires the original SW0143-T factorized SourceT path")
    source_sha = calibration._source_digest(found)
    train._validate_source_proof_digest(prepared["source_proof"], source_sha)
    base_encoder = found.encoder
    admission = prepared["admitted"]["admission_report"]
    if (train.preflight._module_sha(base_encoder) != admission.get("base_encoder_sha256")
            or float(found.feature_clip) != admission.get("initialization", {}).get("feature_clip")
            or train.preflight._integration_values(found.wrapped)
               != admission.get("initialization", {}).get("integration")):
        raise ValueError("SW0177 fresh SourceT base/preprocessing/integration differs from admission")
    from collaborative_test.SW_0171_contextual_instance_graph.contextual_encoder import ContextualFeatureEncoder
    found.encoder = ContextualFeatureEncoder(base_encoder, seed=train.CONTEXT_SEED).to(device)
    found.encoder.eval()
    if train.preflight._module_sha(found.encoder) != _context_sha(admission):
        raise ValueError("SW0177 fresh contextual source differs from admitted zero-context initialization")
    train._named_families(found)
    state_sha = train._state_sha(found.wrapped, found.encoder)
    return found, base_encoder, source_sha, state_sha


def build_source(prepared, device="cpu"):
    """Public actual-model factory for CPU admission inspection (no rollout)."""
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0177 CPU model factory must precede CUDA initialization")
    return _build_source(prepared, device)


def _trace_sha(trace):
    fields = ("theta", "membrane", "spikes", "component_membrane",
              "component_spikes", "component_gates")
    return {name: train._tensor_sha(trace[name]) for name in fields}


def _readout(q, a, gain, support):
    solver = _module("collaborative_test.SW_0164_foreground_preserving_partition.foreground_signed")
    labels = np.asarray(support, dtype=np.int32)
    result = solver.foreground_induced_merge(q.detach().cpu().numpy().astype(np.float64),
        a.detach().cpu().numpy().astype(np.float64), float(gain), labels,
        grid_shape=(32, 32))
    output = np.asarray(result["labels"], dtype=np.int32)
    if output.shape != (32, 32) or not np.array_equal(output > 0, labels > 0):
        raise ValueError("SW0177 conditional readout changed SourceT foreground support")
    return output


def run_preflight(*, device="cuda:0", prepared=None, output):
    """Compare original SourceT and a zero-head signed arm on TRAIN position 0."""
    prepared = prepare_inputs_cpu() if prepared is None else prepared
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0177 source checks must finish before CUDA initialization")
    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError("SW0177 source-backed preflight requires an exclusive CUDA lease")
    if prepared.get("implementation_fingerprint") != implementation_fingerprint():
        raise ValueError("SW0177 source code changed after CPU admission")
    torch.set_num_threads(1)
    torch.cuda.set_device(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    start = time.perf_counter()
    original_rng = train.preflight._state_rng(dev)
    found = base_encoder = None
    try:
        found, base_encoder, source_sha, initial_state_sha = _build_source(prepared, dev)
        arrays = prepared["arrays"]
        position = prepared["selected_position"]
        image_id = int(np.asarray(arrays["image_ids"])[position])
        if image_id != prepared["selected_image_id"]:
            raise ValueError("SW0177 selected TRAIN identity changed")
        rgb = torch.as_tensor(np.asarray(arrays["rgb"][position]), device=dev).unsqueeze(0)
        target = np.asarray(arrays["feasible_targets"][position], dtype=np.int32)
        support = np.asarray(arrays["source_support"][position], dtype=np.int32)
        head = seeded_edge_multiplier(seed=SEED, mode="signed").to(dev)
        head.eval()
        source_before = train._source_sha(found, base_encoder)
        if source_before != source_sha:
            raise ValueError("SW0177 original source changed before rollout")

        from collaborative_test.SW_0136_native32_transfer.evaluate import _qcc_labels
        from collaborative_test.SW_0162_connected_signed_partition import preflight as coupling_capture
        from collaborative_test.SW_0175_critical_component_binding import critical_component_objective as objective

        initial_rng = train.preflight._state_rng(dev)
        gamma0, trace0, a0, z0, q0, rho0 = train._native_forward(found, rgb, live_tail_steps=TAIL)
        source_trace_hash = _trace_sha(trace0)
        source_gamma_hash = train._tensor_sha(gamma0)
        source_a_hash, source_z_hash = train._tensor_sha(a0), train._tensor_sha(z0)
        source_q_hash = train._tensor_sha(q0)
        source_qcc = np.asarray(_qcc_labels(trace0)[0], dtype=np.int32)
        bound_gain = float(prepared["admitted"]["prepared"]["producer_bindings"]["source_gain"])
        actual_gain = float(torch.exp(found.wrapped.core.graph_generator.log_coupling_gain.detach()).cpu())
        if not np.isclose(actual_gain, bound_gain, rtol=1e-7, atol=0.0):
            raise ValueError("SW0177 loaded source gain differs from admitted source gain")
        source_conditional = _readout(q0[0], a0[0], bound_gain, support)
        source_standalone = _readout(q0[0], a0[0], bound_gain, source_qcc)
        source_conditional_sha = hashlib.sha256(source_conditional.astype("<i4", copy=False).tobytes()).hexdigest()
        source_standalone_sha = hashlib.sha256(source_standalone.astype("<i4", copy=False).tobytes()).hexdigest()
        source_qcc_sha = hashlib.sha256(source_qcc.astype("<i4", copy=False).tobytes()).hexdigest()
        del gamma0, trace0, a0, z0, q0, rho0, source_qcc, source_conditional, source_standalone

        train.preflight._restore_rng(dev, initial_rng)
        with install_polarity_prepare(found.wrapped.core.graph_generator,
                                      found.wrapped.core.kuramoto, head):
            gamma, trace, a, z, q, rho = train._native_forward(found, rgb, live_tail_steps=TAIL)
        trace_hash = _trace_sha(trace)
        parity = {
            "gamma_exact": train._tensor_sha(gamma) == source_gamma_hash,
            "six_traces_exact": trace_hash == source_trace_hash,
            "actual_A0_exact": train._tensor_sha(a) == source_a_hash,
            "projection_z_exact": train._tensor_sha(z) == source_z_hash,
            "production_Q_exact": train._tensor_sha(q) == source_q_hash,
        }
        source_qcc = np.asarray(_qcc_labels(trace)[0], dtype=np.int32)
        conditional = _readout(q[0], a[0], bound_gain, support)
        standalone = _readout(q[0], a[0], bound_gain, source_qcc)
        parity["conditional_labels_exact"] = hashlib.sha256(
            conditional.astype("<i4", copy=False).tobytes()).hexdigest() == source_conditional_sha
        parity["standalone_qcc_exact"] = hashlib.sha256(
            source_qcc.astype("<i4", copy=False).tobytes()).hexdigest() == source_qcc_sha
        parity["standalone_labels_exact"] = hashlib.sha256(
            standalone.astype("<i4", copy=False).tobytes()).hexdigest() == source_standalone_sha
        parity["conditional_foreground_exact"] = np.array_equal(conditional > 0, support > 0)
        parity["standalone_qcc_nonempty"] = bool(np.any(source_qcc > 0))
        if not all(parity.values()):
            raise ValueError("SW0177 zero-head SourceT forward/readout parity failed")

        loss = train._loss_for_arm("critical_component", q[0], rho[0], target, support, objective)
        if loss.ndim != 0 or not bool(torch.isfinite(loss)):
            raise FloatingPointError("SW0177 actual TRAIN critical loss is nonfinite")
        final = head.layers[-1]
        hidden = head.layers[0]
        emitted_gradient, head_weight_gradient, head_bias_gradient, hidden_weight_gradient, hidden_bias_gradient = torch.autograd.grad(
            loss, (trace["component_spikes"], final.weight, final.bias,
                   hidden.weight, hidden.bias), retain_graph=False, allow_unused=True)
        gradient_info = {}
        for label, grad in (("emitted_component_spikes", emitted_gradient),
                            ("head_final_weight", head_weight_gradient),
                            ("head_final_bias", head_bias_gradient)):
            finite = bool(grad is not None and torch.isfinite(grad).all())
            norm = float(torch.linalg.vector_norm(grad.detach().double()).cpu()) if finite else None
            gradient_info[label] = {"present": grad is not None, "finite": finite, "l2": norm}
        hidden_grads = (hidden_weight_gradient, hidden_bias_gradient)
        hidden_finite = all(g is None or bool(torch.isfinite(g).all()) for g in hidden_grads)
        gradient_info["hidden_layer_initial_gradient"] = {
            "all_finite": bool(hidden_finite),
            "l2": (float(sum(float(torch.linalg.vector_norm(g.detach().double()).cpu())
                             for g in hidden_grads if g is not None)) if hidden_finite else None),
            "expected_zero_at_zero_final_layer": True}
        final_norm = math.sqrt(sum((row["l2"] or 0.0) ** 2 for row in
            (gradient_info["head_final_weight"], gradient_info["head_final_bias"])))
        gradient_info["head_final_layer_combined_l2"] = final_norm
        reasons = []
        for label in ("emitted_component_spikes",):
            row = gradient_info[label]
            if not row["present"] or not row["finite"] or row["l2"] <= 0:
                reasons.append(f"{label} lacks finite positive gradient credit")
        if (not gradient_info["head_final_weight"]["finite"]
                or not gradient_info["head_final_bias"]["finite"] or final_norm <= 0):
            reasons.append("head final layer lacks finite positive combined gradient credit")
        if not gradient_info["hidden_layer_initial_gradient"]["all_finite"]:
            reasons.append("zero-final head hidden gradient is nonfinite")
        if not gradient_info["emitted_component_spikes"]["finite"]:
            reasons.append("emitted component-spike gradient is nonfinite")
        source_after = train._source_sha(found, base_encoder)
        if source_after != source_sha:
            raise ValueError("SW0177 read-only preflight mutated the original source foundation")
        report = {"experiment": EXPERIMENT,
            "status": "scientific_rejection" if reasons else "complete_source_preflight",
            "rejection_reasons": reasons, "arm": ARM, "seed": 1,
            "source_proof": prepared["source_proof"],
            "implementation_fingerprint": prepared["implementation_fingerprint"],
            "selected_train_position": position, "selected_train_image_id": image_id,
            "source_state_sha256_before": source_before, "source_state_sha256_after": source_after,
            "fresh_initial_model_state_sha256": initial_state_sha,
            "source_gamma_sha256": source_gamma_hash, "source_trace_sha256": source_trace_hash,
            "source_A0_sha256": source_a_hash, "source_projection_sha256": source_z_hash,
            "source_Q_sha256": source_q_hash, "zero_head_trace_sha256": trace_hash,
            "zero_head_A0_sha256": train._tensor_sha(a),
            "zero_head_projection_sha256": train._tensor_sha(z),
            "zero_head_Q_sha256": train._tensor_sha(q), "forward_parity": parity,
            "source_support_sha256": hashlib.sha256(support.astype("<i4", copy=False).tobytes()).hexdigest(),
            "standalone_qcc_sha256": hashlib.sha256(source_qcc.astype("<i4", copy=False).tobytes()).hexdigest(),
            "source_conditional_labels_sha256": source_conditional_sha,
            "source_standalone_labels_sha256": source_standalone_sha,
            "head_initial_state_sha256": train._state_sha(torch.nn.Identity(), head),
            "head_seed": SEED, "head_mode": "signed", "critical_loss": float(loss.detach().cpu()),
            "gradient_evidence": gradient_info, "steps": STEPS, "settle": SETTLE,
            "live_tail_steps": TAIL, "optimizer_updates": 0,
            "validation_data_read": False, "ground_truth_scope": "TRAIN position 0 feasible target/source support",
            "elapsed_seconds": time.perf_counter() - start,
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(dev)),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
        train.preflight._write_report(output, report)
        return report
    finally:
        train.preflight._restore_rng(dev, original_rng)
        if found is not None and base_encoder is not None:
            found.encoder = base_encoder


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_preflight(device=args.device, output=args.output)
