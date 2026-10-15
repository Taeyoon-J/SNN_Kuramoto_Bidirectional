"""Source-bound, zero-update SW0172 dual-gamma admission."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = HERE / "results_archive" / "dual_gamma_admission_v4_20261015" / "preflight.json"
EXPERIMENT = "SW0172_graph_drive_separation_preflight"
POSITIONS = tuple(range(8))
TRACE_FIELDS = ("theta", "membrane", "spikes", "component_membrane",
                "component_spikes", "component_gates")
REGISTERED_LAMBDA = 0.35786938902827736
REQUIRED_STEPS = 1024
REQUIRED_SETTLE = 512
REQUIRED_LIVE_TAIL = 64


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


def implementation_fingerprint() -> dict:
    paths = (
        HERE / "DESIGN_KO.md", HERE / "dual_gamma_rollout.py", HERE / "preflight.py",
        ROOT / "collaborative_test/SW_0171_contextual_instance_graph/contextual_encoder.py",
        ROOT / "collaborative_test/SW_0171_contextual_instance_graph/instance_objective.py",
        ROOT / "collaborative_test/SW_0171_contextual_instance_graph/preflight.py",
        ROOT / "collaborative_test/SW_0171_contextual_instance_graph/train_prefix.py",
        ROOT / "collaborative_test/SW_0169_normalized_hard_reset/calibration_v2.py",
        ROOT / "collaborative_test/SW_0134_native_spike_binding/rollout.py",
        ROOT / "collaborative_test/SW_0162_connected_signed_partition/preflight.py",
        ROOT / "collaborative_test/SW_0136_native32_transfer/evaluate.py",
        ROOT / "collaborative_test/SW_0164_foreground_preserving_partition/foreground_signed.py",
        ROOT / "snn_kuramoto_bidirectional/gamma_initializer.py",
        ROOT / "snn_kuramoto_bidirectional/graph_generator.py",
        ROOT / "snn_kuramoto_bidirectional/spike_classifier.py",
    )
    files = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"SW0172 implementation dependency missing: {path}")
        files[path.relative_to(ROOT).as_posix()] = _sha_file(path)
    return {"files": files, "experiment": EXPERIMENT,
            "source_experiment": "SW0171_contextual_instance_graph",
            "positions": list(POSITIONS), "steps": 1024, "settle": 512,
            "live_tail": 64, "context_seed": 171001,
            "routing_arms": ["shared_gamma", "separated_gamma"],
            "lambda": REGISTERED_LAMBDA, "optimizer_updates": 0,
            "validation_data_read": False}


def fingerprint_sha256() -> str:
    raw = json.dumps(implementation_fingerprint(), sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return _sha_bytes(raw)


def prepare_inputs_cpu() -> dict:
    """Reuse the completed SW0171 source/input admission; select only its first B8."""
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0172 source preparation must run before CUDA initialization")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    trainer = _module("collaborative_test.SW_0171_contextual_instance_graph.train_prefix")
    admitted = trainer._load_admitted_inputs()
    if torch.cuda.is_initialized() or os.environ.get("CUDA_VISIBLE_DEVICES") != visible:
        raise RuntimeError("SW0171 source admission initialized CUDA or changed visibility")
    prepared = admitted["prepared"]
    if float(admitted["lambda"]) != REGISTERED_LAMBDA:
        raise ValueError("SW0172 must retain the completed SW0171 TRAIN-only lambda")
    arrays = prepared["arrays"]
    ids = list(map(int, admitted["proof"]["ordered_train_ids"]))
    if len(ids) != 512 or len(set(ids)) != 512:
        raise ValueError("SW0172 requires the admitted ordered SW0171 TRAIN512 source")
    rgb = np.ascontiguousarray(arrays["rgb"][list(POSITIONS)]).copy()
    support = np.ascontiguousarray(arrays["source_support"][list(POSITIONS)]).copy()
    if rgb.shape != (8, 128, 128, 3) or rgb.dtype != np.uint8:
        raise ValueError("admitted first-eight TRAIN RGB rows have invalid schema")
    if support.shape != (8, 32, 32) or support.dtype != np.int32:
        raise ValueError("admitted first-eight source-QCC rows have invalid schema")
    if np.any(support < 0):
        raise ValueError("source-QCC support labels must be nonnegative")
    result = {
        "rgb8": rgb, "source_support8": support,
        "selected_ids": [ids[index] for index in POSITIONS],
        "selected_positions": list(POSITIONS),
        "source_proof": admitted["proof"],
        "source_admission_report_sha256": admitted["admission_report_sha256"],
        "source_admission_queue_fingerprint_sha256": admitted["queue_fingerprint_sha256"],
        "source_admission_preflight_fingerprint": admitted["preflight_fingerprint"],
        "source_cache_manifest_sha256": prepared["source_proof"]["source_cache_manifest_sha256"],
        "context_cache_manifest_sha256": prepared["source_proof"]["context_cache_manifest_sha256"],
        "source_checkpoint_sha256": prepared["source_checkpoint_sha256"],
        "source_manifest_sha256": prepared["source_manifest_sha256"],
        "archived_reference_proof": prepared["archived_reference_proof"],
        "input_cache_manifest_sha256": prepared["cache_manifest_sha256"],
        "input_cache_arrays_sha256": prepared["arrays_sha256"],
        "source_gain": float(prepared["producer_bindings"]["source_gain"]),
        "lambda": float(admitted["lambda"]),
        "implementation_fingerprint": implementation_fingerprint(),
        "cuda_initialized": False, "validation_data_read": False,
        "optimizer_updates": 0,
    }
    result["rgb8_sha256"] = _sha_bytes(rgb.tobytes())
    result["source_support8_sha256"] = _sha_bytes(support.tobytes())
    return result


def _rng_state(device):
    state = {"cpu": torch.random.get_rng_state().clone()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device).clone()
    return state


def _restore_rng(device, state):
    torch.random.set_rng_state(state["cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def _source_run(found, gamma):
    from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
    from collaborative_test.SW_0162_connected_signed_partition.preflight import capture_source_coupling
    projection = []
    handle = found.wrapped.core.graph_generator.projection.register_forward_hook(
        lambda _module, _inputs, output: projection.append(output))
    try:
        trace, adjacency = capture_source_coupling(
            found.wrapped.core.kuramoto,
            lambda: late_rollout(found.wrapped, gamma, total_steps=1024,
                                 live_tail_steps=64))
    finally:
        handle.remove()
    if len(projection) != 1:
        raise ValueError("the native source graph projection was not called exactly once")
    _validate_trace_geometry(trace, gamma.shape[0], source=True)
    return {"trace": trace, "adjacency": adjacency,
            "projection_raw": projection[0], "z": F.normalize(projection[0], dim=-1)}


def _validate_trace_geometry(trace, batch, *, source=False):
    """Reject inherited short-core defaults before any settled-Q computation."""
    expected = set(TRACE_FIELDS)
    if not isinstance(trace, dict) or not expected.issubset(trace):
        raise ValueError("native rollout did not return all six registered trace fields")
    for key in TRACE_FIELDS:
        value = trace[key]
        if not torch.is_tensor(value):
            raise TypeError(f"native trace field {key} is not a tensor")
        if key == "theta":
            valid = (value.ndim == 4 and value.shape[0] == batch
                     and value.shape[1] == REQUIRED_STEPS
                     and value.shape[2] == 1024 and value.shape[3] == 4)
        elif key in ("component_membrane", "component_spikes", "component_gates"):
            valid = (value.ndim == 4 and value.shape[0] == batch
                     and value.shape[1] == 4 and value.shape[2] == 1024
                     and value.shape[-1] == REQUIRED_STEPS)
        else:
            valid = (value.ndim == 3 and value.shape[0] == batch
                     and value.shape[1] == 1024 and value.shape[-1] == REQUIRED_STEPS)
        if not valid:
            raise ValueError(f"native trace field {key} has wrong 1024-step geometry: {tuple(value.shape)}")
    if trace.get("live_tail_steps") != REQUIRED_LIVE_TAIL:
        raise ValueError("native rollout did not retain the registered 64-step live tail")
    if trace.get("prefix_steps") != REQUIRED_STEPS - REQUIRED_LIVE_TAIL:
        raise ValueError("native rollout did not detach at the registered 960-step boundary")


def _dual_run(wrapped, gamma_graph, gamma_drive, runner=None):
    """Call the rollout at the registered duration, never the core's 64-step default."""
    if runner is None:
        from collaborative_test.SW_0172_graph_drive_separation.dual_gamma_rollout import run_dual_gamma_rollout
        runner = run_dual_gamma_rollout
    result = runner(wrapped, gamma_graph, gamma_drive,
                    total_steps=REQUIRED_STEPS, live_tail_steps=REQUIRED_LIVE_TAIL)
    _validate_trace_geometry(result["trace"], gamma_graph.shape[0])
    return result


def _q(trace):
    from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity
    return spike_synchrony_affinity(trace["spikes"], trace["component_spikes"],
                                    settle=REQUIRED_SETTLE, affinity_mode="spike")


def _qcc(trace):
    from collaborative_test.SW_0136_native32_transfer.evaluate import _qcc_labels
    return _qcc_labels(trace)


def _readout_digest(q, adjacency, gain, labels):
    from collaborative_test.SW_0164_foreground_preserving_partition.foreground_signed import (
        foreground_induced_merge,
    )
    outputs = []
    for index in range(q.shape[0]):
        result = foreground_induced_merge(
            q[index].detach().cpu().numpy(), adjacency[index].detach().cpu().numpy(),
            gain, labels[index], grid_shape=(32, 32))
        energy = np.ascontiguousarray(result["pair_energy"])
        out_labels = np.ascontiguousarray(result["labels"])
        encoded_log = json.dumps(result["merge_log"], sort_keys=True,
                                 separators=(",", ":"), allow_nan=False).encode()
        outputs.append({"energy_sha256": _sha_bytes(energy.tobytes()),
                        "labels_sha256": _sha_bytes(out_labels.tobytes()),
                        "merge_log_sha256": _sha_bytes(encoded_log),
                        "merge_count": int(result["merge_count"])})
    return outputs


def _compare_trace(expected, actual):
    return {key: bool(torch.equal(expected["trace"][key], actual["trace"][key]))
            for key in TRACE_FIELDS}


def _cpu_trace_summary(trace):
    return {key: _sha_bytes(trace[key].detach().cpu().contiguous().numpy().tobytes())
            for key in TRACE_FIELDS}


def _source_identity(found, source_digest, checkpoint_sha256):
    base_hash = hashlib.sha256()
    for key, value in sorted(found.encoder.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        base_hash.update(key.encode("utf-8") + b"\0")
        base_hash.update(str(tensor.dtype).encode("ascii") + b"\0")
        base_hash.update(tensor.numpy().tobytes())
    return {"source_checkpoint_sha256": checkpoint_sha256,
            "source_cache_state_sha256": source_digest,
            "base_encoder_state_sha256": base_hash.hexdigest(),
            "graph_backend": found.wrapped.core.kuramoto_backend,
            "graph_operator": getattr(found.wrapped.core.graph_generator,
                                       "_sw0143_geodesic_arm", None),
            "feature_clip": float(found.feature_clip),
            "feature_mean_sha256": _sha_bytes(found.feature_mean.detach().cpu().contiguous().numpy().tobytes()),
            "feature_std_sha256": _sha_bytes(found.feature_std.detach().cpu().contiguous().numpy().tobytes())}


def run_preflight(*, device="cuda:0", prepared=None, output=OUTPUT) -> dict:
    """Verify fresh-source/shared/separated zero-residual parity and route credit."""
    started = time.perf_counter()
    prepared = prepare_inputs_cpu() if prepared is None else prepared
    fingerprint = implementation_fingerprint()
    if prepared.get("implementation_fingerprint") != fingerprint:
        raise ValueError("SW0172 CPU preparation fingerprint is stale")
    # Reuse the established atomic, create-only receipt writer. Resolve it
    # before CUDA initialization so the producer contract is explicit.
    report_module = _module("collaborative_test.SW_0171_contextual_instance_graph.preflight")
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0172 GPU preflight must own first CUDA initialization")
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0172 source parity admission requires CUDA")
    torch.cuda.set_device(dev)
    from collaborative_test.SW_0169_normalized_hard_reset import calibration_v2 as source_cal
    from collaborative_test.SW_0172_graph_drive_separation.dual_gamma_rollout import (
        encode_gamma_pair, run_dual_gamma_rollout,
    )
    from collaborative_test.SW_0171_contextual_instance_graph.contextual_encoder import (
        ContextualFeatureEncoder,
    )
    found = source_cal._load_source(dev)
    found.encoder.eval(); found.patcher.eval(); found.wrapped.eval()
    if found.wrapped.core.kuramoto_backend != "factorized":
        raise ValueError("SW0172 requires the registered factorized native core")
    if getattr(found.wrapped.core.graph_generator, "_sw0143_geodesic_arm", None) != "T":
        raise ValueError("SW0172 requires the frozen SW0143-T source graph")
    source_before = source_cal._source_digest(found)
    if source_before != prepared["source_proof"]["source_cache_state_sha256"]:
        raise ValueError("fresh seed-1 source differs from the admitted source cache lineage")
    base_encoder = found.encoder
    report = {
        "experiment": EXPERIMENT, "status": "running", "training_admission": False,
        "total_steps": REQUIRED_STEPS, "settle_steps": REQUIRED_SETTLE,
        "live_tail_steps": REQUIRED_LIVE_TAIL,
        "source_proof": prepared["source_proof"],
        "source_admission_report_sha256": prepared["source_admission_report_sha256"],
        "source_admission_queue_fingerprint_sha256": prepared["source_admission_queue_fingerprint_sha256"],
        "implementation_fingerprint": fingerprint,
        "lambda": REGISTERED_LAMBDA, "optimizer_updates": 0,
        "validation_data_read": False,
        "ground_truth_scope": "CPU reuses the verified TRAIN512 modal-mask cache; GPU admission reads only first-eight TRAIN source support, no VAL masks or counts",
        "selected_ids": prepared["selected_ids"],
        "selected_positions": prepared["selected_positions"],
        "rgb8_sha256": prepared["rgb8_sha256"],
        "source_support8_sha256": prepared["source_support8_sha256"],
        "initial_context_seed": 171001, "routing_arms": ["shared_gamma", "separated_gamma"],
        "rejection_reasons": [], "batches": [],
    }
    try:
        from collaborative_test.SW_0135_native32_spike_binding.foundation import encode_native32
        found.encoder = ContextualFeatureEncoder(base_encoder, seed=171001).to(dev)
        found.encoder.eval()
        context_state_initial_sha = source_cal._source_digest(found)
        for start in (0, 4):
            rgb = torch.from_numpy(np.ascontiguousarray(prepared["rgb8"][start:start + 4])).to(dev)
            support = prepared["source_support8"][start:start + 4]
            pair = encode_gamma_pair(found, rgb)
            original_gamma = encode_native32(base_encoder, found.patcher,
                found.feature_mean, found.feature_std, found.feature_clip, rgb)
            gamma_preprocess_exact = torch.equal(original_gamma, pair["gamma_base"])
            gamma_base = original_gamma.detach().requires_grad_(True)
            gamma_context = pair["gamma_context"]
            batch_row = {"start": start, "image_ids": prepared["selected_ids"][start:start + 4],
                         "gamma_base_context_exact": bool(torch.equal(gamma_base, gamma_context)),
                         "new_base_preprocessing_matches_native_exact": bool(gamma_preprocess_exact),
                         "trace_exact": {}, "q_exact": False, "actual_coupling_exact": False,
                         "z_exact": False, "source_qcc_exact": False,
                         "sw165_readout_exact": False}
            if not batch_row["gamma_base_context_exact"]:
                report["rejection_reasons"].append(f"zero_residual_gamma_mismatch_batch_{start}")
            if not gamma_preprocess_exact:
                report["rejection_reasons"].append(f"native_base_gamma_preprocessing_mismatch_batch_{start}")
            rng = _rng_state(dev)
            source = _source_run(found, original_gamma)
            q_source = _q(source["trace"])
            source_qcc = _qcc(source["trace"])
            if not np.array_equal(source_qcc, support):
                report["rejection_reasons"].append(f"source_qcc_cache_mismatch_batch_{start}")
            source_gain_actual = float(found.wrapped.core.graph_generator.log_coupling_gain.detach().exp().cpu())
            source_gain_bound = float(prepared["source_gain"])
            if not np.isclose(source_gain_actual, source_gain_bound, rtol=1e-7, atol=0.0):
                report["rejection_reasons"].append(f"source_gain_binding_mismatch_batch_{start}")
            source_gain = source_gain_bound
            source_readout = _readout_digest(q_source, source["adjacency"], source_gain, support)
            source_trace_summary = _cpu_trace_summary(source["trace"])
            source_a = source["adjacency"].detach().cpu().clone()
            source_q = q_source.detach().cpu().clone()
            source_z = source["z"].detach().cpu().clone()
            del source, q_source

            _restore_rng(dev, rng)
            shared = _dual_run(found.wrapped, gamma_context, gamma_context, run_dual_gamma_rollout)
            q_shared = _q(shared["trace"])
            # Compare against source tensors before releasing the source graph.
            # CPU copies above are exact; this avoids retaining multiple rollout graphs.
            shared_trace_sha = _cpu_trace_summary(shared["trace"])
            shared_trace_exact = shared_trace_sha == source_trace_summary
            shared_exact = (shared_trace_exact
                            and torch.equal(q_shared.detach().cpu(), source_q)
                            and torch.equal(shared["adjacency"].detach().cpu(), source_a)
                            and torch.equal(shared["z"].detach().cpu(), source_z))
            shared_summary = {"trace_sha256": shared_trace_sha,
                              "q_sha256": _sha_bytes(q_shared.detach().cpu().contiguous().numpy().tobytes()),
                              "actual_coupling_sha256": _sha_bytes(shared["adjacency"].detach().cpu().contiguous().numpy().tobytes()),
                              "z_sha256": _sha_bytes(shared["z"].detach().cpu().contiguous().numpy().tobytes()),
                              "exact_source_parity": bool(shared_exact)}
            shared_readout = _readout_digest(q_shared, shared["adjacency"], source_gain, support)
            shared_summary["sw165_readout_exact"] = shared_readout == source_readout
            del shared, q_shared
            core = found.wrapped.core
            core.last_component_out = None; core.last_component_spikes = None; core.last_gate_history = None

            _restore_rng(dev, rng)
            separated = _dual_run(found.wrapped, gamma_context, gamma_base, run_dual_gamma_rollout)
            q_separated = _q(separated["trace"])
            separated_trace_sha = _cpu_trace_summary(separated["trace"])
            separated_exact = (separated_trace_sha == source_trace_summary
                               and torch.equal(q_separated.detach().cpu(), source_q)
                               and torch.equal(separated["adjacency"].detach().cpu(), source_a)
                               and torch.equal(separated["z"].detach().cpu(), source_z))
            separated_readout = _readout_digest(q_separated, separated["adjacency"], source_gain, support)
            context_drive_grad = torch.autograd.grad(
                separated["drive"].sum(), gamma_context, retain_graph=True,
                allow_unused=True)[0]
            drive_base_grad = torch.autograd.grad(
                separated["drive"].sum(), gamma_base, retain_graph=True,
                allow_unused=True)[0]
            q_graph_grad = torch.autograd.grad(
                q_separated.sum(), gamma_context, retain_graph=True,
                allow_unused=True)[0]
            tail_spikes = separated["trace"]["component_spikes"][..., -64:]
            weights = torch.linspace(0.25, 1.25, tail_spikes.numel(),
                                     device=dev, dtype=tail_spikes.dtype).reshape_as(tail_spikes)
            graph_task_grad = torch.autograd.grad(
                (tail_spikes * weights).sum(), gamma_context,
                retain_graph=False, allow_unused=True)[0]
            graph_credit = (0.0 if graph_task_grad is None else
                            float(graph_task_grad.detach().double().norm().cpu()))
            q_graph_credit = (0.0 if q_graph_grad is None else
                              float(q_graph_grad.detach().double().norm().cpu()))
            drive_credit = (0.0 if drive_base_grad is None else
                            float(drive_base_grad.detach().double().norm().cpu()))
            candidate_drive_clean = context_drive_grad is None
            candidate_credit = (candidate_drive_clean and np.isfinite(drive_credit)
                                and drive_credit > 0 and np.isfinite(graph_credit)
                                and graph_credit > 0 and np.isfinite(q_graph_credit)
                                and q_graph_credit > 0)
            separated_summary = {
                "trace_sha256": separated_trace_sha,
                "q_sha256": _sha_bytes(q_separated.detach().cpu().contiguous().numpy().tobytes()),
                "actual_coupling_sha256": _sha_bytes(separated["adjacency"].detach().cpu().contiguous().numpy().tobytes()),
                "z_sha256": _sha_bytes(separated["z"].detach().cpu().contiguous().numpy().tobytes()),
                "exact_source_parity": bool(separated_exact),
                "sw165_readout_exact": separated_readout == source_readout,
                "context_to_direct_drive_gradient_absent": candidate_drive_clean,
                "base_gamma_to_direct_drive_gradient_norm": drive_credit,
                "context_gamma_to_actual_Q_gradient_norm": q_graph_credit,
                "context_gamma_to_actual_tail_spikes_gradient_norm": graph_credit,
                "routing_partial_credit": bool(candidate_credit),
            }
            batch_row.update({"trace_exact": {key: bool(separated_trace_sha[key] == source_trace_summary[key])
                                                for key in TRACE_FIELDS},
                              "q_exact": bool(torch.equal(q_separated.detach().cpu(), source_q)),
                              "actual_coupling_exact": bool(torch.equal(separated["adjacency"].detach().cpu(), source_a)),
                              "z_exact": bool(torch.equal(separated["z"].detach().cpu(), source_z)),
                              "source_qcc_exact": bool(np.array_equal(source_qcc, support)),
                              "sw165_readout_exact": bool(separated_readout == source_readout),
                              "shared_gamma": shared_summary,
                              "separated_gamma": separated_summary,
                              "source_trace_sha256": source_trace_summary,
                              "source_q_sha256": _sha_bytes(source_q.contiguous().numpy().tobytes()),
                              "source_actual_coupling_sha256": _sha_bytes(source_a.contiguous().numpy().tobytes()),
                              "source_z_sha256": _sha_bytes(source_z.contiguous().numpy().tobytes()),
                              "source_readout": source_readout,
                              "source_gain": source_gain})
            if not shared_exact or shared_readout != source_readout:
                report["rejection_reasons"].append(f"shared_gamma_source_parity_failed_batch_{start}")
            if not separated_exact or separated_readout != source_readout:
                report["rejection_reasons"].append(f"separated_gamma_source_parity_failed_batch_{start}")
            if not candidate_credit:
                report["rejection_reasons"].append(f"separated_gamma_routing_credit_failed_batch_{start}")
            report["batches"].append(batch_row)
            del separated, q_separated, original_gamma, gamma_base, gamma_context, pair
            core.last_component_out = None; core.last_component_spikes = None; core.last_gate_history = None
        found.encoder = base_encoder
        source_after = source_cal._source_digest(found)
        report["source_state_before_sha256"] = source_before
        report["source_state_after_sha256"] = source_after
        report["source_state_unchanged"] = source_before == source_after
        report["source_identity"] = _source_identity(
            found, source_before, prepared["source_proof"]["source_checkpoint_sha256"])
        report["context_wrapper_initial_state_sha256"] = context_state_initial_sha
        if source_before != source_after:
            report["rejection_reasons"].append("source_state_changed")
        report["status"] = "complete_admitted" if not report["rejection_reasons"] else "complete_rejected"
        report["training_admission"] = report["status"] == "complete_admitted"
        report["optimizer_updates"] = 0
        report["cuda_initialized"] = bool(torch.cuda.is_initialized())
        report["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated(dev))
        report["elapsed_seconds"] = time.perf_counter() - started
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        return report_module._write_report(target, report)
    finally:
        found.encoder = base_encoder


def validate_report(report: dict, *, source_proof: dict,
                    queue_fingerprint_sha256: str,
                    preflight_fingerprint: dict) -> str:
    """Validate the owner-published preflight envelope and its parity criteria."""
    if not isinstance(report, dict) or report.get("experiment") != EXPERIMENT:
        raise ValueError("SW0172 report experiment mismatch")
    if report.get("source_proof") != source_proof:
        raise ValueError("SW0172 report source proof mismatch")
    if report.get("queue_fingerprint_sha256") != queue_fingerprint_sha256:
        raise ValueError("SW0172 queue fingerprint mismatch")
    if report.get("implementation_fingerprint") != preflight_fingerprint:
        raise ValueError("SW0172 preflight fingerprint mismatch")
    if (report.get("optimizer_updates") != 0 or report.get("validation_data_read") is not False
            or report.get("total_steps") != REQUIRED_STEPS
            or report.get("settle_steps") != REQUIRED_SETTLE
            or report.get("live_tail_steps") != REQUIRED_LIVE_TAIL
            or report.get("lambda") != REGISTERED_LAMBDA
            or report.get("source_state_unchanged") is not True
            or report.get("cuda_initialized") is not True):
        raise ValueError("SW0172 admission report violates source/zero-update scope")
    if (report.get("selected_positions") != list(POSITIONS)
            or len(report.get("selected_ids", [])) != len(POSITIONS)
            or len(set(report.get("selected_ids", []))) != len(POSITIONS)):
        raise ValueError("SW0172 admission report does not bind exact eight TRAIN positions")
    rows = report.get("batches")
    if (not isinstance(rows, list) or len(rows) != 2
            or [row.get("start") for row in rows] != [0, 4]):
        raise ValueError("SW0172 admission report omitted the two original B4 rollouts")
    status, admitted = report.get("status"), report.get("training_admission")
    if status not in ("complete_admitted", "complete_rejected") or admitted is not (status == "complete_admitted"):
        raise ValueError("SW0172 report has inconsistent terminal admission status")
    for row in rows:
        row_checks = (row.get("gamma_base_context_exact"),
                      row.get("new_base_preprocessing_matches_native_exact"),
                      row.get("q_exact"), row.get("actual_coupling_exact"),
                      row.get("z_exact"), row.get("source_qcc_exact"),
                      row.get("sw165_readout_exact"))
        if (set(row.get("trace_exact", {})) != set(TRACE_FIELDS)
                or any(type(value) is not bool for value in row["trace_exact"].values())
                or any(type(value) is not bool for value in row_checks)):
            raise ValueError("SW0172 row parity evidence is incomplete or malformed")
        for arm in ("shared_gamma", "separated_gamma"):
            summary = row.get(arm)
            if (not isinstance(summary, dict)
                    or type(summary.get("exact_source_parity")) is not bool
                    or type(summary.get("sw165_readout_exact")) is not bool):
                raise ValueError(f"SW0172 {arm} parity evidence is malformed")
        candidate = row["separated_gamma"]
        if type(candidate.get("routing_partial_credit")) is not bool:
            raise ValueError("SW0172 routing credit evidence is malformed")
    if status == "complete_admitted":
        if (report.get("rejection_reasons")
                or not all(all(row["trace_exact"].values()) and all(
                    row.get(key) is True for key in (
                        "gamma_base_context_exact", "new_base_preprocessing_matches_native_exact",
                        "q_exact", "actual_coupling_exact", "z_exact", "source_qcc_exact",
                        "sw165_readout_exact")) for row in rows)
                or not all(row[arm]["exact_source_parity"] is True
                           and row[arm]["sw165_readout_exact"] is True
                           for row in rows for arm in ("shared_gamma", "separated_gamma"))
                or not all(row["separated_gamma"]["routing_partial_credit"] is True for row in rows)):
            raise ValueError("SW0172 admitted report does not satisfy source parity/credit criteria")
        return status
    if status == "complete_rejected" and report.get("rejection_reasons"):
        return status
    raise ValueError("SW0172 rejected report lacks rejection evidence")
