"""Eight-image SW0169 source parity and two-update admission preflight."""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from collaborative_test.SW_0169_normalized_hard_reset import calibration_v2 as calibration, hard_reset

ARMS = ("history_live", "history_detached")
STEPS, SETTLE, TAIL = 1024, 512, 64
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def _import_keep_cuda(name):
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    module = importlib.import_module(name)
    if inherited is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = inherited
    return module


def _sha_tensor(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def implementation_fingerprint():
    return {**calibration.implementation_fingerprint(),
            "collaborative_test/SW_0169_normalized_hard_reset/preflight_v3.py":
                hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _modal32_grid(flat_labels):
    """Reshape the real SW0155 flattened modal32 output without relabeling."""
    labels = np.asarray(flat_labels)
    if labels.ndim != 2 or labels.shape[1] != 32 * 32:
        raise ValueError("SW0155 modal_train_targets32 must return [B,1024]")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("SW0155 modal32 labels must have integer dtype")
    return labels.reshape(labels.shape[0], 32, 32)


def _source_identity(prepared):
    return {
        "seed": 1, "selected_ids": list(map(int, prepared["selected_ids"])),
        "positions": list(map(int, prepared["positions"])),
        "source_checkpoint_sha256": prepared["source_checkpoint_sha256"],
        "source_manifest_sha256": prepared["source_manifest_sha256"],
        "source_cache_manifest_sha256": prepared["source_cache_manifest_sha256"],
        "source_cache_bindings_sha256": prepared["source_cache_bindings_sha256"],
        "source_cache_state_sha256": prepared["source_cache_state_sha256"],
        "source_cache_trace_sha256": list(prepared["source_cache_trace_sha256"]),
        "context_cache_manifest_sha256": prepared["context_cache_manifest_sha256"],
        "context_cache_bindings_sha256": prepared["context_cache_bindings_sha256"],
        "context_cache_state_sha256": prepared["context_cache_state_sha256"],
        "context_rows": prepared["context_rows"],
        "rgb_sha256": list(prepared["rgb_sha256"]),
        "rgb_manifest_sha256": prepared["train_rgb_manifest_sha256"],
        "hdf5_path": prepared["train_hdf5_path"],
        "hdf5_size_bytes": int(prepared["train_hdf5_size_bytes"]),
        "hdf5_mtime_ns": int(prepared["train_hdf5_mtime_ns"]),
        "implementation_fingerprint": prepared["implementation_fingerprint"],
    }


def validate_report(report, *, expected_source, expected_fingerprint):
    if report.get("experiment") != "SW0169_normalized_hard_reset":
        raise ValueError("SW0169 report experiment mismatch")
    if report.get("source") != expected_source:
        raise ValueError("SW0169 report source/cache identity mismatch")
    current = implementation_fingerprint()
    if report.get("implementation_fingerprint") != current:
        raise ValueError("SW0169 report implementation fingerprint mismatch")
    if report.get("queue_fingerprint_sha256") != expected_fingerprint:
        raise ValueError("SW0169 queue fingerprint mismatch")
    calibration_receipt = report.get("calibration", {})
    if (calibration_receipt.get("source_six_trace_parity") is not True
            or calibration_receipt.get("source_q_parity") is not True
            or calibration_receipt.get("source_a_parity") is not True
            or calibration_receipt.get("source_cache_replay_exact") is not True
            or calibration_receipt.get("source_model_state_sha256_before")
               != expected_source["source_cache_state_sha256"]
            or calibration_receipt.get("source_model_state_sha256_after")
               != expected_source["source_cache_state_sha256"]):
        raise ValueError("SW0169 source parity/calibration receipt is incomplete")
    source_rows = calibration_receipt.get("source_rows")
    if not isinstance(source_rows, list) or len(source_rows) != 8:
        raise ValueError("SW0169 calibration must report all eight cache-Q replay rows")
    for index, row in enumerate(source_rows):
        expected_mode = "live_trace_strided_b4" if index < 4 else "saved_contiguous_b4"
        if (row.get("index") != index
                or row.get("image_id") != expected_source["selected_ids"][index]
                or row.get("cache_q_producer_mode") != expected_mode
                or row.get("cache_context_verified") is not True):
            raise ValueError("SW0169 historical cache-Q producer mode/order mismatch")
        for key in ("q_sha256", "a_sha256", "cache_q_sha256", "live_q_sha256",
                    "cache_replay_q_sha256"):
            value = row.get(key)
            if (not isinstance(value, str) or len(value) != 64
                    or any(ch not in "0123456789abcdef" for ch in value)):
                raise ValueError(f"SW0169 cache-Q replay row has invalid {key}")
        trace_hashes = row.get("trace_sha256")
        if not isinstance(trace_hashes, dict) or set(trace_hashes) != set(calibration.TRACE_FIELDS):
            raise ValueError("SW0169 calibration row is missing native trace hashes")
        if any(not isinstance(value, str) or len(value) != 64
               or any(ch not in "0123456789abcdef" for ch in value)
               for value in trace_hashes.values()):
            raise ValueError("SW0169 calibration row contains an invalid native trace hash")
        if (not math.isfinite(float(row.get("live_vs_cache_replay_max_abs_diff", float("nan"))))
                or row["live_vs_cache_replay_max_abs_diff"] < 0
                or isinstance(row.get("live_vs_cache_replay_threshold_0_5_edge_differences"), bool)
                or not isinstance(row.get("live_vs_cache_replay_threshold_0_5_edge_differences"), int)
                or row["live_vs_cache_replay_threshold_0_5_edge_differences"] < 0):
            raise ValueError("SW0169 live/cache Q diagnostic is invalid")
        for key in ("live_component_spike_strides", "cache_q_input_strides"):
            strides = row.get(key)
            if (not isinstance(strides, list) or len(strides) != 3
                    or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0
                           for item in strides)):
                raise ValueError(f"SW0169 cache-Q replay row has invalid {key}")
        if expected_mode == "saved_contiguous_b4" and row["cache_q_input_strides"][-1] != 1:
            raise ValueError("SW0169 saved-row cache-Q input is not contiguous")
    admitted = report.get("training_admission")
    if admitted not in (True, False):
        raise ValueError("SW0169 admission result must be boolean")
    if report.get("training_ids") != expected_source["selected_ids"]:
        raise ValueError("SW0169 report TRAIN IDs differ from admitted source prefix")
    if report.get("validation_data_read") is not False or report.get(
            "ground_truth_scope") != "TRAIN8 only":
        raise ValueError("SW0169 admission exceeded registered data scope")
    if admitted:
        if set(report.get("arms", {})) != set(ARMS):
            raise ValueError("SW0169 admitted receipt must contain both history arms")
        for arm in ARMS:
            rows = report["arms"][arm].get("updates", [])
            logical = [row for row in rows if row.get("batch_start") == 4]
            if (len(rows) != 4 or len(logical) != 2
                    or [row.get("update") for row in logical] != [1, 2]
                    or any(row.get("critical_credit") is not True for row in logical)
                    or any(not all(math.isfinite(float(row[key])) for key in
                                   ("mean_loss_micro", "dL_dQ_norm", "dL_dS_norm"))
                           for row in rows)):
                raise ValueError(f"SW0169 {arm} did not complete both disposable updates")
    return report


def _atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def prepare_train_targets(calibration_input, calibration_report, masks32):
    """Bind modal32 TRAIN targets to completed source parity/calibration first."""
    _require_calibration_receipt(calibration_input, calibration_report)
    masks = np.asarray(masks32)
    if masks.shape != (8, 32, 32) or not np.issubdtype(masks.dtype, np.integer):
        raise ValueError("SW0169 admission requires eight modal32 TRAIN masks")
    if np.any(masks < 0):
        raise ValueError("TRAIN target labels must be nonnegative")

    qcc = _import_keep_cuda("collaborative_test.SW_0167_structured_relation_energy.qcc_readout")
    structured = _import_keep_cuda(
        "collaborative_test.SW_0167_structured_relation_energy.structured_energy")
    audit = _import_keep_cuda("collaborative_test.SW_0167_structured_relation_energy.audit_train_cache")
    source_q = audit._recompute_cpu_q(
        np.stack(calibration_input["cache_component_spikes"][:4]))
    source_q_second = audit._recompute_cpu_q(
        np.stack(calibration_input["cache_component_spikes"][4:8]))
    source_q = np.concatenate((source_q, source_q_second), axis=0)
    qcc_labels = qcc.qcc_labels_from_affinity(source_q)
    qcc_labels = np.asarray(qcc_labels, dtype=np.int32)
    if qcc_labels.shape != (8, 32, 32):
        raise ValueError("production CPU-B4 QCC did not return eight native32 supports")
    supports, targets, receipts = [], [], []
    for index, target in enumerate(masks):
        labels = qcc_labels[index]
        feasible = structured.feasible_target(labels, target)
        feasible = np.asarray(feasible, dtype=np.int32)
        if labels.shape != (32, 32) or feasible.shape != (32, 32):
            raise ValueError("SW0169 source support/feasible target is not native32")
        if np.any(feasible[labels == 0] != 0):
            raise ValueError("feasible target labels escaped the fixed source support")
        supports.append(labels)
        targets.append(feasible)
        receipts.append({"index": index,
                         "image_id": int(calibration_input["selected_ids"][index]),
                         "source_f_sha256": hashlib.sha256(labels.astype("<i4").tobytes()).hexdigest(),
                         "target_sha256": hashlib.sha256(feasible.astype("<i4").tobytes()).hexdigest()})
    return {"supports": supports, "targets": targets, "target_receipts": receipts,
            "ground_truth_scope": "eight modal32 TRAIN masks only"}


def _require_calibration_receipt(calibration_input, calibration_report):
    if (calibration_report.get("source_six_trace_parity") is not True
            or calibration_report.get("source_q_parity") is not True
            or calibration_report.get("source_a_parity") is not True
            or calibration_report.get("source_cache_replay_exact") is not True):
        raise ValueError("TRAIN masks cannot be opened before exact source parity/calibration")
    if calibration_report.get("implementation_fingerprint") != calibration_input.get(
            "implementation_fingerprint"):
        raise ValueError("calibration report belongs to a different source implementation")
    if calibration_report.get("source_model_state_sha256_after") != calibration_input.get(
            "source_cache_state_sha256"):
        raise ValueError("source model changed during current calibration")


def prepare_train_inputs_cpu(calibration_report, calibration_input=None):
    """Revalidate source/cache bindings, then read only the eight TRAIN masks."""
    # This function refuses to open masks until the GPU source-parity stage has
    # completed and produced its immutable six-trace/Q/A receipt.
    if calibration_input is not None:
        raise ValueError("TRAIN targets must be prepared in a fresh CPU process")
    source = calibration.prepare_calibration_cpu()
    _require_calibration_receipt(source, calibration_report)
    sw155 = _import_keep_cuda("collaborative_test.SW_0155_supervised_spike_relations.train")
    objective = _import_keep_cuda("collaborative_test.SW_0155_supervised_spike_relations.objective")
    hdf_stat = os.stat(sw155.HDF5_PATH)
    aligned, masks = sw155.load_aligned_train_rows(
        sw155.HDF5_PATH, source["selected_ids"], source["rgb8"], 0, 8)
    if aligned.dtype != torch.uint8 or not np.array_equal(aligned.cpu().numpy(), source["rgb8"]):
        raise ValueError("canonical HDF5 TRAIN RGB differs from registered cache rows")
    modal = _modal32_grid(objective.modal_train_targets32(masks).cpu().numpy())
    if (int(os.stat(sw155.HDF5_PATH).st_size) != hdf_stat.st_size
            or int(os.stat(sw155.HDF5_PATH).st_mtime_ns) != hdf_stat.st_mtime_ns):
        raise ValueError("canonical TRAIN HDF5 changed while aligned targets were read")
    target_data = prepare_train_targets(source, calibration_report, modal)
    return {**source, **target_data,
            "hdf5_path": os.path.realpath(sw155.HDF5_PATH),
            "hdf5_size_bytes": hdf_stat.st_size, "hdf5_mtime_ns": hdf_stat.st_mtime_ns,
            "ground_truth_read": True, "validation_data_read": False}


def _cpu_target_worker(report_path, output_path):
    """Fresh-interpreter target preparation entrypoint; never initializes CUDA."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    training = prepare_train_inputs_cpu(report)
    np.savez_compressed(output_path, rgb8=training["rgb8"],
                        supports=np.stack(training["supports"]).astype(np.int32),
                        targets=np.stack(training["targets"]).astype(np.int32))
    metadata = {key: training[key] for key in
                ("selected_ids", "target_receipts", "hdf5_path", "hdf5_size_bytes",
                 "hdf5_mtime_ns", "ground_truth_read", "validation_data_read",
                 "ground_truth_scope", "source_cache_manifest_sha256",
                 "context_cache_manifest_sha256", "source_cache_state_sha256",
                 "context_cache_state_sha256", "implementation_fingerprint")}
    Path(str(output_path) + ".json").write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")), encoding="utf-8")


def _prepare_targets_fresh_cpu(calibration_report):
    """Open TRAIN masks after source parity, in a separate CUDA-hidden process."""
    temp_root = ROOT / "collaborative_testing" / "temporary"
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sw169_targets_", dir=temp_root) as temp:
        temp = Path(temp)
        report_path, arrays_path = temp / "calibration.json", temp / "targets.npz"
        report_path.write_text(json.dumps(calibration_report, sort_keys=True), encoding="utf-8")
        code = (
            "import os,sys,json,numpy,torch; "
            "os.environ['CUDA_VISIBLE_DEVICES']=''; "
            "from collaborative_test.SW_0169_normalized_hard_reset.preflight_v3 import _cpu_target_worker; "
            "_cpu_target_worker(sys.argv[1],sys.argv[2]); "
            "print(json.dumps({'cuda_initialized':torch.cuda.is_initialized(),'ok':True}))"
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        proc = subprocess.run([sys.executable, "-c", code, str(report_path), str(arrays_path)],
                              cwd=str(ROOT), env=env, text=True, capture_output=True,
                              timeout=600, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"fresh CPU TRAIN target preparation failed: {proc.stderr[-4000:]}")
        status = json.loads(proc.stdout.strip().splitlines()[-1])
        if status != {"cuda_initialized": False, "ok": True}:
            raise RuntimeError("fresh CPU target worker initialized CUDA or failed receipt")
        metadata = json.loads(Path(str(arrays_path) + ".json").read_text(encoding="utf-8"))
        with np.load(arrays_path, allow_pickle=False) as arrays:
            result = {**metadata, "rgb8": arrays["rgb8"].copy(),
                      "supports": [row.copy() for row in arrays["supports"]],
                      "targets": [row.copy() for row in arrays["targets"]]}
        expected_ids = [row["image_id"] for row in calibration_report["source_rows"]]
        if result["selected_ids"] != expected_ids:
            raise ValueError("fresh CPU target preparation changed ordered source IDs")
        if result["source_cache_state_sha256"] != calibration_report[
                "source_model_state_sha256_after"]:
            raise ValueError("fresh CPU target preparation changed source model state")
        for key in ("source_cache_manifest_sha256", "context_cache_manifest_sha256",
                    "source_cache_state_sha256", "context_cache_state_sha256",
                    "implementation_fingerprint"):
            if key in calibration_report and result.get(key) != calibration_report[key]:
                raise ValueError(f"fresh CPU target preparation changed source binding: {key}")
        return result


def _event_diagnostics(trace):
    raw = trace["raw_events"].detach()[..., SETTLE:] > 0
    gate = trace["component_gates"].detach()[..., SETTLE:] > 0
    emitted = trace["component_spikes"].detach()[..., SETTLE:]
    if not torch.equal(emitted, trace["raw_events"].detach()[..., SETTLE:] *
                       trace["component_gates"].detach()[..., SETTLE:]):
        raise ValueError("actual emitted component spike is not raw event times gate")
    rows = []
    for image in range(raw.shape[0]):
        for component in range(raw.shape[1]):
            active = gate[image, component]
            count = int(active.sum().cpu())
            if count == 0:
                occupancy = None
                transitions = 0
                eligible_nodes = 0
                flipping_nodes = 0
            else:
                eligible = raw[image, component][active]
                occupancy = float(eligible.float().mean().cpu())
                adjacent = active[..., 1:] & active[..., :-1]
                flips = (raw[image, component, ..., 1:] !=
                         raw[image, component, ..., :-1]) & adjacent
                eligible_nodes_mask = adjacent.any(dim=-1)
                flipping_nodes_mask = flips.any(dim=-1)
                eligible_nodes = int(eligible_nodes_mask.sum().cpu())
                flipping_nodes = int((flipping_nodes_mask & eligible_nodes_mask).sum().cpu())
                transitions = int(flips.sum().cpu())
            total_nodes = int(active.shape[0])
            flip_fraction = flipping_nodes / total_nodes if total_nodes else None
            rows.append({"image_index": image, "component": component,
                         "active_samples": count, "raw_event_on_fraction_when_active": occupancy,
                         "total_node_units": total_nodes,
                         "eligible_node_units": eligible_nodes,
                         "eligible_node_fraction": eligible_nodes / total_nodes if total_nodes else None,
                         "node_units_with_active_event_flip": flipping_nodes,
                         "active_event_flip_node_fraction": flip_fraction,
                         "active_to_active_event_transitions": transitions,
                         "event_history_hold_samples": int((~active).sum().cpu()),
                         "emitted_spike_nonzero_fraction_all_settled": float(
                             (emitted[image, component] != 0).float().mean().cpu())})
    return rows


def _event_gate_failures(rows):
    failures = []
    for row in rows:
        occupancy = row["raw_event_on_fraction_when_active"]
        if (row["active_samples"] == 0 or occupancy is None
                or not 0.0 < occupancy < 1.0
                or row.get("eligible_node_units", 0) == 0):
            failures.append({"image_index": row["image_index"],
                             "component": row["component"],
                             "reason": "event_mix_or_eligible_sample_gate"})
    for component in sorted({int(row["component"]) for row in rows}):
        selected = [row for row in rows if int(row["component"]) == component]
        denominator = sum(int(row.get("total_node_units", 0)) for row in selected)
        flipped = sum(int(row.get("node_units_with_active_event_flip", 0)) for row in selected)
        fraction = flipped / denominator if denominator else 0.0
        if fraction < 0.10:
            failures.append({"component": component,
                             "reason": "component_image_node_flip_fraction_below_10pct",
                             "node_flip_fraction": fraction,
                             "node_units_with_flip": flipped,
                             "image_node_units": denominator})
    return failures


def _parameter_families(found):
    wrapped = found.wrapped
    graph = wrapped.core.graph_generator
    if getattr(graph, "_sw0143_geodesic_arm", None) != "T":
        raise ValueError("SW0169 requires the registered SW0143-T graph operator")
    if wrapped.core.kuramoto_backend != "factorized":
        raise ValueError("SW0169 requires the registered factorized Kuramoto backend")
    if getattr(graph, "geodesic_contrast", None) is not None:
        graph.geodesic_contrast.requires_grad_(False)
    families = {
        "encoder": list(found.encoder.parameters()),
        "graph": [p for name, p in graph.named_parameters() if name != "geodesic_contrast"],
        "native_core": [p for name, p in wrapped.core.named_parameters()
                        if not name.startswith("graph_generator.")],
        "integration": [wrapped.a_d, wrapped.a_m, wrapped.b],
    }
    params = [p for values in families.values() for p in values]
    if not params or len({id(p) for p in params}) != len(params):
        raise ValueError("SW0169 trainable parameter set is empty or duplicated")
    for p in params:
        p.requires_grad_(True)
    return families


def _norm(tensors):
    grads = [x for x in tensors if x is not None]
    if not grads:
        return 0.0
    if any(not torch.isfinite(g).all() for g in grads):
        raise FloatingPointError("nonfinite SW0169 gradient")
    return math.sqrt(sum(float(g.detach().double().square().sum().cpu()) for g in grads))


def _family_gradients(families):
    return {name: _norm([p.grad for p in params]) for name, params in families.items()}


def _named_core_gradients(found):
    groups = {"kuramoto": [], "dendric_layer": [], "membrane_layer": []}
    for name, parameter in found.wrapped.core.named_parameters():
        for family in groups:
            if family in name:
                groups[family].append(parameter.grad)
                break
    return {name: _norm(values) for name, values in groups.items()}


def _named_core_deltas(found, snapshot):
    groups = {"kuramoto": [], "dendric_layer": [], "membrane_layer": []}
    old = {name: value for name, value in snapshot}
    for name, parameter in found.wrapped.core.named_parameters():
        for family in groups:
            if family in name:
                if name in old:
                    groups[family].append(parameter.detach() - old[name].to(parameter.device))
                break
    return {name: _norm(values) for name, values in groups.items()}


def _named_core_snapshot(found):
    return [(name, parameter.detach().cpu().clone())
            for name, parameter in found.wrapped.core.named_parameters()]


def _parameter_delta(before, families):
    return {name: math.sqrt(sum(float((old.detach().cpu().double() - p.detach().cpu().double()).square().sum())
                                for old, p in zip(before[name], params)))
            for name, params in families.items()}


def _snapshot(families):
    return {name: [p.detach().clone() for p in params] for name, params in families.items()}


def _objective(arm, q, target, support):
    objective = _import_keep_cuda("collaborative_test.SW_0168_end_to_end_connectivity.connectivity_objective")
    foreground = np.asarray(support) > 0
    if arm in ARMS:
        return objective.constrained_maximin_loss(q, target, foreground)
    raise ValueError("unknown SW0169 membrane-history arm")


def _new_source(device, r):
    found = calibration._load_source(device)
    found.encoder.eval(); found.patcher.eval(); found.wrapped.eval()
    source_digest = calibration._source_digest(found)
    if getattr(found.wrapped.core.graph_generator, "_sw0143_geodesic_arm", None) != "T":
        raise ValueError("loaded graph is not SW0143 T")
    initialization = hard_reset.initialize_slow_membrane_(found.wrapped)
    return found, source_digest, initialization


def _rollout_capture_final_carries(found, gamma, r, *, detach_history):
    """Observe actual V/E carry operands at the last live timestep, then restore."""
    original = hard_reset.normalized_membrane_step
    observed = {}
    calls = 0

    def capture_step(**kwargs):
        nonlocal calls
        calls += 1
        if calls == STEPS:
            for name in ("previous_membrane", "previous_event"):
                value = kwargs[name]
                if value.requires_grad:
                    value.retain_grad()
                observed[name] = value
        result = original(**kwargs)
        if calls == STEPS:
            if result.emitted_spike.requires_grad:
                result.emitted_spike.retain_grad()
            observed["step_kwargs"] = kwargs
            observed["actual_emitted"] = result.emitted_spike
        return result

    hard_reset.normalized_membrane_step = capture_step
    try:
        trace = hard_reset.rollout(found.wrapped, gamma, r, total_steps=STEPS,
                                   live_tail_steps=TAIL, detach_history=detach_history)
    finally:
        hard_reset.normalized_membrane_step = original
    if calls != STEPS or set(observed) != {
            "previous_membrane", "previous_event", "step_kwargs", "actual_emitted"}:
        raise RuntimeError("did not observe the actual final V/E carry operands")
    return trace, observed


def _direct_carry_partial_gradient(observed, *, detach_history):
    """Isolate V/E carry partials using the actual step's upstream spike gradient."""
    if observed["actual_emitted"].grad is None:
        raise ValueError("actual final emitted spike has no objective gradient")
    kwargs = {}
    for key, value in observed["step_kwargs"].items():
        kwargs[key] = value.detach() if torch.is_tensor(value) else value
    previous_v = observed["previous_membrane"].detach().clone().requires_grad_(True)
    previous_e = observed["previous_event"].detach().clone().requires_grad_(True)
    kwargs.update(previous_membrane=previous_v, previous_event=previous_e,
                  detach_history=detach_history, validate_inputs=False)
    output = hard_reset.normalized_membrane_step(**kwargs).emitted_spike
    actual = observed["actual_emitted"].detach()
    if not torch.equal(output.detach(), actual):
        raise ValueError("isolated direct-carry replay changed the actual final-step output")
    scalar = (output * observed["actual_emitted"].grad.detach()).sum()
    if scalar.requires_grad:
        grad_v, grad_e = torch.autograd.grad(scalar, (previous_v, previous_e),
                                            allow_unused=True)
    else:
        grad_v = grad_e = None
    return {"previous_membrane": _norm([grad_v]) if grad_v is not None else 0.0,
            "previous_event": _norm([grad_e]) if grad_e is not None else 0.0,
            "scope": "local final-step partial; current/gate/threshold/upstream fixed"}


def run_two_update_admission(prepared, *, device="cuda:0", calibration_output=None):
    """Run exact source calibration and two disposable B4 updates per arm."""
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0169 admission requires an exclusive CUDA device")
    torch.cuda.set_device(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    calibration_receipt = calibration.run_source_calibration_gpu(prepared, device=str(dev))
    handoff_receipt = {**calibration_receipt,
                       "source_cache_manifest_sha256": prepared["source_cache_manifest_sha256"],
                       "context_cache_manifest_sha256": prepared["context_cache_manifest_sha256"],
                       "source_cache_state_sha256": prepared["source_cache_state_sha256"],
                       "context_cache_state_sha256": prepared["context_cache_state_sha256"]}
    if calibration_output is not None:
        # Save successful GPU source evidence before the fresh CPU TRAIN-target
        # handoff, which is an independent failure point.
        _atomic_json(calibration_output, handoff_receipt)
    training = _prepare_targets_fresh_cpu(handoff_receipt)
    r = torch.tensor(calibration_receipt["r"], dtype=torch.float32, device=dev)
    if r.shape != (4,) or not torch.isfinite(r).all() or torch.any(r <= 0):
        raise ValueError("calibration receipt lacks the four fixed positive RMS values")
    objective = _import_keep_cuda("collaborative_test.SW_0168_end_to_end_connectivity.connectivity_objective")
    q_function = _import_keep_cuda("snn_kuramoto_bidirectional.spike_classifier").spike_synchrony_affinity
    reports, initial_hashes = {}, None
    for arm in ARMS:
        torch.cuda.reset_peak_memory_stats(dev)
        found, source_digest, init_report = _new_source(dev, r)
        families = _parameter_families(found)
        source_sha_initial = calibration._source_digest(found)
        if source_sha_initial == prepared["source_cache_state_sha256"]:
            raise ValueError("slow tau/b initialization was not applied")
        # The source model is loaded afresh per arm, then identically initialized.
        initial_rows, initial_ref = [], []
        for start in (0, 4):
            rgb = torch.from_numpy(np.ascontiguousarray(training["rgb8"][start:start + 4])).to(dev)
            gamma = found.encode(rgb)
            trace = hard_reset.rollout(found.wrapped, gamma, r, total_steps=STEPS,
                                       live_tail_steps=TAIL,
                                       detach_history=(arm == "history_detached"))
            q = q_function(trace["spikes"], trace["component_spikes"],
                           settle=SETTLE, affinity_mode="spike")
            if not torch.isfinite(q).all():
                raise FloatingPointError("modified SW0169 initial Q is nonfinite")
            row_stats = _event_diagnostics(trace)
            for row in row_stats:
                row["image_index"] += start
                row["image_id"] = int(training["selected_ids"][row["image_index"]])
            initial_rows.extend(row_stats)
            for j in range(4):
                h = {key: _sha_tensor(trace[key][j]) for key in
                     ("theta", "membrane", "spikes", "component_membrane",
                      "component_spikes", "component_gates", "raw_events", "event_history")}
                initial_ref.append({"index": start + j, "trace_sha256": h,
                                    "q_sha256": _sha_tensor(q[j])})
            del rgb, gamma, trace, q
        if initial_hashes is None:
            initial_hashes = initial_ref
        elif initial_ref != initial_hashes:
            raise ValueError("history arms do not have identical initial forward traces/Q")
        event_failures = _event_gate_failures(initial_rows)
        if event_failures:
            # Persist a complete scientific rejection receipt rather than losing
            # the observed source-bound diagnostics in an exception path.
            arm_reports = {arm: {"status": "complete_rejected_event_gate",
                                 "source_state_sha256_initial": source_digest,
                                 "initialization": init_report,
                                 "initial_trace_rows": initial_ref,
                                 "initial_event_diagnostics": initial_rows,
                                 "event_gate_passed": False},
                           ARMS[1]: {"status": "not_run_event_gate_rejected"}}
            return {"experiment": "SW0169_normalized_hard_reset",
                    "arms": arm_reports, "calibration": calibration_receipt,
                    "training_ids": training["selected_ids"],
                    "target_receipts": training["target_receipts"],
                    "event_gate_failures": event_failures,
                    "validation_data_read": False, "ground_truth_scope": "TRAIN8 only",
                    "implementation_fingerprint": implementation_fingerprint(),
                    "training_admission": False}

        groups = [dict(params=params, lr=3e-5, name=name)
                  for name, params in families.items() if name != "integration"]
        groups.append(dict(params=families["integration"], lr=1e-3, name="integration"))
        optimizer = torch.optim.Adam(groups)
        update_rows = []
        start_time = time.perf_counter()
        for update in (1, 2):
            optimizer.zero_grad(set_to_none=True)
            for start in (0, 4):
                rgb = torch.from_numpy(np.ascontiguousarray(training["rgb8"][start:start + 4])).to(dev)
                gamma = found.encode(rgb)
                carries = None
                if update == 1 and start == 0:
                    trace, carries = _rollout_capture_final_carries(
                        found, gamma, r, detach_history=(arm == "history_detached"))
                else:
                    trace = hard_reset.rollout(found.wrapped, gamma, r, total_steps=STEPS,
                                               live_tail_steps=TAIL,
                                               detach_history=(arm == "history_detached"))
                q = q_function(trace["spikes"], trace["component_spikes"],
                               settle=SETTLE, affinity_mode="spike")
                q.retain_grad(); trace["component_spikes"].retain_grad()
                losses = [_objective(arm, q[i], training["targets"][start + i],
                                     training["supports"][start + i]) for i in range(4)]
                loss = torch.stack(losses).sum() / 8.0
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite disposable SW0169 loss")
                loss.backward()
                qgrad = _norm([q.grad])
                spikegrad = _norm([trace["component_spikes"].grad])
                carry_gradients = None
                if carries is not None:
                    carry_gradients = _direct_carry_partial_gradient(
                        carries, detach_history=(arm == "history_detached"))
                update_rows.append({"update": update, "batch_start": start,
                                    "mean_loss_micro": float(torch.stack(losses).mean().detach().cpu()),
                                    "dL_dQ_norm": qgrad, "dL_dS_norm": spikegrad,
                                    "direct_previous_carry_gradient_norms": carry_gradients})
                del rgb, gamma, trace, q, losses, loss
            grads = _family_gradients(families)
            named_core_grads = _named_core_gradients(found)
            critical = (grads["encoder"] > 0 and grads["graph"] > 0
                        and grads["native_core"] > 0
                        and _norm([found.wrapped.core.membrane_layer.tau_m.grad]) > 0
                        and _norm([found.wrapped.b.grad]) > 0)
            if not critical:
                raise ValueError("SW0169 source/graph/native-membrane credit gate failed")
            before = _snapshot(families)
            named_before = _named_core_snapshot(found)
            tau_before = found.wrapped.core.membrane_layer.tau_m.detach().clone()
            b_before = found.wrapped.b.detach().clone()
            torch.nn.utils.clip_grad_norm_([p for ps in families.values() for p in ps], 1.0)
            optimizer.step()
            found.wrapped.project_integrations_()
            deltas = _parameter_delta(before, families)
            named_deltas = _named_core_deltas(found, named_before)
            if any(not math.isfinite(v) for v in deltas.values()):
                raise FloatingPointError("nonfinite SW0169 update delta")
            tau_delta = _norm([found.wrapped.core.membrane_layer.tau_m.detach() - tau_before])
            b_delta = _norm([found.wrapped.b.detach() - b_before])
            if tau_delta <= 0 or b_delta <= 0:
                raise ValueError("registered membrane tau_m/b did not update")
            if arm == "history_live" and update == 1:
                direct = update_rows[0]["direct_previous_carry_gradient_norms"]
                if (not direct or direct["previous_membrane"] <= 0
                        or direct["previous_event"] <= 0):
                    raise ValueError("actual live previous V/E carry received no direct gradient")
            if arm == "history_detached" and update == 1:
                direct = update_rows[0]["direct_previous_carry_gradient_norms"]
                if (not direct or direct["previous_membrane"] != 0.0
                        or direct["previous_event"] != 0.0):
                    raise ValueError("detached control leaked direct previous V/E carry credit")
            update_rows[-1]["family_gradient_norms"] = grads
            update_rows[-1]["named_core_gradient_norms"] = named_core_grads
            update_rows[-1]["family_parameter_deltas"] = deltas
            update_rows[-1]["named_core_parameter_deltas"] = named_deltas
            update_rows[-1]["tau_m_delta"] = tau_delta
            update_rows[-1]["b_delta"] = b_delta
            update_rows[-1]["critical_credit"] = True
        final_rows, final_q = [], []
        with torch.no_grad():
            for start in (0, 4):
                rgb = torch.from_numpy(np.ascontiguousarray(training["rgb8"][start:start + 4])).to(dev)
                gamma = found.encode(rgb)
                trace = hard_reset.rollout(found.wrapped, gamma, r, total_steps=STEPS,
                                           live_tail_steps=TAIL,
                                           detach_history=(arm == "history_detached"))
                q = q_function(trace["spikes"], trace["component_spikes"],
                               settle=SETTLE, affinity_mode="spike")
                rows = _event_diagnostics(trace)
                for row in rows:
                    row["image_index"] += start
                    row["image_id"] = int(training["selected_ids"][row["image_index"]])
                final_rows.extend(rows)
                final_q.extend(_sha_tensor(q[i]) for i in range(4))
                del rgb, gamma, trace, q
        reports[arm] = {"source_state_sha256_initial": source_digest,
                        "initialization": init_report,
                        "initial_trace_rows": initial_ref,
                        "initial_event_diagnostics": initial_rows,
                        "final_event_diagnostics": final_rows,
                        "final_q_sha256": final_q,
                        "updates": update_rows,
                        "elapsed_seconds": time.perf_counter() - start_time,
                        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(dev))}
        del optimizer, families, found
        torch.cuda.empty_cache()
    return {"experiment": "SW0169_normalized_hard_reset",
            "arms": reports, "calibration": calibration_receipt,
            "training_ids": training["selected_ids"],
            "target_receipts": training["target_receipts"],
            "validation_data_read": False, "ground_truth_scope": "TRAIN8 only",
            "implementation_fingerprint": implementation_fingerprint(),
            "training_admission": True}
