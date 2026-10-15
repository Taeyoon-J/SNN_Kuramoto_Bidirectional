"""Source-bound SW0171 initialization, lambda calibration, and TRAIN8 gate."""
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

from collaborative_test.SW_0169_normalized_hard_reset import calibration_v2 as source_cal
from collaborative_test.SW_0171_contextual_instance_graph.contextual_encoder import ContextualFeatureEncoder
from collaborative_test.SW_0171_contextual_instance_graph.instance_objective import (
    capture_native_projection, leave_one_out_instance_loss,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INPUT_ROOT = HERE / "results_archive" / "train_inputs_20261015"
OUTPUT = HERE / "results_archive" / "source_admission_20261015" / "preflight.json"
EXPERIMENT = "SW0171_contextual_instance_graph_preflight"
COUNT, GRID, PATCHES = 512, 32, 1024
CALIBRATION_POSITIONS = tuple(range(16))
GATE_POSITIONS = tuple(range(256, 264))
STEPS, SETTLE, TAIL = 1024, 512, 64
ARMS = ("Q_only", "Q_plus_instance")
TRACE_FIELDS = ("theta", "membrane", "spikes", "component_membrane",
                "component_spikes", "component_gates")
ARRAY_SCHEMA = {
    "rgb": ((512, 128, 128, 3), "uint8"),
    "source_support": ((512, 32, 32), "int32"),
    "feasible_targets": ((512, 32, 32), "int32"),
    "modal_gt": ((512, 32, 32), "int32"),
    "image_ids": ((512,), "int64"),
}
REGISTERED_CACHE_MANIFEST_SHA256 = "388e8748f5ebf7e8155e95b0651ddc59f239a467ea9351f26d2f6b16cebb8a67"
REGISTERED_CACHE_ARRAYS_SHA256 = "4006f191baff3b9e1c9effc8ec67421c011726a5b02b05fdb8c9a432499819ba"


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_tensor(value) -> str:
    return _sha_bytes(value.detach().cpu().contiguous().numpy().tobytes())


def _import(name):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        return importlib.import_module(name)
    finally:
        if visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible


def _producer_bindings_match(manifest: dict, arrays: dict, producer_bindings: dict) -> None:
    if not isinstance(producer_bindings, dict):
        raise ValueError("cache producer_bindings must be an object")
    ids = np.asarray(arrays["image_ids"], dtype="<i8")
    if list(map(int, ids)) != list(map(int, producer_bindings.get("ordered_train_ids", []))):
        raise ValueError("cache IDs differ from frozen producer TRAIN order")
    if producer_bindings.get("source_support_sha256") != _sha_bytes(
            np.ascontiguousarray(arrays["source_support"], dtype="<i4").tobytes()):
        raise ValueError("cached source support differs from producer binding")
    if producer_bindings.get("feasible_targets_sha256") != _sha_bytes(
            np.ascontiguousarray(arrays["feasible_targets"], dtype="<i4").tobytes()):
        raise ValueError("cached feasible targets differ from producer binding")
    rgb_hashes = producer_bindings.get("train_rgb_sha256")
    if (not isinstance(rgb_hashes, list) or len(rgb_hashes) != COUNT
            or any(value != _sha_bytes(np.ascontiguousarray(row).tobytes())
                   for value, row in zip(rgb_hashes, arrays["rgb"]))):
        raise ValueError("cached RGB rows differ from frozen producer bindings")
    if (manifest.get("canonical_hdf5_path") != producer_bindings.get("hdf5_path")
            or manifest.get("canonical_hdf5_size_bytes") != producer_bindings.get("hdf5_size_bytes")
            or manifest.get("canonical_hdf5_mtime_ns") != producer_bindings.get("hdf5_mtime_ns")):
        raise ValueError("cache HDF5 provenance differs from frozen producer bindings")


def validate_dataset_cache(manifest: dict, arrays: dict, *, arrays_sha256: str,
                           producer_fingerprint: dict, expected_ids) -> dict:
    """Validate producer-shaped CPU cache data without loading a model or CUDA."""
    if not isinstance(manifest, dict) or manifest.get("experiment") != "SW0171_original_source_TRAIN_input_cache":
        raise ValueError("unexpected SW0171 dataset-cache experiment")
    if manifest.get("status") != "complete" or manifest.get("count") != COUNT:
        raise ValueError("SW0171 TRAIN input cache is not complete")
    if manifest.get("arrays_sha256") != arrays_sha256:
        raise ValueError("SW0171 arrays archive hash differs from manifest")
    if manifest.get("producer_implementation") != producer_fingerprint:
        raise ValueError("TRAIN cache producer implementation changed")
    if manifest.get("validation_data_read") is not False or manifest.get("candidate_initialization_not_loaded") is not True:
        raise ValueError("cache producer exceeded its registered data/model scope")
    if manifest.get("optimizer_updates") != 0 or manifest.get("cuda_initialized") is not False:
        raise ValueError("cache producer must be data-only with zero optimizer/CUDA")
    if manifest.get("TRAIN_masks_read") is not True or manifest.get("modal_GT_verified_against_canonical_RGB_rows") is not True:
        raise ValueError("cache producer did not attest canonical TRAIN mask alignment")
    if set(arrays) != set(ARRAY_SCHEMA):
        raise ValueError("TRAIN cache array keys differ from the registered schema")
    expected_schema = {key: {"shape": list(shape), "dtype": dtype}
                       for key, (shape, dtype) in ARRAY_SCHEMA.items()}
    if manifest.get("schema") != expected_schema:
        raise ValueError("TRAIN cache manifest schema differs from producer contract")
    for name, (shape, dtype) in ARRAY_SCHEMA.items():
        value = np.asarray(arrays[name])
        if value.shape != shape or value.dtype != np.dtype(dtype):
            raise ValueError(f"TRAIN cache {name} has unexpected shape/dtype")
        if not np.issubdtype(value.dtype, np.integer) and not np.isfinite(value).all():
            raise ValueError(f"TRAIN cache {name} contains nonfinite values")
    ids = list(map(int, np.asarray(arrays["image_ids"])))
    if ids != list(map(int, expected_ids)) or len(set(ids)) != COUNT:
        raise ValueError("TRAIN cache IDs differ from canonical ordered source IDs")
    if set(ids).intersection(range(1320, 1640)) or set(ids).intersection(range(90640, 90960)):
        raise ValueError("TRAIN input cache overlaps registered validation/reserve IDs")
    support = np.asarray(arrays["source_support"])
    target = np.asarray(arrays["feasible_targets"])
    modal = np.asarray(arrays["modal_gt"])
    if np.any(support < 0) or np.any(target < 0) or np.any(modal < 0):
        raise ValueError("source, feasible, and modal labels must be nonnegative")
    if not np.array_equal(support > 0, target > 0):
        raise ValueError("feasible targets must preserve exact frozen source foreground")
    _producer_bindings_match(manifest, arrays, manifest.get("producer_bindings"))
    return {"image_ids": ids, "count": COUNT, "arrays_sha256": arrays_sha256,
            "producer_implementation": producer_fingerprint,
            "canonical_hdf5_path": manifest["canonical_hdf5_path"],
            "canonical_hdf5_size_bytes": int(manifest["canonical_hdf5_size_bytes"]),
            "canonical_hdf5_mtime_ns": int(manifest["canonical_hdf5_mtime_ns"]),
            "validation_data_read": False, "training_ground_truth_scope": "TRAIN512 only"}


def implementation_fingerprint() -> dict:
    source_fp = source_cal.implementation_fingerprint()
    relative = (
        "collaborative_test/SW_0171_contextual_instance_graph/DESIGN_KO.md",
        "collaborative_test/SW_0171_contextual_instance_graph/contextual_encoder.py",
        "collaborative_test/SW_0171_contextual_instance_graph/instance_objective.py",
        "collaborative_test/SW_0171_contextual_instance_graph/preflight.py",
        "collaborative_test/SW_0169_normalized_hard_reset/train.py",
        "collaborative_test/SW_0164_foreground_preserving_partition/foreground_signed.py",
        "collaborative_test/SW_0164_foreground_preserving_partition/predict_evaluate.py",
        "collaborative_test/SW_0164_foreground_preserving_partition/protocol.json",
        "collaborative_test/SW_0168_end_to_end_connectivity/connectivity_objective.py",
        "collaborative_test/SW_0162_connected_signed_partition/preflight.py",
        "collaborative_test/SW_0136_native32_transfer/evaluate.py",
        "collaborative_test/SW_0155_supervised_spike_relations/objective.py",
        "collaborative_test/SW_0134_native_spike_binding/rollout.py",
        "collaborative_test/SW_0135_native32_spike_binding/foundation.py",
        "collaborative_test/SW_0143_native32_selective_geodesic/geodesic_operator.py",
        "snn_kuramoto_bidirectional/graph_generator.py",
        "snn_kuramoto_bidirectional/spike_classifier.py",
    )
    files = {}
    for rel in relative:
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(f"required SW0171 implementation dependency missing: {rel}")
        files[rel] = _sha_file(path)
    return {"files": files, "source_calibration_dependencies": source_fp,
            "experiment": EXPERIMENT, "seed": 1, "context_seed": 171001,
            "calibration_positions": list(CALIBRATION_POSITIONS),
            "gate_positions": list(GATE_POSITIONS), "steps": STEPS,
            "settle": SETTLE, "live_tail": TAIL}


def prepare_inputs_cpu(*, cache_dir=None) -> dict:
    """Verify the frozen data-only producer artifact and source TRAIN contract."""
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0171 CPU preparation must run before CUDA initialization")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    producer = _import("collaborative_test.SW_0169_normalized_hard_reset.train")
    sw130 = _import("collaborative_test.SW_0130_phase_state_integration.run")
    sw155_train = _import("collaborative_test.SW_0155_supervised_spike_relations.train")

    cache_root = Path(cache_dir) if cache_dir is not None else INPUT_ROOT
    manifest_path, arrays_path = cache_root / "manifest.json", cache_root / "arrays.npz"
    if not manifest_path.is_file() or not arrays_path.is_file():
        raise FileNotFoundError("complete SW0171 TRAIN512 producer cache is not available")
    manifest_bytes = manifest_path.read_bytes()
    if _sha_bytes(manifest_bytes) != REGISTERED_CACHE_MANIFEST_SHA256:
        raise ValueError("TRAIN input manifest differs from the registered completed producer receipt")
    manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
    arrays_hash = _sha_file(arrays_path)
    if arrays_hash != REGISTERED_CACHE_ARRAYS_SHA256:
        raise ValueError("TRAIN input arrays differ from the registered completed producer archive")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}

    checkpoint, source_manifest_path, _source_manifest, pool_rows, ordered_ids, checkpoint_sha = sw130.source_contract(1)
    if list(map(int, manifest.get("image_ids", []))) != list(map(int, ordered_ids[:COUNT])):
        raise ValueError("producer IDs do not equal the frozen ordered source TRAIN512 prefix")
    producer_fp = producer.implementation_fingerprint()
    cache_proof = validate_dataset_cache(manifest, arrays, arrays_sha256=arrays_hash,
                                         producer_fingerprint=producer_fp,
                                         expected_ids=ordered_ids[:COUNT])
    bindings = manifest["producer_bindings"]
    if bindings.get("source_checkpoint_sha256") != checkpoint_sha:
        raise ValueError("producer used a different seed-1 source checkpoint")
    actual_source_manifest_sha = _sha_file(source_manifest_path)
    if bindings.get("source_manifest_sha256") != actual_source_manifest_sha:
        raise ValueError("producer used a different canonical source manifest")
    if list(map(int, bindings.get("ordered_train_ids", []))) != cache_proof["image_ids"]:
        raise ValueError("producer source order does not match the canonical source contract")
    pool_rows = np.asarray(pool_rows, dtype=np.int64)
    if pool_rows.shape != (4096,) or np.any(pool_rows[:COUNT] < 0):
        raise ValueError("seed-1 source contract returned invalid ordered TRAIN RGB rows")
    if bindings.get("source_pool_rows_sha256") != _sha_bytes(pool_rows[:COUNT].astype("<i8").tobytes()):
        raise ValueError("cache producer RGB row mapping differs from source contract")
    rgb_path = Path(sw130.TRAIN_RGB)
    rgb_meta_path = rgb_path.with_suffix(rgb_path.suffix + ".complete.json")
    rgb_meta_sha = _sha_file(rgb_meta_path)
    if bindings.get("train_rgb_manifest_sha256") != rgb_meta_sha:
        raise ValueError("cache producer RGB manifest differs from canonical RGB asset")
    rgb_map = np.load(rgb_path, mmap_mode="r", allow_pickle=False)
    if rgb_map.dtype != np.uint8 or tuple(rgb_map.shape) != (70000, 128, 128, 3):
        raise ValueError("canonical registered TRAIN RGB asset schema changed")
    canonical_rgb = np.asarray(rgb_map[pool_rows[:COUNT]]).copy()
    del rgb_map
    if not np.array_equal(arrays["rgb"], canonical_rgb):
        raise ValueError("cache RGB rows differ from canonical source-contract RGB order")
    hdf_path = Path(sw155_train.HDF5_PATH).resolve()
    hdf_stat = hdf_path.stat()
    if (str(hdf_path) != cache_proof["canonical_hdf5_path"]
            or hdf_stat.st_size != cache_proof["canonical_hdf5_size_bytes"]
            or hdf_stat.st_mtime_ns != cache_proof["canonical_hdf5_mtime_ns"]):
        raise ValueError("canonical HDF5 TRAIN source changed since cache production")

    calibration_inputs = source_cal.prepare_calibration_cpu()
    if (calibration_inputs["selected_ids"] != cache_proof["image_ids"][:8]
            or not np.array_equal(calibration_inputs["rgb8"], arrays["rgb"][:8])):
        raise ValueError("first-eight source calibration rows do not match producer cache")
    if (bindings.get("source_state_sha256") != calibration_inputs["source_cache_state_sha256"]
            or bindings.get("source_cache_manifest_sha256") != calibration_inputs["source_cache_manifest_sha256"]
            or bindings.get("context_cache_manifest_sha256") != calibration_inputs["context_cache_manifest_sha256"]):
        raise ValueError("TRAIN data producer used different frozen source/context cache lineage")
    source_refs_module = _import("collaborative_test.SW_0164_foreground_preserving_partition.predict_evaluate")
    source_refs = source_refs_module._references()
    source_reference_fingerprint = source_refs_module.implementation_fingerprint()
    expected_eval_ids = list(map(int, source_refs_module.IMAGE_IDS))
    if (not np.array_equal(source_refs["source_T"]["ids"], expected_eval_ids)
            or not np.array_equal(source_refs["source_qcc"]["ids"], expected_eval_ids)):
        raise ValueError("archived SourceT/SW165 references do not bind the registered fixed320 IDs")
    archived_reference_proof = {
        "source_T_prediction_sha256": source_refs["source_T"]["prediction_sha256"],
        "source_T_protocol_sha256": source_refs["source_T"]["protocol_sha256"],
        "source_qcc_prediction_sha256": source_refs["source_qcc"]["prediction_sha256"],
        "source_qcc_protocol_sha256": source_refs["source_qcc"]["protocol_sha256"],
        "slot_prediction_sha256": source_refs["slot"]["prediction_sha256"],
        "slot_protocol_sha256": source_refs["slot"]["protocol_sha256"],
        "slot_baseline_sha256": source_refs["slot"]["baseline_sha256"],
        "reference_implementation_fingerprint": source_reference_fingerprint,
        "fixed320_ids": expected_eval_ids,
        "prediction_references_only_no_VAL_RGB_or_GT": True,
    }
    fingerprint = implementation_fingerprint()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != visible or torch.cuda.is_initialized():
        raise RuntimeError("CPU preparation changed CUDA visibility or initialized CUDA")
    return {"arrays": arrays, "cache_root": str(cache_root.resolve()),
            "cache_manifest_sha256": _sha_bytes(manifest_bytes),
            "arrays_sha256": arrays_hash, "manifest": manifest,
            "cache_proof": cache_proof, "source_calibration_cpu": calibration_inputs,
            "producer_bindings": bindings, "source_checkpoint_sha256": checkpoint_sha,
            "source_manifest_sha256": actual_source_manifest_sha,
            "archived_reference_proof": archived_reference_proof,
            "implementation_fingerprint": fingerprint,
            "source_proof": {"seed": 1, "ordered_train_ids": cache_proof["image_ids"],
                "source_checkpoint_sha256": checkpoint_sha,
                "source_manifest_sha256": actual_source_manifest_sha,
                "source_cache_manifest_sha256": calibration_inputs["source_cache_manifest_sha256"],
                "source_cache_bindings_sha256": calibration_inputs["source_cache_bindings_sha256"],
                "source_cache_state_sha256": calibration_inputs["source_cache_state_sha256"],
                "context_cache_manifest_sha256": calibration_inputs["context_cache_manifest_sha256"],
                "context_cache_bindings_sha256": calibration_inputs["context_cache_bindings_sha256"],
                "context_cache_state_sha256": calibration_inputs["context_cache_state_sha256"],
                "producer_manifest_sha256": _sha_bytes(manifest_bytes),
                "producer_arrays_sha256": arrays_hash,
                "hdf5_path": cache_proof["canonical_hdf5_path"],
                "hdf5_size_bytes": cache_proof["canonical_hdf5_size_bytes"],
                "hdf5_mtime_ns": cache_proof["canonical_hdf5_mtime_ns"],
                "implementation_fingerprint": fingerprint,
                "validation_data_read": False, "training_ground_truth_scope": "TRAIN512 only"}}


def _state_rng(device):
    cpu = torch.random.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    return cpu, cuda


def _restore_rng(device, state):
    cpu, cuda = state
    torch.random.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state(cuda, device)


def _source_rollout(found, rgb, *, capture_z=False):
    from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
    from collaborative_test.SW_0162_connected_signed_partition.preflight import capture_source_coupling
    graph = found.wrapped.core.graph_generator
    gamma = found.encode(rgb)
    if capture_z:
        with capture_native_projection(graph) as captured:
            trace, adjacency = capture_source_coupling(
                found.wrapped.core.kuramoto,
                lambda: late_rollout(found.wrapped, gamma, total_steps=STEPS,
                                     live_tail_steps=TAIL))
        if len(captured) != 1:
            raise ValueError("native graph projection was not captured in the source rollout")
        z = captured[0]
    else:
        trace, adjacency = capture_source_coupling(
            found.wrapped.core.kuramoto,
            lambda: late_rollout(found.wrapped, gamma, total_steps=STEPS,
                                 live_tail_steps=TAIL))
        z = None
    missing = set(TRACE_FIELDS) - set(trace)
    if missing:
        raise ValueError(f"native source rollout omitted trace fields: {sorted(missing)}")
    for name in TRACE_FIELDS:
        if not bool(torch.isfinite(trace[name]).all()):
            raise FloatingPointError(f"nonfinite source trace field {name}")
    return gamma, trace, adjacency, z


def _solver_readout(q, adjacency, gain, source_labels):
    from collaborative_test.SW_0164_foreground_preserving_partition.foreground_signed import foreground_induced_merge
    result = foreground_induced_merge(q.detach().cpu().numpy(), adjacency.detach().cpu().numpy(),
                                      gain, source_labels, grid_shape=(32, 32))
    return {"energy": result["pair_energy"], "labels": result["labels"],
            "merge_log": result["merge_log"], "merge_count": result["merge_count"]}


def _losses(q, z, feasible, support, modal, objective):
    modal = np.asarray(modal)
    if modal.ndim == 3 and modal.shape[1:] == (GRID, GRID):
        modal = modal.reshape(modal.shape[0], PATCHES)
    if modal.ndim != 2 or modal.shape != tuple(z.shape[:2]):
        raise ValueError("modal TRAIN labels must flatten exactly to native graph nodes")
    losses_q, losses_i = [], []
    for i in range(q.shape[0]):
        lq = objective.constrained_maximin_loss(
            q[i], torch.as_tensor(feasible[i], device=q.device),
            torch.as_tensor(support[i] > 0, device=q.device))
        li = leave_one_out_instance_loss(z[i:i+1], modal[i:i+1])
        losses_q.append(lq)
        losses_i.append(li)
    return torch.stack(losses_q), torch.stack(losses_i)


def _b4_mean_gradient_norm(per_image_losses, gamma) -> float:
    if per_image_losses.ndim != 1 or per_image_losses.shape[0] != gamma.shape[0]:
        raise ValueError("lambda calibration requires one scalar loss per B4 image")
    gradient = torch.autograd.grad(per_image_losses.mean(), gamma, retain_graph=True,
                                   allow_unused=True)[0]
    if gradient is None:
        return 0.0
    if not bool(torch.isfinite(gradient).all()):
        raise FloatingPointError("nonfinite B4-mean loss gradient with respect to gamma")
    return float(gradient.detach().double().norm().cpu())


def _integration_values(wrapped) -> dict:
    values = {}
    for name in ("a_d", "a_m", "b"):
        tensor = getattr(wrapped, name).detach().cpu().reshape(-1)
        if tensor.numel() != 4 or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"native integration parameter {name} must contain four finite values")
        values[name] = tensor.tolist()
    return values


def run_preflight(*, device="cuda:0", prepared=None, output=OUTPUT):
    """Run source parity, TRAIN-only lambda calibration, then fixed TRAIN8 gate."""
    prepared = prepare_inputs_cpu() if prepared is None else prepared
    if prepared.get("implementation_fingerprint") != implementation_fingerprint():
        raise ValueError("SW0171 CPU preparation fingerprint is stale")
    if (_sha_file(Path(prepared["cache_root"]) / "manifest.json") != prepared["cache_manifest_sha256"]
            or _sha_file(Path(prepared["cache_root"]) / "arrays.npz") != prepared["arrays_sha256"]):
        raise ValueError("SW0171 immutable TRAIN input cache changed after CPU preparation")
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0171 source admission requires an available CUDA device")
    if torch.cuda.is_initialized():
        torch.cuda.set_device(dev)
    else:
        torch.cuda.set_device(dev)
    started = time.perf_counter()
    report = {"experiment": EXPERIMENT, "status": "running", "training_admission": False,
              "source_proof": prepared["source_proof"],
              "implementation_fingerprint": prepared["implementation_fingerprint"],
              "ground_truth_scope": "TRAIN calibration positions 0..15 and TRAIN gate positions 256..263",
              "validation_data_read": False, "optimizer_updates": 0}
    try:
        from collaborative_test.SW_0169_normalized_hard_reset.calibration_v2 import _load_source, _source_digest
        from collaborative_test.SW_0135_native32_spike_binding import foundation
        from collaborative_test.SW_0168_end_to_end_connectivity import connectivity_objective
        from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity

        found = _load_source(dev)
        found.encoder.eval(); found.patcher.eval(); found.wrapped.eval()
        graph = found.wrapped.core.graph_generator
        if getattr(graph, "_sw0143_geodesic_arm", None) != "T":
            raise ValueError("SW0171 requires the original SW0143-T graph operator")
        if found.wrapped.core.kuramoto_backend != "factorized":
            raise ValueError("SW0171 requires the frozen native factorized backend")
        source_state_before = _source_digest(found)
        if source_state_before != prepared["source_calibration_cpu"]["source_cache_state_sha256"]:
            raise ValueError("strictly loaded source differs from the seed-1 source cache")
        source_base_encoder = found.encoder
        base_encoder_sha = _module_sha(source_base_encoder)
        graph.geodesic_contrast.requires_grad_(False)
        initial = {"integration": _integration_values(found.wrapped),
                   "feature_clip": float(found.feature_clip),
                   "base_encoder_sha256": base_encoder_sha,
                   "source_state_sha256": source_state_before}
        found.encoder = ContextualFeatureEncoder(source_base_encoder, seed=171001).to(dev)
        found.encoder.eval()
        context_initial_sha = _module_sha(found.encoder)
        rgb_all = prepared["arrays"]["rgb"]
        support = prepared["arrays"]["source_support"]
        target = prepared["arrays"]["feasible_targets"]
        modal = prepared["arrays"]["modal_gt"]
        base_rows, wrapped_rows = [], []
        base_rng = None
        q_losses, i_losses, q_grad_norms, i_grad_norms = [], [], [], []
        for start in (0, 4):
            rgb = torch.from_numpy(np.ascontiguousarray(rgb_all[start:start+4])).to(dev)
            # The original source path and zero-residual wrapper use identical
            # RGB, source state, and RNG for a same-forward initialization check.
            found.encoder = source_base_encoder
            rng = _state_rng(dev)
            with torch.no_grad():
                gamma0, trace0, a0, z0 = _source_rollout(found, rgb, capture_z=True)
                q0 = spike_synchrony_affinity(trace0["spikes"], trace0["component_spikes"],
                                              settle=SETTLE, affinity_mode="spike")
            rng_after_base = _state_rng(dev)
            found.encoder = ContextualFeatureEncoder(source_base_encoder, seed=171001).to(dev)
            found.encoder.eval()
            _restore_rng(dev, rng)
            with torch.enable_grad():
                gamma1, trace1, a1, z1 = _source_rollout(found, rgb, capture_z=True)
                q1 = spike_synchrony_affinity(trace1["spikes"], trace1["component_spikes"],
                                              settle=SETTLE, affinity_mode="spike")
            if (not torch.equal(gamma0, gamma1) or not torch.equal(a0, a1)
                    or not torch.equal(q0, q1) or not torch.equal(z0, z1)):
                raise ValueError(f"zero-residual source gamma/A/Q parity failed for B4 at {start}")
            if any(not torch.equal(trace0[key], trace1[key]) for key in TRACE_FIELDS):
                raise ValueError(f"zero-residual native six-trace parity failed for B4 at {start}")
            labels0, labels1 = _native_qcc(trace0), _native_qcc(trace1)
            for offset in range(4):
                index = start + offset
                source_labels = support[index]
                if (not np.array_equal(labels0[offset], source_labels)
                        or not np.array_equal(labels1[offset], source_labels)):
                    raise ValueError(f"fresh B4 SourceT QCC differs from TRAIN cache at row {index}")
                energy0 = _solver_readout(q0[offset], a0[offset],
                                          prepared["producer_bindings"]["source_gain"], source_labels)
                energy1 = _solver_readout(q1[offset], a1[offset],
                                          prepared["producer_bindings"]["source_gain"], source_labels)
                if (not np.array_equal(energy0["energy"], energy1["energy"])
                        or not np.array_equal(energy0["labels"], energy1["labels"])
                        or energy0["merge_log"] != energy1["merge_log"]):
                    raise ValueError(f"zero-residual SW165 solver parity failed at TRAIN row {index}")
                base_rows.append({"position": index, "image_id": int(prepared["cache_proof"]["image_ids"][index]),
                                  "gamma_sha256": _sha_tensor(gamma0[offset]),
                                  "trace_sha256": {name: _sha_tensor(trace0[name][offset]) for name in TRACE_FIELDS},
                                  "q_sha256": _sha_tensor(q0[offset]), "a_sha256": _sha_tensor(a0[offset]),
                                  "z_sha256": _sha_tensor(z0[offset]),
                                  "source_qcc_sha256": _sha_bytes(labels0[offset].tobytes()),
                                  "solver_labels_sha256": _sha_bytes(energy0["labels"].tobytes()),
                                  "merge_count": energy0["merge_count"]})
                wrapped_rows.append({"position": index, "gamma_sha256": _sha_tensor(gamma1[offset]),
                                     "trace_sha256": {name: _sha_tensor(trace1[name][offset]) for name in TRACE_FIELDS},
                                     "q_sha256": _sha_tensor(q1[offset]), "a_sha256": _sha_tensor(a1[offset]),
                                     "z_sha256": _sha_tensor(z1[offset]),
                                     "source_qcc_sha256": _sha_bytes(labels1[offset].tobytes()),
                                     "solver_labels_sha256": _sha_bytes(energy1["labels"].tobytes())})
            if start == 0:
                lq, li = _losses(q1, z1, target[start:start+4], support[start:start+4],
                                 modal[start:start+4], connectivity_objective)
                q_grad_norms.append(_b4_mean_gradient_norm(lq, gamma1))
                i_grad_norms.append(_b4_mean_gradient_norm(li, gamma1))
                q_losses.extend(float(value.detach().cpu()) for value in lq)
                i_losses.extend(float(value.detach().cpu()) for value in li)
            del rgb, gamma0, trace0, a0, q0, z0, gamma1, trace1, a1, q1, z1, labels0, labels1

        # Also verify the unchanged original SourceT/SW165 path one image at a
        # time on the same registered TRAIN RGB. This is B1 TRAIN parity, not
        # a claim that this TRAIN image matches an archived VAL prediction.
        b1_rgb = torch.from_numpy(np.ascontiguousarray(rgb_all[:1])).to(dev)
        found.encoder = source_base_encoder
        b1_rng = _state_rng(dev)
        with torch.no_grad():
            b1_gamma0, b1_trace0, b1_a0, b1_z0 = _source_rollout(found, b1_rgb, capture_z=True)
            b1_q0 = spike_synchrony_affinity(b1_trace0["spikes"], b1_trace0["component_spikes"],
                                             settle=SETTLE, affinity_mode="spike")
        found.encoder = ContextualFeatureEncoder(source_base_encoder, seed=171001).to(dev)
        found.encoder.eval()
        _restore_rng(dev, b1_rng)
        with torch.enable_grad():
            b1_gamma1, b1_trace1, b1_a1, b1_z1 = _source_rollout(found, b1_rgb, capture_z=True)
            b1_q1 = spike_synchrony_affinity(b1_trace1["spikes"], b1_trace1["component_spikes"],
                                             settle=SETTLE, affinity_mode="spike")
        if (not torch.equal(b1_gamma0, b1_gamma1) or not torch.equal(b1_a0, b1_a1)
                or not torch.equal(b1_q0, b1_q1) or not torch.equal(b1_z0, b1_z1)
                or any(not torch.equal(b1_trace0[key], b1_trace1[key]) for key in TRACE_FIELDS)):
            raise ValueError("B1 original SourceT/context-zero source parity failed")
        b1_labels0, b1_labels1 = _native_qcc(b1_trace0), _native_qcc(b1_trace1)
        if (not np.array_equal(b1_labels0[0], support[0])
                or not np.array_equal(b1_labels1[0], support[0])):
            raise ValueError("B1 production SourceT QCC differs from first TRAIN cache row")
        b1_solver0 = _solver_readout(b1_q0[0], b1_a0[0],
                                     prepared["producer_bindings"]["source_gain"], support[0])
        b1_solver1 = _solver_readout(b1_q1[0], b1_a1[0],
                                     prepared["producer_bindings"]["source_gain"], support[0])
        if (not np.array_equal(b1_solver0["energy"], b1_solver1["energy"])
                or not np.array_equal(b1_solver0["labels"], b1_solver1["labels"])
                or b1_solver0["merge_log"] != b1_solver1["merge_log"]):
            raise ValueError("B1 source/context-zero SW165 solver parity failed")
        b1_reference = {"image_id": int(prepared["cache_proof"]["image_ids"][0]),
            "gamma_sha256": _sha_tensor(b1_gamma1[0]),
            "trace_sha256": {name: _sha_tensor(b1_trace1[name][0]) for name in TRACE_FIELDS},
            "q_sha256": _sha_tensor(b1_q1[0]), "a_sha256": _sha_tensor(b1_a1[0]),
            "z_sha256": _sha_tensor(b1_z1[0]),
            "qcc_sha256": _sha_bytes(b1_labels1[0].tobytes()),
            "sw165_labels_sha256": _sha_bytes(b1_solver1["labels"].tobytes()),
            "parity_exact": True, "uses_train_rgb_not_archived_val_bytes": True}
        del b1_rgb, b1_gamma0, b1_trace0, b1_a0, b1_z0, b1_q0
        del b1_gamma1, b1_trace1, b1_a1, b1_z1, b1_q1, b1_labels0, b1_labels1

        # Continue the registered four-B4 lambda calibration (rows 4..15) on
        # the same zero-initialized candidate, without optimizer updates.
        for start in (4, 8, 12):
            rgb = torch.from_numpy(np.ascontiguousarray(rgb_all[start:start+4])).to(dev)
            with torch.enable_grad():
                gamma, trace, _a, z = _source_rollout(found, rgb, capture_z=True)
                q = spike_synchrony_affinity(trace["spikes"], trace["component_spikes"],
                                             settle=SETTLE, affinity_mode="spike")
                lq, li = _losses(q, z, target[start:start+4], support[start:start+4],
                                 modal[start:start+4], connectivity_objective)
                q_grad_norms.append(_b4_mean_gradient_norm(lq, gamma))
                i_grad_norms.append(_b4_mean_gradient_norm(li, gamma))
                q_losses.extend(float(value.detach().cpu()) for value in lq)
                i_losses.extend(float(value.detach().cpu()) for value in li)
            del rgb, gamma, trace, z, q, lq, li

        q_norm_sum, i_norm_sum = float(sum(q_grad_norms)), float(sum(i_grad_norms))
        lambda_value = q_norm_sum / i_norm_sum if i_norm_sum > 0 else math.nan
        if not math.isfinite(lambda_value) or lambda_value <= 0 or q_norm_sum <= 0:
            report.update({"status": "complete_rejected", "training_admission": False,
                           "rejection_reasons": ["lambda_calibration_norms_not_finite_positive"]})
            report["lambda_calibration"] = {"positions": list(CALIBRATION_POSITIONS),
                "lq_per_image": q_losses, "linst_per_image": i_losses,
                "gq_norm_per_b4": q_grad_norms, "ginst_norm_per_b4": i_grad_norms,
                "gq_norm_sum": q_norm_sum, "ginst_norm_sum": i_norm_sum,
                "lambda": None}
            return _write_report(output, report)

        # Fixed TRAIN8 go/no-go measurement, before any optimizer update.
        gate_rows = []
        for start in (256, 260):
            rgb = torch.from_numpy(np.ascontiguousarray(rgb_all[start:start+4])).to(dev)
            with torch.no_grad():
                gamma, trace, adjacency, z = _source_rollout(found, rgb, capture_z=True)
                q = spike_synchrony_affinity(trace["spikes"], trace["component_spikes"],
                                             settle=SETTLE, affinity_mode="spike")
                qloss, iloss = _losses(q, z, target[start:start+4], support[start:start+4],
                                       modal[start:start+4], connectivity_objective)
                labels_native = _native_qcc(trace)
                for j in range(4):
                    row = start + j
                    gate_rows.append({"position": row, "image_id": int(prepared["cache_proof"]["image_ids"][row]),
                                      "q_loss": float(qloss[j].cpu()), "instance_loss": float(iloss[j].cpu()),
                                      "native_qcc_foreground_nonempty": bool(np.any(labels_native[j] > 0)),
                                      "q_sha256": _sha_tensor(q[j]), "z_sha256": _sha_tensor(z[j])})
            del rgb, gamma, trace, adjacency, z, q, qloss, iloss
        # Evaluate the original source digest with its original encoder object
        # installed. The contextual wrapper intentionally adds state.
        contextual_encoder = found.encoder
        found.encoder = source_base_encoder
        source_state_after = _source_digest(found)
        found.encoder = contextual_encoder
        base_encoder_after = _module_sha(source_base_encoder)
        if source_state_after != source_state_before or base_encoder_after != base_encoder_sha:
            raise ValueError("original source state changed during optimizer-zero preflight")
        report.update({"status": "complete_admitted", "training_admission": True,
            "source_state_sha256_before": source_state_before,
            "source_state_sha256_after": source_state_after,
            "base_encoder_sha256": base_encoder_sha,
            "base_encoder_sha256_after": base_encoder_after,
            "context_zero_init_sha256": context_initial_sha,
            "initialization": initial,
            "source_b4_reference_rows": base_rows,
            "zero_context_b4_rows": wrapped_rows,
            "source_b1_train_reference": b1_reference,
            "initial_b4_gamma_z_six_trace_q_a_sw165_exact": True,
            "lambda_calibration": {"positions": list(CALIBRATION_POSITIONS),
                "lq_per_image": q_losses, "linst_per_image": i_losses,
                "gq_norm_per_b4": q_grad_norms, "ginst_norm_per_b4": i_grad_norms,
                "gq_norm_sum": q_norm_sum, "ginst_norm_sum": i_norm_sum,
                "lambda": lambda_value},
            "fixed_initial_train8": gate_rows,
            "validation_data_read": False, "optimizer_updates": 0,
            "elapsed_seconds": time.perf_counter() - started,
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(dev))})
        return _write_report(output, report)
    except Exception as exc:
        report.update({"status": "failed_preflight", "training_admission": False,
                       "error": f"{type(exc).__name__}: {exc}",
                       "elapsed_seconds": time.perf_counter() - started})
        _write_report(output, report)
        raise


def _module_sha(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8")); digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _native_qcc(trace):
    evaluator = _import("collaborative_test.SW_0136_native32_transfer.evaluate")
    labels = evaluator._qcc_labels(trace)
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    result = np.asarray(labels, dtype=np.int32)
    if result.shape != (trace["component_spikes"].shape[0], 32, 32):
        raise ValueError("native SourceT QCC returned an unexpected batch/grid schema")
    return result


def _write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with staged.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staged, path)
    finally:
        if staged.exists():
            staged.unlink()
    return report
