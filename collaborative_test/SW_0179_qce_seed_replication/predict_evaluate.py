"""Fixed320 GT-free predictions for SW0179 source seeds 0 and 2."""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np
import h5py
import torch

from collaborative_test.SW_0179_qce_seed_replication import train
from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract
from collaborative_test.SW_0136_native32_transfer import evaluate as sw136
from collaborative_test.SW_0143_native32_selective_geodesic import run as sw143
from collaborative_test.SW_0162_connected_signed_partition import preflight as capture
from collaborative_test.SW_0164_foreground_preserving_partition import (
    foreground_signed as solver, predict_evaluate as sw164_eval,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS = (0, 2)
SEED = 0  # Set once per isolated prediction worker via configure_seed().
COUNT, GRID = contract.COUNT, contract.GRID
IMAGE_IDS = np.asarray(contract.IMAGE_IDS, dtype=np.int64)
ARMS = train.ARMS
EXPERIMENT = "SW0179_qce_seed_replication"
PREDICTION_EXPERIMENT = "SW0179_fixed320_prediction"
TRAIN_QUEUE_EXPERIMENT = "SW0179_training_queue"
ENDPOINTS = ("conditional", "standalone")
PREDICTION_ROOT = HERE / "results_archive" / "predictions_20261016"
WORK_ROOT = HERE / "results_archive" / "prediction_work_20261016"
SCORE_PATH = HERE / "results_archive" / "evaluation_seed1_20261016.json"
BOOTSTRAP_SEED, BOOTSTRAP_REPLICATES = 178, 10_000
TRAIN_QUEUE_MODULE = "collaborative_test.SW_0179_qce_seed_replication.training_queue"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def _is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _fingerprint_paths():
    relatives = (
        "collaborative_test/SW_0179_qce_seed_replication/DESIGN_KO.md",
        "collaborative_test/SW_0179_qce_seed_replication/train.py",
        "collaborative_test/SW_0179_qce_seed_replication/training_queue.py",
        "collaborative_test/SW_0179_qce_seed_replication/predict_evaluate.py",
        "collaborative_test/SW_0179_qce_seed_replication/test_predict_evaluate.py",
        "collaborative_test/SW_0171_contextual_instance_graph/preflight.py",
        "collaborative_test/SW_0178_temporal_class_competition/class_competition.py",
        "collaborative_test/SW_0130_phase_state_integration/run.py",
        "collaborative_test/SW_0134_native_spike_binding/rollout.py",
        "collaborative_test/SW_0135_native32_spike_binding/evaluation_contract32.py",
        "collaborative_test/SW_0136_native32_transfer/evaluate.py",
        "collaborative_test/SW_0143_native32_selective_geodesic/run.py",
        "collaborative_test/SW_0143_native32_selective_geodesic/geodesic_operator.py",
        "collaborative_test/SW_0161_supervised_spike_partition/cache.py",
        "collaborative_test/SW_0162_connected_signed_partition/preflight.py",
        "collaborative_test/SW_0164_foreground_preserving_partition/foreground_signed.py",
        "collaborative_test/SW_0164_foreground_preserving_partition/predict_evaluate.py",
        "collaborative_test/SW_0175_critical_component_binding/critical_component_objective.py",
        "collaborative_test/SW_0175_critical_component_binding/train_v2.py",
        "collaborative_test/SW_0148_checkpointable_boundary_q/stage_queue_v4.py",
        "snn_kuramoto_bidirectional/evaluation.py",
        "snn_kuramoto_bidirectional/graph_generator.py",
        "snn_kuramoto_bidirectional/spike_classifier.py",
    )
    return tuple(ROOT / rel for rel in relatives)

def implementation_fingerprint():
    paths = _fingerprint_paths()
    if any(not path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise FileNotFoundError(f"SW0179 evaluation dependency missing: {missing}")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def _training_state_path(arm, seed=None):
    seed = SEED if seed is None else int(seed)
    if arm not in ARMS:
        raise ValueError("unregistered SW0179 arm")
    queue = train._module(TRAIN_QUEUE_MODULE)
    return queue._state_path(arm, seed)


def _training_records(seed, training_inputs=None, expected_queue_fingerprint=None):
    """Validate one terminal seed artifact. Rejections remain terminal and unpredicted."""
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("SW0179 predictions are registered only for source seeds 0 and 2")
    records = {"seed": seed, "artifacts": {}, "manifests": {}, "states": {},
               "state_sha256": {}, "rejected": {}}
    queue = train._module(TRAIN_QUEUE_MODULE)
    state_path = queue._state_path(ARMS[0], seed)
    if not state_path.is_file():
        raise FileNotFoundError(f"SW0179 source seed {seed} training state is missing")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("experiment") != TRAIN_QUEUE_EXPERIMENT or state.get("seed") != seed or state.get("arm") != ARMS[0]:
        raise ValueError("SW0179 training state identity mismatch")
    if state.get("status") == "complete_rejected":
        work = train.OUTPUT_ROOT / "work" / f"{ARMS[0]}_seed{seed}"
        rejection_path = work / "credit_rejection.json"
        if (not rejection_path.is_file() or state.get("artifact_sha256") != sha256_file(rejection_path)
                or state.get("validation_data_read") is not False):
            raise ValueError("SW0179 scientific rejection receipt is missing or changed")
        rejection = json.loads(rejection_path.read_text(encoding="utf-8"))
        if rejection.get("experiment") != train.EXPERIMENT or rejection.get("arm") != ARMS[0]                 or rejection.get("status") != "scientific_rejection" or rejection.get("bindings", {}).get("seed") != seed:
            raise ValueError("SW0179 rejection is not bound to this seed/objective")
        records["rejected"][seed] = {"state": state, "state_sha256": sha256_file(state_path),
                                      "receipt": rejection, "receipt_sha256": sha256_file(rejection_path)}
        return records
    if state.get("status") != "complete" or state.get("validation_data_read") is not False:
        raise RuntimeError(f"SW0179 seed{seed} training is not terminal complete")
    artifact = train.OUTPUT_ROOT / f"{ARMS[0]}_seed{seed}"
    manifest = train.validate_training_artifact(ARMS[0], artifact, seed=seed)
    manifest_sha = sha256_file(artifact / "manifest.json")
    if (expected_queue_fingerprint is not None
            and state.get("queue_fingerprint_sha256") != expected_queue_fingerprint):
        raise ValueError("SW0179 training queue fingerprint differs from current seed-specific CPU admission")
    if (state.get("completed_updates") != train.UPDATES or state.get("manifest_sha256") != manifest_sha
            or manifest.get("experiment") != train.EXPERIMENT or manifest.get("seed") != seed
            or manifest.get("arm") != ARMS[0] or manifest.get("validation_data_read") is not False):
        raise ValueError("SW0179 complete training artifact does not match queue receipt")
    bindings = manifest.get("bindings")
    if (not isinstance(bindings, dict) or bindings.get("seed") != seed
            or bindings.get("mask_supervised") is not True
            or bindings.get("unsupervised_leaderboard_eligible") is not False
            or bindings.get("source_binding", {}).get("seed") != seed
            or not isinstance(bindings.get("dataset_producer_proof"), dict)):
        raise ValueError("SW0179 prediction training bindings are invalid")
    records["artifacts"][ARMS[0]] = artifact
    records["manifests"][ARMS[0]] = manifest
    records["states"][ARMS[0]] = state
    records["state_sha256"][ARMS[0]] = sha256_file(state_path)
    return records

def _references(seed):
    seed = int(seed)
    source_t = sw143.validate_arm_prediction(sw143._arm_dir("T", seed), seed=seed)
    if not np.array_equal(source_t["ids"], IMAGE_IDS) or source_t.get("arm") != "T":
        raise ValueError(f"SW0179 seed{seed} Source-T fixed320 reference IDs changed")
    source_t["prediction_path"] = Path(source_t.get("npz_path", source_t.get("prediction_path")))
    source_t["protocol_path"] = Path(source_t["protocol_path"])
    if np.asarray(source_t["labels"]).shape != (COUNT, GRID, GRID):
        raise ValueError("SW0179 Source-T fixed320 reference schema changed")
    return {"source_T": source_t}


def configure_seed(seed):
    global SEED, PREDICTION_ROOT, WORK_ROOT
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("SW0179 prediction supports only source seeds 0 and 2")
    SEED = seed
    PREDICTION_ROOT = HERE / "results_archive" / "predictions_20261016" / f"seed{seed}"
    WORK_ROOT = HERE / "results_archive" / "prediction_work_20261016" / f"seed{seed}"
    return SEED


def _assert_prepared_seed(prepared, seed):
    if int(prepared.get("seed", -1)) != int(seed):
        raise ValueError("prepared SW0179 prediction inputs belong to a different source seed")


def prepare_prediction_inputs_cpu(seed=None, *, dataset=None):
    """Prepare one seed's completed model and fixed RGB/source refs without opening GT."""
    seed = configure_seed(SEED if seed is None else seed)
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0179 CPU prediction preparation must precede CUDA initialization")
    training_inputs = train.prepare_training_inputs_cpu()
    # Recreate the queue's seed-specific source proof before trusting its receipt.
    fresh = train.prepare_fresh_training_state_cpu(seed, training_inputs)
    proof = fresh["proof"]
    del fresh
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0179 source factory admission unexpectedly initialized CUDA")
    queue = train._module(TRAIN_QUEUE_MODULE)
    identity = {key: training_inputs[key] for key in ("dataset_manifest_sha256", "dataset_arrays_sha256",
        "dataset_producer_proof", "ordered_train_ids")}
    identity["source_factory_proof"] = proof
    _queue_payload, queue_fingerprint = queue._queue_fingerprint(train, ARMS[0], seed, identity)
    records = _training_records(seed, training_inputs, queue_fingerprint)
    if records["rejected"]:
        return {"training_inputs": training_inputs, **records, "rejected": True,
                "references": _references(seed)}
    refs = _references(seed)
    model_sha = {}
    cpu_prepared = {"training_inputs": training_inputs, "artifacts": records["artifacts"],
                    "manifests": records["manifests"], "seed": seed}
    for arm in records["manifests"]:
        found, actual_sha = _load_model(arm, seed, cpu_prepared, "cpu")
        if actual_sha != records["manifests"][arm]["final_model_state_sha256"] or torch.cuda.is_initialized():
            raise ValueError(f"SW0179 seed{seed} CPU checkpoint validation failed")
        model_sha[arm] = actual_sha
        del found
    rgb_path = Path(sw130.VAL_RGB)
    meta_path = rgb_path.with_suffix(rgb_path.suffix + ".complete.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    expected_ids_sha = hashlib.sha256(IMAGE_IDS.astype("<i8").tobytes()).hexdigest()
    rgb = np.load(rgb_path, mmap_mode="r", allow_pickle=False)
    if (meta.get("status") != "complete" or meta.get("cache_shape") != [COUNT, 128, 128, 3]
            or meta.get("ids_mapping_sha256") != expected_ids_sha or rgb.dtype != np.uint8
            or tuple(rgb.shape) != (COUNT, 128, 128, 3)):
        raise ValueError("SW0179 registered fixed320 RGB cache schema changed")
    rgb_hashes = [hashlib.sha256(np.ascontiguousarray(rgb[i]).tobytes()).hexdigest() for i in range(COUNT)]
    canonical_dataset = Path(sw130.source97.DATASET)
    dataset_path = canonical_dataset if dataset is None else Path(dataset)
    if dataset_path.resolve() != canonical_dataset.resolve() or not dataset_path.is_file():
        raise ValueError("SW0179 prediction dataset differs from canonical HDF5")
    stat = dataset_path.stat()
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0179 CPU prediction preparation initialized CUDA")
    return {"training_inputs": training_inputs, **records, "model_state_sha256": model_sha,
        "seed": seed, "references": refs, "rgb": rgb, "rgb_path": rgb_path,
        "rgb_meta_path": meta_path, "rgb_meta_sha256": sha256_file(meta_path),
        "rgb_hashes": rgb_hashes, "dataset": dataset_path, "dataset_path": str(dataset_path.resolve()),
        "dataset_size_bytes": stat.st_size, "dataset_mtime_ns": stat.st_mtime_ns}




def _bindings(prepared, arm):
    if arm != ARMS[0]:
        raise ValueError("unregistered SW0179 objective")
    seed = int(prepared["seed"])
    if seed not in SEEDS:
        raise ValueError("unregistered SW0179 prediction seed")
    manifest = prepared["manifests"][arm]
    artifact = prepared["artifacts"][arm]
    state = prepared["states"][arm]
    refs = prepared["references"]
    return {"experiment": PREDICTION_EXPERIMENT, "seed": seed, "arm": arm,
        "image_ids": IMAGE_IDS.tolist(), "implementation_fingerprint": implementation_fingerprint(),
        "training_manifest_sha256": sha256_file(artifact / "manifest.json"),
        "training_checkpoint_sha256": sha256_file(artifact / "checkpoint.pt"),
        "training_optimizer_sha256": sha256_file(artifact / "optimizer.pt"),
        "training_state_sha256": prepared["state_sha256"][arm],
        "training_queue_fingerprint_sha256": state["queue_fingerprint_sha256"],
        "training_bindings_sha256": _json_sha(manifest["bindings"]),
        "training_model_state_sha256": manifest["final_model_state_sha256"],
        "training_fingerprint": manifest["training_fingerprint"],
        "source_model_binding": manifest["bindings"]["source_binding"],
        "source_checkpoint_sha256": manifest["bindings"]["source_binding"]["source_checkpoint_sha256"],
        "source_state_sha256": manifest["source_model_state_before_sha256"],
        "dataset_producer_proof": manifest["bindings"]["dataset_producer_proof"],
        "dataset_manifest_sha256": manifest["bindings"]["dataset_manifest_sha256"],
        "dataset_arrays_sha256": manifest["bindings"]["dataset_arrays_sha256"],
        "source_T_prediction_sha256": refs["source_T"]["prediction_sha256"],
        "source_T_protocol_sha256": refs["source_T"]["protocol_sha256"],
        "rgb_meta_sha256": prepared["rgb_meta_sha256"],
        "rgb_hashes_sha256": _json_sha(prepared["rgb_hashes"]),
        "image_ids_sha256": hashlib.sha256(IMAGE_IDS.astype("<i8").tobytes()).hexdigest(),
        "dataset_path": prepared["dataset_path"], "dataset_size_bytes": prepared["dataset_size_bytes"],
        "dataset_mtime_ns": prepared["dataset_mtime_ns"], "steps": 1024, "settle": 512,
        "live_tail": 0, "optimizer_updates": 0, "ground_truth_used_for_prediction": False,
        "mask_supervised": True, "unsupervised_leaderboard_eligible": False}




def _load_model(arm, seed, prepared, device):
    seed = int(seed)
    """Strict-load one finished source-seed model through its own fresh factory."""
    if arm != ARMS[0]:
        raise ValueError("unregistered SW0179 objective")
    dev = torch.device(device)
    training_inputs = prepared["training_inputs"]
    if dev.type == "cuda" and not torch.cuda.is_initialized():
        raise RuntimeError("SW0179 GPU model loading requires worker CUDA setup")
    if dev.type == "cpu" and torch.cuda.is_initialized():
        raise RuntimeError("SW0179 CPU model loader cannot run after CUDA initialization")
    found, _optimizer, _families, fresh_binding, source_parts = train._make_fresh(
        arm, seed, training_inputs, dev)
    base_encoder = source_parts[2]
    artifact = prepared["artifacts"][arm]
    manifest = prepared["manifests"][arm]
    fresh_sha = train._state_sha(found.wrapped, found.encoder)
    if (fresh_sha != manifest.get("initial_model_state_sha256")
            or fresh_binding != manifest.get("bindings")):
        raise ValueError("SW0179 fresh seed-specific SourceT/context factory differs from training lineage")
    checkpoint_path = artifact / "checkpoint.pt"
    if sha256_file(checkpoint_path) != manifest.get("checkpoint_sha256"):
        raise ValueError("SW0179 checkpoint bytes changed after training validation")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (payload.get("experiment") != train.EXPERIMENT or payload.get("arm") != arm
            or payload.get("bindings") != manifest.get("bindings")
            or payload.get("model_state_sha256") != manifest.get("final_model_state_sha256")
            or train._state_sha_from_payload(payload) != manifest.get("final_model_state_sha256")):
        raise ValueError("SW0179 checkpoint payload identity/state SHA invalid")
    found.wrapped.load_state_dict(payload["wrapped"], strict=True)
    found.encoder.load_state_dict(payload["encoder"], strict=True)
    found.wrapped.eval(); found.encoder.eval(); found.patcher.eval(); found.encoder.base_encoder.eval()
    actual_state = train._state_sha(found.wrapped, found.encoder)
    if actual_state != manifest.get("final_model_state_sha256"):
        raise ValueError("SW0179 strict-loaded model tensor digest differs from final manifest")
    source_utils = train._module(train.SOURCE_UTIL_MODULE)
    frozen = source_utils._frozen_source_components(found, base_encoder)
    if frozen != manifest.get("frozen_source_components_after"):
        raise ValueError("SW0179 trained model changed frozen source/base preprocessing components")
    if sha256_file(artifact / "optimizer.pt") != manifest.get("optimizer_sha256"):
        raise ValueError("SW0179 saved optimizer bytes differ from training manifest")
    return found, actual_state


def _rowdir(work, index):
    return Path(work) / "rows" / f"{index:04d}"


def _read_row(work, index, image_id, rgb_sha, bindings_sha):
    folder = _rowdir(work, index)
    record_path, archive_path = folder / "row.json", folder / "row.npz"
    row = json.loads(record_path.read_text(encoding="utf-8"))
    if (row.get("index") != index or row.get("image_id") != image_id
            or row.get("rgb_sha256") != rgb_sha or row.get("bindings_sha256") != bindings_sha
            or row.get("npz_sha256") != sha256_file(archive_path)):
        raise ValueError(f"SW0179 row {index} identity/hash mismatch")
    with np.load(archive_path, allow_pickle=False) as data:
        if set(data.files) != set(ENDPOINTS) | {"standalone_support"}:
            raise ValueError("SW0179 row archive keys changed")
        labels = {endpoint: np.asarray(data[endpoint]) for endpoint in ENDPOINTS}
        support = np.asarray(data["standalone_support"])
    if (support.dtype != np.int32 or support.shape != (GRID, GRID)
            or support.min() < 0 or support.max(initial=0) > 1024
            or not np.array_equal(labels["standalone"] > 0, support > 0)):
        raise ValueError("SW0179 standalone output does not preserve its native QCC support")
    for endpoint, array in labels.items():
        if (array.dtype != np.int32 or array.shape != (GRID, GRID)
                or array.min() < 0 or array.max(initial=0) > 1024
                or row.get("labels_sha256", {}).get(endpoint)
                   != hashlib.sha256(array.astype("<i4", copy=False).tobytes()).hexdigest()):
            raise ValueError(f"SW0179 {endpoint} row labels are malformed or hash-mismatched")
    if (not np.array_equal(labels["conditional"] > 0,
                           row.get("source_T_foreground", np.zeros((GRID, GRID), dtype=bool)))
            or row.get("standalone_support_sha256") != hashlib.sha256(
                support.astype("<i4", copy=False).tobytes()).hexdigest()
            or not isinstance(row.get("trace_sha256"), dict)
            or set(row["trace_sha256"]) != set(capture.TRACE_FIELDS)
            or not _is_sha(row.get("q_sha256")) or not _is_sha(row.get("actual_a_sha256"))
            or not np.isfinite(float(row.get("gain", np.nan))) or row["gain"] <= 0):
        raise ValueError("SW0179 row readout/trace metadata is invalid")
    for value in row["trace_sha256"].values():
        if not _is_sha(value):
            raise ValueError("SW0179 trace digest is invalid")
    return labels, support, row


def validate_prediction_arrays(labels, standalone_support, source_t_foreground):
    """Validate endpoint-specific foreground contracts without equating them."""
    if set(labels) != set(ENDPOINTS):
        raise ValueError("SW0179 prediction endpoint keys changed")
    support = np.asarray(standalone_support)
    source = np.asarray(source_t_foreground, dtype=bool)
    if (support.dtype != np.int32 or support.shape != (COUNT, GRID, GRID)
            or support.min() < 0 or support.max(initial=0) > 1024
            or source.shape != (COUNT, GRID, GRID)):
        raise ValueError("SW0179 endpoint support shapes/capacities are invalid")
    checked = {}
    for endpoint, raw in labels.items():
        values = np.asarray(raw)
        if (values.dtype != np.int32 or values.shape != (COUNT, GRID, GRID)
                or values.min() < 0 or values.max(initial=0) > 1024):
            raise ValueError(f"SW0179 {endpoint} label schema is invalid")
        checked[endpoint] = values
    if not np.array_equal(checked["conditional"] > 0, source):
        raise ValueError("SW0179 conditional endpoint must preserve SourceT foreground")
    if not np.array_equal(checked["standalone"] > 0, support > 0):
        raise ValueError("SW0179 standalone endpoint must preserve its own native QCC support")
    return checked, support


def _predict_one(found, image, refs, index):
    gamma = found.encode(image)
    graph = found.wrapped.core.graph_generator
    kuramoto = found.wrapped.core.kuramoto
    trace, actual_a = capture.capture_source_coupling(kuramoto,
        lambda: late_rollout(found.wrapped, gamma, total_steps=1024, live_tail_steps=0))
    capture._trace_check(trace); capture._validate_actual_A(actual_a)
    q = capture._actual_q(trace)
    qcc = np.asarray(sw136._qcc_labels(trace)[0], dtype=np.int32)
    if qcc.shape != (GRID, GRID) or not np.issubdtype(qcc.dtype, np.integer):
        raise ValueError("SW0179 production CPU native QCC returned malformed labels")
    gain = float(torch.exp(found.wrapped.core.graph_generator.log_coupling_gain.detach()).cpu())
    if not np.isfinite(gain) or gain <= 0:
        raise FloatingPointError("SW0179 actual trained graph gain is invalid")
    q_np = q[0].detach().cpu().numpy().astype(np.float64, copy=True)
    a_np = actual_a[0].detach().cpu().numpy().astype(np.float64, copy=True)
    source_fg = np.asarray(refs["source_T"]["labels"][index] > 0, dtype=bool)
    outputs, details = {}, {}
    for endpoint, support in (("conditional", refs["source_T"]["labels"][index]),
                              ("standalone", qcc)):
        merged = solver.foreground_induced_merge(q_np, a_np, gain,
            np.asarray(support, dtype=np.int32), grid_shape=(GRID, GRID))
        labels = np.asarray(merged["labels"], dtype=np.int32)
        if (labels.shape != (GRID, GRID) or labels.min() < 0
                or labels.max(initial=0) > 1024
                or not np.array_equal(labels > 0, np.asarray(support) > 0)):
            raise ValueError(f"SW0179 {endpoint} readout changed its registered foreground support")
        outputs[endpoint] = labels
        details[endpoint] = {"merge_count": int(merged["merge_count"]),
                              "support_foreground_patches": int(np.count_nonzero(support))}
    if not np.array_equal(outputs["conditional"] > 0, source_fg):
        raise AssertionError("SW0179 conditional readout changed SourceT foreground")
    return {"labels": outputs, "standalone_support": qcc,
        "details": details, "gain": gain,
        "q_sha256": hashlib.sha256(np.ascontiguousarray(q_np).tobytes()).hexdigest(),
        "actual_a_sha256": hashlib.sha256(np.ascontiguousarray(a_np).tobytes()).hexdigest(),
        "trace_sha256": {name: capture._tensor_sha(trace[name]) for name in capture.TRACE_FIELDS}}


def _write_json_create(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with tmp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.link(tmp, path)
    finally:
        if tmp.exists(): tmp.unlink()


def _ledger_sha(work):
    row_hashes = []
    for i, image_id in enumerate(IMAGE_IDS.tolist()):
        row_path = _rowdir(work, i) / "row.json"
        if not row_path.is_file():
            raise ValueError("SW0179 per-image prediction ledger is incomplete")
        row_hashes.append(sha256_file(row_path))
    return hashlib.sha256(json.dumps(row_hashes, separators=(",", ":")).encode()).hexdigest()


def _predict_row(work, index, image_id, rgb_sha, bindings_sha, result, *, seconds, peak_bytes):
    folder = _rowdir(work, index)
    if folder.exists():
        raise FileExistsError(f"preserving existing SW0179 prediction row {index}")
    stage = folder.with_name(folder.name + f".stage-{uuid.uuid4().hex}")
    stage.mkdir(parents=True, exist_ok=False)
    try:
        archive = stage / "row.npz"
        with archive.open("xb") as stream:
            np.savez_compressed(stream, **result["labels"], standalone_support=result["standalone_support"])
            stream.flush(); os.fsync(stream.fileno())
        source_fg = np.asarray(result["labels"]["conditional"] > 0, dtype=np.uint8)
        labels_hash = {key: hashlib.sha256(value.astype("<i4", copy=False).tobytes()).hexdigest()
                       for key, value in result["labels"].items()}
        row = {"index": int(index), "image_id": int(image_id), "rgb_sha256": rgb_sha,
            "bindings_sha256": bindings_sha, "labels_sha256": labels_hash,
            "source_T_foreground": source_fg.tolist(),
            "standalone_support_sha256": hashlib.sha256(
                result["standalone_support"].astype("<i4", copy=False).tobytes()).hexdigest(),
            "q_sha256": result["q_sha256"], "actual_a_sha256": result["actual_a_sha256"],
            "trace_sha256": result["trace_sha256"], "gain": float(result["gain"]),
            "details": result["details"], "rollout_seconds": float(seconds),
            "cuda_peak_allocated_bytes": int(peak_bytes), "npz_sha256": sha256_file(archive)}
        _write_json_create(stage / "row.json", row)
        os.rename(stage, folder)
    except BaseException:
        raise
    return row


def _atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with tmp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        if path.exists(): raise FileExistsError(path)
        os.link(tmp, path)
    finally:
        if tmp.exists(): tmp.unlink()


def _out(arm, root=None, seed=None):
    seed = SEED if seed is None else int(seed)
    root = PREDICTION_ROOT if root is None else root
    if arm not in ARMS: raise ValueError("unregistered SW0179 prediction arm")
    return Path(root) / f"{arm}_seed{seed}"


def _work(arm, root=None, seed=None):
    seed = SEED if seed is None else int(seed)
    root = WORK_ROOT if root is None else root
    root = WORK_ROOT if root is None else root
    if arm not in ARMS: raise ValueError("unregistered SW0179 objective")
    return Path(root) / f"{arm}_seed{seed}"


@torch.no_grad()
def predict_arm(arm, *, seed=None, device="cuda:0", prediction_root=None,
                work_root=None, prepared=None):
    seed = configure_seed(SEED if seed is None else seed)
    prediction_root = PREDICTION_ROOT if prediction_root is None else prediction_root
    work_root = WORK_ROOT if work_root is None else work_root
    if arm not in ARMS:
        raise ValueError("unregistered SW0179 prediction arm")
    p = prepared or prepare_prediction_inputs_cpu(seed)
    _assert_prepared_seed(p, seed)
    bindings = _bindings(p, arm); bindings_sha = _json_sha(bindings)
    work = _work(arm, work_root, seed); work.mkdir(parents=True, exist_ok=True)
    binding_file = work / "bindings.json"
    wrapper = {"bindings": bindings, "bindings_sha256": bindings_sha}
    if binding_file.exists():
        if json.loads(binding_file.read_text(encoding="utf-8")) != wrapper:
            raise ValueError("existing SW0179 prediction rows have different training/reference bindings")
    else:
        _write_json_create(binding_file, wrapper)
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0179 prediction requires an exclusive leased CUDA device")
    torch.cuda.set_device(dev)
    found, model_before = _load_model(arm, seed, p, dev)
    for index, image_id in enumerate(IMAGE_IDS.tolist()):
        folder = _rowdir(work, index)
        if folder.exists():
            _read_row(work, index, int(image_id), p["rgb_hashes"][index], bindings_sha)
            continue
        rgb = torch.from_numpy(np.ascontiguousarray(p["rgb"][index:index + 1])).to(dev)
        torch.cuda.reset_peak_memory_stats(dev); started = time.perf_counter()
        result = _predict_one(found, rgb, p["references"], index)
        _predict_row(work, index, int(image_id), p["rgb_hashes"][index], bindings_sha, result,
            seconds=time.perf_counter() - started,
            peak_bytes=torch.cuda.max_memory_allocated(dev))
        del rgb, result
        if (index + 1) % 40 == 0:
            print(json.dumps({"stage": "SW0179_fixed320_prediction", "arm": arm,
                              "done": index + 1, "count": COUNT}), flush=True)
    model_after = train._state_sha(found.wrapped, found.encoder)
    if model_after != model_before:
        raise RuntimeError("SW0179 trained model tensors changed during fixed320 inference")
    row_ledger_sha = _ledger_sha(work)
    source_state = {"experiment": PREDICTION_EXPERIMENT, "seed": seed, "arm": arm,
        "model_state_sha256_before": model_before,
        "model_state_sha256_after": model_after, "model_unchanged": True,
        "bindings_sha256": bindings_sha, "row_commits_sha256": row_ledger_sha}
    state_file = work / "source_state.json"
    if state_file.exists():
        if json.loads(state_file.read_text(encoding="utf-8")) != source_state:
            raise ValueError("existing SW0179 immutable-model prediction receipt changed")
    else:
        _write_json_create(state_file, source_state)
    return publish_arm(arm, seed=seed, prediction_root=prediction_root, work_root=work_root, prepared=p)


def publish_arm(arm, *, seed=None, prediction_root=None, work_root=None, prepared=None):
    seed = configure_seed(SEED if seed is None else seed)
    prediction_root = PREDICTION_ROOT if prediction_root is None else prediction_root
    work_root = WORK_ROOT if work_root is None else work_root
    p = prepared or prepare_prediction_inputs_cpu(seed)
    _assert_prepared_seed(p, seed)
    bindings = _bindings(p, arm); bsha = _json_sha(bindings); work = _work(arm, work_root, seed)
    state = json.loads((work / "source_state.json").read_text(encoding="utf-8"))
    if (state.get("experiment") != PREDICTION_EXPERIMENT or state.get("seed") != seed
            or state.get("arm") != arm or state.get("model_unchanged") is not True
            or state.get("bindings_sha256") != bsha or state.get("row_commits_sha256") != _ledger_sha(work)
            or state.get("model_state_sha256_before") != p["manifests"][arm]["final_model_state_sha256"]
            or state.get("model_state_sha256_after") != state.get("model_state_sha256_before")):
        raise ValueError("SW0179 immutable-model/row receipt is invalid")
    labels = {endpoint: [] for endpoint in ENDPOINTS}
    qcc_rows = []
    for i, image_id in enumerate(IMAGE_IDS.tolist()):
        row_labels, qcc, _ = _read_row(work, i, int(image_id), p["rgb_hashes"][i], bsha)
        for endpoint in ENDPOINTS: labels[endpoint].append(row_labels[endpoint])
        qcc_rows.append(qcc)
    labels = {key: np.stack(rows).astype(np.int32) for key, rows in labels.items()}
    support = np.stack(qcc_rows).astype(np.int32)
    output = _out(arm, prediction_root, seed)
    if output.exists():
        return validate_prediction(arm, seed=seed, prediction_root=prediction_root, work_root=work_root, prepared=p)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(output.name + f".stage-{uuid.uuid4().hex}")
    stage.mkdir(parents=True, exist_ok=False)
    try:
        archive = stage / "predictions.npz"
        with archive.open("xb") as stream:
            np.savez_compressed(stream, image_ids=IMAGE_IDS,
                conditional=labels["conditional"], standalone=labels["standalone"],
                standalone_support=support)
            stream.flush(); os.fsync(stream.fileno())
        protocol = {"experiment": PREDICTION_EXPERIMENT,
            "status": "complete", "seed": int(p["seed"]), "arm": arm, "image_ids": IMAGE_IDS.tolist(),
            "count": COUNT, "prediction_shape": [COUNT, GRID, GRID], "label_capacity": 1024,
            "endpoints": list(ENDPOINTS), "implementation_fingerprint": implementation_fingerprint(),
            "bindings": bindings, "bindings_sha256": bsha,
            "prediction_sha256": sha256_file(archive), "row_commits_sha256": state["row_commits_sha256"],
            "model_state_receipt_sha256": sha256_file(work / "source_state.json"),
            "ground_truth_used_for_prediction": False, "optimizer_updates": 0,
            "steps": 1024, "settle": 512, "live_tail": 0,
            "conditional_foreground_contract": "exact SourceT membership",
            "standalone_foreground_contract": "exact native production QCC membership"}
        _write_json_create(stage / "protocol.json", protocol)
        os.rename(stage, output)
    except BaseException:
        raise
    return validate_prediction(arm, seed=seed, prediction_root=prediction_root, work_root=work_root, prepared=p)


def validate_prediction(arm, *, seed=None, prediction_root=None, work_root=None, prepared=None):
    seed = configure_seed(SEED if seed is None else seed)
    prediction_root = PREDICTION_ROOT if prediction_root is None else prediction_root
    work_root = WORK_ROOT if work_root is None else work_root
    if arm not in ARMS:
        raise ValueError("unregistered SW0179 prediction arm")
    p = prepared or prepare_prediction_inputs_cpu(seed)
    _assert_prepared_seed(p, seed)
    bindings = _bindings(p, arm); bsha = _json_sha(bindings)
    work, out = _work(arm, work_root, seed), _out(arm, prediction_root, seed)
    source_state = json.loads((work / "source_state.json").read_text(encoding="utf-8"))
    ledger = _ledger_sha(work)
    if (source_state.get("experiment") != PREDICTION_EXPERIMENT
            or source_state.get("seed") != seed or source_state.get("arm") != arm
            or source_state.get("model_unchanged") is not True
            or source_state.get("bindings_sha256") != bsha
            or source_state.get("row_commits_sha256") != ledger
            or source_state.get("model_state_sha256_before") != p["manifests"][arm]["final_model_state_sha256"]
            or source_state.get("model_state_sha256_after") != source_state.get("model_state_sha256_before")):
        raise ValueError("SW0179 immutable model or row ledger receipt is invalid")
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    archive = out / "predictions.npz"
    expected = {"experiment": PREDICTION_EXPERIMENT, "status": "complete",
        "seed": int(p["seed"]), "arm": arm, "image_ids": IMAGE_IDS.tolist(), "count": COUNT,
        "prediction_shape": [COUNT, GRID, GRID], "label_capacity": 1024,
        "endpoints": list(ENDPOINTS), "implementation_fingerprint": implementation_fingerprint(),
        "bindings": bindings, "bindings_sha256": bsha, "row_commits_sha256": ledger,
        "model_state_receipt_sha256": sha256_file(work / "source_state.json"),
        "ground_truth_used_for_prediction": False, "optimizer_updates": 0,
        "steps": 1024, "settle": 512, "live_tail": 0,
        "conditional_foreground_contract": "exact SourceT membership",
        "standalone_foreground_contract": "exact native production QCC membership"}
    if any(protocol.get(key) != value for key, value in expected.items()) \
            or protocol.get("prediction_sha256") != sha256_file(archive):
        raise ValueError("SW0179 prediction protocol/bindings/hash mismatch")
    with np.load(archive, allow_pickle=False) as data:
        ids = contract.require_fixed_image_ids(data["image_ids"], name="SW0179 fixed320 IDs")
        labels = {key: np.asarray(data[key]) for key in ENDPOINTS}
        support = np.asarray(data["standalone_support"])
    labels, support = validate_prediction_arrays(labels, support,
        p["references"]["source_T"]["labels"] > 0)
    for index, image_id in enumerate(IMAGE_IDS.tolist()):
        row_labels, qcc, row = _read_row(work, index, int(image_id), p["rgb_hashes"][index], bsha)
        for endpoint in ENDPOINTS:
            if not np.array_equal(row_labels[endpoint], labels[endpoint][index]):
                raise ValueError("SW0179 published labels differ from immutable row ledger")
        if not np.array_equal(qcc, support[index]):
            raise ValueError("SW0179 published QCC supports differ from immutable row ledger")
    if not np.array_equal(labels["conditional"] > 0,
                          p["references"]["source_T"]["labels"] > 0):
        raise ValueError("SW0179 conditional endpoint changed fixed SourceT foreground")
    return {"arm": arm, "ids": ids, "labels": labels, "standalone_support": support,
        "prediction_sha256": protocol["prediction_sha256"],
        "protocol_sha256": sha256_file(out / "protocol.json"), "protocol": protocol,
        "directory": out}



def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--source-seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--physical-gpu", type=int, choices=range(4), required=True)
    args = parser.parse_args(argv)
    if not args.worker:
        raise SystemExit("prediction entrypoint is owned by prediction_queue")
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu)
            or torch.cuda.is_initialized()):
        raise RuntimeError("SW0179 prediction worker CUDA visibility/initialization does not match its lease")
    configure_seed(args.source_seed)
    prepared = prepare_prediction_inputs_cpu(args.source_seed)
    predict_arm(ARMS[0], seed=args.source_seed, device="cuda:0", prepared=prepared)


if __name__ == "__main__":
    main()
