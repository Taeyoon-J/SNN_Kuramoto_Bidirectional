"""Fixed320 GT-last inference for the frozen SW0178 objective arms."""
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

from collaborative_test.SW_0178_temporal_class_competition import train
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
SEED = 1
COUNT, GRID = contract.COUNT, contract.GRID
IMAGE_IDS = np.asarray(contract.IMAGE_IDS, dtype=np.int64)
ARMS = train.ARMS
ENDPOINTS = ("conditional", "standalone")
PREDICTION_ROOT = HERE / "results_archive" / "predictions_20261016"
WORK_ROOT = HERE / "results_archive" / "prediction_work_20261016"
SCORE_PATH = HERE / "results_archive" / "evaluation_seed1_20261016.json"
BOOTSTRAP_SEED, BOOTSTRAP_REPLICATES = 178, 10_000
SW0164_SCORE_SHA256 = "978a9d97658837eb19f414a7fb7dbb35d86c6135cb1213e12e63e2837fd65df6"
TRAIN_QUEUE_MODULE = "collaborative_test.SW_0178_temporal_class_competition.training_queue"
TRAIN_QUEUE_EXPERIMENT = "SW0178_training_queue"


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
        "collaborative_test/SW_0178_temporal_class_competition/DESIGN_KO.md",
        "collaborative_test/SW_0178_temporal_class_competition/class_competition.py",
        "collaborative_test/SW_0178_temporal_class_competition/preflight.py",
        "collaborative_test/SW_0178_temporal_class_competition/train.py",
        "collaborative_test/SW_0178_temporal_class_competition/training_queue.py",
        "collaborative_test/SW_0178_temporal_class_competition/predict_evaluate.py",
        "collaborative_test/SW_0178_temporal_class_competition/prediction_queue.py",
        "collaborative_test/SW_0178_temporal_class_competition/score_queue.py",
        "collaborative_test/SW_0178_temporal_class_competition/test_predict_evaluate.py",
        "collaborative_test/SW_0178_temporal_class_competition/test_prediction_score_queue.py",
        "collaborative_test/SW_0178_temporal_class_competition/test_score_queue.py",
        "collaborative_test/SW_0178_temporal_class_competition/test_train.py",
        "collaborative_test/SW_0178_temporal_class_competition/test_training_queue.py",
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
        raise FileNotFoundError(f"SW0178 evaluation dependency missing: {missing}")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def _training_state_path(arm):
    if arm not in ARMS:
        raise ValueError("unregistered SW0178 arm")
    queue = train._module(TRAIN_QUEUE_MODULE)
    return queue._state_path(arm)


def _training_records(training_inputs=None):
    """Validate each terminal arm; a bound scientific rejection is not a failure."""
    artifacts, manifests, states, state_hashes, rejected = {}, {}, {}, {}, {}
    for arm in ARMS:
        state_path = _training_state_path(arm)
        if not state_path.is_file():
            raise FileNotFoundError(f"SW0178 {arm} queue terminal state is missing")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("experiment") != TRAIN_QUEUE_EXPERIMENT or state.get("arm") != arm:
            raise ValueError(f"SW0178 {arm} training queue state identity is invalid")
        if state.get("status") == "complete_rejected":
            work = train.OUTPUT_ROOT / "work" / f"{arm}_seed1"
            rejection_path = work / "credit_rejection.json"
            if (not rejection_path.is_file()
                    or state.get("artifact_sha256") != sha256_file(rejection_path)
                    or state.get("validation_data_read") is not False):
                raise ValueError(f"SW0178 {arm} rejected-arm receipt is missing or changed")
            rejection = json.loads(rejection_path.read_text(encoding="utf-8"))
            rb = rejection.get("bindings")
            wrong_source = (training_inputs is not None and isinstance(rb, dict) and (
                rb.get("source_state_sha256") != training_inputs.get("source_state_sha256")
                or rb.get("ordered_train_ids") != training_inputs.get("ordered_train_ids")))
            if (rejection.get("experiment") != train.EXPERIMENT
                    or rejection.get("arm") != arm
                    or rejection.get("status") != "scientific_rejection"
                    or rejection.get("completed_updates") not in (1, 2)
                    or not isinstance(rb, dict)
                    or rb.get("training_fingerprint") != train.fingerprint_sha256()
                    or wrong_source):
                raise ValueError(f"SW0178 {arm} scientific rejection is not bound to current TRAIN inputs")
            rejected[arm] = {"state": state, "state_sha256": sha256_file(state_path),
                             "receipt": rejection, "receipt_sha256": sha256_file(rejection_path)}
            continue
        if state.get("status") != "complete":
            raise RuntimeError(f"SW0178 {arm} training is not terminal: {state.get('status')}")
        artifact = train.OUTPUT_ROOT / f"{arm}_seed1"
        manifest = train.validate_training_artifact(arm, artifact)
        manifest_path = artifact / "manifest.json"
        manifest_sha = sha256_file(manifest_path)
        if (state.get("experiment") != TRAIN_QUEUE_EXPERIMENT
                or state.get("arm") != arm or state.get("status") != "complete"
                or state.get("completed_updates") != train.UPDATES
                or state.get("manifest_sha256") != manifest_sha
                or state.get("validation_data_read") is not False):
            raise ValueError(f"SW0178 {arm} training queue does not bind its completed artifact")
        bindings = manifest.get("bindings")
        if (manifest.get("experiment") != train.EXPERIMENT
                or manifest.get("status") != "training_complete"
                or manifest.get("arm") != arm or manifest.get("seed") != SEED
                or manifest.get("completed_updates") != train.UPDATES
                or manifest.get("validation_data_read") is not False
                or not isinstance(bindings, dict)
                or bindings.get("experiment") != train.EXPERIMENT
                or bindings.get("arm") != arm or bindings.get("seed") != SEED
                or bindings.get("mask_supervised") is not True
                or bindings.get("unsupervised_leaderboard_eligible") is not False
                or not _is_sha(bindings.get("source_cache_manifest_sha256"))
                or not _is_sha(bindings.get("source_cache_ledger_sha256"))
                or not isinstance(bindings.get("source_proof"), dict)):
            raise ValueError(f"SW0178 {arm} training manifest lineage is invalid")
        artifacts[arm], manifests[arm], states[arm] = artifact, manifest, state
        state_hashes[arm] = sha256_file(state_path)
    if len(manifests) == len(ARMS):
        first, second = (manifests[arm]["bindings"] for arm in ARMS)
        for key in ("source_proof", "source_state_sha256", "source_checkpoint_sha256",
                    "dataset_manifest_sha256", "dataset_arrays_sha256", "source_cache_manifest_sha256",
                    "source_cache_ledger_sha256", "source_cache_bindings_sha256",
                    "source_cache_first8_row_sha256", "ordered_train_ids",
                    "base_encoder_sha256", "optimizer_groups", "training_fingerprint"):
            if first.get(key) != second.get(key):
                raise ValueError(f"SW0178 paired arms differ in initial registered binding: {key}")
        if manifests[ARMS[0]].get("initial_model_state_sha256") != manifests[ARMS[1]].get("initial_model_state_sha256"):
            raise ValueError("SW0178 arms do not share the same fresh SourceT+zero-context initial state")
    if set(manifests) | set(rejected) != set(ARMS):
        raise ValueError("SW0178 training terminal arms are incomplete")
    return {"artifacts": artifacts, "manifests": manifests, "states": states,
            "state_sha256": state_hashes, "rejected": rejected}


def _references():
    source_t = sw143.validate_arm_prediction(sw143._arm_dir("T", SEED), seed=SEED)
    sw164 = sw164_eval.validate_prediction("foreground_induced")
    if (not np.array_equal(source_t["ids"], IMAGE_IDS)
            or not np.array_equal(sw164["ids"], IMAGE_IDS)
            or source_t.get("arm") != "T"):
        raise ValueError("SW0178 fixed320 Source-T/SW0164 reference IDs changed")
    source_t["prediction_path"] = Path(source_t.get("npz_path", source_t.get("prediction_path")))
    source_t["protocol_path"] = Path(source_t["protocol_path"])
    sw164["prediction_path"] = Path(sw164["directory"]) / "predictions.npz"
    sw164["protocol_path"] = Path(sw164["directory"]) / "protocol.json"
    if (np.asarray(source_t["labels"]).shape != (COUNT, GRID, GRID)
            or np.asarray(sw164["labels"]).shape != (COUNT, GRID, GRID)):
        raise ValueError("SW0178 fixed320 reference label schema changed")
    score_path = Path(sw164_eval.SCORE_PATH)
    score_sha = sha256_file(score_path)
    if score_sha != SW0164_SCORE_SHA256:
        raise ValueError("immutable SW0164 fixed320 score SHA changed")
    score = json.loads(score_path.read_text(encoding="utf-8-sig"))
    return {"source_T": source_t, "SW0164": sw164,
            "SW0164_score": score, "SW0164_score_sha256": score_sha,
            "SW0164_score_path": score_path}


def prepare_prediction_inputs_cpu(*, dataset=None):
    """Validate models/references/RGB cache without reading validation masks."""
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0178 CPU prediction preparation must precede CUDA initialization")
    training_inputs = train.prepare_training_inputs_cpu()
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0178 source/cache preparation unexpectedly initialized CUDA")
    records = _training_records(training_inputs)
    refs = _references()
    # Exercise the exact strict checkpoint loader for both completed producers
    # before prediction workers request an exclusive GPU.
    model_sha = {}
    for arm in records["manifests"]:
        found, actual_sha = _load_model(arm, {"training_inputs": training_inputs,
            "artifacts": records["artifacts"], "manifests": records["manifests"]}, "cpu")
        expected_sha = records["manifests"][arm]["final_model_state_sha256"]
        if actual_sha != expected_sha or torch.cuda.is_initialized():
            raise ValueError(f"SW0178 {arm} CPU checkpoint validation failed")
        model_sha[arm] = actual_sha
        del found
    rgb_path = Path(sw130.VAL_RGB)
    meta_path = rgb_path.with_suffix(rgb_path.suffix + ".complete.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    expected_ids_sha = hashlib.sha256(np.arange(1320, 1640, dtype="<i8").tobytes()).hexdigest()
    rgb = np.load(rgb_path, mmap_mode="r", allow_pickle=False)
    if (meta.get("status") != "complete" or meta.get("cache_shape") != [COUNT, 128, 128, 3]
            or meta.get("ids_mapping_sha256") != expected_ids_sha
            or rgb.dtype != np.uint8 or tuple(rgb.shape) != (COUNT, 128, 128, 3)):
        raise ValueError("SW0178 registered fixed320 RGB cache schema changed")
    rgb_hashes = [hashlib.sha256(np.ascontiguousarray(rgb[i]).tobytes()).hexdigest()
                  for i in range(COUNT)]
    canonical_dataset = Path(sw130.source97.DATASET)
    dataset_path = canonical_dataset if dataset is None else Path(dataset)
    if dataset_path.resolve() != canonical_dataset.resolve() or not dataset_path.is_file():
        raise ValueError("SW0178 validation dataset differs from registered canonical HDF5")
    stat = dataset_path.stat()
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0178 CPU prediction preparation unexpectedly initialized CUDA")
    return {"training_inputs": training_inputs, **records, "model_state_sha256": model_sha,
        "references": refs, "rgb": rgb, "rgb_path": rgb_path,
        "rgb_meta_path": meta_path, "rgb_meta_sha256": sha256_file(meta_path),
        "rgb_hashes": rgb_hashes, "dataset": dataset_path,
        "dataset_path": str(dataset_path.resolve()), "dataset_size_bytes": stat.st_size,
        "dataset_mtime_ns": stat.st_mtime_ns}


def _bindings(prepared, arm):
    if arm not in ARMS:
        raise ValueError("unregistered SW0178 prediction arm")
    manifest = prepared["manifests"][arm]
    artifact = prepared["artifacts"][arm]
    state = prepared["states"][arm]
    refs = prepared["references"]
    return {"experiment": "SW0178_temporal_class_competition_fixed320_prediction", "seed": SEED,
        "arm": arm, "image_ids": IMAGE_IDS.tolist(),
        "implementation_fingerprint": implementation_fingerprint(),
        "training_manifest_sha256": sha256_file(artifact / "manifest.json"),
        "training_checkpoint_sha256": sha256_file(artifact / "checkpoint.pt"),
        "training_optimizer_sha256": sha256_file(artifact / "optimizer.pt"),
        "training_state_sha256": prepared["state_sha256"][arm],
        "training_queue_fingerprint_sha256": state["queue_fingerprint_sha256"],
        "training_bindings_sha256": _json_sha(manifest["bindings"]),
        "training_model_state_sha256": manifest["final_model_state_sha256"],
        "training_fingerprint": manifest["training_fingerprint"],
        "source_cache_manifest_sha256": manifest["bindings"]["source_cache_manifest_sha256"],
        "source_checkpoint_sha256": manifest["bindings"]["source_checkpoint_sha256"],
        "source_state_sha256": manifest["source_model_state_before_sha256"],
        "source_T_prediction_sha256": refs["source_T"]["prediction_sha256"],
        "source_T_protocol_sha256": refs["source_T"]["protocol_sha256"],
        "SW0164_prediction_sha256": refs["SW0164"]["prediction_sha256"],
        "SW0164_protocol_sha256": refs["SW0164"]["protocol_sha256"],
        "SW0164_score_sha256": refs["SW0164_score_sha256"],
        "rgb_meta_sha256": prepared["rgb_meta_sha256"],
        "rgb_hashes_sha256": _json_sha(prepared["rgb_hashes"]),
        "image_ids_sha256": hashlib.sha256(IMAGE_IDS.astype("<i8").tobytes()).hexdigest(),
        "dataset_path": prepared["dataset_path"],
        "dataset_size_bytes": prepared["dataset_size_bytes"],
        "dataset_mtime_ns": prepared["dataset_mtime_ns"],
        "steps": 1024, "settle": 512, "live_tail": 0,
        "optimizer_updates": 0, "ground_truth_used_for_prediction": False}


def _load_model(arm, prepared, device):
    """Strict-load one completed SW0178 model through a fresh SourceT factory."""
    if arm not in ARMS:
        raise ValueError("unregistered SW0178 model arm")
    dev = torch.device(device)
    training_inputs = prepared["training_inputs"]
    if dev.type == "cuda" and not torch.cuda.is_initialized():
        raise RuntimeError("SW0178 GPU model loading requires worker CUDA setup")
    if dev.type == "cpu" and torch.cuda.is_initialized():
        raise RuntimeError("SW0178 CPU model loader cannot run after CUDA initialization")
    found, _optimizer, _families, _binding, source_parts = train._make_fresh(
        arm, training_inputs, dev)
    base_encoder = source_parts[2]
    artifact = prepared["artifacts"][arm]
    manifest = prepared["manifests"][arm]
    checkpoint_path = artifact / "checkpoint.pt"
    if sha256_file(checkpoint_path) != manifest.get("checkpoint_sha256"):
        raise ValueError("SW0178 checkpoint bytes changed after training validation")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (payload.get("experiment") != train.EXPERIMENT or payload.get("arm") != arm
            or payload.get("bindings") != manifest.get("bindings")
            or payload.get("model_state_sha256") != manifest.get("final_model_state_sha256")
            or train._state_sha_from_payload(payload) != manifest.get("final_model_state_sha256")):
        raise ValueError("SW0178 checkpoint payload identity/state SHA invalid")
    found.wrapped.load_state_dict(payload["wrapped"], strict=True)
    found.encoder.load_state_dict(payload["encoder"], strict=True)
    found.wrapped.eval(); found.encoder.eval(); found.patcher.eval(); found.encoder.base_encoder.eval()
    actual_state = train._state_sha(found.wrapped, found.encoder)
    if actual_state != manifest.get("final_model_state_sha256"):
        raise ValueError("SW0178 strict-loaded model tensor digest differs from final manifest")
    source_utils = train._module(train.SOURCE_UTIL_MODULE)
    frozen = source_utils._frozen_source_components(found, base_encoder)
    if frozen != manifest.get("frozen_source_components_after"):
        raise ValueError("SW0178 trained model changed frozen source/base preprocessing components")
    if sha256_file(artifact / "optimizer.pt") != manifest.get("optimizer_sha256"):
        raise ValueError("SW0178 saved optimizer bytes differ from training manifest")
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
        raise ValueError(f"SW0178 row {index} identity/hash mismatch")
    with np.load(archive_path, allow_pickle=False) as data:
        if set(data.files) != set(ENDPOINTS) | {"standalone_support"}:
            raise ValueError("SW0178 row archive keys changed")
        labels = {endpoint: np.asarray(data[endpoint]) for endpoint in ENDPOINTS}
        support = np.asarray(data["standalone_support"])
    if (support.dtype != np.int32 or support.shape != (GRID, GRID)
            or support.min() < 0 or support.max(initial=0) > 1024
            or not np.array_equal(labels["standalone"] > 0, support > 0)):
        raise ValueError("SW0178 standalone output does not preserve its native QCC support")
    for endpoint, array in labels.items():
        if (array.dtype != np.int32 or array.shape != (GRID, GRID)
                or array.min() < 0 or array.max(initial=0) > 1024
                or row.get("labels_sha256", {}).get(endpoint)
                   != hashlib.sha256(array.astype("<i4", copy=False).tobytes()).hexdigest()):
            raise ValueError(f"SW0178 {endpoint} row labels are malformed or hash-mismatched")
    if (not np.array_equal(labels["conditional"] > 0,
                           row.get("source_T_foreground", np.zeros((GRID, GRID), dtype=bool)))
            or row.get("standalone_support_sha256") != hashlib.sha256(
                support.astype("<i4", copy=False).tobytes()).hexdigest()
            or not isinstance(row.get("trace_sha256"), dict)
            or set(row["trace_sha256"]) != set(capture.TRACE_FIELDS)
            or not _is_sha(row.get("q_sha256")) or not _is_sha(row.get("actual_a_sha256"))
            or not np.isfinite(float(row.get("gain", np.nan))) or row["gain"] <= 0):
        raise ValueError("SW0178 row readout/trace metadata is invalid")
    for value in row["trace_sha256"].values():
        if not _is_sha(value):
            raise ValueError("SW0178 trace digest is invalid")
    return labels, support, row


def validate_prediction_arrays(labels, standalone_support, source_t_foreground):
    """Validate endpoint-specific foreground contracts without equating them."""
    if set(labels) != set(ENDPOINTS):
        raise ValueError("SW0178 prediction endpoint keys changed")
    support = np.asarray(standalone_support)
    source = np.asarray(source_t_foreground, dtype=bool)
    if (support.dtype != np.int32 or support.shape != (COUNT, GRID, GRID)
            or support.min() < 0 or support.max(initial=0) > 1024
            or source.shape != (COUNT, GRID, GRID)):
        raise ValueError("SW0178 endpoint support shapes/capacities are invalid")
    checked = {}
    for endpoint, raw in labels.items():
        values = np.asarray(raw)
        if (values.dtype != np.int32 or values.shape != (COUNT, GRID, GRID)
                or values.min() < 0 or values.max(initial=0) > 1024):
            raise ValueError(f"SW0178 {endpoint} label schema is invalid")
        checked[endpoint] = values
    if not np.array_equal(checked["conditional"] > 0, source):
        raise ValueError("SW0178 conditional endpoint must preserve SourceT foreground")
    if not np.array_equal(checked["standalone"] > 0, support > 0):
        raise ValueError("SW0178 standalone endpoint must preserve its own native QCC support")
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
        raise ValueError("SW0178 production CPU native QCC returned malformed labels")
    gain = float(torch.exp(found.wrapped.core.graph_generator.log_coupling_gain.detach()).cpu())
    if not np.isfinite(gain) or gain <= 0:
        raise FloatingPointError("SW0178 actual trained graph gain is invalid")
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
            raise ValueError(f"SW0178 {endpoint} readout changed its registered foreground support")
        outputs[endpoint] = labels
        details[endpoint] = {"merge_count": int(merged["merge_count"]),
                              "support_foreground_patches": int(np.count_nonzero(support))}
    if not np.array_equal(outputs["conditional"] > 0, source_fg):
        raise AssertionError("SW0178 conditional readout changed SourceT foreground")
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
            raise ValueError("SW0178 per-image prediction ledger is incomplete")
        row_hashes.append(sha256_file(row_path))
    return hashlib.sha256(json.dumps(row_hashes, separators=(",", ":")).encode()).hexdigest()


def _predict_row(work, index, image_id, rgb_sha, bindings_sha, result, *, seconds, peak_bytes):
    folder = _rowdir(work, index)
    if folder.exists():
        raise FileExistsError(f"preserving existing SW0178 prediction row {index}")
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


def _out(arm, root=PREDICTION_ROOT):
    if arm not in ARMS: raise ValueError("unregistered SW0178 prediction arm")
    return Path(root) / f"{arm}_seed1"


def _work(arm, root=WORK_ROOT):
    if arm not in ARMS: raise ValueError("unregistered SW0178 prediction arm")
    return Path(root) / f"{arm}_seed1"


@torch.no_grad()
def predict_arm(arm, *, device="cuda:0", prediction_root=PREDICTION_ROOT,
                work_root=WORK_ROOT, prepared=None):
    if arm not in ARMS:
        raise ValueError("unregistered SW0178 prediction arm")
    p = prepared or prepare_prediction_inputs_cpu()
    bindings = _bindings(p, arm); bindings_sha = _json_sha(bindings)
    work = _work(arm, work_root); work.mkdir(parents=True, exist_ok=True)
    binding_file = work / "bindings.json"
    wrapper = {"bindings": bindings, "bindings_sha256": bindings_sha}
    if binding_file.exists():
        if json.loads(binding_file.read_text(encoding="utf-8")) != wrapper:
            raise ValueError("existing SW0178 prediction rows have different training/reference bindings")
    else:
        _write_json_create(binding_file, wrapper)
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0178 prediction requires an exclusive leased CUDA device")
    torch.cuda.set_device(dev)
    found, model_before = _load_model(arm, p, dev)
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
            print(json.dumps({"stage": "SW0178_fixed320_prediction", "arm": arm,
                              "done": index + 1, "count": COUNT}), flush=True)
    model_after = train._state_sha(found.wrapped, found.encoder)
    if model_after != model_before:
        raise RuntimeError("SW0178 trained model tensors changed during fixed320 inference")
    row_ledger_sha = _ledger_sha(work)
    source_state = {"arm": arm, "model_state_sha256_before": model_before,
        "model_state_sha256_after": model_after, "model_unchanged": True,
        "bindings_sha256": bindings_sha, "row_commits_sha256": row_ledger_sha}
    state_file = work / "source_state.json"
    if state_file.exists():
        if json.loads(state_file.read_text(encoding="utf-8")) != source_state:
            raise ValueError("existing SW0178 immutable-model prediction receipt changed")
    else:
        _write_json_create(state_file, source_state)
    return publish_arm(arm, prediction_root=prediction_root, work_root=work_root, prepared=p)


def publish_arm(arm, *, prediction_root=PREDICTION_ROOT, work_root=WORK_ROOT, prepared=None):
    p = prepared or prepare_prediction_inputs_cpu()
    bindings = _bindings(p, arm); bsha = _json_sha(bindings); work = _work(arm, work_root)
    state = json.loads((work / "source_state.json").read_text(encoding="utf-8"))
    if (state.get("arm") != arm or state.get("model_unchanged") is not True
            or state.get("bindings_sha256") != bsha or state.get("row_commits_sha256") != _ledger_sha(work)
            or state.get("model_state_sha256_before") != p["manifests"][arm]["final_model_state_sha256"]
            or state.get("model_state_sha256_after") != state.get("model_state_sha256_before")):
        raise ValueError("SW0178 immutable-model/row receipt is invalid")
    labels = {endpoint: [] for endpoint in ENDPOINTS}
    qcc_rows = []
    for i, image_id in enumerate(IMAGE_IDS.tolist()):
        row_labels, qcc, _ = _read_row(work, i, int(image_id), p["rgb_hashes"][i], bsha)
        for endpoint in ENDPOINTS: labels[endpoint].append(row_labels[endpoint])
        qcc_rows.append(qcc)
    labels = {key: np.stack(rows).astype(np.int32) for key, rows in labels.items()}
    support = np.stack(qcc_rows).astype(np.int32)
    output = _out(arm, prediction_root)
    if output.exists():
        return validate_prediction(arm, prediction_root=prediction_root, work_root=work_root, prepared=p)
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
        protocol = {"experiment": "SW0178_temporal_class_competition_fixed320_prediction",
            "status": "complete", "seed": SEED, "arm": arm, "image_ids": IMAGE_IDS.tolist(),
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
    return validate_prediction(arm, prediction_root=prediction_root, work_root=work_root, prepared=p)


def validate_prediction(arm, *, prediction_root=PREDICTION_ROOT, work_root=WORK_ROOT, prepared=None):
    if arm not in ARMS:
        raise ValueError("unregistered SW0178 prediction arm")
    p = prepared or prepare_prediction_inputs_cpu()
    bindings = _bindings(p, arm); bsha = _json_sha(bindings)
    work, out = _work(arm, work_root), _out(arm, prediction_root)
    source_state = json.loads((work / "source_state.json").read_text(encoding="utf-8"))
    ledger = _ledger_sha(work)
    if (source_state.get("arm") != arm or source_state.get("model_unchanged") is not True
            or source_state.get("bindings_sha256") != bsha
            or source_state.get("row_commits_sha256") != ledger
            or source_state.get("model_state_sha256_before") != p["manifests"][arm]["final_model_state_sha256"]
            or source_state.get("model_state_sha256_after") != source_state.get("model_state_sha256_before")):
        raise ValueError("SW0178 immutable model or row ledger receipt is invalid")
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    archive = out / "predictions.npz"
    expected = {"experiment": "SW0178_temporal_class_competition_fixed320_prediction", "status": "complete",
        "seed": SEED, "arm": arm, "image_ids": IMAGE_IDS.tolist(), "count": COUNT,
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
        raise ValueError("SW0178 prediction protocol/bindings/hash mismatch")
    with np.load(archive, allow_pickle=False) as data:
        ids = contract.require_fixed_image_ids(data["image_ids"], name="SW0178 fixed320 IDs")
        labels = {key: np.asarray(data[key]) for key in ENDPOINTS}
        support = np.asarray(data["standalone_support"])
    labels, support = validate_prediction_arrays(labels, support,
        p["references"]["source_T"]["labels"] > 0)
    for index, image_id in enumerate(IMAGE_IDS.tolist()):
        row_labels, qcc, row = _read_row(work, index, int(image_id), p["rgb_hashes"][index], bsha)
        for endpoint in ENDPOINTS:
            if not np.array_equal(row_labels[endpoint], labels[endpoint][index]):
                raise ValueError("SW0178 published labels differ from immutable row ledger")
        if not np.array_equal(qcc, support[index]):
            raise ValueError("SW0178 published QCC supports differ from immutable row ledger")
    if not np.array_equal(labels["conditional"] > 0,
                          p["references"]["source_T"]["labels"] > 0):
        raise ValueError("SW0178 conditional endpoint changed fixed SourceT foreground")
    return {"arm": arm, "ids": ids, "labels": labels, "standalone_support": support,
        "prediction_sha256": protocol["prediction_sha256"],
        "protocol_sha256": sha256_file(out / "protocol.json"), "protocol": protocol,
        "directory": out}


def _bootstrap(candidate, baseline):
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    if delta.shape != (COUNT,) or not np.isfinite(delta).all():
        raise ValueError("SW0178 paired bootstrap requires two finite fixed320 vectors")
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    indices = rng.randint(0, COUNT, size=(BOOTSTRAP_REPLICATES, COUNT))
    return {"mean": float(delta.mean()),
        "ci95": [float(x) for x in np.percentile(delta[indices].mean(axis=1), [2.5, 97.5])],
        "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES}


def _score_metrics(labels, gt):
    return {name: contract.production_metrics32(np.asarray(value), gt)
            for name, value in labels.items()}


def _score_gates(scores, paired):
    """Evaluate each arm against SW0164 and the registered rho-vs-Q comparison."""
    gates = {"arm_eligibility": {}, "mean_rho_advantage": {}}
    baseline = scores["SW0164_foreground_induced"]
    available = [arm for arm in ARMS if f"{arm}_conditional" in scores]
    for arm in available:
        endpoint_results = {}
        eligible = True
        for endpoint in ENDPOINTS:
            name = f"{arm}_{endpoint}"
            metric = scores[name]
            boot_key = f"{endpoint}_{arm}_minus_SW0164_fg_ari"
            if boot_key not in paired:
                raise ValueError(f"missing registered paired bootstrap result {boot_key}")
            result = {
                "fg_mean_strictly_above_SW0164": metric["fg_ari"]["mean"] > baseline["fg_ari"]["mean"],
                "paired_fg_ci95_lower_above_zero_vs_SW0164": paired[boot_key]["ci95"][0] > 0,
                "foreground_iou_at_least_SW0164": metric["foreground_iou"]["mean"] >= baseline["foreground_iou"]["mean"],
                "matched_object_iou_at_least_SW0164": metric["matched_object_iou"]["mean"] >= baseline["matched_object_iou"]["mean"],
            }
            endpoint_results[endpoint] = result
            eligible = eligible and all(result.values())
        gates["arm_eligibility"][arm] = {"endpoints": endpoint_results, "eligible": bool(eligible)}
    for endpoint in ENDPOINTS:
        if not set(ARMS).issubset(available):
            gates["mean_rho_advantage"][endpoint] = {"status": "unavailable_training_arm_rejected"}
            continue
        rho = scores[f"mean_rho_ce_{endpoint}"]
        qce = scores[f"product_q_ce_{endpoint}"]
        boot_key = f"{endpoint}_mean_rho_ce_minus_product_q_ce_fg_ari"
        if boot_key not in paired:
            raise ValueError(f"missing registered matched-control bootstrap result {boot_key}")
        gates["mean_rho_advantage"][endpoint] = {
            "fg_mean_strictly_above_product_q_ce": rho["fg_ari"]["mean"] > qce["fg_ari"]["mean"],
            "paired_fg_ci95_lower_above_zero_vs_product_q_ce": paired[boot_key]["ci95"][0] > 0,
            "foreground_iou_at_least_product_q_ce": rho["foreground_iou"]["mean"] >= qce["foreground_iou"]["mean"],
            "matched_object_iou_at_least_product_q_ce": rho["matched_object_iou"]["mean"] >= qce["matched_object_iou"]["mean"],
        }
    gates["co_primary_any_arm_eligible"] = any(
        item["eligible"] for item in gates["arm_eligibility"].values())
    gates["mean_rho_mechanism_promoted"] = (
        set(ARMS).issubset(available)
        and gates["arm_eligibility"]["mean_rho_ce"]["eligible"]
        and all(all(result.values()) for result in gates["mean_rho_advantage"].values()))
    return gates


def _prediction_receipts(prepared, prediction_roots=PREDICTION_ROOT, work_root=WORK_ROOT):
    found, rejected = {}, {}
    for arm in prepared["manifests"]:
        result = validate_prediction(arm, prediction_root=prediction_roots, work_root=work_root, prepared=prepared)
        state_path = HERE / "results_archive" / "prediction_queue_v1_20261016" / f"{arm}_state.json"
        if not state_path.is_file():
            raise FileNotFoundError(f"SW0178 {arm} prediction queue state missing before GT")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (state.get("experiment") != "SW0178_prediction_queue_v1"
                or state.get("arm") != arm or state.get("status") != "complete"
                or state.get("prediction_sha256") != result["prediction_sha256"]
                or state.get("protocol_sha256") != result["protocol_sha256"]
                or state.get("ground_truth_used_for_prediction") is not False):
            raise ValueError(f"SW0178 {arm} queue receipt does not bind fixed320 prediction")
        found[arm] = {"result": result, "state": state,
                      "state_sha256": sha256_file(state_path)}
    for arm in prepared["rejected"]:
        state_path = HERE / "results_archive" / "prediction_queue_v1_20261016" / f"{arm}_state.json"
        if not state_path.is_file():
            raise FileNotFoundError(f"SW0178 {arm} terminal rejected prediction queue state missing")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (state.get("experiment") != "SW0178_prediction_queue_v1"
                or state.get("arm") != arm or state.get("status") != "complete_training_rejected"
                or state.get("ground_truth_used_for_prediction") is not False):
            raise ValueError(f"SW0178 {arm} rejected prediction state is invalid")
        rejected[arm] = {"state": state, "state_sha256": sha256_file(state_path)}
    if set(found) | set(rejected) != set(ARMS):
        raise ValueError("SW0178 must terminally freeze every arm prediction or training rejection before GT")
    return {**found, "_rejected": rejected}


def score_all(*, output=SCORE_PATH, prediction_roots=PREDICTION_ROOT,
              work_root=WORK_ROOT, prepared=None):
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    p = prepared or prepare_prediction_inputs_cpu()
    predictions = _prediction_receipts(p, prediction_roots, work_root)
    refs = p["references"]
    source_t_path = refs["source_T"]["prediction_path"]
    sw164_path = refs["SW0164"]["prediction_path"]
    score_path = refs["SW0164_score_path"]
    # Freeze every prediction, score receipt, protocol and dataset identity before
    # the first read of validation masks.
    expected_refs = {key: (value["prediction_sha256"], value["protocol_sha256"])
                     for key, value in refs.items() if key in ("source_T", "SW0164")}
    if sha256_file(score_path) != SW0164_SCORE_SHA256:
        raise ValueError("SW0164 immutable score changed before GT scoring")
    score_baseline = json.loads(score_path.read_text(encoding="utf-8-sig"))
    dataset = p["dataset"]
    before = dataset.stat()
    if before.st_size != p["dataset_size_bytes"] or before.st_mtime_ns != p["dataset_mtime_ns"]:
        raise ValueError("canonical HDF5 identity changed before SW0178 GT scoring")
    with h5py.File(dataset, "r") as hdf5:
        if "mask" not in hdf5 or hdf5["mask"].shape[0] <= int(IMAGE_IDS[-1]):
            raise ValueError("fixed320 canonical HDF5 masks are missing")
        raw = np.asarray(hdf5["mask"][int(IMAGE_IDS[0]):int(IMAGE_IDS[-1]) + 1])
    gt = contract.modal_native32(raw)
    labels = {"SourceT": refs["source_T"]["labels"],
              "SW0164_foreground_induced": refs["SW0164"]["labels"]}
    for arm in p["manifests"]:
        for endpoint in ENDPOINTS:
            labels[f"{arm}_{endpoint}"] = predictions[arm]["result"]["labels"][endpoint]
    scores = _score_metrics(labels, gt)
    expected_sw164 = score_baseline["scores"]["foreground_induced"]
    for metric in contract.METRICS:
        if _json_sha(scores["SW0164_foreground_induced"][metric]) != _json_sha(expected_sw164[metric]):
            raise ValueError(f"SW0164 score metrics fail production recomputation: {metric}")
    paired = {}
    for endpoint in ENDPOINTS:
        baseline_metrics = scores["SW0164_foreground_induced"]
        for arm in p["manifests"]:
            candidate_metrics = scores[f"{arm}_{endpoint}"]
            key = f"{endpoint}_{arm}_minus_SW0164_fg_ari"
            paired[key] = _bootstrap(candidate_metrics["fg_ari"]["per_image"],
                                     baseline_metrics["fg_ari"]["per_image"])
        if set(ARMS).issubset(p["manifests"]):
            rho_metrics = scores[f"mean_rho_ce_{endpoint}"]
            q_metrics = scores[f"product_q_ce_{endpoint}"]
            key = f"{endpoint}_mean_rho_ce_minus_product_q_ce_fg_ari"
            paired[key] = _bootstrap(rho_metrics["fg_ari"]["per_image"],
                                     q_metrics["fg_ari"]["per_image"])
    gates = _score_gates(scores, paired)
    gates["endpoint_foreground_contracts"] = {}
    for arm in p["manifests"]:
        gates["endpoint_foreground_contracts"][arm] = {
            "conditional_exact_SourceT": all(
                np.array_equal(labels[f"{arm}_conditional"][i] > 0, labels["SourceT"][i] > 0)
                for i in range(COUNT)),
            "standalone_exact_native_QCC": all(
                np.array_equal(labels[f"{arm}_standalone"][i] > 0,
                    predictions[arm]["result"]["standalone_support"][i] > 0)
                for i in range(COUNT))}
        contract_ok = all(gates["endpoint_foreground_contracts"][arm].values())
        gates["arm_eligibility"][arm]["eligible"] = (
            gates["arm_eligibility"][arm]["eligible"] and contract_ok)
    gates["co_primary_any_arm_eligible"] = any(
        item["eligible"] for item in gates["arm_eligibility"].values())
    gates["mean_rho_mechanism_promoted"] = (
        gates["mean_rho_mechanism_promoted"]
        and gates["arm_eligibility"].get("mean_rho_ce", {}).get("eligible", False))
    for key, pair in expected_refs.items():
        ref = refs[key]
        if (sha256_file(ref["prediction_path"]) != pair[0]
                or sha256_file(ref["protocol_path"]) != pair[1]):
            raise ValueError(f"immutable {key} reference changed during GT scoring")
    if sha256_file(score_path) != SW0164_SCORE_SHA256:
        raise ValueError("SW0164 score baseline changed during SW0178 scoring")
    current_receipts = _prediction_receipts(p, prediction_roots, work_root)
    for arm in p["manifests"]:
        current = validate_prediction(arm, prediction_root=prediction_roots, work_root=work_root, prepared=p)
        if (current["prediction_sha256"] != predictions[arm]["result"]["prediction_sha256"]
                or current["protocol_sha256"] != predictions[arm]["result"]["protocol_sha256"]
                or current_receipts[arm]["state_sha256"] != predictions[arm]["state_sha256"]):
            raise ValueError(f"SW0178 {arm} predictions changed during GT scoring")
    for arm in p["rejected"]:
        if current_receipts["_rejected"][arm]["state_sha256"] != predictions["_rejected"][arm]["state_sha256"]:
            raise ValueError(f"SW0178 {arm} rejection state changed during GT scoring")
    after = dataset.stat()
    if after.st_size != before.st_size or after.st_mtime_ns != before.st_mtime_ns:
        raise ValueError("canonical HDF5 identity changed during SW0178 scoring")
    report = {"experiment": "SW0178_temporal_class_competition_fixed320_score", "status": "complete",
        "seed_scope": [SEED], "arm_scope": list(p["manifests"]), "rejected_arms": list(p["rejected"]),
        "endpoint_scope": list(ENDPOINTS),
        "implementation_fingerprint": implementation_fingerprint(),
        "prediction_sha256": {arm: predictions[arm]["result"]["prediction_sha256"] for arm in p["manifests"]},
        "prediction_protocol_sha256": {arm: predictions[arm]["result"]["protocol_sha256"] for arm in p["manifests"]},
        "prediction_queue_state_sha256": {arm: predictions[arm]["state_sha256"] for arm in p["manifests"]},
        "rejected_training_state_sha256": {arm: predictions["_rejected"][arm]["state_sha256"] for arm in p["rejected"]},
        "reference_prediction_sha256": {key: refs[key]["prediction_sha256"] for key in ("source_T", "SW0164")},
        "reference_protocol_sha256": {key: refs[key]["protocol_sha256"] for key in ("source_T", "SW0164")},
        "SW0164_score_sha256": SW0164_SCORE_SHA256,
        "training_manifest_sha256": {arm: sha256_file(p["artifacts"][arm] / "manifest.json") for arm in p["manifests"]},
        "scores": scores, "paired_bootstrap": paired, "gate": gates,
        "ground_truth_used_for_prediction": False, "ground_truth_used_for_scoring": True,
        "mask_supervised": True, "unsupervised_leaderboard_eligible": False,
        "ground_truth_scope": "fixed320 canonical validation modal-native32 masks",
        "dataset_path": p["dataset_path"], "dataset_size_bytes": p["dataset_size_bytes"],
        "dataset_mtime_ns": p["dataset_mtime_ns"],
        "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES}
    _atomic_json(output, report)
    return report


def validate_score(path=SCORE_PATH, *, prepared=None):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if (report.get("experiment") != "SW0178_temporal_class_competition_fixed320_score"
            or report.get("status") != "complete"
            or report.get("implementation_fingerprint") != implementation_fingerprint()
            or report.get("ground_truth_used_for_prediction") is not False
            or report.get("ground_truth_used_for_scoring") is not True):
        raise ValueError("SW0178 score provenance or GT scope is invalid")
    # Recompute through the same GT-last producer; the output path already
    # exists, so validate the durable scores without rewriting any report.
    p = prepared or prepare_prediction_inputs_cpu()
    path_expected = Path(path)
    validation_file = path_expected.with_name(
        path_expected.name + f".validation-{os.getpid()}-{uuid.uuid4().hex}.json")
    expected = score_all(output=validation_file, prepared=p)
    try:
        if any(report.get(key) != value for key, value in expected.items() if key != "experiment"):
            raise ValueError("SW0178 independent score recomputation differs from frozen score")
    finally:
        if validation_file.exists():
            validation_file.unlink()
    return report

