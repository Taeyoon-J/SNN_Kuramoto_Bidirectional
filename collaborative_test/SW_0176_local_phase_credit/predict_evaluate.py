"""Fixed-320 SW0176 prediction and strict GT-last evaluation."""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import h5py
import numpy as np
import torch

from collaborative_test.SW_0176_local_phase_credit import train
from collaborative_test.SW_0175_critical_component_binding import (
    train_v2 as train175, predict_evaluate_v2 as eval175,
)
from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract
from collaborative_test.SW_0136_native32_transfer import evaluate as sw136
from collaborative_test.SW_0143_native32_selective_geodesic import run as sw143, geodesic_operator
from collaborative_test.SW_0161_supervised_spike_partition import cache as source_cache
from collaborative_test.SW_0162_connected_signed_partition import preflight as capture
from collaborative_test.SW_0164_foreground_preserving_partition import (
    foreground_signed as solver, predict_evaluate as sw164_eval,
)
from collaborative_test.SW_0176_local_phase_credit.policy import detach_previous_theta

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEED = 1
COUNT = 320
GRID = 32
IMAGE_IDS = np.asarray(contract.IMAGE_IDS, dtype=np.int64)
ARMS = (train.ARM,)
ENDPOINTS = ("conditional", "standalone")
PREDICTION_ROOT = HERE / "results_archive" / "predictions_20261016"
WORK_ROOT = HERE / "results_archive" / "prediction_work_20261016"
SCORE_PATH = HERE / "results_archive" / "evaluation_seed1_20261016.json"
BOOTSTRAP_SEED = 176
BOOTSTRAP_REPLICATES = 10_000
SW0164_SCORE_SHA256 = "978a9d97658837eb19f414a7fb7dbb35d86c6135cb1213e12e63e2837fd65df6"
SW0175_SCORE_SHA256 = "7311ee86f0ad30a3670edf8bf798f379f200fcd0372cd199e5b0af95fbe5fd02"
SW0175_CONTROL = "critical_component"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _source_cache_manifest_sha256(training_bindings):
    """Read the cache identity from the exact train_v2 producer binding schema."""
    proof = training_bindings.get("source_proof")
    value = proof.get("source_cache_manifest_sha256") if isinstance(proof, dict) else None
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError("training binding lacks its registered nested source-cache proof")
    return value


def _validate_training_lineage(manifest, training_state, manifest_sha256):
    """Check the producer's final manifest and queue-state fields before references or GT."""
    bindings = manifest.get("bindings") if isinstance(manifest, dict) else None
    if (manifest.get("experiment") != train.EXPERIMENT
            or manifest.get("status") != "training_complete"
            or manifest.get("arm") != train.ARM or manifest.get("seed") != SEED
            or manifest.get("completed_updates") != train.UPDATES
            or manifest.get("validation_data_read") is not False
            or not isinstance(bindings, dict)
            or bindings.get("experiment") != train.EXPERIMENT
            or bindings.get("arm") != train.ARM or bindings.get("seed") != SEED
            or not _is_sha256(manifest.get("source_state_sha256_before"))
            or not _is_sha256(manifest.get("final_model_state_sha256"))):
        raise ValueError("SW0176 completed training manifest schema is invalid")
    _source_cache_manifest_sha256(bindings)
    control = bindings.get("control_receipt")
    if (not isinstance(control, dict)
            or control.get("experiment") != train.base.EXPERIMENT
            or control.get("arm") != train.base.ARMS[1]
            or control.get("updates") != 64
            or not _is_sha256(control.get("manifest_sha256"))):
        raise ValueError("SW0176 training manifest lacks its exact frozen SW0175 control receipt")
    if (training_state.get("experiment") != "SW0176_local_phase_training_queue"
            or training_state.get("status") != "complete"
            or training_state.get("manifest_sha256") != manifest_sha256
            or training_state.get("completed_updates") != train.UPDATES):
        raise ValueError("SW0176 prediction requires the exact complete 64-update queue artifact")
    return {"manifest_sha256": manifest_sha256,
            "source_state_sha256": manifest["source_state_sha256_before"],
            "source_cache_manifest_sha256": _source_cache_manifest_sha256(bindings),
            "control_manifest_sha256": control["manifest_sha256"]}


def _fingerprint_paths():
    return tuple(ROOT / rel for rel in (
        "collaborative_test/SW_0176_local_phase_credit/DESIGN_KO.md",
        "collaborative_test/SW_0176_local_phase_credit/policy.py",
        "collaborative_test/SW_0176_local_phase_credit/train.py",
        "collaborative_test/SW_0176_local_phase_credit/training_queue.py",
        "collaborative_test/SW_0176_local_phase_credit/preflight_v4.py",
        "collaborative_test/SW_0176_local_phase_credit/predict_evaluate.py",
        "collaborative_test/SW_0176_local_phase_credit/prediction_queue.py",
        "collaborative_test/SW_0176_local_phase_credit/score_queue.py",
        "collaborative_test/SW_0176_local_phase_credit/test_predict_evaluate.py",
        "collaborative_test/SW_0176_local_phase_credit/test_prediction_score_queue.py",
        "collaborative_test/SW_0175_critical_component_binding/DESIGN_KO.md",
        "collaborative_test/SW_0175_critical_component_binding/critical_component_objective.py",
        "collaborative_test/SW_0175_critical_component_binding/train_v2.py",
        "collaborative_test/SW_0175_critical_component_binding/predict_evaluate_v2.py",
        "collaborative_test/SW_0175_critical_component_binding/prediction_queue_v3.py",
        "collaborative_test/SW_0175_critical_component_binding/score_queue_v3.py",
        "collaborative_test/SW_0130_phase_state_integration/run.py",
        "collaborative_test/SW_0134_native_spike_binding/rollout.py",
        "collaborative_test/SW_0135_native32_spike_binding/foundation.py",
        "collaborative_test/SW_0135_native32_spike_binding/evaluation_contract32.py",
        "collaborative_test/SW_0136_native32_transfer/evaluate.py",
        "collaborative_test/SW_0143_native32_selective_geodesic/run.py",
        "collaborative_test/SW_0143_native32_selective_geodesic/geodesic_operator.py",
        "collaborative_test/SW_0161_supervised_spike_partition/cache.py",
        "collaborative_test/SW_0162_connected_signed_partition/preflight.py",
        "collaborative_test/SW_0164_foreground_preserving_partition/foreground_signed.py",
        "collaborative_test/SW_0164_foreground_preserving_partition/predict_evaluate.py",
        "collaborative_test/SW_0171_contextual_instance_graph/contextual_encoder.py",
        "collaborative_test/SW_0148_checkpointable_boundary_q/stage_queue_v4.py",
        "snn_kuramoto_bidirectional/evaluation.py",
        "snn_kuramoto_bidirectional/graph_generator.py",
        "snn_kuramoto_bidirectional/spike_classifier.py",
    ))


def implementation_fingerprint():
    paths = _fingerprint_paths()
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("SW0175 prediction dependency missing")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def _out(arm, root=PREDICTION_ROOT):
    if arm not in ARMS:
        raise ValueError("unregistered SW0175 prediction arm")
    return Path(root) / f"{arm}_seed1"


def _work(arm, root=WORK_ROOT):
    if arm not in ARMS:
        raise ValueError("unregistered SW0175 prediction arm")
    return Path(root) / f"{arm}_seed1"


def _rowdir(work, index):
    return Path(work) / "rows" / f"{index:04d}"


def _references():
    source_t = sw143.validate_arm_prediction(sw143._arm_dir("T", SEED), seed=SEED)
    sw164 = sw164_eval.validate_prediction("foreground_induced")
    if (not np.array_equal(source_t["ids"], IMAGE_IDS)
            or not np.array_equal(sw164["ids"], IMAGE_IDS)
            or source_t.get("arm") != "T"):
        raise ValueError("SW0175 registered Source-T/SW164 fixed320 IDs changed")
    source_t["prediction_path"] = Path(source_t["npz_path"])
    source_t["protocol_path"] = Path(source_t["protocol_path"])
    sw164["prediction_path"] = Path(sw164["directory"]) / "predictions.npz"
    sw164["protocol_path"] = Path(sw164["directory"]) / "protocol.json"
    sw175 = eval175.validate_prediction(SW0175_CONTROL)
    sw175["prediction_path"] = Path(sw175["directory"]) / "predictions.npz"
    sw175["protocol_path"] = Path(sw175["directory"]) / "protocol.json"
    if (not np.array_equal(sw175["ids"], IMAGE_IDS)
            or source_t["labels"].shape != (COUNT, GRID, GRID)
            or sw164["labels"].shape != (COUNT, GRID, GRID)
            or any(sw175["labels"][endpoint].shape != (COUNT, GRID, GRID)
                   for endpoint in ENDPOINTS)):
        raise ValueError("SW0175 reference label schema changed")
    return {"source_T": source_t, "SW0164": sw164, "SW0175_control": sw175}


def _hash_rgb_rows(rgb):
    return [hashlib.sha256(np.ascontiguousarray(rgb[i]).tobytes()).hexdigest()
            for i in range(COUNT)]


def prepare_prediction_inputs_cpu(*, dataset=None):
    """Validate the completed SW0176 model and frozen references without reading GT."""
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0176 CPU prediction admission must precede CUDA initialization")
    training_inputs = train.prepare_training_inputs_cpu()
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0176 CPU training-input validation unexpectedly initialized CUDA")
    artifact = train.OUTPUT_ROOT / f"{train.ARM}_seed1"
    manifest = train.validate_training_artifact(artifact)
    state_path = train.OUTPUT_ROOT.parent / "training_queue_v1_20261016" / "state.json"
    if not state_path.is_file():
        raise FileNotFoundError("SW0176 training queue terminal state is missing")
    training_state = json.loads(state_path.read_text(encoding="utf-8"))
    manifest_sha = sha256_file(artifact / "manifest.json")
    lineage = _validate_training_lineage(manifest, training_state, manifest_sha)
    bindings0 = manifest["bindings"]
    refs = _references()
    control_manifest = train175.OUTPUT_ROOT / f"{SW0175_CONTROL}_seed1" / "manifest.json"
    if (not control_manifest.is_file()
            or sha256_file(control_manifest) != bindings0.get("control_receipt", {}).get("manifest_sha256")):
        raise ValueError("SW0176 training artifact does not bind the frozen SW0175 critical control")
    baseline_score = Path(eval175.SCORE_PATH)
    if sha256_file(baseline_score) != SW0175_SCORE_SHA256:
        raise ValueError("frozen SW0175 fixed320 score SHA changed")
    rgb_path = Path(sw130.VAL_RGB)
    meta_path = rgb_path.with_suffix(rgb_path.suffix + ".complete.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    expected_map = hashlib.sha256(np.arange(1320, 1640, dtype="<i8").tobytes()).hexdigest()
    rgb = np.load(rgb_path, mmap_mode="r", allow_pickle=False)
    if (meta.get("status") != "complete" or meta.get("cache_shape") != [COUNT, 128, 128, 3]
            or meta.get("ids_mapping_sha256") != expected_map or rgb.dtype != np.uint8
            or tuple(rgb.shape) != (COUNT, 128, 128, 3)):
        raise ValueError("registered fixed320 validation RGB cache changed")
    rgb_hashes = _hash_rgb_rows(rgb)
    dataset_path = Path(dataset) if dataset is not None else Path(sw130.source97.DATASET)
    if dataset_path.resolve() != Path(sw130.source97.DATASET).resolve() or not dataset_path.is_file():
        raise ValueError("scoring dataset differs from registered canonical HDF5")
    st = dataset_path.stat()
    checkpoint_path, source_manifest_path, _manifest, _pool, _ids, checkpoint_sha = (
        sw130.source_contract(SEED))
    if (checkpoint_sha != bindings0.get("source_checkpoint_sha256")
            or sha256_file(source_manifest_path) != bindings0.get("source_manifest_sha256")):
        raise ValueError("trained SW0176 model no longer matches the registered source checkpoint")
    source_cache_manifest = source_cache.CACHE_ROOT / "manifest.json"
    source_cache_hash = sha256_file(source_cache_manifest)
    expected_cache_hash = lineage["source_cache_manifest_sha256"]
    if expected_cache_hash != source_cache_hash:
        raise ValueError("source cache manifest changed after training")
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0176 CPU prediction admission unexpectedly initialized CUDA")
    return {"manifests": {train.ARM: manifest}, "training_state": training_state,
            "training_artifact": artifact, "training_inputs": training_inputs,
            "references": refs, "rgb": rgb, "rgb_path": rgb_path,
            "rgb_manifest_path": meta_path, "rgb_manifest_sha256": sha256_file(meta_path),
            "rgb_hashes": rgb_hashes, "dataset": dataset_path,
            "dataset_path": str(dataset_path.resolve()), "dataset_size_bytes": st.st_size,
            "dataset_mtime_ns": st.st_mtime_ns, "source_checkpoint_path": checkpoint_path,
            "source_manifest_path": source_manifest_path,
            "source_checkpoint_sha256": checkpoint_sha}


def _bindings(prepared, arm):
    if arm not in ARMS:
        raise ValueError("unregistered SW0175 prediction arm")
    manifest = prepared["manifests"][arm]
    refs = prepared["references"]
    stat = prepared["rgb_path"].stat()
    return {"experiment": "SW0176_local_phase_fixed320_prediction", "seed": SEED,
        "arm": arm, "image_ids": IMAGE_IDS.tolist(),
        "implementation_fingerprint": implementation_fingerprint(),
        "training_manifest_sha256": sha256_file(prepared["training_artifact"] / "manifest.json"),
        "training_checkpoint_sha256": sha256_file(prepared["training_artifact"] / "checkpoint.pt"),
        "training_bindings_sha256": _json_sha(manifest["bindings"]),
        "training_model_state_sha256": manifest["final_model_state_sha256"],
        "training_preflight_sha256": manifest["bindings"]["preflight_report_sha256"],
        "source_checkpoint_sha256": prepared["source_checkpoint_sha256"],
        "source_manifest_sha256": sha256_file(prepared["source_manifest_path"]),
        "source_state_sha256": manifest["source_state_sha256_before"],
        "source_state_digest_family": "SW0161 cache._state_digest",
        "Source_T_prediction_sha256": refs["source_T"]["prediction_sha256"],
        "Source_T_protocol_sha256": refs["source_T"]["protocol_sha256"],
        "SW0164_prediction_sha256": refs["SW0164"]["prediction_sha256"],
        "SW0164_protocol_sha256": refs["SW0164"]["protocol_sha256"],
        "SW0175_control_prediction_sha256": refs["SW0175_control"]["prediction_sha256"],
        "SW0175_control_protocol_sha256": refs["SW0175_control"]["protocol_sha256"],
        "rgb_manifest_sha256": prepared["rgb_manifest_sha256"],
        "rgb_size_bytes": stat.st_size, "rgb_mtime_ns": stat.st_mtime_ns,
        "rgb_hashes_sha256": _json_sha(prepared["rgb_hashes"]),
        "dataset_path": prepared["dataset_path"],
        "dataset_size_bytes": prepared["dataset_size_bytes"],
        "dataset_mtime_ns": prepared["dataset_mtime_ns"],
        "steps": 1024, "settle": 512, "live_tail": 0,
        "gradient_intervention": "detach_only_theta_prev_each_kuramoto_transition",
        "matched_control_manifest_sha256": manifest["bindings"]["control_receipt"]["manifest_sha256"],
        "optimizer_updates": 0, "ground_truth_used_for_prediction": False,
        "conditional_support": "registered Source-T native32 labels; all foreground retained",
        "standalone_support": "production SW136 native CPU QCC from the same B1 trace"}


def _sha_tensor(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _load_model(arm, prepared, device):
    if arm != train.ARM:
        raise ValueError("SW0176 inference only loads its registered local-phase candidate")
    artifact = prepared["training_artifact"]
    manifest = prepared["manifests"][arm]
    found, base_encoder, _source_sha, _base_sha, _init_sha, _families = \
        train.preflight_v4._build_source(prepared["training_inputs"]["prepared"], device)
    payload = torch.load(artifact / "checkpoint.pt", map_location="cpu", weights_only=False)
    if (payload.get("arm") != arm or payload.get("bindings") != manifest["bindings"]
            or payload.get("model_state_sha256") != manifest["final_model_state_sha256"]):
        raise ValueError("SW0175 trained checkpoint identity mismatch")
    found.wrapped.load_state_dict(payload["wrapped"], strict=True)
    found.encoder.load_state_dict(payload["encoder"], strict=True)
    found.encoder.eval(); found.patcher.eval(); found.wrapped.eval()
    for module in (found.encoder, found.patcher, found.wrapped):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    before = train.base._state_sha(found.wrapped, found.encoder)
    if before != manifest["final_model_state_sha256"]:
        raise ValueError("loaded SW0175 checkpoint state does not match its completion manifest")
    frozen_components = train.base._frozen_source_components(found, base_encoder)
    if frozen_components != manifest.get("frozen_source_components_after"):
        raise ValueError("loaded SW0175 preprocessing/base/geodesic components differ from training provenance")
    if getattr(found.wrapped.core, "kuramoto_backend", None) != "factorized":
        raise ValueError("SW0175 inference requires the registered factorized backend")
    return found, before


@torch.no_grad()
def _predict_one(found, image, refs, index):
    gamma = found.encode(image)
    with detach_previous_theta(found.wrapped.core.kuramoto):
        trace, actual_a = capture.capture_source_coupling(
            found.wrapped.core.kuramoto,
            lambda: late_rollout(found.wrapped, gamma, total_steps=1024, live_tail_steps=0))
    capture._trace_check(trace)
    capture._validate_actual_A(actual_a)
    q = capture._actual_q(trace)
    gain = float(torch.exp(found.wrapped.core.graph_generator.log_coupling_gain.detach()).cpu())
    if not np.isfinite(gain) or gain <= 0:
        raise FloatingPointError("SW0175 actual trained graph gain is invalid")
    qcc = np.asarray(sw136._qcc_labels(trace)[0], dtype=np.int32)
    source_t = np.asarray(refs["source_T"]["labels"][index], dtype=np.int32)
    if qcc.shape != (GRID, GRID) or source_t.shape != (GRID, GRID):
        raise ValueError("SW0175 support readout has invalid native32 geometry")
    q_np = q[0].detach().cpu().numpy().astype(np.float64, copy=True)
    a_np = actual_a[0].detach().cpu().numpy().astype(np.float64, copy=True)
    outputs = {}
    details = {}
    for endpoint, support_labels in (("conditional", source_t), ("standalone", qcc)):
        result = solver.foreground_induced_merge(q_np, a_np, gain, support_labels,
                                                 grid_shape=(GRID, GRID))
        labels = np.asarray(result["labels"], dtype=np.int32)
        if (labels.shape != (GRID, GRID) or labels.min() < 0
                or not np.array_equal(labels > 0, support_labels > 0)
                or labels.max(initial=0) > 1024):
            raise ValueError(f"SW0175 {endpoint} readout changed foreground membership/capacity")
        outputs[endpoint] = labels
        details[endpoint] = {"merge_count": int(result["merge_count"]),
                             "support_foreground_patches": int(np.count_nonzero(support_labels))}
    return {"labels": outputs, "details": details, "gain": gain,
        "q_sha256": hashlib.sha256(np.ascontiguousarray(q_np).tobytes()).hexdigest(),
        "actual_a_sha256": hashlib.sha256(np.ascontiguousarray(a_np).tobytes()).hexdigest(),
        "trace_sha256": {name: capture._tensor_sha(trace[name]) for name in capture.TRACE_FIELDS},
        "standalone_support": qcc,
        "standalone_qcc_sha256": hashlib.sha256(qcc.astype("<i4", copy=False).tobytes()).hexdigest()}


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


def _read_row(work, index, image_id, rgb_sha, bindings_sha):
    folder = _rowdir(work, index); record_path, archive_path = folder / "row.json", folder / "row.npz"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if (record.get("index") != index or record.get("image_id") != image_id
            or record.get("rgb_sha256") != rgb_sha or record.get("bindings_sha256") != bindings_sha
            or record.get("npz_sha256") != sha256_file(archive_path)):
        raise ValueError(f"SW0175 row {index} provenance/hash mismatch")
    traces = record.get("trace_sha256")
    if (not isinstance(traces, dict) or set(traces) != set(capture.TRACE_FIELDS)
            or any(not _is_sha256(value) for value in traces.values())
            or not _is_sha256(record.get("q_sha256")) or not _is_sha256(record.get("actual_a_sha256"))
            or not _is_sha256(record.get("standalone_qcc_sha256"))
            or not np.isfinite(float(record.get("gain", np.nan))) or record["gain"] <= 0
            or not np.isfinite(float(record.get("rollout_seconds", np.nan)))
            or record["rollout_seconds"] <= 0
            or type(record.get("cuda_peak_allocated_bytes")) is not int
            or record["cuda_peak_allocated_bytes"] < 0
            or not isinstance(record.get("details"), dict)
            or set(record["details"]) != set(ENDPOINTS)):
        raise ValueError("SW0175 row trace/diagnostic metadata invalid")
    with np.load(archive_path, allow_pickle=False) as data:
        if set(data.files) != set(ENDPOINTS) | {"standalone_support"}:
            raise ValueError("SW0175 row archive schema changed")
        labels = {key: np.asarray(data[key]) for key in ENDPOINTS}
        standalone_support = np.asarray(data["standalone_support"])
    if (standalone_support.dtype != np.int32 or standalone_support.shape != (GRID, GRID)
            or standalone_support.min() < 0
            or not np.array_equal(labels["standalone"] > 0, standalone_support > 0)
            or record["standalone_qcc_sha256"] != hashlib.sha256(
                standalone_support.astype("<i4", copy=False).tobytes()).hexdigest()):
        raise ValueError("SW0175 standalone support/QCC row binding is invalid")
    for endpoint, arr in labels.items():
        if arr.dtype != np.int32 or arr.shape != (GRID, GRID) or arr.min() < 0 or arr.max(initial=0) > 1024:
            raise ValueError(f"SW0175 {endpoint} prediction schema invalid")
        expected_sha = record.get("labels_sha256", {}).get(endpoint)
        if expected_sha != hashlib.sha256(arr.astype("<i4", copy=False).tobytes()).hexdigest():
            raise ValueError("SW0175 row label digest mismatch")
    for endpoint, item in record["details"].items():
        if (not isinstance(item, dict) or type(item.get("merge_count")) is not int
                or item["merge_count"] < 0 or type(item.get("support_foreground_patches")) is not int
                or not 0 <= item["support_foreground_patches"] <= GRID * GRID):
            raise ValueError(f"SW0175 {endpoint} per-image details are invalid")
    return labels, record


def _ledger_sha(work):
    hashes = []
    for index in range(COUNT):
        path = _rowdir(work, index) / "row.json"
        if not path.is_file(): raise ValueError("SW0175 prediction row ledger is incomplete")
        hashes.append(sha256_file(path))
    return hashlib.sha256(json.dumps(hashes, separators=(",", ":")).encode()).hexdigest()


def _commit_prediction_row(work, index, image_id, rgb_sha, bindings_sha, result,
                           *, rollout_seconds, cuda_peak_allocated_bytes):
    folder = _rowdir(work, index)
    if folder.exists():
        raise FileExistsError(f"preserving existing SW0175 row {index}")
    stage = folder.with_name(folder.name + f".stage-{uuid.uuid4().hex}")
    stage.mkdir(parents=True, exist_ok=False)
    try:
        archive = stage / "row.npz"
        with archive.open("xb") as stream:
            np.savez_compressed(stream, **result["labels"], standalone_support=result["standalone_support"])
            stream.flush(); os.fsync(stream.fileno())
        record = {"index": index, "image_id": int(image_id), "rgb_sha256": rgb_sha,
            "bindings_sha256": bindings_sha,
            "labels_sha256": {endpoint: hashlib.sha256(labels.astype("<i4", copy=False).tobytes()).hexdigest()
                              for endpoint, labels in result["labels"].items()},
            "q_sha256": result["q_sha256"], "actual_a_sha256": result["actual_a_sha256"],
            "standalone_qcc_sha256": result["standalone_qcc_sha256"],
            "trace_sha256": result["trace_sha256"], "gain": result["gain"],
            "details": result["details"], "rollout_seconds": float(rollout_seconds),
            "cuda_peak_allocated_bytes": int(cuda_peak_allocated_bytes),
            "npz_sha256": sha256_file(archive)}
        with (stage / "row.json").open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.rename(stage, folder)
    except BaseException:
        raise
    return record


@torch.no_grad()
def predict_arm(arm, *, device="cuda:0", prediction_root=PREDICTION_ROOT,
                work_root=WORK_ROOT, prepared=None):
    if arm not in ARMS: raise ValueError("unregistered SW0175 prediction arm")
    p = prepared or prepare_prediction_inputs_cpu()
    bindings = _bindings(p, arm); bindings_sha = _json_sha(bindings)
    work = _work(arm, work_root); work.mkdir(parents=True, exist_ok=True)
    bind_file = work / "bindings.json"
    wrapper = {"bindings": bindings, "bindings_sha256": bindings_sha}
    if bind_file.exists():
        if json.loads(bind_file.read_text(encoding="utf-8")) != wrapper:
            raise ValueError("existing SW0175 per-image work has different bindings")
    else: _atomic_json(bind_file, wrapper)
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0175 fixed320 prediction requires owner-leased CUDA")
    torch.cuda.set_device(device)
    found, model_before = _load_model(arm, p, device)
    for index, image_id in enumerate(IMAGE_IDS.tolist()):
        folder = _rowdir(work, index)
        if folder.exists():
            _read_row(work, index, int(image_id), p["rgb_hashes"][index], bindings_sha)
            continue
        image = torch.from_numpy(np.ascontiguousarray(p["rgb"][index:index+1])).to(device)
        torch.cuda.reset_peak_memory_stats(device); started = time.perf_counter()
        result = _predict_one(found, image, p["references"], index)
        _commit_prediction_row(work, index, int(image_id), p["rgb_hashes"][index], bindings_sha,
            result, rollout_seconds=time.perf_counter()-started,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
        del image, result
        if (index+1) % 40 == 0:
            print(json.dumps({"stage": "SW0175_prediction", "arm": arm,
                              "done": index+1, "count": COUNT}), flush=True)
    model_after = train.base._state_sha(found.wrapped, found.encoder)
    if model_after != model_before:
        raise RuntimeError("SW0175 trained model changed during inference")
    state = {"arm": arm, "model_state_sha256_before": model_before,
        "model_state_sha256_after": model_after, "model_unchanged": True,
        "bindings_sha256": bindings_sha, "row_commits_sha256": _ledger_sha(work)}
    state_path = work / "source_state.json"
    if state_path.exists():
        if json.loads(state_path.read_text(encoding="utf-8")) != state:
            raise ValueError("existing SW0175 inference-state receipt changed")
    else: _atomic_json(state_path, state)
    return publish_arm(arm, prediction_root=prediction_root, work_root=work_root, prepared=p)


def publish_arm(arm, *, prediction_root=PREDICTION_ROOT, work_root=WORK_ROOT, prepared=None):
    p = prepared or prepare_prediction_inputs_cpu()
    bindings = _bindings(p, arm); bsha = _json_sha(bindings); work = _work(arm, work_root)
    state = json.loads((work / "source_state.json").read_text(encoding="utf-8"))
    if (state.get("model_unchanged") is not True or state.get("bindings_sha256") != bsha
            or state.get("row_commits_sha256") != _ledger_sha(work)
            or state.get("model_state_sha256_before") != p["manifests"][arm]["final_model_state_sha256"]
            or state.get("model_state_sha256_after") != state.get("model_state_sha256_before")):
        raise ValueError("SW0175 source/model-state or prediction-row receipt invalid")
    arrays = {endpoint: [] for endpoint in ENDPOINTS}
    for index, image_id in enumerate(IMAGE_IDS.tolist()):
        labels, _ = _read_row(work, index, int(image_id), p["rgb_hashes"][index], bsha)
        for endpoint in ENDPOINTS: arrays[endpoint].append(labels[endpoint])
    arrays = {endpoint: np.stack(rows).astype(np.int32) for endpoint, rows in arrays.items()}
    out = _out(arm, prediction_root)
    if out.exists(): return validate_prediction(arm, prediction_root=prediction_root, work_root=work_root, prepared=p)
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = out.with_name(out.name + f".stage-{uuid.uuid4().hex}"); stage.mkdir(parents=True, exist_ok=False)
    try:
        archive = stage / "predictions.npz"
        with archive.open("xb") as stream:
            np.savez_compressed(stream, image_ids=IMAGE_IDS, conditional=arrays["conditional"],
                                standalone=arrays["standalone"])
            stream.flush(); os.fsync(stream.fileno())
        protocol = {"experiment": "SW0176_local_phase_fixed320_prediction", "status": "complete",
            "arm": arm, "seed": SEED, "image_ids": IMAGE_IDS.tolist(), "count": COUNT,
            "prediction_shape": [COUNT, GRID, GRID], "label_capacity": 1024,
            "endpoints": list(ENDPOINTS), "implementation_fingerprint": implementation_fingerprint(),
            "bindings": bindings, "bindings_sha256": bsha,
            "prediction_sha256": sha256_file(archive), "row_commits_sha256": state["row_commits_sha256"],
            "model_state_receipt_sha256": sha256_file(work / "source_state.json"),
            "ground_truth_used_for_prediction": False, "optimizer_updates": 0,
            "steps": 1024, "settle": 512, "live_tail": 0}
        _atomic_json(stage / "protocol.json", protocol); os.rename(stage, out)
    except BaseException: raise
    return validate_prediction(arm, prediction_root=prediction_root, work_root=work_root, prepared=p)


def validate_prediction(arm, *, prediction_root=PREDICTION_ROOT, work_root=WORK_ROOT, prepared=None):
    if arm not in ARMS: raise ValueError("unregistered SW0175 prediction arm")
    p = prepared or prepare_prediction_inputs_cpu()
    bindings = _bindings(p, arm); bsha = _json_sha(bindings); work = _work(arm, work_root); out = _out(arm, prediction_root)
    state = json.loads((work / "source_state.json").read_text(encoding="utf-8"))
    if (state.get("model_unchanged") is not True or state.get("bindings_sha256") != bsha
            or state.get("row_commits_sha256") != _ledger_sha(work)
            or state.get("model_state_sha256_before") != p["manifests"][arm]["final_model_state_sha256"]
            or state.get("model_state_sha256_after") != state.get("model_state_sha256_before")):
        raise ValueError("SW0175 immutable-model/row ledger receipt invalid")
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8")); archive = out / "predictions.npz"
    expected = {"experiment": "SW0176_local_phase_fixed320_prediction", "status": "complete",
        "arm": arm, "seed": SEED, "image_ids": IMAGE_IDS.tolist(), "count": COUNT,
        "prediction_shape": [COUNT, GRID, GRID], "label_capacity": 1024,
        "endpoints": list(ENDPOINTS), "implementation_fingerprint": implementation_fingerprint(),
        "bindings": bindings, "bindings_sha256": bsha, "row_commits_sha256": state["row_commits_sha256"],
        "model_state_receipt_sha256": sha256_file(work / "source_state.json"),
        "ground_truth_used_for_prediction": False, "optimizer_updates": 0,
        "steps": 1024, "settle": 512, "live_tail": 0}
    if any(protocol.get(k) != v for k, v in expected.items()) or protocol.get("prediction_sha256") != sha256_file(archive):
        raise ValueError("SW0175 prediction protocol/bindings/hash mismatch")
    with np.load(archive, allow_pickle=False) as data:
        ids = contract.require_fixed_image_ids(data["image_ids"], name="SW0175 fixed320 prediction IDs")
        published = {endpoint: np.asarray(data[endpoint]) for endpoint in ENDPOINTS}
    for endpoint, values in published.items():
        if values.shape != (COUNT, GRID, GRID) or values.dtype != np.int32 or values.min() < 0 or values.max(initial=0) > 1024:
            raise ValueError(f"SW0175 published {endpoint} schema invalid")
    for index, image_id in enumerate(IMAGE_IDS.tolist()):
        row, _ = _read_row(work, index, int(image_id), p["rgb_hashes"][index], bsha)
        for endpoint in ENDPOINTS:
            if not np.array_equal(row[endpoint], published[endpoint][index]):
                raise ValueError("SW0175 published predictions differ from committed row ledger")
        if not np.array_equal(published["conditional"][index] > 0, p["references"]["source_T"]["labels"][index] > 0):
            raise ValueError("SW0175 conditional prediction changed fixed Source-T foreground")
    return {"arm": arm, "ids": ids, "labels": published,
        "prediction_sha256": protocol["prediction_sha256"],
        "protocol_sha256": sha256_file(out / "protocol.json"), "protocol": protocol,
        "directory": out}


def _bootstrap(candidate, baseline):
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    if delta.shape != (COUNT,) or not np.isfinite(delta).all():
        raise ValueError("SW0175 paired vectors must be finite fixed320 arrays")
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    indices = rng.randint(0, COUNT, size=(BOOTSTRAP_REPLICATES, COUNT))
    sampled = delta[indices].mean(axis=1)
    return {"mean": float(delta.mean()), "ci95": [float(x) for x in np.percentile(sampled, [2.5, 97.5])],
            "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES}


def _score_metrics(labels, gt):
    return {name: contract.production_metrics32(value, gt) for name, value in labels.items()}


def _reference_hashes(references):
    """Recheck immutable predictions/protocols loaded by the public validators."""
    result = {}
    for name, reference in references.items():
        prediction_path = Path(reference["prediction_path"])
        protocol_path = Path(reference["protocol_path"])
        prediction_sha = sha256_file(prediction_path)
        protocol_sha = sha256_file(protocol_path)
        if (prediction_sha != reference.get("prediction_sha256")
                or protocol_sha != reference.get("protocol_sha256")):
            raise ValueError(f"immutable SW0176 comparison reference changed: {name}")
        result[name] = {"prediction_sha256": prediction_sha,
                        "protocol_sha256": protocol_sha}
    return result


def _is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


# The fixed320 SW0176 producer has one learned arm and two immutable controls;
# these final definitions intentionally replace the copied two-arm SW0175 template.
def _score_fixed320_core(*, prediction_root, work_root, prepared):
    candidate = validate_prediction(train.ARM, prediction_root=prediction_root,
                                    work_root=work_root, prepared=prepared)
    refs = prepared["references"]
    _reference_hashes(refs)
    sw164_score_path = Path(sw164_eval.SCORE_PATH)
    sw175_score_path = Path(eval175.SCORE_PATH)
    if (sha256_file(sw164_score_path) != SW0164_SCORE_SHA256
            or sha256_file(sw175_score_path) != SW0175_SCORE_SHA256):
        raise ValueError("fixed320 immutable SW0164/SW0175 score receipt changed before GT")
    sw164_report = json.loads(sw164_score_path.read_text(encoding="utf-8-sig"))
    sw175_report = json.loads(sw175_score_path.read_text(encoding="utf-8-sig"))
    before = prepared["dataset"].stat()
    if before.st_size != prepared["dataset_size_bytes"] or before.st_mtime_ns != prepared["dataset_mtime_ns"]:
        raise ValueError("canonical HDF5 identity changed before GT scoring")
    with h5py.File(prepared["dataset"], "r") as hdf5:
        if "mask" not in hdf5 or hdf5["mask"].shape[0] <= int(IMAGE_IDS[-1]):
            raise ValueError("canonical HDF5 validation mask range is missing")
        raw = np.asarray(hdf5["mask"][int(IMAGE_IDS[0]):int(IMAGE_IDS[-1]) + 1])
    gt = contract.modal_native32(raw)
    label_sets = {
        "SW0164": refs["SW0164"]["labels"],
        "Source-T": refs["source_T"]["labels"],
        "SW0175_critical_component_conditional": refs["SW0175_control"]["labels"]["conditional"],
        "SW0175_critical_component_standalone": refs["SW0175_control"]["labels"]["standalone"],
        "local_phase_conditional": candidate["labels"]["conditional"],
        "local_phase_standalone": candidate["labels"]["standalone"],
    }
    scores = _score_metrics(label_sets, gt)
    for metric in contract.METRICS:
        if _json_sha(scores["SW0164"][metric]) != _json_sha(
                sw164_report["scores"]["foreground_induced"][metric]):
            raise ValueError(f"SW0164 reference metrics fail production recomputation: {metric}")
        for endpoint in ENDPOINTS:
            key = f"SW0175_critical_component_{endpoint}"
            expected = sw175_report["scores"][f"critical_component_{endpoint}"][metric]
            if _json_sha(scores[key][metric]) != _json_sha(expected):
                raise ValueError(f"SW0175 matched control metrics fail production recomputation: {endpoint}/{metric}")
    for endpoint in ENDPOINTS:
        if not np.array_equal(label_sets[f"local_phase_{endpoint}"] > 0,
                              label_sets["Source-T"] > 0):
            raise ValueError("SW0176 readout changed fixed Source-T foreground membership")
        if not np.array_equal(scores[f"local_phase_{endpoint}"]["foreground_iou"]["per_image"],
                              scores["Source-T"]["foreground_iou"]["per_image"]):
            raise ValueError("SW0176 readout changed fixed Source-T foreground IoU")
    paired, gates = {}, {}
    for endpoint in ENDPOINTS:
        cand = scores[f"local_phase_{endpoint}"]
        comparisons = {
            "SW0164": scores["SW0164"],
            "SW0175_critical_component": scores[f"SW0175_critical_component_{endpoint}"],
        }
        for name, baseline in comparisons.items():
            key = f"{endpoint}_local_phase_minus_{name}_fg_ari"
            paired[key] = _bootstrap(cand["fg_ari"]["per_image"], baseline["fg_ari"]["per_image"])
            gates[f"{endpoint}_fg_ari_mean_above_{name}"] = (
                cand["fg_ari"]["mean"] > baseline["fg_ari"]["mean"])
            gates[f"{endpoint}_fg_ari_ci95_lower_above_zero_{name}"] = (
                paired[key]["ci95"][0] > 0)
            gates[f"{endpoint}_foreground_iou_not_below_{name}"] = (
                cand["foreground_iou"]["mean"] >= baseline["foreground_iou"]["mean"])
            gates[f"{endpoint}_matched_object_iou_not_below_{name}"] = (
                cand["matched_object_iou"]["mean"] >= baseline["matched_object_iou"]["mean"])
    gates["conditional_foreground_membership_exact_Source_T"] = True
    gates["co_primary_gate_passed"] = all(gates.values())
    _reference_hashes(refs)
    if (sha256_file(sw164_score_path) != SW0164_SCORE_SHA256
            or sha256_file(sw175_score_path) != SW0175_SCORE_SHA256):
        raise ValueError("immutable baseline score changed during SW0176 scoring")
    after = prepared["dataset"].stat()
    if after.st_size != prepared["dataset_size_bytes"] or after.st_mtime_ns != prepared["dataset_mtime_ns"]:
        raise ValueError("canonical HDF5 changed during SW0176 scoring")
    fresh = validate_prediction(train.ARM, prediction_root=prediction_root,
                                work_root=work_root, prepared=prepared)
    if fresh["prediction_sha256"] != candidate["prediction_sha256"]:
        raise ValueError("SW0176 fixed320 prediction changed during GT scoring")
    return candidate, scores, paired, gates


def score_all(*, output=SCORE_PATH, prediction_root=PREDICTION_ROOT, work_root=WORK_ROOT, prepared=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    p = prepared or prepare_prediction_inputs_cpu()
    candidate, scores, paired, gates = _score_fixed320_core(
        prediction_root=prediction_root, work_root=work_root, prepared=p)
    report = {"experiment": "SW0176_local_phase_fixed320_score", "status": "complete",
        "seed_scope": [SEED], "endpoint_scope": list(ENDPOINTS),
        "implementation_fingerprint": implementation_fingerprint(),
        "prediction_sha256": candidate["prediction_sha256"],
        "prediction_protocol_sha256": candidate["protocol_sha256"],
        "reference_prediction_sha256": {key: value["prediction_sha256"] for key, value in p["references"].items()},
        "reference_protocol_sha256": {key: value["protocol_sha256"] for key, value in p["references"].items()},
        "SW0164_score_sha256": SW0164_SCORE_SHA256,
        "SW0175_score_sha256": SW0175_SCORE_SHA256,
        "matched_control_training_manifest_sha256": p["manifests"][train.ARM]["bindings"]["control_receipt"]["manifest_sha256"],
        "scores": scores, "paired_bootstrap": paired, "gate": gates,
        "ground_truth_used_for_prediction": False, "ground_truth_used_for_scoring": True,
        "ground_truth_scope": "fixed320 canonical validation modal-native32 masks",
        "dataset_path": p["dataset_path"], "dataset_size_bytes": p["dataset_size_bytes"],
        "dataset_mtime_ns": p["dataset_mtime_ns"], "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES}
    _atomic_json(output, report)
    validate_score(output, prediction_root=prediction_root, work_root=work_root, prepared=p)
    return report


def validate_score(path=SCORE_PATH, *, prediction_root=PREDICTION_ROOT, work_root=WORK_ROOT, prepared=None):
    p = prepared or prepare_prediction_inputs_cpu()
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if (report.get("experiment") != "SW0176_local_phase_fixed320_score"
            or report.get("status") != "complete" or report.get("seed_scope") != [SEED]
            or report.get("implementation_fingerprint") != implementation_fingerprint()
            or report.get("ground_truth_used_for_prediction") is not False
            or report.get("ground_truth_used_for_scoring") is not True):
        raise ValueError("SW0176 score provenance or GT scope is invalid")
    candidate, scores, paired, gates = _score_fixed320_core(
        prediction_root=prediction_root, work_root=work_root, prepared=p)
    expected = {"prediction_sha256": candidate["prediction_sha256"],
        "prediction_protocol_sha256": candidate["protocol_sha256"],
        "reference_prediction_sha256": {key: value["prediction_sha256"] for key, value in p["references"].items()},
        "reference_protocol_sha256": {key: value["protocol_sha256"] for key, value in p["references"].items()},
        "SW0164_score_sha256": SW0164_SCORE_SHA256,
        "SW0175_score_sha256": SW0175_SCORE_SHA256,
        "matched_control_training_manifest_sha256": p["manifests"][train.ARM]["bindings"]["control_receipt"]["manifest_sha256"],
        "scores": scores, "paired_bootstrap": paired, "gate": gates}
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("SW0176 independent score recomputation differs from report")
    return report

