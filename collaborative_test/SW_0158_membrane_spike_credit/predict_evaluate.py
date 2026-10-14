"""Fixed320 native QCC prediction and GT-last scoring for SW0158."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"),
                str(ROOT / "snn_kuramoto_bidirectional")]

import h5py
import numpy as np
import torch

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract
from collaborative_test.SW_0136_native32_transfer import evaluate as sw136
from collaborative_test.SW_0137_native32_source_qcc import run as sw137
from collaborative_test.SW_0143_native32_selective_geodesic import run as sw143
from collaborative_test.SW_0158_membrane_spike_credit import spike_credit, train

ARMS = train.ARMS
SEED = train.SEED
IMAGE_IDS = np.asarray(contract.IMAGE_IDS, dtype=np.int64)
COUNT, GRID = contract.COUNT, contract.GRID
PREDICTION_ROOT = HERE / "predictions"
PREDICTION_WORK_ROOT = HERE / "prediction_work"
SCORE_PATH = HERE / "results_archive" / "evaluation_seed1.json"
BASELINE_T = Path(sw143._arm_dir("T", SEED))
DATASET = Path(sw130.source97.DATASET)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint():
    paths = [HERE / name for name in (
        "predict_evaluate.py", "protocol.json", "train.py", "spike_credit.py")]
    paths.extend(ROOT / rel for rel in (
        "collaborative_test/SW_0130_phase_state_integration/run.py",
        "collaborative_test/SW_0135_native32_spike_binding/evaluation_contract32.py",
        "collaborative_test/SW_0136_native32_transfer/evaluate.py",
        "collaborative_test/SW_0137_native32_source_qcc/run.py",
        "collaborative_test/SW_0143_native32_selective_geodesic/run.py",
        "collaborative_test/SW_0143_native32_selective_geodesic/geodesic_operator.py",
        "collaborative_test/SW_0155_supervised_spike_relations/train.py",
        "snn_kuramoto_bidirectional/spike_classifier.py",
        "snn_kuramoto_bidirectional/evaluation.py"))
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("SW0158 prediction dependency missing")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def _source_and_t():
    source = sw137.validate_prediction(sw137.OUTPUT_ROOT / "seed1", seed=SEED)
    t = sw143.validate_arm_prediction(BASELINE_T, seed=SEED)
    if (not np.array_equal(source["ids"], IMAGE_IDS)
            or not np.array_equal(t["ids"], IMAGE_IDS)
            or t.get("arm") != "T"):
        raise ValueError("SW0158 fixed320 source/T prediction contract mismatch")
    return source, t


def _validated_training(arm, training_root=train.TRAIN_ROOT, *, directory=None):
    if arm not in ARMS:
        raise ValueError("unregistered SW0158 arm")
    folder = (Path(directory) if directory is not None
              else Path(training_root) / f"{arm}_seed1")
    mp = folder / "manifest.json"
    cp = folder / "checkpoint.pt"
    op = folder / "optimizer.pt"
    hp = folder / "history.json"
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    bindings = manifest.get("bindings", {})
    src = train.source_contract()
    assets = sw130.validate_rgb_assets()
    upstream = train.preflight.validate_upstream()
    preflight_path = train.PREFLIGHT_PATH
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    before_frozen = manifest.get("frozen_module_state_sha256_before_after")
    after_frozen = manifest.get("frozen_module_state_sha256_final")
    initial_families = manifest.get("initial_parameter_sha256")
    final_families = manifest.get("final_parameter_sha256")
    if (manifest.get("status") != "training_complete" or manifest.get("experiment") != train.EXPERIMENT
            or manifest.get("arm") != arm or manifest.get("seed") != SEED
            or manifest.get("completed_updates") != train.UPDATES
            or manifest.get("exposures") != train.BATCH * train.UPDATES
            or manifest.get("train_ids") != src["image_ids"][:2048]
            or manifest.get("ground_truth_used") is not True
            or manifest.get("ground_truth_scope") != "TRAIN masks only"
            or not isinstance(before_frozen, dict) or before_frozen != after_frozen
            or not isinstance(initial_families, dict) or not isinstance(final_families, dict)
            or set(initial_families) != {"native_core", "integration_tensors", "eta"}
            or set(final_families) != set(initial_families)
            or any(initial_families[name] == final_families[name] for name in initial_families)
            or manifest.get("implementation_fingerprint") != train.implementation_fingerprint()
            or bindings.get("implementation_fingerprint") != manifest.get("implementation_fingerprint")
            or bindings.get("arm") != arm or bindings.get("seed") != SEED
            or bindings.get("source_checkpoint_sha256") != src["source_sha256"]
            or bindings.get("source_manifest_sha256") != sha256_file(src["manifest_path"])
            or bindings.get("source_training_ids_sha256") != train.sw155.ids_sha(src["image_ids"])
            or bindings.get("ordered_train_ids_sha256") != train.sw155.ids_sha(src["image_ids"][:2048])
            or bindings.get("rgb_asset_validation") != assets
            or bindings.get("preflight_sha256") != sha256_file(preflight_path)
            or bindings.get("preflight_implementation_fingerprint") != preflight.get("implementation_fingerprint")
            or bindings.get("sw0157_calibration_sha256") != spike_credit.SW157_REPORT_SHA256
            or bindings.get("component_margin_rms") != list(spike_credit.registered_margin_rms())
            or preflight.get("training_admission") is not True
            or preflight.get("upstream") != upstream
            or not all(path.is_file() for path in (cp, op, hp))
            or manifest.get("checkpoint_sha256") != sha256_file(cp)
            or manifest.get("optimizer_sha256") != sha256_file(op)
            or manifest.get("history_sha256") != sha256_file(hp)):
        raise ValueError(f"SW0158 {arm} training lineage is invalid")
    history = json.loads(hp.read_text(encoding="utf-8"))
    expected_ids = src["image_ids"][:train.BATCH * train.UPDATES]
    if (len(history) != train.UPDATES
            or any(row.get("update") != i + 1
                   or row.get("image_ids") != expected_ids[i * train.BATCH:(i + 1) * train.BATCH]
                   or not np.isfinite(float(row["mean_loss"]))
                   or set(row.get("family_gradient_norms", {}))
                        != {"native_core", "integration_tensors", "eta"}
                   or any(not np.isfinite(float(value)) or float(value) <= 0
                          for value in row["family_gradient_norms"].values())
                   for i, row in enumerate(history))):
        raise ValueError("SW0158 completed training history is incomplete or nonfinite")
    checkpoint = torch.load(cp, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"wrapped", "encoder"}:
        raise ValueError("SW0158 completed checkpoint has an invalid schema")
    checkpoint_state = {**{"wrapped." + k: v for k, v in checkpoint["wrapped"].items()},
                        **{"encoder." + k: v for k, v in checkpoint["encoder"].items()}}
    checkpoint_model_sha = train.sw155.source_training.state_sha(checkpoint_state)
    optimizer = torch.load(op, map_location="cpu", weights_only=False)
    groups = optimizer.get("param_groups") if isinstance(optimizer, dict) else None
    expected_group_spec = (("native_core", 3e-5), ("integration_tensors", 1e-3), ("eta", 1e-3))
    if not isinstance(groups, list) or len(groups) != len(expected_group_spec):
        raise ValueError("SW0158 optimizer group layout is invalid")
    actual_steps = {}
    state = optimizer.get("state", {})
    for group, (name, lr) in zip(groups, expected_group_spec):
        if group.get("name") != name or float(group.get("lr", -1)) != lr:
            raise ValueError("SW0158 optimizer group order/rates changed")
        steps = []
        group_shapes = []
        for param_id in group.get("params", []):
            entry = state.get(param_id)
            if entry is None:
                continue
            step = entry.get("step")
            step_value = int(step.item()) if torch.is_tensor(step) else int(step)
            if not 1 <= step_value <= train.UPDATES:
                raise ValueError("SW0158 optimizer state step is outside the registered update budget")
            for key in ("exp_avg", "exp_avg_sq"):
                tensor = entry.get(key)
                if not torch.is_tensor(tensor) or not torch.isfinite(tensor).all():
                    raise ValueError(f"SW0158 optimizer moment {key} is invalid")
                if key == "exp_avg":
                    group_shapes.append(tuple(tensor.shape))
            steps.append(step_value)
        if name == "eta" and (len(steps) != 1 or group_shapes != [(4,)]):
            raise ValueError("SW0158 vector eta Adam state is incomplete")
        if not steps:
            raise ValueError(f"SW0158 optimizer group {name} has no state")
        actual_steps[name] = max(steps)
        if actual_steps[name] != train.UPDATES:
            raise ValueError(f"SW0158 optimizer group {name} did not reach final update")
    if manifest.get("optimizer_step_summary") != actual_steps:
        raise ValueError("SW0158 optimizer step summary differs from serialized Adam state")
    return {"folder": folder, "manifest": manifest, "manifest_path": mp,
            "checkpoint_path": cp, "manifest_sha256": sha256_file(mp),
            "checkpoint_model_state_sha256": checkpoint_model_sha}


def validate_training_artifact(arm, directory=None):
    """Validate a finished arm from disk without constructing a model or using GT."""
    return _validated_training(arm, directory=directory)


def _load_trained(arm, device, training_root=train.TRAIN_ROOT):
    artifact = _validated_training(arm, training_root)
    foundation = train._load_foundation(device)
    payload = torch.load(artifact["checkpoint_path"], map_location=device, weights_only=False)
    if not isinstance(payload, dict) or set(payload) != {"wrapped", "encoder"}:
        raise ValueError("SW0158 model checkpoint schema mismatch")
    foundation.wrapped.load_state_dict(payload["wrapped"], strict=True)
    foundation.encoder.load_state_dict(payload["encoder"], strict=True)
    foundation.wrapped.eval().requires_grad_(False)
    foundation.encoder.eval().requires_grad_(False)
    return foundation, artifact


def _prediction_bindings(arm, artifact, source, t, dataset, cache):
    """One writer/validator lineage record for every fixed320 row commit."""
    training_bindings = artifact["manifest"]["bindings"]
    dataset = Path(dataset)
    ds_stat = dataset.stat()
    if (str(dataset.resolve()) != training_bindings["hdf5_path"]
            or ds_stat.st_size != training_bindings["hdf5_size"]
            or ds_stat.st_mtime_ns != training_bindings["hdf5_mtime_ns"]):
        raise ValueError("SW0158 inference dataset differs from training identity")
    if cache.dtype != np.uint8 or cache.shape != (COUNT, 128, 128, 3):
        raise ValueError("registered SW0158 fixed320 RGB cache schema changed")
    src = train.source_contract()
    return {"experiment": train.EXPERIMENT, "arm": arm, "seed": SEED,
        "prediction_implementation_fingerprint": implementation_fingerprint(),
        "source_checkpoint_sha256": src["source_sha256"],
        "source_manifest_sha256": sha256_file(src["manifest_path"]),
        "source_training_ids_sha256": train.sw155.ids_sha(src["image_ids"]),
        "training_manifest_sha256": artifact["manifest_sha256"],
        "training_checkpoint_sha256": artifact["manifest"]["checkpoint_sha256"],
        "training_model_state_sha256": artifact["checkpoint_model_state_sha256"],
        "training_implementation_fingerprint": artifact["manifest"]["implementation_fingerprint"],
        "rgb_asset_validation": train.sw155.sw130.validate_rgb_assets(),
        "source_qcc_prediction_sha256": source["prediction_sha256"],
        "source_qcc_protocol_sha256": source["protocol_sha256"],
        "source_T_prediction_sha256": t["prediction_sha256"],
        "source_T_protocol_sha256": t["protocol_sha256"],
        "dataset_path": str(dataset.resolve()), "dataset_size": ds_stat.st_size,
        "dataset_mtime_ns": ds_stat.st_mtime_ns,
        "validation_rgb_shape": list(cache.shape), "validation_rgb_dtype": str(cache.dtype),
        "steps": train.STEPS, "settle": train.SETTLE,
        "readout": "native32_actual_spike_QCC"}


def _training_dataset_fields(training_bindings):
    """Keep training HDF5 identity distinct from prediction binding fields."""
    return {"training_hdf5_path": training_bindings["hdf5_path"],
            "training_hdf5_size": training_bindings["hdf5_size"],
            "training_hdf5_mtime_ns": training_bindings["hdf5_mtime_ns"]}


def _write_prediction(folder, labels, protocol):
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError(f"preserve existing SW0158 prediction output: {folder}")
    folder.parent.mkdir(parents=True, exist_ok=True)
    staging = folder.with_name(folder.name + f".{os.getpid()}.staging")
    staging.mkdir(parents=True, exist_ok=False)
    with (staging / "predictions.npz").open("xb") as stream:
        np.savez_compressed(stream, image_ids=IMAGE_IDS, qcc_labels=labels)
        stream.flush(); os.fsync(stream.fileno())
    protocol["prediction_sha256"] = sha256_file(staging / "predictions.npz")
    protocol["implementation_fingerprint"] = implementation_fingerprint()
    with (staging / "protocol.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(protocol, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    protocol_sha = sha256_file(staging / "protocol.json")
    if folder.exists():
        raise FileExistsError(f"preserve concurrently-created SW0158 prediction: {folder}")
    os.rename(staging, folder)
    return {"prediction_sha256": protocol["prediction_sha256"],
            "protocol_sha256": protocol_sha}


def _row_digest(path):
    return sha256_file(path)


def _write_prediction_row(work_dir, index, image_id, labels, rgb_sha, bindings_sha):
    row_dir = Path(work_dir) / "rows" / f"{index:04d}"
    if row_dir.exists():
        raise FileExistsError(f"preserve existing SW0158 prediction row: {row_dir}")
    row_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = row_dir.with_name(row_dir.name + f".staging-{os.getpid()}-{uuid.uuid4().hex}")
    staging.mkdir(parents=True, exist_ok=False)
    npz = staging / "row.npz"
    with npz.open("xb") as stream:
        np.savez_compressed(stream, image_index=np.asarray(index, dtype=np.int64),
            image_id=np.asarray(image_id, dtype=np.int64),
            qcc_labels=np.asarray(labels, dtype=np.int64))
        stream.flush(); os.fsync(stream.fileno())
    record = {"image_index": int(index), "image_id": int(image_id),
        "rgb_sha256": rgb_sha, "bindings_sha256": bindings_sha,
        "labels_sha256": hashlib.sha256(np.asarray(labels, dtype="<i8").tobytes()).hexdigest(),
        "npz_sha256": sha256_file(npz)}
    with (staging / "row.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    if row_dir.exists():
        raise FileExistsError(f"preserve concurrently-created SW0158 prediction row: {row_dir}")
    os.rename(staging, row_dir)
    return record


def _load_prediction_row(work_dir, index, image_id, rgb_sha, bindings_sha):
    row_dir = Path(work_dir) / "rows" / f"{index:04d}"
    record = json.loads((row_dir / "row.json").read_text(encoding="utf-8"))
    npz = row_dir / "row.npz"
    if (record != {"image_index": index, "image_id": image_id, "rgb_sha256": rgb_sha,
            "bindings_sha256": bindings_sha, "labels_sha256": record.get("labels_sha256"),
            "npz_sha256": record.get("npz_sha256")}
            or record["npz_sha256"] != sha256_file(npz)):
        raise ValueError(f"SW0158 prediction resume row {index} has invalid lineage/digest")
    with np.load(npz, allow_pickle=False) as row:
        if (int(row["image_index"]) != index or int(row["image_id"]) != image_id
                or row["qcc_labels"].shape != (GRID, GRID)
                or not np.issubdtype(row["qcc_labels"].dtype, np.integer)
                or np.any(row["qcc_labels"] < 0)):
            raise ValueError(f"SW0158 prediction resume row {index} has invalid labels")
        labels = np.asarray(row["qcc_labels"], dtype=np.int64)
    label_sha = hashlib.sha256(labels.astype("<i8", copy=False).tobytes()).hexdigest()
    if label_sha != record["labels_sha256"]:
        raise ValueError(f"SW0158 prediction resume row {index} label hash mismatch")
    return labels


def _committed_rows_digest(work_dir, cache, labels, bindings_sha):
    commits = []
    for index in range(COUNT):
        rgb_np = np.asarray(cache[index:index + 1])
        rgb_sha = hashlib.sha256(rgb_np.tobytes()).hexdigest()
        saved = _load_prediction_row(work_dir, index, int(IMAGE_IDS[index]),
                                     rgb_sha, bindings_sha)
        if not np.array_equal(saved, labels[index]):
            raise ValueError(f"SW0158 final archive differs from committed prediction row {index}")
        commits.append(sha256_file(Path(work_dir) / "rows" / f"{index:04d}" / "row.json"))
    raw = json.dumps(commits, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _validate_trace(trace, batch):
    n, t = 1024, train.STEPS
    expected = {"theta": (batch, t, n, 4), "membrane": (batch, n, t),
        "spikes": (batch, n, t), "component_membrane": (batch, 4, n, t),
        "component_spikes": (batch, 4, n, t), "component_gates": (batch, 4, n, t)}
    for name, shape in expected.items():
        value = trace.get(name)
        if not torch.is_tensor(value) or tuple(value.shape) != shape:
            raise ValueError(f"SW0158 trace field {name} must have shape {shape}")
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"nonfinite SW0158 trace field {name}")


@torch.no_grad()
def predict(arm, *, device="cuda:0", prediction_root=PREDICTION_ROOT,
            training_root=train.TRAIN_ROOT, work_root=PREDICTION_WORK_ROOT):
    if arm not in ARMS:
        raise ValueError("unregistered SW0158 prediction arm")
    output = Path(prediction_root) / f"{arm}_seed1"
    if output.exists():
        return validate_prediction(arm, prediction_root=prediction_root,
                                   training_root=training_root, work_root=work_root)
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0158 fixed320 prediction requires assigned CUDA")
    torch.cuda.set_device(dev)
    source, t = _source_and_t()
    foundation, artifact = _load_trained(arm, dev, training_root)
    training_bindings = artifact["manifest"]["bindings"]
    dataset = Path(training_bindings["hdf5_path"])
    cache = np.load(sw130.VAL_RGB, mmap_mode="r", allow_pickle=False)
    prediction_bindings = _prediction_bindings(arm, artifact, source, t, dataset, cache)
    model_before = train.sw155.source_training.state_sha({
        **{"wrapped." + k: v for k, v in foundation.wrapped.state_dict().items()},
        **{"encoder." + k: v for k, v in foundation.encoder.state_dict().items()}})
    model_after_load = train.sw155.source_training.state_sha({
        **{"wrapped." + k: v for k, v in foundation.wrapped.state_dict().items()},
        **{"encoder." + k: v for k, v in foundation.encoder.state_dict().items()}})
    if (model_before != model_after_load
            or model_before != artifact["checkpoint_model_state_sha256"]):
        raise AssertionError("SW0158 restored model differs from validated trained checkpoint")
    bindings_sha = hashlib.sha256(json.dumps(prediction_bindings, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    work_dir = Path(work_root) / f"{arm}_seed1"
    work_dir.parent.mkdir(parents=True, exist_ok=True)
    work_manifest = work_dir / "bindings.json"
    if work_dir.exists():
        if not work_manifest.is_file() or json.loads(work_manifest.read_text(encoding="utf-8")) != {
                "bindings": prediction_bindings, "bindings_sha256": bindings_sha}:
            raise ValueError("preserve SW0158 partial predictions with different lineage")
    else:
        work_dir.mkdir()
        with work_manifest.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump({"bindings": prediction_bindings, "bindings_sha256": bindings_sha}, stream,
                      sort_keys=True, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    labels = np.empty((COUNT, GRID, GRID), dtype=np.int64)
    for index in range(COUNT):
        rgb_np = np.asarray(cache[index:index + 1]).copy()
        rgb_sha = hashlib.sha256(rgb_np.tobytes()).hexdigest()
        row_dir = work_dir / "rows" / f"{index:04d}"
        if row_dir.exists():
            labels[index] = _load_prediction_row(work_dir, index, int(IMAGE_IDS[index]),
                                                  rgb_sha, bindings_sha)
        else:
            rgb = torch.from_numpy(rgb_np).to(dev)
            gamma = foundation.encode(rgb)
            trace = spike_credit.late_rollout_credit(foundation.wrapped, gamma,
                total_steps=train.STEPS, live_tail_steps=0,
                detached_control=(arm == "detached_membrane_control"))
            _validate_trace(trace, 1)
            readout = sw136._qcc_labels(trace)
            if readout.shape != (1, GRID, GRID) or not np.isfinite(readout).all():
                raise ValueError(f"invalid SW0158 actual-spike QCC at validation row {index}")
            labels[index] = readout[0]
            _write_prediction_row(work_dir, index, int(IMAGE_IDS[index]), labels[index],
                                  rgb_sha, bindings_sha)
        if index % 32 == 31:
            print(json.dumps({"stage": "predict", "arm": arm,
                              "predicted": index + 1, "count": COUNT}), flush=True)
    model_after = train.sw155.source_training.state_sha({
        **{"wrapped." + k: v for k, v in foundation.wrapped.state_dict().items()},
        **{"encoder." + k: v for k, v in foundation.encoder.state_dict().items()}})
    if model_before != model_after:
        raise AssertionError("SW0158 inference changed trained weights/buffers")
    rows_digest = _committed_rows_digest(work_dir, cache, labels, bindings_sha)
    protocol = {"experiment": train.EXPERIMENT, "status": "complete", "arm": arm,
        "seed": SEED, "detached_amplitude_gradient": arm == "detached_membrane_control",
        "image_ids": [1320, 1639], "count": COUNT, "grid_size": [GRID, GRID],
        "prediction_shape": [COUNT, GRID, GRID],
        "readout": "unchanged production native32 actual-spike QCC from T1024 trace",
        "steps": train.STEPS, "settle": train.SETTLE, "live_tail": 0,
        "ground_truth_used_for_prediction": False, "optimizer_updates": 0,
        "source_checkpoint_sha256": prediction_bindings["source_checkpoint_sha256"],
        "source_manifest_sha256": prediction_bindings["source_manifest_sha256"],
        "source_training_ids_sha256": prediction_bindings["source_training_ids_sha256"],
        "source_qcc_prediction_sha256": prediction_bindings["source_qcc_prediction_sha256"],
        "source_qcc_protocol_sha256": prediction_bindings["source_qcc_protocol_sha256"],
        "source_T_prediction_sha256": prediction_bindings["source_T_prediction_sha256"],
        "source_T_protocol_sha256": prediction_bindings["source_T_protocol_sha256"],
        "training_manifest_sha256": artifact["manifest_sha256"],
        "training_checkpoint_sha256": artifact["manifest"]["checkpoint_sha256"],
        "checkpoint_model_state_sha256": artifact["checkpoint_model_state_sha256"],
        "training_implementation_fingerprint": artifact["manifest"]["implementation_fingerprint"],
        "prediction_work_bindings_sha256": bindings_sha,
        "prediction_row_commits_sha256": rows_digest,
        **_training_dataset_fields(training_bindings),
        "model_state_sha256_before": model_before, "model_state_sha256_after": model_after}
    result = _write_prediction(output, labels, protocol)
    return result


def validate_prediction(arm, *, prediction_root=PREDICTION_ROOT, training_root=train.TRAIN_ROOT,
                        work_root=PREDICTION_WORK_ROOT):
    if arm not in ARMS:
        raise ValueError("unregistered SW0158 prediction arm")
    folder = Path(prediction_root) / f"{arm}_seed1"
    npz, pp = folder / "predictions.npz", folder / "protocol.json"
    protocol_bytes = pp.read_bytes()
    protocol = json.loads(protocol_bytes.decode("utf-8"))
    source, t = _source_and_t()
    artifact = _validated_training(arm, training_root)
    manifest = artifact["manifest"]
    training_bindings = manifest["bindings"]
    dataset = Path(training_bindings["hdf5_path"])
    cache = np.load(sw130.VAL_RGB, mmap_mode="r", allow_pickle=False)
    expected_work_bindings = _prediction_bindings(arm, artifact, source, t, dataset, cache)
    expected_work_sha = hashlib.sha256(json.dumps(expected_work_bindings, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    expected = {"experiment": train.EXPERIMENT, "status": "complete", "arm": arm,
        "seed": SEED, "detached_amplitude_gradient": arm == "detached_membrane_control",
        "image_ids": [1320, 1639], "count": COUNT, "grid_size": [GRID, GRID],
        "prediction_shape": [COUNT, GRID, GRID], "steps": train.STEPS,
        "settle": train.SETTLE, "live_tail": 0, "ground_truth_used_for_prediction": False,
        "optimizer_updates": 0, "source_checkpoint_sha256": train.source_contract()["source_sha256"],
        "source_manifest_sha256": sha256_file(train.source_contract()["manifest_path"]),
        "source_training_ids_sha256": train.sw155.ids_sha(train.source_contract()["image_ids"]),
        "source_qcc_prediction_sha256": sha256_file(source["archive_path"]),
        "source_qcc_protocol_sha256": sha256_file(source["protocol_path"]),
        "source_T_prediction_sha256": sha256_file(t["npz_path"]),
        "source_T_protocol_sha256": sha256_file(t["protocol_path"]),
        "training_manifest_sha256": artifact["manifest_sha256"],
        "training_checkpoint_sha256": manifest["checkpoint_sha256"],
        "checkpoint_model_state_sha256": artifact["checkpoint_model_state_sha256"],
        "training_implementation_fingerprint": manifest["implementation_fingerprint"],
        **_training_dataset_fields(training_bindings),
        "model_state_sha256_before": artifact["checkpoint_model_state_sha256"],
        "model_state_sha256_after": artifact["checkpoint_model_state_sha256"]}
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"SW0158 prediction provenance mismatch: {key}")
    if (protocol.get("model_state_sha256_before") != protocol.get("model_state_sha256_after")
            or protocol.get("implementation_fingerprint") != implementation_fingerprint()
            or protocol.get("prediction_sha256") != sha256_file(npz)):
        raise ValueError("SW0158 prediction fingerprint/model state is invalid")
    with np.load(npz, allow_pickle=False) as saved:
        ids = contract.require_fixed_image_ids(saved["image_ids"])
        labels = np.asarray(saved["qcc_labels"])
    if labels.shape != (COUNT, GRID, GRID) or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError("SW0158 QCC archive violates fixed native32 schema")
    if not np.array_equal(ids, IMAGE_IDS):
        raise ValueError("SW0158 prediction IDs differ from fixed320 contract")
    bindings_sha = protocol.get("prediction_work_bindings_sha256")
    rows_sha = protocol.get("prediction_row_commits_sha256")
    if (not isinstance(bindings_sha, str) or len(bindings_sha) != 64
            or not isinstance(rows_sha, str) or len(rows_sha) != 64):
        raise ValueError("SW0158 prediction row-ledger provenance is incomplete")
    work_dir = Path(work_root) / f"{arm}_seed1"
    work_record = json.loads((work_dir / "bindings.json").read_text(encoding="utf-8"))
    if not isinstance(work_record.get("bindings"), dict):
        raise ValueError("SW0158 prediction work binding record is malformed")
    computed_bindings_sha = hashlib.sha256(json.dumps(work_record["bindings"],
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    if (work_record.get("bindings_sha256") != bindings_sha
            or computed_bindings_sha != bindings_sha
            or bindings_sha != expected_work_sha
            or work_record.get("bindings") != expected_work_bindings):
        raise ValueError("SW0158 prediction work bindings differ from final artifact")
    if cache.dtype != np.uint8 or cache.shape != (COUNT, 128, 128, 3):
        raise ValueError("registered SW0158 fixed320 RGB cache schema changed")
    if _committed_rows_digest(work_dir, cache, labels, bindings_sha) != rows_sha:
        raise ValueError("SW0158 committed row ledger differs from fixed320 artifact")
    return {"ids": ids, "labels": labels.astype(np.int64, copy=False), "protocol": protocol,
            "npz_path": npz, "protocol_path": pp,
            "prediction_sha256": sha256_file(npz),
            "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest()}


def _paired_bootstrap(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != (COUNT,) or right.shape != (COUNT,):
        raise ValueError("SW0158 paired bootstrap needs fixed320 per-image metrics")
    delta = left - right
    rng = np.random.RandomState(139)
    draws = rng.randint(0, COUNT, size=(10000, COUNT))
    values = delta[draws].mean(axis=1)
    return {"mean": float(delta.mean()),
            "ci95": [float(x) for x in np.percentile(values, [2.5, 97.5])]}


def score_all(*, output=SCORE_PATH, prediction_root=PREDICTION_ROOT,
              training_root=train.TRAIN_ROOT, dataset=DATASET):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserve existing SW0158 evaluation: {output}")
    # Freeze and validate all model/reference outputs before opening any GT.
    predictions = {arm: validate_prediction(arm, prediction_root=prediction_root,
        training_root=training_root) for arm in ARMS}
    source, t = _source_and_t()
    if not np.array_equal(source["ids"], IMAGE_IDS) or not np.array_equal(t["ids"], IMAGE_IDS):
        raise ValueError("SW0158 source/T fixed320 IDs changed")
    hashes = {}
    for name, row in [*predictions.items(), ("source97", source), ("SW0143_T", t)]:
        npz = row["npz_path"] if "npz_path" in row else row["archive_path"]
        pp = row["protocol_path"]
        hashes[f"{name}:prediction"] = sha256_file(npz)
        hashes[f"{name}:protocol"] = sha256_file(pp)
        if hashes[f"{name}:prediction"] != row["prediction_sha256"] or hashes[f"{name}:protocol"] != row["protocol_sha256"]:
            raise ValueError("SW0158 prediction/reference changed before GT read")
    dataset = Path(dataset)
    stat = dataset.stat()
    for row in predictions.values():
        p = row["protocol"]
        if (p["training_hdf5_path"] != str(dataset.resolve())
                or p["training_hdf5_size"] != stat.st_size
                or p["training_hdf5_mtime_ns"] != stat.st_mtime_ns):
            raise ValueError("SW0158 scoring HDF5 differs from the trained TRAIN source")
    # Re-hash immediately before the first mask access.
    for name, row in [*predictions.items(), ("source97", source), ("SW0143_T", t)]:
        npz = row["npz_path"] if "npz_path" in row else row["archive_path"]
        if sha256_file(npz) != hashes[f"{name}:prediction"] or sha256_file(row["protocol_path"]) != hashes[f"{name}:protocol"]:
            raise ValueError("SW0158 frozen prediction changed immediately before GT access")
    with h5py.File(dataset, "r") as h5:
        if "mask" not in h5 or h5["mask"].shape[0] <= 1639:
            raise ValueError("SW0158 score dataset lacks registered fixed320 GT masks")
        target = contract.modal_native32(np.asarray(h5["mask"][1320:1640]))
    scores = {arm: contract.production_metrics32(predictions[arm]["labels"], target) for arm in ARMS}
    scores["source97"] = contract.production_metrics32(source["labels"], target)
    scores["SW0143_T"] = contract.production_metrics32(t["labels"], target)
    paired = {}
    for arm in ARMS:
        for reference in ("source97", "SW0143_T", *[a for a in ARMS if a != arm]):
            for metric in contract.METRICS:
                paired[f"{arm}_minus_{reference}_{metric}"] = _paired_bootstrap(
                    scores[arm][metric]["per_image"], scores[reference][metric]["per_image"])
    means = {name: {metric: float(value[metric]["mean"]) for metric in contract.METRICS}
             for name, value in scores.items()}
    live_fg = paired["live_membrane_credit_minus_SW0143_T_fg_ari"]
    live_detached_fg = paired["live_membrane_credit_minus_detached_membrane_control_fg_ari"]
    report = {"experiment": train.EXPERIMENT, "status": "complete", "seed_scope": [SEED],
        "implementation_fingerprint": implementation_fingerprint(),
        "scores": scores, "means": means, "paired_bootstrap": paired,
        "bootstrap": {"rng": "numpy RandomState", "seed": 139, "replicates": 10000,
                      "common_image_indices": True},
        "promotion_gate": {"live_fg_ari_at_least_0.01_above_SW0143_T":
                means["live_membrane_credit"]["fg_ari"] >= means["SW0143_T"]["fg_ari"] + .01,
            "live_fg_ari_at_least_0.01_above_detached":
                means["live_membrane_credit"]["fg_ari"] >= means["detached_membrane_control"]["fg_ari"] + .01,
            "paired_ci_lower_live_minus_T_positive": live_fg["ci95"][0] > 0,
            "paired_ci_lower_live_minus_detached_positive": live_detached_fg["ci95"][0] > 0},
        "ground_truth_opened_after_all_predictions_validated": True,
        "ground_truth_used_for_prediction": False, "ground_truth_used_for_scoring": True,
        "scientific_limits": "Single seed1 pilot; no binary event desaturation, gate necessity, or generalization claim.",
        "prediction_hashes_before_gt": hashes,
        "scoring_dataset": {"path": str(dataset.resolve()), "size": stat.st_size,
                            "mtime_ns": stat.st_mtime_ns}}
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + f".{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, output)
    return report


def validate_score(path=SCORE_PATH, *, prediction_root=PREDICTION_ROOT,
                   training_root=train.TRAIN_ROOT, dataset=DATASET):
    """Recompute the GT-last score from frozen predictions before adoption."""
    path = Path(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    required = {"experiment": train.EXPERIMENT, "status": "complete",
        "seed_scope": [SEED], "ground_truth_opened_after_all_predictions_validated": True,
        "ground_truth_used_for_prediction": False, "ground_truth_used_for_scoring": True}
    if any(record.get(key) != value for key, value in required.items()):
        raise ValueError("SW0158 existing score violates the registered score contract")
    scratch = path.with_name(path.name + f".verify-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        expected = score_all(output=scratch, prediction_root=prediction_root,
                             training_root=training_root, dataset=dataset)
        if record != expected:
            raise ValueError("SW0158 existing score differs from recomputed fixed320 metrics/gates")
    finally:
        try:
            scratch.unlink()
        except FileNotFoundError:
            pass
    return record


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("predict", "score"), required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.stage == "predict":
        if args.arm is None:
            parser.error("--arm is required for prediction")
        result = predict(args.arm, device=args.device)
    else:
        result = score_all()
    print(json.dumps({"stage": args.stage, "status": result.get("status", "prediction_complete"),
        "arm": args.arm, "ground_truth_used_for_prediction": False,
        "ground_truth_used_for_scoring": args.stage == "score"}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
