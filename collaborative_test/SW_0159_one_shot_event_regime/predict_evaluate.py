"""Native32 SW0159 prediction archives and GT-last fixed320 evaluation."""
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
from collaborative_test.SW_0159_one_shot_event_regime import train

ARMS = train.ARMS
SEED = train.SEED
IMAGE_IDS = np.asarray(contract.IMAGE_IDS, dtype=np.int64)
COUNT, GRID = contract.COUNT, contract.GRID
PREDICTION_ROOT = HERE / "results_archive" / "predictions_20261014"
WORK_ROOT = HERE / "results_archive" / "prediction_work_20261014"
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
    paths = [HERE / name for name in ("predict_evaluate.py", "protocol.json", "train.py")]
    paths.extend(ROOT / rel for rel in (
        "collaborative_test/SW_0130_phase_state_integration/run.py",
        "collaborative_test/SW_0135_native32_spike_binding/evaluation_contract32.py",
        "collaborative_test/SW_0136_native32_transfer/evaluate.py",
        "collaborative_test/SW_0137_native32_source_qcc/run.py",
        "collaborative_test/SW_0143_native32_selective_geodesic/run.py",
        "collaborative_test/SW_0143_native32_selective_geodesic/geodesic_operator.py",
        "collaborative_test/SW_0148_checkpointable_boundary_q/checkpoint.py",
        "collaborative_test/SW_0155_supervised_spike_relations/train.py",
        "collaborative_test/SW_0155_supervised_spike_relations/objective.py",
        "collaborative_test/SW_0152_source_preserving_partition_equivariance/training.py",
        "snn_kuramoto_bidirectional/spike_classifier.py",
        "snn_kuramoto_bidirectional/evaluation.py"))
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("SW0159 prediction dependency missing")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def _source_and_t():
    source = sw137.validate_prediction(sw137.OUTPUT_ROOT / "seed1", seed=SEED)
    reference = sw143.validate_arm_prediction(BASELINE_T, seed=SEED)
    if (not np.array_equal(source["ids"], IMAGE_IDS)
            or not np.array_equal(reference["ids"], IMAGE_IDS)
            or reference.get("arm") != "T"):
        raise ValueError("SW0159 fixed320 source/T prediction contract mismatch")
    return source, reference


def _validated_training(arm, training_root=train.TRAIN_ROOT, *, directory=None):
    if arm not in ARMS:
        raise ValueError("unregistered SW0159 arm")
    folder = Path(directory) if directory is not None else Path(training_root) / f"{arm}_seed1"
    if not train.validate_training_artifact(arm, directory=folder):
        raise ValueError(f"SW0159 {arm} training artifact failed strict validation")
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_path = folder / "checkpoint.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    flat = {**{"wrapped." + key: value for key, value in payload["wrapped"].items()},
            **{"encoder." + key: value for key, value in payload["encoder"].items()}}
    model_sha = train.source_training.state_sha(flat)
    return {"folder": folder, "manifest": manifest, "manifest_sha256": sha256_file(manifest_path),
            "checkpoint_path": checkpoint_path, "checkpoint_model_state_sha256": model_sha}


def validate_training_artifact(arm, directory=None):
    """Return a fully lineage-checked completed training record, or raise."""
    root = Path(directory).parent if directory is not None else train.TRAIN_ROOT
    folder = Path(directory) if directory is not None else root / f"{arm}_seed1"
    return _validated_training(arm, root, directory=folder)


def _load_trained(arm, device, training_root=train.TRAIN_ROOT):
    artifact = _validated_training(arm, training_root)
    foundation = train._load_foundation(device)
    payload = torch.load(artifact["checkpoint_path"], map_location=device, weights_only=False)
    if set(payload) != {"wrapped", "encoder"}:
        raise ValueError("SW0159 checkpoint schema changed")
    foundation.wrapped.load_state_dict(payload["wrapped"], strict=True)
    foundation.encoder.load_state_dict(payload["encoder"], strict=True)
    foundation.wrapped.eval().requires_grad_(False)
    foundation.encoder.eval().requires_grad_(False)
    model_sha = train.source_training.state_sha({
        **{"wrapped." + key: value for key, value in foundation.wrapped.state_dict().items()},
        **{"encoder." + key: value for key, value in foundation.encoder.state_dict().items()}})
    if model_sha != artifact["checkpoint_model_state_sha256"]:
        raise ValueError("SW0159 loaded model differs from its validated checkpoint")
    return foundation, artifact


def _prediction_bindings(arm, artifact, source, reference, cache, dataset):
    dataset = Path(dataset)
    stat = dataset.stat()
    train_bindings = artifact["manifest"]["bindings"]
    if (str(dataset.resolve()) != train_bindings["hdf5_path"]
            or stat.st_size != train_bindings["hdf5_size_bytes"]
            or stat.st_mtime_ns != train_bindings["hdf5_mtime_ns"]):
        raise ValueError("SW0159 prediction dataset differs from training HDF5 identity")
    if cache.dtype != np.uint8 or cache.shape != (COUNT, 128, 128, 3):
        raise ValueError("SW0159 fixed320 RGB cache schema changed")
    source_contract = train.sw155.source_contract()
    return {"experiment": train.EXPERIMENT, "arm": arm, "seed": SEED,
        "implementation_fingerprint": implementation_fingerprint(),
        "training_implementation_fingerprint": artifact["manifest"]["implementation_fingerprint"],
        "training_manifest_sha256": artifact["manifest_sha256"],
        "training_checkpoint_sha256": artifact["manifest"]["checkpoint_sha256"],
        "training_model_state_sha256": artifact["checkpoint_model_state_sha256"],
        "source_checkpoint_sha256": source_contract["source_sha256"],
        "source_manifest_sha256": sha256_file(source_contract["manifest_path"]),
        "source_training_ids_sha256": train.sw155.ids_sha(source_contract["image_ids"]),
        "ordered_train_ids_sha256": train.sw155.ids_sha(source_contract["image_ids"][:2048]),
        "rgb_asset_validation": train.sw130.validate_rgb_assets(),
        "source_qcc_prediction_sha256": source["prediction_sha256"],
        "source_qcc_protocol_sha256": source["protocol_sha256"],
        "source_T_prediction_sha256": reference["prediction_sha256"],
        "source_T_protocol_sha256": reference["protocol_sha256"],
        "dataset_path": str(dataset.resolve()), "dataset_size_bytes": stat.st_size,
        "dataset_mtime_ns": stat.st_mtime_ns, "validation_rgb_shape": list(cache.shape),
        "validation_rgb_dtype": str(cache.dtype), "steps": train.STEPS,
        "settle": train.SETTLE, "readout": "original SW0134 actual-spike native32 QCC"}


def _row_dir(work, index):
    return Path(work) / "rows" / f"{index:04d}"


def _write_row(work, index, image_id, rgb_sha, bindings_sha, labels):
    target = _row_dir(work, index)
    if target.exists():
        raise FileExistsError(f"preserving existing SW0159 committed row {index}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(target.name + f".staging-{os.getpid()}-{uuid.uuid4().hex}")
    staging.mkdir()
    npz = staging / "row.npz"
    labels = np.asarray(labels, dtype=np.int64)
    with npz.open("xb") as stream:
        np.savez_compressed(stream, image_index=np.asarray(index, np.int64),
            image_id=np.asarray(image_id, np.int64), qcc_labels=labels)
        stream.flush(); os.fsync(stream.fileno())
    record = {"image_index": index, "image_id": image_id, "rgb_sha256": rgb_sha,
        "bindings_sha256": bindings_sha,
        "labels_sha256": hashlib.sha256(labels.astype("<i8", copy=False).tobytes()).hexdigest(),
        "npz_sha256": sha256_file(npz)}
    with (staging / "row.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.rename(staging, target)
    return record


def _load_row(work, index, image_id, rgb_sha, bindings_sha):
    folder = _row_dir(work, index)
    record = json.loads((folder / "row.json").read_text(encoding="utf-8"))
    expected_keys = {"image_index", "image_id", "rgb_sha256", "bindings_sha256",
                     "labels_sha256", "npz_sha256"}
    npz = folder / "row.npz"
    if (set(record) != expected_keys or record.get("image_index") != index
            or record.get("image_id") != image_id or record.get("rgb_sha256") != rgb_sha
            or record.get("bindings_sha256") != bindings_sha
            or record.get("npz_sha256") != sha256_file(npz)):
        raise ValueError(f"SW0159 committed prediction row {index} is corrupt or misbound")
    with np.load(npz, allow_pickle=False) as data:
        labels = np.asarray(data["qcc_labels"])
        if (int(data["image_index"]) != index or int(data["image_id"]) != image_id
                or labels.shape != (GRID, GRID) or not np.issubdtype(labels.dtype, np.integer)
                or np.any(labels < 0)):
            raise ValueError(f"SW0159 row {index} has invalid native32 labels")
    labels = labels.astype(np.int64, copy=False)
    if record.get("labels_sha256") != hashlib.sha256(labels.astype("<i8", copy=False).tobytes()).hexdigest():
        raise ValueError(f"SW0159 row {index} label digest mismatch")
    return labels


def _row_commit_digest(work):
    commits = []
    for index in range(COUNT):
        row_path = _row_dir(work, index) / "row.json"
        if not row_path.is_file():
            raise ValueError(f"SW0159 prediction row {index} is not committed")
        commits.append(sha256_file(row_path))
    return hashlib.sha256(json.dumps(commits, separators=(",", ":")).encode()).hexdigest()


def _assemble_final(folder, labels, protocol):
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError(f"preserving existing SW0159 prediction: {folder}")
    folder.parent.mkdir(parents=True, exist_ok=True)
    staging = folder.with_name(folder.name + f".staging-{uuid.uuid4().hex}")
    staging.mkdir()
    npz = staging / "predictions.npz"
    with npz.open("xb") as stream:
        np.savez_compressed(stream, image_ids=IMAGE_IDS, qcc_labels=labels)
        stream.flush(); os.fsync(stream.fileno())
    protocol["prediction_sha256"] = sha256_file(npz)
    with (staging / "protocol.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(protocol, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    protocol_sha = sha256_file(staging / "protocol.json")
    os.rename(staging, folder)
    return {"prediction_sha256": protocol["prediction_sha256"], "protocol_sha256": protocol_sha}


@torch.no_grad()
def predict(arm, *, device="cuda:0", prediction_root=PREDICTION_ROOT,
            training_root=train.TRAIN_ROOT, work_root=WORK_ROOT, dataset=DATASET):
    if arm not in ARMS:
        raise ValueError("unregistered SW0159 arm")
    output = Path(prediction_root) / f"{arm}_seed1"
    if output.exists():
        return validate_prediction(arm, prediction_root=prediction_root,
                                   training_root=training_root, work_root=work_root, dataset=dataset)
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0159 fixed320 prediction requires assigned CUDA")
    torch.cuda.set_device(dev)
    source, reference = _source_and_t()
    foundation, artifact = _load_trained(arm, dev, training_root)
    cache = np.load(sw130.VAL_RGB, mmap_mode="r", allow_pickle=False)
    bindings = _prediction_bindings(arm, artifact, source, reference, cache, dataset)
    bindings_sha = hashlib.sha256(json.dumps(bindings, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()
    work = Path(work_root) / f"{arm}_seed1"
    work.parent.mkdir(parents=True, exist_ok=True)
    marker = work / "bindings.json"
    if work.exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved != {"bindings": bindings, "bindings_sha256": bindings_sha}:
            raise ValueError("preserving partial predictions with different SW0159 lineage")
    else:
        work.mkdir()
        with marker.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump({"bindings": bindings, "bindings_sha256": bindings_sha}, stream,
                      sort_keys=True, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    labels = np.empty((COUNT, GRID, GRID), dtype=np.int64)
    model_before = train.source_training.state_sha({
        **{"wrapped." + key: value for key, value in foundation.wrapped.state_dict().items()},
        **{"encoder." + key: value for key, value in foundation.encoder.state_dict().items()}})
    if model_before != artifact["checkpoint_model_state_sha256"]:
        raise ValueError("SW0159 model differs from exact trained checkpoint")
    for index, image_id in enumerate(IMAGE_IDS.tolist()):
        rgb_np = np.asarray(cache[index:index + 1]).copy()
        rgb_sha = hashlib.sha256(rgb_np.tobytes()).hexdigest()
        if _row_dir(work, index).exists():
            labels[index] = _load_row(work, index, int(image_id), rgb_sha, bindings_sha)
            continue
        rgb = torch.from_numpy(rgb_np).to(dev)
        gamma = foundation.encode(rgb)
        trace = foundation.rollout(gamma, total_steps=train.STEPS, live_tail_steps=0)
        for key in sw143.TRACE_FIELDS:
            value = trace.get(key)
            if not torch.is_tensor(value) or not torch.isfinite(value).all():
                raise FloatingPointError(f"SW0159 nonfinite or missing trace field {key}")
        readout = sw136._qcc_labels(trace)
        if readout.shape != (1, GRID, GRID) or not np.isfinite(readout).all() or np.any(readout < 0):
            raise ValueError(f"SW0159 invalid actual-spike QCC for row {index}")
        labels[index] = readout[0]
        _write_row(work, index, int(image_id), rgb_sha, bindings_sha, labels[index])
        if index % 32 == 31:
            print(json.dumps({"stage": "predict", "arm": arm, "predicted": index + 1,
                              "count": COUNT}), flush=True)
        del rgb, gamma, trace
    model_after = train.source_training.state_sha({
        **{"wrapped." + key: value for key, value in foundation.wrapped.state_dict().items()},
        **{"encoder." + key: value for key, value in foundation.encoder.state_dict().items()}})
    if model_after != model_before:
        raise AssertionError("SW0159 inference changed model state")
    rows_sha = _row_commit_digest(work)
    protocol = {"experiment": train.EXPERIMENT, "status": "complete", "arm": arm,
        "seed": SEED, "image_ids": [int(IMAGE_IDS[0]), int(IMAGE_IDS[-1])], "count": COUNT,
        "grid_size": [GRID, GRID], "prediction_shape": [COUNT, GRID, GRID],
        "readout": "original SW0134 actual-spike native32 QCC from source-T T1024 rollout",
        "steps": train.STEPS, "settle": train.SETTLE, "live_tail": 0,
        "optimizer_updates": 0, "ground_truth_used_for_prediction": False,
        "prediction_bindings": bindings, "prediction_bindings_sha256": bindings_sha,
        "prediction_row_commits_sha256": rows_sha,
        "implementation_fingerprint": implementation_fingerprint(),
        "source_checkpoint_sha256": bindings["source_checkpoint_sha256"],
        "source_manifest_sha256": bindings["source_manifest_sha256"],
        "source_training_ids_sha256": bindings["source_training_ids_sha256"],
        "source_qcc_prediction_sha256": bindings["source_qcc_prediction_sha256"],
        "source_T_prediction_sha256": bindings["source_T_prediction_sha256"],
        "training_manifest_sha256": artifact["manifest_sha256"],
        "training_checkpoint_sha256": artifact["manifest"]["checkpoint_sha256"],
        "training_model_state_sha256": model_before,
        "initial_thresholds": artifact["manifest"]["initial_thresholds"],
        "actual_thresholds": (0.06 * torch.exp(foundation.wrapped.b.detach())).cpu().tolist(),
        "model_state_sha256_before": model_before, "model_state_sha256_after": model_after}
    return _assemble_final(output, labels, protocol)


def validate_prediction(arm, *, prediction_root=PREDICTION_ROOT,
                        training_root=train.TRAIN_ROOT, work_root=WORK_ROOT, dataset=DATASET):
    if arm not in ARMS:
        raise ValueError("unregistered SW0159 prediction arm")
    folder = Path(prediction_root) / f"{arm}_seed1"
    protocol_path, archive_path = folder / "protocol.json", folder / "predictions.npz"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    source, reference = _source_and_t()
    artifact = _validated_training(arm, training_root)
    cache = np.load(sw130.VAL_RGB, mmap_mode="r", allow_pickle=False)
    bindings = _prediction_bindings(arm, artifact, source, reference, cache, dataset)
    expected_bind_sha = hashlib.sha256(json.dumps(bindings, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    expected = {"experiment": train.EXPERIMENT, "status": "complete", "arm": arm,
        "seed": SEED, "image_ids": [int(IMAGE_IDS[0]), int(IMAGE_IDS[-1])], "count": COUNT,
        "grid_size": [GRID, GRID], "prediction_shape": [COUNT, GRID, GRID],
        "steps": train.STEPS, "settle": train.SETTLE, "live_tail": 0,
        "optimizer_updates": 0, "ground_truth_used_for_prediction": False,
        "prediction_bindings": bindings, "prediction_bindings_sha256": expected_bind_sha,
        "prediction_row_commits_sha256": _row_commit_digest(Path(work_root) / f"{arm}_seed1"),
        "implementation_fingerprint": implementation_fingerprint(),
        "source_checkpoint_sha256": bindings["source_checkpoint_sha256"],
        "source_manifest_sha256": bindings["source_manifest_sha256"],
        "source_training_ids_sha256": bindings["source_training_ids_sha256"],
        "source_qcc_prediction_sha256": bindings["source_qcc_prediction_sha256"],
        "source_T_prediction_sha256": bindings["source_T_prediction_sha256"],
        "training_manifest_sha256": artifact["manifest_sha256"],
        "training_checkpoint_sha256": artifact["manifest"]["checkpoint_sha256"],
        "training_model_state_sha256": artifact["checkpoint_model_state_sha256"],
        "initial_thresholds": artifact["manifest"]["initial_thresholds"]}
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"SW0159 prediction provenance mismatch: {key}")
    if (protocol.get("implementation_fingerprint") != implementation_fingerprint()
            or protocol.get("prediction_sha256") != sha256_file(archive_path)
            or protocol.get("model_state_sha256_before") != artifact["checkpoint_model_state_sha256"]
            or protocol.get("model_state_sha256_after") != artifact["checkpoint_model_state_sha256"]):
        raise ValueError("SW0159 prediction implementation/checkpoint state changed")
    with np.load(archive_path, allow_pickle=False) as saved:
        ids = contract.require_fixed_image_ids(saved["image_ids"])
        labels = np.asarray(saved["qcc_labels"])
    if (not np.array_equal(ids, IMAGE_IDS) or labels.shape != (COUNT, GRID, GRID)
            or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0)):
        raise ValueError("SW0159 prediction archive violates fixed320 native32 schema")
    work = Path(work_root) / f"{arm}_seed1"
    marker = json.loads((work / "bindings.json").read_text(encoding="utf-8"))
    if marker != {"bindings": bindings, "bindings_sha256": expected_bind_sha}:
        raise ValueError("SW0159 row ledger is bound to different source/model lineage")
    for index, image_id in enumerate(IMAGE_IDS.tolist()):
        rgb = np.asarray(cache[index:index + 1])
        rgb_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
        if not np.array_equal(_load_row(work, index, int(image_id), rgb_sha, expected_bind_sha), labels[index]):
            raise ValueError(f"SW0159 final prediction differs from committed row {index}")
    return {"ids": ids, "labels": labels.astype(np.int64, copy=False), "protocol": protocol,
        "npz_path": archive_path, "protocol_path": protocol_path,
        "prediction_sha256": sha256_file(archive_path),
        "protocol_sha256": sha256_file(protocol_path)}


def _paired_bootstrap(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != (COUNT,) or right.shape != (COUNT,):
        raise ValueError("SW0159 paired bootstrap requires fixed320 per-image values")
    delta = left - right
    rng = np.random.RandomState(139)
    samples = delta[rng.randint(0, COUNT, size=(10000, COUNT))].mean(axis=1)
    return {"mean": float(delta.mean()),
            "ci95": [float(x) for x in np.percentile(samples, [2.5, 97.5])]}


def score_all(*, output=SCORE_PATH, prediction_root=PREDICTION_ROOT,
              training_root=train.TRAIN_ROOT, dataset=DATASET):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserving existing SW0159 score: {output}")
    # All model and registered reference predictions are validated before mask access.
    predictions = {arm: validate_prediction(arm, prediction_root=prediction_root,
        training_root=training_root, dataset=dataset) for arm in ARMS}
    source, reference = _source_and_t()
    frozen = {"control": predictions["control"], "candidate": predictions["candidate"],
              "source97": source, "SW0143_T": reference}
    hashes = {}
    for name, record in frozen.items():
        archive = record.get("npz_path", record.get("archive_path"))
        protocol_path = record["protocol_path"]
        hashes[f"{name}:prediction"] = sha256_file(archive)
        hashes[f"{name}:protocol"] = sha256_file(protocol_path)
        if (hashes[f"{name}:prediction"] != record["prediction_sha256"]
                or hashes[f"{name}:protocol"] != record["protocol_sha256"]):
            raise ValueError("SW0159 fixed320 predictions changed before GT access")
    dataset = Path(dataset)
    dataset_stat = dataset.stat()
    for arm, record in predictions.items():
        bindings = record["protocol"]["prediction_bindings"]
        if (bindings["dataset_path"] != str(dataset.resolve())
                or bindings["dataset_size_bytes"] != dataset_stat.st_size
                or bindings["dataset_mtime_ns"] != dataset_stat.st_mtime_ns):
            raise ValueError(f"SW0159 {arm} HDF5 identity differs before GT access")
    for name, record in frozen.items():
        archive = record.get("npz_path", record.get("archive_path"))
        if (sha256_file(archive) != hashes[f"{name}:prediction"]
                or sha256_file(record["protocol_path"]) != hashes[f"{name}:protocol"]):
            raise ValueError("SW0159 frozen output changed immediately before GT access")
    with h5py.File(dataset, "r") as h5:
        if "mask" not in h5 or h5["mask"].shape[0] <= int(IMAGE_IDS[-1]):
            raise ValueError("SW0159 score HDF5 does not contain fixed320 masks")
        target = contract.modal_native32(np.asarray(h5["mask"][int(IMAGE_IDS[0]):int(IMAGE_IDS[-1]) + 1]))
    score = {name: contract.production_metrics32(record["labels"], target)
             for name, record in frozen.items()}
    paired = {}
    for arm in ARMS:
        for baseline in ("source97", "SW0143_T", *[other for other in ARMS if other != arm]):
            for metric in contract.METRICS:
                paired[f"{arm}_minus_{baseline}_{metric}"] = _paired_bootstrap(
                    score[arm][metric]["per_image"], score[baseline][metric]["per_image"])
    means = {name: {metric: float(values[metric]["mean"]) for metric in contract.METRICS}
             for name, values in score.items()}
    report = {"experiment": train.EXPERIMENT, "status": "complete", "seed_scope": [SEED],
        "implementation_fingerprint": implementation_fingerprint(), "scores": score,
        "means": means, "paired_bootstrap": paired,
        "bootstrap": {"rng": "numpy RandomState", "seed": 139, "replicates": 10000,
                      "common_image_indices": True},
        "ground_truth_opened_after_all_predictions_validated": True,
        "ground_truth_used_for_prediction": False, "ground_truth_used_for_scoring": True,
        "prediction_hashes_before_gt": hashes,
        "dataset": {"path": str(dataset.resolve()), "size_bytes": dataset_stat.st_size,
                    "mtime_ns": dataset_stat.st_mtime_ns},
        "interpretation_limits": ["single seed1 pilot", "b jointly changes event cutoff and reset",
            "no threshold sweep or seed expansion is registered"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.rename(temp, output)
    return report


def validate_score(path=SCORE_PATH, *, prediction_root=PREDICTION_ROOT,
                   training_root=train.TRAIN_ROOT, dataset=DATASET):
    """Recompute score from independently validated frozen prediction archives."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"experiment": train.EXPERIMENT, "status": "complete", "seed_scope": [SEED],
        "ground_truth_opened_after_all_predictions_validated": True,
        "ground_truth_used_for_prediction": False, "ground_truth_used_for_scoring": True}
    if any(record.get(key) != value for key, value in required.items()):
        raise ValueError("SW0159 score has invalid GT-last contract")
    scratch = Path(path).with_name(Path(path).name + f".verify-{uuid.uuid4().hex}")
    try:
        recomputed = score_all(output=scratch, prediction_root=prediction_root,
                               training_root=training_root, dataset=dataset)
        if record != recomputed:
            raise ValueError("SW0159 stored score differs from recomputed fixed320 metrics")
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
        stage = "predict"
    else:
        result = score_all()
        stage = "score"
    print(json.dumps({"stage": stage, "status": result.get("status", "prediction_complete"),
        "arm": args.arm, "ground_truth_used_for_prediction": False,
        "ground_truth_used_for_scoring": stage == "score"}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
