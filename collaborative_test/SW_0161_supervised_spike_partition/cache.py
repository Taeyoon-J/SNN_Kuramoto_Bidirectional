"""Create-once, resumable cache of exact frozen SW143-T source spike traces.

The public runner is fixed to the first 512 registered source TRAIN images and
requires a completed SW0160 score before it touches CUDA. Internal storage
helpers accept shorter fixtures for CPU tests; that is not a public run option.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"),
                str(ROOT / "snn_kuramoto_bidirectional")]

import h5py
import numpy as np
import torch

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0135_native32_spike_binding import foundation as sw135
from collaborative_test.SW_0143_native32_selective_geodesic import geodesic_operator, run as sw143
from collaborative_test.SW_0155_supervised_spike_relations import objective, train as sw155

EXPERIMENT = "SW0161_supervised_spike_partition_source_cache"
COUNT, BATCH, GRID, COMPONENTS, STEPS, SETTLE = 512, 4, 1024, 4, 1024, 512
SLOT_CAPACITY = 11
CACHE_ROOT = HERE / "results_archive" / "sourceT_cache_20261014"
SCORE_DEPENDENCY_EXPERIMENT = "SW0160_end_to_end_source_encoder"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value.lower()))


def _json_digest(value) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                   allow_nan=False).encode("utf-8"))


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _atomic_replace_json(path: Path, value) -> None:
    temp = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    raw = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    try:
        with temp.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        _fsync_dir(path.parent)
    finally:
        if temp.exists():
            temp.unlink()


def _atomic_create_json(path: Path, value) -> None:
    """Publish a JSON document without replacing an existing final artifact."""
    temp = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    raw = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    try:
        with temp.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
        _fsync_dir(path.parent)
    finally:
        if temp.exists():
            temp.unlink()


def _trace_row_array(value) -> np.ndarray:
    """Validate settled actual component spikes without dtype conversion."""
    if torch.is_tensor(value):
        if value.dtype != torch.float32:
            raise ValueError("source component spikes must remain native float32")
        array = value.detach().contiguous().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.dtype != np.float32:
        raise ValueError("source component spikes must remain native float32")
    if array.shape != (COMPONENTS, GRID, SETTLE):
        raise ValueError("settled trace row must be float32 [4,1024,512]")
    if not np.isfinite(array).all():
        raise FloatingPointError("settled source spikes contain nonfinite values")
    return np.ascontiguousarray(array)


def _target_row_array(value) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.shape == (1, GRID):
        array = array[0]
    if array.dtype.kind not in "iu" or array.shape != (GRID,):
        raise ValueError("TRAIN modal target row must be integer [1024]")
    if np.any(array < 0):
        raise ValueError("TRAIN modal target IDs must be nonnegative")
    if np.unique(array).size > SLOT_CAPACITY:
        raise ValueError("TRAIN modal target exceeds the common 11-slot capacity")
    return np.ascontiguousarray(array.astype(np.int64, copy=False))


def _array_sha(array: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(array).tobytes())


def implementation_fingerprint():
    """Hash the exact local cache/protocol and training/source APIs it calls."""
    relatives = (
        "collaborative_test/SW_0161_supervised_spike_partition/cache.py",
        "collaborative_test/SW_0161_supervised_spike_partition/protocol.json",
        "collaborative_test/SW_0161_supervised_spike_partition/objective.py",
        "collaborative_test/SW_0155_supervised_spike_relations/train.py",
        "collaborative_test/SW_0155_supervised_spike_relations/objective.py",
        "collaborative_test/SW_0130_phase_state_integration/run.py",
        "collaborative_test/SW_0134_native_spike_binding/rollout.py",
        "collaborative_test/SW_0135_native32_spike_binding/foundation.py",
        "collaborative_test/SW_0143_native32_selective_geodesic/run.py",
        "collaborative_test/SW_0143_native32_selective_geodesic/geodesic_operator.py",
        "collaborative_test/SW_0152_source_preserving_partition_equivariance/training.py",
        "collaborative_test/SW_0160_end_to_end_source_encoder/train.py",
        "collaborative_test/SW_0160_end_to_end_source_encoder/score_queue.py",
        "collaborative_test/SW_0160_end_to_end_source_encoder/predict_evaluate.py",
    )
    paths = [ROOT / rel for rel in relatives]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"SW0161 cache dependency missing: {missing[0]}")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


class _AtomicTraceCache:
    """Row-atomic storage primitive. Production construction fixes 512 rows."""

    def __init__(self, root, bindings, image_ids, rgb_sha256):
        self.root = Path(root)
        self.rows_dir = self.root / "rows"
        self.bindings = bindings
        self.bindings_sha256 = _json_digest(bindings)
        self.image_ids = [int(v) for v in image_ids]
        self.rgb_sha256 = list(rgb_sha256)
        if (not self.image_ids or len(set(self.image_ids)) != len(self.image_ids)
                or len(self.image_ids) != len(self.rgb_sha256)
                or any(not isinstance(value, str) or len(value) != 64
                       or any(ch not in "0123456789abcdef" for ch in value.lower())
                       for value in self.rgb_sha256)):
            raise ValueError("cache image IDs/RGB digest list is invalid")
        self.ledger_path = self.root / "ledger.json"
        self.bindings_path = self.root / "bindings.json"
        if not self.root.exists():
            self.root.mkdir(parents=True, exist_ok=False)
            self.rows_dir.mkdir()
            _atomic_create_json(self.bindings_path, {"bindings": bindings,
                                                     "bindings_sha256": self.bindings_sha256})
            _atomic_create_json(self.ledger_path, {"rows": []})
            self.records = []
        else:
            self._open_existing()

    def _open_existing(self):
        if not self.bindings_path.is_file() or not self.ledger_path.is_file() or not self.rows_dir.is_dir():
            raise ValueError("preserving incomplete SW0161 cache root")
        saved = json.loads(self.bindings_path.read_text(encoding="utf-8"))
        if saved != {"bindings": self.bindings, "bindings_sha256": self.bindings_sha256}:
            raise ValueError("existing SW0161 cache has different source/order/RGB bindings")
        ledger = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        records = ledger.get("rows")
        if not isinstance(records, list) or len(records) > len(self.image_ids):
            raise ValueError("invalid SW0161 cache ledger")
        self.records = []
        for index, record in enumerate(records):
            self._verify_row(index, record)
            self.records.append(record)

        # A crash after atomic row publication but before ledger replacement
        # leaves a complete, verifiable row. Adopt only a contiguous such row.
        while len(self.records) < len(self.image_ids):
            index = len(self.records)
            path = self._row_path(index)
            if not path.exists():
                break
            record = self._verify_row(index, None)
            self.records.append(record)
            _atomic_replace_json(self.ledger_path, {"rows": self.records})

        expected_names = {self._row_path(index).name for index in range(len(self.records))}
        for path in self.rows_dir.glob("*.npz"):
            if path.name not in expected_names:
                raise ValueError("unexpected or out-of-order SW0161 row artifact; preserving it")

    def _row_path(self, index):
        return self.rows_dir / f"{index:04d}.npz"

    def _verify_row(self, index, record):
        path = self._row_path(index)
        if not path.is_file():
            raise ValueError(f"SW0161 cache row {index} is missing")
        if record is not None:
            expected = {"index": index, "image_id": self.image_ids[index],
                        "rgb_sha256": self.rgb_sha256[index]}
            if any(record.get(key) != value for key, value in expected.items()):
                raise ValueError(f"SW0161 row {index} ID/order/RGB binding mismatch")
            if record.get("row_file_sha256") != sha256_file(path):
                raise ValueError(f"SW0161 row {index} file digest mismatch")
        try:
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {"component_spikes", "targets32", "index", "image_id",
                                           "rgb_sha256", "bindings_sha256"}:
                    raise ValueError("SW0161 row archive has unexpected fields")
                if (int(archive["index"]) != index
                        or int(archive["image_id"]) != self.image_ids[index]
                        or str(archive["rgb_sha256"].item()) != self.rgb_sha256[index]
                        or str(archive["bindings_sha256"].item()) != self.bindings_sha256):
                    raise ValueError("SW0161 row archive identity/source binding mismatch")
                components = _trace_row_array(archive["component_spikes"])
                targets = _target_row_array(archive["targets32"])
        except Exception as exc:
            raise ValueError(f"SW0161 row {index} is incomplete or invalid") from exc
        trace_sha, target_sha = _array_sha(components), _array_sha(targets)
        if record is not None and (record.get("trace_sha256") != trace_sha
                                   or record.get("targets_sha256") != target_sha):
            raise ValueError(f"SW0161 row {index} payload digest mismatch")
        return {"index": index, "image_id": self.image_ids[index],
                "rgb_sha256": self.rgb_sha256[index], "trace_sha256": trace_sha,
                "targets_sha256": target_sha, "row_file_sha256": sha256_file(path)}

    def load_row(self, index):
        if index >= len(self.records):
            raise ValueError(f"SW0161 row {index} has not been committed")
        self._verify_row(index, self.records[index])
        with np.load(self._row_path(index), allow_pickle=False) as archive:
            return archive["component_spikes"].copy(), archive["targets32"].copy()

    def write_row(self, index, image_id, rgb_sha, component_spikes, targets32):
        if index < len(self.records):
            if image_id != self.image_ids[index] or rgb_sha != self.rgb_sha256[index]:
                raise ValueError("live source row differs from already cached ID/RGB binding")
            old_trace, old_targets = self.load_row(index)
            if (not np.array_equal(old_trace, _trace_row_array(component_spikes))
                    or not np.array_equal(old_targets, _target_row_array(targets32))):
                raise ValueError("live replay differs from the committed exact source cache row")
            return self.records[index]
        if index != len(self.records):
            raise ValueError("cache rows must be committed in registered source order")
        if image_id != self.image_ids[index] or rgb_sha != self.rgb_sha256[index]:
            raise ValueError("source row ID/RGB differs from registered cache bindings")
        components = _trace_row_array(component_spikes)
        targets = _target_row_array(targets32)
        path = self._row_path(index)
        if path.exists():
            # Existing unledgered rows are adopted only through strict parsing.
            record = self._verify_row(index, None)
            old_trace, old_targets = self.load_uncommitted(path)
            if not np.array_equal(old_trace, components) or not np.array_equal(old_targets, targets):
                raise ValueError("preserving uncommitted row that differs from source replay")
        else:
            temp = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
            try:
                with temp.open("xb") as stream:
                    np.savez(stream, component_spikes=components, targets32=targets,
                             index=np.asarray(index, dtype=np.int64),
                             image_id=np.asarray(image_id, dtype=np.int64),
                             rgb_sha256=np.asarray(rgb_sha),
                             bindings_sha256=np.asarray(self.bindings_sha256))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temp, path)
                _fsync_dir(path.parent)
            finally:
                if temp.exists():
                    temp.unlink()
            record = self._verify_row(index, None)
        self.records.append(record)
        _atomic_replace_json(self.ledger_path, {"rows": self.records})
        return record

    @staticmethod
    def load_uncommitted(path):
        with np.load(path, allow_pickle=False) as archive:
            return (_trace_row_array(archive["component_spikes"]),
                    _target_row_array(archive["targets32"]))

    def finalize(self, *, bindings, source_model_state_before, source_model_state_after):
        if bindings != self.bindings:
            raise ValueError("final cache bindings changed during generation")
        if len(self.records) != len(self.image_ids):
            raise ValueError("cannot finalize incomplete source spike cache")
        if (not _is_sha256(source_model_state_before) or not _is_sha256(source_model_state_after)
                or source_model_state_before != source_model_state_after):
            raise AssertionError("frozen source model or buffers changed during cache generation")
        for index, record in enumerate(self.records):
            self._verify_row(index, record)
        manifest = {
            "experiment": EXPERIMENT,
            "status": "complete",
            "count": len(self.image_ids),
            "trace_dtype": "float32",
            "trace_shape_per_image": [COMPONENTS, GRID, SETTLE],
            "target_dtype": "int64",
            "target_shape_per_image": [GRID],
            "bindings": self.bindings,
            "bindings_sha256": self.bindings_sha256,
            "ledger_sha256": sha256_file(self.ledger_path),
            "rows_sha256": [row["row_file_sha256"] for row in self.records],
            "source_model_state_before_sha256": source_model_state_before,
            "source_model_state_after_sha256": source_model_state_after,
            "source_model_unchanged": True,
        }
        path = self.root / "manifest.json"
        _atomic_create_json(path, manifest)
        return manifest


def validate_final_cache(root=CACHE_ROOT, *, expected_bindings=None):
    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("SW0161 source cache has no complete final manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("experiment") != EXPERIMENT or manifest.get("status") != "complete"
            or manifest.get("count") != COUNT or manifest.get("source_model_unchanged") is not True
            or not _is_sha256(manifest.get("source_model_state_before_sha256"))
            or not _is_sha256(manifest.get("source_model_state_after_sha256"))
            or manifest.get("source_model_state_before_sha256") != manifest.get("source_model_state_after_sha256")
            or manifest.get("trace_dtype") != "float32"
            or manifest.get("trace_shape_per_image") != [COMPONENTS, GRID, SETTLE]
            or manifest.get("target_dtype") != "int64"
            or manifest.get("target_shape_per_image") != [GRID]
            or manifest.get("bindings_sha256") != _json_digest(manifest.get("bindings"))):
        raise ValueError("SW0161 final source cache manifest is invalid")
    if expected_bindings is not None and manifest.get("bindings") != expected_bindings:
        raise ValueError("SW0161 final cache lineage differs from current source")
    ledger_path = root / "ledger.json"
    if not ledger_path.is_file() or manifest.get("ledger_sha256") != sha256_file(ledger_path):
        raise ValueError("SW0161 final cache ledger digest mismatch")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ids = manifest.get("bindings", {}).get("image_ids")
    rgb_hashes = manifest.get("bindings", {}).get("image_rgb_sha256")
    if not isinstance(ids, list) or len(ids) != COUNT or not isinstance(rgb_hashes, list) or len(rgb_hashes) != COUNT:
        raise ValueError("SW0161 final cache ordered TRAIN IDs/RGB hashes are invalid")
    store = _AtomicTraceCache(root, manifest["bindings"], ids, rgb_hashes)
    if len(store.records) != COUNT or [r["row_file_sha256"] for r in store.records] != manifest.get("rows_sha256"):
        raise ValueError("SW0161 final cache row ledger differs from manifest")
    return manifest


def _state_digest(foundation):
    from collaborative_test.SW_0152_source_preserving_partition_equivariance import training as source_training
    values = {}
    for prefix, module in (("wrapped", foundation.wrapped), ("encoder", foundation.encoder),
                           ("patcher", foundation.patcher)):
        values.update({f"{prefix}.parameter.{name}": tensor for name, tensor in module.named_parameters()})
        values.update({f"{prefix}.buffer.{name}": tensor for name, tensor in module.named_buffers()})
    values.update({"feature_mean": foundation.feature_mean, "feature_std": foundation.feature_std,
                   "feature_clip": torch.tensor(float(foundation.feature_clip), dtype=torch.float64)})
    return source_training.state_sha(values)


def _score_dependency_from_records(state, report, score_sha, queue_fp, score_fp,
                                   *, score_owner_live=False):
    """Strict SW0160 (not SW0159) final-score state/report binding."""
    from collaborative_test.SW_0160_end_to_end_source_encoder import predict_evaluate as evaluation
    expected_score_names = {"source97", "SW0143_T", *evaluation.ARMS}
    means = report.get("means")
    metrics = tuple(evaluation.contract.METRICS)
    if (not isinstance(means, dict) or set(means) != expected_score_names
            or any(not isinstance(means[name], dict) or set(means[name]) != set(metrics)
                   or any(type(means[name][metric]) not in (int, float)
                          or not np.isfinite(means[name][metric])
                          for metric in metrics)
                   for name in expected_score_names)):
        raise ValueError("SW0160 score report has invalid registered means schema")
    if state.get("experiment") != SCORE_DEPENDENCY_EXPERIMENT:
        raise ValueError("SW0161 requires the registered SW0160 score queue")
    if state.get("status") != "complete":
        if (state.get("status") in {"waiting_for_both_predictions", "waiting_for_prediction_owner", "running"}
                and score_owner_live):
            raise RuntimeError("SW0160 score is published or pending while its registered owner is live")
        raise ValueError("SW0160 score queue is not terminal-complete")
    if (report.get("experiment") != SCORE_DEPENDENCY_EXPERIMENT
            or report.get("status") != "complete"
            or report.get("seed_scope") != [1]
            or report.get("ground_truth_opened_after_all_predictions_validated") is not True
            or report.get("ground_truth_used_for_prediction") is not False
            or report.get("ground_truth_used_for_scoring") is not True
            or state.get("score_sha256") != score_sha
            or state.get("queue_fingerprint") != queue_fp
            or state.get("score_fingerprint") != score_fp
            or report.get("implementation_fingerprint") != score_fp):
        raise ValueError("SW0160 completed score failed its current fingerprint/GT-last binding")
    return {"experiment": SCORE_DEPENDENCY_EXPERIMENT, "status": "complete",
            "score_sha256": score_sha, "queue_fingerprint": queue_fp,
            "score_fingerprint": score_fp,
            "ground_truth_used_for_prediction": False,
            "ground_truth_used_for_scoring": True}


def _score_queue_owner_live(pid):
    if type(pid) is not int or pid <= 0:
        return False
    process = Path(f"/proc/{pid}")
    try:
        stat = (process / "stat").read_text(encoding="ascii")
        status = stat[stat.rfind(")") + 2:].split()[0]
        argv = [part.decode("utf-8", errors="replace") for part in
                (process / "cmdline").read_bytes().split(b"\0") if part]
    except (OSError, ValueError):
        return False
    return (status != "Z" and
            "collaborative_test.SW_0160_end_to_end_source_encoder.score_queue" in argv)


def _current_score_dependency():
    # SW0160.train._score_dependency() protects the earlier SW0159 prerequisite;
    # this cache instead requires SW0160's own final, independently validated score.
    from collaborative_test.SW_0160_end_to_end_source_encoder import (
        predict_evaluate as evaluation, score_queue)
    state_path, score_path = score_queue.STATE, evaluation.SCORE_PATH
    if not state_path.is_file():
        raise ValueError("SW0161 cache requires a terminal SW0160 score queue state")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("experiment") != SCORE_DEPENDENCY_EXPERIMENT:
        raise ValueError("unexpected SW0160 score queue state")
    queue_fp = score_queue.implementation_fingerprint()
    score_fp = evaluation.implementation_fingerprint()
    if state.get("status") != "complete":
        if (state.get("status") in {"waiting_for_both_predictions", "waiting_for_prediction_owner", "running"}
                and _score_queue_owner_live(state.get("supervisor_pid"))):
            raise RuntimeError("SW0160 final score queue is still running; cache waits without GPU use")
        # Publication can win the race between the first state read and the
        # owner liveness probe. Re-read once before classifying a dead owner as
        # an orphan, then validate the newly published immutable receipt.
        if state.get("status") in {"waiting_for_both_predictions", "waiting_for_prediction_owner", "running"}:
            latest = json.loads(state_path.read_text(encoding="utf-8"))
            if latest.get("status") == "complete":
                state = latest
            elif _score_queue_owner_live(latest.get("supervisor_pid")):
                raise RuntimeError("SW0160 final score queue is still running; cache waits without GPU use")
            else:
                raise ValueError("SW0160 final score queue is failed, orphaned, or incomplete")
        else:
            raise ValueError("SW0160 final score queue is failed, orphaned, or incomplete")
    if not score_path.is_file():
        raise ValueError("SW0160 score queue is complete but its score artifact is missing")
    # The score queue publishes "complete" only after its expensive independent
    # validate_score recomputation. Recheck the immutable receipt/report bindings
    # here without rerunning the 320-image bootstrap on every cache validation.
    report = json.loads(score_path.read_text(encoding="utf-8"))
    return _score_dependency_from_records(state, report, sha256_file(score_path), queue_fp, score_fp)


def _source_bindings(src, rgb_rows, rgb_asset_validation, score_dependency):
    ids = [int(value) for value in src["image_ids"][:COUNT]]
    rows = np.asarray(src["rows"][:COUNT], dtype=np.int64)
    if len(ids) != COUNT or len(set(ids)) != COUNT or rows.shape != (COUNT,) or len(np.unique(rows)) != COUNT:
        raise ValueError("SW0161 requires first512 unique ordered source TRAIN IDs/rows")
    rgb_hashes = [sha256_bytes(np.ascontiguousarray(rgb).tobytes()) for rgb in rgb_rows]
    source_manifest_sha = sha256_file(src["manifest_path"])
    return {
        "experiment": EXPERIMENT,
        "source_seed": 1,
        "image_ids": ids,
        "source_rows": rows.tolist(),
        "ordered_ids_sha256": sw155.ids_sha(ids),
        "source_checkpoint_sha256": src["source_sha256"],
        "source_manifest_sha256": source_manifest_sha,
        "source_train_ids_sha256": sw155.ids_sha(src["image_ids"]),
        "source_rgb_asset_validation": rgb_asset_validation,
        "train_rgb_cache_size": Path(sw130.TRAIN_RGB).stat().st_size,
        "train_rgb_cache_mtime_ns": Path(sw130.TRAIN_RGB).stat().st_mtime_ns,
        "image_rgb_sha256": rgb_hashes,
        "hdf5_path": str(Path(sw155.HDF5_PATH).resolve()),
        "hdf5_size": Path(sw155.HDF5_PATH).stat().st_size,
        "hdf5_mtime_ns": Path(sw155.HDF5_PATH).stat().st_mtime_ns,
        "score_dependency": score_dependency,
        "source_implementation_fingerprint": sw135.implementation_fingerprint(),
        "cache_implementation_fingerprint": implementation_fingerprint(),
        "operator": "SW0143-T",
        "operator_files_sha256": {
            "run.py": sha256_file(ROOT / "collaborative_test/SW_0143_native32_selective_geodesic/run.py"),
            "geodesic_operator.py": sha256_file(ROOT / "collaborative_test/SW_0143_native32_selective_geodesic/geodesic_operator.py"),
        },
    }, rgb_hashes


def _require_frozen_source(foundation, src):
    if (foundation.provenance.get("source_core_sha256") != src["source_sha256"]
            or foundation.provenance.get("source_manifest_sha256") != sha256_file(src["manifest_path"])
            or list(map(int, foundation.image_ids)) != list(map(int, src["image_ids"]))
            or not np.array_equal(foundation.pool_indices, np.asarray(src["rows"], dtype=np.int64))):
        raise ValueError("fresh SW0161 source97 foundation differs from registered source contract")


def prepare_cache_cpu():
    """Validate and materialize the fixed512 source TRAIN inputs before CUDA use."""
    score_dependency = _current_score_dependency()
    src = sw155.source_contract()
    rgb_assets = sw130.validate_rgb_assets()
    rgb_memmap = np.load(sw130.TRAIN_RGB, mmap_mode="r", allow_pickle=False)
    if rgb_memmap.dtype != np.uint8 or tuple(rgb_memmap.shape) != (70000, 128, 128, 3):
        raise ValueError("registered source TRAIN RGB cache schema changed")
    rows = np.asarray(src["rows"][:COUNT], dtype=np.int64)
    rgb_rows = np.asarray(rgb_memmap[rows]).copy()
    if rgb_rows.dtype != np.uint8 or rgb_rows.shape != (COUNT, 128, 128, 3):
        raise ValueError("SW0161 ordered source RGB batch schema changed")
    ids = [int(value) for value in src["image_ids"][:COUNT]]
    aligned_rgb, masks = sw155.load_aligned_train_rows(
        sw155.HDF5_PATH, ids, rgb_rows, 0, COUNT)
    if not np.array_equal(aligned_rgb.numpy(), rgb_rows):
        raise ValueError("SW0161 canonical TRAIN RGB does not match source cache rows")
    targets = objective.modal_train_targets32(masks).cpu().numpy()
    if targets.shape != (COUNT, GRID) or targets.dtype != np.int64:
        raise ValueError("SW0161 TRAIN mask modal32 target schema changed")
    for row in targets:
        _target_row_array(row)
    bindings, rgb_hashes = _source_bindings(src, rgb_rows, rgb_assets, score_dependency)
    return {"score_dependency": score_dependency, "source_contract": src,
            "rgb_asset_validation": rgb_assets, "rgb_rows": rgb_rows,
            "image_ids": ids, "targets32": targets,
            "bindings": bindings, "rgb_sha256": rgb_hashes}


def run_cache(*, device="cuda:0", output=CACHE_ROOT):
    """Build or resume the fixed512 exact float32 source-T trace cache."""
    output = Path(output)
    if (output / "manifest.json").is_file():
        expected_score = _current_score_dependency()
        src = sw155.source_contract()
        saved_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        rgb_assets = saved_manifest.get("bindings", {}).get("source_rgb_asset_validation")
        if not isinstance(rgb_assets, dict):
            raise ValueError("existing SW0161 cache lacks its registered RGB asset receipt")
        rgb_meta = sw130.TRAIN_RGB.with_suffix(sw130.TRAIN_RGB.suffix + ".complete.json")
        if (not sw130.TRAIN_RGB.is_file() or not rgb_meta.is_file()
                or sha256_file(rgb_meta) != rgb_assets.get("train_manifest_sha256")
                or sw130.TRAIN_RGB.stat().st_size != saved_manifest["bindings"].get("train_rgb_cache_size")
                or sw130.TRAIN_RGB.stat().st_mtime_ns != saved_manifest["bindings"].get("train_rgb_cache_mtime_ns")
                or sha256_file(sw130.ENCODER_SOURCE) != rgb_assets.get("encoder_source_sha256")
                or sha256_file(sw130.FEATURE_STATS) != rgb_assets.get("feature_preprocessing_sha256")):
            raise ValueError("registered source TRAIN RGB asset changed since cache completion")
        rgb_memmap = np.load(sw130.TRAIN_RGB, mmap_mode="r", allow_pickle=False)
        rows = np.asarray(src["rows"][:COUNT], dtype=np.int64)
        rgb_rows = np.asarray(rgb_memmap[rows]).copy()
        bindings, _ = _source_bindings(src, rgb_rows, rgb_assets, expected_score)
        manifest = validate_final_cache(output, expected_bindings=bindings)
        return manifest

    # Complete upstream score, source/RGB alignment, and all512 target checks
    # before any CUDA initialization or cache-row publication.
    prepared = prepare_cache_cpu()
    src = prepared["source_contract"]
    rgb_rows = prepared["rgb_rows"]
    ids = prepared["image_ids"]
    targets = prepared["targets32"]
    bindings = prepared["bindings"]
    rgb_hashes = prepared["rgb_sha256"]
    if output.exists():
        store = _AtomicTraceCache(output, bindings, ids, rgb_hashes)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        store = _AtomicTraceCache(output, bindings, ids, rgb_hashes)

    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0161 source trace cache requires its assigned CUDA device")
    torch.cuda.set_device(dev)
    foundation = sw155._load_foundation(dev)
    _require_frozen_source(foundation, src)
    foundation.encoder.eval(); foundation.patcher.eval(); foundation.wrapped.eval()
    foundation.encoder.requires_grad_(False)
    foundation.patcher.requires_grad_(False)
    foundation.wrapped.requires_grad_(False)
    geodesic_operator.install_operator(foundation.wrapped.core.graph_generator, "T")
    source_state_before = _state_digest(foundation)

    def rollout_batch(start):
        rgb = torch.from_numpy(rgb_rows[start:start+BATCH].copy()).to(dev)
        with torch.no_grad():
            gamma = foundation.encode(rgb)
            trace = foundation.rollout(gamma, total_steps=STEPS, live_tail_steps=64)
        fields = tuple(sw143.TRACE_FIELDS)
        for key in fields:
            value = trace.get(key)
            if not torch.is_tensor(value) or not torch.isfinite(value).all():
                raise FloatingPointError(f"SW0161 source trace missing/nonfinite {key}")
        components = trace.get("component_spikes")
        if (not torch.is_tensor(components) or components.dtype != torch.float32
                or tuple(components.shape) != (BATCH, COMPONENTS, GRID, STEPS)):
            raise ValueError("source T rollout component_spikes schema changed")
        return components[..., SETTLE:].detach().contiguous(), trace

    for start in range(0, COUNT, BATCH):
        # Preserve the registered B4 grouping. Complete later batches are
        # reused on resume; a partially committed batch is recomputed whole.
        if start != 0 and len(store.records) >= start + BATCH:
            continue
        rng_cpu_before = torch.get_rng_state()
        rng_cuda_before = torch.cuda.get_rng_state(dev)
        settled, trace = rollout_batch(start)
        for offset in range(BATCH):
            store.write_row(start + offset, ids[start + offset], rgb_hashes[start + offset],
                            settled[offset], targets[start + offset])
        if start == 0:
            # Replay the first actual B4 against saved float32 rows exactly.
            rng_cpu_after = torch.get_rng_state()
            rng_cuda_after = torch.cuda.get_rng_state(dev)
            try:
                torch.set_rng_state(rng_cpu_before)
                torch.cuda.set_rng_state(rng_cuda_before, dev)
                replay, replay_trace = rollout_batch(0)
                cached = np.stack([store.load_row(i)[0] for i in range(BATCH)])
                if not np.array_equal(replay.cpu().numpy(), cached):
                    raise AssertionError("first cached B4 does not exactly replay source-T float32 spikes")
                for key in sw143.TRACE_FIELDS:
                    if not torch.equal(trace[key], replay_trace[key]):
                        raise AssertionError(f"first cached B4 full trace replay differs for {key}")
                del replay, replay_trace, cached
            finally:
                torch.set_rng_state(rng_cpu_after)
                torch.cuda.set_rng_state(rng_cuda_after, dev)
        del settled, trace
        if start % 32 == 0:
            print(json.dumps({"stage": "source_cache", "cached_rows": min(start + BATCH, COUNT),
                              "count": COUNT}), flush=True)

    source_state_after = _state_digest(foundation)
    manifest = store.finalize(bindings=bindings, source_model_state_before=source_state_before,
                              source_model_state_after=source_state_after)
    validate_final_cache(output, expected_bindings=bindings)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=CACHE_ROOT)
    args = parser.parse_args(argv)
    # Parent/CPU entrypoint hides CUDA before importing torch in deployed use.
    result = run_cache(device=args.device, output=args.output)
    print(json.dumps({"status": result["status"], "count": result["count"],
                      "manifest_sha256": sha256_file(Path(args.output) / "manifest.json")},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
