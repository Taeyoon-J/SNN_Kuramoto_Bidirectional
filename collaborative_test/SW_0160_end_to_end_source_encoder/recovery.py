"""Metadata-only SW0160 protocol repair and GT-last score recovery.

The prediction archives and row ledgers are never rewritten. This narrowly adds
the two redundant source-base hash mirrors omitted by the frozen producer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"),
                str(ROOT / "snn_kuramoto_bidirectional")]

import numpy as np

try:
    import fcntl
except ImportError:
    fcntl = None

_unset = object()
_cuda_visibility = os.environ.get("CUDA_VISIBLE_DEVICES", _unset)
from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4 as owner
if _cuda_visibility is _unset:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = _cuda_visibility

from collaborative_test.SW_0160_end_to_end_source_encoder import (
    predict_evaluate as evaluation, prediction_queue, score_queue, train, training_queue)

ARMS = tuple(evaluation.ARMS)
PREDICTION_ROOT = Path(evaluation.PREDICTION_ROOT)
WORK_ROOT = Path(evaluation.WORK_ROOT)
RECOVERY_ROOT = HERE / "results_archive" / "protocol_mirror_recovery_20261014"
ACTIVE_STATUSES = {"waiting_for_training_artifact", "waiting_stable_gpu",
                   "waiting_for_exclusive_gpu", "running", "launching",
                   "waiting_for_prediction_owner", "waiting_for_both_predictions"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(ch in "0123456789abcdef" for ch in value.lower()))


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    raw = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    with temp.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    try:
        fd = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def source_base_mirrors(arm, manifest):
    """Read only the independently validated training-manifest source hashes."""
    if arm not in ARMS or not isinstance(manifest, dict):
        raise ValueError("unregistered arm or malformed training manifest")
    before = manifest.get("source_base_before_sha256")
    after = manifest.get("source_base_after_sha256")
    if not _valid_sha(before) or not _valid_sha(after):
        raise ValueError("training manifest has invalid source-base hashes")
    if arm == "frozen_base_joint" and before != after:
        raise ValueError("frozen-base control training manifest changed its source base")
    if arm == "full_encoder_joint" and before == after:
        raise ValueError("full-encoder candidate training manifest did not change its source base")
    return {"source_base_before_sha256": before, "source_base_after_sha256": after}


def validate_and_migrate_protocol_fields(protocol, expected, arm, manifest):
    """Validate every frozen field, then add only the two omitted mirrors."""
    if not isinstance(protocol, dict) or not isinstance(expected, dict):
        raise ValueError("prediction protocol or expected contract is not an object")
    mirrors = source_base_mirrors(arm, manifest)
    for key, value in expected.items():
        if key in mirrors:
            continue
        if protocol.get(key) != value:
            raise ValueError(f"SW0160 legacy prediction protocol mismatch: {key}")
    present = [key in protocol for key in mirrors]
    if any(present) and not all(present):
        raise ValueError("prediction protocol has a partial source-base mirror pair")
    for key, value in mirrors.items():
        if key in protocol and protocol[key] != value:
            raise ValueError(f"existing source-base mirror does not match training manifest: {key}")
    migrated = dict(protocol)
    migrated.update(mirrors)
    return migrated


def validate_legacy_protocol_shape(protocol, expected, arm, manifest):
    """Accept the actual producer schema, omitting only the two known mirrors."""
    mirrors = source_base_mirrors(arm, manifest)
    base = (set(expected)
            | {"prediction_sha256", "model_state_sha256_before",
               "model_state_sha256_after", "readout", "actual_thresholds"})
    keys = set(protocol)
    if keys not in (base, base | set(mirrors)):
        raise ValueError("legacy prediction protocol has missing or unexpected fields")
    if protocol.get("readout") != "original SW0134 actual-spike native32 QCC from source-T T1024 rollout":
        raise ValueError("legacy prediction readout contract changed")
    thresholds = protocol.get("actual_thresholds")
    if (not isinstance(thresholds, list) or len(thresholds) != 4
            or any(not isinstance(value, (int, float)) or not np.isfinite(value)
                   or value <= 0 for value in thresholds)):
        raise ValueError("legacy actual-threshold diagnostic is malformed")
    for key in ("prediction_sha256", "model_state_sha256_before", "model_state_sha256_after"):
        if not _valid_sha(protocol.get(key)):
            raise ValueError(f"legacy prediction protocol has invalid {key}")


def validate_prediction_artifact_mirrors(protocol, *, prediction_sha256, model_state_sha256):
    """Bind producer mirrors to the already-computed archive/checkpoint hashes."""
    if protocol.get("prediction_sha256") != prediction_sha256:
        raise ValueError("prediction NPZ digest differs from frozen protocol")
    if (protocol.get("model_state_sha256_before") != model_state_sha256
            or protocol.get("model_state_sha256_after") != model_state_sha256):
        raise ValueError("prediction model-state hashes differ from validated checkpoint")


def _legacy_expected(arm, artifact, source, reference, cache, dataset, work_root=WORK_ROOT):
    """Rebuild the frozen validator's full expected contract except its two mirrors."""
    bindings = evaluation._prediction_bindings(arm, artifact, source, reference, cache, dataset)
    bindings_sha = hashlib.sha256(_json_bytes(bindings)).hexdigest()
    ids = evaluation.IMAGE_IDS
    expected = {"experiment": train.EXPERIMENT, "status": "complete", "arm": arm,
        "seed": evaluation.SEED, "image_ids": [int(ids[0]), int(ids[-1])],
        "count": evaluation.COUNT, "grid_size": [evaluation.GRID, evaluation.GRID],
        "prediction_shape": [evaluation.COUNT, evaluation.GRID, evaluation.GRID],
        "steps": train.STEPS, "settle": train.SETTLE, "live_tail": 0,
        "optimizer_updates": 0, "ground_truth_used_for_prediction": False,
        "prediction_bindings": bindings, "prediction_bindings_sha256": bindings_sha,
        "prediction_row_commits_sha256": evaluation._row_commit_digest(
            Path(work_root) / f"{arm}_seed1"),
        "implementation_fingerprint": evaluation.implementation_fingerprint(),
        "source_checkpoint_sha256": bindings["source_checkpoint_sha256"],
        "source_manifest_sha256": bindings["source_manifest_sha256"],
        "source_training_ids_sha256": bindings["source_training_ids_sha256"],
        "source_qcc_prediction_sha256": bindings["source_qcc_prediction_sha256"],
        "source_T_prediction_sha256": bindings["source_T_prediction_sha256"],
        "training_manifest_sha256": artifact["manifest_sha256"],
        "training_checkpoint_sha256": artifact["manifest"]["checkpoint_sha256"],
        "training_model_state_sha256": artifact["checkpoint_model_state_sha256"]}
    return expected, bindings, bindings_sha


def validate_legacy_prediction(arm, *, prediction_root=PREDICTION_ROOT,
                               training_root=train.TRAIN_ROOT, work_root=WORK_ROOT,
                               dataset=evaluation.DATASET):
    """Deep validate a frozen prediction before the protocol migration or GT read."""
    if arm not in ARMS:
        raise ValueError("unregistered SW0160 prediction arm")
    folder = Path(prediction_root) / f"{arm}_seed1"
    protocol_path, archive_path = folder / "protocol.json", folder / "predictions.npz"
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    artifact = evaluation._validated_training(arm, training_root)
    mirrors = source_base_mirrors(arm, artifact["manifest"])
    source, reference = evaluation._source_and_t()
    cache = np.load(evaluation.sw130.VAL_RGB, mmap_mode="r", allow_pickle=False)
    expected, bindings, bindings_sha = _legacy_expected(
        arm, artifact, source, reference, cache, dataset, work_root)
    validate_legacy_protocol_shape(protocol, expected, arm, artifact["manifest"])
    migrated = validate_and_migrate_protocol_fields(protocol, expected, arm, artifact["manifest"])
    archive_sha = sha256_file(archive_path)
    validate_prediction_artifact_mirrors(protocol, prediction_sha256=archive_sha,
        model_state_sha256=artifact["checkpoint_model_state_sha256"])
    with np.load(archive_path, allow_pickle=False) as saved:
        ids = evaluation.contract.require_fixed_image_ids(saved["image_ids"])
        labels = np.asarray(saved["qcc_labels"])
    if (not np.array_equal(ids, evaluation.IMAGE_IDS)
            or labels.shape != (evaluation.COUNT, evaluation.GRID, evaluation.GRID)
            or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0)):
        raise ValueError("prediction archive is not the registered fixed320 native32 schema")
    work = Path(work_root) / f"{arm}_seed1"
    marker = json.loads((work / "bindings.json").read_text(encoding="utf-8"))
    if marker != {"bindings": bindings, "bindings_sha256": bindings_sha}:
        raise ValueError("prediction row ledger is bound to different source/model lineage")
    actual_row_digest = evaluation._row_commit_digest(work)
    if actual_row_digest != protocol.get("prediction_row_commits_sha256"):
        raise ValueError("prediction row ledger digest changed")
    for index, image_id in enumerate(evaluation.IMAGE_IDS.tolist()):
        rgb = np.asarray(cache[index:index + 1])
        rgb_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
        row = evaluation._load_row(work, index, int(image_id), rgb_sha, bindings_sha)
        if not np.array_equal(row, labels[index]):
            raise ValueError(f"prediction row {index} differs from frozen NPZ")
    return {"arm": arm, "folder": folder, "protocol_path": protocol_path,
        "protocol_bytes": protocol_bytes, "protocol_sha256_before": hashlib.sha256(protocol_bytes).hexdigest(),
        "migrated_protocol": migrated, "prediction_path": archive_path,
        "prediction_sha256": archive_sha, "row_ledger_sha256": actual_row_digest,
        "bindings_sha256": bindings_sha, "training_manifest_sha256": artifact["manifest_sha256"],
        "training_checkpoint_sha256": artifact["manifest"]["checkpoint_sha256"],
        "training_model_state_sha256": artifact["checkpoint_model_state_sha256"],
        "source_base_mirrors": mirrors,
        "prediction_protocol_fingerprint": evaluation.implementation_fingerprint()}


def _queue_specs():
    entries = []
    for arm in ARMS:
        train_cfg = training_queue.configure_arm(arm)
        pred_cfg = prediction_queue.configure_arm(arm)
        entries.extend([
            {"kind": "training", "arm": arm, "state": train_cfg["state"],
             "lock": train_cfg["lock"], "module": "collaborative_test.SW_0160_end_to_end_source_encoder.training_queue",
             "child_module": "collaborative_test.SW_0160_end_to_end_source_encoder.train",
             "terminal_status": "complete"},
            {"kind": "prediction", "arm": arm, "state": pred_cfg["state"],
             "lock": pred_cfg["lock"], "module": "collaborative_test.SW_0160_end_to_end_source_encoder.prediction_queue",
             "child_module": "collaborative_test.SW_0160_end_to_end_source_encoder.predict_evaluate",
             "terminal_status": "failed",
             "terminal_error": "ValueError('SW0160 prediction provenance mismatch: source_base_before_sha256')"},
        ])
    entries.append({"kind": "score", "arm": None, "state": score_queue.STATE,
        "lock": score_queue.LOCK, "module": "collaborative_test.SW_0160_end_to_end_source_encoder.score_queue",
        "child_module": None, "terminal_status": "failed",
        "terminal_error": "ValueError('SW0160 prediction provenance mismatch: source_base_before_sha256')"})
    return entries


def _state_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(f"cannot read terminal SW0160 queue state {path}") from exc


def _assert_pid_dead(pid, module, arm=None, *, proc_reader=None):
    if pid is None:
        return
    if type(pid) is not int or pid <= 0:
        raise ValueError(f"malformed recorded predecessor PID: {pid!r}")
    reader = proc_reader or owner._proc_record
    row = reader(pid)
    if row is None or row.get("state") == "Z":
        return
    argv = row.get("argv", [])
    if module not in argv or (arm is not None and ("--arm" not in argv or arm not in argv)):
        raise RuntimeError(f"recorded PID {pid} is live but identity does not match predecessor")
    raise RuntimeError(f"registered SW0160 predecessor PID {pid} is still live")


def _acquire_queue_locks(specs):
    if fcntl is None:
        raise RuntimeError("SW0160 recovery requires Linux flock for safe queue handoff")
    streams = []
    try:
        for spec in specs:
            lock = Path(spec["lock"])
            if not lock.is_file():
                raise RuntimeError(f"missing old queue lock; cannot prove exclusive recovery: {lock}")
            stream = lock.open("r+")
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                stream.close()
                raise RuntimeError(f"old SW0160 queue lock is still held: {lock}") from exc
            streams.append(stream)
        return streams
    except BaseException:
        _release_locks(streams)
        raise


def _release_locks(streams):
    for stream in reversed(streams):
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()


def assert_old_queues_terminal(specs=None, *, proc_reader=None):
    states = {}
    for spec in specs or _queue_specs():
        state = _state_json(spec["state"])
        if state.get("experiment") != train.EXPERIMENT:
            raise ValueError(f"unexpected SW0160 {spec['kind']} queue experiment")
        if spec["arm"] is not None and state.get("arm") != spec["arm"]:
            raise ValueError(f"SW0160 {spec['kind']} state arm mismatch")
        if state.get("status") != spec["terminal_status"]:
            raise RuntimeError(f"SW0160 {spec['kind']} queue is not at registered terminal failure/completion")
        if spec.get("terminal_error") is not None and state.get("error") != spec["terminal_error"]:
            raise RuntimeError(f"SW0160 {spec['kind']} terminal failure is not the registered mirror mismatch")
        _assert_pid_dead(state.get("supervisor_pid"), spec["module"], spec["arm"],
                         proc_reader=proc_reader)
        _assert_pid_dead(state.get("child_pid"), spec.get("child_module"), spec["arm"],
                         proc_reader=proc_reader)
        states[str(Path(spec["state"]))] = {"bytes": Path(spec["state"]).read_bytes(), "record": state}
    return states


def _archive_bytes(path, archive_path):
    source = Path(path)
    raw = source.read_bytes()
    target = Path(archive_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def _atomic_protocol_update(path, original_bytes, migrated):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(original_bytes).digest():
        raise RuntimeError(f"prediction protocol changed after validation: {path}")
    temp = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    raw = json.dumps(migrated, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    with temp.open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)
    return hashlib.sha256(raw).hexdigest()


def recover_and_score(*, archive_root=RECOVERY_ROOT):
    """Lock dead queue handoff, migrate two mirrors, validate, then GT-last score."""
    specs = _queue_specs()
    streams = _acquire_queue_locks(specs)
    archive_root = Path(archive_root)
    try:
        old_states = assert_old_queues_terminal(specs)
        # Deep-validate both predictions before writing an archive or touching either protocol.
        validated = {arm: validate_legacy_prediction(arm) for arm in ARMS}
        if archive_root.exists():
            raise FileExistsError(f"preserving existing SW0160 recovery archive: {archive_root}")
        archive_root.mkdir(parents=True, exist_ok=False)
        records = {"schema": "sw0160_source_base_mirror_recovery_v1",
            "experiment": train.EXPERIMENT, "reason": "frozen prediction producer omitted two redundant source-base mirror fields",
            "prediction_implementation_fingerprint": evaluation.implementation_fingerprint(),
            "source_score_recomputed": False, "prediction_rerun": False,
            "arms": {}, "queue_states": {}}
        for key, saved in old_states.items():
            state_path = Path(key)
            digest = _archive_bytes(state_path, archive_root / "queue_states" / f"{state_path.parent.parent.name}_{state_path.parent.name}_state.json")
            records["queue_states"][key] = {"sha256": digest, "status": saved["record"].get("status")}
        for arm in ARMS:
            item = validated[arm]
            backup = archive_root / "protocols" / f"{arm}_protocol.original.json"
            backup_sha = _archive_bytes(item["protocol_path"], backup)
            if backup_sha != item["protocol_sha256_before"]:
                raise RuntimeError("protocol backup hash differs from prevalidation snapshot")
            records["arms"][arm] = {"protocol_path": str(item["protocol_path"]),
                "protocol_backup": str(backup), "protocol_sha256_before": backup_sha,
                "prediction_sha256": item["prediction_sha256"],
                "row_ledger_sha256": item["row_ledger_sha256"],
                "bindings_sha256": item["bindings_sha256"],
                "training_manifest_sha256": item["training_manifest_sha256"],
                "training_checkpoint_sha256": item["training_checkpoint_sha256"],
                "training_model_state_sha256": item["training_model_state_sha256"],
                **item["source_base_mirrors"]}
        _atomic_json(archive_root / "recovery_started.json", records)
        for arm in ARMS:
            item = validated[arm]
            after_sha = _atomic_protocol_update(item["protocol_path"], item["protocol_bytes"],
                                                item["migrated_protocol"])
            records["arms"][arm]["protocol_sha256_after"] = after_sha
        # The frozen original validator must pass for both arms before score_all opens GT.
        post = {arm: evaluation.validate_prediction(arm) for arm in ARMS}
        for arm in ARMS:
            if (post[arm]["prediction_sha256"] != validated[arm]["prediction_sha256"]
                    or sha256_file(post[arm]["npz_path"]) != validated[arm]["prediction_sha256"]):
                raise RuntimeError("prediction bytes changed during metadata-only migration")
        records["post_migration_original_validator"] = {arm: True for arm in ARMS}
        records["all_predictions_validated_before_gt"] = True
        _atomic_json(archive_root / "migration_validated.json", records)
        output = Path(evaluation.SCORE_PATH)
        if output.exists():
            report = evaluation.validate_score(output)
        else:
            report = evaluation.score_all(output=output)
            report = evaluation.validate_score(output)
        score_sha = sha256_file(output)
        current_score_fp = evaluation.implementation_fingerprint()
        current_queue_fp = score_queue.implementation_fingerprint()
        old_score_state = old_states[str(Path(score_queue.STATE))]["record"]
        new_state = {"experiment": train.EXPERIMENT, "status": "complete",
            "supervisor_pid": os.getpid(), "queue_fingerprint": current_queue_fp,
            "score_fingerprint": current_score_fp, "score_sha256": score_sha,
            "ground_truth_used_for_prediction": False,
            "ground_truth_used_for_scoring": True,
            "promotion_gate": report.get("registered_seed1_gate"),
            "recovery": {"kind": "metadata_only_prediction_protocol_migration",
                         "archive": str(archive_root),
                         "prior_failed_state_sha256": hashlib.sha256(old_states[str(Path(score_queue.STATE))]["bytes"]).hexdigest(),
                         "prior_status": old_score_state.get("status"),
                         "prediction_rerun": False}, "finished": time.time()}
        _atomic_json(score_queue.STATE, new_state)
        records["score"] = {"path": str(output), "sha256": score_sha,
            "validated_by_original_validator": True, "score_fingerprint": current_score_fp,
            "queue_fingerprint": current_queue_fp,
            "published_state": str(score_queue.STATE), "published_status": "complete"}
        _atomic_json(archive_root / "recovery_complete.json", records)
        return records
    finally:
        _release_locks(streams)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"status": "planned", "experiment": train.EXPERIMENT,
            "arms": list(ARMS), "prediction_rerun": False,
            "score_output": str(evaluation.SCORE_PATH),
            "archive": str(RECOVERY_ROOT)}, sort_keys=True), flush=True)
        return 0
    result = recover_and_score()
    print(json.dumps({"status": "score_recovered", "score_sha256": result["score"]["sha256"],
        "archive": str(RECOVERY_ROOT), "prediction_rerun": False,
        "both_predictions_validated_before_gt": result.get("all_predictions_validated_before_gt")},
        sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
