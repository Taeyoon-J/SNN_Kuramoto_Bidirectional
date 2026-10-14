"""Read-only row-root repair and GT-last score recovery for completed SW161 outputs.

The frozen predictor wrote rows under work/<arm>_seed1/rows, while its frozen
validator looked for work/rows. This module temporarily adapts only that path
lookup in-process, calls the original validator and scorer, and records a
separate recovery receipt. It does not modify prediction bytes or frozen code.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid

from collaborative_test.SW_0161_supervised_spike_partition import predict_evaluate as evaluation
from collaborative_test.SW_0161_supervised_spike_partition import prediction_queue, score_queue, train
from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4 as owner

try:
    import fcntl
except ImportError:  # Windows CPU fixtures never acquire the Linux server locks.
    fcntl = None

HERE = Path(__file__).resolve().parent
RECOVERY_ROOT = HERE / "results_archive" / "score_recovery_20261014"
ARCHIVE = RECOVERY_ROOT / "original_failure_snapshot"
RECOVERY_STATE = RECOVERY_ROOT / "recovery_state.json"
FROZEN_VALIDATE = evaluation.validate_prediction
ROW_ROOT_REPAIR = "map frozen validator work root to work/<registered-arm>_seed1 for row reads only"
FROZEN_EVALUATOR_SHA256 = "4364d5c9d0bcf1d9af7496eb0794a58941769f7a5f9db77dddbafb697494596f"
CAPTURED_FAILURE_RECORDS = HERE / "results_archive" / "prediction_validation_failure_20261014" / "server_records.json"
CAPTURED_FAILURE_RECORDS_SHA256 = "d0a2bcba00d065d13040ff309da6976a52a065aad03bbcfcfc16da5a1f4e9501"
DEPLOYMENT_CPU_LOGS = HERE / "results_archive" / "prediction_deployment_CPU_20261014"
EXPECTED_FAILURE_RECORDS = {
    "pair_supervised_P": {
        "source_name": "prediction_queues_20261014/pair_supervised_P/state.json",
        "sha256": "d68a7f6656b2a793cee131027a4666773d6032b58a4a422118ed5673adbabba6",
        "supervisor_pid": 3181663, "child_pid": 3205891,
    },
    "mask_supervised_P": {
        "source_name": "prediction_queues_20261014/mask_supervised_P/state.json",
        "sha256": "b811bed238b5ef854a904d74e9354c50dba8117cf4b3b142467e549e54c9a827",
        "supervisor_pid": 3181681, "child_pid": 3205747,
    },
    "score": {
        "source_name": "score_queue_20261014/state.json",
        "sha256": "66429bc7526895ffd56d87b5ef47644babf155cb363f36eb5c4bc06a1923ffab",
        "supervisor_pid": 3181695, "child_pid": None,
    },
}
PINNED_PREDICTIONS = {
    "pair_supervised_P": {
        "prediction_sha256": "93da7689640d19ec21647fe98c62362e9973e8896de6c86000624a8661f1e6e1",
        "protocol_sha256": "24905de22dc73b89f136dbd32183d132131972608b382f141c7a958ec7636b06",
    },
    "mask_supervised_P": {
        "prediction_sha256": "f0fbb25da8a6801139eaf8439b5b1b95e8e4db00e46e9f96898c741a35960eaa",
        "protocol_sha256": "f7e8a0a733d7b1f644b1b104f15c8db2bc748fa006df8ae8eb12d8bfa94fdf05",
    },
}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fingerprint():
    paths = (HERE / "recovery.py", HERE / "predict_evaluate.py", HERE / "prediction_queue.py",
             HERE / "score_queue.py", HERE / "train.py", HERE / "preflight.py")
    if any(not p.is_file() for p in paths):
        raise FileNotFoundError("SW161 recovery dependency missing")
    return {p.name: sha256_file(p) for p in paths}


def _require_frozen_evaluator():
    actual = sha256_file(HERE / "predict_evaluate.py")
    if actual != FROZEN_EVALUATOR_SHA256:
        raise RuntimeError("frozen SW161 evaluator bytes differ from the original completed prediction run")


@contextmanager
def _arm_scoped_original_validator():
    """Keep all frozen checks, changing only the row lookup root for the active arm."""
    original_load_row = evaluation._load_row
    original_validate = evaluation.validate_prediction
    active = {"arm": None}

    def scoped_load_row(work_root, index, image_id, rgb_sha, bindings_sha):
        arm = active["arm"]
        if arm not in train.ARMS:
            raise RuntimeError("SW161 recovery row lookup has no active registered arm")
        return original_load_row(Path(work_root) / f"{arm}_seed1", index, image_id, rgb_sha, bindings_sha)

    def scoped_validate(arm, **kwargs):
        if arm not in train.ARMS or active["arm"] is not None:
            raise ValueError("SW161 recovery validator accepts one registered arm at a time")
        active["arm"] = arm
        try:
            return FROZEN_VALIDATE(arm, **kwargs)
        finally:
            active["arm"] = None

    evaluation._load_row = scoped_load_row
    evaluation.validate_prediction = scoped_validate
    try:
        yield
    finally:
        evaluation._load_row = original_load_row
        evaluation.validate_prediction = original_validate


def validate_prediction(arm, *, prediction_root=evaluation.PREDICTION_ROOT,
                        work_root=evaluation.WORK_ROOT, training_root=train.TRAIN_ROOT,
                        dataset=evaluation.DATASET):
    if arm not in train.ARMS:
        raise ValueError("unregistered SW161 recovery arm")
    with _arm_scoped_original_validator():
        return evaluation.validate_prediction(arm, prediction_root=prediction_root,
            work_root=work_root, training_root=training_root, dataset=dataset)


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read original SW161 failure record {path}") from exc


def _original_failure_evidence(*, records_path=CAPTURED_FAILURE_RECORDS,
                               deployment_logs=DEPLOYMENT_CPU_LOGS):
    records_path = Path(records_path)
    if sha256_file(records_path) != CAPTURED_FAILURE_RECORDS_SHA256:
        raise RuntimeError("SW161 captured failure records do not match the authoritative snapshot")
    captured = json.loads(records_path.read_text(encoding="utf-8-sig"))
    by_name = {item.get("source_name"): item for item in captured}
    evidence = {}
    for arm in train.ARMS:
        config = prediction_queue.configure_arm(arm)
        state = _read_json(config["state"])
        expected = EXPECTED_FAILURE_RECORDS[arm]
        snapshot = by_name.get(expected["source_name"])
        log = Path(config["log"])
        supervisor_log = Path(deployment_logs) / f"{arm}_supervisor.log"
        if (snapshot is None or snapshot.get("sha256") != expected["sha256"]
                or state != snapshot.get("record")
                or state.get("experiment") != train.EXPERIMENT or state.get("arm") != arm
                or state.get("status") != "failed"
                or state.get("error") != "FileNotFoundError(2, 'No such file or directory')"
                or state.get("supervisor_pid") != expected["supervisor_pid"]
                or state.get("child_pid") != expected["child_pid"]
                or not log.is_file() or not supervisor_log.is_file()):
            raise RuntimeError(f"SW161 {arm} is not the registered row-root validation failure")
        child_log_text = log.read_text(encoding="utf-8", errors="replace")
        supervisor_log_text = supervisor_log.read_text(encoding="utf-8", errors="replace")
        if ("prediction_complete" not in child_log_text or "320" not in child_log_text
                or "FileNotFoundError" not in supervisor_log_text):
            raise RuntimeError(f"SW161 {arm} logs do not match successful child plus failed parent validation")
        evidence[arm] = {"state_path": config["state"], "log_path": log,
            "supervisor_log_path": supervisor_log,
            "state": state, "state_sha256": sha256_file(config["state"]),
            "log_sha256": sha256_file(log), "supervisor_log_sha256": sha256_file(supervisor_log)}

    score_state = _read_json(score_queue.STATE)
    expected = EXPECTED_FAILURE_RECORDS["score"]
    snapshot = by_name.get(expected["source_name"])
    score_supervisor_log = Path(deployment_logs) / "score_supervisor.log"
    if (snapshot is None or snapshot.get("sha256") != expected["sha256"]
            or score_state != snapshot.get("record")
            or score_state.get("experiment") != train.EXPERIMENT
            or score_state.get("status") != "failed"
            or score_state.get("error") != "FileNotFoundError(2, 'No such file or directory')"
            or score_state.get("supervisor_pid") != expected["supervisor_pid"]
            or Path(score_queue.LOG).exists() or not score_supervisor_log.is_file()):
        raise RuntimeError("SW161 score queue is not the registered row.json validation failure")
    score_log_text = score_supervisor_log.read_text(encoding="utf-8", errors="replace")
    if "FileNotFoundError" not in score_log_text:
        raise RuntimeError("SW161 score log does not match the known row.json defect")
    evidence["score"] = {"state_path": score_queue.STATE, "log_path": score_supervisor_log,
        "state": score_state, "state_sha256": sha256_file(score_queue.STATE),
        "log_sha256": sha256_file(score_supervisor_log)}
    return evidence


def _require_pids_dead(evidence):
    for name, item in evidence.items():
        state = item["state"]
        for key in ("supervisor_pid", "child_pid"):
            pid = state.get(key)
            if pid is None:
                continue
            if type(pid) is not int or pid <= 0:
                raise ValueError(f"invalid PID in original {name} queue state")
            try:
                proc = owner._proc_record(pid)
            except (FileNotFoundError, ProcessLookupError):
                proc = None
            except PermissionError as exc:
                raise RuntimeError(f"cannot prove original {name} PID {pid} is dead") from exc
            if proc is not None and proc.get("state") != "Z":
                raise RuntimeError(f"original {name} queue PID {pid} is still live")


@contextmanager
def _exclusive_old_queue_locks(evidence):
    if fcntl is None:
        raise RuntimeError("SW161 recovery requires Linux flock")
    paths = [prediction_queue.configure_arm(arm)["lock"] for arm in train.ARMS] + [score_queue.LOCK]
    streams = []
    try:
        for path in paths:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open("a+")
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                stream.close()
                raise RuntimeError(f"original SW161 queue lock is still owned: {path}") from exc
            streams.append(stream)
        _require_pids_dead(evidence)
        yield
    finally:
        for stream in reversed(streams):
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            finally:
                stream.close()


def _archive_failure(evidence, *, archive=ARCHIVE):
    archive = Path(archive)
    if archive.exists():
        raise FileExistsError(f"preserving existing SW161 recovery archive {archive}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    staging = archive.with_name(archive.name + f".staging-{os.getpid()}-{uuid.uuid4().hex}")
    staging.mkdir()
    snapshot = {}
    try:
        for name, item in evidence.items():
            for kind in ("state", "log", "supervisor_log"):
                if f"{kind}_path" not in item:
                    continue
                src = Path(item[f"{kind}_path"])
                dest = staging / f"{name}_{kind}{src.suffix}"
                raw = src.read_bytes()
                with dest.open("xb") as stream:
                    stream.write(raw); stream.flush(); os.fsync(stream.fileno())
                if sha256_file(dest) != item[f"{kind}_sha256"]:
                    raise RuntimeError("SW161 archived failure evidence hash changed")
                snapshot[f"{name}_{kind}"] = {"sha256": item[f"{kind}_sha256"], "size": len(raw)}
        metadata = {"experiment": train.EXPERIMENT, "captured_at": time.time(),
            "reason": "completed fixed320 predictions rejected by frozen validator's wrong row root",
            "evidence": snapshot}
        with (staging / "archive_manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(metadata, stream, indent=2, allow_nan=False); stream.write("\n")
            stream.flush(); os.fsync(stream.fileno())
        os.rename(staging, archive)
    except BaseException:
        # Keep any staged forensic bytes; never delete or overwrite them.
        raise
    return {"path": str(archive), "manifest_sha256": sha256_file(archive / "archive_manifest.json"),
            "evidence": snapshot}


def _write_create_once(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"preserving existing SW161 recovery record {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    os.rename(temp, path)


def recover_and_score(*, prediction_root=evaluation.PREDICTION_ROOT,
                      work_root=evaluation.WORK_ROOT, training_root=train.TRAIN_ROOT,
                      dataset=evaluation.DATASET, output=evaluation.SCORE_PATH,
                      archive=ARCHIVE, recovery_state=RECOVERY_STATE):
    """Validate unchanged completed predictions, run frozen GT-last scoring, publish separate receipt."""
    output, recovery_state = Path(output), Path(recovery_state)
    if output.exists() or recovery_state.exists():
        raise FileExistsError("preserving existing SW161 score/recovery result")
    if fcntl is None:
        raise RuntimeError("SW161 recovery requires Linux flock")
    evidence = _original_failure_evidence()
    with _exclusive_old_queue_locks(evidence):
        _require_frozen_evaluator()
        if (output.exists() or recovery_state.exists() or Path(archive).exists()):
            raise FileExistsError("SW161 recovery output/archive appeared before scoring")
        frozen_fp = evaluation.implementation_fingerprint()
        before = {}
        recovered_predictions = {}
        for arm in train.ARMS:
            record = validate_prediction(arm, prediction_root=prediction_root,
                work_root=work_root, training_root=training_root, dataset=dataset)
            recovered_predictions[arm] = record
            if {key: record[key] for key in ("prediction_sha256", "protocol_sha256")} != PINNED_PREDICTIONS[arm]:
                raise ValueError(f"SW161 {arm} prediction/protocol hashes differ from the completed run")
            before[arm] = {"prediction_sha256": record["prediction_sha256"],
                "protocol_sha256": record["protocol_sha256"],
                "rows_sha256": evaluation._rows_digest(Path(work_root) / f"{arm}_seed1")}
        archived = _archive_failure(evidence, archive=archive)
        with _arm_scoped_original_validator():
            report = evaluation.score_all(output=output, prediction_root=prediction_root, dataset=dataset)
            evaluation.validate_score(output, prediction_root=prediction_root, dataset=dataset)
        if evaluation.implementation_fingerprint() != frozen_fp:
            raise RuntimeError("frozen SW161 score implementation changed during recovery")
        for arm in train.ARMS:
            record = recovered_predictions[arm]
            after = {"prediction_sha256": sha256_file(record["npz_path"]),
                "protocol_sha256": sha256_file(record["protocol_path"]),
                "rows_sha256": evaluation._rows_digest(Path(work_root) / f"{arm}_seed1")}
            if after != before[arm]:
                raise RuntimeError(f"SW161 {arm} prediction artifacts changed during recovery scoring")
        receipt = {"experiment": train.EXPERIMENT, "status": "complete_recovered",
            "completed_at": time.time(), "score_path": str(output),
            "score_sha256": sha256_file(output), "score_report_status": report.get("status"),
            "registered_seed1_gate": report.get("registered_seed1_gate"),
            "ground_truth_opened_after_all_predictions_validated": report.get("ground_truth_opened_after_all_predictions_validated"),
            "ground_truth_used_for_prediction": report.get("ground_truth_used_for_prediction"),
            "ground_truth_used_for_scoring": report.get("ground_truth_used_for_scoring"),
            "original_evaluation_fingerprint": frozen_fp,
            "recovery_fingerprint": _fingerprint(), "recovery_validator": ROW_ROOT_REPAIR,
            "original_score_queue_state_preserved": True, "original_prediction_queue_states_preserved": True,
            "archived_failure_evidence": archived, "validated_predictions": before,
            "interpretation": "read-only corrected row-path validation and frozen scorer; no prediction regeneration or model change"}
        _write_create_once(recovery_state, receipt)
        return receipt


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"status": "planned", "arms": list(train.ARMS),
            "score_path": str(evaluation.SCORE_PATH), "recovery_state": str(RECOVERY_STATE),
            "no_gpu": True, "ground_truth_opened_only_after_recovery_validation": True}, allow_nan=False))
        return 0
    receipt = recover_and_score()
    print(json.dumps({"status": receipt["status"], "score_sha256": receipt["score_sha256"],
        "recovery_state": str(RECOVERY_STATE), "registered_seed1_gate": receipt["registered_seed1_gate"]},
        allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
