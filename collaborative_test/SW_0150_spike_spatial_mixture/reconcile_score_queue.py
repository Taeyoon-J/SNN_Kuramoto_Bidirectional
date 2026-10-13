"""Guarded, score-only reconciliation for the known SW0150 baseline-schema failure.

This utility does not launch training, prediction, or scoring. It verifies the
already-written canonical recovery score, archives the failed queue evidence
byte-for-byte, then records that score as recovered in the existing queue state.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from collaborative_test.SW_0150_spike_spatial_mixture import evaluate, score_recovery, stage_queue


EXPECTED_STAGE_QUEUE_SHA256 = "69a21766e76a11dbcd3b31b1093a515d14b0b49d54ceb47e7ee89de2f5131465"
EXPECTED_EVALUATE_SHA256 = score_recovery.EXPECTED_EVALUATE_SHA256
EXPECTED_BASELINE_SHA256 = evaluate.SLOT_BASELINE_SHA256
EXPECTED_FAILURE_TEXT = "ValueError: registered Slot32 baseline contract/schema changed"
DEFAULT_RECEIPT_DIR = stage_queue.RUNTIME / "score_recovery_receipt_20261013"

try:
    import fcntl
except ImportError:  # The actual queue reconciler requires POSIX flock.
    fcntl = None


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_bytes(path: Path):
    raw = Path(path).read_bytes()
    return raw, json.loads(raw.decode("utf-8-sig"))


def _recorded_pids(state: dict) -> set[int]:
    result = set()
    supervisor = state.get("supervisor_pid")
    if supervisor is not None:
        if isinstance(supervisor, bool) or not isinstance(supervisor, int) or supervisor <= 0:
            raise ValueError("old queue supervisor PID is malformed")
        result.add(supervisor)
    tasks = state.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError("old queue task state is malformed")
    for task_row in tasks.values():
        pid = task_row.get("child_pid") if isinstance(task_row, dict) else None
        if pid is not None:
            if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
                raise ValueError("old queue child PID is malformed")
            result.add(pid)
    return result


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def _exclusive_queue_lock(path: Path):
    if fcntl is None:
        raise RuntimeError("score-state reconciliation requires POSIX flock")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"preserve missing original queue lock: {path}")
    with path.open("r+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("original SW0150 queue lock is currently held") from exc
        try:
            yield stream
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _validate_score_report(score_task: dict, report_path: Path, report: dict) -> None:
    if Path(score_task.get("output", "")).resolve() != Path(report_path).resolve():
        raise ValueError("recovery score is not at the canonical queue score path")
    if not stage_queue.valid_result(score_task):
        raise ValueError("frozen queue validator rejects the recovered canonical score")
    provenance = report.get("score_recovery_provenance")
    if (not isinstance(provenance, dict)
            or provenance.get("sealed_original_evaluate_sha256") != EXPECTED_EVALUATE_SHA256
            or provenance.get("baseline_contract_sha256") != EXPECTED_BASELINE_SHA256
            or provenance.get("recovery_module_sha256")
            != sha256_file(Path(score_recovery.__file__))):
        raise ValueError("score recovery provenance does not match frozen scorer/baseline bytes")
    refs = report.get("reference_provenance")
    if (not isinstance(refs, dict)
            or refs.get("slot32_three_seed_baseline_sha256") != EXPECTED_BASELINE_SHA256):
        raise ValueError("recovered score does not bind the registered Slot32 baseline")
    prediction_rows = report.get("prediction_provenance")
    if (not isinstance(prediction_rows, dict)
            or set(prediction_rows) != {*stage_queue.train.ARMS, "source_p"}):
        raise ValueError("recovered score has incomplete prediction provenance")
    for row in prediction_rows.values():
        if (not isinstance(row, dict)
                or any(not isinstance(row.get(key), str) or len(row[key]) != 64
                       for key in ("prediction_sha256", "protocol_sha256"))):
            raise ValueError("recovered score contains malformed prediction hashes")
    scores = report.get("scores")
    expected_scores = {"encoder_graph", "joint", "source_p", "source97_qcc", "slot32_seed1"}
    if not isinstance(scores, dict) or set(scores) != expected_scores:
        raise ValueError("recovered score metric table is incomplete")
    for readouts in scores.values():
        if not isinstance(readouts, dict) or not readouts:
            raise ValueError("recovered score readout table is malformed")
        for metrics in readouts.values():
            if not isinstance(metrics, dict) or set(metrics) != set(evaluate.contract.METRICS):
                raise ValueError("recovered score metric schema changed")
            for metric in metrics.values():
                if not isinstance(metric, dict):
                    raise ValueError("recovered score metric entry is malformed")
                values = metric.get("per_image")
                mean = metric.get("mean")
                if (metric.get("valid_count") != 320 or not isinstance(values, list)
                        or len(values) != 320 or isinstance(mean, bool)
                        or not isinstance(mean, (int, float)) or not math.isfinite(mean)
                        or not all(not isinstance(value, bool)
                                   and isinstance(value, (int, float))
                                   and math.isfinite(value) for value in values)):
                    raise ValueError("recovered score contains incomplete/nonfinite 320-image metrics")


def _known_failure(state: dict, state_path: Path):
    if state.get("status") != "score_failed":
        raise ValueError("queue is not the known terminal score_failed state")
    expected_fp = stage_queue.implementation_fingerprint()
    expected_sha = stage_queue.implementation_sha256()
    if (sha256_file(stage_queue.HERE / "stage_queue.py") != EXPECTED_STAGE_QUEUE_SHA256
            or sha256_file(stage_queue.HERE / "evaluate.py") != EXPECTED_EVALUATE_SHA256
            or sha256_file(evaluate.SLOT_BASELINE) != EXPECTED_BASELINE_SHA256
            or state.get("implementation_fingerprint") != expected_fp
            or state.get("implementation_sha256") != expected_sha
            or state.get("owner_dispatcher_sha256") != stage_queue.OWNER_GUARD_SHA256):
        raise ValueError("original SW0150 queue/scorer/baseline fingerprint changed")
    task_plan = stage_queue.task_plan()
    state_tasks = state.get("tasks")
    if not isinstance(state_tasks, dict) or set(state_tasks) != {
            row["task_id"] for row in task_plan}:
        raise ValueError("old queue does not contain the exact registered task plan")
    score_row = state_tasks.get("score")
    if (not isinstance(score_row, dict) or score_row.get("status") != "failed"
            or score_row.get("artifact_valid") is not False
            or not isinstance(score_row.get("returncode"), int)
            or score_row["returncode"] == 0):
        raise ValueError("score task is not the recorded failed score attempt")
    score_task = score_row.get("task")
    expected_score_task = next(row for row in task_plan if row["task_id"] == "score")
    if score_task != expected_score_task:
        raise ValueError("score task definition differs from the frozen registered task")
    log_value = score_row.get("log")
    if not isinstance(log_value, str) or not Path(log_value).is_file():
        raise ValueError("known failed score log is missing")
    log_bytes = Path(log_value).read_bytes()
    log_text = log_bytes.decode("utf-8", errors="replace")
    if EXPECTED_FAILURE_TEXT not in log_text:
        raise ValueError("old score log does not contain the exact registered-schema failure")
    for task in task_plan:
        task_id = task["task_id"]
        row = state_tasks[task_id]
        if task_id == "score":
            continue
        if (row.get("task") != task
                or row.get("status") not in {"passed", "verified_existing"}
                or not stage_queue.valid_result(task)):
            raise ValueError(f"upstream artifact {task_id} is not passed and valid")
    score_path = Path(expected_score_task["output"])
    if not score_path.is_file():
        raise FileNotFoundError("canonical recovered score report is not present")
    report_bytes, report = _read_json_bytes(score_path)
    _validate_score_report(expected_score_task, score_path, report)
    return {"score_task": score_task, "score_task_row": score_row,
            "score_path": score_path, "score_bytes": report_bytes,
            "score_sha256": sha256_bytes(report_bytes), "score_log_path": Path(log_value),
            "score_log_bytes": log_bytes, "report": report,
            "queue_fingerprint": expected_fp, "queue_sha256": expected_sha}


def _archive_failed_evidence(receipt_dir: Path, state_bytes: bytes, log_bytes: bytes,
                             metadata: dict):
    receipt_dir = Path(receipt_dir)
    receipt_dir.mkdir(parents=True, exist_ok=False)
    state_copy = receipt_dir / "original_state.json"
    log_copy = receipt_dir / "original_score_failure.log"
    for path, contents in ((state_copy, state_bytes), (log_copy, log_bytes)):
        with path.open("xb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
    receipt = {**metadata,
               "original_state_sha256": sha256_bytes(state_bytes),
               "original_score_log_sha256": sha256_bytes(log_bytes),
               "archived_state": state_copy.name, "archived_score_log": log_copy.name}
    receipt_path = receipt_dir / "reconciliation_receipt.json"
    with receipt_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return receipt


def _atomic_json_replace(path: Path, value: dict):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                     dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def reconcile(*, state_path=stage_queue.STATE, lock_path=stage_queue.QUEUE_LOCK,
              receipt_dir=DEFAULT_RECEIPT_DIR, process_alive=_pid_alive, now=None):
    """Record a validated recovered score without rerunning any scientific stage."""
    state_path, lock_path = Path(state_path), Path(lock_path)
    now = time.time() if now is None else float(now)
    with _exclusive_queue_lock(lock_path):
        state_bytes, state = _read_json_bytes(state_path)
        pids = sorted(_recorded_pids(state))
        live = [pid for pid in pids if process_alive(pid)]
        if live:
            raise RuntimeError(f"recorded SW0150 supervisor/child processes remain live: {live}")
        verified = _known_failure(state, state_path)
        recovery_module_sha = sha256_file(Path(__file__))
        archived = _archive_failed_evidence(
            receipt_dir, state_bytes, verified["score_log_bytes"], {
                "status": "score_only_reconciled",
                "experiment": stage_queue.NAME,
                "reconciled_at": now,
                "score_report_path": str(verified["score_path"]),
                "score_report_sha256": verified["score_sha256"],
                "score_recovery_module_sha256": recovery_module_sha,
                "frozen_queue_sha256": verified["queue_sha256"],
                "frozen_evaluate_sha256": EXPECTED_EVALUATE_SHA256,
                "registered_baseline_sha256": EXPECTED_BASELINE_SHA256,
                "recorded_pids_checked": pids,
                "recorded_pids_live": [],
                "upstream_artifacts_validated": True,
                "training_or_prediction_relaunched": False,
            })
        updated = json.loads(state_bytes.decode("utf-8-sig"))
        row = updated["tasks"]["score"]
        row.update({"status": "passed_recovered", "artifact_valid": True,
                    "recovery_report_sha256": verified["score_sha256"],
                    "recovery_receipt": str(Path(receipt_dir) / "reconciliation_receipt.json"),
                    "recovered_at": now})
        updated.update({"status": "complete", "finished": now,
                        "score_recovery": {
                            "status": "score_only_reconciled",
                            "receipt_sha256": sha256_file(
                                Path(receipt_dir) / "reconciliation_receipt.json"),
                            "original_state_sha256": archived["original_state_sha256"],
                            "score_report_sha256": verified["score_sha256"],
                            "training_or_prediction_relaunched": False}})
        _atomic_json_replace(state_path, updated)
    return updated


def _queue_module(kind):
    if kind == "sw150_diagnosis":
        from collaborative_test.SW_0150_spike_spatial_mixture import diagnosis_queue
        return diagnosis_queue, "blocked_main_queue_failure", "main SW0150 queue terminal status: score_failed"
    if kind == "sw151_preflight":
        from collaborative_test.SW_0151_source_preserving_event_history import preflight_queue
        return preflight_queue, "blocked_sw150_terminal_failure", "main SW0150 queue terminal status: score_failed"
    raise ValueError("only the registered SW0150 diagnosis and SW0151 preflight reservations may be recovered")


def _validate_reconciled_main_schema_failure():
    state = stage_queue._read_json(stage_queue.STATE)
    if (state.get("status") != "complete"
            or state.get("score_recovery", {}).get("status") != "score_only_reconciled"):
        raise ValueError("SW0150 main score queue is not reconciled complete")
    score_row = state.get("tasks", {}).get("score", {})
    receipt_path = Path(score_row.get("recovery_receipt", ""))
    if not receipt_path.is_file():
        raise ValueError("SW0150 score-only recovery receipt is missing")
    if sha256_file(receipt_path) != state["score_recovery"].get("receipt_sha256"):
        raise ValueError("SW0150 score recovery receipt changed")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    if (receipt.get("status") != "score_only_reconciled"
            or receipt.get("frozen_evaluate_sha256") != EXPECTED_EVALUATE_SHA256
            or receipt.get("registered_baseline_sha256") != EXPECTED_BASELINE_SHA256
            or receipt.get("training_or_prediction_relaunched") is not False):
        raise ValueError("SW0150 main recovery receipt does not bind the registered schema failure")
    old_log = receipt_path.parent / receipt.get("archived_score_log", "")
    if (not old_log.is_file()
            or sha256_file(old_log) != receipt.get("original_score_log_sha256")
            or EXPECTED_FAILURE_TEXT not in old_log.read_text(
                encoding="utf-8", errors="replace")):
        raise ValueError("archived SW0150 score log does not prove the known baseline-schema failure")


def _validate_blocked_reservation(kind, module, state, log_path, output_path):
    expected_status, failure_reason = _queue_module(kind)[1:]
    if state.get("status") != expected_status or state.get("reason") != failure_reason:
        raise ValueError("prerequisite queue is not blocked by the exact reconciled SW0150 score failure")
    if not isinstance(state.get("supervisor_pid"), int) or isinstance(state.get("supervisor_pid"), bool):
        raise ValueError("blocked prerequisite queue lacks a valid recorded supervisor PID")
    _validate_reconciled_main_schema_failure()
    if kind == "sw150_diagnosis":
        expected_queue = module.implementation_fingerprint()
        expected_queue_sha = module.implementation_sha256()
        expected_diag = module.diagnosis.implementation_fingerprint()
        expected_diag_sha = hashlib.sha256(json.dumps(
            expected_diag, sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode("utf-8")).hexdigest()
        if (state.get("queue_fingerprint") != expected_queue
                or state.get("implementation_sha256") != expected_queue_sha
                or state.get("diagnostic_fingerprint") != expected_diag
                or state.get("diagnostic_fingerprint_sha256") != expected_diag_sha):
            raise ValueError("SW0150 diagnostic or queue implementation fingerprint changed")
        task_rows = state.get("tasks")
        task_id = module.diagnosis.DIAGNOSTIC
        if not isinstance(task_rows, dict) or set(task_rows) != {task_id}:
            raise ValueError("SW0150 diagnostic blocked task schema changed")
        task_row = task_rows[task_id]
        if (not isinstance(task_row, dict)
                or task_row.get("status") != expected_status
                or task_row.get("child_pid") is not None
                or task_row.get("physical_gpu") is not None
                or task_row.get("gpu_uuid") is not None):
            raise ValueError("SW0150 diagnostic queue shows a GPU attempt or unexpected task state")
    else:
        expected_fp = module.implementation_fingerprint()
        expected_preflight_fp = module.preflight.implementation_fingerprint()
        expected_fp_sha = hashlib.sha256(json.dumps(
            expected_fp, sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode("utf-8")).hexdigest()
        expected_preflight_sha = hashlib.sha256(json.dumps(
            expected_preflight_fp, sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode("utf-8")).hexdigest()
        if (state.get("queue_fingerprint") != expected_fp
                or state.get("queue_fingerprint_sha256") != expected_fp_sha
                or state.get("preflight_fingerprint") != expected_preflight_fp
                or state.get("preflight_fingerprint_sha256") != expected_preflight_sha):
            raise ValueError("SW0151 preflight reservation fingerprint changed")
        if (state.get("task") != module.task_plan()[0]
                or state.get("attempt_policy") != "one_shot_no_automatic_retry"):
            raise ValueError("SW0151 blocked reservation task contract changed")
        if (state.get("child_pid") is not None
                or state.get("physical_gpu") is not None
                or state.get("gpu_uuid") is not None):
            raise ValueError("SW0151 preflight queue shows a GPU attempt")
    if Path(log_path).exists() or Path(output_path).exists():
        raise FileExistsError("preserve prerequisite queue log/output; it is not an untouched blocked reservation")


def requeue_blocked_prerequisite(kind: str, *, archive_dir: Path,
                                  process_alive=_pid_alive, popen=None, now=None):
    """Archive one untouched schema-blocked reservation, then launch its sealed queue.

    This is deliberately not called by ``reconcile``. It can only reserve the
    original diagnostic/preflight queue after SW0150 main score reconciliation.
    """
    module, expected_status, _failure_text = _queue_module(kind)
    if fcntl is None:
        raise RuntimeError("prerequisite reservation recovery requires POSIX flock")
    if kind == "sw150_diagnosis":
        state_path, lock_path = Path(module.STATE), Path(module.LOCK)
        log_path, output_path = Path(module.LOG), Path(module.OUTPUT)
        if module.main_queue_state() != ("ready", None):
            raise ValueError("SW0150 main queue is not reconciled and prediction-complete")
        argv = [sys.executable,
                "-m", "collaborative_test.SW_0150_spike_spatial_mixture.diagnosis_queue"]
    else:
        state_path, lock_path = Path(module.STATE), Path(module.LOCK)
        log_path, output_path = Path(module.LOG), Path(module.OUTPUT)
        if module.upstream_readiness() != ("ready", None):
            raise ValueError("SW0151 main and diagnostic prerequisites are not complete")
        argv = [sys.executable,
                "-m", "collaborative_test.SW_0151_source_preserving_event_history.preflight_queue"]

    archive_dir = Path(archive_dir)
    now = time.time() if now is None else float(now)
    with _exclusive_queue_lock(lock_path):
        state_bytes, state = _read_json_bytes(state_path)
        _validate_blocked_reservation(kind, module, state, log_path, output_path)
        pid = state["supervisor_pid"]
        if process_alive(pid):
            raise RuntimeError(f"blocked {kind} supervisor {pid} is still live")
        archive_dir.mkdir(parents=True, exist_ok=False)
        archived_state = archive_dir / "blocked_queue_state.json"
        if state_path.read_bytes() != state_bytes:
            raise RuntimeError("blocked queue state changed during reservation recovery")
        os.replace(state_path, archived_state)
        intent = {
            "status": "blocked_reservation_archived_before_relaunch",
            "queue_kind": kind,
            "original_status": expected_status,
            "archived_state": archived_state.name,
            "archived_state_sha256": sha256_bytes(state_bytes),
            "original_supervisor_pid": pid,
            "original_supervisor_dead": True,
            "output_absent": True,
            "attempt_log_absent": True,
            "recovered_at": now,
            "argv": argv,
        }
        intent_path = archive_dir / "requeue_intent.json"
        with intent_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(intent, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

    if not archive_dir.is_dir():
        raise RuntimeError("archived blocked-reservation directory disappeared before launch")
    supervisor_log = archive_dir / f"sup_{int(now)}_{uuid.uuid4().hex[:8]}.log"
    popen = subprocess.Popen if popen is None else popen
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    try:
        with supervisor_log.open("x", encoding="utf-8") as stream:
            process = popen(argv, cwd=module.ROOT, env=env,
                            stdin=subprocess.DEVNULL,
                            stdout=stream, stderr=subprocess.STDOUT,
                            start_new_session=True)
        receipt = {**intent, "status": "original_queue_relaunched",
                   "new_supervisor_pid": int(process.pid),
                   "supervisor_log": supervisor_log.name}
    except BaseException as exc:
        receipt = {**intent, "status": "relaunch_failed_state_preserved",
                   "error": repr(exc), "supervisor_log": supervisor_log.name}
        with (archive_dir / "relaunch_failure.json").open(
                "x", encoding="utf-8", newline="\n") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        raise
    receipt_path = archive_dir / "relaunch_receipt.json"
    with receipt_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=stage_queue.STATE)
    parser.add_argument("--lock", type=Path, default=stage_queue.QUEUE_LOCK)
    parser.add_argument("--receipt-dir", type=Path, default=DEFAULT_RECEIPT_DIR)
    args = parser.parse_args(argv)
    state = reconcile(state_path=args.state, lock_path=args.lock,
                      receipt_dir=args.receipt_dir)
    print(json.dumps({"status": state["status"], "score_status": state["tasks"]["score"]["status"],
                      "recovery": state["score_recovery"]}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
