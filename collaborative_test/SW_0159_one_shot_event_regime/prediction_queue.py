"""Owner-aware SW0159 per-arm fixed320 prediction reservation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for item in (ROOT, ROOT / "collaborative_test"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

try:
    import fcntl
except ImportError:
    fcntl = None

_MISSING = object()
_saved_visibility = os.environ.get("CUDA_VISIBLE_DEVICES", _MISSING)
from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4 as owner
if _saved_visibility is _MISSING:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = _saved_visibility

from collaborative_test.SW_0159_one_shot_event_regime import predict_evaluate as evaluation
from collaborative_test.SW_0159_one_shot_event_regime import train

RUNTIME_ROOT = HERE / "results_archive" / "prediction_queues_20261014"
TRAINING_QUEUE_ROOT = HERE / "results_archive" / "training_queues_20261014"
MAX_MEMORY_MIB, STABLE_SECONDS = 512, 5.0
SAMPLE_SECONDS, POLL_SECONDS, WATCHDOG_SECONDS = 2.0, 5.0, 7200.0
DEVICE = "cuda:0"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_arm(arm):
    if arm not in train.ARMS:
        raise ValueError("unregistered SW0159 arm")
    runtime = RUNTIME_ROOT / arm
    return {"arm": arm, "runtime": runtime, "state": runtime / "state.json",
        "lock": runtime / "queue.lock", "log": runtime / "attempt01.log",
        "output": evaluation.PREDICTION_ROOT / f"{arm}_seed1"}


def command(arm):
    return [sys.executable, "-m", "collaborative_test.SW_0159_one_shot_event_regime.predict_evaluate",
            "--stage", "predict", "--arm", arm, "--device", DEVICE]


def implementation_fingerprint():
    guard = Path(owner.OWNER_GUARD)
    paths = (HERE / "prediction_queue.py", HERE / "predict_evaluate.py", HERE / "train.py",
        HERE / "preflight.py", HERE / "protocol.json", Path(owner.__file__), guard)
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("SW0159 prediction queue dependency missing")
    if sha256_file(guard) != owner.OWNER_GUARD_SHA256:
        raise ValueError("shared GPU owner guard checksum changed")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def _process_has_args(pid, module, arm):
    if type(pid) is not int or pid <= 0:
        return False
    try:
        row = owner._proc_record(pid)
    except (OSError, RuntimeError):
        return False
    if row is None or row.get("state") == "Z":
        return False
    argv = row.get("argv", [])
    return module in argv and "--arm" in argv and arm in argv


def _training_queue_is_live(arm):
    path = TRAINING_QUEUE_ROOT / arm / "state.json"
    if not path.is_file():
        return False
    state = json.loads(path.read_text(encoding="utf-8"))
    active = {"waiting_for_preflight", "waiting_stable_gpu", "waiting_for_exclusive_gpu",
              "running", "launching"}
    if (state.get("experiment") != train.EXPERIMENT or state.get("arm") != arm
            or state.get("status") not in active):
        return False
    supervisor = _process_has_args(state.get("supervisor_pid"),
        "collaborative_test.SW_0159_one_shot_event_regime.training_queue", arm)
    child = _process_has_args(state.get("child_pid"),
        "collaborative_test.SW_0159_one_shot_event_regime.train", arm)
    return supervisor or child


def _final_gpu_admission(gpu, expected_uuid):
    try:
        actual_uuid, memory, owners = owner.gpu_guard._gpu_info(gpu)
    except Exception as exc:
        return False, {"reason": "gpu_query_error", "detail": repr(exc)}
    result = {"gpu_uuid": actual_uuid, "memory_used_mib": int(memory), "owner_pids": list(owners)}
    if actual_uuid != expected_uuid:
        result["reason"] = "gpu_uuid_changed"
    elif owners:
        result["reason"] = "gpu_has_owners"
    elif int(memory) > MAX_MEMORY_MIB:
        result["reason"] = "gpu_memory_over_limit"
    else:
        result["reason"] = "idle"
        return True, result
    return False, result


def _atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


class PredictionQueue:
    def __init__(self, arm):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if fcntl is None:
            raise RuntimeError("SW0159 prediction queue requires Linux flock")
        self.config = configure_arm(arm)
        self.arm = arm
        self.runtime, self.state_path = self.config["runtime"], self.config["state"]
        self.lock_path, self.log_path, self.output = (
            self.config["lock"], self.config["log"], self.config["output"])
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.lock_stream = self.lock_path.open("a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock_stream.close()
            raise RuntimeError(f"another SW0159 {arm} prediction queue owns the lock") from exc
        self.state_lock = threading.RLock()
        self.stop_event = threading.Event()
        self.queue_fp = implementation_fingerprint()
        self.report_fp = evaluation.implementation_fingerprint()
        self.existing_complete = False
        try:
            if self.output.exists():
                evaluation.validate_prediction(arm)
                self.existing_complete = True
            elif self.state_path.exists() or self.log_path.exists():
                raise FileExistsError("preserve existing SW0159 prediction state/log")
            else:
                _atomic_json(self.state_path, {"experiment": train.EXPERIMENT, "arm": arm,
                    "status": "waiting_for_training_artifact", "supervisor_pid": os.getpid(),
                    "queue_fingerprint": self.queue_fp, "prediction_fingerprint": self.report_fp,
                    "output": str(self.output), "resume": "atomic per-image prediction rows"})
        except BaseException:
            self._unlock()
            raise

    def _unlock(self):
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_UN)
        finally:
            self.lock_stream.close()

    def _update(self, status, **fields):
        with self.state_lock:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            state.update(status=status, updated=time.time(), **fields)
            _atomic_json(self.state_path, state)

    def _verify(self):
        if (implementation_fingerprint() != self.queue_fp
                or evaluation.implementation_fingerprint() != self.report_fp):
            raise RuntimeError("SW0159 prediction dependencies changed after registration")

    def run(self):
        if self.existing_complete:
            self._unlock(); return 0
        proc = lease = None
        try:
            while not self.stop_event.is_set():
                manifest = train.TRAIN_ROOT / f"{self.arm}_seed1" / "manifest.json"
                if manifest.is_file():
                    # A present malformed artifact is terminal, never treated as pending.
                    artifact = evaluation.validate_training_artifact(self.arm)
                    break
                if not _training_queue_is_live(self.arm):
                    raise RuntimeError("SW0159 training artifact absent and no live arm owner remains")
                self._update("waiting_for_training_artifact", training_queue_state=str(
                    TRAINING_QUEUE_ROOT / self.arm / "state.json"))
                self.stop_event.wait(POLL_SECONDS)
            if self.stop_event.is_set():
                self._update("cancelled_before_gpu_reservation"); return 1
            self._verify()
            while not self.stop_event.is_set():
                gpu, gpu_uuid, memory, lease = owner._acquire(self.stop_event)
                try:
                    stable = owner._stable_idle(gpu, gpu_uuid, self.stop_event,
                        lambda elapsed, used: self._update("waiting_stable_gpu", gpu=gpu,
                            gpu_uuid=gpu_uuid, stable_seconds=elapsed, memory_used_mib=used))
                    if not stable:
                        fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                        if self.stop_event.is_set():
                            self._update("cancelled_before_child"); return 1
                        self._update("waiting_for_exclusive_gpu", reservation_released=True)
                        continue
                    self._verify()
                    fresh = evaluation.validate_training_artifact(self.arm)
                    if (fresh["manifest_sha256"] != artifact["manifest_sha256"]
                            or self.output.exists()):
                        raise RuntimeError("SW0159 training artifact changed or output appeared before launch")
                    admitted, admission = _final_gpu_admission(gpu, gpu_uuid)
                    if not admitted:
                        fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                        self._update("waiting_for_exclusive_gpu", reservation_released=True,
                                     final_gpu_check=admission)
                        continue
                    argv = command(self.arm)
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                               MKL_NUM_THREADS="1")
                    with self.log_path.open("x", encoding="utf-8") as log:
                        proc = subprocess.Popen(argv, cwd=ROOT, env=env,
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True, pass_fds=(lease.fileno(),))
                        started = time.monotonic()
                        self._update("running", child_pid=proc.pid, child_argv=argv,
                            physical_gpu=gpu, gpu_uuid=gpu_uuid, memory_before_mib=memory,
                            inherited_lease_fd=True, trained_manifest_sha256=artifact["manifest_sha256"],
                            log=str(self.log_path), watchdog_seconds=WATCHDOG_SECONDS)
                        while proc.poll() is None and not self.stop_event.is_set():
                            if time.monotonic() - started > WATCHDOG_SECONDS:
                                owner._stop_owned(proc)
                                self._update("failed_watchdog", child_pid=proc.pid,
                                    rows_preserved=True, finished=time.time()); return 1
                            self.stop_event.wait(SAMPLE_SECONDS)
                            if proc.poll() is not None: break
                            try:
                                now_uuid, _used, owners = owner.gpu_guard._gpu_info(gpu)
                                foreign = (["gpu_uuid_changed"] if now_uuid != gpu_uuid else
                                    owner.gpu_guard._foreign_owners(owners,
                                        owner.gpu_guard._process_tree(proc.pid)))
                            except Exception as exc:
                                self._update("running", last_gpu_query_error=repr(exc)); continue
                            if foreign:
                                owner._stop_owned(proc)
                                self._update("failed_foreign_gpu_owner", foreign_owner_pids=foreign,
                                    rows_preserved=True, child_pid=proc.pid, finished=time.time()); return 1
                    rc = int(proc.wait())
                    result = (evaluation.validate_prediction(self.arm)
                              if rc == 0 else None)
                    valid = rc == 0 and bool(result)
                    self._update("complete" if valid else "failed_child_or_prediction_validation",
                        child_returncode=rc, prediction_valid=bool(valid),
                        prediction_sha256=result["prediction_sha256"] if valid else None,
                        rows_preserved=True, finished=time.time())
                    return 0 if valid else 1
                finally:
                    try:
                        if proc is not None and proc.poll() is None:
                            owner._stop_owned(proc)
                    finally:
                        if lease is not None:
                            try: fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                            finally: lease.close(); lease = None
            return 1
        except BaseException as exc:
            if proc is not None and proc.poll() is None:
                owner._stop_owned(proc)
            self._update("failed", error=repr(exc), finished=time.time())
            raise
        finally:
            self._unlock()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=train.ARMS, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = configure_arm(args.arm)
    if args.dry_run:
        print(json.dumps({"status": "planned", "arm": args.arm, "output": str(config["output"]),
            "argv": command(args.arm), "queue_fingerprint": implementation_fingerprint(),
            "prediction_fingerprint": evaluation.implementation_fingerprint()}, allow_nan=False), flush=True)
        return 0
    return PredictionQueue(args.arm).run()


if __name__ == "__main__":
    raise SystemExit(main())
