"""Owner-aware fixed320 prediction reservation for SW0158."""
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

try:
    import fcntl
except ImportError:
    fcntl = None

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for item in (ROOT, ROOT / "collaborative_test"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

_MISSING = object()
_saved_visibility = os.environ.get("CUDA_VISIBLE_DEVICES", _MISSING)
from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4 as owner
if _saved_visibility is _MISSING:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = _saved_visibility

from collaborative_test.SW_0158_membrane_spike_credit import predict_evaluate as evaluation
from collaborative_test.SW_0158_membrane_spike_credit import train

RUNTIME_ROOT = HERE / "results_archive" / "prediction_queues_20261014"
LEASE_DIR = Path("/tmp/kevinswk_sw0113_gpu_leases")
MAX_MEMORY_MIB = 512
STABLE_SECONDS = 5.0
SAMPLE_SECONDS = 2.0
POLL_SECONDS = 5.0
WATCHDOG_SECONDS = 7200.0
DEVICE = "cuda:0"
TRAINING_QUEUE_ROOT = HERE / "results_archive" / "training_queues_20261014"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_arm(arm):
    if arm not in train.ARMS:
        raise ValueError("unregistered SW0158 prediction arm")
    runtime = RUNTIME_ROOT / arm
    return {"arm": arm, "runtime": runtime, "state": runtime / "state.json",
        "lock": runtime / "queue.lock", "log": runtime / "attempt01.log",
        "output": evaluation.PREDICTION_ROOT / f"{arm}_seed1"}


def command(arm):
    return [sys.executable, "-m", "collaborative_test.SW_0158_membrane_spike_credit.predict_evaluate",
            "--stage", "predict", "--arm", arm, "--device", DEVICE]


def implementation_fingerprint():
    owner_path = Path(owner.__file__)
    paths = (HERE / "prediction_queue.py", HERE / "predict_evaluate.py",
        HERE / "train.py", HERE / "spike_credit.py", HERE / "protocol.json",
        owner_path, owner.OWNER_GUARD)
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("SW0158 prediction queue dependency missing")
    if sha256_file(owner.OWNER_GUARD) != owner.OWNER_GUARD_SHA256:
        raise ValueError("shared GPU owner guard checksum changed")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def _atomic_json(path, record):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def _final_gpu_admission(gpu, expected_uuid):
    try:
        actual_uuid, memory, owners = owner.gpu_guard._gpu_info(gpu)
    except Exception as exc:
        return False, {"reason": "gpu_query_error", "detail": repr(exc)}
    result = {"gpu_uuid": actual_uuid, "memory_used_mib": int(memory),
              "owner_pids": list(owners)}
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


def _process_has_args(pid, module_or_script, arm):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        row = owner._proc_record(pid)
    except (OSError, RuntimeError):
        return False
    if row is None or row.get("state") == "Z":
        return False
    argv = row.get("argv", [])
    return module_or_script in argv and "--arm" in argv and arm in argv


def _training_queue_is_live(arm):
    """Wait only while the registered arm's actual supervisor/child is live."""
    path = TRAINING_QUEUE_ROOT / arm / "state.json"
    if not path.is_file():
        return False
    state = json.loads(path.read_text(encoding="utf-8"))
    active = {"waiting_for_upstream", "waiting_stable_gpu", "waiting_for_exclusive_gpu",
              "launching", "running", "queued_after_foreign_owner"}
    if (state.get("experiment") != train.EXPERIMENT or state.get("task", {}).get("arm") != arm
            or state.get("status") not in active):
        return False
    supervisor_live = _process_has_args(state.get("supervisor_pid"),
        "collaborative_test.SW_0158_membrane_spike_credit.training_queue", arm)
    child_live = _process_has_args(state.get("child_pid"),
        "collaborative_test.SW_0158_membrane_spike_credit.train", arm)
    return supervisor_live or child_live


class PredictionQueue:
    def __init__(self, arm):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if fcntl is None:
            raise RuntimeError("SW0158 prediction queue requires Linux flock")
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
            raise RuntimeError("another SW0158 prediction supervisor owns this arm") from exc
        self.state_lock = threading.RLock()
        self.stop_event = threading.Event()
        self.queue_fp = implementation_fingerprint()
        self.report_fp = evaluation.implementation_fingerprint()
        self.existing_complete = False
        try:
            if self.output.exists():
                evaluation.validate_prediction(self.arm)
                self.existing_complete = True
            else:
                if self.state_path.exists() or self.log_path.exists():
                    raise FileExistsError("preserve existing SW0158 prediction queue state/log")
                _atomic_json(self.state_path, {"experiment": train.EXPERIMENT,
                    "arm": self.arm, "status": "waiting_for_training_artifact",
                    "supervisor_pid": os.getpid(), "queue_fingerprint": self.queue_fp,
                    "prediction_fingerprint": self.report_fp,
                    "task": {"output": str(self.output), "resume": "validated per-image row commits"}})
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
        if implementation_fingerprint() != self.queue_fp:
            raise RuntimeError("SW0158 prediction queue dependencies changed after registration")
        if evaluation.implementation_fingerprint() != self.report_fp:
            raise RuntimeError("SW0158 prediction implementation changed after registration")

    def _training_ready(self):
        artifact = evaluation.validate_training_artifact(self.arm)
        return artifact

    def run(self):
        if self.existing_complete:
            self._unlock()
            return 0
        proc = lease = None
        try:
            while not self.stop_event.is_set():
                try:
                    artifact = self._training_ready()
                    break
                except FileNotFoundError as exc:
                    manifest = train.TRAIN_ROOT / f"{self.arm}_seed1" / "manifest.json"
                    if manifest.exists() or not _training_queue_is_live(self.arm):
                        raise RuntimeError("SW0158 training dependency is missing and no live owner remains") from exc
                    self._update("waiting_for_training_artifact", last_validation=repr(exc))
                    self.stop_event.wait(POLL_SECONDS)
            if self.stop_event.is_set():
                self._update("cancelled_before_gpu_reservation")
                return 1
            self._verify()
            while not self.stop_event.is_set():
                gpu, gpu_uuid, memory, lease = owner._acquire(self.stop_event)
                stable = owner._stable_idle(gpu, gpu_uuid, self.stop_event,
                    lambda elapsed, used: self._update("waiting_stable_gpu", gpu=gpu,
                        gpu_uuid=gpu_uuid, stable_seconds=elapsed, memory_used_mib=used))
                if not stable:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                    if self.stop_event.is_set():
                        self._update("cancelled_before_child")
                        return 1
                    self._update("waiting_for_exclusive_gpu", reservation_released=True)
                    continue
                self._verify()
                fresh = self._training_ready()
                if (fresh["manifest_sha256"] != artifact["manifest_sha256"]
                        or self.output.exists()):
                    raise RuntimeError("SW0158 trained artifact changed or prediction appeared before launch")
                admitted, final_state = _final_gpu_admission(gpu, gpu_uuid)
                if not admitted:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                    self._update("waiting_for_exclusive_gpu", reservation_released=True,
                                 final_gpu_check=final_state)
                    continue
                argv = command(self.arm)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                    OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                _atomic_json(self.state_path, {**json.loads(self.state_path.read_text()),
                    "status": "launching", "gpu": gpu, "gpu_uuid": gpu_uuid,
                    "trained_manifest_sha256": artifact["manifest_sha256"],
                    "child_argv": argv, "updated": time.time()})
                child_start = time.monotonic()
                with self.log_path.open("x", encoding="utf-8") as log:
                    proc = subprocess.Popen(argv, cwd=ROOT, env=env,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True, pass_fds=(lease.fileno(),))
                    self._update("running", child_pid=proc.pid, child_argv=argv,
                        physical_gpu=gpu, gpu_uuid=gpu_uuid, inherited_lease_fd=True,
                        log=str(self.log_path), trained_manifest_sha256=artifact["manifest_sha256"])
                    while proc.poll() is None and not self.stop_event.is_set():
                        if time.monotonic() - child_start > WATCHDOG_SECONDS:
                            owner._stop_owned(proc)
                            self._update("failed_watchdog_timeout", child_pid=proc.pid,
                                         output_preserved=self.output.exists(), finished=time.time())
                            return 1
                        self.stop_event.wait(SAMPLE_SECONDS)
                        if proc.poll() is not None: break
                        try:
                            current_uuid, _memory, owners = owner.gpu_guard._gpu_info(gpu)
                            foreign = (["gpu_uuid_changed"] if current_uuid != gpu_uuid else
                                owner.gpu_guard._foreign_owners(owners,
                                    owner.gpu_guard._process_tree(proc.pid)))
                        except Exception as exc:
                            self._update("running", last_gpu_query_error=repr(exc))
                            continue
                        if foreign:
                            owner._stop_owned(proc)
                            self._update("failed_foreign_gpu_owner", foreign_owner_pids=foreign,
                                child_pid=proc.pid, per_image_work_preserved=True,
                                output_preserved=self.output.exists(), finished=time.time())
                            return 1
                if proc.poll() is None:
                    owner._stop_owned(proc)
                    self._update("cancelled_owned_prediction", child_pid=proc.pid,
                        per_image_work_preserved=True, output_preserved=self.output.exists(),
                        finished=time.time())
                    return 1
                rc = int(proc.wait())
                if rc == 0:
                    result = evaluation.validate_prediction(self.arm)
                    self._update("complete", child_returncode=rc,
                        prediction_sha256=result["prediction_sha256"],
                        protocol_sha256=result["protocol_sha256"], finished=time.time())
                    return 0
                self._update("failed_child", child_returncode=rc,
                    per_image_work_preserved=True, output_preserved=self.output.exists(), finished=time.time())
                return 1
            return 1
        except BaseException as exc:
            if proc is not None and proc.poll() is None:
                owner._stop_owned(proc)
            self._update("failed", error=repr(exc), finished=time.time())
            raise
        finally:
            try:
                if proc is not None and proc.poll() is None:
                    owner._stop_owned(proc)
            finally:
                if lease is not None:
                    try: fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                    finally: lease.close()
                self._unlock()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=train.ARMS, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"status": "planned", "task": configure_arm(args.arm),
            "argv": command(args.arm), "cuda_visible_devices_parent": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
            "queue_fingerprint": implementation_fingerprint(),
            "prediction_fingerprint": evaluation.implementation_fingerprint()}, allow_nan=False), flush=True)
        return 0
    return PredictionQueue(args.arm).run()


if __name__ == "__main__":
    raise SystemExit(main())
