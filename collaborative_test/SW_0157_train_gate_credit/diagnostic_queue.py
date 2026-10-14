"""Single-job owner-aware GPU reservation for the SW0157 TRAIN diagnostic."""
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

# A CLI supervisor must hide CUDA before importing any experiment/Torch modules.
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
except ImportError:  # CPU tests can inspect the plan but cannot reserve a GPU.
    fcntl = None

_MISSING = object()
_saved_visibility = os.environ.get("CUDA_VISIBLE_DEVICES", _MISSING)
from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4 as owner
if _saved_visibility is _MISSING:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = _saved_visibility

from collaborative_test.SW_0143_native32_selective_geodesic import run as sw143
from collaborative_test.SW_0155_supervised_spike_relations import train
from collaborative_test.SW_0157_train_gate_credit import diagnostic

EXPERIMENT = diagnostic.EXPERIMENT
RUNTIME = HERE / "results_archive" / "queue_20261014"
STATE, LOCK = RUNTIME / "state.json", RUNTIME / "queue.lock"
LOG = RUNTIME / "attempt01.log"
OUTPUT = diagnostic.OUTPUT
GPU_LEASE_DIR = Path("/tmp/kevinswk_sw0113_gpu_leases")
MAX_MEMORY_MIB = 512
STABLE_SECONDS = 5.0
SAMPLE_SECONDS = 2.0
POLL_SECONDS = 3.0
WATCHDOG_SECONDS = diagnostic.WATCHDOG_SECONDS
DEVICE = "cuda:0"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def report_fingerprint():
    return diagnostic.implementation_fingerprint()


def implementation_fingerprint():
    paths = (HERE / "diagnostic_queue.py", HERE / "diagnostic.py",
             HERE / "protocol.json", HERE / "README.md",
             Path(owner.__file__), owner.OWNER_GUARD)
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("SW0157 queue dependency missing")
    if sha256_file(owner.OWNER_GUARD) != owner.OWNER_GUARD_SHA256:
        raise ValueError("registered shared GPU owner guard changed")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def task_plan():
    return {"task_id": "train_gate_credit_eight_train_images", "device": DEVICE,
            "models": list(diagnostic.MODEL_ORDER), "source_positions": list(diagnostic.POSITIONS),
            "steps": diagnostic.STEPS, "settle": diagnostic.SETTLE,
            "live_tail_steps": diagnostic.LIVE_TAIL, "watchdog_seconds": WATCHDOG_SECONDS,
            "output": str(OUTPUT), "log": str(LOG), "depends_on": [
                "SW0143_T_seed1_complete", "SW0155_encoder_graph_training_complete",
                "SW0155_joint_training_complete"]}


def command():
    return [sys.executable, "-m", "collaborative_test.SW_0157_train_gate_credit.diagnostic",
            "--device", DEVICE, "--output", str(OUTPUT),
            "--watchdog-seconds", str(WATCHDOG_SECONDS)]


def _upstream_ready():
    t_dir = sw143._arm_dir("T", 1)
    if (not sw143._preflight_path(1).is_file()
            or not (t_dir / "predictions.npz").is_file()
            or not (t_dir / "protocol.json").is_file()):
        return False, "waiting_for_sw0143_T_seed1"
    missing_arms = [arm for arm in train.ARMS
                    if not (train.TRAIN_ROOT / f"{arm}_seed1" / "manifest.json").is_file()]
    if missing_arms:
        return False, "waiting_for_sw0155_training:" + ",".join(missing_arms)
    return True, diagnostic.validate_upstream()


def validate_report(path=OUTPUT, *, expected_fingerprint=None, upstream=None):
    return diagnostic.validate_report(path, expected_fingerprint=expected_fingerprint,
                                      upstream=upstream)


def _check_absent(paths):
    present = [str(Path(path)) for path in paths if Path(path).exists()]
    if present:
        raise FileExistsError(f"preserve preexisting SW0157 outputs/log/state: {present}")


def _final_gpu_admission(gpu, expected_uuid):
    """Recheck exclusivity after upstream hashing, while the lease is still held."""
    try:
        actual_uuid, memory, owners = owner.gpu_guard._gpu_info(gpu)
    except Exception as exc:
        return False, {"reason": "gpu_query_error", "detail": repr(exc)}
    state = {"gpu_uuid": actual_uuid, "memory_used_mib": int(memory),
             "owner_pids": list(owners)}
    if actual_uuid != expected_uuid:
        state["reason"] = "gpu_uuid_changed"
        return False, state
    if owners:
        state["reason"] = "gpu_has_owners"
        return False, state
    if int(memory) > MAX_MEMORY_MIB:
        state["reason"] = "gpu_memory_over_limit"
        return False, state
    state["reason"] = "idle"
    return True, state


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


class DiagnosticQueue:
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if fcntl is None:
            raise RuntimeError("SW0157 GPU queue requires Linux flock")
        RUNTIME.mkdir(parents=True, exist_ok=True)
        self.lock_stream = LOCK.open("a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock_stream.close()
            raise RuntimeError("another SW0157 diagnostic supervisor owns the queue") from exc
        self.stop_event = threading.Event()
        self.state_lock = threading.RLock()
        self.plan = task_plan()
        self.report_fp = report_fingerprint()
        self.queue_fp = implementation_fingerprint()
        self.existing_complete = False
        try:
            if OUTPUT.exists():
                if not validate_report(OUTPUT, expected_fingerprint=self.report_fp):
                    raise FileExistsError("preserve invalid or differently bound SW0157 report")
                self.existing_complete = True
            else:
                _check_absent((STATE, LOG))
                _atomic_json(STATE, {"experiment": EXPERIMENT, "status": "waiting_for_upstream",
                    "supervisor_pid": os.getpid(), "task": self.plan,
                    "report_fingerprint": self.report_fp, "queue_fingerprint": self.queue_fp,
                    "optimizer_updates": 0, "ground_truth_scope": "TRAIN masks only"})
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
            state = json.loads(STATE.read_text(encoding="utf-8"))
            state.update(status=status, updated=time.time(), **fields)
            _atomic_json(STATE, state)

    def _verify(self):
        if report_fingerprint() != self.report_fp or implementation_fingerprint() != self.queue_fp:
            raise RuntimeError("SW0157 dependencies changed after queue registration")

    def run(self):
        if self.existing_complete:
            self._unlock()
            return 0
        proc = lease = None
        try:
            while not self.stop_event.is_set():
                ready, upstream = _upstream_ready()
                if ready:
                    break
                self._update("waiting_for_upstream", upstream_state=upstream)
                self.stop_event.wait(POLL_SECONDS)
            if self.stop_event.is_set():
                self._update("cancelled_before_gpu_reservation")
                return 1
            self._verify()
            _check_absent((OUTPUT, LOG))
            while not self.stop_event.is_set():
                gpu, gpu_uuid, memory, lease = owner._acquire(self.stop_event)
                try:
                    stable = owner._stable_idle(gpu, gpu_uuid, self.stop_event,
                        lambda elapsed, used: self._update("waiting_stable_gpu", gpu=gpu,
                            gpu_uuid=gpu_uuid, stable_seconds=elapsed, memory_used_mib=used))
                    if not stable:
                        fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                        lease.close(); lease = None
                        if self.stop_event.is_set():
                            return 1
                        self._update("waiting_for_exclusive_gpu", reservation_released=True)
                        continue
                    self._verify()
                    # Revalidate complete upstream lineage after waiting for a lease.
                    ready, current_upstream = _upstream_ready()
                    if not ready or current_upstream != upstream:
                        raise RuntimeError("SW0157 upstream changed while waiting for GPU")
                    _check_absent((OUTPUT, LOG))
                    argv = command()
                    child_env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                    admitted, admission_state = _final_gpu_admission(gpu, gpu_uuid)
                    if not admitted:
                        fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                        lease.close(); lease = None
                        self._update("waiting_for_exclusive_gpu",
                            reservation_released=True, final_gpu_check=admission_state)
                        if self.stop_event.is_set():
                            return 1
                        continue
                    child_started = time.monotonic()
                    with LOG.open("x", encoding="utf-8") as log:
                        proc = subprocess.Popen(argv, cwd=ROOT, env=child_env,
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True, pass_fds=(lease.fileno(),))
                        self._update("running", child_pid=proc.pid, child_argv=argv,
                            physical_gpu=gpu, gpu_uuid=gpu_uuid, memory_before_mib=memory,
                            inherited_lease_fd=True, log=str(LOG), watchdog_seconds=WATCHDOG_SECONDS)
                        while proc.poll() is None and not self.stop_event.is_set():
                            if time.monotonic() - child_started > WATCHDOG_SECONDS:
                                owner._stop_owned(proc)
                                self._update("failed_watchdog_timeout", child_pid=proc.pid,
                                             report_preserved=OUTPUT.exists(), finished=time.time())
                                return 1
                            time.sleep(SAMPLE_SECONDS)
                            if proc.poll() is not None:
                                break
                            try:
                                current_uuid, _used, owners = owner.gpu_guard._gpu_info(gpu)
                                foreign = (["gpu_uuid_changed"] if current_uuid != gpu_uuid else
                                    owner.gpu_guard._foreign_owners(owners,
                                        owner.gpu_guard._process_tree(proc.pid)))
                            except Exception as exc:
                                self._update("running", last_gpu_query_error=repr(exc))
                                continue
                            if foreign:
                                owner._stop_owned(proc)
                                self._update("failed_foreign_gpu_owner", foreign_owner_pids=foreign,
                                    child_pid=proc.pid, report_preserved=OUTPUT.exists(),
                                    finished=time.time())
                                return 1
                    rc = int(proc.wait())
                    valid = (rc == 0 and validate_report(OUTPUT,
                        expected_fingerprint=self.report_fp, upstream=upstream))
                    self._update("complete" if valid else "failed_child_or_report_validation",
                        child_returncode=rc, report_valid=bool(valid), finished=time.time(),
                        report_sha256=sha256_file(OUTPUT) if OUTPUT.is_file() else None)
                    return 0 if valid else 1
                finally:
                    try:
                        if proc is not None and proc.poll() is None:
                            owner._stop_owned(proc)
                    finally:
                        if lease is not None:
                            try:
                                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                            finally:
                                lease.close(); lease = None
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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"status": "planned", "task": task_plan(), "argv": command(),
            "cuda_visible_devices_parent": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
            "watchdog_seconds": WATCHDOG_SECONDS,
            "report_fingerprint": report_fingerprint(),
            "queue_fingerprint": implementation_fingerprint()}, allow_nan=False), flush=True)
        return 0
    return DiagnosticQueue().run()


if __name__ == "__main__":
    raise SystemExit(main())
