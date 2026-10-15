"""One-arm-at-a-time GPU owner queue for the SW0168 256-update pilot."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

_WORKER = "--worker" in sys.argv
if not _WORKER:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import hashlib

import numpy as np

from collaborative_test.SW_0168_end_to_end_connectivity import train

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNTIME = HERE / "results_archive" / "training_queue_20261015"
OWNER_MODULE = "collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4"
_stop = threading.Event()


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _load_owner_preserving_visibility():
    import importlib
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    owner = importlib.import_module(OWNER_MODULE)
    if visible is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
    return owner


def _state_path(arm):
    return RUNTIME / f"{arm}_state.json"


def _write_state(arm, status, **fields):
    state = {"experiment": "SW0168_training_queue", "arm": arm,
             "supervisor_pid": os.getpid(), "status": status,
             "updated": time.time(), **fields}
    _atomic_json(_state_path(arm), state)


def _worker(arm, physical_gpu):
    result = train.train_arm(arm, physical_gpu=physical_gpu)
    print(json.dumps({"status": "training_complete", "arm": arm,
                      "completed_updates": result["completed_updates"]}, sort_keys=True))
    return result


def _worker_command(arm, gpu):
    if arm not in train.ARMS or type(gpu) is not int or gpu not in range(4):
        raise ValueError("worker command requires a registered arm and physical GPU")
    return [sys.executable, "-m",
            "collaborative_test.SW_0168_end_to_end_connectivity.training_queue",
            "--worker", "--arm", arm, "--physical-gpu", str(int(gpu))]


def _foreign_owner_changed(expected_uuid, actual_uuid, owners, child_pid):
    """Yield only when hardware identity changed or a non-owned GPU PID appeared."""
    return actual_uuid != expected_uuid or any(int(pid) != int(child_pid) for pid in owners)


def run(arm, *, timeout_seconds=8 * 3600):
    if arm not in train.ARMS:
        raise ValueError("unregistered SW0168 arm")
    import fcntl
    signal.signal(signal.SIGTERM, lambda *_: _stop.set())
    signal.signal(signal.SIGINT, lambda *_: _stop.set())
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("queue supervisor must keep CUDA hidden")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    state_path = _state_path(arm)
    lock_path = RUNTIME / f"{arm}.lock"
    if state_path.exists() or any(RUNTIME.glob(f"{arm}_worker_attempt*.log")):
        raise FileExistsError(f"preserving existing SW0168 {arm} queue artifacts")
    lock = lock_path.open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    owner = None
    lease = None
    child = None
    attempt = 0
    try:
        _write_state(arm, "validating_admission", trainer_sha256=_sha(HERE / "train.py"),
                     queue_sha256=_sha(__file__))
        # Strict CPU preflight/source/cache admission before reserving a GPU.
        prep, proof, admission_sha = train._load_admission()
        del prep
        trainer_fp = train.implementation_fingerprint()
        queue_sha = _sha(__file__)
        if not trainer_fp:
            raise AssertionError("trainer fingerprint unavailable")
        owner = _load_owner_preserving_visibility()
        deadline = time.monotonic() + timeout_seconds
        while not _stop.is_set():
            if time.monotonic() > deadline:
                raise TimeoutError("SW0168 training queue exceeded its bounded wait")
            gpu, uuid, memory, lease = owner._acquire(_stop)
            _write_state(arm, "waiting_for_stable_gpu", physical_gpu=gpu, gpu_uuid=uuid,
                         memory_used_mib=int(memory), preflight_report_sha256=admission_sha)
            stable = owner._stable_idle(gpu, uuid, _stop,
                lambda elapsed, used: _write_state(arm, "waiting_for_stable_gpu",
                    physical_gpu=gpu, gpu_uuid=uuid, stable_seconds=elapsed,
                    memory_used_mib=int(used), preflight_report_sha256=admission_sha))
            if not stable:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                continue
            # Re-read the terminal receipt while holding the physical lease.
            prep2, proof2, admission_sha2 = train._load_admission()
            del prep2
            if (proof2 != proof or admission_sha2 != admission_sha
                    or train.implementation_fingerprint() != trainer_fp
                    or _sha(__file__) != queue_sha):
                raise RuntimeError("SW0168 admission/source changed while waiting for GPU")
            actual_uuid, used, owners = owner.gpu_guard._gpu_info(gpu)
            if actual_uuid != uuid or owners or int(used) > owner.MAX_MEMORY_MIB:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                _write_state(arm, "waiting_for_exclusive_gpu", gpu_uuid=actual_uuid,
                             memory_used_mib=int(used), owners=list(owners))
                continue
            attempt += 1
            log_path = RUNTIME / f"{arm}_worker_attempt{attempt:02d}.log"
            command = _worker_command(arm, gpu)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
            retry_after_owner_change = False
            with log_path.open("x", encoding="utf-8", newline="\n") as stream:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                    pass_fds=(lease.fileno(),))
                _write_state(arm, "running_training", child_pid=child.pid, child_pgid=child.pid,
                    child_argv=command, physical_gpu=gpu, gpu_uuid=uuid,
                    preflight_report_sha256=admission_sha, attempt=attempt)
                end = time.monotonic() + timeout_seconds
                while child.poll() is None and not _stop.is_set():
                    if time.monotonic() > end:
                        owner._stop_owned(child)
                        raise TimeoutError("SW0168 training worker timed out")
                    now_uuid, now_memory, gpu_owners = owner.gpu_guard._gpu_info(gpu)
                    # Recheck the owned child after a potentially slow NVML read.
                    if child.poll() is None and _foreign_owner_changed(uuid, now_uuid, gpu_owners, child.pid):
                        owner._stop_owned(child)
                        retry_after_owner_change = True
                        _write_state(arm, "yielding_foreign_gpu", child_pid=child.pid,
                            physical_gpu=gpu, gpu_uuid=uuid, observed_gpu_uuid=now_uuid,
                            owners=list(gpu_owners), attempt=attempt)
                        break
                    if child.poll() is None:
                        proc = owner._proc_record(child.pid)
                        identity_bad = (not proc or proc.get("state") == "Z"
                                or command[1] not in proc.get("argv", [])
                                or command[2] not in proc.get("argv", []))
                        if identity_bad:
                            time.sleep(0.05)
                            if child.poll() is None:
                                owner._stop_owned(child)
                                raise RuntimeError("owned training child identity changed")
                    _write_state(arm, "running_training", child_pid=child.pid,
                        physical_gpu=gpu, gpu_uuid=uuid, memory_used_mib=int(now_memory),
                        preflight_report_sha256=admission_sha, attempt=attempt)
                    time.sleep(3)
                rc = child.wait()
            if retry_after_owner_change:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                child = None
                continue
            if rc != 0:
                raise RuntimeError(f"SW0168 training worker exited {rc}; inspect {log_path}")
            artifact = train.validate_training_artifact(arm)
            if artifact.get("bindings", {}).get("preflight_report_sha256") != admission_sha:
                raise ValueError("completed training artifact is not bound to admitted preflight")
            _write_state(arm, "complete", output=str(train.TRAIN_ROOT / f"{arm}_seed1"),
                manifest_sha256=_sha(train.TRAIN_ROOT / f"{arm}_seed1" / "manifest.json"),
                child_pid=child.pid, physical_gpu=gpu, gpu_uuid=uuid,
                preflight_report_sha256=admission_sha, attempt=attempt)
            return artifact
        else:
            raise RuntimeError("SW0168 training queue stopped before GPU admission")

    except BaseException as exc:
        if child is not None and child.poll() is None:
            owner._stop_owned(child)
        _write_state(arm, "failed", error=repr(exc), finished=time.time())
        raise
    finally:
        if child is not None and child.poll() is None:
            owner._stop_owned(child)
        if lease is not None:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
            lease.close()
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=train.ARMS, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--physical-gpu", type=int)
    args = parser.parse_args()
    if args.worker:
        if args.physical_gpu not in range(4):
            raise SystemExit("worker requires its leased physical GPU")
        _worker(args.arm, args.physical_gpu)
    else:
        run(args.arm)


if __name__ == "__main__":
    main()
