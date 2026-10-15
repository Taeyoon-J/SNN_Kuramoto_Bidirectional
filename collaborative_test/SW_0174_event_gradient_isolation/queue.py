"""Single exclusive-GPU resource measurement; never retries a completed worker."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODULE = "collaborative_test.SW_0174_event_gradient_isolation.queue"
RUNTIME = HERE / "results_archive" / "event_gradient_audit_20261015"
STOP = threading.Event()


def atomic_state(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def worker_command(gpu, output):
    return [sys.executable, "-u", "-m", MODULE, "--worker", "--physical-gpu",
            str(gpu), "--output", str(output)]


def load_probe():
    # Historical imports may hide CUDA; preserve the leased physical visibility.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        return importlib.import_module("collaborative_test.SW_0174_event_gradient_isolation.audit")
    finally:
        if visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible


def run_worker(gpu, output):
    if gpu not in range(4) or os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise RuntimeError("worker visibility must match its exclusive physical GPU")
    probe = load_probe()
    prepared = probe.prepare_cpu()
    if probe.torch.cuda.is_initialized() or os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise RuntimeError("CPU admission initialized CUDA or changed physical visibility")
    if probe.torch.cuda.device_count() != 1:
        raise RuntimeError("worker requires exactly one leased CUDA device")
    return probe.run_audit(prepared=prepared, device="cuda:0", output=output)


def supervise():
    import fcntl
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: STOP.set())
    RUNTIME.mkdir(parents=True, exist_ok=True)
    lock = (RUNTIME / "queue.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    state, report, log = [RUNTIME / name for name in ("state.json", "report.json", "worker.log")]
    child = lease = owner = None
    def record(status, **fields):
        atomic_state(state, dict(status=status, observed_unix=time.time(), **fields))
    try:
        if any(path.exists() for path in (state, report, log)):
            raise FileExistsError("preserve existing resource measurement; inspect before any new run")
        probe = load_probe()
        prepared = probe.prepare_cpu()
        proof = prepared["source_proof"]
        fingerprint = probe.fingerprint()
        del prepared
        owner = importlib.import_module("collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4")
        record("waiting_for_exclusive_gpu", source_proof=proof, fingerprint=fingerprint)
        while not STOP.is_set():
            gpu, uuid, _, lease = owner._acquire(STOP)
            if owner._stable_idle(gpu, uuid, STOP, lambda *_: None):
                live_uuid, used, owners = owner.gpu_guard._gpu_info(gpu)
                if live_uuid == uuid and not owners and int(used) <= owner.MAX_MEMORY_MIB:
                    break
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
            lease.close()
            lease = None
        else:
            raise RuntimeError("stopped before resource worker launch")
        if probe.fingerprint() != fingerprint:
            raise RuntimeError("source files changed after CPU admission")
        command = worker_command(gpu, report)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        with log.open("x", encoding="utf-8") as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                pass_fds=(lease.fileno(),))
            record("running", pid=child.pid, argv=command, physical_gpu=gpu,
                   gpu_uuid=uuid, source_proof=proof, fingerprint=fingerprint)
            deadline = time.monotonic() + 1800
            while child.poll() is None and not STOP.wait(5):
                if time.monotonic() > deadline:
                    raise TimeoutError("bounded resource probe exceeded 1800 seconds")
                live_uuid, _, owners = owner.gpu_guard._gpu_info(gpu)
                if child.poll() is not None:
                    break
                if live_uuid != uuid or any(int(pid) != child.pid for pid in owners):
                    raise RuntimeError("foreign GPU process appeared; stop only owned worker")
            if STOP.is_set() and child.poll() is None:
                raise RuntimeError("resource supervisor stopped")
            if child.returncode != 0 or not report.is_file():
                raise RuntimeError(f"resource worker exited {child.returncode}; inspect immutable log")
        result = json.loads(report.read_text(encoding="utf-8"))
        if result.get("status") not in ("complete_event_gradient_audit", "event_gradient_audit_rejected"):
            raise ValueError("unexpected resource measurement status")
        if (result.get("implementation_fingerprint") != fingerprint
                or result.get("source_proof") != proof
                or result.get("optimizer_updates") != 0
                or result.get("validation_data_read") is not False
                or result.get("cuda_visible_devices") != str(gpu)):
            raise ValueError("resource report does not bind the admitted source and leased worker")
        record(result["status"], pid=child.pid, returncode=child.returncode,
               physical_gpu=gpu, gpu_uuid=uuid,
               report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(), fingerprint=fingerprint)
    except BaseException as exc:
        if child is not None and child.poll() is None and owner is not None:
            owner._stop_owned(child)
        # Do not overwrite an earlier run's state on duplicate launch.
        if child is not None or not state.exists():
            record("failed", error=repr(exc))
        raise
    finally:
        if child is not None and child.poll() is None and owner is not None:
            owner._stop_owned(child)
        if lease is not None:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
            lease.close()
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.physical_gpu is None or args.output is None:
            parser.error("worker requires physical GPU and output")
        run_worker(args.physical_gpu, args.output)
    else:
        supervise()


if __name__ == "__main__":
    main()
