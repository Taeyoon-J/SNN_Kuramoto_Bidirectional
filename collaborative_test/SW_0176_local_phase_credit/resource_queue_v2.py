"""Exclusive-owner supervisor for the bounded SW0176 source preflight."""
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
MODULE = "collaborative_test.SW_0176_local_phase_credit.resource_queue_v2"
RUNTIME = HERE / "results_archive" / "source_preflight_v2_20261016"
STOP = threading.Event()


def _atomic_state(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _create_report(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def worker_command(gpu, output):
    return [sys.executable, "-u", "-m", MODULE, "--worker", "--physical-gpu",
            str(gpu), "--output", str(output)]


def _load_preflight():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        return importlib.import_module("collaborative_test.SW_0176_local_phase_credit.preflight_v2")
    finally:
        if visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible


def implementation_fingerprint():
    prep = _load_preflight()
    return prep.implementation_fingerprint()


def _wait_for_stable_idle(owner, gpu, uuid, stop_event):
    return owner._stable_idle(gpu, uuid, stop_event, lambda *_: None)


def run_worker(gpu, output):
    if gpu not in range(4) or os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise RuntimeError("SW0176 worker visibility must match its exclusive physical GPU")
    prep = _load_preflight()
    prepared = prep.prepare_inputs_cpu()
    if (prep.torch.cuda.is_initialized()
            or os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu)):
        raise RuntimeError("SW0176 CPU source admission initialized CUDA or changed GPU visibility")
    if prep.torch.cuda.device_count() != 1:
        raise RuntimeError("SW0176 worker must see exactly its leased CUDA device")
    return prep.run_preflight(device="cuda:0", prepared=prepared, output=output)


def supervise():
    import fcntl
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: STOP.set())
    RUNTIME.mkdir(parents=True, exist_ok=True)
    lock = (RUNTIME / "queue.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    state, report, log = [RUNTIME / name for name in
                          ("state.json", "preflight.json", "worker.log")]
    child = lease = owner = None
    owned_state = False

    def record(status, **fields):
        _atomic_state(state, {"experiment": "SW0176_local_phase_credit_source_preflight",
            "status": status, "observed_unix": time.time(),
            "supervisor_pid": os.getpid(), **fields})

    try:
        if any(path.exists() for path in (state, report, log)):
            raise FileExistsError("preserve existing SW0176 preflight run; inspect before retry")
        record("validating_source_cpu")
        owned_state = True
        prep = _load_preflight()
        prepared = prep.prepare_inputs_cpu()
        proof = prepared["source_proof"]
        fingerprint = prep.implementation_fingerprint()
        if prepared.get("implementation_fingerprint") != fingerprint:
            raise ValueError("SW0176 CPU preparation has a stale implementation fingerprint")
        del prepared
        owner = importlib.import_module(
            "collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4")
        record("waiting_for_exclusive_gpu", source_proof=proof,
               implementation_fingerprint=fingerprint)
        while not STOP.is_set():
            gpu, uuid, _, lease = owner._acquire(STOP)
            # The production lease helper requires all four parameters.
            stable = _wait_for_stable_idle(owner, gpu, uuid, STOP)
            if stable:
                live_uuid, used, owners = owner.gpu_guard._gpu_info(gpu)
                if live_uuid == uuid and not owners and int(used) <= owner.MAX_MEMORY_MIB:
                    break
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
            lease.close()
            lease = None
        else:
            raise RuntimeError("stopped before SW0176 worker launch")
        if prep.implementation_fingerprint() != fingerprint:
            raise RuntimeError("SW0176 source files changed after CPU admission")
        command = worker_command(gpu, report)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
                   USE_PYTORCH_KERNEL_CACHE="0")
        with log.open("x", encoding="utf-8") as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                pass_fds=(lease.fileno(),))
            record("running", pid=child.pid, argv=command, physical_gpu=gpu,
                   gpu_uuid=uuid, source_proof=proof,
                   implementation_fingerprint=fingerprint)
            deadline = time.monotonic() + 1800
            while child.poll() is None and not STOP.wait(5):
                if time.monotonic() > deadline:
                    raise TimeoutError("bounded SW0176 source preflight exceeded 1800 seconds")
                live_uuid, _, owners = owner.gpu_guard._gpu_info(gpu)
                if child.poll() is not None:
                    break
                if live_uuid != uuid or any(int(pid) != child.pid for pid in owners):
                    raise RuntimeError("foreign GPU owner detected; stopping only owned worker")
            if STOP.is_set() and child.poll() is None:
                raise RuntimeError("SW0176 source preflight supervisor stopped")
            if child.returncode != 0 or not report.is_file():
                raise RuntimeError(f"SW0176 worker exited {child.returncode}; inspect immutable log")
        result = json.loads(report.read_text(encoding="utf-8"))
        if (result.get("status") != "complete_source_preflight"
                or result.get("implementation_fingerprint") != fingerprint
                or result.get("source_proof") != proof
                or result.get("optimizer_updates") != 0
                or result.get("validation_data_read") is not False
                or result.get("cuda_visible_devices") != str(gpu)):
            raise ValueError("SW0176 report does not bind the admitted source and leased worker")
        record("complete", pid=child.pid, returncode=child.returncode,
               physical_gpu=gpu, gpu_uuid=uuid,
               report_file_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
               source_proof=proof, implementation_fingerprint=fingerprint)
    except BaseException as exc:
        if child is not None and child.poll() is None and owner is not None:
            owner._stop_owned(child)
        if owned_state:
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
            parser.error("worker requires --physical-gpu and --output")
        run_worker(args.physical_gpu, args.output)
    else:
        supervise()


if __name__ == "__main__":
    main()
