"""Owner-safe retrying supervisor for the frozen SW0177 source preflight."""
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
MODULE = "collaborative_test.SW_0177_learned_coupling_polarity.resource_queue_v2"
RUNTIME = HERE / "results_archive" / "source_preflight_v2_20261016"
STATE, REPORT = RUNTIME / "state.json", RUNTIME / "preflight.json"
STOP = threading.Event()


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_state(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _link_report_bytes(data, destination):
    destination = Path(destination)
    temp = destination.with_name(destination.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temp.open("xb") as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.link(temp, destination)
    finally:
        if temp.exists():
            temp.unlink()


def _load_preflight():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        return importlib.import_module("collaborative_test.SW_0177_learned_coupling_polarity.preflight")
    finally:
        if visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible


def _load_owner():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        return importlib.import_module("collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4")
    finally:
        if visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible


def implementation_fingerprint():
    prep = _load_preflight()
    owner = _load_owner()
    return {"preflight": prep.implementation_fingerprint(),
            "resource_queue_v2.py": _sha_file(__file__),
            "stage_queue_v4.py": _sha_file(owner.__file__)}


def fingerprint_sha256():
    raw = json.dumps(implementation_fingerprint(), sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def worker_command(gpu, output):
    return [sys.executable, "-u", "-m", MODULE, "--worker", "--physical-gpu",
            str(gpu), "--output", str(output)]


def _wait_for_stable_idle(owner, gpu, uuid, stop_event, on_sample=lambda *_: None):
    return owner._stable_idle(gpu, uuid, stop_event, on_sample)


def _validate_worker_report(report, *, preflight_fingerprint, source_proof, physical_gpu):
    status = report.get("status")
    terminal = {"complete_source_preflight": "complete",
                "scientific_rejection": "complete_rejected"}.get(status)
    if (terminal is None
            or report.get("implementation_fingerprint") != preflight_fingerprint
            or report.get("source_proof") != source_proof
            or report.get("optimizer_updates") != 0
            or report.get("validation_data_read") is not False
            or report.get("cuda_visible_devices") != str(physical_gpu)):
        raise ValueError("SW0177 worker report does not bind current preflight, source, and leased GPU")
    if status == "scientific_rejection" and not report.get("rejection_reasons"):
        raise ValueError("SW0177 scientific rejection has no recorded reason")
    return terminal


def run_worker(gpu, output):
    if type(gpu) is not int or gpu not in range(4) or os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise RuntimeError("SW0177 worker visibility must match its exclusively leased physical GPU")
    prep = _load_preflight()
    prepared = prep.prepare_inputs_cpu()
    if prep.torch.cuda.is_initialized() or os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise RuntimeError("SW0177 CPU input admission initialized CUDA or changed worker visibility")
    if prep.torch.cuda.device_count() != 1:
        raise RuntimeError("SW0177 worker must see exactly its leased device")
    return prep.run_preflight(device="cuda:0", prepared=prepared, output=output)


def _write_state(path, status, **fields):
    current = {}
    if Path(path).is_file():
        current = json.loads(Path(path).read_text(encoding="utf-8"))
    current.update({"experiment": "SW0177_learned_coupling_polarity_preflight_v2",
                   "status": status, "supervisor_pid": os.getpid(),
                   "updated_unix": time.time(), **fields})
    _atomic_state(path, current)
    return current


def supervise(*, timeout_seconds=1800):
    import fcntl
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["USE_PYTORCH_KERNEL_CACHE"] = "0"
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: STOP.set())
    RUNTIME.mkdir(parents=True, exist_ok=True)
    lock = (RUNTIME / "queue.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = lease = owner = None
    owned_state = False
    state_path, report_path = STATE, REPORT

    def record(status, **fields):
        return _write_state(state_path, status, **fields)

    try:
        if state_path.exists() or report_path.exists():
            raise FileExistsError("preserving any existing SW0177 v2 runtime; inspect before retry")
        record("validating_source_cpu", attempts=[])
        owned_state = True
        prep = _load_preflight()
        prepared = prep.prepare_inputs_cpu()
        source_proof = prepared.get("source_proof")
        preflight_fingerprint = prep.implementation_fingerprint()
        if (prepared.get("implementation_fingerprint") != preflight_fingerprint
                or not isinstance(source_proof, dict)):
            raise ValueError("SW0177 CPU source admission returned stale or malformed proof")
        fp = implementation_fingerprint()
        if fp.get("preflight") != preflight_fingerprint:
            raise ValueError("SW0177 queue and source-preflight fingerprints disagree")
        del prepared
        owner = _load_owner()
        record("waiting_for_exclusive_gpu", source_proof=source_proof,
               implementation_fingerprint=preflight_fingerprint,
               queue_fingerprint=fp,
               queue_fingerprint_sha256=fingerprint_sha256())
        attempt = 0
        attempts = []
        while not STOP.is_set():
            gpu, gpu_uuid, _initial_memory, lease = owner._acquire(STOP)
            stable = _wait_for_stable_idle(owner, gpu, gpu_uuid, STOP,
                lambda elapsed, used: record("waiting_for_stable_gpu",
                    source_proof=source_proof, implementation_fingerprint=preflight_fingerprint,
                    queue_fingerprint=fp, queue_fingerprint_sha256=fingerprint_sha256(),
                    physical_gpu=int(gpu), gpu_uuid=gpu_uuid,
                    stable_seconds=float(elapsed), memory_used_mib=int(used), attempts=attempts))
            if not stable:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                continue
            live_uuid, used, owners = owner.gpu_guard._gpu_info(gpu)
            if live_uuid != gpu_uuid or owners or int(used) > owner.MAX_MEMORY_MIB:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                record("waiting_for_exclusive_gpu", source_proof=source_proof,
                    implementation_fingerprint=preflight_fingerprint, queue_fingerprint=fp,
                    queue_fingerprint_sha256=fingerprint_sha256(), physical_gpu=int(gpu),
                    gpu_uuid=live_uuid, memory_used_mib=int(used), owners=list(owners), attempts=attempts)
                continue
            if prep.implementation_fingerprint() != preflight_fingerprint:
                raise RuntimeError("SW0177 preflight source changed while waiting for the owner lease")
            attempt += 1
            attempt_dir = RUNTIME / "attempts"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            attempt_report = attempt_dir / f"preflight_attempt{attempt:02d}.json"
            attempt_log = attempt_dir / f"worker_attempt{attempt:02d}.log"
            command = worker_command(int(gpu), attempt_report)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
                USE_PYTORCH_KERNEL_CACHE="0")
            yielded = False
            with attempt_log.open("x", encoding="utf-8", newline="\n") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True, pass_fds=(lease.fileno(),))
                # The bound applies to an active owned child, never to time
                # spent waiting for a free device or yielding to another owner.
                worker_deadline = time.monotonic() + timeout_seconds
                record("running_worker", source_proof=source_proof,
                    implementation_fingerprint=preflight_fingerprint, queue_fingerprint=fp,
                    queue_fingerprint_sha256=fingerprint_sha256(), attempt=attempt,
                    child_pid=child.pid, child_argv=command, physical_gpu=int(gpu), gpu_uuid=gpu_uuid,
                    attempt_report=attempt_report.name, attempt_log=attempt_log.name, attempts=attempts)
                while child.poll() is None and not STOP.wait(2):
                    if time.monotonic() >= worker_deadline:
                        owner._stop_owned(child)
                        raise TimeoutError("SW0177 owned source-preflight worker timed out")
                    observed_uuid, _used_now, gpu_owners = owner.gpu_guard._gpu_info(gpu)
                    if child.poll() is not None:
                        break
                    foreign = (observed_uuid != gpu_uuid
                               or any(int(pid) != child.pid for pid in gpu_owners))
                    if foreign:
                        owner._stop_owned(child)
                        child.wait()
                        yielded = True
                        attempt_row = {"attempt": attempt, "status": "yielded_foreign_owner",
                            "pid": child.pid, "physical_gpu": int(gpu), "gpu_uuid": gpu_uuid,
                            "observed_uuid": observed_uuid, "owners": list(gpu_owners),
                            "returncode": child.returncode, "log": attempt_log.name,
                            "partial_report_sha256": _sha_file(attempt_report) if attempt_report.is_file() else None}
                        attempts.append(attempt_row)
                        record("yielding_foreign_owner", source_proof=source_proof,
                            implementation_fingerprint=preflight_fingerprint, queue_fingerprint=fp,
                            queue_fingerprint_sha256=fingerprint_sha256(), attempts=attempts,
                            current_attempt=attempt_row)
                        break
                    proc = owner._proc_record(child.pid)
                    if proc is None or proc.get("state") == "Z":
                        try:
                            child.wait(timeout=1.0)
                        except subprocess.TimeoutExpired:
                            pass
                        if child.poll() is not None:
                            break
                        continue
                    argv = proc.get("argv")
                    if argv and list(argv) != command:
                        owner._stop_owned(child)
                        raise RuntimeError("SW0177 source-preflight worker argv identity changed")
                if STOP.is_set() and child.poll() is None:
                    owner._stop_owned(child)
                rc = child.wait()
            if yielded:
                child = None
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                continue
            if rc != 0:
                raise RuntimeError(f"SW0177 source-preflight worker exited {rc}; inspect {attempt_log}")
            if not attempt_report.is_file():
                raise RuntimeError("SW0177 worker exited successfully without its immutable report")
            result = json.loads(attempt_report.read_text(encoding="utf-8"))
            terminal = _validate_worker_report(result, preflight_fingerprint=preflight_fingerprint,
                source_proof=source_proof, physical_gpu=int(gpu))
            _link_report_bytes(attempt_report.read_bytes(), report_path)
            attempt_row = {"attempt": attempt, "status": terminal, "pid": child.pid,
                "physical_gpu": int(gpu), "gpu_uuid": gpu_uuid, "returncode": rc,
                "report_file": attempt_report.name, "report_sha256": _sha_file(attempt_report),
                "log": attempt_log.name}
            attempts.append(attempt_row)
            record(terminal, report_status=result["status"], rejection_reasons=result.get("rejection_reasons", []),
                source_proof=source_proof, implementation_fingerprint=preflight_fingerprint,
                queue_fingerprint=fp, queue_fingerprint_sha256=fingerprint_sha256(), attempts=attempts,
                report_file_sha256=_sha_file(report_path), report_attempt=attempt_report.name,
                pid=child.pid, returncode=rc, physical_gpu=int(gpu), gpu_uuid=gpu_uuid)
            return
        raise RuntimeError("stopped before SW0177 preflight completion")
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
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close()
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN); lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.worker:
        if args.physical_gpu is None or args.output is None:
            parser.error("worker requires --physical-gpu and --output")
        run_worker(args.physical_gpu, args.output)
    else:
        supervise()


if __name__ == "__main__":
    main()
