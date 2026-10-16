"""Exclusive-GPU supervisor for the single SW0176 local-phase training arm."""
from __future__ import annotations

import hashlib
import importlib
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
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("USE_PYTORCH_KERNEL_CACHE", "0")

try:
    import fcntl
except ImportError:  # pragma: no cover - production worker is Linux
    fcntl = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODULE = "collaborative_test.SW_0176_local_phase_credit.training_queue"
TRAINER_MODULE = "collaborative_test.SW_0176_local_phase_credit.train"
PREFLIGHT_MODULE = "collaborative_test.SW_0176_local_phase_credit.preflight_v4"
OWNER_MODULE = "collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4"
RUNTIME = HERE / "results_archive" / "training_queue_v1_20261016"
STOP = threading.Event()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_preserving_visibility(module_name):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        return importlib.import_module(module_name)
    finally:
        if visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible


def _runtime_modules():
    return (_load_preserving_visibility(TRAINER_MODULE),
            _load_preserving_visibility(OWNER_MODULE))


def _write_json(path, value, *, create_only=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        if create_only:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _state_path():
    return RUNTIME / "state.json"


def _log_path(attempt):
    return RUNTIME / f"worker_attempt{attempt:02d}.log"


def _fingerprint(trainer, admitted):
    proof = admitted["source_proof"]
    fields = {
        "experiment": trainer.EXPERIMENT,
        "queue_module": MODULE,
        "queue_sha256": _sha(__file__),
        "trainer_fingerprint": trainer.implementation_fingerprint(),
        "preflight_fingerprint": admitted["preflight_fingerprint"],
        "preflight_report_sha256": admitted["preflight_report_sha256"],
        "preflight_state_sha256": admitted["preflight_state_sha256"],
        "control_receipt_sha256": admitted["control_receipt"]["manifest_sha256"],
        "source_proof": proof,
        "validation_data_read": False,
    }
    encoded = json.dumps(fields, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return fields, hashlib.sha256(encoded).hexdigest()


def _write_state(status, **fields):
    current = {}
    path = _state_path()
    if path.is_file():
        current = json.loads(path.read_text(encoding="utf-8"))
    current.update({"experiment": "SW0176_local_phase_training_queue",
        "arm": "local_phase", "supervisor_pid": os.getpid(),
        "status": status, "updated_unix": time.time(), **fields})
    _write_json(path, current)
    return current


def worker_command(gpu):
    if type(gpu) is not int or gpu not in range(4):
        raise ValueError("SW0176 worker requires a physical GPU index in [0,3]")
    return [sys.executable, "-u", "-m", MODULE, "--worker",
            "--physical-gpu", str(gpu)]


def _worker(gpu):
    trainer, _owner = _runtime_modules()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise RuntimeError("SW0176 worker visibility differs from its leased physical GPU")
    admitted = trainer.prepare_training_inputs_cpu()
    cpu_receipt = trainer.prepare_fresh_training_state_cpu(admitted)
    if (cpu_receipt.get("status") != "complete_fresh_cpu_factory"
            or trainer.torch.cuda.is_initialized()
            or os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu)):
        raise RuntimeError("SW0176 worker CPU factory failed before CUDA initialization")
    result = trainer.run_training(physical_gpu=gpu, admitted=admitted)
    status = result.get("status")
    if status == "training_complete":
        message = {"status": status, "arm": trainer.ARM,
                   "completed_updates": trainer.UPDATES}
    elif status == "scientific_rejection":
        message = {"status": status, "arm": trainer.ARM,
                   "completed_updates": result.get("completed_updates"),
                   "rejection_reasons": result.get("rejection_reasons", [])}
    else:
        raise ValueError("SW0176 trainer returned an unregistered terminal status")
    print(json.dumps(message, sort_keys=True), flush=True)
    return result


def _proc_matches(owner, pid, command):
    row = owner._proc_record(pid)
    if row is None or row.get("state") == "Z":
        return False
    return row.get("argv") == command


def _validate_rejection(trainer, admitted):
    import io
    import torch
    work = Path(trainer.OUTPUT_ROOT) / "work" / f"{trainer.ARM}_seed1"
    boundary = _load_preserving_visibility(
        "collaborative_test.SW_0148_checkpointable_boundary_q.checkpoint")
    metadata = boundary.inspect_committed_metadata(work)
    raw = (work / "boundary.ptx").read_bytes()
    outer = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    payload_bytes = outer.get("payload_bytes")
    if (not isinstance(payload_bytes, bytes)
            or hashlib.sha256(payload_bytes).hexdigest() != outer.get("payload_sha256")):
        raise ValueError("SW0176 rejection boundary payload hash mismatch")
    payload = torch.load(io.BytesIO(payload_bytes), map_location="cpu", weights_only=False)
    if (payload.get("completed_updates") != metadata.get("completed_updates")
            or payload.get("bindings") != metadata.get("bindings")
            or payload.get("history") is None
            or len(payload["history"]) != payload.get("completed_updates")):
        raise ValueError("SW0176 rejection payload differs from verified boundary metadata")
    bindings, history = payload["bindings"], payload["history"]
    report = trainer._validate_rejection(work, completed=payload["completed_updates"],
                                         history=history, bindings=bindings)
    if (report is None or bindings.get("source_proof") != admitted["source_proof"]
            or bindings.get("preflight_report_sha256") != admitted["preflight_report_sha256"]):
        raise ValueError("SW0176 scientific rejection does not bind current admission")
    return report


def supervise(*, timeout_seconds=8 * 3600):
    if fcntl is None:
        raise RuntimeError("SW0176 owner queue requires Linux flock")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("SW0176 supervisor must keep CUDA hidden")
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    RUNTIME.mkdir(parents=True, exist_ok=True)
    state_path = _state_path()
    if state_path.exists() or any(RUNTIME.glob("worker_attempt*.log")):
        raise FileExistsError("preserving existing SW0176 training queue state/logs")
    lock = (RUNTIME / "queue.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = owner = lease = None
    owned_state = False
    attempt = 0
    try:
        _write_state("validating_source_cpu")
        owned_state = True
        trainer, owner = _runtime_modules()
        admitted = trainer.prepare_training_inputs_cpu()
        fields, fp = _fingerprint(trainer, admitted)
        if not admitted["source_proof"]:
            raise ValueError("SW0176 queue requires a validated source proof")
        preflight = admitted["preflight_report"]
        if preflight.get("status") == "scientific_rejection":
            _write_state("complete_rejected", queue_fingerprint=fields,
                queue_fingerprint_sha256=fp, source_proof=admitted["source_proof"],
                preflight_report_sha256=admitted["preflight_report_sha256"],
                rejection_reasons=preflight.get("rejection_reasons", []),
                optimizer_updates=0, validation_data_read=False)
            return {"status": "complete_rejected", "rejection_reasons": preflight.get("rejection_reasons", [])}
        if preflight.get("status") != "complete_source_preflight":
            raise ValueError("SW0176 preflight is not terminally admitted")
        del admitted
        _write_state("waiting_for_exclusive_gpu", queue_fingerprint=fields,
            queue_fingerprint_sha256=fp, source_proof=fields["source_proof"],
            preflight_report_sha256=fields["preflight_report_sha256"],
            validation_data_read=False)
        deadline = time.monotonic() + timeout_seconds
        while not STOP.is_set():
            if time.monotonic() >= deadline:
                raise TimeoutError("SW0176 exclusive GPU wait exceeded timeout")
            gpu, gpu_uuid, _, lease = owner._acquire(STOP)
            stable = owner._stable_idle(gpu, gpu_uuid, STOP,
                lambda elapsed, used: _write_state("waiting_for_stable_gpu",
                    physical_gpu=int(gpu), gpu_uuid=gpu_uuid, stable_seconds=float(elapsed),
                    memory_used_mib=int(used), queue_fingerprint_sha256=fp))
            if not stable:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                continue
            live_uuid, used, owners = owner.gpu_guard._gpu_info(gpu)
            if live_uuid != gpu_uuid or owners or int(used) > owner.MAX_MEMORY_MIB:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                _write_state("waiting_for_exclusive_gpu", physical_gpu=int(gpu),
                    gpu_uuid=live_uuid, owners=list(owners), memory_used_mib=int(used),
                    queue_fingerprint_sha256=fp)
                continue
            if trainer.implementation_fingerprint() != fields["trainer_fingerprint"]:
                raise RuntimeError("SW0176 trainer source changed after CPU admission")
            break
        else:
            raise RuntimeError("SW0176 queue stopped before worker launch")

        attempt = 1
        command = worker_command(int(gpu))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", USE_PYTORCH_KERNEL_CACHE="0")
        with _log_path(attempt).open("x", encoding="utf-8", newline="\n") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                pass_fds=(lease.fileno(),))
            _write_state("running_training", child_pid=child.pid, child_pgid=child.pid,
                child_argv=command, physical_gpu=int(gpu), gpu_uuid=gpu_uuid,
                queue_fingerprint_sha256=fp, preflight_report_sha256=fields["preflight_report_sha256"],
                source_proof=fields["source_proof"], attempt=attempt)
            deadline = time.monotonic() + timeout_seconds
            while child.poll() is None and not STOP.wait(3):
                if time.monotonic() >= deadline:
                    owner._stop_owned(child)
                    raise TimeoutError("SW0176 owned training process exceeded timeout")
                live_uuid, used_now, gpu_owners = owner.gpu_guard._gpu_info(gpu)
                if child.poll() is not None:
                    break
                if live_uuid != gpu_uuid or any(int(pid) != child.pid for pid in gpu_owners):
                    owner._stop_owned(child)
                    raise RuntimeError("foreign GPU owner detected; owned worker stopped")
                proc = owner._proc_record(child.pid)
                if proc is None or proc.get("state") == "Z":
                    try:
                        child.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        pass
                    if child.poll() is not None:
                        break
                    continue
                if not _proc_matches(owner, child.pid, command):
                    owner._stop_owned(child)
                    raise RuntimeError("SW0176 owned child argv changed")
                _write_state("running_training", child_pid=child.pid,
                    physical_gpu=int(gpu), gpu_uuid=gpu_uuid,
                    memory_used_mib=int(used_now), attempt=attempt,
                    queue_fingerprint_sha256=fp)
            if STOP.is_set() and child.poll() is None:
                owner._stop_owned(child)
            rc = child.wait()
        if rc != 0:
            raise RuntimeError(f"SW0176 worker exited {rc}; inspect immutable worker log")
        output = Path(trainer.OUTPUT_ROOT) / f"{trainer.ARM}_seed1"
        work = Path(trainer.OUTPUT_ROOT) / "work" / f"{trainer.ARM}_seed1"
        if output.is_dir():
            manifest = trainer.validate_training_artifact(output)
            terminal = "complete"
            artifact_sha = _sha(output / "manifest.json")
            fields_out = {"manifest_sha256": artifact_sha,
                "completed_updates": manifest["completed_updates"], "output": str(output)}
        elif (work / "credit_rejection.json").is_file():
            rejection = _validate_rejection(trainer, fields)
            terminal = "complete_rejected"
            artifact_sha = _sha(work / "credit_rejection.json")
            fields_out = {"rejection_sha256": artifact_sha,
                "completed_updates": rejection["completed_updates"],
                "rejection_reasons": rejection["rejection_reasons"]}
        else:
            raise FileNotFoundError("SW0176 worker has no final artifact or terminal rejection")
        _write_state(terminal, **fields_out, child_pid=child.pid,
            physical_gpu=int(gpu), gpu_uuid=gpu_uuid, attempt=attempt,
            queue_fingerprint_sha256=fp, preflight_report_sha256=fields["preflight_report_sha256"],
            source_proof=fields["source_proof"], validation_data_read=False,
            finished_unix=time.time())
        return {"status": terminal, "artifact_sha256": artifact_sha,
            "queue_fingerprint_sha256": fp, **fields_out}
    except BaseException as exc:
        if child is not None and child.poll() is None and owner is not None:
            owner._stop_owned(child)
        if owned_state:
            _write_state("failed", error=repr(exc), finished_unix=time.time())
        raise
    finally:
        if child is not None and child.poll() is None and owner is not None:
            owner._stop_owned(child)
        if lease is not None:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close()
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN); lock.close()


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--physical-gpu", type=int)
    args = parser.parse_args(argv)
    if args.worker:
        if args.physical_gpu not in range(4):
            parser.error("worker requires an assigned physical GPU")
        _worker(args.physical_gpu)
    else:
        result = supervise()
        print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
