"""Owner-safe per-arm supervisor for the bounded SW0175 training prefix."""
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

try:
    import fcntl
except ImportError:  # pragma: no cover - production queue runs on Linux
    fcntl = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNTIME = HERE / "results_archive" / "training_queue_v2_20261015"
TRAINER_MODULE = "collaborative_test.SW_0175_critical_component_binding.train_v2"
OWNER_MODULE = "collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4"
_stop = threading.Event()


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
    trainer = _load_preserving_visibility(TRAINER_MODULE)
    return trainer, _load_preserving_visibility(OWNER_MODULE)


def _admit(trainer=None):
    if trainer is None:
        trainer, _owner = _runtime_modules()
    admitted = trainer._load_admitted_inputs()
    return trainer, admitted


def _fingerprint(trainer, admitted):
    values = {
        "experiment": "SW0175_training_queue_v2",
        "queue_sha256": _sha(__file__),
        "trainer_file_sha256": _sha(HERE / "train_v2.py"),
        "trainer_fingerprint": trainer._training_fingerprint(),
        "admission_queue_fingerprint_sha256": admitted["queue_fingerprint_sha256"],
        "preflight_fingerprint": admitted["preflight_fingerprint"],
        "admission_report_sha256": admitted["admission_report_sha256"],
        "source_proof": admitted["proof"],
    }
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return values, hashlib.sha256(payload).hexdigest()


def _state_path(arm):
    return RUNTIME / f"{arm}_state.json"


def _log_path(arm, attempt):
    return RUNTIME / f"{arm}_worker_attempt{attempt:02d}.log"


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _write_state(arm, status, **fields):
    path = _state_path(arm)
    state = {}
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
    state.update({"experiment": "SW0175_training_queue_v2", "arm": arm,
                  "supervisor_pid": os.getpid(), "status": status,
                  "updated": time.time(), **fields})
    _atomic_json(path, state)
    return state


def worker_command(arm, gpu):
    trainer, _owner = _runtime_modules()
    if arm not in trainer.ARMS or type(gpu) is not int or gpu not in range(4):
        raise ValueError("SW0175 worker command requires a registered arm and physical GPU")
    return [sys.executable, "-m", TRAINER_MODULE.rsplit(".", 1)[0] + ".training_queue_v2",
            "--worker", "--arm", arm, "--physical-gpu", str(gpu)]


def _worker(arm, physical_gpu):
    trainer, _owner = _runtime_modules()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("SW0175 worker visibility differs from its leased physical GPU")
    result = trainer.train(arm, physical_gpu=physical_gpu)
    if result.get("status") == "scientific_rejection":
        printable = {key: result[key] for key in
                     ("status", "arm", "completed_updates", "rejection_reasons")}
    else:
        printable = {"status": result.get("status", "training_complete"), "arm": arm,
                     "completed_updates": result.get("completed_updates", trainer.UPDATES)}
    print(json.dumps(printable, sort_keys=True))
    return result


def _validate_admission_snapshot(trainer, current, first, fp):
    _values, current_fp = _fingerprint(trainer, current)
    if current_fp != fp:
        raise RuntimeError("SW0175 source, admission, trainer, or queue fingerprint changed")
    for key in ("proof", "queue_fingerprint_sha256", "preflight_fingerprint",
                "admission_report_sha256"):
        if current.get(key) != first.get(key):
            raise RuntimeError(f"SW0175 terminal admission changed while waiting: {key}")


def _validate_training_result(trainer, arm, admitted):
    output = Path(trainer.OUTPUT_ROOT) / f"{arm}_seed1"
    manifest = trainer.validate_training_artifact(arm, output)
    bindings = manifest.get("bindings", {})
    checks = {
        "admission_report_sha256": admitted["admission_report_sha256"],
        "admission_queue_fingerprint_sha256": admitted["queue_fingerprint_sha256"],
        "preflight_implementation_fingerprint": admitted["preflight_fingerprint"],
        "source_proof": admitted["proof"],
        "source_state_sha256": admitted["proof"].get("source_cache_state_sha256"),
        "training_fingerprint": trainer._training_fingerprint(),
        
    }
    if any(bindings.get(key) != value for key, value in checks.items()):
        raise ValueError("SW0175 completed training does not bind the current admitted source")
    return manifest, _sha(output / "manifest.json")


def _validate_rejection(trainer, arm, admitted):
    runtime = _load_preserving_visibility("collaborative_test.SW_0148_checkpointable_boundary_q.checkpoint")
    work = Path(trainer.OUTPUT_ROOT) / "work" / f"{arm}_seed1"
    metadata = runtime.inspect_committed_metadata(work)
    # The checkpoint API intentionally exposes only small metadata. Read the
    # serialized payload after its container/metadata validation, then bind
    # the history to the same committed cursor and bindings.
    import io
    import torch
    raw = (work / "boundary.ptx").read_bytes()
    outer = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    payload_bytes = outer.get("payload_bytes")
    if not isinstance(payload_bytes, bytes) or hashlib.sha256(payload_bytes).hexdigest() != outer.get("payload_sha256"):
        raise ValueError("SW0175 rejection boundary payload hash mismatch")
    payload = torch.load(io.BytesIO(payload_bytes), map_location="cpu", weights_only=False)
    if (payload.get("completed_updates") != metadata.get("completed_updates")
            or payload.get("bindings") != metadata.get("bindings")
            or payload.get("runtime") != metadata.get("runtime")
            or payload.get("history") is None
            or len(payload["history"]) != payload.get("completed_updates")):
        raise ValueError("SW0175 rejection boundary payload differs from verified envelope")
    history = payload.get("history")
    bindings = payload.get("bindings")
    if not isinstance(history, list) or not isinstance(bindings, dict):
        raise ValueError("SW0175 rejection lacks committed boundary/history")
    report = trainer._read_credit_rejection(work, arm=arm,
        bindings=bindings, completed=metadata.get("completed_updates"), history=history)
    if (report is None or report.get("source_proof") != admitted["proof"]
            or report.get("training_fingerprint") != trainer._training_fingerprint()
            or bindings.get("admission_report_sha256") != admitted["admission_report_sha256"]):
        raise ValueError("SW0175 scientific rejection is not bound to current admission")
    return report, _sha(work / "credit_rejection.json")


def _proc_matches(owner, pid, command):
    row = owner._proc_record(pid)
    if row is None or row.get("state") == "Z":
        return False
    return row.get("argv") == command


def run(arm, *, timeout_seconds=8 * 3600):
    trainer, _ = _runtime_modules()
    if arm not in trainer.ARMS:
        raise ValueError("unregistered SW0175 training arm")
    if fcntl is None:
        raise RuntimeError("SW0175 owner queue requires Linux flock")
    signal.signal(signal.SIGTERM, lambda *_: _stop.set())
    signal.signal(signal.SIGINT, lambda *_: _stop.set())
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("SW0175 queue supervisor must keep CUDA hidden")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    state_path = _state_path(arm)
    if state_path.exists() or any(RUNTIME.glob(f"{arm}_worker_attempt*.log")):
        raise FileExistsError(f"preserving existing SW0175 {arm} queue state/logs")
    lock = (RUNTIME / f"{arm}.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = lease = owner = None
    attempt = 0
    try:
        _write_state(arm, "validating_admission")
        trainer, admitted = _admit(trainer)
        fingerprint, fp = _fingerprint(trainer, admitted)
        if not admitted["proof"] or admitted["admission_report"].get("training_admission") is not True:
            raise ValueError("SW0175 training queue requires terminal admitted preflight")
        _write_state(arm, "waiting_for_exclusive_gpu",
            queue_fingerprint=fingerprint, queue_fingerprint_sha256=fp,
            admission_report_sha256=admitted["admission_report_sha256"],
            source_proof=admitted["proof"], validation_data_read=False)
        owner = _load_preserving_visibility(OWNER_MODULE)
        deadline = time.monotonic() + timeout_seconds
        while not _stop.is_set():
            if time.monotonic() >= deadline:
                raise TimeoutError("SW0175 bounded training queue timed out")
            gpu, gpu_uuid, initial_memory, lease = owner._acquire(_stop)
            stable = owner._stable_idle(gpu, gpu_uuid, _stop,
                lambda elapsed, used: _write_state(arm, "waiting_for_stable_gpu",
                    physical_gpu=gpu, gpu_uuid=gpu_uuid, stable_seconds=float(elapsed),
                    memory_used_mib=int(used), queue_fingerprint_sha256=fp))
            if not stable:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                continue
            # Heavy source/cache admission is revalidated under the exclusive lease.
            trainer2, current = _admit(trainer)
            _validate_admission_snapshot(trainer2, current, admitted, fp)
            actual_uuid, used, owners = owner.gpu_guard._gpu_info(gpu)
            if actual_uuid != gpu_uuid or owners or int(used) > owner.MAX_MEMORY_MIB:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                _write_state(arm, "waiting_for_exclusive_gpu", physical_gpu=gpu,
                    gpu_uuid=actual_uuid, memory_used_mib=int(used), owners=list(owners),
                    queue_fingerprint_sha256=fp)
                continue
            attempt += 1
            command = worker_command(arm, int(gpu))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
                       USE_PYTORCH_KERNEL_CACHE="0")
            log_path = _log_path(arm, attempt)
            retry_after_owner_change = False
            with log_path.open("x", encoding="utf-8", newline="\n") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    pass_fds=(lease.fileno(),))
                _write_state(arm, "running_training", child_pid=child.pid, child_pgid=child.pid,
                    child_argv=command, physical_gpu=int(gpu), gpu_uuid=gpu_uuid,
                    preflight_report_sha256=admitted["admission_report_sha256"],
                    queue_fingerprint_sha256=fp, attempt=attempt)
                while child.poll() is None and not _stop.is_set():
                    if time.monotonic() >= deadline:
                        owner._stop_owned(child)
                        raise TimeoutError("SW0175 owned training worker timed out")
                    live_uuid, used_now, gpu_owners = owner.gpu_guard._gpu_info(gpu)
                    if child.poll() is not None:
                        break
                    if live_uuid != gpu_uuid or any(int(pid) != child.pid for pid in gpu_owners):
                        owner._stop_owned(child)
                        retry_after_owner_change = True
                        _write_state(arm, "yielding_foreign_gpu", child_pid=child.pid,
                            physical_gpu=int(gpu), gpu_uuid=gpu_uuid,
                            observed_gpu_uuid=live_uuid, owners=list(gpu_owners), attempt=attempt)
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
                    if not _proc_matches(owner, child.pid, command):
                        owner._stop_owned(child)
                        raise RuntimeError("SW0175 owned child argv changed")
                    _write_state(arm, "running_training", child_pid=child.pid,
                        physical_gpu=int(gpu), gpu_uuid=gpu_uuid,
                        memory_used_mib=int(used_now), attempt=attempt,
                        preflight_report_sha256=admitted["admission_report_sha256"],
                        queue_fingerprint_sha256=fp)
                    time.sleep(3)
                if _stop.is_set() and child.poll() is None:
                    owner._stop_owned(child)
                rc = child.wait()
            if retry_after_owner_change:
                child = None
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                continue
            if rc != 0:
                raise RuntimeError(f"SW0175 training worker exited {rc}; inspect {log_path}")
            output = Path(trainer.OUTPUT_ROOT) / f"{arm}_seed1"
            work = Path(trainer.OUTPUT_ROOT) / "work" / f"{arm}_seed1"
            if output.is_dir():
                manifest, artifact_sha = _validate_training_result(trainer, arm, admitted)
                terminal = "complete"
                terminal_fields = {"manifest_sha256": artifact_sha,
                    "completed_updates": manifest["completed_updates"],
                    "output": str(output)}
            elif (work / "credit_rejection.json").is_file():
                rejection, artifact_sha = _validate_rejection(trainer, arm, admitted)
                terminal = "complete_rejected"
                terminal_fields = {"rejection_sha256": artifact_sha,
                    "completed_updates": rejection["completed_updates"],
                    "rejection_reasons": rejection["rejection_reasons"]}
            else:
                raise FileNotFoundError("SW0175 worker exited successfully without training completion or rejection receipt")
            _write_state(arm, terminal, **terminal_fields, child_pid=child.pid,
                physical_gpu=int(gpu), gpu_uuid=gpu_uuid, attempt=attempt,
                queue_fingerprint_sha256=fp,
                preflight_report_sha256=admitted["admission_report_sha256"],
                validation_data_read=False, finished=time.time())
            return {"status": terminal, "artifact_sha256": artifact_sha,
                    "arm": arm, "queue_fingerprint_sha256": fp,
                    "admission_report_sha256": admitted["admission_report_sha256"],
                    **terminal_fields}
        raise RuntimeError("SW0175 queue stopped before terminal arm result")
    except BaseException as exc:
        if child is not None and child.poll() is None and owner is not None:
            owner._stop_owned(child)
        _write_state(arm, "failed", error=repr(exc), finished=time.time())
        raise
    finally:
        if child is not None and child.poll() is None and owner is not None:
            owner._stop_owned(child)
        if lease is not None:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close()
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN); lock.close()


def main(argv=None):
    import argparse
    trainer, _owner = _runtime_modules()
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--arm", choices=trainer.ARMS, required=True)
    parser.add_argument("--physical-gpu", type=int)
    args = parser.parse_args(argv)
    if args.worker:
        if args.physical_gpu not in range(4):
            parser.error("worker requires an assigned physical GPU")
        _worker(args.arm, args.physical_gpu)
        return 0
    result = run(args.arm)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
