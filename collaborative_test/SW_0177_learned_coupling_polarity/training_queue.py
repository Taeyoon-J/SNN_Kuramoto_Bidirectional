"""Owner-exclusive, resumable supervisor for SW0177's matched training arms."""
from __future__ import annotations

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

_WORKER = "--worker" in sys.argv
if not _WORKER:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ["USE_PYTORCH_KERNEL_CACHE"] = "0"

try:
    import fcntl
except ImportError:  # pragma: no cover - workers execute on Linux
    fcntl = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODULE = "collaborative_test.SW_0177_learned_coupling_polarity.training_queue"
TRAINER_MODULE = "collaborative_test.SW_0177_learned_coupling_polarity.train"
OWNER_MODULE = "collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4"
PREFLIGHT_QUEUE = "collaborative_test.SW_0177_learned_coupling_polarity.resource_queue_v2"
RUNTIME = HERE / "results_archive" / "training_queue_v1_20261016"
_stop = threading.Event()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(name):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        return importlib.import_module(name)
    finally:
        if visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible


def _modules():
    return _load(TRAINER_MODULE), _load(OWNER_MODULE), _load(PREFLIGHT_QUEUE)


def worker_command(arm, gpu):
    trainer = _load(TRAINER_MODULE)
    if arm not in trainer.ARMS or type(gpu) is not int or gpu not in range(4):
        raise ValueError("SW0177 worker requires a registered arm and physical GPU")
    return [sys.executable, "-u", "-m", MODULE, "--worker", "--arm", arm,
            "--physical-gpu", str(gpu)]


def _state_path(arm):
    return RUNTIME / f"{arm}_state.json"


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with tmp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists(): tmp.unlink()


def _write_state(arm, status, **fields):
    path = _state_path(arm)
    state = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    state.update({"experiment": "SW0177_learned_coupling_polarity_training_queue",
        "arm": arm, "status": status, "supervisor_pid": os.getpid(),
        "updated_unix": time.time(), **fields})
    _atomic_json(path, state)
    return state


def _preflight_parent_is_live(owner, state):
    pid = state.get("supervisor_pid")
    if type(pid) is not int or pid <= 0:
        raise RuntimeError("pending SW0177 v2 preflight state lacks supervisor PID")
    proc = owner._proc_record(pid)
    if proc is None or proc.get("state") == "Z":
        raise RuntimeError("pending SW0177 v2 preflight supervisor is not live")
    argv = proc.get("argv")
    if argv and ("-m" not in argv or PREFLIGHT_QUEUE not in argv):
        raise RuntimeError("pending SW0177 v2 preflight PID has the wrong module argv")
    if not argv:
        raise RuntimeError("pending SW0177 v2 preflight supervisor argv is unavailable")
    return True


def _wait_for_successful_preflight(queue, owner, *, deadline, arm):
    """Wait only on v2's runtime; v1 terminal failure is never consulted."""
    state_path, report_path = queue.STATE, queue.REPORT
    while not _stop.is_set():
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("SW0177 training queue timed out waiting for v2 preflight")
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            status = state.get("status")
            if status == "complete":
                if not report_path.is_file():
                    raise FileNotFoundError("v2 preflight state is complete without its report")
                return state
            if status == "complete_rejected":
                raise RuntimeError("SW0177 v2 source preflight scientifically rejected the candidate")
            if status == "failed":
                raise RuntimeError("SW0177 v2 source preflight failed operationally")
            _preflight_parent_is_live(owner, state)
        _write_state(arm, "waiting_for_v2_source_preflight",
                     v2_state_path=str(state_path), validation_data_read=False)
        _stop.wait(2.0)
    raise RuntimeError("SW0177 training supervisor stopped while waiting for source admission")


def _admission_snapshot(trainer):
    admitted = trainer._load_admitted_inputs()
    receipt = admitted["v2_receipt"]
    if not receipt.get("source_proof") or receipt.get("report", {}).get("status") != "complete_source_preflight":
        raise ValueError("SW0177 trainer did not receive successful v2 source admission")
    return admitted


def _queue_fingerprint(trainer, admitted, owner):
    body = {"experiment": "SW0177_training_queue_v1",
        "queue_module_sha256": _sha(__file__),
        "trainer_sha256": _sha(HERE / "train.py"),
        "trainer_fingerprint": trainer._training_fingerprint(),
        "v2_preflight_queue_fingerprint_sha256": admitted["v2_receipt"]["queue_fingerprint_sha256"],
        "preflight_implementation_fingerprint": admitted["v2_receipt"]["preflight_fingerprint"],
        "preflight_report_sha256": admitted["v2_receipt"]["report_sha256"],
        "source_proof": admitted["proof"],
        "owner_sha256": _sha(owner.__file__)}
    data = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return body, hashlib.sha256(data).hexdigest()


def _worker(arm, gpu):
    trainer = _load(TRAINER_MODULE)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise RuntimeError("SW0177 worker visibility differs from its assigned physical GPU")
    result = trainer.train(arm, physical_gpu=gpu)
    if result.get("status") == "scientific_rejection":
        print(json.dumps({"status": "scientific_rejection", "arm": arm,
            "completed_updates": result.get("completed_updates"),
            "rejection_reasons": result.get("rejection_reasons", [])}, sort_keys=True))
    else:
        print(json.dumps({"status": "training_complete", "arm": arm,
                          "completed_updates": trainer.UPDATES}, sort_keys=True))
    return result


def _validate_result(trainer, arm, admitted):
    root = Path(trainer.OUTPUT_ROOT) / f"{arm}_seed1"
    manifest = trainer.validate_training_artifact(arm, root)
    bindings = manifest.get("bindings", {})
    receipt = admitted["v2_receipt"]
    expected = {"v2_preflight_report_sha256": receipt["report_sha256"],
        "v2_queue_fingerprint_sha256": receipt["queue_fingerprint_sha256"],
        "preflight_implementation_fingerprint": receipt["preflight_fingerprint"],
        "source_proof": admitted["proof"], "training_fingerprint": trainer._training_fingerprint()}
    if any(bindings.get(key) != value for key, value in expected.items()):
        raise ValueError("SW0177 completed training does not bind successful v2 preflight")
    return manifest


def run(arm, *, timeout_seconds=12 * 3600):
    trainer = _load(TRAINER_MODULE)
    if arm not in trainer.ARMS:
        raise ValueError("unregistered SW0177 training arm")
    if fcntl is None:
        raise RuntimeError("SW0177 owner queue requires Linux flock")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("SW0177 queue supervisor must keep CUDA hidden")
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: _stop.set())
    RUNTIME.mkdir(parents=True, exist_ok=True)
    state_path = _state_path(arm)
    if (state_path.exists() or list(RUNTIME.glob(f"{arm}_attempt*.log"))
            or (Path(trainer.OUTPUT_ROOT) / f"{arm}_seed1").exists()):
        raise FileExistsError("preserving prior SW0177 arm queue state/attempt logs")
    lock = (RUNTIME / f"{arm}.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = lease = owner = None
    own_state = False
    attempts = []
    attempt = 0
    resume_gpu = None
    try:
        own_state = True
        _write_state(arm, "waiting_for_v2_source_preflight", validation_data_read=False)
        queue = _load(PREFLIGHT_QUEUE)
        owner = _load(OWNER_MODULE)
        preflight_state = _wait_for_successful_preflight(queue, owner,
            deadline=None, arm=arm)
        admitted = _admission_snapshot(trainer)
        qfp, fp_sha = _queue_fingerprint(trainer, admitted, owner)
        _write_state(arm, "waiting_for_exclusive_gpu", v2_preflight_state_sha256=_sha(queue.STATE),
            v2_preflight_report_sha256=admitted["v2_receipt"]["report_sha256"],
            v2_queue_fingerprint_sha256=admitted["v2_receipt"]["queue_fingerprint_sha256"],
            source_proof=admitted["proof"], queue_fingerprint=qfp,
            queue_fingerprint_sha256=fp_sha, attempts=attempts, validation_data_read=False)
        while not _stop.is_set():
            # Checkpoint runtime signatures bind the physical visible device.
            # After an owner yield, resume only on that same device.
            gpu, gpu_uuid, _, lease = owner._acquire(_stop, preferred_gpu=resume_gpu)
            stable = owner._stable_idle(gpu, gpu_uuid, _stop,
                lambda elapsed, used: _write_state(arm, "waiting_for_stable_gpu",
                    physical_gpu=int(gpu), gpu_uuid=gpu_uuid, stable_seconds=float(elapsed),
                    memory_used_mib=int(used), queue_fingerprint_sha256=fp_sha,
                    attempts=attempts))
            if not stable:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                continue
            current = _admission_snapshot(trainer)
            current_qfp, current_sha = _queue_fingerprint(trainer, current, owner)
            if current_sha != fp_sha or current_qfp != qfp:
                raise RuntimeError("SW0177 trainer/source admission changed while waiting")
            actual_uuid, used, owners = owner.gpu_guard._gpu_info(gpu)
            if actual_uuid != gpu_uuid or owners or int(used) > owner.MAX_MEMORY_MIB:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                _write_state(arm, "waiting_for_exclusive_gpu", physical_gpu=int(gpu),
                    gpu_uuid=actual_uuid, memory_used_mib=int(used), owners=list(owners),
                    queue_fingerprint_sha256=fp_sha, attempts=attempts)
                continue
            attempt += 1
            command = [sys.executable, "-u", "-m", MODULE, "--worker", "--arm", arm,
                       "--physical-gpu", str(int(gpu))]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", USE_PYTORCH_KERNEL_CACHE="0")
            log_path = RUNTIME / f"{arm}_attempt{attempt:02d}.log"
            yielded = False
            with log_path.open("x", encoding="utf-8", newline="\n") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    pass_fds=(lease.fileno(),))
                worker_deadline = time.monotonic() + timeout_seconds
                _write_state(arm, "running_worker", queue_fingerprint_sha256=fp_sha,
                    child_pid=child.pid, child_argv=command, physical_gpu=int(gpu), gpu_uuid=gpu_uuid,
                    attempt=attempt, attempts=attempts, worker_log=log_path.name)
                while child.poll() is None and not _stop.wait(5):
                    if time.monotonic() >= worker_deadline:
                        owner._stop_owned(child)
                        raise TimeoutError("SW0177 owned training child exceeded active runtime")
                    observed_uuid, used_now, gpu_owners = owner.gpu_guard._gpu_info(gpu)
                    if child.poll() is not None:
                        break
                    if observed_uuid != gpu_uuid or any(int(pid) != child.pid for pid in gpu_owners):
                        owner._stop_owned(child)
                        child.wait()
                        attempts.append({"attempt": attempt, "status": "yielded_foreign_owner",
                            "physical_gpu": int(gpu), "gpu_uuid": gpu_uuid,
                            "observed_uuid": observed_uuid, "owners": list(gpu_owners),
                            "returncode": child.returncode, "log": log_path.name})
                        _write_state(arm, "yielding_foreign_owner", attempts=attempts,
                            queue_fingerprint_sha256=fp_sha)
                        yielded = True
                        break
                    proc = owner._proc_record(child.pid)
                    if proc is None or proc.get("state") == "Z":
                        try: child.wait(timeout=1.0)
                        except subprocess.TimeoutExpired: pass
                        if child.poll() is not None: break
                        continue
                    argv = proc.get("argv")
                    if argv and list(argv) != command:
                        owner._stop_owned(child)
                        raise RuntimeError("SW0177 training child argv identity changed")
                if _stop.is_set() and child.poll() is None:
                    owner._stop_owned(child)
                rc = child.wait()
            if yielded:
                boundary = Path(trainer.OUTPUT_ROOT) / "work" / f"{arm}_seed1" / "boundary.ptx"
                if boundary.is_file():
                    resume_gpu = int(gpu)
                    attempts[-1]["resume_gpu"] = resume_gpu
                child = None
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                continue
            if rc != 0:
                raise RuntimeError(f"SW0177 owned training worker exited {rc}; inspect {log_path}")
            rejection_work = Path(trainer.OUTPUT_ROOT) / "work" / f"{arm}_seed1"
            rejection_path = rejection_work / "credit_rejection.json"
            if rejection_path.is_file():
                rejection = trainer.validate_training_rejection(arm, rejection_work)
                attempts.append({"attempt": attempt, "status": "scientific_rejection",
                    "physical_gpu": int(gpu), "gpu_uuid": gpu_uuid,
                    "returncode": rc, "log": log_path.name,
                    "rejection_sha256": _sha(rejection_path)})
                _write_state(arm, "complete_rejected", rejection_reasons=rejection["rejection_reasons"],
                    rejection_sha256=_sha(rejection_path), queue_fingerprint_sha256=fp_sha,
                    v2_preflight_report_sha256=admitted["v2_receipt"]["report_sha256"],
                    source_proof=admitted["proof"], attempts=attempts,
                    physical_gpu=int(gpu), gpu_uuid=gpu_uuid, returncode=rc,
                    validation_data_read=False)
                return
            result = _validate_result(trainer, arm, admitted)
            attempts.append({"attempt": attempt, "status": "training_complete",
                "physical_gpu": int(gpu), "gpu_uuid": gpu_uuid, "returncode": rc,
                "log": log_path.name})
            _write_state(arm, "complete", result_manifest_sha256=_sha(
                Path(trainer.OUTPUT_ROOT) / f"{arm}_seed1" / "manifest.json"),
                completed_updates=trainer.UPDATES, queue_fingerprint_sha256=fp_sha,
                v2_preflight_report_sha256=admitted["v2_receipt"]["report_sha256"],
                source_proof=admitted["proof"], attempts=attempts,
                physical_gpu=int(gpu), gpu_uuid=gpu_uuid, returncode=rc,
                validation_data_read=False)
            return
        raise RuntimeError("SW0177 training supervisor stopped")
    except BaseException as exc:
        if child is not None and child.poll() is None and owner is not None:
            owner._stop_owned(child)
        if own_state:
            _write_state(arm, "failed", error=repr(exc), attempts=attempts)
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
    parser.add_argument("--arm", choices=("signed", "nonnegative"), required=True)
    parser.add_argument("--physical-gpu", type=int)
    args = parser.parse_args(argv)
    if args.worker:
        if args.physical_gpu is None:
            parser.error("worker requires --physical-gpu")
        _worker(args.arm, args.physical_gpu)
    else:
        run(args.arm)


if __name__ == "__main__":
    main()
