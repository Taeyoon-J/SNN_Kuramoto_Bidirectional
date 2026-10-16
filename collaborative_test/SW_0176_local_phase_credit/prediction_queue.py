"""Exclusive-GPU owner queue for the single immutable SW0176 fixed320 endpoint."""
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
_ASSIGNED_VISIBILITY = os.environ.get("CUDA_VISIBLE_DEVICES") if _WORKER else None
if not _WORKER:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("USE_PYTORCH_KERNEL_CACHE", "0")

import numpy as np  # NumPy/OpenMP before any legacy torch imports.

from collaborative_test.SW_0176_local_phase_credit import predict_evaluate as evaluate
from collaborative_test.SW_0176_local_phase_credit import train

if _WORKER:
    if _ASSIGNED_VISIBILITY is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = _ASSIGNED_VISIBILITY

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODULE = "collaborative_test.SW_0176_local_phase_credit.prediction_queue"
TRAIN_MODULE = "collaborative_test.SW_0176_local_phase_credit.training_queue"
OWNER_MODULE = "collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4"
RUNTIME = HERE / "results_archive" / "prediction_queue_v1_20261016"
STOP = threading.Event()
ACTIVE_TRAIN = {"validating_source_cpu", "waiting_for_exclusive_gpu",
                "waiting_for_stable_gpu", "running_training"}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        if replace:
            os.replace(temp, path)
        else:
            os.link(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _state_path():
    return RUNTIME / "state.json"


def _write_state(status, **fields):
    _atomic_json(_state_path(), {"experiment": "SW0176_prediction_queue_v1",
        "status": status, "supervisor_pid": os.getpid(), "updated_unix": time.time(),
        **fields}, replace=True)


def _module_live(owner, pid, module):
    if type(pid) is not int or pid <= 0:
        return False
    record = owner._proc_record(pid)
    if not record or record.get("state") == "Z":
        return False
    argv = record.get("argv", [])
    if argv.count("-m") != 1:
        return False
    index = argv.index("-m")
    return index + 1 < len(argv) and argv[index + 1] == module


def _owner_preserve_visibility():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        owner = importlib.import_module(OWNER_MODULE)
    finally:
        if visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible
    return owner


def _training_terminal(owner):
    path = train.OUTPUT_ROOT.parent / "training_queue_v1_20261016" / "state.json"
    if not path.is_file():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("experiment") != "SW0176_local_phase_training_queue":
        raise ValueError("SW0176 training queue dependency identity mismatch")
    status = state.get("status")
    if status == "complete":
        artifact = train.OUTPUT_ROOT / f"{train.ARM}_seed1"
        manifest = train.validate_training_artifact(artifact)
        return _validate_training_state(state, owner, manifest=manifest,
            manifest_sha256=_sha(artifact / "manifest.json"))
    return _validate_training_state(state, owner)


def _validate_training_state(state, owner, *, manifest=None, manifest_sha256=None):
    """Validate the queue producer's public state schema independently of disk I/O."""
    if state.get("experiment") != "SW0176_local_phase_training_queue":
        raise ValueError("SW0176 training queue dependency identity mismatch")
    status = state.get("status")
    if status == "complete":
        if (manifest is None or manifest_sha256 is None
                or state.get("manifest_sha256") != manifest_sha256
                or state.get("completed_updates") != train.UPDATES
                or manifest.get("completed_updates") != train.UPDATES):
            raise ValueError("SW0176 training queue state does not bind complete 64-update artifact")
        return state
    if status in ACTIVE_TRAIN and _module_live(owner, state.get("supervisor_pid"), TRAIN_MODULE):
        return False
    raise RuntimeError(f"SW0176 training dependency is failed/orphaned: {status}")


def _validate_completed_model_cpu(prepared):
    """Exercise the real completed-checkpoint loader before reserving a GPU."""
    before = evaluate.torch.cuda.is_initialized()
    if before:
        raise RuntimeError("SW0176 parent model validation must precede CUDA initialization")
    found, state_sha = evaluate._load_model(train.ARM, prepared, "cpu")
    expected = prepared["manifests"][train.ARM]["final_model_state_sha256"]
    if state_sha != expected or evaluate.torch.cuda.is_initialized():
        raise ValueError("SW0176 CPU checkpoint loader did not validate the frozen trained model")
    del found
    return state_sha


def worker_command(gpu):
    if type(gpu) is not int or gpu not in range(4):
        raise ValueError("SW0176 prediction worker requires physical GPU 0..3")
    return [sys.executable, "-u", "-m", MODULE, "--worker", "--physical-gpu", str(gpu)]


def _worker(gpu):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise RuntimeError("SW0176 prediction worker visibility differs from leased GPU")
    prepared = evaluate.prepare_prediction_inputs_cpu()
    if (evaluate.torch.cuda.is_initialized()
            or os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu)):
        raise RuntimeError("SW0176 prediction CPU admission changed CUDA state/visibility")
    return evaluate.predict_arm(train.ARM, device="cuda:0", prepared=prepared)


def run(*, timeout_seconds=8 * 3600):
    import fcntl
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("SW0176 prediction supervisor must keep CUDA hidden")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    state_path = _state_path()
    if state_path.exists() or any(RUNTIME.glob("worker_attempt*.log")):
        raise FileExistsError("preserving existing SW0176 prediction state/logs")
    lock = (RUNTIME / "queue.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    owner = lease = child = None
    owned_state = False
    attempt = 0
    try:
        _write_state("waiting_for_training")
        owned_state = True
        owner = _owner_preserve_visibility()
        deadline = time.monotonic() + timeout_seconds
        while not STOP.is_set():
            if time.monotonic() >= deadline:
                raise TimeoutError("SW0176 prediction wait for completed training timed out")
            state = _training_terminal(owner)
            if state is not False and state is not None:
                prepared = evaluate.prepare_prediction_inputs_cpu()
                _validate_completed_model_cpu(prepared)
                break
            time.sleep(5)
        else:
            raise RuntimeError("SW0176 prediction queue stopped while waiting for training")
        bindings = evaluate._bindings(prepared, train.ARM)
        binding_sha = evaluate._json_sha(bindings)
        fingerprint = evaluate.implementation_fingerprint()
        _write_state("waiting_for_exclusive_gpu", training_manifest_sha256=
            bindings["training_manifest_sha256"], bindings_sha256=binding_sha)
        while not STOP.is_set():
            if time.monotonic() >= deadline:
                raise TimeoutError("SW0176 prediction GPU wait timed out")
            gpu, gpu_uuid, used, lease = owner._acquire(STOP)
            stable = owner._stable_idle(gpu, gpu_uuid, STOP,
                lambda elapsed, memory: _write_state("waiting_for_stable_gpu",
                    physical_gpu=int(gpu), gpu_uuid=gpu_uuid,
                    stable_seconds=float(elapsed), memory_used_mib=int(memory)))
            if not stable:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                continue
            latest_training = _training_terminal(owner)
            if latest_training is False or latest_training is None:
                raise RuntimeError("SW0176 completed training artifact disappeared under GPU lease")
            current = evaluate.prepare_prediction_inputs_cpu()
            _validate_completed_model_cpu(current)
            if (evaluate._bindings(current, train.ARM) != bindings
                    or evaluate.implementation_fingerprint() != fingerprint):
                raise RuntimeError("SW0176 prediction source/artifact bindings changed before launch")
            actual_uuid, memory, owners = owner.gpu_guard._gpu_info(gpu)
            if actual_uuid != gpu_uuid or owners or int(memory) > owner.MAX_MEMORY_MIB:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                _write_state("waiting_for_exclusive_gpu", physical_gpu=int(gpu),
                    gpu_uuid=actual_uuid, memory_used_mib=int(memory), owners=list(owners))
                continue
            attempt += 1
            command = worker_command(int(gpu))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
                USE_PYTORCH_KERNEL_CACHE="0")
            yielded = False
            log_path = RUNTIME / f"worker_attempt{attempt:02d}.log"
            with log_path.open("x", encoding="utf-8", newline="\n") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True, pass_fds=(lease.fileno(),))
                _write_state("running_prediction", child_pid=child.pid, child_pgid=child.pid,
                    child_argv=command, physical_gpu=int(gpu), gpu_uuid=gpu_uuid,
                    training_manifest_sha256=bindings["training_manifest_sha256"], attempt=attempt)
                while child.poll() is None and not STOP.is_set():
                    if time.monotonic() >= deadline:
                        owner._stop_owned(child)
                        raise TimeoutError("SW0176 fixed320 prediction worker timed out")
                    current_uuid, used_now, gpu_owners = owner.gpu_guard._gpu_info(gpu)
                    if child.poll() is not None:
                        break
                    if (current_uuid != gpu_uuid
                            or any(int(pid) != child.pid for pid in gpu_owners)):
                        owner._stop_owned(child); yielded = True
                        _write_state("yielding_foreign_gpu", child_pid=child.pid,
                            physical_gpu=int(gpu), gpu_uuid=gpu_uuid,
                            observed_gpu_uuid=current_uuid, owners=list(gpu_owners), attempt=attempt)
                        break
                    proc = owner._proc_record(child.pid) if child.poll() is None else None
                    if proc is None or proc.get("state") == "Z":
                        try:
                            child.wait(timeout=1.0)
                        except subprocess.TimeoutExpired:
                            pass
                        if child.poll() is not None:
                            break
                        continue
                    if proc.get("argv") != command:
                        owner._stop_owned(child)
                        raise RuntimeError("SW0176 owned prediction child argv changed")
                    _write_state("running_prediction", child_pid=child.pid,
                        physical_gpu=int(gpu), gpu_uuid=gpu_uuid,
                        memory_used_mib=int(used_now), attempt=attempt)
                    time.sleep(3)
                if STOP.is_set() and child.poll() is None:
                    owner._stop_owned(child)
                rc = child.wait()
            if yielded:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                child = None
                continue
            if rc != 0:
                raise RuntimeError(f"SW0176 prediction worker exited {rc}; inspect {log_path}")
            after = evaluate.prepare_prediction_inputs_cpu()
            result = evaluate.validate_prediction(train.ARM, prepared=after)
            _write_state("complete", child_pid=child.pid, physical_gpu=int(gpu),
                gpu_uuid=gpu_uuid, training_manifest_sha256=bindings["training_manifest_sha256"],
                bindings_sha256=binding_sha, output=str(result["directory"]),
                prediction_sha256=result["prediction_sha256"],
                protocol_sha256=result["protocol_sha256"], attempt=attempt,
                ground_truth_used_for_prediction=False)
            return result
        raise RuntimeError("SW0176 prediction queue stopped before prediction launch")
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
            parser.error("worker requires leased physical GPU [0,3]")
        _worker(args.physical_gpu)
    else:
        run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
