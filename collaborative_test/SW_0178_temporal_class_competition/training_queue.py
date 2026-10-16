"""Owner-safe SW0178 per-arm training queue."""
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
except ImportError:  # pragma: no cover
    fcntl = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNTIME = HERE / "results_archive" / "training_queue_20261016"
TRAINER_MODULE = "collaborative_test.SW_0178_temporal_class_competition.train"
OWNER_MODULE = "collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4"
_stop = threading.Event()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _import_preserving_visibility(name):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        return importlib.import_module(name)
    finally:
        if visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible


def _runtime_modules():
    trainer = _import_preserving_visibility(TRAINER_MODULE)
    owner = _import_preserving_visibility(OWNER_MODULE)
    return trainer, owner


def _state_path(arm): return RUNTIME / f"{arm}_state.json"
def _log_path(arm, attempt): return RUNTIME / f"{arm}_attempt{attempt:02d}.log"


def _write_state(arm, status, **fields):
    path = _state_path(arm)
    state = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    state.update({"experiment": "SW0178_training_queue", "arm": arm,
                  "supervisor_pid": os.getpid(), "status": status,
                  "updated_unix": time.time(), **fields})
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(state, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists(): temp.unlink()
    return state


def _input_admission(trainer=None):
    trainer = _import_preserving_visibility(TRAINER_MODULE) if trainer is None else trainer
    inputs = trainer.prepare_training_inputs_cpu()
    identity = {key: inputs[key] for key in ("dataset_manifest_sha256", "dataset_arrays_sha256",
        "source_cache_manifest_sha256", "source_cache_ledger_sha256",
        "source_cache_bindings_sha256", "source_state_sha256", "ordered_train_ids")}
    return trainer, inputs, identity


def _queue_fingerprint(trainer, arm, identity):
    payload = {"experiment": "SW0178_training_queue", "arm": arm,
        "queue_sha256": _sha(__file__),
        "trainer_sha256": _sha(HERE / "train.py"),
        "trainer_fingerprint_sha256": trainer.fingerprint_sha256(),
        "source_input_identity": identity,
        "cache_mode": "USE_PYTORCH_KERNEL_CACHE=0"}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return payload, hashlib.sha256(raw).hexdigest()


def _proc_matches(owner, pid, command):
    row = owner._proc_record(pid)
    return row is not None and row.get("state") != "Z" and row.get("argv") == command


def worker_command(arm, gpu):
    trainer, _ = _runtime_modules()
    if arm not in trainer.ARMS or type(gpu) is not int or gpu not in range(4):
        raise ValueError("SW0178 worker requires registered arm and physical GPU")
    return [sys.executable, "-m", "collaborative_test.SW_0178_temporal_class_competition.training_queue",
            "--worker", "--arm", arm, "--physical-gpu", str(gpu)]


def _resume_gpu(trainer, arm):
    """Keep the checkpoint's physical CUDA visibility on same-runtime resume."""
    work = Path(trainer.OUTPUT_ROOT) / "work" / f"{arm}_seed1"
    boundary = work / "boundary.ptx"
    if not boundary.is_file():
        return None
    runtime = _import_preserving_visibility(trainer.CHECKPOINT_MODULE).inspect_committed_metadata(work).get("runtime")
    visible = runtime.get("cuda_visible_devices") if isinstance(runtime, dict) else None
    if not isinstance(visible, str) or not visible.isdigit() or int(visible) not in range(4):
        raise ValueError("SW0178 resume boundary lacks registered physical GPU visibility")
    return int(visible)


def _worker(arm, gpu):
    trainer, _ = _runtime_modules()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise RuntimeError("SW0178 worker visibility differs from leased physical GPU")
    result = trainer.train(arm, physical_gpu=gpu)
    print(json.dumps({"arm": arm, "status": result.get("status", "training_complete"),
                      "completed_updates": result.get("completed_updates", trainer.UPDATES)},
                     sort_keys=True))
    return result


def run(arm, *, worker_timeout_seconds=12 * 3600):
    trainer, owner = _runtime_modules()
    if arm not in trainer.ARMS:
        raise ValueError("unregistered SW0178 arm")
    if fcntl is None:
        raise RuntimeError("SW0178 GPU queue requires Linux flock")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("SW0178 parent must keep CUDA hidden")
    signal.signal(signal.SIGTERM, lambda *_: _stop.set())
    signal.signal(signal.SIGINT, lambda *_: _stop.set())
    RUNTIME.mkdir(parents=True, exist_ok=True)
    state_path = _state_path(arm)
    if state_path.exists() or any(RUNTIME.glob(f"{arm}_attempt*.log")):
        raise FileExistsError(f"preserving existing SW0178 queue state/logs for {arm}")
    lock = (RUNTIME / f"{arm}.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = lease = None
    owned_state = False
    attempt = 0
    try:
        _write_state(arm, "validating_train_cache")
        owned_state = True
        trainer, _inputs, identity = _input_admission(trainer)
        queue, queue_sha = _queue_fingerprint(trainer, arm, identity)
        _write_state(arm, "waiting_for_exclusive_gpu", queue_fingerprint=queue,
            queue_fingerprint_sha256=queue_sha, source_input_identity=identity,
            validation_data_read=False)
        while not _stop.is_set():
            preferred_gpu = _resume_gpu(trainer, arm)
            gpu, gpu_uuid, initial_memory, lease = owner._acquire(_stop, preferred_gpu=preferred_gpu)
            stable = owner._stable_idle(gpu, gpu_uuid, _stop,
                lambda elapsed, used: _write_state(arm, "waiting_for_stable_gpu",
                    physical_gpu=int(gpu), gpu_uuid=gpu_uuid, stable_seconds=float(elapsed),
                    memory_used_mib=int(used), queue_fingerprint_sha256=queue_sha))
            if not stable:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                continue
            # Revalidate CPU source/data while owning the candidate lease; no CUDA query here.
            trainer, _again, identity_now = _input_admission(trainer)
            queue_now, queue_sha_now = _queue_fingerprint(trainer, arm, identity_now)
            if identity_now != identity or queue_sha_now != queue_sha or queue_now != queue:
                raise RuntimeError("SW0178 source/cache/trainer fingerprint changed while queued")
            actual_uuid, used, owners = owner.gpu_guard._gpu_info(gpu)
            if actual_uuid != gpu_uuid or owners or int(used) > owner.MAX_MEMORY_MIB:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                _write_state(arm, "waiting_for_exclusive_gpu", physical_gpu=int(gpu),
                    gpu_uuid=actual_uuid, memory_used_mib=int(used), owners=list(owners),
                    queue_fingerprint_sha256=queue_sha)
                continue
            attempt += 1
            command = worker_command(arm, int(gpu))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", USE_PYTORCH_KERNEL_CACHE="0")
            log_path = _log_path(arm, attempt)
            worker_deadline = time.monotonic() + worker_timeout_seconds
            yield_to_foreign = False
            with log_path.open("x", encoding="utf-8", newline="\n") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    pass_fds=(lease.fileno(),))
                _write_state(arm, "running_worker", child_pid=child.pid, child_pgid=child.pid,
                    child_argv=command, physical_gpu=int(gpu), gpu_uuid=gpu_uuid,
                    attempt=attempt, queue_fingerprint_sha256=queue_sha)
                while child.poll() is None and not _stop.is_set():
                    if time.monotonic() >= worker_deadline:
                        owner._stop_owned(child)
                        raise TimeoutError("SW0178 active worker exceeded its bound")
                    live_uuid, memory_now, gpu_owners = owner.gpu_guard._gpu_info(gpu)
                    if child.poll() is not None:
                        break
                    foreign = live_uuid != gpu_uuid or any(int(pid) != child.pid for pid in gpu_owners)
                    if foreign:
                        owner._stop_owned(child)
                        yield_to_foreign = True
                        _write_state(arm, "yielding_foreign_owner", child_pid=child.pid,
                            physical_gpu=int(gpu), gpu_uuid=gpu_uuid, observed_gpu_uuid=live_uuid,
                            owners=list(gpu_owners), attempt=attempt)
                        break
                    record = owner._proc_record(child.pid)
                    if record is None or record.get("state") == "Z":
                        try: child.wait(timeout=1.0)
                        except subprocess.TimeoutExpired: pass
                        if child.poll() is not None: break
                        continue
                    if not _proc_matches(owner, child.pid, command):
                        owner._stop_owned(child)
                        raise RuntimeError("SW0178 owned worker argv changed")
                    _write_state(arm, "running_worker", child_pid=child.pid,
                        physical_gpu=int(gpu), gpu_uuid=gpu_uuid, memory_used_mib=int(memory_now),
                        attempt=attempt, queue_fingerprint_sha256=queue_sha)
                    time.sleep(3)
                if _stop.is_set() and child.poll() is None:
                    owner._stop_owned(child)
                return_code = child.wait()
            if yield_to_foreign:
                child = None
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                _write_state(arm, "waiting_for_exclusive_gpu", yielded_attempt=attempt,
                    queue_fingerprint_sha256=queue_sha)
                continue
            if return_code != 0:
                raise RuntimeError(f"SW0178 worker exited {return_code}; inspect {log_path}")
            work = Path(trainer.OUTPUT_ROOT) / "work" / f"{arm}_seed1"
            output = Path(trainer.OUTPUT_ROOT) / f"{arm}_seed1"
            if output.is_dir():
                manifest = trainer.validate_training_artifact(arm, output)
                status, artifact_sha = "complete", _sha(output / "manifest.json")
                fields = {"manifest_sha256": artifact_sha, "completed_updates": manifest["completed_updates"]}
            elif (work / "credit_rejection.json").is_file():
                rejection = json.loads((work / "credit_rejection.json").read_text(encoding="utf-8"))
                if rejection.get("bindings", {}).get("training_fingerprint") != trainer.implementation_fingerprint():
                    raise ValueError("SW0178 rejection receipt has stale trainer fingerprint")
                status, artifact_sha = "complete_rejected", _sha(work / "credit_rejection.json")
                fields = {"rejection_sha256": artifact_sha,
                    "completed_updates": rejection.get("completed_updates"),
                    "rejection_reasons": rejection.get("rejection_reasons")}
            else:
                raise FileNotFoundError("SW0178 worker finished without artifact or rejection receipt")
            _write_state(arm, status, **fields, artifact_sha256=artifact_sha,
                physical_gpu=int(gpu), gpu_uuid=gpu_uuid, attempt=attempt,
                queue_fingerprint_sha256=queue_sha, validation_data_read=False)
            return {"status": status, "arm": arm, "artifact_sha256": artifact_sha,
                    "queue_fingerprint_sha256": queue_sha, **fields}
        raise RuntimeError("SW0178 queue stopped before terminal arm state")
    except BaseException as exc:
        if child is not None and child.poll() is None:
            owner._stop_owned(child)
        if owned_state:
            _write_state(arm, "failed", error=repr(exc), finished_unix=time.time())
        raise
    finally:
        if child is not None and child.poll() is None:
            owner._stop_owned(child)
        if lease is not None:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close()
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN); lock.close()


def main(argv=None):
    import argparse
    trainer, _ = _runtime_modules()
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--arm", choices=trainer.ARMS, required=True)
    parser.add_argument("--physical-gpu", type=int)
    args = parser.parse_args(argv)
    if args.worker:
        if args.physical_gpu not in range(4): parser.error("worker requires physical GPU")
        _worker(args.arm, args.physical_gpu)
        return 0
    print(json.dumps(run(args.arm), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
