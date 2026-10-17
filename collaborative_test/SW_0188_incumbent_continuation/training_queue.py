"""Exclusive, resumable SW0188 proposal queue gated on successful source preflight."""
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
RUNTIME = HERE / "results_archive" / "training_queue_20261017"
SEEDS = (0, 1, 2)
TRAINER_MODULE = "collaborative_test.SW_0188_incumbent_continuation.train"
OWNER_MODULE = "collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4"
PREFLIGHT_QUEUE_MODULE = "collaborative_test.SW_0177_learned_coupling_polarity.resource_queue_v2"
STOP = threading.Event()


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
    return (_import_preserving_visibility(TRAINER_MODULE),
            _import_preserving_visibility(OWNER_MODULE))


def _state_path(seed):
    return RUNTIME / f"product_q_ce_seed{int(seed)}_state.json"


def _log_path(seed, attempt):
    return RUNTIME / f"product_q_ce_seed{int(seed)}_attempt{int(attempt):02d}.log"


def _write_state(seed, status, **fields):
    path = _state_path(seed)
    state = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    state.update({"experiment": "SW0188_training_queue", "arm": "product_q_ce",
        "seed": int(seed), "supervisor_pid": os.getpid(), "status": status,
        "updated_unix": time.time(), **fields})
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(state, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()
    return state


def _wait_preflight(trainer, stop_event=STOP, poll_seconds=5.0):
    queue = _import_preserving_visibility(PREFLIGHT_QUEUE_MODULE)
    while not stop_event.is_set():
        if queue.STATE.is_file() and queue.REPORT.is_file():
            try:
                return trainer._preflight_receipt(1)
            except ValueError as exc:
                state = json.loads(queue.STATE.read_text(encoding="utf-8"))
                if state.get("status") == "complete_rejected":
                    return {"status": "complete_rejected", "state": state,
                            "report_sha256": _sha(queue.REPORT), "reason": str(exc)}
                if state.get("status") == "failed":
                    raise RuntimeError("SW0188 source preflight failed") from exc
                if state.get("status") in {"complete", "complete_rejected"}:
                    raise
        stop_event.wait(poll_seconds)
    raise RuntimeError("SW0188 queue stopped while waiting for successful source preflight")


def _cpu_admission(seed, trainer, preflight_receipt):
    inputs = trainer.prepare_training_inputs_cpu()
    fresh = trainer.prepare_fresh_training_state_cpu(seed, inputs)
    proof = fresh["proof"]
    if proof.get("cuda_initialized") is not False:
        raise RuntimeError("SW0188 CPU factory admission initialized CUDA")
    identity = {key: inputs[key] for key in (
        "dataset_manifest_sha256", "dataset_arrays_sha256", "dataset_producer_proof",
        "ordered_train_ids")}
    identity.update({"source_factory_proof": proof,
        "source_preflight_report_sha256": preflight_receipt["report_sha256"],
        "source_preflight_queue_fingerprint_sha256": preflight_receipt.get("queue_fingerprint_sha256")})
    del fresh
    return inputs, identity


def _queue_fingerprint(trainer, seed, identity):
    payload = {"experiment": "SW0188_training_queue", "arm": "product_q_ce",
        "seed": int(seed), "queue_sha256": _sha(__file__),
        "trainer_sha256": _sha(HERE / "train.py"),
        "trainer_fingerprint_sha256": trainer.fingerprint_sha256(),
        "input_identity": identity, "cache_mode": "USE_PYTORCH_KERNEL_CACHE=0"}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return payload, hashlib.sha256(raw).hexdigest()


def _proc_matches(owner, pid, command):
    record = owner._proc_record(pid)
    return record is not None and record.get("state") != "Z" and record.get("argv") == command


def worker_command(seed, physical_gpu):
    if int(seed) not in (0, 1, 2) or type(physical_gpu) is not int or physical_gpu not in range(4):
        raise ValueError("SW0188 worker requires a registered seed and physical GPU")
    return [sys.executable, "-m", "collaborative_test.SW_0188_incumbent_continuation.training_queue",
        "--worker", "--source-seed", str(int(seed)), "--physical-gpu", str(physical_gpu)]


def _resume_gpu(trainer, seed):
    work, _ = trainer._artifact_paths(trainer.OUTPUT_ROOT, "product_q_ce", seed)
    if not (work / "boundary.ptx").is_file():
        return None
    checkpoint = _import_preserving_visibility(trainer.base.CHECKPOINT_MODULE)
    runtime = checkpoint.inspect_committed_metadata(work).get("runtime")
    visible = runtime.get("cuda_visible_devices") if isinstance(runtime, dict) else None
    if not isinstance(visible, str) or not visible.isdigit() or int(visible) not in range(4):
        raise ValueError("SW0188 boundary lacks its physical-GPU runtime binding")
    return int(visible)


def _validate_credit_rejection(trainer, work, seed):
    path = Path(work) / "credit_rejection.json"
    rejection = json.loads(path.read_text(encoding="utf-8"))
    expected = trainer.implementation_fingerprint()
    bindings = rejection.get("bindings")
    if (rejection.get("experiment") != trainer.EXPERIMENT
            or rejection.get("status") != "scientific_rejection"
            or rejection.get("arm") != "product_q_ce"
            or rejection.get("seed") != int(seed)
            or not isinstance(bindings, dict)
            or bindings.get("training_fingerprint") != expected
            or not rejection.get("rejection_reasons")
            or rejection.get("validation_data_read") is not False):
        raise ValueError("SW0188 scientific rejection is not bound to current trainer/run")
    return {"status": "complete_rejected", "rejection_sha256": _sha(path),
        "proposal_cursor": rejection.get("proposal_cursor"),
        "accepted_updates": rejection.get("accepted_updates"),
        "rejection_reasons": rejection.get("rejection_reasons")}


def _worker(seed, physical_gpu):
    trainer, _ = _runtime_modules()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("SW0188 worker visibility differs from its assigned physical GPU")
    result = trainer.train("product_q_ce", seed=int(seed), physical_gpu=int(physical_gpu))
    print(json.dumps({"seed": int(seed), "status": result.get("status", "training_complete"),
        "proposal_cursor": result.get("proposal_cursor", trainer.PROPOSALS),
        "accepted_updates": result.get("accepted_updates")}, sort_keys=True))
    return result


def run(seed, *, active_worker_timeout_seconds=12 * 3600):
    seed = int(seed)
    if seed not in (0, 1, 2):
        raise ValueError("unregistered SW0188 source seed")
    trainer, owner = _runtime_modules()
    if fcntl is None or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("SW0188 parent requires Linux flock and hidden CUDA visibility")
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    RUNTIME.mkdir(parents=True, exist_ok=True)
    state_path = _state_path(seed)
    if state_path.exists() or any(RUNTIME.glob(f"product_q_ce_seed{seed}_attempt*.log")):
        raise FileExistsError(f"preserving existing SW0188 queue state for seed {seed}")
    lock = (RUNTIME / f"product_q_ce_seed{seed}.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = lease = None
    owned_state = False
    attempt = 0
    try:
        _write_state(seed, "waiting_for_successful_source_preflight", validation_data_read=False)
        owned_state = True
        preflight_receipt = _wait_preflight(trainer)
        if preflight_receipt["status"] == "complete_rejected":
            return _write_state(seed, "complete_rejected", source_preflight_report_sha256=
                preflight_receipt["report_sha256"], reason=preflight_receipt["reason"],
                validation_data_read=False)
        inputs, identity = _cpu_admission(seed, trainer, preflight_receipt)
        queue, queue_sha = _queue_fingerprint(trainer, seed, identity)
        _write_state(seed, "waiting_for_exclusive_gpu", queue_fingerprint=queue,
            queue_fingerprint_sha256=queue_sha, source_input_identity=identity,
            validation_data_read=False)
        while not STOP.is_set():
            preferred_gpu = _resume_gpu(trainer, seed)
            gpu, gpu_uuid, _initial_memory, lease = owner._acquire(STOP, preferred_gpu=preferred_gpu)
            stable = owner._stable_idle(gpu, gpu_uuid, STOP,
                lambda elapsed, used: _write_state(seed, "waiting_for_stable_gpu",
                    physical_gpu=int(gpu), gpu_uuid=gpu_uuid, stable_seconds=float(elapsed),
                    memory_used_mib=int(used), queue_fingerprint_sha256=queue_sha))
            if not stable:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                continue
            current_preflight = _wait_preflight(trainer)
            inputs_now, identity_now = _cpu_admission(seed, trainer, current_preflight)
            queue_now, queue_sha_now = _queue_fingerprint(trainer, seed, identity_now)
            if (identity_now != identity or queue_sha_now != queue_sha or queue_now != queue
                    or current_preflight != preflight_receipt):
                raise RuntimeError("SW0188 source/data/preflight/trainer binding changed while queued")
            live_uuid, used, owners = owner.gpu_guard._gpu_info(gpu)
            if live_uuid != gpu_uuid or owners or int(used) > owner.MAX_MEMORY_MIB:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                _write_state(seed, "waiting_for_exclusive_gpu", physical_gpu=int(gpu),
                    gpu_uuid=live_uuid, memory_used_mib=int(used), owners=list(owners),
                    queue_fingerprint_sha256=queue_sha)
                continue
            attempt += 1
            command = worker_command(seed, int(gpu))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", USE_PYTORCH_KERNEL_CACHE="0")
            log_path = _log_path(seed, attempt)
            deadline = time.monotonic() + active_worker_timeout_seconds
            yielded = False
            with log_path.open("x", encoding="utf-8", newline="\n") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True, pass_fds=(lease.fileno(),))
                _write_state(seed, "running_worker", child_pid=child.pid, child_pgid=child.pid,
                    child_argv=command, physical_gpu=int(gpu), gpu_uuid=gpu_uuid,
                    attempt=attempt, queue_fingerprint_sha256=queue_sha)
                while child.poll() is None and not STOP.is_set():
                    if time.monotonic() >= deadline:
                        owner._stop_owned(child)
                        raise TimeoutError("SW0188 active worker exceeded its registered bound")
                    live_uuid, used, active_owners = owner.gpu_guard._gpu_info(gpu)
                    if child.poll() is not None:
                        break
                    if live_uuid != gpu_uuid or any(int(pid) != child.pid for pid in active_owners):
                        owner._stop_owned(child)
                        yielded = True
                        _write_state(seed, "yielding_foreign_owner", child_pid=child.pid,
                            physical_gpu=int(gpu), gpu_uuid=gpu_uuid,
                            observed_gpu_uuid=live_uuid, owners=list(active_owners), attempt=attempt)
                        break
                    record = owner._proc_record(child.pid)
                    if record is None or record.get("state") == "Z":
                        try: child.wait(timeout=1.0)
                        except subprocess.TimeoutExpired: pass
                        if child.poll() is not None: break
                    elif not _proc_matches(owner, child.pid, command):
                        owner._stop_owned(child)
                        raise RuntimeError("SW0188 owned worker argv changed")
                    else:
                        _write_state(seed, "running_worker", child_pid=child.pid,
                            physical_gpu=int(gpu), gpu_uuid=gpu_uuid,
                            memory_used_mib=int(used), attempt=attempt,
                            queue_fingerprint_sha256=queue_sha)
                    STOP.wait(3)
                if STOP.is_set() and child.poll() is None:
                    owner._stop_owned(child)
                return_code = child.wait()
            if yielded:
                child = None
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                _write_state(seed, "waiting_for_exclusive_gpu", yielded_attempt=attempt,
                    queue_fingerprint_sha256=queue_sha)
                continue
            if STOP.is_set():
                raise RuntimeError("SW0188 queue stopped after stopping its owned worker")
            if return_code != 0:
                raise RuntimeError(f"SW0188 worker exited {return_code}; inspect {log_path}")
            work, output = trainer._artifact_paths(trainer.OUTPUT_ROOT, "product_q_ce", seed)
            if output.is_dir():
                manifest = trainer.validate_training_artifact("product_q_ce", output, seed=seed)
                status = "complete"
                fields = {"manifest_sha256": _sha(output / "manifest.json"),
                    "proposal_cursor": manifest["proposal_cursor"],
                    "accepted_updates": manifest["accepted_updates"]}
            elif (work / "credit_rejection.json").is_file():
                rejected = _validate_credit_rejection(trainer, work, seed)
                status = rejected.pop("status")
                fields = rejected
            else:
                raise FileNotFoundError("SW0188 worker ended without a complete artifact/rejection")
            return _write_state(seed, status, **fields, artifact_sha256=(
                fields.get("manifest_sha256") or fields.get("rejection_sha256")),
                physical_gpu=int(gpu), gpu_uuid=gpu_uuid, attempt=attempt,
                queue_fingerprint_sha256=queue_sha, validation_data_read=False)
        raise RuntimeError("SW0188 queue stopped before a terminal training state")
    except BaseException as exc:
        if child is not None and child.poll() is None:
            owner._stop_owned(child)
        if owned_state:
            _write_state(seed, "failed", error=repr(exc), finished_unix=time.time())
        raise
    finally:
        if child is not None and child.poll() is None:
            owner._stop_owned(child)
        if lease is not None:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close()
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN); lock.close()


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--source-seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--physical-gpu", type=int)
    args = parser.parse_args(argv)
    if args.worker:
        if args.physical_gpu not in range(4):
            parser.error("worker requires a physical GPU 0..3")
        _worker(args.source_seed, args.physical_gpu)
        return 0
    if args.physical_gpu is not None:
        parser.error("parent supervisor does not accept --physical-gpu")
    result = run(args.source_seed)
    print(json.dumps({"seed": args.source_seed, "status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
