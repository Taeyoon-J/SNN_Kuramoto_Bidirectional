"""Exclusive, durable SW0187 per-seed worker; prefix and both forks share one lease."""
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
except ImportError:  # pragma: no cover - execution is on the Linux research host.
    fcntl = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNTIME = HERE / "results_archive" / "training_queue_20261017"
SEEDS = (0, 1, 2)
TRAINER_MODULE = "collaborative_test.SW_0187_joint_encoder_unique_scaling.train"
OWNER_MODULE = "collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4"
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


def _modules():
    return (_import_preserving_visibility(TRAINER_MODULE),
            _import_preserving_visibility(OWNER_MODULE))


def state_path(seed):
    return RUNTIME / f"seed{int(seed)}_state.json"


def _log_path(seed, attempt):
    return RUNTIME / f"seed{int(seed)}_attempt{int(attempt):02d}.log"


def _write_state(seed, status, **fields):
    path = state_path(seed)
    current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    current.update({"experiment": "SW0187_training_queue", "seed": int(seed),
        "supervisor_pid": os.getpid(), "status": status, "updated_unix": time.time(), **fields})
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(current, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists(): temp.unlink()
    return current


def _cpu_admission(seed, trainer):
    inputs = trainer.prepare_training_inputs_cpu()
    fresh = trainer.source.prepare_fresh_training_state_cpu(seed, inputs)
    if torch_cuda_initialized(trainer):
        raise RuntimeError("SW0187 CPU admission initialized CUDA")
    control = trainer._prefix_control(seed)
    proof = fresh["proof"]
    if set(fresh["families"]) != trainer.source.FAMILIES:
        raise ValueError("SW0187 requires five disjoint trainable optimizer families")
    if proof["initial_model_state_sha256"] != control["manifest"]["initial_model_state_sha256"]:
        raise ValueError("SW0187 fresh source differs from SW0186 control initialization")
    identity = {"dataset_manifest_sha256": inputs["dataset_manifest_sha256"],
        "dataset_arrays_sha256": inputs["dataset_arrays_sha256"],
        "ordered_train_ids_sha256": trainer._sha_bytes(
            __import__("numpy").asarray(inputs["ordered_train_ids"], dtype="<i8").tobytes()),
        "fresh_source_proof": proof,
        "control_manifest_sha256": control["manifest_sha256"],
        "control_history_sha256": control["history_sha256"],
        "validation_data_read": False}
    del inputs, fresh
    return identity


def torch_cuda_initialized(trainer):
    torch = trainer.torch
    return bool(torch.cuda.is_initialized())


def _queue_fingerprint(trainer, seed, identity):
    value = {"experiment": "SW0187_training_queue", "seed": int(seed),
        "queue_sha256": _sha(__file__), "trainer_sha256": _sha(HERE / "train.py"),
        "training_fingerprint": trainer.fingerprint_sha256(),
        "input_identity": identity, "cache_mode": "USE_PYTORCH_KERNEL_CACHE=0"}
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return value, hashlib.sha256(raw).hexdigest()


def worker_command(seed, physical_gpu):
    if int(seed) not in SEEDS or type(physical_gpu) is not int or physical_gpu not in range(4):
        raise ValueError("SW0187 worker needs a registered seed and physical GPU")
    return [sys.executable, "-m", "collaborative_test.SW_0187_joint_encoder_unique_scaling.training_queue",
        "--worker", "--source-seed", str(int(seed)), "--physical-gpu", str(physical_gpu)]


def _proc_matches(owner, pid, command):
    record = owner._proc_record(pid)
    return record is not None and record.get("state") != "Z" and record.get("argv") == command


def _resume_gpu(trainer, seed):
    paths = [_ for _ in [trainer._paths(trainer.OUTPUT_ROOT, "shared_prefix", seed)[0]]]
    paths += [trainer._paths(trainer.OUTPUT_ROOT, arm, seed)[0] for arm in trainer.ARMS]
    checkpoint = trainer._module(trainer.prior.base.CHECKPOINT_MODULE)
    observed = set()
    for work in paths:
        boundary = work / "boundary.ptx"
        if boundary.is_file():
            runtime = checkpoint.inspect_committed_metadata(work).get("runtime")
            gpu = runtime.get("cuda_visible_devices") if isinstance(runtime, dict) else None
            if not isinstance(gpu, str) or not gpu.isdigit() or int(gpu) not in range(4):
                raise ValueError("SW0187 boundary lacks its physical GPU runtime binding")
            observed.add(int(gpu))
    if len(observed) > 1:
        raise ValueError("SW0187 shared prefix and branch checkpoints use different physical GPUs")
    return next(iter(observed)) if observed else None


def _worker(seed, physical_gpu):
    trainer = _import_preserving_visibility(TRAINER_MODULE)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("SW0187 worker visibility differs from leased GPU")
    result = trainer.train_seed(seed, physical_gpu)
    print(json.dumps({"seed": int(seed), "status": result["status"],
        "arms": list(result.get("arms", {})), "proposals": result.get("actual_proposals", 2)},
        sort_keys=True), flush=True)
    return result


def run(seed, *, active_worker_timeout_seconds=12 * 3600):
    seed = int(seed)
    if seed not in SEEDS: raise ValueError("unregistered SW0187 seed")
    trainer, owner = _modules()
    if fcntl is None or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("SW0187 parent requires Linux flock and hidden CUDA")
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    RUNTIME.mkdir(parents=True, exist_ok=True)
    state = state_path(seed)
    if state.exists() or any(RUNTIME.glob(f"seed{seed}_attempt*.log")):
        raise FileExistsError(f"preserving existing SW0187 queue state for seed {seed}")
    lock = (RUNTIME / f"seed{seed}.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = lease = None; owned_state = False; attempt = 0
    try:
        _write_state(seed, "validating_train_inputs", validation_data_read=False); owned_state = True
        identity = _cpu_admission(seed, trainer)
        queue, queue_sha = _queue_fingerprint(trainer, seed, identity)
        _write_state(seed, "waiting_for_exclusive_gpu", queue_fingerprint=queue,
            queue_fingerprint_sha256=queue_sha, input_identity=identity, validation_data_read=False)
        while not STOP.is_set():
            preferred = _resume_gpu(trainer, seed)
            gpu, gpu_uuid, _memory, lease = owner._acquire(STOP, preferred_gpu=preferred)
            stable = owner._stable_idle(gpu, gpu_uuid, STOP,
                lambda elapsed, used: _write_state(seed, "waiting_for_stable_gpu",
                    physical_gpu=int(gpu), gpu_uuid=gpu_uuid, stable_seconds=float(elapsed),
                    memory_used_mib=int(used), queue_fingerprint_sha256=queue_sha))
            if not stable:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None; continue
            identity_now = _cpu_admission(seed, trainer)
            queue_now, sha_now = _queue_fingerprint(trainer, seed, identity_now)
            if identity_now != identity or queue_now != queue or sha_now != queue_sha:
                raise RuntimeError("SW0187 source/cache/fingerprint changed while queued")
            live_uuid, used, owners = owner.gpu_guard._gpu_info(gpu)
            if live_uuid != gpu_uuid or owners or int(used) > owner.MAX_MEMORY_MIB:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None; continue
            attempt += 1; command = worker_command(seed, int(gpu))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", USE_PYTORCH_KERNEL_CACHE="0")
            log_path = _log_path(seed, attempt); deadline = time.monotonic() + active_worker_timeout_seconds
            yielded = False
            with log_path.open("x", encoding="utf-8", newline="\n") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    pass_fds=(lease.fileno(),))
                _write_state(seed, "running_worker", child_pid=child.pid, child_pgid=child.pid,
                    child_argv=command, physical_gpu=int(gpu), gpu_uuid=gpu_uuid,
                    attempt=attempt, queue_fingerprint_sha256=queue_sha)
                while child.poll() is None and not STOP.is_set():
                    if time.monotonic() >= deadline:
                        owner._stop_owned(child); raise TimeoutError("SW0187 worker exceeded active bound")
                    current_uuid, used, active = owner.gpu_guard._gpu_info(gpu)
                    if child.poll() is not None: break
                    if current_uuid != gpu_uuid or any(int(pid) != child.pid for pid in active):
                        owner._stop_owned(child); yielded = True
                        _write_state(seed, "yielding_foreign_owner", child_pid=child.pid,
                            physical_gpu=int(gpu), gpu_uuid=gpu_uuid,
                            observed_gpu_uuid=current_uuid, owners=list(active), attempt=attempt)
                        break
                    record = owner._proc_record(child.pid)
                    if record is None or record.get("state") == "Z":
                        try: child.wait(timeout=1.0)
                        except subprocess.TimeoutExpired: pass
                        if child.poll() is not None: break
                    elif not _proc_matches(owner, child.pid, command):
                        owner._stop_owned(child); raise RuntimeError("SW0187 worker argv changed")
                    else:
                        _write_state(seed, "running_worker", child_pid=child.pid,
                            physical_gpu=int(gpu), gpu_uuid=gpu_uuid, memory_used_mib=int(used),
                            attempt=attempt, queue_fingerprint_sha256=queue_sha)
                    STOP.wait(3)
                if STOP.is_set() and child.poll() is None: owner._stop_owned(child)
                code = child.wait()
            if yielded:
                child = None; fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                _write_state(seed, "waiting_for_exclusive_gpu", yielded_attempt=attempt,
                    queue_fingerprint_sha256=queue_sha); continue
            if STOP.is_set(): raise RuntimeError("SW0187 queue stopped after owned worker stop")
            if code != 0: raise RuntimeError(f"SW0187 worker exited {code}; inspect {log_path}")
            prefix_work, prefix_output = trainer._paths(trainer.OUTPUT_ROOT, "shared_prefix", seed)
            rejection_path = prefix_work / "credit_rejection.json"
            if rejection_path.is_file():
                rejection = json.loads(rejection_path.read_text(encoding="utf-8"))
                if (rejection.get("experiment") != trainer.EXPERIMENT
                        or rejection.get("status") != "scientific_rejection"
                        or rejection.get("seed") != seed
                        or rejection.get("training_fingerprint") != trainer.fingerprint_sha256()
                        or not rejection.get("rejection_reasons")
                        or rejection.get("validation_data_read") is not False):
                    raise ValueError("SW0187 prefix rejection receipt is invalid")
                return _write_state(seed, "complete_rejected", rejection_sha256=_sha(rejection_path),
                    proposal_cursor=rejection["proposal_cursor"],
                    accepted_updates=rejection["accepted_updates"],
                    rejection_reasons=rejection["rejection_reasons"],
                    physical_gpu=int(gpu), gpu_uuid=gpu_uuid, attempt=attempt,
                    queue_fingerprint_sha256=queue_sha, validation_data_read=False)
            arms = {}
            for arm in trainer.ARMS:
                _work, output = trainer._paths(trainer.OUTPUT_ROOT, arm, seed)
                arms[arm] = trainer.validate_training_artifact(arm, output, seed=seed)
            prefix = trainer.validate_prefix_artifact(prefix_output, seed=seed,
                expected_fingerprint=trainer.fingerprint_sha256())
            return _write_state(seed, "complete", arm_manifests={arm: row["manifest_sha256"]
                for arm, row in arms.items()}, prefix_manifest_sha256=_sha(prefix_output / "manifest.json"),
                prefix_state_sha256=prefix["prefix_state_sha256"], proposal_count=96,
                accepted_updates={arm: row["manifest"]["accepted_updates"] for arm, row in arms.items()},
                physical_gpu=int(gpu), gpu_uuid=gpu_uuid, attempt=attempt,
                queue_fingerprint_sha256=queue_sha, validation_data_read=False)
        raise RuntimeError("SW0187 queue stopped before terminal status")
    except BaseException as exc:
        if child is not None and child.poll() is None: owner._stop_owned(child)
        if owned_state: _write_state(seed, "failed", error=repr(exc), finished_unix=time.time())
        raise
    finally:
        if child is not None and child.poll() is None: owner._stop_owned(child)
        if lease is not None: fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close()
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN); lock.close()


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--source-seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--physical-gpu", type=int)
    args = parser.parse_args(argv)
    if args.worker:
        if args.physical_gpu not in range(4): parser.error("worker requires physical GPU 0..3")
        _worker(args.source_seed, args.physical_gpu); return 0
    if args.physical_gpu is not None: parser.error("parent does not accept --physical-gpu")
    result = run(args.source_seed)
    print(json.dumps({"seed": args.source_seed, "status": result["status"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
