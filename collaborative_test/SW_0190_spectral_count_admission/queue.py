"""Exclusive-owner queue for one SW0190 TRAIN16 source seed per worker."""
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
_ASSIGNED = os.environ.get("CUDA_VISIBLE_DEVICES") if _WORKER else None
if not _WORKER:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ["USE_PYTORCH_KERNEL_CACHE"] = "0"

import numpy as np  # Keep NumPy ahead of legacy Torch imports.


def _import_preserving_visibility(name):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        return importlib.import_module(name)
    finally:
        if visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible


run_api = _import_preserving_visibility(
    "collaborative_test.SW_0190_spectral_count_admission.run")
if _WORKER:
    if _ASSIGNED is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = _ASSIGNED
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODULE = "collaborative_test.SW_0190_spectral_count_admission.queue"
OWNER_MODULE = "collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4"
RUNTIME = HERE / "results_archive" / "train16_queue_20261017"
REPORT_ROOT = HERE / "results_archive" / "train16_replay_20261017"
STOP = threading.Event()
TRAIN_ACTIVE = {"validating_train_inputs", "validating_train_cache", "waiting_for_exclusive_gpu",
    "waiting_for_stable_gpu", "running_worker", "yielding_foreign_owner", "waiting_for_training"}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, value, *, replace=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        if replace:
            os.replace(temp, path)
        else:
            os.link(temp, path)
    finally:
        if temp.exists(): temp.unlink()


def _state_path(seed):
    if int(seed) not in run_api.SEEDS:
        raise ValueError("unregistered SW0190 source seed")
    return RUNTIME / f"seed{int(seed)}_state.json"


def _write_state(seed, status, **fields):
    _atomic_json(_state_path(seed), {"experiment": "SW0190_train16_queue",
        "seed": int(seed), "status": status, "supervisor_pid": os.getpid(),
        "updated_unix": time.time(), **fields}, replace=True)


def _load(name):
    return _import_preserving_visibility(name)


def _training_ready(owner):
    trainer = run_api.train
    queue = _load(trainer.QUEUE_MODULE)
    states = {}
    for seed in run_api.SEEDS:
        path = queue._state_path(seed)
        state = json.loads(path.read_text(encoding="utf-8"))
        if (state.get("status") != "complete" or state.get("seed") != seed
                or state.get("experiment") != "SW0186_training_queue"
                or state.get("proposal_cursor") != 64
                or state.get("validation_data_read") is not False):
            raise ValueError("SW0190 requires frozen complete SW0183 training")
        states[str(seed)] = {"path": str(path), "sha256": _sha(path),
            "manifest_sha256": state["manifest_sha256"]}
    return states


def worker_command(seed, physical_gpu):
    if int(seed) not in run_api.SEEDS or type(physical_gpu) is not int or physical_gpu not in range(4):
        raise ValueError("SW0190 worker requires registered seed and physical GPU")
    return [sys.executable, "-u", "-m", MODULE, "--worker", "--seed", str(int(seed)),
            "--physical-gpu", str(physical_gpu)]


def _worker(seed, physical_gpu):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("SW0190 worker visibility differs from assigned physical GPU")
    prepared = run_api.prepare_inputs_cpu(seed_subset=(int(seed),))
    if run_api.torch.cuda.is_initialized() or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("SW0190 CPU admission initialized CUDA or changed visibility")
    run_api.torch.set_num_threads(1)
    run_api.torch.cuda.set_device(torch_device := run_api.torch.device("cuda:0"))
    report = run_api.run_diagnostic(prepared, seed_subset=(int(seed),), device=torch_device)
    if report.get("optimizer_updates") != 0 or report.get("validation_data_read") is not False:
        raise ValueError("SW0190 replay violated its no-update/ TRAIN-only scope")
    output = REPORT_ROOT / f"seed{int(seed)}" / "train16_replay.json"
    run_api.write_report(output, report)
    return {"path": str(output), "sha256": _sha(output), "report": report}


def run(seed, *, active_timeout_seconds=6 * 3600):
    import fcntl
    seed = int(seed)
    if seed not in run_api.SEEDS:
        raise ValueError("unregistered SW0190 seed")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("SW0190 supervisor must keep CUDA hidden")
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    RUNTIME.mkdir(parents=True, exist_ok=True)
    state_path = _state_path(seed)
    if state_path.exists() or any(RUNTIME.glob(f"seed{seed}_attempt*.log")):
        raise FileExistsError("preserving existing SW0190 queue state/logs")
    lock = (RUNTIME / f"seed{seed}.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    owner = lease = child = None
    own_state = False
    attempt = 0
    try:
        _write_state(seed, "waiting_for_training", validation_data_read=False)
        own_state = True
        owner = _load(OWNER_MODULE)
        while not STOP.is_set():
            train_states = _training_ready(owner)
            if train_states is not False:
                break
            _write_state(seed, "waiting_for_training", validation_data_read=False)
            STOP.wait(5)
        else:
            raise RuntimeError("SW0190 queue stopped while waiting for training")
        prepared = run_api.prepare_inputs_cpu(seed_subset=(seed,))
        report_fp = run_api.implementation_fingerprint()
        queue_fp = {"experiment": "SW0190_train16_queue", "seed": seed,
                    "run_fingerprint": report_fp, "queue_sha256": _sha(__file__),
                    "training_states": train_states, "cache0": True}
        queue_sha = hashlib.sha256(json.dumps(queue_fp, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        _write_state(seed, "waiting_for_exclusive_gpu", queue_fingerprint_sha256=queue_sha,
                     implementation_fingerprint=report_fp, training_states=train_states,
                     validation_data_read=False)
        del prepared
        while not STOP.is_set():
            gpu, gpu_uuid, _used, lease = owner._acquire(STOP)
            if STOP.is_set():
                break
            stable = owner._stable_idle(gpu, gpu_uuid, STOP,
                lambda elapsed, memory: _write_state(seed, "waiting_for_stable_gpu",
                    physical_gpu=int(gpu), gpu_uuid=gpu_uuid, stable_seconds=float(elapsed),
                    memory_used_mib=int(memory), queue_fingerprint_sha256=queue_sha))
            if not stable:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                continue
            observed_uuid, memory, owners = owner.gpu_guard._gpu_info(gpu)
            if observed_uuid != gpu_uuid or owners or int(memory) > owner.MAX_MEMORY_MIB:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                _write_state(seed, "waiting_for_exclusive_gpu", gpu_uuid=observed_uuid,
                    owners=list(owners), memory_used_mib=int(memory), queue_fingerprint_sha256=queue_sha)
                continue
            attempt += 1
            command = worker_command(seed, int(gpu))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", USE_PYTORCH_KERNEL_CACHE="0")
            log_path = RUNTIME / f"seed{seed}_attempt{attempt:02d}.log"
            yielded = False
            with log_path.open("x", encoding="utf-8", newline="\n") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    pass_fds=(lease.fileno(),))
                deadline = time.monotonic() + active_timeout_seconds
                _write_state(seed, "running_worker", child_pid=child.pid, child_argv=command,
                    physical_gpu=int(gpu), gpu_uuid=gpu_uuid, attempt=attempt,
                    queue_fingerprint_sha256=queue_sha, validation_data_read=False)
                while child.poll() is None and not STOP.wait(5):
                    if time.monotonic() >= deadline:
                        owner._stop_owned(child)
                        raise TimeoutError("SW0190 owned TRAIN16 replay exceeded active timeout")
                    live_uuid, _mem_now, gpu_owners = owner.gpu_guard._gpu_info(gpu)
                    if child.poll() is not None:
                        break
                    if live_uuid != gpu_uuid or any(int(pid) != child.pid for pid in gpu_owners):
                        owner._stop_owned(child); child.wait(); yielded = True
                        _write_state(seed, "yielding_foreign_owner", child_pid=child.pid,
                            physical_gpu=int(gpu), gpu_uuid=gpu_uuid, observed_gpu_uuid=live_uuid,
                            owners=list(gpu_owners), attempt=attempt)
                        break
                    proc = owner._proc_record(child.pid) if child.poll() is None else None
                    if proc is None or proc.get("state") == "Z":
                        try: child.wait(timeout=1)
                        except subprocess.TimeoutExpired: pass
                        if child.poll() is not None: break
                        continue
                    argv = proc.get("argv")
                    if argv and list(argv) != command:
                        owner._stop_owned(child)
                        raise RuntimeError("SW0190 owned worker argv identity changed")
                if STOP.is_set() and child.poll() is None:
                    owner._stop_owned(child)
                rc = child.wait()
            # A complete atomic report may precede the foreign-owner observation.
            if yielded and (REPORT_ROOT / f"seed{seed}" / "train16_replay.json").is_file():
                yielded=False;rc=0
            if yielded:
                old = json.loads(state_path.read_text(encoding="utf-8"))
                attempts = list(old.get("attempts", [])) + [{"attempt": attempt,
                    "status": "yielded_foreign_owner", "physical_gpu": int(gpu),
                    "gpu_uuid": gpu_uuid, "returncode": rc, "log": log_path.name}]
                _write_state(seed, "waiting_for_exclusive_gpu", attempts=attempts,
                    queue_fingerprint_sha256=queue_sha)
                child = None
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                continue
            if STOP.is_set():
                raise RuntimeError("SW0190 queue stopped during worker")
            if rc != 0:
                raise RuntimeError(f"SW0190 worker exited {rc}; inspect {log_path}")
            output = REPORT_ROOT / f"seed{seed}" / "train16_replay.json"
            report = json.loads(output.read_text(encoding="utf-8"))
            if (report.get("experiment") != run_api.EXPERIMENT or report.get("status") != "complete"
                    or set(report.get("seeds", {})) != {str(seed)}
                    or report.get("optimizer_updates") != 0
                    or report.get("validation_data_read") is not False
                    or report.get("implementation_fingerprint") != report_fp):
                raise ValueError("SW0190 worker report fails its producer contract")
            report_sha = _sha(output)
            terminal = {"status": "complete", "report_path": str(output),
                "report_sha256": report_sha, "queue_fingerprint_sha256": queue_sha,
                "implementation_fingerprint": report_fp, "physical_gpu": int(gpu),
                "gpu_uuid": gpu_uuid, "attempt": attempt, "optimizer_updates": 0,
                "validation_data_read": False}
            _publish_complete_state(seed, terminal)
            return terminal
        raise RuntimeError("SW0190 supervisor stopped before completing the TRAIN16 replay")
    except BaseException as exc:
        if child is not None and child.poll() is None and owner is not None:
            owner._stop_owned(child)
        if own_state:
            _write_state(seed, "failed", error=repr(exc))
        raise
    finally:
        if child is not None and child.poll() is None and owner is not None:
            owner._stop_owned(child)
        if lease is not None:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close()
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN); lock.close()


def _worker_entry(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--seed", type=int, choices=run_api.SEEDS, required=True)
    parser.add_argument("--physical-gpu", type=int)
    args = parser.parse_args(argv)
    if args.worker:
        if args.physical_gpu not in range(4):
            parser.error("worker requires --physical-gpu in [0,3]")
        _worker(args.seed, args.physical_gpu)
        return 0
    result = run(args.seed)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


def _publish_complete_state(seed, terminal):
    if terminal.get("status") != "complete" or int(terminal.get("seed", seed)) != int(seed):
        raise ValueError("SW0190 terminal result identity/status mismatch")
    fields = {key: value for key, value in terminal.items() if key not in {"status", "seed"}}
    _write_state(seed, "complete", **fields)
    return fields


if __name__ == "__main__":
    raise SystemExit(_worker_entry())
