"""Exclusive-GPU prediction supervisors for completed SW0185 seeds."""
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
_ASSIGNED = os.environ.get("CUDA_VISIBLE_DEVICES") if _WORKER else None
if not _WORKER:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ["USE_PYTORCH_KERNEL_CACHE"] = "0"

import numpy as np

from collaborative_test.SW_0185_qce_spatial_connectivity import predict_evaluate as evaluate
from collaborative_test.SW_0185_qce_spatial_connectivity import train

if _WORKER:
    if _ASSIGNED is None: os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else: os.environ["CUDA_VISIBLE_DEVICES"] = _ASSIGNED

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODULE = "collaborative_test.SW_0185_qce_spatial_connectivity.prediction_queue"
TRAIN_MODULE = "collaborative_test.SW_0185_qce_spatial_connectivity.training_queue"
OWNER_MODULE = "collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4"
RUNTIME = HERE / "results_archive" / "prediction_queue_20261016"
STOP = threading.Event()
SEEDS = evaluate.SEEDS
ACTIVE_TRAIN = {"waiting_for_successful_source_preflight", "waiting_for_exclusive_gpu",
    "waiting_for_stable_gpu", "running_worker", "yielding_foreign_owner"}


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def _load(name):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try: return importlib.import_module(name)
    finally:
        if visible is None: os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else: os.environ["CUDA_VISIBLE_DEVICES"] = visible


def _state_path(seed):
    if int(seed) not in SEEDS: raise ValueError("unregistered SW0185 prediction seed")
    return RUNTIME / f"qce_spatial_connectivity_seed{int(seed)}_state.json"


def _write_state(seed, status, **fields):
    path = _state_path(seed); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    state = {"experiment": "SW0185_prediction_queue", "seed": int(seed), "arm": evaluate.ARM,
        "status": status, "supervisor_pid": os.getpid(), "updated_unix": time.time(), **fields}
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(state, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists(): temp.unlink()
    return state


def _module_live(owner, pid, module, seed, *, worker):
    if type(pid) is not int or pid <= 0: return False
    record = owner._proc_record(pid)
    if not record or record.get("state") == "Z": return False
    argv = list(record.get("argv") or [])
    if argv.count("-m") != 1: return False
    pos = argv.index("-m")
    if pos + 1 >= len(argv) or argv[pos + 1] != module: return False
    if ("--worker" in argv) != bool(worker): return False
    for flag, value in (("--source-seed", str(int(seed))),):
        if argv.count(flag) != 1 or argv[argv.index(flag) + 1:argv.index(flag) + 2] != [value]:
            return False
    return True


def _training_terminal(owner, seed):
    training_queue = _load(TRAIN_MODULE)
    path = training_queue._state_path(seed)
    if not path.is_file(): return None
    state = json.loads(path.read_text(encoding="utf-8"))
    if (state.get("experiment") != "SW0185_training_queue" or state.get("arm") != evaluate.ARM
            or state.get("seed") != int(seed)):
        raise ValueError("SW0185 prediction saw a mismatched training state")
    if state.get("status") == "complete_rejected":
        work, _ = train._artifact_paths(train.OUTPUT_ROOT, evaluate.ARM, seed)
        receipt = work / "credit_rejection.json"
        if (not receipt.is_file() or state.get("artifact_sha256") != _sha(receipt)
                or state.get("validation_data_read") is not False):
            raise ValueError("SW0185 terminal rejection receipt is missing or changed")
        return {"status": "complete_rejected", "state": state,
            "state_path": path, "state_sha256": _sha(path), "rejection_path": receipt,
            "rejection_sha256": _sha(receipt)}
    if state.get("status") != "complete":
        if (state.get("status") in ACTIVE_TRAIN and state.get("seed") == int(seed)
                and _module_live(owner, state.get("supervisor_pid"), TRAIN_MODULE, seed, worker=False)):
            return False
        raise RuntimeError(f"SW0185 seed{seed} training state is not live or terminal-complete")
    work, artifact = train._artifact_paths(train.OUTPUT_ROOT, evaluate.ARM, seed)
    manifest_path = artifact / "manifest.json"
    manifest = train.validate_training_artifact(evaluate.ARM, artifact, seed=seed)
    manifest_sha = _sha(manifest_path)
    if (state.get("manifest_sha256") != manifest_sha or state.get("artifact_sha256") != manifest_sha
            or state.get("proposal_cursor") != train.PROPOSALS
            or state.get("validation_data_read") is not False):
        raise ValueError("SW0185 training state does not bind a complete final artifact")
    return {"status": "complete", "state": state, "state_path": path,
        "state_sha256": _sha(path), "manifest": manifest, "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha, "artifact": artifact}


def worker_command(seed, physical_gpu):
    if int(seed) not in SEEDS or type(physical_gpu) is not int or physical_gpu not in range(4):
        raise ValueError("SW0185 worker requires registered seed and physical GPU")
    return [sys.executable, "-u", "-m", MODULE, "--worker", "--source-seed", str(int(seed)),
        "--physical-gpu", str(physical_gpu)]


def _worker(seed, gpu):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu) or evaluate.torch.cuda.is_initialized():
        raise RuntimeError("SW0185 worker visibility/initialization is invalid before CPU admission")
    prepared = evaluate.prepare_prediction_inputs_cpu(seed)
    if evaluate.torch.cuda.is_initialized() or os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise RuntimeError("SW0185 CPU prediction preparation changed CUDA state/visibility")
    result = evaluate.predict_seed(seed, device="cuda:0", prepared=prepared)
    print(json.dumps({"seed": int(seed), "status": "complete",
        "prediction_sha256": result["prediction_sha256"]}, sort_keys=True), flush=True)
    return result


def _worker_fingerprint(seed, prepared):
    binding = evaluate._bindings(prepared)
    training_queue = _load(TRAIN_MODULE)
    return {"experiment": "SW0185_prediction_queue", "seed": int(seed),
        "arm": evaluate.ARM, "prediction_fingerprint": evaluate.implementation_fingerprint(),
        "prediction_queue_sha256": _sha(__file__),
        "training_queue_fingerprint_sha256": prepared["queue_fingerprint_sha256"],
        "training_queue_code_sha256": training_queue._sha(training_queue.__file__),
        "bindings_sha256": evaluate._sha_json(binding), "cache_mode": "USE_PYTORCH_KERNEL_CACHE=0"}


def _record_training_rejection(seed, terminal):
    result = {"status": "complete_training_rejected", "seed": int(seed),
        "training_rejection_sha256": terminal["rejection_sha256"],
        "ground_truth_used_for_prediction": False}
    _write_state(seed, result["status"], **{key: value for key, value in result.items()
        if key not in {"status", "seed"}})
    return result


def run(seed, *, active_worker_timeout_seconds=8 * 3600):
    import fcntl
    seed = int(seed)
    if seed not in SEEDS or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("SW0185 prediction parent needs registered seed and hidden CUDA")
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    RUNTIME.mkdir(parents=True, exist_ok=True)
    state_path = _state_path(seed)
    if state_path.exists() or any(RUNTIME.glob(f"qce_spatial_connectivity_seed{seed}_attempt*.log")):
        raise FileExistsError("preserving existing SW0185 prediction runtime state/logs")
    lock = (RUNTIME / f"qce_spatial_connectivity_seed{seed}.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    owner = lease = child = None; own_state = False; attempt = 0
    try:
        _write_state(seed, "waiting_for_training", validation_data_read=False); own_state = True
        owner = _load(OWNER_MODULE)
        while not STOP.is_set():
            terminal = _training_terminal(owner, seed)
            if terminal is not False and terminal is not None: break
            _write_state(seed, "waiting_for_training", validation_data_read=False)
            STOP.wait(5)
        else:
            raise RuntimeError("SW0185 prediction stopped while waiting for training")
        if terminal["status"] == "complete_rejected": return _record_training_rejection(seed, terminal)
        prepared = evaluate.prepare_prediction_inputs_cpu(seed)
        worker_fp = _worker_fingerprint(seed, prepared)
        bindings_sha = evaluate._sha_json(evaluate._bindings(prepared))
        _write_state(seed, "waiting_for_exclusive_gpu",
            training_manifest_sha256=terminal["manifest_sha256"],
            training_state_sha256=terminal["state_sha256"], bindings_sha256=bindings_sha,
            implementation_fingerprint=worker_fp, validation_data_read=False)
        while not STOP.is_set():
            gpu, gpu_uuid, _used, lease = owner._acquire(STOP)
            stable = owner._stable_idle(gpu, gpu_uuid, STOP,
                lambda elapsed, memory: _write_state(seed, "waiting_for_stable_gpu",
                    physical_gpu=int(gpu), gpu_uuid=gpu_uuid, stable_seconds=float(elapsed),
                    memory_used_mib=int(memory), bindings_sha256=bindings_sha))
            if not stable:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                continue
            current = _training_terminal(owner, seed)
            if current is None or current is False or current.get("status") != "complete" \
                    or current["manifest_sha256"] != terminal["manifest_sha256"] \
                    or current["state_sha256"] != terminal["state_sha256"]:
                raise RuntimeError("SW0185 training artifact/state changed while prediction was queued")
            attempt += 1
            command = worker_command(seed, int(gpu))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", USE_PYTORCH_KERNEL_CACHE="0")
            log_path = RUNTIME / f"qce_spatial_connectivity_seed{seed}_attempt{attempt:02d}.log"
            yielded = False; deadline = time.monotonic() + active_worker_timeout_seconds
            with log_path.open("x", encoding="utf-8", newline="\n") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    pass_fds=(lease.fileno(),))
                _write_state(seed, "running_worker", child_pid=child.pid, child_pgid=child.pid,
                    child_argv=command, physical_gpu=int(gpu), gpu_uuid=gpu_uuid, attempt=attempt,
                    bindings_sha256=bindings_sha)
                while child.poll() is None and not STOP.is_set():
                    if time.monotonic() >= deadline:
                        owner._stop_owned(child); raise TimeoutError("SW0185 prediction active worker timed out")
                    live_uuid, _memory, active_owners = owner.gpu_guard._gpu_info(gpu)
                    if child.poll() is not None: break
                    if live_uuid != gpu_uuid or any(int(pid) != child.pid for pid in active_owners):
                        owner._stop_owned(child); yielded = True
                        _write_state(seed, "yielding_foreign_owner", child_pid=child.pid,
                            physical_gpu=int(gpu), gpu_uuid=gpu_uuid, owners=list(active_owners))
                        break
                    record = owner._proc_record(child.pid)
                    if record and record.get("state") != "Z":
                        argv = list(record.get("argv") or [])
                        if argv != command:
                            owner._stop_owned(child); raise RuntimeError("SW0185 prediction worker argv changed")
                    STOP.wait(3)
                if STOP.is_set() and child.poll() is None: owner._stop_owned(child)
                return_code = child.wait()
            if yielded:
                child = None; fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                _write_state(seed, "waiting_for_exclusive_gpu", yielded_attempt=attempt,
                    bindings_sha256=bindings_sha); continue
            if STOP.is_set(): raise RuntimeError("SW0185 prediction stopped its owned worker")
            if return_code != 0: raise RuntimeError(f"SW0185 prediction worker exited {return_code}; see {log_path}")
            result = evaluate.validate_prediction(seed, prepared=prepared)
            final_state = _write_state(seed, "complete", prediction_sha256=result["prediction_sha256"],
                protocol_sha256=result["protocol_sha256"], training_manifest_sha256=terminal["manifest_sha256"],
                training_state_sha256=terminal["state_sha256"], bindings_sha256=bindings_sha,
                implementation_fingerprint=worker_fp, ground_truth_used_for_prediction=False,
                validation_data_read=False)
            final_state["state_sha256"] = _sha(state_path)
            return final_state
        raise RuntimeError("SW0185 prediction queue stopped before terminal state")
    except BaseException as exc:
        if child is not None and child.poll() is None: owner._stop_owned(child)
        if own_state: _write_state(seed, "failed", error=repr(exc), validation_data_read=False)
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
    if args.physical_gpu is not None: parser.error("supervisor does not accept --physical-gpu")
    result = run(args.source_seed)
    print(json.dumps({"seed": args.source_seed, "status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
