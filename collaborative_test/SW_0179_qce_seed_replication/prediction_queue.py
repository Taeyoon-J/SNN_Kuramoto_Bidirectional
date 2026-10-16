"""Exclusive-GPU fixed320 prediction worker per SW0179 source seed."""
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

import numpy as np  # Keep NumPy/OpenMP initialization before legacy Torch imports.

from collaborative_test.SW_0179_qce_seed_replication import predict_evaluate as evaluate
from collaborative_test.SW_0179_qce_seed_replication import train

if _WORKER:
    if _ASSIGNED is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = _ASSIGNED

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODULE = "collaborative_test.SW_0179_qce_seed_replication.prediction_queue"
TRAIN_MODULE = "collaborative_test.SW_0179_qce_seed_replication.training_queue"
OWNER_MODULE = "collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4"
RUNTIME = HERE / "results_archive" / "prediction_queue_20261016"
STOP = threading.Event()
ACTIVE_TRAIN = {"validating_train_inputs", "waiting_for_exclusive_gpu",
                "waiting_for_stable_gpu", "running_worker", "yielding_foreign_owner"}
SOURCE_SEEDS = evaluate.SEEDS


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, value, *, replace=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with tmp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        if replace: os.replace(tmp, path)
        else: os.link(tmp, path)
    finally:
        if tmp.exists(): tmp.unlink()


def _state_path(seed):
    seed = int(seed)
    if seed not in SOURCE_SEEDS:
        raise ValueError("unregistered SW0179 prediction seed")
    return RUNTIME / f"product_q_ce_seed{seed}_state.json"


def _write_state(seed, status, **fields):
    _atomic_json(_state_path(seed), {"experiment": "SW0179_prediction_queue",
        "seed": int(seed), "arm": train.ARMS[0], "status": status,
        "supervisor_pid": os.getpid(), "updated_unix": time.time(), **fields}, replace=True)


def _load(name):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        return importlib.import_module(name)
    finally:
        if visible is None: os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else: os.environ["CUDA_VISIBLE_DEVICES"] = visible


def _module_live(owner, pid, module, arm, seed, *, worker=False):
    if type(pid) is not int or pid <= 0: return False
    record = owner._proc_record(pid)
    if not record or record.get("state") == "Z": return False
    argv = list(record.get("argv") or [])
    if argv.count("-m") != 1: return False
    pos = argv.index("-m")
    if pos + 1 >= len(argv) or argv[pos + 1] != module: return False
    if ("--worker" in argv) != bool(worker): return False
    for flag, value in (("--arm", arm), ("--source-seed", str(int(seed)))):
        if argv.count(flag) != 1 or argv[argv.index(flag) + 1:argv.index(flag) + 2] != [value]:
            return False
    return True


def _training_pending(owner, state, seed):
    return (state.get("status") in ACTIVE_TRAIN and state.get("seed") == int(seed)
            and _module_live(owner, state.get("supervisor_pid"), TRAIN_MODULE,
                             train.ARMS[0], seed, worker=False))


def _training_state(owner, seed):
    queue = _load(TRAIN_MODULE)
    path = queue._state_path(train.ARMS[0], seed)
    if not path.is_file(): return None
    state = json.loads(path.read_text(encoding="utf-8"))
    if (state.get("experiment") != "SW0179_training_queue" or state.get("arm") != train.ARMS[0]
            or state.get("seed") != int(seed)):
        raise ValueError("SW0179 training queue seed identity mismatch")
    if state.get("status") == "complete_rejected":
        rejection = train.OUTPUT_ROOT / "work" / f"{train.ARMS[0]}_seed{seed}" / "credit_rejection.json"
        if (not rejection.is_file() or state.get("artifact_sha256") != _sha(rejection)
                or state.get("validation_data_read") is not False):
            raise ValueError("SW0179 training rejection receipt is missing or changed")
        return {"status": "complete_rejected", "state": state, "state_sha256": _sha(path),
                "rejection_path": rejection, "rejection_sha256": _sha(rejection)}
    if state.get("status") != "complete":
        if _training_pending(owner, state, seed): return False
        raise RuntimeError(f"SW0179 seed{seed} training is not complete: {state.get('status')}")
    artifact = train.OUTPUT_ROOT / f"{train.ARMS[0]}_seed{seed}"
    manifest_path = artifact / "manifest.json"
    if not manifest_path.is_file(): raise FileNotFoundError("SW0179 final training manifest missing")
    manifest_sha = _sha(manifest_path)
    if (state.get("completed_updates") != train.UPDATES or state.get("manifest_sha256") != manifest_sha
            or state.get("validation_data_read") is not False):
        raise ValueError("SW0179 training queue state does not bind final artifact")
    manifest = train.validate_training_artifact(train.ARMS[0], artifact, seed=seed)
    if manifest.get("status") != "training_complete" or manifest.get("seed") != int(seed):
        raise ValueError("SW0179 training manifest source-seed mismatch")
    return {"status": "complete", "state": state, "state_sha256": _sha(path),
            "manifest": manifest, "manifest_path": manifest_path, "manifest_sha256": manifest_sha,
            "artifact": artifact}


def worker_command(seed, gpu):
    seed = int(seed)
    if seed not in SOURCE_SEEDS or type(gpu) is not int or gpu not in range(4):
        raise ValueError("SW0179 prediction worker requires registered source seed and physical GPU")
    return [sys.executable, "-u", "-m", MODULE, "--worker", "--arm", train.ARMS[0],
            "--source-seed", str(seed), "--physical-gpu", str(gpu)]


def _worker(seed, gpu):
    seed, gpu = int(seed), int(gpu)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise RuntimeError("SW0179 prediction worker visibility differs from assigned physical GPU")
    if evaluate.torch.cuda.is_initialized():
        raise RuntimeError("SW0179 worker initialized CUDA before CPU checkpoint/cache admission")
    prepared = evaluate.prepare_prediction_inputs_cpu(seed)
    if evaluate.torch.cuda.is_initialized() or os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise RuntimeError("SW0179 CPU prediction preparation changed CUDA visibility/state")
    return evaluate.predict_arm(train.ARMS[0], seed=seed, device="cuda:0", prepared=prepared)


def _worker_fingerprint(seed, prepared):
    queue = _load(TRAIN_MODULE)
    binding = evaluate._bindings(prepared, train.ARMS[0])
    return {"experiment": "SW0179_prediction_queue", "seed": int(seed),
        "arm": train.ARMS[0], "prediction_implementation_fingerprint": evaluate.implementation_fingerprint(),
        "prediction_queue_sha256": _sha(__file__),
        "training_queue_fingerprint_sha256": prepared["states"][train.ARMS[0]]["queue_fingerprint_sha256"],
        "training_queue_code_sha256": queue._sha(queue.__file__),
        "bindings_sha256": evaluate._json_sha(binding),
        "cache_mode": "USE_PYTORCH_KERNEL_CACHE=0"}


def _publish_training_rejection(seed, terminal):
    rejection = json.loads(terminal["rejection_path"].read_text(encoding="utf-8"))
    if rejection.get("bindings", {}).get("seed") != int(seed):
        raise ValueError("SW0179 rejected-training seed binding mismatch")
    result = {"status": "complete_training_rejected", "seed": int(seed),
        "training_rejection_sha256": terminal["rejection_sha256"],
        "ground_truth_used_for_prediction": False}
    _write_state(seed, result["status"], **{k:v for k,v in result.items() if k not in {"status", "seed"}})
    return result


def run(seed, *, active_timeout_seconds=8 * 3600):
    import fcntl
    seed = int(seed)
    if seed not in SOURCE_SEEDS: raise ValueError("unregistered SW0179 prediction seed")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("SW0179 prediction coordinator must keep CUDA hidden")
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    RUNTIME.mkdir(parents=True, exist_ok=True)
    state_path = _state_path(seed)
    if state_path.exists() or any(RUNTIME.glob(f"product_q_ce_seed{seed}_attempt*.log")):
        raise FileExistsError("preserving existing SW0179 prediction state/logs")
    lock = (RUNTIME / f"product_q_ce_seed{seed}.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    owner = lease = child = None
    own_state = False; attempt = 0
    try:
        _write_state(seed, "waiting_for_training", validation_data_read=False)
        own_state = True
        owner = _load(OWNER_MODULE)
        while not STOP.is_set():
            terminal = _training_state(owner, seed)
            if terminal is not False and terminal is not None: break
            _write_state(seed, "waiting_for_training", validation_data_read=False)
            STOP.wait(5)
        else:
            raise RuntimeError("SW0179 prediction queue stopped while waiting for training")
        if terminal["status"] == "complete_rejected":
            return _publish_training_rejection(seed, terminal)
        prepared = evaluate.prepare_prediction_inputs_cpu(seed)
        if prepared.get("rejected"):
            return _publish_training_rejection(seed, terminal)
        binding = evaluate._bindings(prepared, train.ARMS[0])
        bindings_sha = evaluate._json_sha(binding)
        fingerprint = _worker_fingerprint(seed, prepared)
        _write_state(seed, "waiting_for_exclusive_gpu", training_manifest_sha256=terminal["manifest_sha256"],
            training_state_sha256=terminal["state_sha256"], bindings_sha256=bindings_sha,
            implementation_fingerprint=fingerprint, validation_data_read=False)
        while not STOP.is_set():
            gpu, gpu_uuid, _used, lease = owner._acquire(STOP)
            stable = owner._stable_idle(gpu, gpu_uuid, STOP,
                lambda elapsed, memory: _write_state(seed, "waiting_for_stable_gpu",
                    physical_gpu=int(gpu), gpu_uuid=gpu_uuid, stable_seconds=float(elapsed),
                    memory_used_mib=int(memory), bindings_sha256=bindings_sha))
            if not stable:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                continue
            latest = _training_state(owner, seed)
            if latest is False or latest is None or latest.get("status") != "complete":
                raise RuntimeError("SW0179 completed training artifact changed before prediction")
            current = evaluate.prepare_prediction_inputs_cpu(seed)
            current_binding = evaluate._bindings(current, train.ARMS[0])
            current_fingerprint = _worker_fingerprint(seed, current)
            if (current_binding != binding or current_fingerprint != fingerprint
                    or latest["state_sha256"] != terminal["state_sha256"]
                    or latest["manifest_sha256"] != terminal["manifest_sha256"]):
                raise RuntimeError("SW0179 prediction inputs changed while waiting for a GPU")
            live_uuid, memory, owners = owner.gpu_guard._gpu_info(gpu)
            if live_uuid != gpu_uuid or owners or int(memory) > owner.MAX_MEMORY_MIB:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                _write_state(seed, "waiting_for_exclusive_gpu", physical_gpu=int(gpu),
                    gpu_uuid=live_uuid, memory_used_mib=int(memory), owners=list(owners),
                    bindings_sha256=bindings_sha)
                continue
            attempt += 1
            command = worker_command(seed, int(gpu))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", USE_PYTORCH_KERNEL_CACHE="0")
            log_path = RUNTIME / f"product_q_ce_seed{seed}_attempt{attempt:02d}.log"
            yielded = False
            with log_path.open("x", encoding="utf-8", newline="\n") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    pass_fds=(lease.fileno(),))
                deadline = time.monotonic() + active_timeout_seconds
                _write_state(seed, "running_prediction", child_pid=child.pid, child_argv=command,
                    physical_gpu=int(gpu), gpu_uuid=gpu_uuid, attempt=attempt,
                    implementation_fingerprint=fingerprint, bindings_sha256=bindings_sha,
                    validation_data_read=False)
                while child.poll() is None and not STOP.wait(5):
                    if time.monotonic() >= deadline:
                        owner._stop_owned(child)
                        raise TimeoutError("SW0179 owned prediction worker exceeded active runtime")
                    observed_uuid, _used_now, gpu_owners = owner.gpu_guard._gpu_info(gpu)
                    if child.poll() is not None: break
                    if observed_uuid != gpu_uuid or any(int(pid) != child.pid for pid in gpu_owners):
                        owner._stop_owned(child); child.wait(); yielded = True
                        _write_state(seed, "yielding_foreign_owner", child_pid=child.pid,
                            physical_gpu=int(gpu), gpu_uuid=gpu_uuid, observed_gpu_uuid=observed_uuid,
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
                        raise RuntimeError("SW0179 prediction child argv identity changed")
                if STOP.is_set() and child.poll() is None:
                    owner._stop_owned(child)
                rc = child.wait()
            if yielded:
                attempt_record = {"attempt": attempt, "status": "yielded_foreign_owner",
                    "physical_gpu": int(gpu), "gpu_uuid": gpu_uuid, "returncode": rc,
                    "log": log_path.name}
                old = json.loads(state_path.read_text(encoding="utf-8"))
                _write_state(seed, "waiting_for_exclusive_gpu",
                    attempts=list(old.get("attempts", [])) + [attempt_record],
                    bindings_sha256=bindings_sha, implementation_fingerprint=fingerprint)
                child = None
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                continue
            if STOP.is_set() and rc == 0: raise RuntimeError("SW0179 prediction stopped during worker completion")
            if rc != 0: raise RuntimeError(f"SW0179 prediction worker exited {rc}; inspect {log_path}")
            published = evaluate.validate_prediction(train.ARMS[0], seed=seed, prepared=current)
            result = {"status": "complete", "seed": seed, "arm": train.ARMS[0],
                "supervisor_pid": os.getpid(), "child_pid": child.pid, "physical_gpu": int(gpu),
                "gpu_uuid": gpu_uuid, "attempt": attempt,
                "prediction_sha256": published["prediction_sha256"],
                "protocol_sha256": published["protocol_sha256"], "output": str(published["directory"]),
                "bindings_sha256": bindings_sha, "implementation_fingerprint": fingerprint,
                "ground_truth_used_for_prediction": False, "completed_unix": time.time()}
            _write_state(seed, "complete", **{k:v for k,v in result.items() if k not in {"status", "seed"}})
            return result
        raise RuntimeError("SW0179 prediction supervisor stopped")
    except BaseException as exc:
        if child is not None and child.poll() is None and owner is not None: owner._stop_owned(child)
        if own_state: _write_state(seed, "failed", error=repr(exc))
        raise
    finally:
        if child is not None and child.poll() is None and owner is not None: owner._stop_owned(child)
        if lease is not None:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close()
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN); lock.close()


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--arm", choices=train.ARMS, required=True)
    parser.add_argument("--source-seed", type=int, choices=SOURCE_SEEDS, required=True)
    parser.add_argument("--physical-gpu", type=int)
    args = parser.parse_args(argv)
    if args.arm != train.ARMS[0]: parser.error("SW0179 registers only product_q_ce")
    if args.worker:
        if args.physical_gpu not in range(4): parser.error("worker requires physical GPU")
        _worker(args.source_seed, args.physical_gpu)
    else:
        print(json.dumps(run(args.source_seed), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
