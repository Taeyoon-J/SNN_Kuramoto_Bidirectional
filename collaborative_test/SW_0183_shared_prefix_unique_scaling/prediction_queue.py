"""Exclusive per-seed prediction supervisor for both frozen SW0183 arms."""
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
if not _WORKER: os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1"); os.environ["USE_PYTORCH_KERNEL_CACHE"] = "0"

_visible_before_import = os.environ.get("CUDA_VISIBLE_DEVICES")
try:
    evaluate = importlib.import_module(
        "collaborative_test.SW_0183_shared_prefix_unique_scaling.predict_evaluate")
finally:
    if _visible_before_import is None: os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else: os.environ["CUDA_VISIBLE_DEVICES"] = _visible_before_import

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
MODULE = "collaborative_test.SW_0183_shared_prefix_unique_scaling.prediction_queue"
TRAIN_MODULE = "collaborative_test.SW_0183_shared_prefix_unique_scaling.training_queue"
OWNER_MODULE = "collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4"
RUNTIME = HERE / "results_archive" / "prediction_queue_20261016"
SEEDS = evaluate.SEEDS; STOP = threading.Event()
ACTIVE_TRAIN = {"validating_train_inputs", "waiting_for_exclusive_gpu", "waiting_for_stable_gpu",
    "running_worker", "yielding_foreign_owner"}


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def _module(name):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try: return importlib.import_module(name)
    finally:
        if visible is None: os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else: os.environ["CUDA_VISIBLE_DEVICES"] = visible


def state_path(seed): return RUNTIME / f"seed{int(seed)}_state.json"


def _write_state(seed, status, **fields):
    path = state_path(seed); path.parent.mkdir(parents=True, exist_ok=True)
    value = {"experiment": "SW0183_prediction_queue", "seed": int(seed), "status": status,
        "supervisor_pid": os.getpid(), "updated_unix": time.time(), **fields}
    temp = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists(): temp.unlink()
    return value


def _module_live(owner, pid, module, seed, *, worker=False):
    if type(pid) is not int or pid <= 0: return False
    proc = owner._proc_record(pid)
    if not proc or proc.get("state") == "Z": return False
    argv = list(proc.get("argv") or [])
    if argv.count("-m") != 1: return False
    pos = argv.index("-m")
    if pos + 1 >= len(argv) or argv[pos + 1] != module or (("--worker" in argv) != worker): return False
    return argv.count("--source-seed") == 1 and argv[argv.index("--source-seed") + 1:][:1] == [str(int(seed))]


def training_terminal(owner, seed, state=None):
    tq = _module(TRAIN_MODULE); path = tq.state_path(seed)
    if state is None:
        if not path.is_file(): return None
        state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("experiment") != "SW0183_training_queue" or state.get("seed") != int(seed):
        raise ValueError("SW0183 training queue state identity mismatch")
    if state.get("status") == "complete_rejected":
        return {"status": "complete_rejected", "state": state, "path": path, "sha256": _sha(path)}
    if state.get("status") != "complete":
        if state.get("status") in ACTIVE_TRAIN and _module_live(owner, state.get("supervisor_pid"),
                TRAIN_MODULE, seed, worker=False): return False
        raise RuntimeError(f"SW0183 training seed{seed} is not live or complete")
    if (state.get("validation_data_read") is not False or state.get("proposal_count") != 96
            or set(state.get("arm_manifests", {})) != set(evaluate.ARMS)):
        raise ValueError("SW0183 training terminal state lacks both arms or violates GT-last scope")
    return {"status": "complete", "state": state, "path": path, "sha256": _sha(path)}


def worker_command(seed, gpu):
    if int(seed) not in SEEDS or type(gpu) is not int or gpu not in range(4):
        raise ValueError("SW0183 prediction worker assignment is invalid")
    return [sys.executable, "-u", "-m", MODULE, "--worker", "--source-seed", str(int(seed)),
        "--physical-gpu", str(gpu)]


def _worker(seed, gpu):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu) or evaluate.torch.cuda.is_initialized():
        raise RuntimeError("SW0183 prediction worker visibility must be set before CUDA initialization")
    prepared = evaluate.prepare_prediction_inputs_cpu(seed)
    if evaluate.torch.cuda.is_initialized() or os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise RuntimeError("SW0183 CPU prediction preparation changed CUDA state")
    results = {arm: evaluate.predict_arm(seed, arm, device="cuda:0", prepared=prepared)
               for arm in evaluate.ARMS}
    print(json.dumps({"seed": int(seed), "status": "complete", "predictions": {
        arm: {"prediction_sha256": row["prediction_sha256"], "protocol_sha256": row["protocol_sha256"]}
        for arm, row in results.items()}}, sort_keys=True), flush=True)
    return results


def _worker_fp(seed, prepared):
    return {"experiment": "SW0183_prediction_queue", "seed": int(seed),
        "prediction_fingerprint": evaluate.implementation_fingerprint(),
        "queue_sha256": _sha(__file__), "training_state_sha256": prepared["queue_state_sha256"],
        "training_queue_fingerprint_sha256": prepared["queue_fingerprint_sha256"],
        "branch_manifests": {arm: prepared["artifacts"][arm]["manifest_sha256"] for arm in evaluate.ARMS},
        "cache_mode": "USE_PYTORCH_KERNEL_CACHE=0"}


def _lease_idle(owner, gpu, expected_uuid):
    current_uuid, used, owners = owner.gpu_guard._gpu_info(int(gpu))
    return current_uuid == expected_uuid and not owners and int(used) <= int(owner.MAX_MEMORY_MIB)


def run(seed, *, active_worker_timeout_seconds=8 * 3600, poll_seconds=5):
    import fcntl
    seed = int(seed)
    if seed not in SEEDS or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("SW0183 prediction supervisor requires a registered seed and hidden CUDA")
    signal.signal(signal.SIGTERM, lambda *_: STOP.set()); signal.signal(signal.SIGINT, lambda *_: STOP.set())
    RUNTIME.mkdir(parents=True, exist_ok=True)
    if state_path(seed).exists() or any(RUNTIME.glob(f"seed{seed}_attempt*.log")):
        raise FileExistsError("preserving existing SW0183 prediction state/logs")
    lock = (RUNTIME / f"seed{seed}.lock").open("a+"); fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    owner = lease = child = None; own_state = False; attempt = 0
    try:
        _write_state(seed, "waiting_for_training", validation_data_read=False); own_state = True
        owner = _module(OWNER_MODULE)
        while not STOP.is_set():
            terminal = training_terminal(owner, seed)
            if terminal is not False and terminal is not None: break
            _write_state(seed, "waiting_for_training", validation_data_read=False); STOP.wait(poll_seconds)
        else: raise RuntimeError("SW0183 prediction stopped while waiting for training")
        if terminal["status"] == "complete_rejected":
            return _write_state(seed, "complete_training_rejected", training_state_sha256=terminal["sha256"],
                ground_truth_used_for_prediction=False, validation_data_read=False)
        prepared = evaluate.prepare_prediction_inputs_cpu(seed)
        fp = _worker_fp(seed, prepared)
        _write_state(seed, "waiting_for_exclusive_gpu", training_state_sha256=terminal["sha256"],
            implementation_fingerprint=fp, validation_data_read=False)
        while not STOP.is_set():
            gpu, gpu_uuid, _used, lease = owner._acquire(STOP)
            stable = owner._stable_idle(gpu, gpu_uuid, STOP, lambda elapsed, memory: _write_state(seed,
                "waiting_for_stable_gpu", physical_gpu=int(gpu), gpu_uuid=gpu_uuid,
                stable_seconds=float(elapsed), memory_used_mib=int(memory)))
            if not stable:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None; continue
            current = training_terminal(owner, seed)
            if current is False or current is None or current["status"] != "complete" or current["sha256"] != terminal["sha256"]:
                raise RuntimeError("SW0183 training state changed while prediction waited")
            if not _lease_idle(owner, gpu, gpu_uuid):
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None; continue
            attempt += 1; command = worker_command(seed, int(gpu)); yielded = False
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", USE_PYTORCH_KERNEL_CACHE="0")
            log_path = RUNTIME / f"seed{seed}_attempt{attempt:02d}.log"
            deadline = time.monotonic() + active_worker_timeout_seconds
            with log_path.open("x", encoding="utf-8", newline="\n") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lease.fileno(),))
                _write_state(seed, "running_worker", child_pid=child.pid, child_pgid=child.pid,
                    child_argv=command, physical_gpu=int(gpu), gpu_uuid=gpu_uuid, attempt=attempt)
                while child.poll() is None and not STOP.is_set():
                    if time.monotonic() >= deadline:
                        owner._stop_owned(child); raise TimeoutError("SW0183 prediction worker timed out")
                    live_uuid, _memory, owners = owner.gpu_guard._gpu_info(gpu)
                    if child.poll() is not None: break
                    if live_uuid != gpu_uuid or any(int(pid) != child.pid for pid in owners):
                        owner._stop_owned(child); yielded = True
                        _write_state(seed, "yielding_foreign_owner", child_pid=child.pid,
                            physical_gpu=int(gpu), gpu_uuid=gpu_uuid, owners=list(owners)); break
                    record = owner._proc_record(child.pid)
                    if record and record.get("state") != "Z" and list(record.get("argv") or []) != command:
                        owner._stop_owned(child); raise RuntimeError("SW0183 prediction worker argv changed")
                    STOP.wait(3)
                if STOP.is_set() and child.poll() is None: owner._stop_owned(child)
                code = child.wait()
            if yielded:
                child = None; fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                _write_state(seed, "waiting_for_exclusive_gpu", yielded_attempt=attempt); continue
            if STOP.is_set(): raise RuntimeError("SW0183 prediction stopped its owned worker")
            if code != 0: raise RuntimeError(f"SW0183 prediction worker exited {code}; inspect {log_path}")
            results = {arm: evaluate.validate_prediction(seed, arm, prepared=prepared) for arm in evaluate.ARMS}
            return _write_state(seed, "complete", predictions={arm: {
                "prediction_sha256": row["prediction_sha256"], "protocol_sha256": row["protocol_sha256"]}
                for arm, row in results.items()}, training_state_sha256=terminal["sha256"],
                training_manifest_sha256={arm: prepared["artifacts"][arm]["manifest_sha256"] for arm in evaluate.ARMS},
                implementation_fingerprint=fp, ground_truth_used_for_prediction=False,
                validation_data_read=False, physical_gpu=int(gpu), gpu_uuid=gpu_uuid)
        raise RuntimeError("SW0183 prediction stopped before terminal state")
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
    parser = argparse.ArgumentParser(); parser.add_argument("--worker", action="store_true")
    parser.add_argument("--source-seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--physical-gpu", type=int)
    args = parser.parse_args(argv)
    if args.worker:
        if args.physical_gpu not in range(4) or os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
            parser.error("worker physical GPU must match leased visibility")
        _worker(args.source_seed, args.physical_gpu); return 0
    if args.physical_gpu is not None: parser.error("supervisor does not accept --physical-gpu")
    result = run(args.source_seed); print(json.dumps({"seed": args.source_seed, "status": result["status"]})); return 0


if __name__ == "__main__": raise SystemExit(main())
