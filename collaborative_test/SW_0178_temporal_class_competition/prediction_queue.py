"""Owner-safe resumable fixed320 prediction queues for both SW0178 arms."""
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

import numpy as np  # NumPy/OpenMP must precede legacy Torch imports.

from collaborative_test.SW_0178_temporal_class_competition import predict_evaluate as evaluate
from collaborative_test.SW_0178_temporal_class_competition import train

if _WORKER:
    if _ASSIGNED is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = _ASSIGNED

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODULE = "collaborative_test.SW_0178_temporal_class_competition.prediction_queue"
TRAIN_MODULE = "collaborative_test.SW_0178_temporal_class_competition.training_queue"
OWNER_MODULE = "collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4"
RUNTIME = HERE / "results_archive" / "prediction_queue_v1_20261016"
STOP = threading.Event()
ACTIVE_TRAIN = {"validating_train_cache", "waiting_for_exclusive_gpu",
                "waiting_for_stable_gpu", "running_worker", "yielding_foreign_owner"}


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


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


def _state_path(arm):
    if arm not in train.ARMS: raise ValueError("unregistered SW0178 arm")
    return RUNTIME / f"{arm}_state.json"


def _write_state(arm, status, **fields):
    _atomic_json(_state_path(arm), {"experiment": "SW0178_prediction_queue_v1",
        "arm": arm, "status": status, "supervisor_pid": os.getpid(),
        "updated_unix": time.time(), **fields}, replace=True)


def _load(name):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        return importlib.import_module(name)
    finally:
        if visible is None: os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else: os.environ["CUDA_VISIBLE_DEVICES"] = visible


def _module_live(owner, pid, module, arm):
    if type(pid) is not int or pid <= 0: return False
    rec = owner._proc_record(pid)
    if not rec or rec.get("state") == "Z": return False
    argv = rec.get("argv", [])
    if argv.count("-m") != 1: return False
    pos = argv.index("-m")
    if pos + 1 >= len(argv) or argv[pos + 1] != module or "--worker" in argv:
        return False
    return argv.count("--arm") == 1 and argv[argv.index("--arm") + 1:argv.index("--arm") + 2] == [arm]


def _training_is_pending(owner, state, arm):
    return (state.get("status") in ACTIVE_TRAIN
            and _module_live(owner, state.get("supervisor_pid"), TRAIN_MODULE, arm))


def _training_states(owner):
    queue = _load(TRAIN_MODULE)
    states, rejected = {}, []
    pending, missing = False, False
    for arm in train.ARMS:
        path = queue._state_path(arm)
        if not path.is_file():
            missing = True
            continue
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("experiment") != "SW0178_training_queue" or state.get("arm") != arm:
            raise ValueError("SW0178 training queue identity mismatch")
        if state.get("status") == "complete_rejected":
            work = Path(train.OUTPUT_ROOT) / "work" / f"{arm}_seed1"
            rejection_path = work / "credit_rejection.json"
            if (not rejection_path.is_file()
                    or state.get("artifact_sha256") != _sha(rejection_path)
                    or state.get("validation_data_read") is not False):
                raise ValueError(f"SW0178 {arm} scientific rejection receipt is missing or changed")
            rejected.append({"arm": arm, "state_sha256": _sha(path),
                             "rejection_sha256": _sha(rejection_path)})
            continue
        if state.get("status") != "complete":
            if _training_is_pending(owner, state, arm):
                pending = True
                continue
            raise RuntimeError(f"SW0178 {arm} training is not complete: {state.get('status')}")
        artifact = train.OUTPUT_ROOT / f"{arm}_seed1"
        manifest_path = artifact / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"SW0178 {arm} final training manifest missing")
        manifest_sha = _sha(manifest_path)
        _validate_training_receipt(arm, state, manifest_path=manifest_path,
                                   manifest_sha256=manifest_sha)
        states[arm] = {"state": state, "state_sha256": _sha(path),
                       "manifest_sha256": manifest_sha}
    if pending: return False
    if missing: return None
    if rejected: return {**states, "rejected": rejected}
    if set(states) != set(train.ARMS):
        raise ValueError("SW0178 training terminal state set is incomplete")
    return states


def _validate_training_receipt(arm, state, *, manifest_path, manifest_sha256):
    if (state.get("experiment") != "SW0178_training_queue"
            or state.get("arm") != arm or state.get("status") != "complete"
            or state.get("completed_updates") != train.UPDATES
            or state.get("manifest_sha256") != manifest_sha256
            or state.get("validation_data_read") is not False
            or not Path(manifest_path).is_file()):
        raise ValueError(f"SW0178 {arm} training receipt does not bind final artifact")
    return True


def worker_command(arm, gpu):
    if arm not in train.ARMS or type(gpu) is not int or gpu not in range(4):
        raise ValueError("SW0178 prediction worker requires registered arm and physical GPU")
    return [sys.executable, "-u", "-m", MODULE, "--worker", "--arm", arm,
            "--physical-gpu", str(gpu)]


def _worker(arm, gpu):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise RuntimeError("SW0178 prediction worker visibility differs from assigned GPU")
    prepared = evaluate.prepare_prediction_inputs_cpu()
    if evaluate.torch.cuda.is_initialized() or os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise RuntimeError("SW0178 CPU model admission changed CUDA state/visibility")
    return evaluate.predict_arm(arm, device="cuda:0", prepared=prepared)


def _publish_training_rejection(arm, rejection):
    if arm not in train.ARMS or rejection.get("arm") != arm:
        raise ValueError("SW0178 rejected-training arm identity mismatch")
    result = {"status": "complete_training_rejected", "arm": arm,
              "training_rejections": [rejection],
              "ground_truth_used_for_prediction": False}
    _write_state(arm, result["status"], **{
        key: value for key, value in result.items() if key not in {"status", "arm"}})
    return result


def run(arm, *, active_timeout_seconds=8 * 3600):
    import fcntl
    if arm not in train.ARMS: raise ValueError("unregistered SW0178 prediction arm")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("SW0178 prediction coordinator must keep CUDA hidden")
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    RUNTIME.mkdir(parents=True, exist_ok=True)
    state_path = _state_path(arm)
    if state_path.exists() or any(RUNTIME.glob(f"{arm}_attempt*.log")):
        raise FileExistsError("preserving existing SW0178 prediction state and logs")
    lock = (RUNTIME / f"{arm}.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    owner = lease = child = None
    own_state = False
    attempt = 0
    try:
        _write_state(arm, "waiting_for_training", validation_data_read=False)
        own_state = True
        owner = _load(OWNER_MODULE)
        # Waiting for the other arm and a free GPU is intentionally unbounded.
        while not STOP.is_set():
            terminal = _training_states(owner)
            if terminal is not False and terminal is not None:
                break
            _write_state(arm, "waiting_for_training", validation_data_read=False)
            STOP.wait(5)
        else:
            raise RuntimeError("SW0178 prediction queue stopped while waiting for training")
        rejection_by_arm = {row["arm"]: row for row in terminal.get("rejected", [])}
        if arm in rejection_by_arm:
            return _publish_training_rejection(arm, rejection_by_arm[arm])
        prepared = evaluate.prepare_prediction_inputs_cpu()
        cpu_sha = evaluate._load_model(arm, {"training_inputs": prepared["training_inputs"],
            "artifacts": prepared["artifacts"], "manifests": prepared["manifests"]}, "cpu")[1]
        bindings = evaluate._bindings(prepared, arm)
        bindings_sha = evaluate._json_sha(bindings)
        fingerprint = evaluate.implementation_fingerprint()
        _write_state(arm, "waiting_for_exclusive_gpu", training_manifest_sha256=
            bindings["training_manifest_sha256"], training_states={
                name: (terminal[name]["state_sha256"] if name in terminal else rejection_by_arm[name]["state_sha256"])
                for name in train.ARMS},
            bindings_sha256=bindings_sha, model_state_sha256=cpu_sha,
            implementation_fingerprint=fingerprint, validation_data_read=False)
        while not STOP.is_set():
            gpu, gpu_uuid, _used, lease = owner._acquire(STOP)
            stable = owner._stable_idle(gpu, gpu_uuid, STOP,
                lambda elapsed, memory: _write_state(arm, "waiting_for_stable_gpu",
                    physical_gpu=int(gpu), gpu_uuid=gpu_uuid,
                    stable_seconds=float(elapsed), memory_used_mib=int(memory),
                    bindings_sha256=bindings_sha))
            if not stable:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                continue
            latest = _training_states(owner)
            if latest is False or latest is None:
                raise RuntimeError("SW0178 paired training artifacts disappeared before prediction")
            current = evaluate.prepare_prediction_inputs_cpu()
            current_binding = evaluate._bindings(current, arm)
            check_model = evaluate._load_model(arm, {"training_inputs": current["training_inputs"],
                "artifacts": current["artifacts"], "manifests": current["manifests"]}, "cpu")[1]
            if (current_binding != bindings or check_model != cpu_sha
                    or evaluate.implementation_fingerprint() != fingerprint
                    or {name: (latest[name]["state_sha256"] if name in latest else
                                {item["arm"]: item for item in latest["rejected"]}[name]["state_sha256"])
                        for name in train.ARMS}
                       != {name: (terminal[name]["state_sha256"] if name in terminal else
                                  rejection_by_arm[name]["state_sha256"])
                           for name in train.ARMS}):
                raise RuntimeError("SW0178 prediction inputs changed while waiting for GPU")
            live_uuid, memory, owners = owner.gpu_guard._gpu_info(gpu)
            if live_uuid != gpu_uuid or owners or int(memory) > owner.MAX_MEMORY_MIB:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                _write_state(arm, "waiting_for_exclusive_gpu", physical_gpu=int(gpu),
                    gpu_uuid=live_uuid, memory_used_mib=int(memory), owners=list(owners),
                    bindings_sha256=bindings_sha)
                continue
            attempt += 1
            command = worker_command(arm, int(gpu))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", USE_PYTORCH_KERNEL_CACHE="0")
            log_path = RUNTIME / f"{arm}_attempt{attempt:02d}.log"
            yielded = False
            with log_path.open("x", encoding="utf-8", newline="\n") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    pass_fds=(lease.fileno(),))
                deadline = time.monotonic() + active_timeout_seconds
                _write_state(arm, "running_prediction", child_pid=child.pid, child_argv=command,
                    physical_gpu=int(gpu), gpu_uuid=gpu_uuid, attempt=attempt,
                    implementation_fingerprint=fingerprint, bindings_sha256=bindings_sha,
                    validation_data_read=False)
                while child.poll() is None and not STOP.wait(5):
                    if time.monotonic() >= deadline:
                        owner._stop_owned(child)
                        raise TimeoutError("SW0178 owned prediction worker exceeded active runtime")
                    observed_uuid, _used_now, gpu_owners = owner.gpu_guard._gpu_info(gpu)
                    if child.poll() is not None: break
                    if observed_uuid != gpu_uuid or any(int(pid) != child.pid for pid in gpu_owners):
                        owner._stop_owned(child); child.wait(); yielded = True
                        _write_state(arm, "yielding_foreign_owner", child_pid=child.pid,
                            physical_gpu=int(gpu), gpu_uuid=gpu_uuid,
                            observed_gpu_uuid=observed_uuid, owners=list(gpu_owners), attempt=attempt)
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
                        raise RuntimeError("SW0178 prediction child argv identity changed")
                rc = child.wait()
            if yielded:
                attempt_record = {"attempt": attempt, "status": "yielded_foreign_owner",
                    "physical_gpu": int(gpu), "gpu_uuid": gpu_uuid,
                    "returncode": rc, "log": log_path.name}
                old = json.loads(state_path.read_text(encoding="utf-8"))
                attempts = list(old.get("attempts", [])) + [attempt_record]
                _write_state(arm, "waiting_for_exclusive_gpu", attempts=attempts,
                    bindings_sha256=bindings_sha, implementation_fingerprint=fingerprint)
                child = None
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                continue
            if STOP.is_set() and rc == 0:
                raise RuntimeError("SW0178 prediction stopped during worker completion")
            if rc != 0:
                raise RuntimeError(f"SW0178 prediction worker exited {rc}; inspect {log_path}")
            published = evaluate.validate_prediction(arm, prepared=current)
            state_out = {"experiment": "SW0178_prediction_queue_v1", "status": "complete",
                "arm": arm, "supervisor_pid": os.getpid(), "child_pid": child.pid,
                "physical_gpu": int(gpu), "gpu_uuid": gpu_uuid, "attempt": attempt,
                "prediction_sha256": published["prediction_sha256"],
                "protocol_sha256": published["protocol_sha256"],
                "output": str(published["directory"]), "bindings_sha256": bindings_sha,
                "implementation_fingerprint": fingerprint,
                "ground_truth_used_for_prediction": False, "completed_unix": time.time()}
            _write_state(arm, "complete", **{k:v for k,v in state_out.items()
                if k not in {"experiment", "status", "arm", "supervisor_pid"}})
            return state_out
        raise RuntimeError("SW0178 prediction supervisor stopped")
    except BaseException as exc:
        if child is not None and child.poll() is None and owner is not None:
            owner._stop_owned(child)
        if own_state:
            _write_state(arm, "failed", error=repr(exc))
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
    parser.add_argument("--arm", choices=("mean_rho_ce", "product_q_ce"), required=True)
    parser.add_argument("--physical-gpu", type=int)
    args = parser.parse_args(argv)
    if args.worker:
        if args.physical_gpu is None: parser.error("worker needs --physical-gpu")
        _worker(args.arm, args.physical_gpu)
    else:
        run(args.arm)


if __name__ == "__main__":
    main()
