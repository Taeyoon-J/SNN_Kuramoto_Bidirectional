"""Per-arm exclusive-GPU prediction coordinator for SW0175 fixed320 rows."""
from __future__ import annotations

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
_WORKER_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES") if _WORKER else None
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import hashlib

from collaborative_test.SW_0175_critical_component_binding import predict_evaluate as evaluate, train_v2 as train

# Several imported legacy helpers hide CUDA for CPU supervisors. A GPU child
# must retain the physical device assigned by its leased parent.
if _WORKER:
    if _WORKER_CUDA_VISIBLE_DEVICES is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = _WORKER_CUDA_VISIBLE_DEVICES

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNTIME = HERE / "results_archive" / "prediction_queue_v2_20261015"
OWNER_MODULE = "collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4"
_stop = threading.Event()
ACTIVE_TRAIN_STATES = {"validating_admission", "waiting_for_stable_gpu", "waiting_for_exclusive_gpu",
                       "running_training", "yielding_foreign_gpu"}


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def _atomic_json(path, value, *, replace=False):
    path = Path(path); tmp = path.with_name(path.name + f".tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with tmp.open("x", encoding="utf-8", newline="\n") as f:
            json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        if replace:
            os.replace(tmp, path)
        else:
            if path.exists(): raise FileExistsError(path)
            os.link(tmp, path)
    finally:
        if tmp.exists(): tmp.unlink()


def _state_path(arm): return RUNTIME / f"{arm}_state.json"


def _write_state(arm, status, **fields):
    _atomic_json(_state_path(arm), {"experiment": "SW0175_prediction_queue", "arm": arm,
        "status": status, "supervisor_pid": os.getpid(), "updated": time.time(), **fields}, replace=True)


def _module_live(owner, pid, module, arm=None):
    if type(pid) is not int or pid <= 0: return False
    record = owner._proc_record(pid)
    if not record or record.get("state") == "Z": return False
    argv = record.get("argv", [])
    if "-m" not in argv: return False
    index = argv.index("-m")
    if index + 1 >= len(argv) or argv[index + 1] != module: return False
    if arm is not None:
        if argv.count("--arm") != 1: return False
        arm_index = argv.index("--arm")
        return arm_index + 1 < len(argv) and argv[arm_index + 1] == arm
    return True


def _owner_preserve_visibility():
    import importlib
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    owner = importlib.import_module(OWNER_MODULE)
    if visible is None: os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else: os.environ["CUDA_VISIBLE_DEVICES"] = visible
    return owner


def _stable_idle(owner, arm, gpu, gpu_uuid):
    return owner._stable_idle(gpu, gpu_uuid, _stop,
        lambda elapsed, used: _write_state(arm, "waiting_for_stable_gpu",
            physical_gpu=gpu, gpu_uuid=gpu_uuid, stable_seconds=float(elapsed),
            memory_used_mib=int(used)))


def _foreign_during_owned_child(child_pid, expected_uuid, observed_uuid, owners):
    return observed_uuid != expected_uuid or any(int(pid) != int(child_pid) for pid in owners)


def _training_ready(arm, owner):
    path = train.OUTPUT_ROOT.parent / "training_queue_v2_20261015" / f"{arm}_state.json"
    if not path.is_file():
        raise FileNotFoundError(f"SW0175 registered training queue state missing for {arm}")
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("experiment") != "SW0175_training_queue_v2" or state.get("arm") != arm:
        raise ValueError("SW0175 training dependency state identity mismatch")
    if state.get("status") == "complete":
        manifest = train.validate_training_artifact(arm, train.OUTPUT_ROOT / f"{arm}_seed1")
        expected = train.OUTPUT_ROOT / f"{arm}_seed1" / "manifest.json"
        if state.get("manifest_sha256") != _sha(expected):
            raise ValueError("SW0175 training terminal state does not bind current manifest")
        return manifest
    if state.get("status") in ACTIVE_TRAIN_STATES and _module_live(owner, state.get("supervisor_pid"),
            "collaborative_test.SW_0175_critical_component_binding.training_queue_v2", arm):
        return False
    raise RuntimeError(f"SW0175 training dependency for {arm} is terminal or orphaned: {state.get('status')}")


def _all_training_ready(owner):
    ready = {arm: _training_ready(arm, owner) for arm in train.ARMS}
    return False if any(value is False for value in ready.values()) else ready


def _worker_command(arm, gpu):
    if arm not in train.ARMS or type(gpu) is not int or gpu not in range(4):
        raise ValueError("worker command requires registered arm and leased GPU")
    return [sys.executable, "-m", "collaborative_test.SW_0175_critical_component_binding.prediction_queue",
            "--worker", "--arm", arm, "--physical-gpu", str(gpu)]


def _worker(arm, physical_gpu):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("SW0175 prediction worker visibility differs from leased GPU")
    # CPU provenance checks happen before the first CUDA operation.
    prepared = evaluate.prepare_prediction_inputs_cpu()
    return evaluate.predict_arm(arm, device="cuda:0", prepared=prepared)


def run(arm, *, timeout_seconds=8 * 3600):
    if arm not in train.ARMS: raise ValueError("unregistered SW0175 prediction arm")
    import fcntl
    signal.signal(signal.SIGTERM, lambda *_: _stop.set())
    signal.signal(signal.SIGINT, lambda *_: _stop.set())
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("prediction supervisor must keep CUDA hidden")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    state_path = _state_path(arm); lock_path = RUNTIME / f"{arm}.lock"
    if state_path.exists() or any(RUNTIME.glob(f"{arm}_attempt*.log")):
        raise FileExistsError("preserving existing SW0175 prediction queue state/logs")
    lock = lock_path.open("a+"); fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    owner = None; lease = None; child = None; attempt = 0
    try:
        _write_state(arm, "waiting_for_training")
        owner = _owner_preserve_visibility()
        deadline = time.monotonic() + timeout_seconds
        prepared = None
        while not _stop.is_set():
            if time.monotonic() > deadline: raise TimeoutError("SW0175 prediction wait timed out")
            training = _all_training_ready(owner)
            if training is not False:
                prepared = evaluate.prepare_prediction_inputs_cpu()
                break
            time.sleep(5)
        if prepared is None: raise RuntimeError("SW0175 prediction queue stopped while waiting for training")
        bindings_before = evaluate._bindings(prepared, arm)
        fp_before = evaluate.implementation_fingerprint()
        while not _stop.is_set():
            if time.monotonic() > deadline: raise TimeoutError("SW0175 prediction queue exceeded deadline")
            gpu, gpu_uuid, memory, lease = owner._acquire(_stop)
            _write_state(arm, "waiting_for_stable_gpu", physical_gpu=gpu, gpu_uuid=gpu_uuid,
                         memory_used_mib=int(memory), training_manifest_sha256=bindings_before["training_manifest_sha256"])
            if not _stable_idle(owner, arm, gpu, gpu_uuid):
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None; continue
            current_training = _training_ready(arm, owner)
            if current_training is False: raise RuntimeError("SW0175 training artifact disappeared under GPU lease")
            prepared_now = evaluate.prepare_prediction_inputs_cpu()
            if (evaluate._bindings(prepared_now, arm) != bindings_before
                    or evaluate.implementation_fingerprint() != fp_before):
                raise RuntimeError("SW0175 prediction bindings changed while waiting for GPU")
            actual_uuid, used, owners = owner.gpu_guard._gpu_info(gpu)
            if actual_uuid != gpu_uuid or owners or int(used) > owner.MAX_MEMORY_MIB:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                _write_state(arm, "waiting_for_exclusive_gpu", gpu_uuid=actual_uuid,
                             memory_used_mib=int(used), owners=list(owners)); continue
            attempt += 1
            log_path = RUNTIME / f"{arm}_attempt{attempt:02d}.log"
            command = _worker_command(arm, gpu)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
                       USE_PYTORCH_KERNEL_CACHE="0")
            retry = False
            with log_path.open("x", encoding="utf-8", newline="\n") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    pass_fds=(lease.fileno(),))
                _write_state(arm, "running_prediction", child_pid=child.pid, child_pgid=child.pid,
                    child_argv=command, physical_gpu=gpu, gpu_uuid=gpu_uuid, attempt=attempt,
                    training_manifest_sha256=bindings_before["training_manifest_sha256"])
                end = min(deadline, time.monotonic() + timeout_seconds)
                while child.poll() is None and not _stop.is_set():
                    if time.monotonic() > end:
                        owner._stop_owned(child); raise TimeoutError("SW0175 prediction worker timed out")
                    now_uuid, used, gpu_owners = owner.gpu_guard._gpu_info(gpu)
                    if child.poll() is None and _foreign_during_owned_child(
                            child.pid, gpu_uuid, now_uuid, gpu_owners):
                        owner._stop_owned(child); retry = True
                        _write_state(arm, "yielding_foreign_gpu", child_pid=child.pid, physical_gpu=gpu,
                            gpu_uuid=gpu_uuid, observed_gpu_uuid=now_uuid, owners=list(gpu_owners), attempt=attempt)
                        break
                    proc = owner._proc_record(child.pid) if child.poll() is None else None
                    if child.poll() is None and proc and proc.get("state") != "Z" and proc.get("argv") != command:
                        owner._stop_owned(child)
                        raise RuntimeError("SW0175 prediction child identity changed")
                    if child.poll() is None and (not proc or proc.get("state") == "Z"):
                        try:
                            child.wait(timeout=1.0)
                        except subprocess.TimeoutExpired:
                            pass
                        if child.poll() is not None:
                            break
                        continue
                    _write_state(arm, "running_prediction", child_pid=child.pid, physical_gpu=gpu,
                                 gpu_uuid=gpu_uuid, memory_used_mib=int(used), attempt=attempt)
                    time.sleep(3)
                rc = child.wait()
            if retry:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None; child = None
                continue
            if rc != 0: raise RuntimeError(f"SW0175 prediction worker exited {rc}; inspect {log_path}")
            # Revalidate all 320 committed rows and final protocol on CPU under the lease.
            prepared_after = evaluate.prepare_prediction_inputs_cpu()
            result = evaluate.validate_prediction(arm, prepared=prepared_after)
            _write_state(arm, "complete", child_pid=child.pid, output=str(result["directory"]),
                prediction_sha256=result["prediction_sha256"], protocol_sha256=result["protocol_sha256"],
                training_manifest_sha256=bindings_before["training_manifest_sha256"], attempt=attempt)
            return result
        raise RuntimeError("SW0175 prediction queue stopped before an owned prediction")
    except BaseException as exc:
        if child is not None and child.poll() is None: owner._stop_owned(child)
        _write_state(arm, "failed", error=repr(exc), finished=time.time())
        raise
    finally:
        if child is not None and child.poll() is None: owner._stop_owned(child)
        if lease is not None:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close()
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN); lock.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=train.ARMS, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--physical-gpu", type=int)
    args = parser.parse_args()
    if args.worker:
        if args.physical_gpu not in range(4): raise SystemExit("worker needs leased GPU")
        _worker(args.arm, args.physical_gpu)
    else: run(args.arm)


if __name__ == "__main__": main()
