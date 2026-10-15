"""New-namespace SW0168 prediction supervisor adopting CPU-validated completed heads."""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import subprocess
import signal
import threading

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

from collaborative_test.SW_0168_end_to_end_connectivity import recover_completed_training as recovery
from collaborative_test.SW_0168_end_to_end_connectivity import train
from collaborative_test.SW_0168_end_to_end_connectivity import prediction_queue as base

HERE = Path(__file__).resolve().parent
RUNTIME = HERE / "results_archive" / "prediction_queue_v3_20261015"
EXPERIMENT = "SW0168_prediction_queue_v3"
MODULE = "collaborative_test.SW_0168_end_to_end_connectivity.prediction_queue_v3"
_stop = threading.Event()


def _sha(path): return recovery._sha(path)


def _write_state(arm, status, **fields):
    recovery._atomic_replace_json(RUNTIME / f"{arm}_state.json", {
        "experiment": EXPERIMENT, "arm": arm, "status": status,
        "supervisor_pid": os.getpid(), "updated": time.time(),
        "recovery_receipt_sha256": _sha(recovery.RECEIPT), **fields})


def _training_ready(arm, owner):
    receipt = recovery.validate_receipt(deep=False)
    if arm not in train.ARMS: raise ValueError("unregistered SW0168 arm")
    manifest_path = train.TRAIN_ROOT / f"{arm}_seed1" / "manifest.json"
    record = receipt["arms"][arm]
    if _sha(manifest_path) != record["manifest_sha256"]:
        raise ValueError("recovered SW0168 model manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "training_complete" or manifest.get("completed_updates") != 64:
        raise ValueError("recovered SW0168 model is not a complete 64-update head")
    return manifest


def _owned_proc_record(owner, child, command):
    record=owner._proc_record(child.pid)
    missing=(not record or record.get("state")=="Z" or not record.get("argv"))
    if child.poll() is None and missing:
        try: child.wait(timeout=1.0)
        except subprocess.TimeoutExpired: pass
        record=owner._proc_record(child.pid)
    return record


def _foreign_owner_changed(expected_uuid, actual_uuid, owners, child_pid):
    return actual_uuid != expected_uuid or any(int(pid) != int(child_pid) for pid in owners)


def _worker(arm, gpu):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise RuntimeError("SW0168 v2 worker visibility differs from leased GPU")
    prepared = base.evaluate.prepare_prediction_inputs_cpu()
    return base.evaluate.predict_arm(arm, device="cuda:0", prepared=prepared)


def run(arm, *, timeout_seconds=8 * 3600):
    import fcntl
    if arm not in train.ARMS: raise ValueError("unregistered SW0168 recovery prediction arm")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("SW0168 v2 prediction supervisor must remain CUDA-hidden")
    queue_sha=_sha(__file__)
    receipt = recovery.validate_receipt(deep=True)
    signal.signal(signal.SIGTERM, lambda *_: _stop.set())
    signal.signal(signal.SIGINT, lambda *_: _stop.set())
    RUNTIME.mkdir(parents=True, exist_ok=True)
    state_path=RUNTIME/f"{arm}_state.json"; lock_path=RUNTIME/f"{arm}.lock"
    if state_path.exists() or any(RUNTIME.glob(f"{arm}_attempt*.log")):
        raise FileExistsError("preserving existing SW0168 v2 prediction state/logs")
    lock=lock_path.open("a+"); fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    owner=None; lease=None; child=None; attempt=0
    recovery_sha=_sha(recovery.RECEIPT)
    try:
        _write_state(arm,"waiting_for_training")
        owner=base._owner_preserve_visibility()
        deadline=time.monotonic()+timeout_seconds
        prepared=None
        while not _stop.is_set():
            if time.monotonic()>deadline: raise TimeoutError("SW0168 v2 training wait timed out")
            readiness={name:_training_ready(name,owner) for name in train.ARMS}
            if all(readiness.values()):
                prepared=base.evaluate.prepare_prediction_inputs_cpu(); break
            time.sleep(5)
        if prepared is None: raise RuntimeError("SW0168 v2 prediction stopped while waiting for training")
        bindings_before=base.evaluate._bindings(prepared,arm)
        fp_before=base.evaluate.implementation_fingerprint()
        runtime_fp=base.evaluate._json_sha({"evaluator":fp_before,"recovery_receipt_sha256":recovery_sha,
                                           "recovery_module_sha256":queue_sha})
        while not _stop.is_set():
            if time.monotonic()>deadline: raise TimeoutError("SW0168 v2 prediction exceeded deadline")
            gpu,gpu_uuid,memory,lease=owner._acquire(_stop)
            _write_state(arm,"waiting_for_stable_gpu",physical_gpu=gpu,gpu_uuid=gpu_uuid,
                memory_used_mib=int(memory),recovery_runtime_fingerprint=runtime_fp)
            if not owner._stable_idle(gpu,gpu_uuid,_stop,
                    lambda elapsed,used:_write_state(arm,"waiting_for_stable_gpu",
                        physical_gpu=gpu,gpu_uuid=gpu_uuid,stable_seconds=float(elapsed),
                        memory_used_mib=int(used),recovery_runtime_fingerprint=runtime_fp)):
                fcntl.flock(lease.fileno(),fcntl.LOCK_UN); lease.close(); lease=None; continue
            readiness={name:_training_ready(name,owner) for name in train.ARMS}
            if not all(readiness.values()): raise RuntimeError("recovered training artifact changed under GPU lease")
            prepared_now=base.evaluate.prepare_prediction_inputs_cpu()
            if (base.evaluate._bindings(prepared_now,arm)!=bindings_before
                    or base.evaluate.implementation_fingerprint()!=fp_before or _sha(__file__)!=queue_sha):
                raise RuntimeError("SW0168 v2 prediction bindings/code changed while waiting")
            actual_uuid,used,owners=owner.gpu_guard._gpu_info(gpu)
            if actual_uuid!=gpu_uuid or owners or int(used)>owner.MAX_MEMORY_MIB:
                fcntl.flock(lease.fileno(),fcntl.LOCK_UN); lease.close(); lease=None
                _write_state(arm,"waiting_for_exclusive_gpu",gpu_uuid=actual_uuid,
                    memory_used_mib=int(used),owners=list(owners)); continue
            attempt+=1; log_path=RUNTIME/f"{arm}_attempt{attempt:02d}.log"
            command=base._worker_command(arm,gpu)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="1",
                     MKL_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
            retry=False
            with log_path.open("x",encoding="utf-8",newline="\n") as log:
                child=subprocess.Popen(command,cwd=base.ROOT,env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lease.fileno(),))
                _write_state(arm,"running_prediction",child_pid=child.pid,child_pgid=child.pid,
                    child_argv=command,physical_gpu=gpu,gpu_uuid=gpu_uuid,attempt=attempt,
                    recovery_runtime_fingerprint=runtime_fp,
                    training_manifest_sha256=receipt["arms"][arm]["manifest_sha256"])
                while child.poll() is None and not _stop.is_set():
                    if time.monotonic()>deadline:
                        owner._stop_owned(child); raise TimeoutError("SW0168 v2 prediction worker timed out")
                    now_uuid,used,gpu_owners=owner.gpu_guard._gpu_info(gpu)
                    if child.poll() is None and _foreign_owner_changed(gpu_uuid,now_uuid,gpu_owners,child.pid):
                        owner._stop_owned(child); retry=True
                        _write_state(arm,"yielding_foreign_gpu",child_pid=child.pid,physical_gpu=gpu,
                            gpu_uuid=gpu_uuid,observed_gpu_uuid=now_uuid,owners=list(gpu_owners),attempt=attempt)
                        break
                    if child.poll() is None:
                        proc=_owned_proc_record(owner,child,command)
                        if child.poll() is None and proc and proc.get("argv") and (
                                command[1] not in proc["argv"] or command[2] not in proc["argv"]):
                            owner._stop_owned(child); raise RuntimeError("SW0168 v2 owned child identity changed")
                    _write_state(arm,"running_prediction",child_pid=child.pid,physical_gpu=gpu,
                        gpu_uuid=gpu_uuid,memory_used_mib=int(used),attempt=attempt,
                        recovery_runtime_fingerprint=runtime_fp)
                    time.sleep(3)
                rc=child.wait()
            if retry:
                fcntl.flock(lease.fileno(),fcntl.LOCK_UN); lease.close(); lease=None; child=None; continue
            if rc!=0: raise RuntimeError(f"SW0168 v2 prediction worker exited {rc}; inspect {log_path}")
            prepared_after=base.evaluate.prepare_prediction_inputs_cpu()
            result=base.evaluate.validate_prediction(arm,prepared=prepared_after)
            _write_state(arm,"complete",child_pid=child.pid,output=str(result["directory"]),
                prediction_sha256=result["prediction_sha256"],protocol_sha256=result["protocol_sha256"],
                training_manifest_sha256=receipt["arms"][arm]["manifest_sha256"],attempt=attempt,
                recovery_runtime_fingerprint=runtime_fp)
            return result
        raise RuntimeError("SW0168 v2 prediction stopped before owned prediction")
    except BaseException as exc:
        if child is not None and child.poll() is None: owner._stop_owned(child)
        _write_state(arm,"failed",error=repr(exc),finished=time.time(),recovery_runtime_fingerprint=locals().get("runtime_fp"))
        raise
    finally:
        if child is not None and child.poll() is None: owner._stop_owned(child)
        if lease is not None:
            fcntl.flock(lease.fileno(),fcntl.LOCK_UN); lease.close()
        fcntl.flock(lock.fileno(),fcntl.LOCK_UN); lock.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=train.ARMS, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--physical-gpu", type=int)
    args = parser.parse_args()
    if args.worker: raise SystemExit("v2 supervisors dispatch the frozen prediction worker")
    run(args.arm)

if __name__ == "__main__": main()
