"""Run the non-gating TRAIN8 audit only after SW0175 fixed320 scoring finishes."""
from __future__ import annotations

import argparse
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODULE = "collaborative_test.SW_0175_critical_component_binding.train8_queue"
RUNTIME = HERE / "results_archive" / "train8_diagnostic_queue_20261015"
REPORT = HERE / "results_archive" / "train8_audit_20261015" / "report.json"
SCORE_MODULE = "collaborative_test.SW_0175_critical_component_binding.score_queue_v3"
STOP = threading.Event()
_WORKER = "--worker" in sys.argv
_WORKER_VISIBLE = os.environ.get("CUDA_VISIBLE_DEVICES") if _WORKER else None
if not _WORKER:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict, *, replace=False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with tmp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(tmp, path)
        else:
            if path.exists():
                raise FileExistsError(path)
            os.link(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _imports_preserving_cuda():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        audit = importlib.import_module(
            "collaborative_test.SW_0175_critical_component_binding.train8_audit")
        evaluator = importlib.import_module(
            "collaborative_test.SW_0175_critical_component_binding.predict_evaluate_v2")
        score_queue = importlib.import_module(SCORE_MODULE)
        owner = importlib.import_module(
            "collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4")
        return audit, evaluator, score_queue, owner
    finally:
        if visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible


def _supervisor_live(owner, pid: int, module: str) -> bool:
    if type(pid) is not int or pid <= 0:
        return False
    record = owner._proc_record(pid)
    if not record or record.get("state") == "Z":
        return False
    argv = record.get("argv", [])
    if not isinstance(argv, list) or argv.count("-m") != 1:
        return False
    marker = argv.index("-m")
    return marker + 1 < len(argv) and argv[marker + 1] == module


def _score_complete(owner, score_queue, evaluator):
    state_path = Path(score_queue.RUNTIME) / "state.json"
    if not state_path.is_file():
        return False
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("experiment") != "SW0175_score_queue_v3":
        raise ValueError("SW0175 TRAIN8 audit found a foreign score queue state")
    status = state.get("status")
    if status == "complete":
        score_path = Path(state.get("score_path", ""))
        if (score_path.resolve() != Path(evaluator.SCORE_PATH).resolve()
                or not score_path.is_file()
                or _sha(score_path) != state.get("score_sha256")):
            raise ValueError("SW0175 score completion state does not bind the score artifact")
        prepared = evaluator.prepare_prediction_inputs_cpu()
        verified = evaluator.validate_score(path=score_path, prepared=prepared)
        if not isinstance(verified, dict):
            raise ValueError("SW0175 public score validator returned no report")
        return {"state": state, "score_sha256": _sha(score_path),
                "score_path": str(score_path.resolve())}
    active = {"waiting_for_both_predictions", "scoring_gt_last"}
    if status in active and _supervisor_live(owner, state.get("supervisor_pid"), SCORE_MODULE):
        return False
    raise RuntimeError(f"SW0175 score queue is failed, orphaned, or unexpected: {status!r}")


def worker(physical_gpu: int, output: Path, score_sha256: str) -> dict:
    if type(physical_gpu) is not int or physical_gpu not in range(4):
        raise ValueError("TRAIN8 worker requires a physical GPU index in [0,3]")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("TRAIN8 worker visibility differs from leased physical GPU")
    audit, _evaluator, _score, _owner = _imports_preserving_cuda()
    prepared = audit.prepare_inputs_cpu()
    if audit.torch.cuda.is_initialized() or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("TRAIN8 CPU admission initialized CUDA or changed worker visibility")
    if audit.torch.cuda.device_count() != 1:
        raise RuntimeError("TRAIN8 worker must see exactly its leased CUDA device")
    report = audit.run_train8_audit(device="cuda:0", prepared=prepared)
    report.update({"implementation_fingerprint": prepared["fingerprint"],
        "source_proof": prepared["admitted"]["proof"],
        "optimizer_updates": 0, "validation_data_read": False,
        "cuda_visible_devices": str(physical_gpu), "score_sha256": score_sha256})
    if report.get("status") != "complete_non_gating_diagnostic":
        raise ValueError("TRAIN8 diagnostic did not complete non-gating analysis")
    _atomic_json(output, report)
    return report


def worker_command(gpu: int, output=REPORT, score_sha256="") -> list[str]:
    if type(gpu) is not int or gpu not in range(4):
        raise ValueError("worker command needs a leased physical GPU")
    if len(score_sha256) != 64 or any(c not in "0123456789abcdef" for c in score_sha256):
        raise ValueError("worker command must bind the completed score SHA256")
    return [sys.executable, "-u", "-m", MODULE, "--worker", "--physical-gpu",
            str(gpu), "--output", str(Path(output)), "--score-sha256", score_sha256]


def supervise(*, timeout_seconds=8 * 3600) -> dict:
    import fcntl
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: STOP.set())
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    RUNTIME.mkdir(parents=True, exist_ok=True)
    state_path, lock_path, log_path = (RUNTIME / "state.json", RUNTIME / "queue.lock",
                                       RUNTIME / "worker.log")
    lock = lock_path.open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = lease = owner = None
    owns_state = False

    def record(status, **fields):
        value = {"experiment": "SW0175_train8_diagnostic_queue", "status": status,
                 "supervisor_pid": os.getpid(), "updated": time.time(), **fields}
        _atomic_json(state_path, value, replace=state_path.exists())

    try:
        if state_path.exists() or REPORT.exists() or log_path.exists():
            raise FileExistsError("preserving existing TRAIN8 diagnostic queue artifacts")
        audit, evaluator, score_queue, owner = _imports_preserving_cuda()
        record("waiting_for_score_completion")
        owns_state = True
        deadline = time.monotonic() + timeout_seconds
        score = False
        while not STOP.is_set():
            if time.monotonic() > deadline:
                raise TimeoutError("TRAIN8 audit timed out waiting for completed score")
            score = _score_complete(owner, score_queue, evaluator)
            if score is not False:
                break
            time.sleep(5)
        if STOP.is_set():
            raise RuntimeError("TRAIN8 audit stopped before score completion")

        prepared = audit.prepare_inputs_cpu()
        proof = prepared["admitted"]["proof"]
        fingerprint = prepared["fingerprint"]
        del prepared
        fp_before = audit.implementation_fingerprint()
        if fingerprint != fp_before:
            raise RuntimeError("TRAIN8 audit dependencies changed during CPU admission")
        record("waiting_for_exclusive_gpu", score_sha256=score["score_sha256"],
               score_path=score["score_path"], source_proof=proof,
               implementation_fingerprint=fingerprint)

        while not STOP.is_set():
            if time.monotonic() > deadline:
                raise TimeoutError("TRAIN8 audit exceeded GPU admission deadline")
            gpu, gpu_uuid, memory, lease = owner._acquire(STOP)
            record("waiting_for_stable_gpu", physical_gpu=gpu, gpu_uuid=gpu_uuid,
                   memory_used_mib=int(memory), score_sha256=score["score_sha256"])
            stable = owner._stable_idle(gpu, gpu_uuid, STOP,
                lambda elapsed, used: record("waiting_for_stable_gpu", physical_gpu=gpu,
                    gpu_uuid=gpu_uuid, stable_seconds=float(elapsed), memory_used_mib=int(used),
                    score_sha256=score["score_sha256"]))
            if not stable:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                lease.close(); lease = None
                continue
            actual_uuid, used, owners = owner.gpu_guard._gpu_info(gpu)
            if actual_uuid != gpu_uuid or owners or int(used) > owner.MAX_MEMORY_MIB:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                lease.close(); lease = None
                record("waiting_for_exclusive_gpu", physical_gpu=gpu,
                       gpu_uuid=actual_uuid, memory_used_mib=int(used), owners=list(owners),
                       score_sha256=score["score_sha256"])
                continue
            if audit.implementation_fingerprint() != fp_before:
                raise RuntimeError("TRAIN8 audit fingerprint changed before worker launch")
            break
        else:
            raise RuntimeError("TRAIN8 audit stopped before exclusive GPU admission")

        current_state = json.loads((Path(score_queue.RUNTIME) / "state.json").read_text(encoding="utf-8"))
        if (current_state.get("status") != "complete"
                or current_state.get("score_sha256") != score["score_sha256"]
                or _sha(Path(current_state.get("score_path", ""))) != score["score_sha256"]):
            raise RuntimeError("completed SW0175 score artifact changed before TRAIN8 launch")
        command = worker_command(gpu, REPORT, score["score_sha256"])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
                   USE_PYTORCH_KERNEL_CACHE="0")
        with log_path.open("x", encoding="utf-8", newline="\n") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                pass_fds=(lease.fileno(),))
            record("running", child_pid=child.pid, worker_argv=command,
                   physical_gpu=gpu, gpu_uuid=gpu_uuid, score_sha256=score["score_sha256"],
                   source_proof=proof, implementation_fingerprint=fingerprint)
            while child.poll() is None and not STOP.wait(5):
                if time.monotonic() > deadline:
                    raise TimeoutError("TRAIN8 GPU diagnostic exceeded its deadline")
                live_uuid, _used, owners = owner.gpu_guard._gpu_info(gpu)
                if child.poll() is not None:
                    break
                if live_uuid != gpu_uuid or any(int(pid) != child.pid for pid in owners):
                    if child.poll() is not None:
                        break
                    raise RuntimeError("foreign GPU owner appeared during TRAIN8 audit")
            if STOP.is_set() and child.poll() is None:
                raise RuntimeError("TRAIN8 audit supervisor stopped while worker was active")
            if child.returncode != 0 or not REPORT.is_file():
                raise RuntimeError(f"TRAIN8 audit worker exited {child.returncode}; inspect log")
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        if (report.get("status") != "complete_non_gating_diagnostic"
                or report.get("implementation_fingerprint") != fingerprint
                or report.get("source_proof") != proof
                or report.get("optimizer_updates") != 0
                or report.get("validation_data_read") is not False
                or report.get("cuda_visible_devices") != str(gpu)
                or report.get("score_sha256") != score["score_sha256"]):
            raise ValueError("TRAIN8 audit report does not match the completed score/source lease")
        record("complete_non_gating_diagnostic", child_pid=child.pid,
               returncode=child.returncode, physical_gpu=gpu, gpu_uuid=gpu_uuid,
               score_sha256=score["score_sha256"],
               report_sha256=_sha(REPORT), implementation_fingerprint=fingerprint)
        return report
    except BaseException as exc:
        if child is not None and child.poll() is None and owner is not None:
            owner._stop_owned(child)
        if owns_state:
            record("failed", error=repr(exc))
        raise
    finally:
        if child is not None and child.poll() is None and owner is not None:
            owner._stop_owned(child)
        if lease is not None:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
            lease.close()
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--score-sha256")
    args = parser.parse_args()
    if args.worker:
        if args.physical_gpu is None or args.output is None or args.score_sha256 is None:
            parser.error("worker requires --physical-gpu, --output, and --score-sha256")
        if args.output.resolve() != REPORT.resolve():
            parser.error("worker output must be the registered immutable report path")
        worker(args.physical_gpu, args.output, args.score_sha256)
    else:
        supervise()


if __name__ == "__main__":
    main()
