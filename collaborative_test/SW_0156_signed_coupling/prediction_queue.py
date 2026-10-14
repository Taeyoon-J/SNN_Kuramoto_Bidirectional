"""Owner-aware SW0156 fixed320 prediction plus GT-last scoring queue."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
try:
    import fcntl
except ImportError:  # Windows may inspect plans but cannot reserve Linux GPUs.
    fcntl = None
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"),
                str(ROOT / "snn_kuramoto_bidirectional")]

from collaborative_test.SW_0143_native32_selective_geodesic import run as sw143
from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4 as owner
from collaborative_test.SW_0156_signed_coupling import evaluate, predict, source_screen_queue

EXPERIMENT = "SW0156_native32_signed_coupling"
RUNTIME = HERE / "results_archive" / "prediction_queue_20261013"
STATE, LOCK = RUNTIME / "state.json", RUNTIME / "queue.lock"
LOGS = RUNTIME / "logs"
SCORE = HERE / "results_archive" / "evaluation_seed1.json"
GPU_ARMS = ("signed_candidate", "matched_control")
POLL_SECONDS = 2.0


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint():
    paths = (HERE / "protocol.json", HERE / "signed_coupling.py", HERE / "screen.py",
             HERE / "source_screen_queue.py", HERE / "predict.py", HERE / "evaluate.py",
             HERE / "prediction_queue.py", Path(owner.__file__), owner.OWNER_GUARD)
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("SW0156 prediction queue dependency missing")
    result = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}
    result["owner_guard_sha256"] = owner.OWNER_GUARD_SHA256
    return result


def task_plan():
    return [
        {"task_id": "source_T", "kind": "reuse_validated_reference", "gpu": False,
         "output": str(predict.prediction_dir("source_T"))},
        *[{"task_id": arm, "kind": "predict", "gpu": True,
           "depends_on": ["source_T"], "output": str(predict.prediction_dir(arm))}
          for arm in GPU_ARMS],
        {"task_id": "score_seed1", "kind": "gt_last_score", "gpu": False,
         "depends_on": ["source_T", *GPU_ARMS], "output": str(SCORE)},
    ]


def _atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def _target_state(target):
    """Return ``missing`` or ``ledger``; reject ambiguous/complete targets."""
    target = Path(target)
    if not target.exists():
        return "missing"
    if ((target / "protocol.json").exists()
            or not (target / "prediction_header.json").is_file()
            or not (target / "prediction_ledger.jsonl").is_file()):
        raise FileExistsError("preserve unrecognized/complete prediction target")
    return "ledger"


def _cleanup_child_before_lease_release(proc, lease):
    """Stop only the owned child before releasing the physical GPU lease."""
    try:
        if proc is not None and proc.poll() is None:
            owner._stop_owned(proc)
    finally:
        if lease is not None:
            try:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
            finally:
                lease.close()


def _upstream():
    if not source_screen_queue.validate_report(source_screen_queue.OUTPUT):
        if not source_screen_queue.OUTPUT.exists():
            return None
        raise ValueError("preserve invalid/stale SW0156 source-screen report")
    screen_bytes = source_screen_queue.OUTPUT.read_bytes()
    source = json.loads(screen_bytes.decode("utf-8"))
    screen_sha = hashlib.sha256(screen_bytes).hexdigest()
    t = sw143.validate_arm_prediction(sw143._arm_dir("T", 1), seed=1)
    if (t["protocol"].get("source") != source.get("source_lineage")
            or t["prediction_sha256"] != source.get("sw0143_T_prediction_sha256")
            or t["protocol_sha256"] != source.get("sw0143_T_protocol_sha256")):
        raise ValueError("source screen and current SW0143-T reference disagree")
    return {"screen": source, "screen_sha256": screen_sha,
            "source_prediction_sha256": t["prediction_sha256"],
            "source_protocol_sha256": t["protocol_sha256"]}


def _valid_completed(arm):
    return predict.validate_prediction(predict.prediction_dir(arm), arm=arm)


class PredictionQueue:
    def __init__(self):
        if fcntl is None:
            raise RuntimeError("SW0156 GPU queue requires Linux flock")
        RUNTIME.mkdir(parents=True, exist_ok=True)
        self.lock_stream = LOCK.open("a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock_stream.close()
            raise RuntimeError("another SW0156 prediction supervisor is active") from exc
        self.stop_event = threading.Event()
        self.state_lock = threading.RLock()
        self.plan = task_plan()
        self.fingerprint = implementation_fingerprint()
        self.completed = False
        try:
            if SCORE.exists():
                raise FileExistsError("preserve existing SW0156 score; manual validation required")
            if STATE.exists():
                raise FileExistsError("preserve existing SW0156 prediction queue state")
            _atomic_json(STATE, {"experiment": EXPERIMENT, "queue_version": 1,
                "status": "waiting_for_source_screen", "supervisor_pid": os.getpid(),
                "tasks": self.plan, "implementation": self.fingerprint,
                "ground_truth_used_for_prediction": False})
        except BaseException:
            self._unlock(); raise

    def _unlock(self):
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_UN)
        finally:
            self.lock_stream.close()

    def _update(self, status, **fields):
        with self.state_lock:
            record = json.loads(STATE.read_text(encoding="utf-8"))
            record.update(status=status, updated=time.time(), **fields)
            arm = fields.get("arm")
            if arm in GPU_ARMS:
                arm_states = record.setdefault("arm_states", {})
                arm_record = arm_states.setdefault(arm, {})
                arm_record.update(status=status, updated=time.time(), **fields)
            _atomic_json(STATE, record)

    def _verify(self):
        if implementation_fingerprint() != self.fingerprint:
            raise RuntimeError("SW0156 prediction queue dependencies changed")

    def _run_gpu_arm(self, arm, upstream):
        target = predict.prediction_dir(arm)
        if target.exists():
            try:
                _valid_completed(arm)
                self._update("adopted_existing_prediction", arm=arm)
                return
            except (OSError, ValueError, TypeError, KeyError):
                _target_state(target)
        LOGS.mkdir(parents=True, exist_ok=True)
        prior_attempts = list(LOGS.glob(f"{arm}.attempt*.log"))
        attempt = len(prior_attempts) + 1
        log_path = LOGS / f"{arm}.attempt{attempt:02d}.log"
        while log_path.exists():
            attempt += 1
            log_path = LOGS / f"{arm}.attempt{attempt:02d}.log"
        proc = lease = None
        while not self.stop_event.is_set():
            gpu, gpu_uuid, memory, lease = owner._acquire(self.stop_event)
            try:
                stable = owner._stable_idle(gpu, gpu_uuid, self.stop_event,
                    lambda elapsed, used: self._update("waiting_stable_gpu", arm=arm,
                        gpu=gpu, gpu_uuid=gpu_uuid, stable_seconds=elapsed,
                        memory_used_mib=used))
                if not stable:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                    if self.stop_event.is_set():
                        return
                    self._update("waiting_for_exclusive_gpu", arm=arm,
                                 reservation_released=True)
                    continue
                self._verify()
                if _upstream() != upstream:
                    raise RuntimeError("SW0156 source-screen or source reference changed before prediction")
                if log_path.exists():
                    raise FileExistsError("preserve prediction attempt log collision before launch")
                # A recognized ledger is an intentional resume point. The
                # predictor revalidates its header and every committed row.
                _target_state(target)
                argv = [sys.executable, "-m", "collaborative_test.SW_0156_signed_coupling.predict",
                        "--arm", arm, "--device", "cuda:0"]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                with log_path.open("x", encoding="utf-8") as log:
                    proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                        pass_fds=(lease.fileno(),))
                    self._update("running_prediction", arm=arm, child_pid=proc.pid,
                        child_argv=argv, physical_gpu=gpu, gpu_uuid=gpu_uuid,
                        memory_before_mib=memory, inherited_lease_fd=True, log=str(log_path))
                    while proc.poll() is None and not self.stop_event.is_set():
                        time.sleep(POLL_SECONDS)
                        if proc.poll() is not None:
                            break
                        try:
                            current_uuid, _used, owners = owner.gpu_guard._gpu_info(gpu)
                            foreign = (["gpu_uuid_changed"] if current_uuid != gpu_uuid else
                                owner.gpu_guard._foreign_owners(owners,
                                    owner.gpu_guard._process_tree(proc.pid)))
                        except Exception as exc:
                            self._update("running_prediction", arm=arm,
                                         last_gpu_query_error=repr(exc))
                            continue
                        if foreign:
                            owner._stop_owned(proc)
                            self._update("failed_foreign_owner", arm=arm,
                                foreign_owner_pids=foreign, child_pid=proc.pid,
                                finished=time.time())
                            raise RuntimeError("foreign GPU owner collided with SW0156 child")
                rc = int(proc.wait())
                if rc != 0:
                    raise RuntimeError(f"SW0156 {arm} predictor exited {rc}; log preserved at {log_path}")
                _valid_completed(arm)
                self._update("prediction_complete", arm=arm, child_returncode=rc,
                             prediction_sha256=sha256_file(target / "predictions.npz"))
                return
            finally:
                _cleanup_child_before_lease_release(proc, lease)
                lease = None
        raise RuntimeError("SW0156 prediction queue stopped")

    def run(self):
        try:
            while not self.stop_event.is_set():
                upstream = _upstream()
                if upstream is not None:
                    break
                self._update("waiting_for_source_screen")
                self.stop_event.wait(10.0)
            if self.stop_event.is_set():
                return 1
            self._verify()
            if predict.prediction_dir("source_T").exists():
                _valid_completed("source_T")
            else:
                predict.predict_arm("source_T", device="cpu")
                _valid_completed("source_T")
            self._update("source_reference_ready", source_prediction_sha256=
                         _valid_completed("source_T")["prediction_sha256"])
            # Keep both registered interventions moving when two exclusive
            # devices are available; each worker still acquires its own lease.
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(self._run_gpu_arm, arm, upstream)
                           for arm in GPU_ARMS]
                for future in futures:
                    future.result()
            # No GPU lease is held during final scoring; evaluator revalidates
            # every arm and Slot artifact before opening any target mask.
            self._verify()
            if SCORE.exists():
                raise FileExistsError("preserve unexpected SW0156 score output")
            argv = [sys.executable, "-m", "collaborative_test.SW_0156_signed_coupling.evaluate"]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            score_log = LOGS / "score_seed1.log"
            if score_log.exists():
                raise FileExistsError(f"preserve prior score log: {score_log}")
            with score_log.open("x", encoding="utf-8") as log:
                proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                self._update("scoring_gt_last", child_pid=proc.pid, child_argv=argv,
                             log=str(score_log), cuda_visible_devices="")
                rc = int(proc.wait())
            if rc != 0 or not SCORE.is_file() or not evaluate.validate_score(SCORE):
                raise RuntimeError(f"SW0156 GT-last scorer exited {rc}; log preserved")
            self._update("complete", score_sha256=sha256_file(SCORE), finished=time.time(),
                         ground_truth_used_for_prediction=False,
                         ground_truth_used_for_scoring=True)
            return 0
        except BaseException as exc:
            self._update("failed", error=repr(exc), finished=time.time())
            raise
        finally:
            self._unlock()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"status": "planned", "tasks": task_plan(),
            "cuda_visible_devices_parent": "", "implementation": implementation_fingerprint()},
            allow_nan=False), flush=True)
        return 0
    return PredictionQueue().run()


if __name__ == "__main__":
    raise SystemExit(main())
