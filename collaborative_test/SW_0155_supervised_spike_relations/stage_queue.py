"""Owner-aware fixed320 inference and GT-last score queue for SW0155."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

try:
    import fcntl
except ImportError:  # Windows supports plan/CPU tests only.
    fcntl = None

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"),
                str(ROOT / "snn_kuramoto_bidirectional")]

from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4 as owner
from collaborative_test.SW_0155_supervised_spike_relations import (
    artifactvalidator, predict_evaluate as evaluation, train,
)

EXPERIMENT = train.EXPERIMENT
ARMS = train.ARMS
RUNTIME = HERE / "results_archive" / "prediction_queue_20261013"
STATE = RUNTIME / "state.json"
LOCK = RUNTIME / "queue.lock"
EVALUATION_OUTPUT = evaluation.EVALUATION_OUTPUT
GPU_LEASE_DIR = Path("/tmp/kevinswk_sw0113_gpu_leases")
SAMPLE_SECONDS = owner.SAMPLE_SECONDS
OWNER_GUARD = owner.OWNER_GUARD
OWNER_GUARD_SHA256 = owner.OWNER_GUARD_SHA256


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint():
    paths = (HERE / "stage_queue.py", HERE / "predict_evaluate.py", HERE / "protocol.json",
        HERE / "train.py", HERE / "objective.py", HERE / "artifactvalidator.py",
        HERE / "preflight.py", HERE / "preflight_migration.py",
        ROOT / "collaborative_test/SW_0148_checkpointable_boundary_q/stage_queue_v4.py",
        OWNER_GUARD)
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("SW0155 queue dependency is missing")
    if sha256_file(OWNER_GUARD) != OWNER_GUARD_SHA256:
        raise ValueError("registered GPU owner helper changed")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def task_plan():
    return [{"task_id": f"predict_{arm}", "arm": arm, "seed": 1,
             "stage": "fixed320_qcc_prediction", "device": "cuda:0",
             "output": str(evaluation.PREDICTION_ROOT / f"{arm}_seed1")}
            for arm in ARMS]


def command(task, *, physical_gpu):
    rows = {row["task_id"]: row for row in task_plan()}
    if task != rows.get(task.get("task_id")) or physical_gpu not in range(4):
        raise ValueError("only registered SW0155 seed1 QCC tasks may launch")
    return [sys.executable, "-m", "collaborative_test.SW_0155_supervised_spike_relations.predict_evaluate",
            "--stage", "predict", "--arm", task["arm"], "--device", "cuda:0"]


def valid_training(arm):
    task = {"arm": arm, "output": str(train.TRAIN_ROOT / f"{arm}_seed1")}
    return artifactvalidator.validate_training_artifact(task)


def valid_prediction(arm):
    try:
        evaluation.validate_prediction(arm)
        return True
    except (OSError, ValueError, TypeError, KeyError, AssertionError, RuntimeError):
        return False


def valid_evaluation():
    try:
        record = json.loads(EVALUATION_OUTPUT.read_text(encoding="utf-8"))
        return (record.get("status") == "complete" and record.get("experiment") == EXPERIMENT
            and record.get("seed_scope") == [1]
            and record.get("ground_truth_opened_after_all_predictions_validated") is True
            and record.get("ground_truth_used_for_prediction") is False
            and record.get("ground_truth_used_for_scoring") is True
            and record.get("implementation_fingerprint") == evaluation.implementation_fingerprint()
            and set(record.get("scores", {})) == {*ARMS, "source97", "SW0143_T"}
            and record.get("leaderboard") == "supervised SW0155 seed1 pilot; separate from unsupervised leaderboard")
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


class PredictionQueue:
    def __init__(self):
        if fcntl is None:
            raise RuntimeError("SW0155 GPU queue requires Linux flock")
        RUNTIME.mkdir(parents=True, exist_ok=True)
        self.lock_stream = LOCK.open("a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock_stream.close()
            raise RuntimeError("another SW0155 prediction supervisor owns the queue") from exc
        self.guard = threading.RLock()
        self.stop_event = threading.Event()
        try:
            self.queue_fp = implementation_fingerprint()
            self.prediction_fp = evaluation.implementation_fingerprint()
            self.tasks = task_plan()
            for task in self.tasks:
                arm = task["arm"]
                manifest = train.TRAIN_ROOT / f"{arm}_seed1" / "manifest.json"
                if manifest.exists() and not valid_training(arm):
                    raise ValueError(f"preserve invalid completed SW0155 training artifact: {arm}")
                exists = Path(task["output"]).exists()
                if exists and (not valid_training(arm) or not valid_prediction(arm)):
                    raise FileExistsError(f"preserve invalid or partial output: {task['output']}")
            self.existing_score = False
            if EVALUATION_OUTPUT.exists():
                if not valid_evaluation() or not all(valid_prediction(arm) for arm in ARMS):
                    raise FileExistsError("preserve invalid/differently-bound SW0155 score")
                self.existing_score = True
            if STATE.exists():
                raise FileExistsError(f"preserve existing SW0155 queue state: {STATE}")
            self.state = {"experiment": EXPERIMENT, "queue_version": 1,
                "status": "queued", "supervisor_pid": os.getpid(),
                "queue_fingerprint": self.queue_fp,
                "prediction_fingerprint": self.prediction_fp,
                "score_output": str(EVALUATION_OUTPUT),
                "tasks": {row["task_id"]: {"task": row,
                    "status": "verified_existing" if valid_prediction(row["arm"]) else "queued"}
                    for row in self.tasks}}
            _atomic_json(STATE, self.state)
        except BaseException:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_UN)
            self.lock_stream.close()
            raise

    def _update(self, task_id, status, **fields):
        with self.guard:
            self.state.update(status="predictions_running", updated=time.time())
            self.state["tasks"][task_id].update(status=status, updated=time.time(), **fields)
            _atomic_json(STATE, self.state)

    def _verify(self):
        if implementation_fingerprint() != self.queue_fp or evaluation.implementation_fingerprint() != self.prediction_fp:
            raise RuntimeError("SW0155 prediction code changed after queue registration")

    def _wait_for_training(self):
        """Wait without reserving a GPU until both immutable TRAIN artifacts validate."""
        training_state = train.HERE / "results_archive" / "training_queue_20261013" / "state.json"
        while not self.stop_event.is_set():
            manifests = [train.TRAIN_ROOT / f"{arm}_seed1" / "manifest.json" for arm in ARMS]
            if all(path.is_file() for path in manifests):
                if all(valid_training(arm) for arm in ARMS):
                    return True
                self.state["status"] = "blocked_invalid_training_completion"
                _atomic_json(STATE, self.state)
                return False
            if training_state.is_file():
                try:
                    state = json.loads(training_state.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    state = {}
                if state.get("status") == "training_failed":
                    self.state["status"] = "blocked_training_failed"
                    self.state["training_state_path"] = str(training_state)
                    _atomic_json(STATE, self.state)
                    return False
                if state.get("status") == "training_complete":
                    self.state["status"] = "blocked_invalid_training_completion"
                    self.state["training_state_path"] = str(training_state)
                    _atomic_json(STATE, self.state)
                    return False
            time.sleep(10)
        return False

    def _execute(self, task):
        task_id, arm = task["task_id"], task["arm"]
        if valid_prediction(arm):
            self._update(task_id, "verified_existing")
            return True
        if not valid_training(arm):
            self._update(task_id, "failed_training_artifact_invalid")
            return False
        output = Path(task["output"])
        log = RUNTIME / f"{task_id}.attempt01.log"
        if output.exists() or log.exists():
            self._update(task_id, "failed_existing_output_preserved")
            return False
        lease = None
        proc = None
        try:
            while not self.stop_event.is_set():
                gpu, gpu_uuid, memory, lease = owner._acquire(self.stop_event)
                stable = owner._stable_idle(gpu, gpu_uuid, self.stop_event,
                    lambda elapsed, used: self._update(task_id, "waiting_stable_gpu",
                        physical_gpu=gpu, gpu_uuid=gpu_uuid, stable_seconds=elapsed,
                        memory_used_mib=used))
                if stable:
                    break
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                lease.close()
                lease = None
                if self.stop_event.is_set():
                    self._update(task_id, "cancelled_before_launch", reservation_released=True)
                    return False
                self._update(task_id, "queued", reservation_released=True)
            if lease is None:
                self._update(task_id, "cancelled_before_launch")
                return False
            self._verify()
            if valid_prediction(arm):
                self._update(task_id, "verified_existing")
                return True
            if output.exists() or log.exists():
                self._update(task_id, "failed_existing_output_preserved")
                return False
            argv = command(task, physical_gpu=gpu)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                CUDA_DEVICE_ORDER="PCI_BUS_ID", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            with log.open("x", encoding="utf-8") as stream:
                proc = subprocess.Popen(argv, cwd=ROOT, env=env,
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=True, pass_fds=(lease.fileno(),))
                self._update(task_id, "running", child_pid=proc.pid, argv=argv,
                    physical_gpu=gpu, gpu_uuid=gpu_uuid, memory_before_mib=memory,
                    inherited_lease_fd=True, log=str(log))
                while proc.poll() is None and not self.stop_event.is_set():
                    time.sleep(SAMPLE_SECONDS)
                    if proc.poll() is not None:
                        break
                    try:
                        current_uuid, _, owners = owner.gpu_guard._gpu_info(gpu)
                        foreign = (["gpu_uuid_changed"] if current_uuid != gpu_uuid else
                            owner.gpu_guard._foreign_owners(owners,
                                owner.gpu_guard._process_tree(proc.pid)))
                    except Exception as exc:
                        self._update(task_id, "running", last_gpu_query_error=repr(exc))
                        continue
                    if foreign:
                        self._update(task_id, "interrupted_foreign_gpu_owner",
                                     foreign_owner_pids=foreign)
                        owner._stop_owned(proc)
                        return False
            rc = int(proc.wait())
            valid = rc == 0 and valid_prediction(arm)
            self._update(task_id, "passed" if valid else "failed_child_or_artifact_validation",
                         child_returncode=rc, artifact_valid=bool(valid), finished=time.time())
            return bool(valid)
        except BaseException as exc:
            if proc is not None and proc.poll() is None:
                owner._stop_owned(proc)
            self._update(task_id, "failed", error=repr(exc), finished=time.time())
            return False
        finally:
            if lease is not None:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                lease.close()

    def run(self):
        try:
            self._verify()
            if self.existing_score:
                self.state["status"] = "complete_existing_score"
                _atomic_json(STATE, self.state)
                return 0
            if not self._wait_for_training():
                self.state["status"] = self.state.get("status", "stopped_waiting_for_training")
                _atomic_json(STATE, self.state)
                return 1
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = [future.result() for future in
                           [pool.submit(self._execute, task) for task in self.tasks]]
            complete = all(results) and all(valid_prediction(arm) for arm in ARMS)
            if complete:
                self.state["status"] = "scoring_gt_last"
                _atomic_json(STATE, self.state)
                try:
                    report = evaluation.score_all()
                    complete = report.get("status") == "complete" and valid_evaluation()
                except Exception as exc:
                    self.state["score_error"] = repr(exc)
                    complete = False
            self.state["status"] = "complete" if complete else "prediction_or_score_failed"
            self.state["finished"] = time.time()
            _atomic_json(STATE, self.state)
            return 0 if complete else 1
        finally:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_UN)
            self.lock_stream.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"experiment": EXPERIMENT, "status": "dry_run",
            "tasks": task_plan(), "score_output": str(EVALUATION_OUTPUT),
            "queue_fingerprint": implementation_fingerprint()}, allow_nan=False), flush=True)
        return 0
    return PredictionQueue().run()


if __name__ == "__main__":
    raise SystemExit(main())
