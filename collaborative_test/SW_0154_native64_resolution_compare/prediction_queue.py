"""Owner-aware fixed320 native64 source prediction queue for seeds0/1/2."""
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
except ImportError:  # CPU tests work on Windows; GPU reservation is Linux-only.
    fcntl = None

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for item in (ROOT, ROOT / "collaborative_test", ROOT / "snn_kuramoto_bidirectional"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4 as owner
from collaborative_test.SW_0154_native64_resolution_compare import (
    evaluate_native64 as evaluate, predict_native64 as predict,
    resource_probe, resource_queue,
)

EXPERIMENT = "SW0154_native64_resolution_compare"
SEEDS = (0, 1, 2)
RUNTIME = HERE / "results_archive" / "prediction_queue_20261013"
STATE = RUNTIME / "state.json"
QUEUE_LOCK = RUNTIME / "queue.lock"
EVALUATION_OUTPUT = HERE / "results_archive" / "native64_evaluation.json"
SAMPLE_SECONDS = owner.SAMPLE_SECONDS
MAX_MEMORY_MIB = owner.MAX_MEMORY_MIB
OWNER_GUARD = owner.OWNER_GUARD
OWNER_GUARD_SHA256 = owner.OWNER_GUARD_SHA256


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint():
    paths = (HERE / "prediction_queue.py", HERE / "predict_native64.py",
        HERE / "evaluate_native64.py", HERE / "resource_probe.py", HERE / "resource_queue.py",
        HERE / "native64_model.py", HERE / "resolution64.py", HERE / "score_slot64.py",
        HERE / "design.json", ROOT / "collaborative_test/SW_0148_checkpointable_boundary_q/stage_queue_v4.py",
        OWNER_GUARD)
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("SW0154 prediction queue dependency missing")
    if sha256_file(OWNER_GUARD) != OWNER_GUARD_SHA256:
        raise ValueError("registered shared GPU owner helper changed")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def task_plan():
    return [{"task_id": f"predict_seed{seed}", "seed": seed,
             "stage": "native64_fixed320_source_prediction", "device": "cuda:0",
             "output": str(predict.PREDICTION_ROOT / f"seed{seed}"),
             "work": str(predict.WORK_ROOT / f"seed{seed}")}
            for seed in SEEDS]


def command(task, *, physical_gpu):
    expected = {row["task_id"]: row for row in task_plan()}
    if task != expected.get(task.get("task_id")) or physical_gpu not in range(4):
        raise ValueError("only fixed320 native64 predictions and physical GPUs0..3 are allowed")
    return [sys.executable, "-m", "collaborative_test.SW_0154_native64_resolution_compare.predict_native64",
            "--seed", str(task["seed"]), "--device", "cuda:0"]


def valid_prediction(seed):
    try:
        evaluate.validate_native_prediction(seed)
        return True
    except (OSError, ValueError, TypeError, KeyError, AssertionError, RuntimeError):
        return False


def valid_evaluation():
    try:
        record = json.loads(EVALUATION_OUTPUT.read_text(encoding="utf-8"))
        return (record.get("status") == "complete"
            and record.get("experiment") == EXPERIMENT
            and record.get("seed_scope") == list(SEEDS)
            and record.get("ground_truth_opened_after_all_six_predictions_validated") is True
            and record.get("implementation_fingerprint") == evaluate.implementation_fingerprint())
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _check_unwritten(paths):
    existing = [str(Path(path)) for path in paths if Path(path).exists()]
    if existing:
        raise FileExistsError(f"preserve existing SW0154 queue artifacts: {existing}")


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


class PredictionQueue:
    def __init__(self, max_parallel_gpu_tasks=3):
        if not isinstance(max_parallel_gpu_tasks, int) or isinstance(max_parallel_gpu_tasks, bool) or not 1 <= max_parallel_gpu_tasks <= 3:
            raise ValueError("max parallel native64 prediction tasks must be 1..3")
        self.max_parallel_gpu_tasks = max_parallel_gpu_tasks
        if fcntl is None:
            raise RuntimeError("SW0154 prediction queue requires Linux flock")
        RUNTIME.mkdir(parents=True, exist_ok=True)
        self.lock_stream = QUEUE_LOCK.open("a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock_stream.close()
            raise RuntimeError("another SW0154 prediction supervisor owns the queue") from exc
        self.guard = threading.RLock()
        self.stop_event = threading.Event()
        try:
            self.queue_fp = implementation_fingerprint()
            self.prediction_fp = predict.implementation_fingerprint()
            self.resource_fp = resource_probe.implementation_fingerprint()
            if not resource_queue.validate_report(expected_fingerprint=self.resource_fp):
                raise ValueError("SW0154 one-image native64 source resource probe must pass first")
            self.tasks = task_plan()
            existing = {row["task_id"]: valid_prediction(row["seed"]) for row in self.tasks}
            invalid_outputs = [row["output"] for row in self.tasks
                if Path(row["output"]).exists() and not existing[row["task_id"]]]
            if invalid_outputs:
                raise FileExistsError(f"preserve invalid or partial completed prediction directories: {invalid_outputs}")
            self.evaluation_complete = False
            if EVALUATION_OUTPUT.exists():
                if not valid_evaluation() or not all(existing.values()):
                    raise FileExistsError("preserve invalid/differently-bound native64 evaluation output")
                self.evaluation_complete = True
            _check_unwritten((STATE,))
            self.state = {"experiment": EXPERIMENT, "queue_version": 1,
                "status": "queued", "supervisor_pid": os.getpid(),
                "queue_fingerprint": self.queue_fp, "prediction_fingerprint": self.prediction_fp,
                "resource_probe_path": str(resource_probe.HERE / "results_archive" / resource_probe.REPORT_NAME),
                "resource_probe_sha256": sha256_file(resource_probe.HERE / "results_archive" / resource_probe.REPORT_NAME),
                "evaluation_output": str(EVALUATION_OUTPUT),
                "tasks": {row["task_id"]: {"task": row,
                    "status": "verified_existing" if existing[row["task_id"]] else "queued"}
                    for row in self.tasks}}
            _atomic_json(STATE, self.state)
        except BaseException:
            try:
                fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_UN)
            finally:
                self.lock_stream.close()
            raise

    def _update(self, task_id, status, **fields):
        with self.guard:
            self.state.update(status="predictions_running", updated=time.time())
            row = self.state["tasks"][task_id]
            row.update(status=status, updated=time.time(), **fields)
            _atomic_json(STATE, self.state)

    def _verify_fingerprints(self):
        if (implementation_fingerprint() != self.queue_fp
                or predict.implementation_fingerprint() != self.prediction_fp
                or resource_probe.implementation_fingerprint() != self.resource_fp):
            raise RuntimeError("SW0154 prediction/resource implementation changed after registration")
        report = resource_probe.HERE / "results_archive" / resource_probe.REPORT_NAME
        if not resource_queue.validate_report(report, expected_fingerprint=self.resource_fp):
            raise RuntimeError("SW0154 source resource admission report changed")

    def _execute_seed(self, task):
        task_id, seed = task["task_id"], task["seed"]
        if valid_prediction(seed):
            self._update(task_id, "verified_existing")
            return True
        output_dir = Path(task["output"])
        if output_dir.exists():
            self._update(task_id, "failed_existing_output_preserved")
            return False
        attempt = 1
        preferred_gpu = None
        preferred_uuid = None
        while not self.stop_event.is_set():
            self._verify_fingerprints()
            gpu, gpu_uuid, memory, lease = owner._acquire(self.stop_event, preferred_gpu)
            if preferred_uuid is not None and gpu_uuid != preferred_uuid:
                if fcntl is not None:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                lease.close()
                self._update(task_id, "failed_resume_gpu_identity_changed",
                             expected_uuid=preferred_uuid, observed_uuid=gpu_uuid)
                return False
            process = None
            log = RUNTIME / f"{task_id}.attempt{attempt:02d}.log"
            retry_foreign = False
            try:
                if not owner._stable_idle(gpu, gpu_uuid, self.stop_event,
                    lambda elapsed, used: self._update(task_id, "waiting_stable_gpu",
                        gpu=gpu, gpu_uuid=gpu_uuid, stable_seconds=elapsed, memory_used_mib=used)):
                    if fcntl is not None:
                        fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                    lease.close()
                    if self.stop_event.is_set():
                        return False
                    self._update(task_id, "queued", reservation_released=True)
                    continue
                self._verify_fingerprints()
                if valid_prediction(seed):
                    self._update(task_id, "verified_existing")
                    return True
                if output_dir.exists() or log.exists():
                    self._update(task_id, "failed_existing_output_preserved")
                    return False
                argv = command(task, physical_gpu=gpu)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                    CUDA_DEVICE_ORDER="PCI_BUS_ID", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                with log.open("x", encoding="utf-8") as stream:
                    process = subprocess.Popen(argv, cwd=ROOT, env=env,
                        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                        start_new_session=True, pass_fds=(lease.fileno(),))
                    self._update(task_id, "running", child_pid=process.pid, argv=argv,
                        physical_gpu=gpu, gpu_uuid=gpu_uuid, memory_before_mib=memory,
                        inherited_lease_fd=True, log=str(log), attempt=attempt)
                    while process.poll() is None and not self.stop_event.is_set():
                        time.sleep(SAMPLE_SECONDS)
                        if process.poll() is not None:
                            break
                        try:
                            current_uuid, _, owners = owner.gpu_guard._gpu_info(gpu)
                            foreign = (["gpu_uuid_changed"] if current_uuid != gpu_uuid else
                                owner.gpu_guard._foreign_owners(owners,
                                    owner.gpu_guard._process_tree(process.pid)))
                        except Exception as exc:
                            self._update(task_id, "running", last_gpu_query_error=repr(exc))
                            continue
                        if foreign:
                            self._update(task_id, "interrupted_foreign_gpu_owner", foreign_owner_pids=foreign)
                            owner._stop_owned(process)
                            process.wait()
                            retry_foreign = True
                            preferred_gpu, preferred_uuid = gpu, gpu_uuid
                            break
                rc = int(process.wait())
                if retry_foreign:
                    self._update(task_id, "queued_after_foreign_owner",
                        partial_work_preserved=Path(task["work"]).exists(), retry=attempt + 1,
                        prior_foreign_owner_pids=self.state["tasks"][task_id].get(
                            "foreign_owner_pids", []))
                else:
                    valid = rc == 0 and valid_prediction(seed)
                    self._update(task_id, "passed" if valid else "failed_child_or_artifact_validation",
                        child_returncode=rc, artifact_valid=bool(valid), finished=time.time())
                    return bool(valid)
            except BaseException as exc:
                if process is not None and process.poll() is None:
                    owner._stop_owned(process)
                self._update(task_id, "failed", error=repr(exc), finished=time.time())
                return False
            finally:
                if lease is not None:
                    if fcntl is not None:
                        fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                    lease.close()
            if retry_foreign:
                if not valid_prediction(seed) and Path(task["output"]).exists():
                    self._update(task_id, "failed_existing_output_preserved")
                    return False
                attempt += 1
        return False

    def run(self):
        try:
            self._verify_fingerprints()
            if self.evaluation_complete:
                self.state.update(status="complete", evaluation_verified_existing=True)
                _atomic_json(STATE, self.state)
                return 0
            with ThreadPoolExecutor(max_workers=self.max_parallel_gpu_tasks) as pool:
                futures = [pool.submit(self._execute_seed, task) for task in self.tasks]
                passed = [future.result() for future in futures]
            if not all(passed):
                self.state.update(status="predictions_failed", finished=time.time())
                _atomic_json(STATE, self.state)
                return 1
            # Revalidate all six immutable prediction inputs immediately before the CPU scorer opens GT.
            for seed in SEEDS:
                if not valid_prediction(seed):
                    raise ValueError(f"native64 seed{seed} prediction failed terminal validation")
            if EVALUATION_OUTPUT.exists():
                raise FileExistsError("preserve existing native64 score output")
            from collaborative_test.SW_0154_native64_resolution_compare.evaluate_native64 import score_all
            report = score_all()
            with EVALUATION_OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
                stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
            if not valid_evaluation():
                raise ValueError("native64 evaluation artifact failed terminal validation")
            self.state.update(status="complete", evaluation_sha256=sha256_file(EVALUATION_OUTPUT),
                              finished=time.time())
            _atomic_json(STATE, self.state)
            return 0
        finally:
            try:
                fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_UN)
            finally:
                self.lock_stream.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-parallel-gpu-tasks", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args(argv)
    if args.dry_run:
        report = resource_probe.HERE / "results_archive" / resource_probe.REPORT_NAME
        if not resource_queue.validate_report(report):
            raise RuntimeError("native64 resource probe is not valid")
        print(json.dumps({"status": "resource_probe_validated", "tasks": task_plan(),
            "max_parallel_gpu_tasks": args.max_parallel_gpu_tasks,
            "resource_probe_sha256": sha256_file(report),
            "queue_fingerprint": implementation_fingerprint()}, allow_nan=False), flush=True)
        return 0
    return PredictionQueue(args.max_parallel_gpu_tasks).run()


if __name__ == "__main__":
    raise SystemExit(main())
