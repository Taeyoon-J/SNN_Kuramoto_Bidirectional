"""Owner-aware SW0152 training, fixed320 prediction and GT-last score queue."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import threading
import time
try:
    import fcntl
except ImportError:  # Windows is for plan/tests only; the GPU queue requires Linux flock.
    fcntl = None

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"),
               str(ROOT / "snn_kuramoto_bidirectional")]

from collaborative_test.SW_0126_history_event_binding import pilot_queue as gpu_guard
from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4 as owner_queue
from collaborative_test.SW_0152_source_preserving_partition_equivariance import evaluate, preflight, train

NAME = train.NAME
RUNTIME = HERE / "results_archive" / "stage_queue_20261013"
STATE = RUNTIME / "state.json"
QUEUE_LOCK = RUNTIME / "queue.lock"
GPU_LEASE_DIR = Path("/tmp/kevinswk_sw0113_gpu_leases")
OWNER_GUARD = ROOT / "collaborative_test/SW_0126_history_event_binding/pilot_queue.py"
OWNER_GUARD_SHA256 = owner_queue.OWNER_GUARD_SHA256
OWNER_QUEUE = ROOT / "collaborative_test/SW_0148_checkpointable_boundary_q/stage_queue_v4.py"
MAX_MEMORY_MIB = 512
SAMPLE_SECONDS = 5.0


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_fingerprint():
    paths = (HERE / "stage_queue.py", HERE / "train.py", HERE / "evaluate.py",
             HERE / "preflight.py", HERE / "preflight_queue.py", HERE / "protocol.json",
             HERE / "pilot_reference.json",
             HERE / "training.py",
             ROOT / "collaborative_test/SW_0150_spike_spatial_mixture/integration.py", ROOT / "collaborative_test/SW_0150_spike_spatial_mixture/spatial_diffusion.py",
             ROOT / "collaborative_test/SW_0150_spike_spatial_mixture/seed_selection.py",
             HERE / "partition_objective.py",
             HERE / "deep_encoder.py", HERE / "view_augmentation.py",
             ROOT / "collaborative_test/SW_0148_checkpointable_boundary_q/checkpoint.py",
             ROOT / "collaborative_test/SW_0148_checkpointable_boundary_q/stage_queue_v4.py",
             OWNER_GUARD)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"SW0152 stage-queue dependency missing: {missing[0]}")
    if sha256_file(OWNER_GUARD) != OWNER_GUARD_SHA256:
        raise ValueError("registered GPU owner dispatcher changed")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def implementation_sha256():
    payload = json.dumps(implementation_fingerprint(), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def task_plan():
    tasks = []
    for arm in train.ARMS:
        tasks.append({"task_id": f"train_{arm}", "stage": "train", "arm": arm,
                      "depends_on": ["external_preflight"],
                      "output": str(train.TRAIN / f"{arm}_seed1")})
    for arm in train.ARMS:
        tasks.append({"task_id": f"predict_{arm}", "stage": "predict", "arm": arm,
                      "depends_on": [f"train_{arm}"],
                      "output": str(Path(evaluate.PREDICTION_ROOT) / f"{arm}_seed1")})
    tasks.append({"task_id": "predict_source_p", "stage": "predict", "arm": "source_p",
                  "depends_on": ["external_preflight"],
                  "output": str(Path(evaluate.PREDICTION_ROOT) / "source_p_seed1")})
    tasks.append({"task_id": "score", "stage": "score", "arm": None,
                  "depends_on": [f"predict_{arm}" for arm in train.ARMS] + ["predict_source_p"],
                  "output": str(evaluate.SUMMARY)})
    return tasks


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def valid_result(task):
    try:
        stage, arm = task["stage"], task.get("arm")
        if stage == "train":
            train.validate_training(arm)
        elif stage == "predict":
            evaluate.validate_prediction(arm)
        elif stage == "score":
            report = _read_json(task["output"])
            if (report.get("status") != "complete" or report.get("experiment") != NAME
                    or report.get("seed_scope") != [1]
                    or report.get("ground_truth_opened_after_all_predictions_validated") is not True
                    or report.get("ground_truth_used_for_prediction") is not False
                    or report.get("ground_truth_used_for_scoring") is not True
                    or set(report.get("scores", {})) != {
                        *set(train.ARMS), "source_p", "source97_qcc", "slot32_seed1"}):
                return False
        else:
            return False
        return True
    except (OSError, ValueError, TypeError, KeyError, AssertionError, RuntimeError,
            EOFError, pickle.UnpicklingError):
        return False


def _argv(task, *, gpu=0):
    if task["stage"] == "train":
        return [sys.executable, "-m", "collaborative_test.SW_0152_source_preserving_partition_equivariance.train",
                "--arm", task["arm"], "--device", "cuda:0"]
    if task["stage"] == "predict":
        return [sys.executable, "-m", "collaborative_test.SW_0152_source_preserving_partition_equivariance.evaluate",
                "--stage", "predict", "--arm", task["arm"], "--device", "cuda:0"]
    return [sys.executable, "-m", "collaborative_test.SW_0152_source_preserving_partition_equivariance.evaluate",
            "--stage", "score"]


def _required_resume_gpu(task):
    if task.get("stage") != "train":
        return None
    from collaborative_test.SW_0148_checkpointable_boundary_q import checkpoint as durable
    directory = train.WORK / f"{task['arm']}_seed1" / "checkpoints"
    boundary = directory / "boundary.ptx"
    if not boundary.exists():
        if directory.exists() and not durable.is_only_uncommitted_temp(directory):
            raise ValueError("uncommitted work directory is not a recognized atomic temp")
        return None
    try:
        metadata = durable.inspect_committed_metadata(directory)
    except (OSError, ValueError, TypeError, KeyError, EOFError, RuntimeError,
            pickle.UnpicklingError) as exc:
        raise ValueError("SW0150 checkpoint integrity validation failed") from exc
    runtime = metadata.get("runtime")
    visible = runtime.get("cuda_visible_devices") if isinstance(runtime, dict) else None
    completed = metadata.get("completed_updates")
    if (not isinstance(visible, str) or visible not in {"0", "1", "2", "3"}
            or runtime.get("device_type") != "cuda" or runtime.get("device_index") != 0
            or isinstance(completed, bool) or not isinstance(completed, int)
            or not 0 <= completed <= train.UPDATES):
        raise ValueError("SW0150 resume boundary lacks valid physical-GPU binding")
    return int(visible)


def _safe_foreign_retry(task):
    if Path(task["output"]).exists():
        return False
    if task["stage"] in {"predict", "predict_source"}:
        return True
    if task["stage"] != "train":
        return False
    from collaborative_test.SW_0148_checkpointable_boundary_q import checkpoint as durable
    directory = train.WORK / f"{task['arm']}_seed1" / "checkpoints"
    boundary = directory / "boundary.ptx"
    if boundary.is_file():
        try:
            durable.inspect_committed_metadata(directory)
            return True
        except (OSError, ValueError, TypeError, KeyError, EOFError, RuntimeError,
                pickle.UnpicklingError):
            return False
    return not directory.exists() or durable.is_only_uncommitted_temp(directory)


def _external_preflight_state():
    report = Path(preflight.OUTPUT_DEFAULT)
    external_state = preflight_queue_state_path()
    if report.is_file():
        try:
            admission = train.validate_preflight(report)
        except (OSError, ValueError, TypeError, KeyError, RuntimeError, EOFError,
                pickle.UnpicklingError) as exc:
            try:
                state = _read_json(external_state) if external_state.is_file() else {}
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                state = {}
            terminal = state.get("status") in {"passed", "failed", "failed_preflight_or_artifact_validation",
                "failed_child", "cancelled_before_launch", "cancelled"}
            if not terminal:
                return "waiting_for_external_preflight", None, None
            return "blocked_invalid_external_preflight", None, repr(exc)
        return "passed", admission, None
    if external_state.is_file():
        try:
            state = _read_json(external_state)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return "waiting_for_external_preflight", None, None
        if state.get("status") in {"failed", "failed_preflight_or_artifact_validation", "failed_child",
                                    "cancelled_before_launch", "cancelled"}:
            return "blocked_external_preflight_failed", None, state.get("status")
        task = state.get("tasks", {}).get("preflight_seed1", {})
        if state.get("status") == "passed" or task.get("status") == "passed":
            return "blocked_invalid_external_preflight", None, "passed preflight lacks report"
    return "waiting_for_external_preflight", None, None


def preflight_queue_state_path():
    return preflight.HERE / "results_archive" / "preflight_queue_20261013" / "state.json"


def _atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    with tmp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


class StageQueue:
    def __init__(self):
        if fcntl is None:
            raise RuntimeError("SW0150 main queue requires Linux flock")
        RUNTIME.mkdir(parents=True, exist_ok=True)
        self.lock_stream = QUEUE_LOCK.open("a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock_stream.close()
            raise RuntimeError("another SW0150 main supervisor owns this queue") from exc
        if STATE.exists():
            self.lock_stream.close()
            raise FileExistsError(f"preserve existing SW0150 main queue state: {STATE}")
        if sha256_file(OWNER_GUARD) != OWNER_GUARD_SHA256:
            self.lock_stream.close()
            raise RuntimeError("reviewed SW0150 owner dispatcher changed")
        self.state_lock = threading.RLock()
        self.stop_event = threading.Event()
        self.tasks = task_plan()
        self.state = {"experiment": NAME, "queue_version": 1, "status": "running",
                      "supervisor_pid": os.getpid(), "owner_dispatcher_sha256": OWNER_GUARD_SHA256,
                      "owner_queue_sha256": sha256_file(OWNER_QUEUE),
                      "implementation_fingerprint": implementation_fingerprint(),
                      "implementation_sha256": implementation_sha256(),
                      "preflight": {"status": "waiting_for_external_preflight",
                                    "report": str(preflight.OUTPUT_DEFAULT)},
                      "tasks": {task["task_id"]: {"task": task, "status": "queued"}
                                for task in self.tasks}}
        self._save()

    def _save(self):
        with self.state_lock:
            _atomic_json(STATE, self.state)

    def _update(self, task_id, **fields):
        with self.state_lock:
            self.state["tasks"][task_id].update(fields)
            self._save()

    def _update_preflight(self, **fields):
        with self.state_lock:
            self.state["preflight"].update(fields)
            self._save()

    def _finish(self, status):
        with self.state_lock:
            self.state.update(status=status, finished=time.time())
            self._save()

    @staticmethod
    def _terminate_owned(proc):
        if proc is not None and proc.poll() is None:
            owner_queue._stop_owned(proc)

    def _execute_gpu(self, task):
        task_id, output = task["task_id"], Path(task["output"])
        if valid_result(task):
            self._update(task_id, status="verified_existing")
            return True
        if output.exists():
            self._update(task_id, status="failed_existing_output_preserved")
            return False
        attempt = 1
        self._update(task_id, status="waiting_for_exclusive_gpu")
        while not self.stop_event.is_set():
            try:
                preferred = _required_resume_gpu(task)
                gpu, gpu_uuid, memory, lease = owner_queue._acquire(self.stop_event, preferred)
            except BaseException as exc:
                self._update(task_id, status="failed_gpu_or_checkpoint_admission",
                             error=repr(exc), finished=time.time())
                return False
            proc = None
            retry = False
            log = RUNTIME / f"{task_id}.attempt{attempt:02d}.log"
            try:
                if not owner_queue._stable_idle(gpu, gpu_uuid, self.stop_event,
                        lambda elapsed, used: self._update(
                            task_id, status="waiting_stable_gpu", gpu=gpu,
                            gpu_uuid=gpu_uuid, stable_seconds=elapsed,
                            memory_used_mib=used)):
                    continue
                if output.exists() or log.exists():
                    self._update(task_id, status="failed_existing_output_preserved")
                    return False
                argv = _argv(task, gpu=gpu)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                with log.open("x", encoding="utf-8") as stream:
                    proc = subprocess.Popen(argv, cwd=ROOT, env=env,
                                            stdin=subprocess.DEVNULL, stdout=stream,
                                            stderr=subprocess.STDOUT, start_new_session=True,
                                            pass_fds=(lease.fileno(),))
                    self._update(task_id, status="running", child_pid=proc.pid, argv=argv,
                                 gpu=gpu, gpu_uuid=gpu_uuid, memory_before_mib=memory,
                                 inherited_lease_fd=True, log=str(log), attempt=attempt,
                                 started=time.time())
                    while proc.poll() is None and not self.stop_event.is_set():
                        time.sleep(SAMPLE_SECONDS)
                        if proc.poll() is not None:
                            break
                        try:
                            current_uuid, _used, owners = gpu_guard._gpu_info(gpu)
                            foreign = (["gpu_uuid_changed"] if current_uuid != gpu_uuid else
                                       gpu_guard._foreign_owners(owners,
                                           gpu_guard._process_tree(proc.pid)))
                        except Exception as exc:
                            self._update(task_id, last_gpu_query_error=repr(exc))
                            continue
                        if foreign:
                            self._terminate_owned(proc)
                            self._update(task_id, status="interrupted_foreign_gpu_owner",
                                         foreign_owner_pids=foreign, finished=time.time())
                            retry = True
                            break
                    if self.stop_event.is_set() and proc.poll() is None:
                        self._terminate_owned(proc)
                        self._update(task_id, status="cancelled_owned_child_stopped",
                                     finished=time.time())
                code = int(proc.wait())
                good = code == 0 and valid_result(task)
                self._update(task_id, status="passed" if good else "failed",
                             returncode=code, artifact_valid=bool(good), finished=time.time())
                if good:
                    return True
                if not retry:
                    return False
            except BaseException as exc:
                if proc is not None and proc.poll() is None:
                    self._terminate_owned(proc)
                self._update(task_id, status="failed", error=repr(exc), finished=time.time())
                return False
            finally:
                try:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                finally:
                    lease.close()
            if not _safe_foreign_retry(task):
                self._update(task_id, status="failed_foreign_retry_not_safe")
                return False
            attempt += 1
            self._update(task_id, status="queued_after_foreign_owner", attempt=attempt)
        self._update(task_id, status="cancelled")
        return False

    def _execute_cpu(self, task):
        if valid_result(task):
            self._update(task["task_id"], status="verified_existing")
            return True
        output = Path(task["output"])
        log = RUNTIME / f"{task['task_id']}.log"
        if output.exists() or log.exists():
            self._update(task["task_id"], status="failed_existing_artifact_preserved")
            return False
        argv = _argv(task)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        try:
            with log.open("x", encoding="utf-8") as stream:
                proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                        stdout=stream, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                self._update(task["task_id"], status="running", child_pid=proc.pid,
                             argv=argv, cuda_hidden=True, log=str(log))
                code = int(proc.wait())
            good = code == 0 and valid_result(task)
            self._update(task["task_id"], status="passed" if good else "failed",
                         returncode=code, artifact_valid=bool(good), finished=time.time())
            return good
        except BaseException as exc:
            self._update(task["task_id"], status="failed", error=repr(exc), finished=time.time())
            return False

    def _wait_external_preflight(self):
        while not self.stop_event.is_set():
            status, admission, error = _external_preflight_state()
            self._update_preflight(status=status, checked=time.time(), error=error)
            if status == "passed":
                self._update_preflight(report_sha256=admission["report_sha256"],
                                       implementation_sha256=train.implementation_sha())
                return admission
            if status.startswith("blocked_"):
                return None
            time.sleep(SAMPLE_SECONDS)
        return None

    def run(self):
        try:
            source_task = next(task for task in self.tasks if task["task_id"] == "predict_source_p")
            admission = self._wait_external_preflight()
            if admission is None:
                status = self.state["preflight"]["status"]
                self._finish(status if status.startswith("blocked_") else "cancelled")
                return 1
            train_tasks = [task for task in self.tasks if task["stage"] == "train"]
            with ThreadPoolExecutor(max_workers=3) as pool:
                train_futures = [pool.submit(self._execute_gpu, task) for task in train_tasks]
                source_future = pool.submit(self._execute_gpu, source_task)
                train_ok = [future.result() for future in train_futures]
                if not all(train_ok):
                    # A finished sibling is preserved; predict any valid completed arm
                    # only after both training tasks have reached terminal status.
                    valid_trains = [task for task, okay in zip(train_tasks, train_ok)
                                    if okay and valid_result(task)]
                else:
                    valid_trains = train_tasks
                pred_by_arm = {task["arm"]: task for task in self.tasks if task["stage"] == "predict"}
                candidates = [pred_by_arm[task["arm"]] for task in valid_trains]
                candidate_results = list(pool.map(self._execute_gpu, candidates)) if candidates else []
                source_ok = source_future.result()
            candidate_arms = {task.get("arm") for task in candidates}
            ready = (candidate_arms == set(train.ARMS)
                     and len(candidate_results) == len(train.ARMS)
                     and all(candidate_results) and source_ok
                     and all(valid_result(task) for task in candidates)
                     and valid_result(source_task))
            if not ready:
                self._finish("score_blocked_missing_valid_predictions")
                return 1
            score = next(task for task in self.tasks if task["stage"] == "score")
            score_ok = self._execute_cpu(score)
            self._finish("complete" if score_ok else "score_failed")
            return 0 if score_ok else 1
        finally:
            try:
                fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_UN)
                self.lock_stream.close()
            except (AttributeError, OSError):
                pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"experiment": NAME, "tasks": task_plan(),
                          "owner_aware": True, "maximum_parallel_gpu_tasks": 3,
                          "external_preflight_wait": True, "score_cuda_hidden": True,
                          "dry_run": True}, indent=2), flush=True)
        return 0
    return StageQueue().run()


if __name__ == "__main__":
    raise SystemExit(main())
