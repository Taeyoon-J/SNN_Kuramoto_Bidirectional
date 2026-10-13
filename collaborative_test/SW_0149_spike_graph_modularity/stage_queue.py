"""Owner-aware two-arm SW0149 train, prediction, then CPU-score queue."""
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
except ImportError:  # Windows supports validation/dry-run, not Linux GPU leases.
    fcntl = None

# Keep the supervisor and all CPU validation free of CUDA contexts.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0126_history_event_binding import pilot_queue as gpu_guard
from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4 as owner_queue
from collaborative_test.SW_0149_spike_graph_modularity import evaluate, train

RUNTIME = HERE / "results_archive" / "stage_queue_20261012"
STATE = RUNTIME / "state.json"
QUEUE_LOCK = RUNTIME / "queue.lock"
GPU_LEASE_DIR = Path("/tmp/kevinswk_sw0113_gpu_leases")
MAX_MEMORY_MIB = owner_queue.MAX_MEMORY_MIB
STABLE_SECONDS = owner_queue.STABLE_SECONDS
SAMPLE_SECONDS = owner_queue.SAMPLE_SECONDS
OWNER_GUARD = owner_queue.OWNER_GUARD
OWNER_GUARD_SHA256 = owner_queue.OWNER_GUARD_SHA256
OWNER_QUEUE = ROOT / "collaborative_test/SW_0148_checkpointable_boundary_q/stage_queue_v4.py"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_plan():
    rows = []
    for arm in train.ARMS:
        rows.append({"task_id": f"train_{arm}", "stage": "train", "arm": arm,
                     "depends_on": [], "output": str(train.TRAIN / f"{arm}_seed1")})
    for arm in train.ARMS:
        rows.append({"task_id": f"predict_{arm}", "stage": "predict", "arm": arm,
                     "depends_on": [f"train_{arm}"],
                     "output": str(evaluate._prediction_dir(arm))})
    rows.append({"task_id": "score", "stage": "score", "arm": None,
                 "depends_on": [f"predict_{arm}" for arm in train.ARMS],
                 "output": str(evaluate.ARCHIVE / "evaluation_seed1.json")})
    return rows


def valid_result(task):
    try:
        if task["stage"] == "train":
            train.validate_training(task["arm"])
        elif task["stage"] == "predict":
            evaluate.validate_prediction(task["arm"])
        else:
            row = json.loads(Path(task["output"]).read_text(encoding="utf-8"))
            if (row.get("status") != "complete" or row.get("experiment") != train.NAME
                    or row.get("seed") != train.SEED
                    or row.get("ground_truth_opened_after_all_predictions_validated") is not True
                    or row.get("ground_truth_used_for_prediction") is not False
                    or row.get("ground_truth_used_for_scoring") is not True):
                return False
        return True
    except (OSError, ValueError, TypeError, KeyError, AssertionError, RuntimeError,
            EOFError, pickle.UnpicklingError):
        return False


def _argv(task):
    if task["stage"] == "train":
        return [sys.executable, "-m", "collaborative_test.SW_0149_spike_graph_modularity.train",
                "--arm", task["arm"], "--device", "cuda:0"]
    if task["stage"] == "predict":
        return [sys.executable, "-m", "collaborative_test.SW_0149_spike_graph_modularity.evaluate",
                "--stage", "predict", "--arm", task["arm"], "--device", "cuda:0"]
    return [sys.executable, "-m", "collaborative_test.SW_0149_spike_graph_modularity.evaluate",
            "--stage", "score"]


def _required_resume_gpu(task):
    if task["stage"] != "train":
        return None
    from collaborative_test.SW_0148_checkpointable_boundary_q.checkpoint import (
        inspect_committed_metadata, is_only_uncommitted_temp,
    )
    directory = train.WORK / f"{task['arm']}_seed1" / "checkpoints"
    boundary = directory / "boundary.ptx"
    if not boundary.exists():
        if directory.exists() and not is_only_uncommitted_temp(directory):
            raise ValueError("work directory without a committed boundary is not a known atomic temp")
        return None
    try:
        meta = inspect_committed_metadata(directory)
    except (OSError, ValueError, TypeError, KeyError, EOFError,
            pickle.UnpicklingError, RuntimeError) as exc:
        raise ValueError("checkpoint container failed integrity validation") from exc
    runtime = meta.get("runtime")
    visibility = runtime.get("cuda_visible_devices") if isinstance(runtime, dict) else None
    updates = meta.get("completed_updates")
    if (not isinstance(visibility, str) or visibility not in {"0", "1", "2", "3"}
            or runtime.get("device_type") != "cuda" or runtime.get("device_index") != 0
            or not isinstance(updates, int) or isinstance(updates, bool)
            or not 0 <= updates <= train.UPDATES):
        raise ValueError("checkpoint has no valid physical-GPU resume binding")
    return int(visibility)


def _safe_foreign_retry(task):
    output = Path(task["output"])
    if output.exists():
        return False
    if task["stage"] == "predict":
        return True
    if task["stage"] != "train":
        return False
    from collaborative_test.SW_0148_checkpointable_boundary_q.checkpoint import (
        inspect_committed_metadata, is_only_uncommitted_temp,
    )
    directory = train.WORK / f"{task['arm']}_seed1" / "checkpoints"
    boundary = directory / "boundary.ptx"
    if boundary.is_file():
        try:
            inspect_committed_metadata(directory)
            return True
        except (OSError, ValueError, TypeError, KeyError, EOFError,
                pickle.UnpicklingError, RuntimeError):
            return False
    return not directory.exists() or is_only_uncommitted_temp(directory)


class StageQueue:
    def __init__(self):
        if fcntl is None:
            raise RuntimeError("SW0149 stage queue requires Linux flock")
        # Fail before taking a GPU lease if the admitted source preflight is absent/stale.
        self.preflight = train.validate_preflight_admission()
        RUNTIME.mkdir(parents=True, exist_ok=True)
        self.lock_stream = QUEUE_LOCK.open("a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock_stream.close()
            raise RuntimeError("another SW0149 stage supervisor owns this queue") from exc
        if STATE.exists():
            self.lock_stream.close()
            raise FileExistsError(f"preserving existing SW0149 queue state: {STATE}")
        if sha(OWNER_GUARD) != OWNER_GUARD_SHA256:
            self.lock_stream.close()
            raise RuntimeError("reviewed GPU owner guard changed")
        self.stop_event = threading.Event()
        self.state_lock = threading.RLock()
        self.tasks = task_plan()
        self.state = {"experiment": train.NAME, "status": "running", "queue_version": 1,
                      "supervisor_pid": os.getpid(), "no_overwrite": True,
                      "owner_dispatcher_sha256": OWNER_GUARD_SHA256,
                      "owner_queue_sha256": sha(OWNER_QUEUE),
                      "implementation_sha256": train.implementation_sha(),
                      "preflight_sha256": sha(train.PREFLIGHT),
                      "tasks": {row["task_id"]: {"task": row, "status": "queued"}
                                for row in self.tasks}}
        self._save()

    def _save(self):
        with self.state_lock:
            tmp = STATE.with_name(STATE.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
            with tmp.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(self.state, stream, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, STATE)

    def _update(self, task_id, **fields):
        with self.state_lock:
            self.state["tasks"][task_id].update(fields)
            self._save()

    def _execute_gpu(self, task):
        task_id, output = task["task_id"], Path(task["output"])
        if valid_result(task):
            self._update(task_id, status="verified_existing")
            return True
        if output.exists():
            self._update(task_id, status="failed_existing_output_preserved")
            return False
        attempt = 1
        while not self.stop_event.is_set():
            try:
                preferred = _required_resume_gpu(task)
            except (OSError, ValueError, TypeError, KeyError, EOFError,
                    pickle.UnpicklingError, RuntimeError) as exc:
                self._update(task_id, status="failed_checkpoint_admission", error=repr(exc))
                return False
            try:
                gpu, uuid, memory, lease = owner_queue._acquire(self.stop_event, preferred)
            except BaseException as exc:
                self._update(task_id, status="failed_gpu_admission", error=repr(exc),
                             finished=time.time())
                return False
            proc, retry = None, False
            log = RUNTIME / f"{task_id}.attempt{attempt:02d}.log"
            try:
                if not owner_queue._stable_idle(gpu, uuid, self.stop_event,
                        lambda elapsed, used: self._update(
                            task_id, status="waiting_stable_gpu", gpu=gpu,
                            gpu_uuid=uuid, stable_seconds=elapsed, memory_used_mib=used)):
                    continue
                if output.exists() or log.exists():
                    self._update(task_id, status="failed_existing_output_preserved")
                    return False
                env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                if task["stage"] == "score":
                    env["CUDA_VISIBLE_DEVICES"] = ""
                else:
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                argv = _argv(task)
                with log.open("x", encoding="utf-8") as stream:
                    kwargs = dict(cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                  stdout=stream, stderr=subprocess.STDOUT,
                                  start_new_session=True)
                    if task["stage"] != "score":
                        kwargs["pass_fds"] = (lease.fileno(),)
                    proc = subprocess.Popen(argv, **kwargs)
                    self._update(task_id, status="running", child_pid=proc.pid,
                                 argv=argv, gpu=(gpu if task["stage"] != "score" else None),
                                 gpu_uuid=(uuid if task["stage"] != "score" else None),
                                 memory_used_before_mib=(memory if task["stage"] != "score" else None),
                                 cuda_hidden=(task["stage"] == "score"), log=str(log), attempt=attempt)
                    foreign = []
                    while proc.poll() is None:
                        time.sleep(5)
                        if proc.poll() is not None:
                            break
                        try:
                            current, _used, owners = gpu_guard._gpu_info(gpu)
                        except Exception:
                            continue
                        if current != uuid:
                            foreign = ["gpu_uuid_changed"]
                        else:
                            foreign = gpu_guard._foreign_owners(owners, gpu_guard._process_tree(proc.pid))
                        if foreign:
                            self._update(task_id, status="interrupted_foreign_gpu_owner",
                                         foreign_owner_pids=foreign)
                            owner_queue._stop_owned(proc)
                            retry = True
                            break
                code = proc.wait()
                good = code == 0 and valid_result(task)
                self._update(task_id, status="passed" if good else "failed",
                             returncode=code, artifact_valid=bool(good), finished=time.time())
                if good:
                    return True
                if not retry:
                    return False
            except BaseException as exc:
                if proc is not None and proc.poll() is None:
                    owner_queue._stop_owned(proc)
                self._update(task_id, status="failed", error=repr(exc), finished=time.time())
                return False
            finally:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                lease.close()
            if not _safe_foreign_retry(task):
                self._update(task_id, status="failed_foreign_retry_not_safe")
                return False
            self._update(task_id, status="queued_after_foreign_owner", attempt=attempt + 1)
            attempt += 1
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
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        argv = _argv(task)
        try:
            with log.open("x", encoding="utf-8") as stream:
                proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                        stdout=stream, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                self._update(task["task_id"], status="running", child_pid=proc.pid,
                             argv=argv, log=str(log), cuda_hidden=True)
                code = proc.wait()
            good = code == 0 and valid_result(task)
            self._update(task["task_id"], status="passed" if good else "failed",
                         returncode=code, artifact_valid=bool(good), finished=time.time())
            return good
        except BaseException as exc:
            self._update(task["task_id"], status="failed", error=repr(exc), finished=time.time())
            return False

    def run(self):
        try:
            for stage in ("train", "predict"):
                tasks = [task for task in self.tasks if task["stage"] == stage]
                with ThreadPoolExecutor(max_workers=2) as pool:
                    outcomes = list(pool.map(self._execute_gpu, tasks))
                if not all(outcomes):
                    self.state["status"] = f"{stage}_failed"
                    self._save()
                    return 1
            prediction_tasks = [task for task in self.tasks if task["stage"] == "predict"]
            if not all(valid_result(task) for task in prediction_tasks):
                self.state["status"] = "score_blocked_invalid_predictions"
                self._save()
                return 1
            score = next(task for task in self.tasks if task["stage"] == "score")
            result = self._execute_cpu(score)
            self.state["status"] = "complete" if result else "score_failed"
            self._save()
            return 0 if result else 1
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
        print(json.dumps({"experiment": train.NAME, "tasks": task_plan(),
                          "owner_aware": True, "maximum_parallel_gpu_tasks": 2,
                          "preflight_admission_required": True,
                          "score_cuda_hidden": True, "dry_run": True}, indent=2), flush=True)
        return 0
    return StageQueue().run()


if __name__ == "__main__":
    raise SystemExit(main())
