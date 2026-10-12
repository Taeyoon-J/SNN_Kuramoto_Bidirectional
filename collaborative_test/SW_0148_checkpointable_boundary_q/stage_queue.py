"""Owner-aware SW0148 train/resume, predict, then GT-last score supervisor."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
try:
    import fcntl
except ImportError:  # Windows can validate plans/tests but cannot own Linux leases.
    fcntl = None
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0126_history_event_binding import pilot_queue as gpu_guard
from collaborative_test.SW_0146_native32_boundary_q_pilot import run as sw146
from collaborative_test.SW_0148_checkpointable_boundary_q import run as experiment
from collaborative_test.SW_0148_checkpointable_boundary_q import train as trainer

RUNTIME = HERE / "results_archive" / "queue_20261012"
STATE = RUNTIME / "state.json"
QUEUE_LOCK = RUNTIME / "queue.lock"
GPU_LEASE_DIR = Path("/tmp/kevinswk_sw0113_gpu_leases")
MAX_MEMORY_MIB = 512
STABLE_SECONDS = 60.0
SAMPLE_SECONDS = 5.0
OWNER_GUARD = ROOT / "collaborative_test/SW_0126_history_event_binding/pilot_queue.py"
OWNER_GUARD_SHA256 = "c59cdc39af20c824930d1926c8cec1418ee818577b8cc4240ad6eaae836764f6"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_plan():
    tasks = []
    for arm in sw146.ARMS:
        tasks.append({"task_id": f"train_{arm}", "stage": "train", "arm": arm,
                      "depends_on": [], "output": str(trainer.TRAIN / f"{arm}_seed1")})
    for arm in sw146.ARMS:
        tasks.append({"task_id": f"predict_{arm}", "stage": "predict", "arm": arm,
                      "depends_on": [f"train_{arm}"],
                      "output": str(experiment._prediction_dir(arm))})
    tasks.append({"task_id": "score", "stage": "score", "arm": None,
                  "depends_on": [f"predict_{arm}" for arm in sw146.ARMS],
                  "output": str(experiment.ARCHIVE / "evaluation_seed1.json")})
    return tasks


def valid_result(task):
    try:
        if task["stage"] == "train":
            trainer.validate_training(task["arm"])
        elif task["stage"] == "predict":
            experiment.validate_prediction(task["arm"])
        else:
            result = json.loads(Path(task["output"]).read_text(encoding="utf-8"))
            if (result.get("status") != "complete" or result.get("experiment") != trainer.NAME
                    or result.get("seed") != sw146.SEED
                    or result.get("ground_truth_opened_after_all_predictions_validated") is not True
                    or result.get("ground_truth_used_for_prediction") is not False):
                return False
        return True
    except (OSError, ValueError, TypeError, KeyError, AssertionError):
        return False


def _acquire(stop_event):
    while not stop_event.is_set():
        for gpu in range(4):
            lease_path = GPU_LEASE_DIR / f"gpu{gpu}.lock"
            lease_path.parent.mkdir(parents=True, exist_ok=True)
            lease = lease_path.open("a+")
            try:
                fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lease.close()
                continue
            try:
                uuid, memory, owners = gpu_guard._gpu_info(gpu)
            except BaseException:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                lease.close()
                continue
            if not owners and int(memory) <= MAX_MEMORY_MIB:
                return gpu, uuid, int(memory), lease
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
            lease.close()
        time.sleep(SAMPLE_SECONDS)
    raise RuntimeError("queue stopped while waiting for an exclusive GPU")


def _stable_idle(gpu, expected_uuid, stop_event, on_sample):
    started = None
    while not stop_event.is_set():
        try:
            uuid, memory, owners = gpu_guard._gpu_info(gpu)
        except Exception:
            return False
        if uuid != expected_uuid or owners or int(memory) > MAX_MEMORY_MIB:
            return False
        now = time.monotonic()
        if started is None:
            started = now
        elapsed = now - started
        on_sample(elapsed, int(memory))
        if elapsed >= STABLE_SECONDS:
            try:
                uuid, memory, owners = gpu_guard._gpu_info(gpu)
            except Exception:
                return False
            return uuid == expected_uuid and not owners and int(memory) <= MAX_MEMORY_MIB
        time.sleep(min(SAMPLE_SECONDS, STABLE_SECONDS - elapsed))
    return False


def safe_foreign_retry(task, output):
    """Permit only resumable training or absent-output prediction retries."""
    output = Path(output)
    if output.exists():
        return False
    if task["stage"] == "predict":
        return True
    if task["stage"] != "train":
        return False
    from collaborative_test.SW_0148_checkpointable_boundary_q.checkpoint import (
        inspect_committed_metadata, is_only_uncommitted_temp,
    )
    checkpoint_dir = trainer.WORK / f"{task['arm']}_seed1" / "checkpoints"
    boundary = checkpoint_dir / "boundary.ptx"
    if boundary.is_file():
        try:
            inspect_committed_metadata(checkpoint_dir)
            return True
        except (OSError, ValueError, TypeError, KeyError):
            return False
    return not checkpoint_dir.exists() or is_only_uncommitted_temp(checkpoint_dir)


def _argv(task):
    return [sys.executable, "-m", "collaborative_test.SW_0148_checkpointable_boundary_q.run",
            "--stage", task["stage"], *( ["--arm", task["arm"]] if task.get("arm") else [] ),
            *( ["--device", "cuda:0"] if task["stage"] != "score" else [] )]


def _stop_owned(proc):
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


class Queue:
    def __init__(self):
        if fcntl is None:
            raise RuntimeError("SW0148 queue requires Linux flock")
        if fcntl is None:
            raise RuntimeError("SW0148 queue requires Linux flock")
        RUNTIME.mkdir(parents=True, exist_ok=True)
        self.lock_stream = QUEUE_LOCK.open("a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock_stream.close()
            raise RuntimeError("another SW0148 supervisor owns the queue") from exc
        if STATE.exists():
            self.lock_stream.close()
            raise FileExistsError(f"preserving previous SW0148 queue state: {STATE}")
        if _sha(OWNER_GUARD) != OWNER_GUARD_SHA256:
            self.lock_stream.close()
            raise RuntimeError("reviewed SW0148 GPU owner guard changed")
        self.stop_event = threading.Event()
        self.state_lock = threading.RLock()
        self.tasks = task_plan()
        self.state = {"experiment": trainer.NAME, "status": "running",
                      "supervisor_pid": os.getpid(), "no_overwrite": True,
                      "owner_dispatcher_sha256": OWNER_GUARD_SHA256,
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
        task_id = task["task_id"]
        if valid_result(task):
            self._update(task_id, status="verified_existing")
            return True
        output = Path(task["output"])
        if output.exists():
            self._update(task_id, status="failed_existing_output_preserved")
            return False
        proc = None
        attempt = 1
        while not self.stop_event.is_set():
            if valid_result(task):
                self._update(task_id, status="verified_existing")
                return True
            if output.exists():
                self._update(task_id, status="failed_existing_output_preserved")
                return False
            gpu, gpu_uuid, memory, lease = _acquire(self.stop_event)
            retry_foreign = False
            log = RUNTIME / f"{task_id}.attempt{attempt:02d}.log"
            try:
                if not _stable_idle(gpu, gpu_uuid, self.stop_event,
                                    lambda elapsed, used: self._update(
                                        task_id, status="waiting_stable_gpu", gpu=gpu,
                                        gpu_uuid=gpu_uuid, stable_seconds=elapsed,
                                        memory_used_mib=used)):
                    continue
                if output.exists() or log.exists():
                    self._update(task_id, status="failed_existing_output_preserved")
                    return False
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                           MKL_NUM_THREADS="1")
                argv = _argv(task)
                with log.open("x", encoding="utf-8") as stream:
                    proc = subprocess.Popen(argv, cwd=ROOT, env=env,
                        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                        start_new_session=True, pass_fds=(lease.fileno(),))
                    self._update(task_id, status="running", child_pid=proc.pid,
                                 gpu=gpu, gpu_uuid=gpu_uuid, memory_used_before_mib=memory,
                                 argv=argv, log=str(log), inherited_lease_fd=True)
                    foreign = []
                    while proc.poll() is None:
                        time.sleep(5)
                        if proc.poll() is not None:
                            break
                        try:
                            current_uuid, _used, owners = gpu_guard._gpu_info(gpu)
                        except Exception:
                            continue
                        if current_uuid != gpu_uuid:
                            foreign = ["gpu_uuid_changed"]
                        else:
                            foreign = gpu_guard._foreign_owners(
                                owners, gpu_guard._process_tree(proc.pid))
                        if foreign:
                            self._update(task_id, status="interrupted_foreign_gpu_owner",
                                         foreign_owner_pids=foreign)
                            _stop_owned(proc)
                            retry_foreign = True
                            break
                    code = proc.wait()
                    good = code == 0 and valid_result(task)
                    self._update(task_id, status="passed" if good else "failed",
                                 returncode=code, artifact_valid=bool(good),
                                 finished=time.time())
                    if good:
                        return True
                    if not retry_foreign:
                        return False
            except BaseException as exc:
                if proc is not None and proc.poll() is None:
                    _stop_owned(proc)
                self._update(task_id, status="failed", error=repr(exc), finished=time.time())
                return False
            finally:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                lease.close()
            if retry_foreign:
                # train.py resumes only from a self-verifying committed logical boundary.
                if not safe_foreign_retry(task, output):
                    return False
                # This is an explicit foreign-owner recovery, never a numerical retry.
                self._update(task_id, status="queued_after_foreign_owner",
                             prior_foreign_owner_pids=self.state["tasks"][task_id].get("foreign_owner_pids", []),
                             retry_log=str(RUNTIME / f"{task_id}.attempt{attempt:02d}.log"))
                attempt += 1
                continue
        self._update(task_id, status="cancelled")
        return False

    def _execute_cpu(self, task):
        if valid_result(task):
            self._update(task["task_id"], status="verified_existing")
            return True
        if Path(task["output"]).exists():
            self._update(task["task_id"], status="failed_existing_output_preserved")
            return False
        log = RUNTIME / f"{task['task_id']}.log"
        if log.exists():
            self._update(task["task_id"], status="failed_existing_log_preserved")
            return False
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1")
        argv = _argv(task)
        try:
            with log.open("x", encoding="utf-8") as stream:
                proc = subprocess.Popen(argv, cwd=ROOT, env=env,
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
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
            train_tasks = [task for task in self.tasks if task["stage"] == "train"]
            with ThreadPoolExecutor(max_workers=2) as pool:
                outcomes = list(pool.map(self._execute_gpu, train_tasks))
            if not all(outcomes):
                self.state["status"] = "training_failed_or_interrupted"
                self._save()
                return 1
            prediction_tasks = [task for task in self.tasks if task["stage"] == "predict"]
            with ThreadPoolExecutor(max_workers=2) as pool:
                outcomes = list(pool.map(self._execute_gpu, prediction_tasks))
            if not all(outcomes):
                self.state["status"] = "prediction_failed"
                self._save()
                return 1
            score = next(task for task in self.tasks if task["stage"] == "score")
            if not all(valid_result(task) for task in prediction_tasks):
                self.state["status"] = "score_blocked_invalid_predictions"
                self._save()
                return 1
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
        print(json.dumps({"experiment": trainer.NAME, "tasks": task_plan(),
                          "stable_idle_seconds": STABLE_SECONDS,
                          "owner_aware": True, "training_resume": "committed logical update only",
                          "dry_run": True}, indent=2), flush=True)
        return 0
    return Queue().run()


if __name__ == "__main__":
    raise SystemExit(main())
