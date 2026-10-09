"""Owner-aware, no-retry queue for the conditional two-arm seed-1 pilot."""
from __future__ import annotations

import concurrent.futures
import json
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time

try:
    import fcntl
except ImportError:  # Windows can run CPU tests but cannot launch this queue.
    fcntl = None

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0126_history_event_binding import pilot_run, pilot_evaluate
from collaborative_test.SW_0126_history_event_binding import screen_queue as resource_guard

GPU_LEASE_DIR = pathlib.Path("/tmp/kevinswk_sw0113_gpu_leases")
RUNTIME_DIR = pathlib.Path("/tmp/kevinswk_sw0126_pilot_queue")
QUEUE_LOCK = RUNTIME_DIR / "queue.lock"
STATE_PATH = RUNTIME_DIR / "state.json"
MAX_GPU_MEMORY_MIB = 512
PYTHON = "/Data0/kevinswk/envs/snn/bin/python"
EVALUATION_OUTPUT = HERE / "results_archive" / "pilot_seed1_pair_evaluation_20261009"


def task_plan():
    tasks = []
    for arm in pilot_run.ARM_NAMES:
        tasks.append({"task_id": f"sw0126_pilot_seed1_{arm}_preflight", "stage": "preflight",
                      "arm": arm, "output": str(pilot_run._preflight_path(arm)),
                      "depends_on": [f"sw0126_screen_seed{s}" for s in (0, 1, 2)]})
    for arm in pilot_run.ARM_NAMES:
        tasks.append({"task_id": f"sw0126_pilot_seed1_{arm}_train", "stage": "train",
                      "arm": arm, "output": str(pilot_run.training_dir(arm)),
                      "depends_on": [f"sw0126_pilot_seed1_{arm}_preflight"]})
    tasks.append({"task_id": "sw0126_pilot_seed1_pair_evaluate", "stage": "evaluate",
                  "output": str(EVALUATION_OUTPUT),
                  "depends_on": [f"sw0126_pilot_seed1_{arm}_train" for arm in pilot_run.ARM_NAMES]})
    return tasks


def gpu_is_exclusive(memory_mib, owner_pids):
    return int(memory_mib) <= MAX_GPU_MEMORY_MIB and not owner_pids


def _gpu_info(index):
    uuid = subprocess.check_output(
        ["nvidia-smi", "-i", str(index), "--query-gpu=uuid", "--format=csv,noheader"],
        text=True).strip()
    memory = int(subprocess.check_output(
        ["nvidia-smi", "-i", str(index), "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0])
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    owners = []
    for line in apps.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 2 and fields[0] == uuid:
            owners.append(int(fields[1]))
    return uuid, memory, sorted(set(owners))


def _process_tree(root_pid):
    children = {}
    for line in subprocess.check_output(["ps", "-eo", "pid=,ppid="], text=True).splitlines():
        fields = line.split()
        if len(fields) == 2:
            pid, parent = map(int, fields)
            children.setdefault(parent, []).append(pid)
    result, stack = {int(root_pid)}, [int(root_pid)]
    while stack:
        for child in children.get(stack.pop(), ()):
            if child not in result:
                result.add(child)
                stack.append(child)
    return result


def _foreign_owners(owners, owned_tree):
    return resource_guard._foreign_owners(owners, owned_tree)


def _valid_result(task):
    stage, arm = task["stage"], task.get("arm")
    if stage == "preflight":
        try:
            return bool(pilot_run._validate_preflight(arm))
        except (OSError, ValueError, TypeError, KeyError, AssertionError):
            return False
    if stage == "train":
        return pilot_run.valid_training(task["output"], arm)
    path = pathlib.Path(task["output"]) / "evaluation.json"
    manifest = pathlib.Path(task["output"]) / "prediction_manifest.json"
    if not path.is_file() or not manifest.is_file():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        prediction = json.loads(manifest.read_text(encoding="utf-8"))
        if (report.get("status") != "complete" or report.get("experiment") != "SW0126"
                or report.get("seed") != 1 or report.get("ids") != [1320, 1639]
                or report.get("count") != 320
                or report.get("prediction_manifest_sha256") != pilot_run.sha(manifest)
                or prediction.get("status") != "predictions_complete"
                or prediction.get("ground_truth_used_for_prediction") is not False):
            return False
        blob = pathlib.Path(task["output"]) / "paired_predictions.pt"
        if (not blob.is_file() or pilot_run.sha(blob) != prediction.get("prediction_sha256")
                or report.get("prediction_blob_sha256") != pilot_run.sha(blob)):
            return False
        for name in ("history_event", "gate_only"):
            score = report["scores"][name]
            for metric in pilot_evaluate.METRICS:
                values = score[metric]["per_image"]
                if (score[metric]["valid_count"] != 320 or len(values) != 320
                        or not all(__import__("math").isfinite(float(x)) for x in values)):
                    return False
        return True
    except (OSError, ValueError, TypeError, KeyError):
        return False


def _output_exists(task):
    p = pathlib.Path(task["output"])
    return p.exists()


class PilotQueue:
    def __init__(self):
        if fcntl is None:
            raise RuntimeError("SW0126 pilot queue requires Linux flock")
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        self.queue_lock = QUEUE_LOCK.open("a+")
        try:
            fcntl.flock(self.queue_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.queue_lock.close()
            raise RuntimeError("another SW0126 pilot queue owns the lock") from exc
        if STATE_PATH.exists():
            self.queue_lock.close()
            raise FileExistsError(f"preserving existing pilot queue state {STATE_PATH}")
        self.lock = threading.RLock()
        self.tasks = task_plan()
        self.state = {"experiment": "SW0126", "status": "running",
                      "supervisor_pid": os.getpid(), "no_automatic_retry": True,
                      "tasks": {t["task_id"]: {"task": t, "status": "queued"} for t in self.tasks}}
        self._save()

    def _save(self):
        with self.lock:
            tmp = STATE_PATH.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.state, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            os.replace(tmp, STATE_PATH)

    def _update(self, task_id, **fields):
        with self.lock:
            self.state["tasks"][task_id].update(fields)
            self._save()

    def _acquire(self, task):
        while True:
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
                    uuid, memory, owners = _gpu_info(gpu)
                    if gpu_is_exclusive(memory, owners):
                        return gpu, uuid, memory, lease
                except BaseException:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                    lease.close()
                    raise
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                lease.close()
            self._update(task["task_id"], status="waiting_for_exclusive_gpu")
            time.sleep(5)

    @staticmethod
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

    def _argv(self, task, gpu):
        if task["stage"] in ("preflight", "train"):
            return [PYTHON, "-m", "collaborative_test.SW_0126_history_event_binding.pilot_run",
                    task["stage"], "--arm", task["arm"], "--device", "cuda:0",
                    "--output", task["output"]]
        return [PYTHON, "-m", "collaborative_test.SW_0126_history_event_binding.pilot_evaluate",
                "--device", "cuda:0", "--output", task["output"]]

    def _execute(self, task):
        task_id = task["task_id"]
        if _valid_result(task):
            self._update(task_id, status="verified_existing")
            return True
        if _output_exists(task):
            self._update(task_id, status="failed_existing_output_preserved")
            return False
        gpu, uuid, memory, lease = self._acquire(task)
        proc = None
        log_path = RUNTIME_DIR / f"{task_id}.log"
        try:
            if log_path.exists():
                raise FileExistsError(f"preserving existing task log {log_path}")
            triton = pathlib.Path(f"/tmp/kevinswk_sw0126_pilot_triton_{task_id}")
            triton.mkdir(mode=0o700, parents=True, exist_ok=False)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1", TRITON_CACHE_DIR=str(triton))
            argv = self._argv(task, gpu)
            with log_path.open("x", encoding="utf-8") as stream:
                proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                    pass_fds=(lease.fileno(),))
                self._update(task_id, status="running", child_pid=proc.pid, gpu=gpu,
                    gpu_uuid=uuid, memory_used_before_mib=memory, argv=argv,
                    log=str(log_path), triton_cache_dir=str(triton), inherited_lease_fd=True)
                foreign = []
                while proc.poll() is None:
                    time.sleep(5)
                    if proc.poll() is not None:
                        break
                    _uuid, _mem, owners = _gpu_info(gpu)
                    foreign = _foreign_owners(owners, _process_tree(proc.pid))
                    if foreign:
                        self._update(task_id, status="interrupted_foreign_gpu_owner",
                                     foreign_owner_pids=foreign)
                        self._stop_owned(proc)
                        break
                rc = proc.wait()
                accepted = rc == 0 and not foreign and _valid_result(task)
                self._update(task_id, status="passed" if accepted else "failed",
                    returncode=rc, artifact_valid=_valid_result(task), finished=time.time())
                return accepted
        except BaseException as exc:
            if proc is not None and proc.poll() is None:
                self._stop_owned(proc)
            self._update(task_id, status="failed", error=repr(exc), finished=time.time())
            return False
        finally:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
            lease.close()

    def _run_phase(self, tasks):
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            return all(pool.map(self._execute, tasks))

    def run(self):
        try:
            screen_tasks = resource_guard.task_plan()
            if not all(resource_guard.valid_result(t) for t in screen_tasks):
                self.state.update(status="blocked_screen_prerequisite_failed", finished=time.time())
                self._save()
                return
            preflight = [t for t in self.tasks if t["stage"] == "preflight"]
            if not self._run_phase(preflight):
                self.state.update(status="preflight_failed_training_not_started", finished=time.time())
                self._save()
                return
            training = [t for t in self.tasks if t["stage"] == "train"]
            if not self._run_phase(training):
                self.state.update(status="training_failed_evaluation_not_started", finished=time.time())
                self._save()
                return
            evaluation = next(t for t in self.tasks if t["stage"] == "evaluate")
            if not self._execute(evaluation):
                self.state.update(status="evaluation_failed", finished=time.time())
                self._save()
                return
            self.state.update(status="complete", finished=time.time())
            self._save()
        finally:
            fcntl.flock(self.queue_lock.fileno(), fcntl.LOCK_UN)
            self.queue_lock.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({"experiment": "SW0126", "tasks": task_plan()}, indent=2))
    else:
        PilotQueue().run()
