"""Owner-aware no-retry dispatcher for the nine SW0125 tasks."""
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
except ImportError:  # The dispatcher runs only on the Linux research server.
    fcntl = None

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0125_late_rollout_credit import coordinator
from collaborative_test.SW_0123_adaptive_temporal_assignment import dispatcher as owner

ARCHIVE = HERE / "results_archive"
STATE = ARCHIVE / "dispatcher_state_20261009.json"
QUEUE_LOCK = ARCHIVE / "dispatcher.lock"
GPU_LEASE_DIR = pathlib.Path("/tmp/kevinswk_sw0113_gpu_leases")
GPU_MEMORY_LIMIT_MIB = 512
MAX_PARALLEL = 3


def gpu_is_exclusive_candidate(memory_mib, compute_owners):
    return not compute_owners and int(memory_mib) <= GPU_MEMORY_LIMIT_MIB


def atomic_json(path, value):
    path = pathlib.Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _protected_path(task):
    artifact = coordinator.artifact_path(task)
    if task["stage"] == "preflight":
        return artifact
    if task["stage"] == "train":
        return artifact.parent
    return artifact.parent


class Dispatcher:
    def __init__(self):
        if fcntl is None:
            raise RuntimeError("SW0125 dispatcher requires Linux fcntl global GPU leases")
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        self.queue_lock = QUEUE_LOCK.open("a+")
        try:
            fcntl.flock(self.queue_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.queue_lock.close()
            raise RuntimeError("another SW0125 dispatcher owns its queue lock") from exc
        if STATE.exists():
            self.queue_lock.close()
            raise FileExistsError(f"preserving prior SW0125 dispatcher state: {STATE}")
        self.lock = threading.Lock()
        self.tasks = coordinator.task_plan()
        self.state = {
            "experiment": "SW0125", "status": "running", "supervisor_pid": os.getpid(),
            "started": time.time(), "task_count": len(self.tasks),
            "implementation_fingerprint": coordinator.run.implementation_fingerprint(),
            "evaluation_fingerprint": coordinator.evaluate.evaluation_fingerprint(),
            "tasks": {task["task_id"]: {"task": task, "status": "queued"} for task in self.tasks},
        }
        self._save()

    def _save(self):
        with self.lock:
            atomic_json(STATE, self.state)

    def _update(self, task_id, **changes):
        with self.lock:
            self.state["tasks"][task_id].update(changes)
            atomic_json(STATE, self.state)

    def _reserve_gpu(self, task_id):
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
                    gpu_uuid, memory, owners = owner._nvidia_gpu_info(gpu)
                    if gpu_is_exclusive_candidate(memory, owners):
                        self._update(task_id, status="reserved", gpu=gpu,
                                     gpu_uuid=gpu_uuid, used_before_mib=memory,
                                     lease_path=str(lease_path), reservation_pid=os.getpid(),
                                     reserved=time.time())
                        return gpu, lease, gpu_uuid
                except BaseException:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                    lease.close()
                    raise
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                lease.close()
            self._update(task_id, status="waiting_for_exclusive_gpu", last_wait=time.time())
            time.sleep(5)

    @staticmethod
    def _terminate_own_group(proc):
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()

    def _execute(self, task):
        task_id = task["task_id"]
        if coordinator.valid_result(task):
            self._update(task_id, status="reused_verified", verified_at=time.time())
            return "reused_verified"
        protected = _protected_path(task)
        if protected.exists():
            self._update(task_id, status="failed_existing_artifact",
                         error=f"preserving invalid/partial artifact: {protected}", finished=time.time())
            return "failed_existing_artifact"

        gpu, lease, gpu_uuid = self._reserve_gpu(task_id)
        log_path = ARCHIVE / f"{task_id}_20261009.log"
        proc = None
        try:
            if log_path.exists():
                raise FileExistsError(f"preserving prior task log: {log_path}")
            triton_dir = pathlib.Path(f"/tmp/kevinswk_sw0125_triton_{task_id}_{int(time.time()*1000)}")
            triton_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
            command = coordinator.command(task, device="cuda:0")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", TRITON_CACHE_DIR=str(triton_dir))
            self._update(task_id, status="launching", gpu=gpu, gpu_uuid=gpu_uuid,
                         argv=command, log_path=str(log_path), triton_cache=str(triton_dir),
                         launcher_pid=os.getpid(), launched=time.time())
            with log_path.open("x", encoding="utf-8") as log:
                proc = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True,
                                        pass_fds=(lease.fileno(),))
                self._update(task_id, status="running", child_pid=proc.pid, started=time.time())
                while proc.poll() is None:
                    time.sleep(5)
                    if proc.poll() is not None:
                        break
                    _uuid, _memory, owners = owner._nvidia_gpu_info(gpu)
                    descendants = owner._process_tree(proc.pid)
                    foreign = owner.foreign_owner_pids(owners, descendants)
                    if foreign:
                        self._terminate_own_group(proc)
                        self._update(task_id, status="interrupted_foreign_gpu_owner",
                                     foreign_owner_pids=foreign, finished=time.time())
                        return "interrupted_foreign_gpu_owner"
                rc = int(proc.returncode)
            result_valid = rc == 0 and coordinator.valid_result(task)
            status = "passed" if result_valid else "failed"
            self._update(task_id, status=status, returncode=rc,
                         artifact_valid=bool(result_valid), finished=time.time())
            return status
        except BaseException as exc:
            if proc is not None:
                self._terminate_own_group(proc)
            self._update(task_id, status="failed", error=repr(exc), finished=time.time())
            return "failed"
        finally:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
            lease.close()

    def _run_phase(self, tasks):
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
            futures = {executor.submit(self._execute, task): task for task in tasks}
            return {futures[future]["task_id"]: future.result() for future in futures}

    def run(self):
        by_stage = {stage: [task for task in self.tasks if task["stage"] == stage]
                    for stage in ("preflight", "train", "evaluate")}
        pre_status = self._run_phase(by_stage["preflight"])
        if any(value not in ("passed", "reused_verified") for value in pre_status.values()):
            for task in by_stage["train"] + by_stage["evaluate"]:
                self._update(task["task_id"], status="blocked_preflight_failure")
            self._finish("preflight_failed")
            return
        train_status = self._run_phase(by_stage["train"])
        ready_eval = []
        for task in by_stage["evaluate"]:
            train_id = task["depends_on"][0]
            if train_status.get(train_id) in ("passed", "reused_verified"):
                ready_eval.append(task)
            else:
                self._update(task["task_id"], status="blocked_training_failure")
        eval_status = self._run_phase(ready_eval) if ready_eval else {}
        done = all(self.state["tasks"][task["task_id"]]["status"] in
                   ("passed", "reused_verified") for task in by_stage["evaluate"])
        self._finish("complete" if done else "incomplete")

    def _finish(self, status):
        self._save()
        with self.lock:
            self.state.update(status=status, finished=time.time())
            atomic_json(STATE, self.state)


def main():
    if "--dry-run" in sys.argv:
        print(json.dumps({"experiment": "SW0125", "task_count": len(coordinator.task_plan()),
                          "tasks": coordinator.task_plan()}, indent=2))
        return
    Dispatcher().run()


if __name__ == "__main__":
    main()
