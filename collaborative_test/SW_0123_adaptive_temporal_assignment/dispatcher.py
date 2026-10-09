"""Owner-aware, no-retry dispatcher for the registered 27 SW0123 tasks.

This is a separate supervisor; it does not alter the scientific runner's
fingerprint. A single queue process holds each shared GPU lease through its
child's lifetime, and the child inherits the descriptor so an orphan cannot
silently release the lease while still using CUDA.
"""
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
except ImportError:  # local Windows tests do not execute the server dispatcher
    fcntl = None

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0123_adaptive_temporal_assignment import coordinator as adapter

MAX_GPU_MEMORY_MIB = 512
GPU_LEASE_DIR = pathlib.Path("/tmp/kevinswk_sw0113_gpu_leases")
ARCHIVE = HERE / "results_archive"
STATE_PATH = ARCHIVE / "dispatcher_state_20261009.json"
QUEUE_LOCK_PATH = ARCHIVE / "dispatcher.lock"
PYTHON = "/Data0/kevinswk/envs/snn/bin/python"


def foreign_owner_pids(owner_pids, owned_process_pids):
    """Return non-vanished compute PIDs outside this worker's process tree.

    Only a confirmed missing /proc entry proves a PID vanished. Permission or
    other inspection errors remain conservative and are treated as foreign.
    """
    owners = {int(pid) for pid in owner_pids if int(pid) > 0}
    owned = {int(pid) for pid in owned_process_pids if int(pid) > 0}
    foreign = []
    for pid in sorted(owners - owned):
        if not _pid_is_confirmed_gone(pid):
            foreign.append(pid)
    return foreign


def _pid_is_confirmed_gone(pid):
    """Return true only for a positively missing Linux proc entry."""
    if os.name == "nt":
        return False
    try:
        pathlib.Path(f"/proc/{pid}").stat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return False


def adaptive_preflight_all_passed(task_rows):
    """Whether all registered adaptive seed preflights passed their guards."""
    adaptive = [row for row in task_rows if row["task"]["stage"] == "preflight"
                and row["task"]["arm"] == "adaptive_full"]
    return len(adaptive) == 3 and all(
        row["status"] in ("passed", "reused_verified") for row in adaptive)


def _nvidia_gpu_info(gpu):
    uuid = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"],
        text=True).strip()
    memory = int(subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0])
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True)
    owners = []
    for line in apps.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 2 and fields[0] == uuid:
            try:
                owners.append(int(fields[1]))
            except ValueError:
                raise RuntimeError(f"unparseable nvidia-smi compute PID row: {line!r}")
    return uuid, memory, sorted(set(owners))


def _process_tree(root_pid):
    """Return root PID and descendants using Linux ps parent IDs."""
    rows = subprocess.check_output(["ps", "-eo", "pid=,ppid="], text=True)
    children = {}
    for line in rows.splitlines():
        fields = line.split()
        if len(fields) == 2:
            pid, parent = map(int, fields)
            children.setdefault(parent, []).append(pid)
    result, stack = {int(root_pid)}, [int(root_pid)]
    while stack:
        parent = stack.pop()
        for child in children.get(parent, ()):
            if child not in result:
                result.add(child)
                stack.append(child)
    return result


def _protected_attempt_paths(task):
    path = adapter.artifact_path(task)
    stage = task["stage"]
    if stage == "preflight":
        return (path, adapter.run._warmup_path(task["seed"], task["arm"]))
    if stage == "train":
        return (path.parent,)
    if stage == "evaluate":
        return (path.parent,)
    raise ValueError(stage)


def _atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


class Dispatcher:
    def __init__(self):
        if fcntl is None:
            raise RuntimeError("the owner-aware SW0123 dispatcher requires Linux fcntl locks")
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        self.queue_lock = QUEUE_LOCK_PATH.open("a+")
        try:
            fcntl.flock(self.queue_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.queue_lock.close()
            raise RuntimeError("another SW0123 dispatcher owns the queue lock") from exc
        if STATE_PATH.exists():
            self.queue_lock.close()
            raise FileExistsError(f"preserve existing dispatcher state: {STATE_PATH}")
        self.lock = threading.Lock()
        self.tasks = adapter.task_plan()
        self.state = {
            "experiment": "SW0123", "status": "running", "supervisor_pid": os.getpid(),
            "started": time.time(), "task_count": len(self.tasks),
            "runner_fingerprint": adapter.run.implementation_fingerprint(),
            "evaluation_fingerprint": adapter.evaluate.evaluation_fingerprint(),
            "tasks": {task["task_id"]: {"task": task, "status": "queued"} for task in self.tasks},
        }
        self._save()

    def _save(self):
        with self.lock:
            _atomic_json(STATE_PATH, self.state)

    def _update(self, task_id, **fields):
        with self.lock:
            self.state["tasks"][task_id].update(fields)
            _atomic_json(STATE_PATH, self.state)

    def _set_status(self, status, **fields):
        with self.lock:
            self.state.update(status=status, **fields)
            _atomic_json(STATE_PATH, self.state)

    def _acquire_gpu(self, task):
        task_id = task["task_id"]
        while True:
            for gpu in range(4):
                path = GPU_LEASE_DIR / f"gpu{gpu}.lock"
                path.parent.mkdir(parents=True, exist_ok=True)
                stream = path.open("a+")
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    stream.close()
                    continue
                try:
                    uuid, memory, owners = _nvidia_gpu_info(gpu)
                    if not owners and memory <= MAX_GPU_MEMORY_MIB:
                        self._update(task_id, status="reserved", gpu=gpu, gpu_uuid=uuid,
                                     gpu_used_before_mib=memory, gpu_lease_path=str(path),
                                     reservation_pid=os.getpid(), reserved=time.time())
                        return gpu, stream, memory, uuid
                except BaseException:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                    stream.close()
                    raise
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                stream.close()
            self._update(task_id, status="queued_waiting_for_exclusive_gpu",
                         last_wait=time.time())
            time.sleep(5)

    def _terminate_owned_group(self, proc):
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            # The process group was created exclusively for this child. It is
            # safe to finish only that owned group, never a GPU owner by PID.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()

    def _execute(self, task):
        task_id = task["task_id"]
        if adapter.valid_result(task):
            self._update(task_id, status="reused_verified", verified_at=time.time())
            return "reused_verified"
        protected = [path for path in _protected_attempt_paths(task) if path.exists()]
        if protected:
            message = "preserving existing invalid/partial stage artifact(s): " + ", ".join(map(str, protected))
            self._update(task_id, status="failed_existing_artifact", error=message, finished=time.time())
            return "failed_existing_artifact"

        gpu, lease, memory, uuid = self._acquire_gpu(task)
        log_path = ARCHIVE / f"{task_id}_20261009.log"
        proc = None
        try:
            if log_path.exists():
                raise FileExistsError(f"preserve existing log: {log_path}")
            triton_cache = pathlib.Path(
                f"/tmp/kevinswk_sw0123_triton_{task_id}_{int(time.time() * 1000)}")
            triton_cache.mkdir(parents=True, mode=0o700, exist_ok=False)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", TRITON_CACHE_DIR=str(triton_cache))
            cmd = adapter.command(task, "cuda:0")
            cmd[0] = PYTHON
            with log_path.open("x", encoding="utf-8") as log:
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True, pass_fds=(lease.fileno(),))
                self._update(task_id, status="running", stage=task["stage"], seed=task["seed"],
                             arm=task["arm"], child_pid=proc.pid, gpu=gpu, gpu_uuid=uuid,
                             gpu_used_before_mib=memory, gpu_lease_path=str(GPU_LEASE_DIR / f"gpu{gpu}.lock"),
                             inherited_lease_fd=True, triton_cache_dir=str(triton_cache),
                             argv=cmd, log=str(log_path), started=time.time())
                foreign = []
                while proc.poll() is None:
                    time.sleep(5)
                    if proc.poll() is not None:
                        break
                    _uuid, _used, owners = _nvidia_gpu_info(gpu)
                    foreign = foreign_owner_pids(owners, _process_tree(proc.pid))
                    if foreign:
                        self._update(task_id, status="interrupted_foreign_gpu_owner",
                                     foreign_owner_pids=foreign, observed_at=time.time())
                        self._terminate_owned_group(proc)
                        break
                rc = proc.wait()
                valid = rc == 0 and not foreign and adapter.valid_result(task)
                terminal = "passed" if valid else ("interrupted_foreign_gpu_owner" if foreign else "failed")
                self._update(task_id, status=terminal, returncode=rc,
                             artifact_valid=bool(valid), finished=time.time())
                return terminal
        except BaseException as exc:
            if proc is not None and proc.poll() is None:
                self._terminate_owned_group(proc)
            self._update(task_id, status="failed", error=repr(exc), finished=time.time())
            return "failed"
        finally:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
            lease.close()

    def _run_phase(self, stage):
        selected = [task for task in self.tasks if task["stage"] == stage]
        if stage == "train" and not adaptive_preflight_all_passed(
                list(self.state["tasks"].values())):
            for task in selected:
                self._update(task["task_id"], status="blocked_adaptive_recipe_guard",
                             reason="at least one registered adaptive seed preflight failed")
            return
        runnable = []
        for task in selected:
            dependency_states = [self.state["tasks"].get(dep, {}).get("status")
                                 for dep in task["depends_on"]]
            if any(state not in ("passed", "reused_verified") for state in dependency_states):
                self._update(task["task_id"], status="blocked_dependency", dependency_status=dependency_states)
                continue
            runnable.append(task)
        if not runnable:
            return
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(9, len(runnable))) as pool:
            futures = {pool.submit(self._execute, task): task for task in runnable}
            for future in concurrent.futures.as_completed(futures):
                # _execute records each terminal result and never retries it.
                future.result()

    def run(self):
        try:
            for stage in ("preflight", "train", "evaluate"):
                self._set_status("running", current_stage=stage, updated=time.time())
                self._run_phase(stage)
            statuses = [row["status"] for row in self.state["tasks"].values()]
            complete = all(status in ("passed", "reused_verified") for status in statuses)
            self._set_status("complete" if complete else "completed_with_failures",
                             finished=time.time(), completed_scientific_tasks=sum(
                                 status in ("passed", "reused_verified") for status in statuses),
                             failed_or_blocked_tasks=sum(
                                 status not in ("passed", "reused_verified") for status in statuses))
            return self.state
        except BaseException as exc:
            self._set_status("failed", error=repr(exc), finished=time.time())
            raise
        finally:
            fcntl.flock(self.queue_lock.fileno(), fcntl.LOCK_UN)
            self.queue_lock.close()


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the immutable 27-task plan only")
    args = parser.parse_args(argv)
    plan = adapter.task_plan()
    if args.dry_run:
        print(json.dumps({"experiment": "SW0123", "task_count": len(plan),
                          "tasks": plan}, allow_nan=False))
        return
    Dispatcher().run()


if __name__ == "__main__":
    main()
