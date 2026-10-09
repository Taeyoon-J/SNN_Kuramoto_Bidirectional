"""Exclusive-GPU, no-retry launcher for the SW0124 read-only evaluation."""
from __future__ import annotations

import json
import os
import pathlib
import signal
import subprocess
import sys
import time

try:
    import fcntl
except ImportError:  # The dispatcher is only executed on the Linux server.
    fcntl = None

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]
from collaborative_test.SW_0124_temporal_prototype_readout import coordinator

GPU_LEASE_DIR = pathlib.Path("/tmp/kevinswk_sw0113_gpu_leases")
MAX_GPU_MEMORY_MIB = 512
SERVER_PYTHON = "/Data0/kevinswk/envs/snn/bin/python"


def gpu_is_exclusive(memory_mib, compute_pids):
    return int(memory_mib) <= MAX_GPU_MEMORY_MIB and not list(compute_pids)


def _pid_is_confirmed_gone(pid):
    if os.name == "nt":
        return False
    try:
        pathlib.Path(f"/proc/{int(pid)}").stat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return False


def foreign_owner_pids(owner_pids, owned_process_pids):
    """Treat uncertain/live GPU owners conservatively as foreign."""
    owned = {int(pid) for pid in owned_process_pids}
    return sorted({int(pid) for pid in owner_pids if int(pid) not in owned
                   and not _pid_is_confirmed_gone(int(pid))})


def _nvidia_gpu_info(gpu):
    uuid = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"],
        text=True).strip()
    memory = int(subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used",
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
    rows = subprocess.check_output(["ps", "-eo", "pid=,ppid="], text=True)
    children = {}
    for line in rows.splitlines():
        values = line.split()
        if len(values) == 2:
            pid, parent = map(int, values)
            children.setdefault(parent, []).append(pid)
    result, stack = {int(root_pid)}, [int(root_pid)]
    while stack:
        parent = stack.pop()
        for child in children.get(parent, ()):
            if child not in result:
                result.add(child); stack.append(child)
    return result


def _atomic_json(path, value):
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


class Dispatcher:
    """Serial four-task preflight/evaluation queue; it never restarts failures."""
    def __init__(self, attempt_root):
        if fcntl is None:
            raise RuntimeError("SW0124 dispatcher requires Linux flock")
        self.attempt_root = pathlib.Path(attempt_root).resolve()
        self.attempt_root.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.attempt_root.with_suffix(".queue.lock")
        self.queue_lock = self.lock_path.open("a+")
        try:
            fcntl.flock(self.queue_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.queue_lock.close()
            raise RuntimeError("another SW0124 dispatcher owns this queue") from exc
        if self.attempt_root.exists():
            fcntl.flock(self.queue_lock.fileno(), fcntl.LOCK_UN); self.queue_lock.close()
            raise FileExistsError(f"preserve existing SW0124 attempt root: {self.attempt_root}")
        self.attempt_root.mkdir(parents=True, exist_ok=False)
        self.state_path = self.attempt_root / "dispatcher_state.json"
        self.tasks = coordinator.task_plan(self.attempt_root)
        self.state = {"experiment": "SW0124", "status": "running", "supervisor_pid": os.getpid(),
                      "created": time.time(), "tasks": {task["task_id"]: {
                          "task": task, "status": "queued"} for task in self.tasks}}
        self._save()

    def _save(self):
        _atomic_json(self.state_path, self.state)

    def _update(self, task_id, **fields):
        self.state["tasks"][task_id].update(fields)
        self._save()

    def _acquire_gpu(self, task):
        while True:
            for gpu in range(4):
                lease_path = GPU_LEASE_DIR / f"gpu{gpu}.lock"
                lease_path.parent.mkdir(parents=True, exist_ok=True)
                stream = lease_path.open("a+")
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    stream.close(); continue
                try:
                    uuid, memory, owners = _nvidia_gpu_info(gpu)
                    if gpu_is_exclusive(memory, owners):
                        self._update(task["task_id"], status="reserved", gpu=gpu,
                                     gpu_uuid=uuid, memory_before_mib=memory,
                                     gpu_lease_path=str(lease_path), reserved=time.time())
                        return gpu, stream, uuid, memory
                except BaseException:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN); stream.close(); raise
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN); stream.close()
            self._update(task["task_id"], status="waiting_for_exclusive_gpu", checked_at=time.time())
            time.sleep(5)

    def _stop_own_group(self, proc):
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
        artifact = coordinator.artifact_path(task)
        if artifact.exists():
            self._update(task_id, status="failed_existing_artifact", artifact=str(artifact))
            return False
        gpu, lease, uuid, memory = self._acquire_gpu(task)
        log_path = self.attempt_root / "logs" / f"{task_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        proc = None
        try:
            triton_cache = pathlib.Path(f"/tmp/kevinswk_sw0124_triton_{task_id}_{int(time.time()*1000)}")
            triton_cache.mkdir(parents=True, mode=0o700, exist_ok=False)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", TRITON_CACHE_DIR=str(triton_cache))
            argv = coordinator.command(task, "cuda:0")
            argv[0] = SERVER_PYTHON
            with log_path.open("x", encoding="utf-8") as log:
                proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                        pass_fds=(lease.fileno(),))
                self._update(task_id, status="running", pid=proc.pid, argv=argv, gpu=gpu,
                             gpu_uuid=uuid, memory_before_mib=memory,
                             lease_path=str(GPU_LEASE_DIR / f"gpu{gpu}.lock"),
                             inherited_lease=True, triton_cache=str(triton_cache),
                             log=str(log_path), started=time.time())
                foreign = []
                while proc.poll() is None:
                    time.sleep(5)
                    if proc.poll() is not None:
                        break
                    current_uuid, _used, owners = _nvidia_gpu_info(gpu)
                    if current_uuid != uuid:
                        foreign = [-1]
                    else:
                        foreign = foreign_owner_pids(owners, _process_tree(proc.pid))
                    if foreign:
                        self._update(task_id, status="interrupted_foreign_gpu_owner",
                                     foreign_pids=foreign, interrupted=time.time())
                        self._stop_own_group(proc)
                        break
                rc = proc.wait()
            ok = rc == 0 and not foreign and coordinator.valid_result(task)
            self._update(task_id, status="passed" if ok else "failed", returncode=rc,
                         artifact_valid=bool(ok), finished=time.time())
            return bool(ok)
        except BaseException as exc:
            if proc is not None and proc.poll() is None:
                self._stop_own_group(proc)
            self._update(task_id, status="failed", error=repr(exc), finished=time.time())
            return False
        finally:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close()

    def run(self):
        try:
            ok = {}
            for task in self.tasks:
                if any(not ok.get(dep, False) for dep in task["depends_on"]):
                    self._update(task["task_id"], status="blocked_dependency")
                    ok[task["task_id"]] = False
                    continue
                ok[task["task_id"]] = self._execute(task)
            complete = all(ok.values())
            self.state.update(status="complete" if complete else "completed_with_failures",
                              finished=time.time(), completed_tasks=sum(ok.values()),
                              failed_or_blocked_tasks=sum(not value for value in ok.values()))
            self._save()
            return self.state
        finally:
            fcntl.flock(self.queue_lock.fileno(), fcntl.LOCK_UN); self.queue_lock.close()


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-root", type=pathlib.Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    tasks = coordinator.task_plan(args.attempt_root)
    if args.dry_run:
        print(json.dumps({"experiment": "SW0124", "tasks": tasks,
                          "commands": [coordinator.command(task) for task in tasks]}, allow_nan=False))
        return
    state = Dispatcher(args.attempt_root).run()
    print(json.dumps({"status": state["status"], "task_statuses": {
        key: value["status"] for key, value in state["tasks"].items()}}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
