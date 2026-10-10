"""Single-task owner-aware exclusive GPU launcher for the SW0135 resource probe."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

try:
    import fcntl
except ImportError:  # Windows is limited to --dry-run and pure contract tests.
    fcntl = None

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "results_archive"
STATE = ARCHIVE / "resource_probe_queue_state.json"
QUEUE_LOCK = ARCHIVE / "resource_probe_queue.lock"
GPU_LEASE_DIR = Path("/tmp/kevinswk_sw0113_gpu_leases")
GPU_MEMORY_LIMIT_MIB = 512

sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]
from collaborative_test.SW_0123_adaptive_temporal_assignment import dispatcher as owner


def task_plan():
    return [{"task_id": "sw0135_resource_probe_seed0", "seed": 0,
             "stage": "resource_probe", "device": "cuda:0"}]


def result_path(task):
    if task.get("stage") != "resource_probe" or int(task.get("seed", -1)) != 0:
        raise ValueError("SW0135 resource queue only admits its registered seed0 probe")
    return ARCHIVE / "resource_probe_seed0.json"


def log_path(task):
    result_path(task)
    return ARCHIVE / "resource_probe_seed0.log"


def command(task):
    output = result_path(task)
    return [sys.executable, "-m",
            "collaborative_test.SW_0135_native32_spike_binding.resource_probe",
            "--seed", "0", "--device", "cuda:0", "--output", str(output)]


def gpu_is_exclusive_candidate(memory_mib, compute_owners):
    return not compute_owners and int(memory_mib) <= GPU_MEMORY_LIMIT_MIB


def foreign_owners(compute_owners, owned_processes):
    return owner.foreign_owner_pids(compute_owners, owned_processes)


def ensure_absent(paths):
    existing = [str(Path(path)) for path in paths if Path(path).exists()]
    if existing:
        raise FileExistsError("preserve existing resource-probe attempt paths: " + ", ".join(existing))


def validate_probe_record(path):
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        return (record.get("experiment") == "SW0135_native32_spike_binding"
                and record.get("status") == "resource_probe_complete"
                and record.get("resource_only") is True
                and record.get("training_admission") is False
                and record.get("ground_truth_used") is False
                and record.get("optimizer_updates") == 0
                and record.get("resource_optimizer_steps") == {"joint": 1, "head_decoder": 1}
                and record.get("image_id_count") == 16
                and len(record.get("microbatch_seconds", [])) == 16)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _atomic_state(path, record):
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


class ResourceQueue:
    def __init__(self):
        if fcntl is None:
            raise RuntimeError("SW0135 GPU resource queue requires Linux flock")
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        self.queue_lock = QUEUE_LOCK.open("a+")
        try:
            fcntl.flock(self.queue_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.queue_lock.close()
            raise RuntimeError("another SW0135 resource queue owns its lock") from exc
        try:
            ensure_absent([STATE, result_path(task_plan()[0]), log_path(task_plan()[0])])
        except BaseException:
            fcntl.flock(self.queue_lock.fileno(), fcntl.LOCK_UN)
            self.queue_lock.close()
            raise
        self.task = task_plan()[0]
        self.record = {
            "experiment": "SW0135_native32_spike_binding",
            "stage": "resource_probe_only", "status": "running",
            "supervisor_pid": os.getpid(), "started": time.time(),
            "max_parallel": 1, "task_count": 1,
            "owner_dispatcher_sha256": _sha(Path(owner.__file__)),
            "queue_sha256": _sha(HERE / "resource_queue.py"),
            "tasks": {self.task["task_id"]: {"task": self.task, "status": "queued"}},
        }
        self._save()

    def _save(self):
        _atomic_state(STATE, self.record)

    def _update(self, **fields):
        self.record["tasks"][self.task["task_id"]].update(fields)
        self._save()

    def _finish(self, status):
        self.record.update(status=status, finished=time.time())
        self._save()

    def _reserve_gpu(self):
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
                    uuid, memory, owners = owner._nvidia_gpu_info(gpu)
                    if gpu_is_exclusive_candidate(memory, owners):
                        self._update(status="reserved", gpu=gpu, gpu_uuid=uuid,
                                     memory_before_mib=memory, lease_path=str(lease_path),
                                     reservation_pid=os.getpid(), reserved=time.time())
                        return gpu, lease, uuid
                except BaseException:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                    lease.close()
                    raise
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                lease.close()
            self._update(status="waiting_for_exclusive_gpu", last_wait=time.time())
            time.sleep(5)

    @staticmethod
    def _terminate_owned_child(proc):
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

    def run(self):
        task = self.task
        log = log_path(task)
        result = result_path(task)
        triton = None
        lease = None
        proc = None
        try:
            gpu, lease, uuid = self._reserve_gpu()
            triton = Path(f"/tmp/kevinswk_sw0135_resource_{task['task_id']}_{int(time.time() * 1000)}")
            triton.mkdir(parents=True, exist_ok=False)
            argv = command(task)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", TRITON_CACHE_DIR=str(triton))
            self._update(status="launching", gpu=gpu, gpu_uuid=uuid, argv=argv,
                         output=str(result), log_path=str(log), triton_cache=str(triton),
                         environment={"CUDA_VISIBLE_DEVICES": str(gpu),
                                      "TRITON_CACHE_DIR": str(triton)}, launched=time.time())
            with log.open("x", encoding="utf-8") as stream:
                proc = subprocess.Popen(argv, cwd=str(ROOT), env=env,
                                        stdout=stream, stderr=subprocess.STDOUT,
                                        start_new_session=True, pass_fds=(lease.fileno(),))
                self._update(status="running", child_pid=proc.pid, started=time.time())
                while proc.poll() is None:
                    time.sleep(5)
                    if proc.poll() is not None:
                        break
                    _uuid, _memory, owners = owner._nvidia_gpu_info(gpu)
                    foreign = foreign_owners(owners, owner._process_tree(proc.pid))
                    if foreign:
                        self._terminate_owned_child(proc)
                        self._update(status="interrupted_foreign_gpu_owner",
                                     foreign_owner_pids=foreign, finished=time.time())
                        self._finish("interrupted_foreign_gpu_owner")
                        return
            rc = int(proc.returncode)
            valid = rc == 0 and validate_probe_record(result)
            self._update(status="passed" if valid else "failed", returncode=rc,
                         artifact_valid=bool(valid), finished=time.time())
            self._finish("resource_probe_complete" if valid else "resource_probe_failed")
        except BaseException as exc:
            if proc is not None:
                self._terminate_owned_child(proc)
            self._update(status="failed", error=repr(exc), finished=time.time())
            self._finish("resource_probe_failed")
        finally:
            if proc is not None and proc.poll() is None:
                self._terminate_owned_child(proc)
            if lease is not None:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                lease.close()
            self.queue_lock.close()


def _sha(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"experiment": "SW0135_native32_spike_binding",
                          "stage": "resource_probe_only", "max_parallel": 1,
                          "tasks": [{**task, "argv": command(task)} for task in task_plan()]},
                         indent=2, allow_nan=False))
        return
    ResourceQueue().run()


if __name__ == "__main__":
    main()
