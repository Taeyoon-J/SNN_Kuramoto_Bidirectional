"""Exclusive, no-retry sequential queue for the three SW0140 TRAIN screens."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

try:
    import fcntl
except ImportError:  # Windows supports dry-run only.
    fcntl = None

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0123_adaptive_temporal_assignment import dispatcher as owner
from collaborative_test.SW_0140_native32_causal_screen import screen

ARCHIVE = HERE / "results_archive"
STATE = ARCHIVE / "screen_queue_state.json"
LOCK = ARCHIVE / "screen_queue.lock"
LEASES = Path("/tmp/kevinswk_sw0113_gpu_leases")
MAX_MEMORY_MIB = 512


def foreign_owners(pids, owned_tree):
    return owner.foreign_owner_pids(pids, owned_tree)


def task_plan():
    return [
        {"task_id": f"screen_seed{seed}", "stage": "screen", "seed": seed,
         "depends_on": [] if seed == 0 else [f"screen_seed{seed - 1}"]}
        for seed in (0, 1, 2)
    ]


def output_path(task):
    return ARCHIVE / f"causal_screen_seed{int(task['seed'])}.json"


def command(task):
    return [sys.executable, "-m", "collaborative_test.SW_0140_native32_causal_screen.screen",
            "--seed", str(int(task["seed"])), "--device", "cuda:0",
            "--output", str(output_path(task))]


def valid_result(task):
    try:
        path = output_path(task)
        row = json.loads(path.read_text(encoding="utf-8"))
        expected = screen.screen_fingerprint(
            screen.load_native32_foundation(int(task["seed"]), "cpu", verify_rgb_assets=False)
            .provenance["implementation_fingerprint"])
        return (row.get("experiment") == "SW0140_native32_causal_screen"
                and row.get("status") == "screen_complete"
                and row.get("seed") == int(task["seed"])
                and row.get("ground_truth_used") is False
                and row.get("optimizer_updates") == 0
                and row.get("implementation_fingerprint") == expected
                and len(row.get("batches", [])) == 4
                and [b.get("ordinal") for b in row["batches"]] == [0, 1, 2, 3]
                and all(b.get("conditions", {}).get("none", {}).get("parity") is True
                        for b in row["batches"]))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, FloatingPointError):
        return False


def _atomic(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class ScreenQueue:
    def __init__(self):
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        self.queue_lock = LOCK.open("a+")
        try:
            fcntl.flock(self.queue_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.queue_lock.close()
            raise RuntimeError("another SW0140 queue owns its lock") from exc
        tasks = task_plan()
        if STATE.exists():
            self._unlock()
            raise FileExistsError(f"preserve prior SW0140 queue state: {STATE}")
        for task in tasks:
            target, log = output_path(task), ARCHIVE / f"{task['task_id']}.log"
            if target.exists() or log.exists():
                self._unlock()
                raise FileExistsError(f"preserve prior SW0140 output/log: {target if target.exists() else log}")
        self.tasks = tasks
        self.guard = threading.RLock()
        self.state = {"experiment": "SW0140_native32_causal_screen", "status": "running",
                      "supervisor_pid": os.getpid(), "started": time.time(),
                      "screen_fingerprint": screen.screen_fingerprint(
                          screen.load_native32_foundation(0, "cpu", verify_rgb_assets=False)
                          .provenance["implementation_fingerprint"]),
                      "queue_sha256": screen.sha256_file(HERE / "screen_queue.py"),
                      "owner_dispatcher_sha256": screen.sha256_file(owner.__file__),
                      "max_parallel": 1, "no_retries": True,
                      "tasks": {t["task_id"]: {"task": t, "status": "queued"} for t in tasks}}
        _atomic(STATE, self.state)

    def _unlock(self):
        try:
            fcntl.flock(self.queue_lock.fileno(), fcntl.LOCK_UN)
        finally:
            self.queue_lock.close()

    def _update(self, key, **fields):
        with self.guard:
            self.state["tasks"][key].update(fields)
            _atomic(STATE, self.state)

    def _reserve(self, key):
        while True:
            for gpu in range(4):
                lease_path = LEASES / f"gpu{gpu}.lock"
                lease_path.parent.mkdir(parents=True, exist_ok=True)
                lease = lease_path.open("a+")
                try:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lease.close()
                    continue
                try:
                    uuid, memory, owners = owner._nvidia_gpu_info(gpu)
                    if not owners and memory <= MAX_MEMORY_MIB:
                        self._update(key, status="reserved", physical_gpu=gpu,
                                     gpu_uuid=uuid, memory_before_mib=memory,
                                     lease_path=str(lease_path), reservation_pid=os.getpid())
                        return gpu, lease, uuid
                except BaseException:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                    lease.close()
                    raise
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                lease.close()
            self._update(key, status="waiting_for_exclusive_gpu", last_wait=time.time())
            time.sleep(5)

    @staticmethod
    def _stop_owned(proc):
        if proc is None or proc.poll() is not None:
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
        key, artifact = task["task_id"], output_path(task)
        log_path = ARCHIVE / f"{key}.log"
        proc = lease = None
        try:
            if artifact.exists() or log_path.exists():
                raise FileExistsError("preserve existing per-seed screen artifact/log")
            gpu, lease, uuid = self._reserve(key)
            triton = Path(f"/tmp/kevinswk_sw0140_{key}_{int(time.time()*1000)}")
            triton.mkdir(parents=True, exist_ok=False)
            argv = command(task)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", TRITON_CACHE_DIR=str(triton))
            self._update(key, status="launching", argv=argv, physical_gpu=gpu,
                         gpu_uuid=uuid, log_path=str(log_path), triton_cache=str(triton),
                         launched=time.time())
            with log_path.open("x", encoding="utf-8") as stream:
                proc = subprocess.Popen(argv, cwd=str(ROOT), env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, start_new_session=True,
                                        pass_fds=(lease.fileno(),))
                self._update(key, status="running", child_pid=proc.pid, started=time.time())
                while proc.poll() is None:
                    time.sleep(5)
                    if proc.poll() is not None:
                        break
                    _uuid, _memory, pids = owner._nvidia_gpu_info(gpu)
                    foreign = foreign_owners(pids, owner._process_tree(proc.pid))
                    if foreign:
                        self._stop_owned(proc)
                        self._update(key, status="interrupted_foreign_gpu_owner",
                                     foreign_owner_pids=foreign, finished=time.time())
                        return False
            rc = int(proc.returncode)
            valid = rc == 0 and valid_result(task)
            self._update(key, status="passed" if valid else "failed", returncode=rc,
                         artifact_valid=bool(valid), finished=time.time())
            return bool(valid)
        except BaseException as exc:
            self._stop_owned(proc)
            self._update(key, status="failed", error=repr(exc), finished=time.time())
            return False
        finally:
            if lease is not None:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                lease.close()

    def run(self):
        statuses = {t["task_id"]: "queued" for t in self.tasks}
        overall = True
        for task in self.tasks:
            key = task["task_id"]
            if any(statuses[dependency] != "passed" for dependency in task["depends_on"]):
                statuses[key] = "blocked"
                self._update(key, status="blocked_dependency", finished=time.time())
                overall = False
                continue
            statuses[key] = "running"
            ok = self._execute(task)
            statuses[key] = "passed" if ok else "failed"
            overall = overall and ok
            if not ok:
                for later in self.tasks:
                    if statuses[later["task_id"]] == "queued":
                        statuses[later["task_id"]] = "blocked"
                        self._update(later["task_id"], status="blocked_dependency",
                                     finished=time.time())
                break
        with self.guard:
            self.state.update(status="complete" if overall else "screen_incomplete",
                              task_statuses=statuses, finished=time.time())
            _atomic(STATE, self.state)
        self._unlock()
        return overall


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"max_parallel": 1, "no_retries": True,
                          "tasks": task_plan()}, indent=2), flush=True)
        return 0
    if fcntl is None:
        raise RuntimeError("SW0140 screen queue requires Linux fcntl")
    return 0 if ScreenQueue().run() else 1


if __name__ == "__main__":
    raise SystemExit(main())
