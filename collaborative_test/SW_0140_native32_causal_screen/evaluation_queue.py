"""Owner-aware no-retry queue for the fixed SW0140 native32 causal endpoint."""
from __future__ import annotations

import concurrent.futures
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
from collaborative_test.SW_0140_native32_causal_screen import evaluate

ARCHIVE = HERE / "results_archive"
STATE = ARCHIVE / "evaluation_queue_state.json"
LOCK = ARCHIVE / "evaluation_queue.lock"
LEASES = Path("/tmp/kevinswk_sw0113_gpu_leases")
MAX_MEMORY_MIB = 512
MAX_PARALLEL = 3


def foreign_owners(pids, owned_tree):
    return owner.foreign_owner_pids(pids, owned_tree)


def task_plan():
    tasks = [{"task_id": f"predict_{condition}_seed{seed}", "stage": "predict",
              "condition": condition, "seed": seed, "depends_on": []}
             for condition in evaluate.CONDITIONS_TO_SCORE for seed in evaluate.SEEDS]
    tasks.append({"task_id": "score_all", "stage": "score", "depends_on":
                  [task["task_id"] for task in tasks]})
    return tasks


def artifact_path(task):
    if task["stage"] == "predict":
        return evaluate.prediction_dir(task["seed"], task["condition"])
    if task["stage"] == "score":
        return evaluate.SUMMARY_PATH
    raise ValueError("unregistered SW0140 evaluation task")


def command(task, device="cuda:0"):
    if task["stage"] == "predict":
        return [sys.executable, "-m", "collaborative_test.SW_0140_native32_causal_screen.evaluate",
                "--stage", "predict", "--seed", str(task["seed"]),
                "--condition", task["condition"], "--device", device]
    if task["stage"] == "score":
        return [sys.executable, "-m", "collaborative_test.SW_0140_native32_causal_screen.evaluate",
                "--stage", "score"]
    raise ValueError("unregistered SW0140 task stage")


def child_environment(task, physical_gpu=None, triton_cache=None):
    visible = "" if task["stage"] == "score" else str(physical_gpu)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=visible,
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if triton_cache is not None:
        environment["TRITON_CACHE_DIR"] = str(triton_cache)
    return environment


def valid_result(task):
    try:
        if task["stage"] == "predict":
            row = evaluate.validate_prediction(task["seed"], task["condition"])
            return (row["protocol"].get("status") == "predictions_complete"
                    and row["protocol"].get("ground_truth_used_for_prediction") is False)
        if task["stage"] == "score":
            report = json.loads(evaluate.SUMMARY_PATH.read_text(encoding="utf-8"))
            expected = {f"{condition}_seed{seed}" for condition in evaluate.CONDITIONS_TO_SCORE
                        for seed in evaluate.SEEDS}
            return (report.get("status") == "complete"
                    and set(report.get("conditions", [])) == set(evaluate.CONDITIONS_TO_SCORE)
                    and set(report.get("seed_scope", [])) == set(evaluate.SEEDS)
                    and expected.issubset(report.get("scores", {}))
                    and report.get("ground_truth_used_for_prediction") is False
                    and report.get("ground_truth_used_for_scoring") is True)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, FloatingPointError):
        return False
    return False


def _atomic(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


class EvaluationQueue:
    def __init__(self):
        if fcntl is None:
            raise RuntimeError("SW0140 evaluation queue requires Linux fcntl")
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        self.queue_lock = LOCK.open("a+")
        try:
            fcntl.flock(self.queue_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.queue_lock.close()
            raise RuntimeError("another SW0140 evaluation queue owns its lock") from exc
        self.tasks = task_plan()
        if STATE.exists():
            self._unlock()
            raise FileExistsError(f"preserve prior SW0140 evaluation queue state: {STATE}")
        for task in self.tasks:
            target = artifact_path(task)
            log = ARCHIVE / f"{task['task_id']}.log"
            if target.exists() or log.exists():
                self._unlock()
                raise FileExistsError(f"preserve existing SW0140 artifact/log: {target if target.exists() else log}")
        self.guard = threading.RLock()
        self.state = {"experiment": "SW0140_native32_causal_endpoint", "status": "running",
                      "supervisor_pid": os.getpid(), "started": time.time(),
                      "evaluation_fingerprint": evaluate.implementation_fingerprint(),
                      "queue_sha256": evaluate.sha256_file(HERE / "evaluation_queue.py"),
                      "owner_dispatcher_sha256": evaluate.sha256_file(owner.__file__),
                      "max_parallel": MAX_PARALLEL, "no_retries": True,
                      "tasks": {t["task_id"]: {"task": t, "status": "queued"} for t in self.tasks}}
        _atomic(STATE, self.state)

    def _unlock(self):
        try:
            fcntl.flock(self.queue_lock.fileno(), fcntl.LOCK_UN)
        finally:
            self.queue_lock.close()

    def _update(self, task_id, **fields):
        with self.guard:
            self.state["tasks"][task_id].update(fields)
            _atomic(STATE, self.state)

    def _reserve_gpu(self, task_id):
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
                    uuid, memory_mib, owners = owner._nvidia_gpu_info(gpu)
                    if not owners and memory_mib <= MAX_MEMORY_MIB:
                        self._update(task_id, status="reserved", physical_gpu=gpu,
                                     gpu_uuid=uuid, memory_before_mib=memory_mib,
                                     lease_path=str(lease_path), reservation_pid=os.getpid())
                        return gpu, lease, uuid
                except BaseException:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                    lease.close()
                    raise
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                lease.close()
            self._update(task_id, status="waiting_for_exclusive_gpu", last_wait=time.time())
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
        task_id = task["task_id"]
        target = artifact_path(task)
        log = ARCHIVE / f"{task_id}.log"
        proc = lease = None
        try:
            if target.exists() or log.exists():
                raise FileExistsError("preserve existing SW0140 artifact/log; no retry")
            gpu = uuid = None
            triton = None
            if task["stage"] == "predict":
                gpu, lease, uuid = self._reserve_gpu(task_id)
                triton = Path(f"/tmp/kevinswk_sw0140_eval_{task_id}_{int(time.time()*1000)}")
                triton.mkdir(parents=True, exist_ok=False)
            argv = command(task, "cuda:0")
            env = child_environment(task, gpu, triton)
            self._update(task_id, status="launching", argv=argv, physical_gpu=gpu,
                         gpu_uuid=uuid, log_path=str(log), triton_cache=str(triton),
                         launched=time.time())
            with log.open("x", encoding="utf-8") as stream:
                pass_fds = (lease.fileno(),) if lease is not None else ()
                proc = subprocess.Popen(argv, cwd=str(ROOT), env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, start_new_session=True,
                                        pass_fds=pass_fds)
                self._update(task_id, status="running", child_pid=proc.pid, started=time.time())
                while proc.poll() is None:
                    time.sleep(5)
                    if proc.poll() is not None:
                        break
                    if task["stage"] == "predict":
                        _uuid, _mem, pids = owner._nvidia_gpu_info(gpu)
                        foreign = foreign_owners(pids, owner._process_tree(proc.pid))
                        if foreign:
                            self._stop_owned(proc)
                            self._update(task_id, status="interrupted_foreign_gpu_owner",
                                         foreign_owner_pids=foreign, finished=time.time())
                            return False
            rc = int(proc.returncode)
            valid = rc == 0 and valid_result(task)
            self._update(task_id, status="passed" if valid else "failed", returncode=rc,
                         artifact_valid=bool(valid), finished=time.time())
            return bool(valid)
        except BaseException as exc:
            self._stop_owned(proc)
            self._update(task_id, status="failed", error=repr(exc), finished=time.time())
            return False
        finally:
            if lease is not None:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                lease.close()

    def run(self):
        task_map = {task["task_id"]: task for task in self.tasks}
        statuses = {key: "queued" for key in task_map}
        running = {}
        max_workers = MAX_PARALLEL
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            while any(status in {"queued", "running"} for status in statuses.values()):
                scheduled = False
                for key, task in task_map.items():
                    if statuses[key] != "queued" or len(running) >= max_workers:
                        continue
                    deps = task.get("depends_on", [])
                    if task["stage"] == "score":
                        if any(statuses[dep] in {"queued", "running"} for dep in deps):
                            continue
                        if not all(statuses[dep] == "passed" for dep in deps):
                            statuses[key] = "blocked"
                            self._update(key, status="blocked_dependency", finished=time.time())
                            scheduled = True
                            continue
                    elif any(statuses[dep] in {"failed", "blocked"} for dep in deps):
                        statuses[key] = "blocked"
                        self._update(key, status="blocked_dependency", finished=time.time())
                        scheduled = True
                        continue
                    statuses[key] = "running"
                    running[executor.submit(self._execute, task)] = key
                    scheduled = True
                if running:
                    done, _ = concurrent.futures.wait(running,
                        return_when=concurrent.futures.FIRST_COMPLETED, timeout=2)
                    for future in done:
                        key = running.pop(future)
                        try:
                            ok = bool(future.result())
                        except BaseException:
                            ok = False
                        statuses[key] = "passed" if ok else "failed"
                elif not scheduled:
                    break
        complete = statuses.get("score_all") == "passed"
        with self.guard:
            self.state.update(status="complete" if complete else "evaluation_incomplete",
                              task_statuses=statuses, finished=time.time())
            _atomic(STATE, self.state)
        self._unlock()
        return complete


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"max_parallel": MAX_PARALLEL, "no_retries": True,
                          "task_count": len(task_plan()), "tasks": task_plan()}, indent=2), flush=True)
        return 0
    return 0 if EvaluationQueue().run() else 1


if __name__ == "__main__":
    raise SystemExit(main())
