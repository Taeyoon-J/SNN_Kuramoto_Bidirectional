"""Owner-aware no-retry seed1 SW0139 calibration, preflight, train, and QCC queue."""
from __future__ import annotations

import concurrent.futures
import json
import math
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
from collaborative_test.SW_0139_native32_representation import run, evaluate

ARCHIVE = HERE / "results_archive"
STATE = ARCHIVE / "pilot_queue_state.json"
LOCK = ARCHIVE / "pilot_queue.lock"
LEASES = Path("/tmp/kevinswk_sw0113_gpu_leases")
MAX_MEMORY_MIB = 512
MAX_PARALLEL = 3


def task_plan():
    tasks = [{"task_id": "lambda_seed0", "stage": "lambda", "seed": 0, "depends_on": []}]
    for arm in run.ARMS:
        deps = ["lambda_seed0"] if arm == "cross_view" else []
        tasks.append({"task_id": f"preflight_{arm}_seed1", "stage": "preflight",
                      "seed": 1, "arm": arm, "depends_on": deps})
    for arm in run.ARMS:
        deps = [f"preflight_{arm}_seed1"]
        if arm == "cross_view":
            deps.append("lambda_seed0")
        tasks.append({"task_id": f"train_{arm}_seed1", "stage": "train",
                      "seed": 1, "arm": arm, "depends_on": deps})
    for arm in run.ARMS:
        tasks.append({"task_id": f"predict_{arm}_seed1", "stage": "predict",
                      "seed": 1, "arm": arm, "depends_on": [f"train_{arm}_seed1"]})
    tasks.append({"task_id": "score_seed1", "stage": "score", "seed": 1,
                  "depends_on": [f"predict_{arm}_seed1" for arm in run.ARMS]})
    return tasks


def artifact_path(task):
    stage, seed, arm = task["stage"], task.get("seed"), task.get("arm")
    if stage == "lambda":
        return run.LAMBDA_PATH
    if stage == "preflight" and arm in run.ARMS:
        return run.ARCHIVE / f"preflight_{arm}_seed{seed}.json"
    if stage == "train" and arm in run.ARMS:
        return run.OUT / f"{arm}_seed{seed}"
    if stage == "predict" and arm in run.ARMS:
        return evaluate.EVALUATION_ROOT / arm
    if stage == "score":
        return evaluate.EVALUATION_ROOT / "evaluation.json"
    raise ValueError("unregistered SW0139 task")


def command(task, device="cuda:0"):
    py = sys.executable
    if task["stage"] == "lambda":
        return [py, "-m", "collaborative_test.SW_0139_native32_representation.run",
                "--stage", "lambda", "--seed", "0", "--device", device,
                "--output", str(artifact_path(task))]
    if task["stage"] == "preflight":
        return [py, "-m", "collaborative_test.SW_0139_native32_representation.run",
                "--stage", "preflight", "--seed", "1", "--arm", task["arm"],
                "--device", device, "--output", str(artifact_path(task))]
    if task["stage"] == "train":
        return [py, "-m", "collaborative_test.SW_0139_native32_representation.run",
                "--stage", "train", "--seed", "1", "--arm", task["arm"],
                "--device", device,
                "--preflight-path", str(run.ARCHIVE / f"preflight_{task['arm']}_seed1.json"),
                "--lambda-path", str(run.LAMBDA_PATH), "--output-root", str(run.OUT)]
    if task["stage"] == "predict":
        return [py, "-m", "collaborative_test.SW_0139_native32_representation.evaluate",
                "--stage", "predict", "--seed", "1", "--arm", task["arm"],
                "--device", device, "--training-root", str(run.OUT),
                "--prediction-root", str(evaluate.EVALUATION_ROOT)]
    if task["stage"] == "score":
        return [py, "-m", "collaborative_test.SW_0139_native32_representation.evaluate",
                "--stage", "score", "--seed", "1", "--training-root", str(run.OUT),
                "--prediction-root", str(evaluate.EVALUATION_ROOT),
                "--source-root", str(evaluate.sw137.OUTPUT_ROOT),
                "--dataset", str(evaluate.DATASET), "--output", str(artifact_path(task))]
    raise ValueError(task["stage"])


def child_environment(task, physical_gpu, triton_cache):
    """Hide CUDA from the CPU-only scorer before its Python imports torch."""
    visible = "" if task["stage"] == "score" else str(physical_gpu)
    return dict(os.environ, CUDA_VISIBLE_DEVICES=visible, OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1", TRITON_CACHE_DIR=str(triton_cache))


def valid_result(task):
    try:
        path = artifact_path(task)
        if task["stage"] == "lambda":
            value = run.load_lambda_record(path)
            return value.get("status") == "lambda_calibrated"
        if task["stage"] == "preflight":
            row = json.loads(path.read_text(encoding="utf-8"))
            expected_updates = 2 if task["arm"] == "context_residual" else 1
            if (row.get("status") != "disposable_update_complete"
                    or row.get("seed") != 1 or row.get("arm") != task["arm"]
                    or row.get("ground_truth_used") is not False
                    or row.get("implementation_fingerprint") != run.implementation_fingerprint()
                    or row.get("optimizer_updates") != expected_updates
                    or len(row.get("steps", [])) != expected_updates
                    or row.get("source_unchanged_except_trainable_graph") is not True):
                return False
            if task["arm"] == "cross_view":
                if row.get("lambda_sha256") != run.sha256_file(run.LAMBDA_PATH):
                    return False
                families = row.get("separate_old_contrastive_gradient_families", {})
                return all(set(families.get(loss, {})) == {"encoder", "graph"}
                           and all(float(families[loss][k]) > 0 for k in ("encoder", "graph"))
                           and all(math.isfinite(float(families[loss][k])) for k in ("encoder", "graph"))
                           for loss in ("old", "contrastive"))
            return True
        if task["stage"] == "train":
            verified = evaluate.validate_training(1, task["arm"], path)
            return verified["manifest"].get("status") == "training_complete"
        if task["stage"] == "predict":
            row = evaluate.validate_arm_prediction(task["arm"])
            return row["protocol"].get("status") == "predictions_complete"
        if task["stage"] == "score":
            row = json.loads(path.read_text(encoding="utf-8"))
            return (row.get("status") in {"complete", "partial_complete"} and row.get("seed") == 1
                    and row.get("count") == 320
                    and "source97" in row.get("scores", {})
                    and bool(set(row.get("scores", {})) & set(run.ARMS))
                    and row.get("ground_truth_used_for_scoring") is True)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, FloatingPointError):
        return False
    return False


def _atomic(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


class PilotQueue:
    def __init__(self):
        if fcntl is None:
            raise RuntimeError("SW0139 owner-aware queue requires Linux fcntl")
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        self.queue_lock = LOCK.open("a+")
        try:
            fcntl.flock(self.queue_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.queue_lock.close()
            raise RuntimeError("another SW0139 pilot queue owns its lock") from exc
        if STATE.exists():
            self._unlock()
            raise FileExistsError(f"preserve prior SW0139 state: {STATE}")
        self.tasks = task_plan()
        for task in self.tasks:
            target = artifact_path(task)
            log = ARCHIVE / f"{task['task_id']}.log"
            if target.exists() or log.exists():
                self._unlock()
                raise FileExistsError(f"preserve existing SW0139 output/log: {target if target.exists() else log}")
        self.guard = threading.RLock()
        self.state = {"experiment": "SW0139_native32_representation", "status": "running",
                      "supervisor_pid": os.getpid(), "started": time.time(),
                      "runner_fingerprint": run.implementation_fingerprint(),
                      "evaluator_fingerprint": evaluate.implementation_fingerprint(),
                      "queue_sha256": run.sha256_file(HERE / "pilot_queue.py"),
                      "owner_dispatcher_sha256": run.sha256_file(owner.__file__),
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

    def _reserve(self, task_id):
        while True:
            for gpu in range(4):
                path = LEASES / f"gpu{gpu}.lock"
                path.parent.mkdir(parents=True, exist_ok=True)
                lease = path.open("a+")
                try:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lease.close()
                    continue
                try:
                    uuid, memory, owners = owner._nvidia_gpu_info(gpu)
                    if not owners and memory <= MAX_MEMORY_MIB:
                        self._update(task_id, status="reserved", physical_gpu=gpu,
                                     gpu_uuid=uuid, memory_before_mib=memory,
                                     lease_path=str(path), reservation_pid=os.getpid())
                        return gpu, lease, uuid
                except BaseException:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); raise
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close()
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
        key = task["task_id"]
        artifact = artifact_path(task)
        log_path = ARCHIVE / f"{key}.log"
        lease = proc = None
        try:
            if artifact.exists() or log_path.exists():
                self._update(key, status="failed_existing_artifact", finished=time.time())
                return False
            gpu, lease, uuid = self._reserve(key)
            triton = Path(f"/tmp/kevinswk_sw0139_{key}_{int(time.time()*1000)}")
            triton.mkdir(parents=True, exist_ok=False)
            argv = command(task, "cuda:0")
            env = child_environment(task, gpu, triton)
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
                    if task["stage"] != "score":
                        _uuid, _mem, pids = owner._nvidia_gpu_info(gpu)
                        foreign = owner.foreign_owner_pids(pids, owner._process_tree(proc.pid))
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
            if proc is not None:
                self._stop_owned(proc)
            self._update(key, status="failed", error=repr(exc), finished=time.time())
            return False
        finally:
            if lease is not None:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                lease.close()

    def run(self):
        task_map = {t["task_id"]: t for t in self.tasks}
        statuses = {key: "queued" for key in task_map}
        running = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
            while any(v in {"queued", "running"} for v in statuses.values()):
                scheduled = False
                for key, task in task_map.items():
                    if statuses[key] != "queued" or len(running) >= MAX_PARALLEL:
                        continue
                    deps = task.get("depends_on", [])
                    if task["stage"] == "score" and all(
                            statuses[d] not in {"queued", "running"} for d in deps):
                        statuses[key] = "running"
                        future = pool.submit(self._execute, task)
                        running[future] = key
                        scheduled = True
                    elif any(statuses[d] in {"failed", "blocked"} for d in deps):
                        statuses[key] = "blocked"
                        self._update(key, status="blocked_dependency", finished=time.time())
                        scheduled = True
                    elif all(statuses[d] == "passed" for d in deps):
                        statuses[key] = "running"
                        future = pool.submit(self._execute, task)
                        running[future] = key
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
        terminal = "complete" if statuses.get("score_seed1") == "passed" else "pilot_incomplete"
        with self.guard:
            self.state.update(status=terminal, finished=time.time(), task_statuses=statuses)
            _atomic(STATE, self.state)
        self._unlock()
        return terminal == "complete"


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"max_parallel": MAX_PARALLEL, "tasks": task_plan(),
                          "cross_view_waits_for_seed0_lambda": True,
                          "control_context_independent_of_lambda": True,
                          "no_retries": True}, indent=2))
        return 0
    return 0 if PilotQueue().run() else 1


if __name__ == "__main__":
    raise SystemExit(main())
