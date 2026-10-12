"""Create-once owner-aware SW0143 preflight/inference/scoring queue."""
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
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    import fcntl
except ImportError:
    fcntl = None

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]
from collaborative_test.SW_0123_adaptive_temporal_assignment import dispatcher as owner
from collaborative_test.SW_0143_native32_selective_geodesic import run

ARCHIVE = HERE / "results_archive"
STATE = ARCHIVE / "selective_geodesic_queue_state.json"
LOCK = ARCHIVE / "selective_geodesic_queue.lock"
GPU_LEASES = Path("/tmp/kevinswk_sw0113_gpu_leases")
MEMORY_LIMIT_MIB = 512
MAX_PARALLEL = 4


def task_plan():
    tasks = []
    for seed in run.SEEDS:
        tasks.append({"task_id": f"preflight_s{seed}", "stage": "preflight", "seed": seed, "deps": []})
    for seed in run.SEEDS:
        tasks.append({"task_id": f"reference_R_s{seed}", "stage": "reference", "seed": seed,
                      "arm": "R", "deps": [f"preflight_s{seed}"]})
    for arm in ("C", "T"):
        for seed in run.SEEDS:
            tasks.append({"task_id": f"predict_{arm}_s{seed}", "stage": "predict", "seed": seed,
                          "arm": arm, "deps": [f"reference_R_s{seed}"]})
    tasks.append({"task_id": "score_all", "stage": "score", "deps": [
        f"reference_R_s{seed}" for seed in run.SEEDS]
        + [f"predict_{arm}_s{seed}" for arm in ("C", "T") for seed in run.SEEDS]})
    return tasks


def task_groups():
    return [tuple(f"preflight_s{s}" for s in run.SEEDS),
            tuple(f"reference_R_s{s}" for s in run.SEEDS),
            tuple(f"predict_{arm}_s{s}" for arm in ("C", "T") for s in run.SEEDS),
            ("score_all",)]


def output_path(task):
    stage, seed, arm = task["stage"], task.get("seed"), task.get("arm")
    if stage == "preflight":
        return run._preflight_path(seed)
    if stage in ("reference", "predict"):
        return run._arm_dir(arm, seed)
    if stage == "score":
        return run.SUMMARY
    raise ValueError(stage)


def log_path(task):
    return ARCHIVE / f"{task['task_id']}.log"


def command(task, *, device="cuda:0", python=sys.executable):
    stage = task["stage"]
    argv = [python, str(HERE / "run.py"), "--stage", stage]
    if "seed" in task:
        argv += ["--seed", str(int(task["seed"]))]
    if "arm" in task and stage == "predict":
        argv += ["--arm", task["arm"]]
    argv += ["--device", device]
    if stage == "score":
        argv += ["--output", str(run.SUMMARY)]
    return argv


def environment(task, *, physical_gpu=None, triton_cache=None):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PATCH_V2_REPO=str(ROOT))
    if task["stage"] == "score":
        env["CUDA_VISIBLE_DEVICES"] = ""
        env.pop("TRITON_CACHE_DIR", None)
    else:
        if physical_gpu not in range(4):
            raise ValueError("GPU task requires a registered physical device0..3")
        env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
        if triton_cache is not None:
            env["TRITON_CACHE_DIR"] = str(triton_cache)
    return env


def gpu_candidate(memory_mib, compute_owners):
    return not compute_owners and int(memory_mib) <= MEMORY_LIMIT_MIB


def validate_result(task):
    try:
        path = output_path(task)
        if task["stage"] == "preflight":
            row = json.loads(path.read_text(encoding="utf-8"))
            return (row.get("status") == "preflight_complete" and row.get("seed") == task["seed"]
                    and row.get("implementation") == run.implementation_fingerprint()
                    and row.get("ground_truth_used") is False and row.get("optimizer_updates") == 0
                    and row.get("source_r_native_trace_equal") == {
                        batch: {key: True for key in run.TRACE_FIELDS} for batch in ("B1", "B4")}
                    and set(row.get("native_b1_b4_trace_max_abs_descriptive", {})) == set(run.TRACE_FIELDS)
                    and set(row.get("r_b1_b4_trace_max_abs_descriptive", {})) == set(run.TRACE_FIELDS)
                    and set(row.get("intervention_b4_trace_shapes", {})) == {"C", "T"}
                    and row.get("cuda_peak_allocated_bytes", 0) > 0)
        if task["stage"] in {"reference", "predict"}:
            row = run.validate_arm_prediction(path, seed=task["seed"])
            return row["arm"] == task["arm"]
        if task["stage"] == "score":
            row = json.loads(path.read_text(encoding="utf-8"))
            return (row.get("status") == "complete" and row.get("seed_scope") == list(run.SEEDS)
                    and row.get("ground_truth_opened_after_all_predictions_validated") is True
                    and row.get("optimizer_updates") == 0)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, json.JSONDecodeError):
        return False
    return False


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


class SelectiveGeodesicQueue:
    def __init__(self):
        if fcntl is None:
            raise RuntimeError("SW0143 GPU queue requires Linux flock")
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        if STATE.exists() or LOCK.exists():
            raise FileExistsError("preserve existing SW0143 queue state/lock")
        self.tasks = task_plan()
        paths = [output_path(t) for t in self.tasks] + [log_path(t) for t in self.tasks]
        existing = [str(path) for path in paths if path.exists()]
        if existing:
            raise FileExistsError("preserve existing SW0143 outputs: " + ", ".join(existing))
        self.queue_lock = LOCK.open("x+")
        try:
            fcntl.flock(self.queue_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.queue_lock.close(); raise RuntimeError("another SW0143 queue owns the lock") from exc
        self.guard = threading.RLock()
        self.state = {"experiment": "SW0143_native32_selective_geodesic", "status": "running",
                      "supervisor_pid": os.getpid(), "started": time.time(),
                      "implementation_fingerprint": run.implementation_fingerprint(),
                      "queue_sha256": run.sha256_file(Path(__file__)),
                      "owner_dispatcher_sha256": run.sha256_file(Path(owner.__file__)),
                      "max_parallel": MAX_PARALLEL, "gpu_memory_limit_mib": MEMORY_LIMIT_MIB,
                      "ground_truth_used_for_prediction": False, "optimizer_updates": 0,
                      "tasks": {t["task_id"]: {"task": t, "status": "queued"} for t in self.tasks}}
        self._save()

    def _save(self):
        with self.guard:
            _atomic_json(STATE, self.state)

    def _update(self, task_id, **values):
        with self.guard:
            self.state["tasks"][task_id].update(values)
            _atomic_json(STATE, self.state)

    def _reserve(self, task_id):
        while True:
            for gpu in range(4):
                lease_path = GPU_LEASES / f"gpu{gpu}.lock"
                lease_path.parent.mkdir(parents=True, exist_ok=True)
                lease = lease_path.open("a+")
                try:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lease.close(); continue
                try:
                    uuid, memory, owners = owner._nvidia_gpu_info(gpu)
                    if gpu_candidate(memory, owners):
                        self._update(task_id, status="reserved", physical_gpu=gpu, gpu_uuid=uuid,
                                     memory_before_mib=memory, lease_path=str(lease_path))
                        return gpu, lease, uuid
                except BaseException:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); raise
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close()
            self._update(task_id, status="waiting_for_exclusive_gpu", last_wait=time.time())
            time.sleep(5)

    @staticmethod
    def _terminate_owned(proc):
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try: os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            proc.wait()

    def _execute(self, task):
        task_id = task["task_id"]
        gpu = lease = proc = None
        try:
            if task["stage"] == "score":
                env = environment(task)
            else:
                gpu, lease, uuid = self._reserve(task_id)
                triton = Path("/tmp") / f"kevinswk_sw0143_{os.getpid()}_{task_id}_{int(time.time()*1000)}"
                triton.mkdir(exist_ok=False)
                env = environment(task, physical_gpu=gpu, triton_cache=triton)
                self._update(task_id, triton_cache=str(triton))
            argv = command(task, device="cuda:0")
            log = log_path(task)
            self._update(task_id, status="running", argv=argv, log_path=str(log),
                         environment={"CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"]}, started=time.time())
            with log.open("x", encoding="utf-8", newline="\n") as stream:
                options = {"cwd": str(ROOT), "env": env, "stdout": stream,
                           "stderr": subprocess.STDOUT, "start_new_session": True}
                if lease is not None:
                    options["pass_fds"] = (lease.fileno(),)
                proc = subprocess.Popen(argv, **options)
                self._update(task_id, child_pid=proc.pid)
                while proc.poll() is None:
                    time.sleep(5)
                    if proc.poll() is not None or gpu is None:
                        continue
                    _uuid, _memory, owners = owner._nvidia_gpu_info(gpu)
                    foreign = owner.foreign_owner_pids(owners, owner._process_tree(proc.pid))
                    if foreign:
                        self._terminate_owned(proc)
                        self._update(task_id, status="interrupted_foreign_gpu_owner",
                                     foreign_owner_pids=foreign, returncode=proc.returncode)
                        return False
            rc = int(proc.returncode)
            valid = rc == 0 and validate_result(task)
            self._update(task_id, status="passed" if valid else "failed", returncode=rc,
                         artifact_valid=bool(valid), finished=time.time())
            return bool(valid)
        except BaseException as exc:
            self._terminate_owned(proc)
            self._update(task_id, status="failed", error=f"{type(exc).__name__}: {exc}",
                         returncode=(proc.returncode if proc is not None else None), finished=time.time())
            return False
        finally:
            if proc is not None and proc.poll() is None:
                self._terminate_owned(proc)
            if lease is not None:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close()

    def run(self):
        by_id = {t["task_id"]: t for t in self.tasks}
        failed = False
        for group in task_groups():
            batch = [by_id[task_id] for task_id in group]
            for task in batch:
                if any(not validate_result(by_id[dep]) for dep in task["deps"]):
                    self._update(task["task_id"], status="blocked_invalid_dependency")
                    failed = True
            if failed:
                break
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(MAX_PARALLEL, len(batch))) as pool:
                results = list(pool.map(self._execute, batch))
            if not all(results):
                failed = True; break
        self.state.update(status="failed" if failed else "complete", finished=time.time())
        self._save()
        return 1 if failed else 0


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"max_parallel": MAX_PARALLEL, "gpu_policy": "exclusive, <=512MiB, no compute owner",
                          "tasks": [{**task, "argv": command(task)} for task in task_plan()],
                          "groups": task_groups(), "gt_last": True}, indent=2), flush=True)
        return 0
    return SelectiveGeodesicQueue().run()


if __name__ == "__main__":
    raise SystemExit(main())
