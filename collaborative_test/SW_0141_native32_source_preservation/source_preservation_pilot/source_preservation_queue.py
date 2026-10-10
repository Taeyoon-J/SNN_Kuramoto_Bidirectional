"""Owner-aware, no-retry queue draft for the private SW0139 source-Q pilot.

This module is preparation only. It has not been admitted, deployed or run.
The queue uses immutable stage outputs, waits on child processes (not polling
observations), and never kills a process outside its own child process group.
"""
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
except ImportError:  # Local Windows supports planning/tests only.
    fcntl = None

# A deployed copy lives under the checkout's private runtime directory; an
# offline preparation copy lives beside patch_v2_sw. Resolve both layouts and
# pass this explicit root to every stage child.
_source_path = Path(__file__).resolve()
_repo_candidates = [parent for parent in _source_path.parents
                    if (parent / "collaborative_test").is_dir()]
_repo_candidates += [parent / "patch_v2_sw" for parent in _source_path.parents
                     if (parent / "patch_v2_sw" / "collaborative_test").is_dir()]
if _repo_candidates:
    os.environ.setdefault("PATCH_V2_REPO", str(_repo_candidates[0]))

import pilot

ROOT = pilot.ROOT
HERE = pilot.HERE
ARCHIVE = HERE / "queue_runtime"
STATE = ARCHIVE / "queue_state.json"
LOCK = ARCHIVE / "queue.lock"
GPU_LEASE_DIR = Path("/tmp/kevinswk_sw0113_gpu_leases")
GPU_MEMORY_LIMIT_MIB = 512
MAX_PARALLEL = 2
PYTHON = sys.executable

try:
    from collaborative_test.SW_0123_adaptive_temporal_assignment import dispatcher as owner
except ImportError:
    owner = None


def task_plan():
    """Fixed stage DAG; the two caches and each arm pair may run in parallel."""
    return [
        {"task_id": "source_q_cache_s0", "stage": "cache", "seed": 0, "depends_on": []},
        {"task_id": "source_q_cache_s1", "stage": "cache", "seed": 1, "depends_on": []},
        {"task_id": "anchor_lambda_s0", "stage": "calibrate", "depends_on": ["source_q_cache_s0"]},
        {"task_id": "preflight_c_only_s1", "stage": "preflight", "arm": "c_only",
         "depends_on": ["source_q_cache_s1", "anchor_lambda_s0"]},
        {"task_id": "preflight_c_plus_anchor_s1", "stage": "preflight", "arm": "c_plus_anchor",
         "depends_on": ["source_q_cache_s1", "anchor_lambda_s0"]},
        {"task_id": "train_c_only_s1", "stage": "train", "arm": "c_only",
         "depends_on": ["preflight_c_only_s1"]},
        {"task_id": "train_c_plus_anchor_s1", "stage": "train", "arm": "c_plus_anchor",
         "depends_on": ["preflight_c_plus_anchor_s1"]},
        {"task_id": "predict_c_only_s1", "stage": "predict", "arm": "c_only",
         "depends_on": ["train_c_only_s1"]},
        {"task_id": "predict_c_plus_anchor_s1", "stage": "predict", "arm": "c_plus_anchor",
         "depends_on": ["train_c_plus_anchor_s1"]},
        {"task_id": "score_seed1", "stage": "score",
         "depends_on": ["predict_c_only_s1", "predict_c_plus_anchor_s1"]},
    ]


def stage_groups():
    return (("cache",), ("calibrate",), ("preflight",), ("train",), ("predict",), ("score",))


def output_paths(task):
    stage = task["stage"]
    if stage == "cache":
        return (pilot.CACHE_ROOT / f"source{task['seed']}_n{pilot.CAL_N if task['seed'] == 0 else pilot.TRAIN_N}",)
    if stage == "calibrate":
        return (pilot.LAMBDA_A_PATH,)
    if stage == "preflight":
        return (pilot.TRAIN_ROOT / f"preflight_{task['arm']}_seed1.json",)
    if stage == "train":
        return (pilot.TRAIN_ROOT / f"{task['arm']}_seed1",)
    if stage == "predict":
        return (pilot.RESULT_ROOT / task["arm"],)
    if stage == "score":
        return (pilot.RESULT_ROOT / "evaluation.json",)
    raise ValueError(f"unknown pilot stage {stage}")


def command(task):
    exe = PYTHON
    driver = str(HERE / "pilot.py")
    stage = task["stage"]
    if stage == "cache":
        return [exe, driver, "cache", "--seed", str(task["seed"]), "--device", "cuda:0",
                "--output", str(output_paths(task)[0])]
    if stage == "calibrate":
        return [exe, driver, "calibrate", "--source0-cache", str(output_paths(
            {"stage": "cache", "seed": 0})[0]), "--device", "cuda:0",
                "--output", str(pilot.LAMBDA_A_PATH)]
    if stage == "preflight":
        argv = [exe, driver, "preflight", "--arm", task["arm"], "--device", "cuda:0",
                "--lambda-a", str(pilot.LAMBDA_A_PATH), "--output-root", str(pilot.TRAIN_ROOT)]
        if task["arm"] == "c_plus_anchor":
            argv += ["--teacher-cache", str(output_paths({"stage": "cache", "seed": 1})[0])]
        return argv
    if stage == "train":
        argv = [exe, driver, "train", "--arm", task["arm"], "--device", "cuda:0",
                "--lambda-a", str(pilot.LAMBDA_A_PATH), "--output-root", str(pilot.TRAIN_ROOT)]
        if task["arm"] == "c_plus_anchor":
            argv += ["--teacher-cache", str(output_paths({"stage": "cache", "seed": 1})[0])]
        return argv
    if stage == "predict":
        return [exe, driver, "predict", "--arm", task["arm"], "--device", "cuda:0",
                "--training-root", str(pilot.TRAIN_ROOT), "--output-root", str(pilot.RESULT_ROOT)]
    if stage == "score":
        return [exe, driver, "score", "--prediction-root", str(pilot.RESULT_ROOT),
                "--source-root", str(pilot.sw137.OUTPUT_ROOT),
                "--old-c-root", str(pilot.sw139_eval.EVALUATION_ROOT),
                "--slot-root", str(ROOT / "trained_models/SW0092_slot_our70000_eval"),
                "--dataset", str(pilot.sw139_eval.DATASET),
                "--output", str(pilot.RESULT_ROOT / "evaluation.json")]
    raise ValueError(f"unknown pilot stage {stage}")


def environment_for(task, physical_gpu=None, *, triton_cache=None):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env["PATCH_V2_REPO"] = str(ROOT)
    if task["stage"] == "score":
        env["CUDA_VISIBLE_DEVICES"] = ""
        env.pop("TRITON_CACHE_DIR", None)
    else:
        if physical_gpu is None:
            raise ValueError("GPU stages need an exclusively reserved physical GPU")
        env["CUDA_VISIBLE_DEVICES"] = str(int(physical_gpu))
        if triton_cache is not None:
            env["TRITON_CACHE_DIR"] = str(triton_cache)
    return env


def validate_result(task):
    """Validate terminal stage metadata; never treat partial outputs as reusable."""
    try:
        stage = task["stage"]
        if stage == "cache":
            directory = output_paths(task)[0]
            record = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            n = pilot.CAL_N if task["seed"] == 0 else pilot.TRAIN_N
            expected = n * 1024 * 1024 * 4
            return (record.get("status") == "complete" and record.get("seed") == task["seed"]
                    and record.get("q_shape") == [n, 1024, 1024]
                    and record.get("ground_truth_used") is False
                    and record.get("implementation_fingerprint") == pilot.implementation_fingerprint()
                    and (directory / "teacher_q.f32").is_file()
                    and (directory / "teacher_q.f32").stat().st_size == expected)
        if stage == "calibrate":
            return pilot._load_lambda_a(pilot.LAMBDA_A_PATH).get("status") == "calibrated_local_unregistered"
        if stage == "preflight":
            row = json.loads(output_paths(task)[0].read_text(encoding="utf-8"))
            return (row.get("status") == "disposable_preflight_complete_local_unregistered"
                    and row.get("arm") == task["arm"] and row.get("seed") == 1
                    and row.get("ground_truth_used") is False
                    and row.get("implementation_fingerprint") == pilot.implementation_fingerprint())
        if stage == "train":
            row = pilot.validate_pilot_checkpoint(output_paths(task)[0], arm=task["arm"], seed=1)
            return row["manifest"].get("status") == "training_complete_local_unregistered"
        if stage == "predict":
            row = pilot.validate_own_prediction(task["arm"], prediction_root=pilot.RESULT_ROOT)
            return row["training"]["manifest"].get("arm") == task["arm"]
        if stage == "score":
            record = json.loads(output_paths(task)[0].read_text(encoding="utf-8"))
            return (record.get("status") == "complete_local_unregistered"
                    and record.get("ground_truth_used_for_prediction_or_training") is False
                    and record.get("experiment") == pilot.PILOT_NAME)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError):
        return False
    return False


class PilotQueue:
    def __init__(self):
        if fcntl is None or owner is None:
            raise RuntimeError("the owner-aware source-Q queue requires Linux flock and SW0123 owner helpers")
        if STATE.exists() or LOCK.exists():
            raise FileExistsError("preserve existing source-preservation queue state/lock; no restart")
        artifacts = [p for task in task_plan() for p in output_paths(task) if p.exists()]
        if artifacts:
            raise FileExistsError("preserve existing pilot outputs: " + ", ".join(map(str, artifacts)))
        ARCHIVE.mkdir(parents=True, exist_ok=False)
        self.lock_file = LOCK.open("x+")
        try:
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.lock_file.close()
            raise
        self.guard = threading.RLock()
        self.tasks = task_plan()
        self.state = {"status": "running", "pilot": pilot.PILOT_NAME,
                      "registered": False, "supervisor_pid": os.getpid(),
                      "started": time.time(), "pilot_sha256": pilot.sha256_file(HERE / "pilot.py"),
                      "queue_sha256": pilot.sha256_file(HERE / "source_preservation_queue.py"),
                      "owner_dispatcher_sha256": pilot.sha256_file(owner.__file__),
                      "implementation_fingerprint": pilot.implementation_fingerprint(),
                      "task_count": len(self.tasks),
                      "tasks": {t["task_id"]: {"task": t, "status": "queued"} for t in self.tasks}}
        self._save()

    def _save(self):
        with self.guard:
            tmp = STATE.with_name(STATE.name + f".{os.getpid()}.tmp")
            with tmp.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(self.state, stream, indent=2, allow_nan=False); stream.write("\n")
                stream.flush(); os.fsync(stream.fileno())
            os.replace(tmp, STATE)

    def _update(self, task_id, **fields):
        with self.guard:
            self.state["tasks"][task_id].update(fields)
            self._save()

    def _reserve(self, task):
        while True:
            for gpu in range(4):
                lease_path = GPU_LEASE_DIR / f"gpu{gpu}.lock"
                lease_path.parent.mkdir(parents=True, exist_ok=True)
                lease = lease_path.open("a+")
                try:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lease.close(); continue
                try:
                    uuid, memory, owners = owner._nvidia_gpu_info(gpu)
                    if not owners and memory <= GPU_MEMORY_LIMIT_MIB:
                        self._update(task["task_id"], status="reserved", gpu=gpu,
                                     gpu_uuid=uuid, memory_before_mib=memory,
                                     lease_path=str(lease_path), reserved=time.time())
                        return gpu, lease, uuid, memory
                except BaseException:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); raise
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close()
            self._update(task["task_id"], status="waiting_for_exclusive_gpu", last_wait=time.time())
            time.sleep(5)

    @staticmethod
    def _terminate(proc):
        if proc is None or proc.poll() is not None: return
        try: os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError: return
        try: proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try: os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            proc.wait()

    def _execute(self, task):
        task_id = task["task_id"]
        proc = lease = None
        log_path = ARCHIVE / f"{task_id}.log"
        try:
            gpu = uuid = memory = triton = None
            if task["stage"] != "score":
                gpu, lease, uuid, memory = self._reserve(task)
                triton = Path(f"/tmp/kevinswk_sourceq_{task_id}_{int(time.time()*1000)}")
                triton.mkdir(parents=True, exist_ok=False)
            argv = command(task)
            env = environment_for(task, gpu, triton_cache=triton)
            if log_path.exists(): raise FileExistsError(f"preserve existing queue log: {log_path}")
            with log_path.open("x", encoding="utf-8") as log:
                kwargs = {"cwd": ROOT, "env": env, "stdin": subprocess.DEVNULL,
                          "stdout": log, "stderr": subprocess.STDOUT,
                          "start_new_session": True}
                if lease is not None: kwargs["pass_fds"] = (lease.fileno(),)
                proc = subprocess.Popen(argv, **kwargs)
                self._update(task_id, status="running", child_pid=proc.pid, argv=argv,
                             log=str(log_path), gpu=gpu, gpu_uuid=uuid,
                             memory_before_mib=memory, lease_inherited=lease is not None,
                             triton_cache=str(triton) if triton else None, started=time.time())
                foreign = []
                while proc.poll() is None:
                    time.sleep(5)
                    if proc.poll() is not None: break
                    if gpu is not None:
                        _uuid, _memory, owners = owner._nvidia_gpu_info(gpu)
                        foreign = owner.foreign_owner_pids(owners, owner._process_tree(proc.pid))
                        if foreign:
                            self._update(task_id, status="interrupted_foreign_gpu_owner",
                                         foreign_owner_pids=foreign, observed=time.time())
                            self._terminate(proc)
                            break
                rc = proc.wait()
                valid = rc == 0 and not foreign and validate_result(task)
                terminal = "passed" if valid else ("interrupted_foreign_gpu_owner" if foreign else "failed")
                self._update(task_id, status=terminal, returncode=rc,
                             artifact_valid=bool(valid), finished=time.time())
                return terminal
        except BaseException as exc:
            self._terminate(proc)
            self._update(task_id, status="failed", error=repr(exc), finished=time.time())
            return "failed"
        finally:
            if lease is not None:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close()

    def run(self):
        rows = {t["task_id"]: t for t in self.tasks}
        for group in stage_groups():
            selected = [t for t in self.tasks if t["stage"] in group]
            ready, blocked = [], []
            for task in selected:
                states = [self.state["tasks"][dep]["status"] for dep in task["depends_on"]]
                if all(status == "passed" for status in states): ready.append(task)
                else:
                    blocked.append((task, states))
            for task, states in blocked:
                self._update(task["task_id"], status="blocked_dependency", dependency_status=states)
            if ready:
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(MAX_PARALLEL, len(ready))) as pool:
                    futures = [pool.submit(self._execute, task) for task in ready]
                    for future in concurrent.futures.as_completed(futures): future.result()
        terminal = "complete" if all(r["status"] == "passed" for r in self.state["tasks"].values()) else "failed_or_blocked"
        self.state.update(status=terminal, finished=time.time(), training_registered=False)
        self._save()
        fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN); self.lock_file.close()
        return self.state


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print fixed stage plan without creating files or querying GPUs")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"registered": False, "tasks": task_plan(),
                          "groups": stage_groups(), "max_parallel_gpu": MAX_PARALLEL}, indent=2))
        return 0
    state = PilotQueue().run()
    print(json.dumps({"status": state["status"], "state": str(STATE),
                      "supervisor_pid": state["supervisor_pid"]}, allow_nan=False), flush=True)
    return 0 if state["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
