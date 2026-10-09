"""Owner-aware, no-retry parallel dispatcher for three SW0126 TRAIN screens."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time

import numpy as np
try:
    import fcntl
except ImportError:  # Local Windows contract tests do not execute the queue.
    fcntl = None

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]
from collaborative_test.SW_0126_history_event_binding import screen

GPU_LEASE_DIR = pathlib.Path("/tmp/kevinswk_sw0113_gpu_leases")
RUNTIME_DIR = pathlib.Path("/tmp/kevinswk_sw0126_screen_queue")
QUEUE_LOCK = RUNTIME_DIR / "queue.lock"
STATE_PATH = RUNTIME_DIR / "state.json"
MAX_GPU_MEMORY_MIB = 512
PYTHON = "/Data0/kevinswk/envs/snn/bin/python"


def task_plan():
    return [{"task_id": f"sw0126_screen_seed{s}", "seed": s,
             "output": str(HERE / "results_archive" / f"screen_seed{s}_20261009.json")}
            for s in (0, 1, 2)]


def gpu_is_exclusive(memory_mib, owner_pids):
    return int(memory_mib) <= MAX_GPU_MEMORY_MIB and not owner_pids


def validate_screen_record(record, task, ids, source_sha, fingerprint):
    batches = record.get("batches", [])
    if len(batches) != 4:
        return False
    for row in batches:
        arms = row.get("arms", {})
        event = arms.get("history_event", {}).get("gradient_credit", {})
        gate = arms.get("gate_only", {}).get("gradient_credit", {})
        try:
            e_required = [event[k] for k in ("assignment_norm", "beta_norm",
                                             "membrane_tau_m_norm", "dendritic_norm")]
            beta = event["beta_component_abs"]
            g_assignment = gate["assignment_norm"]
            g_upstream = [gate[k] for k in ("beta_norm", "membrane_tau_m_norm", "dendritic_norm")]
            mixed = np.asarray(
                row["activity"]["mixed_unit_pass_by_image_component"], dtype=bool)
            assignment_error = row["arms"]["history_event"]["assignment_patch_mass_max_error"]
            trace_variance = row["centered_event_trace_variance_by_component"]
        except (KeyError, TypeError):
            return False
        if (mixed.shape != (16, 4) or not mixed.all()
                or len(beta) != 4 or not all(np.isfinite(x) and x > 0 for x in e_required + beta)
                or not np.isfinite(g_assignment) or g_assignment <= 0
                or any(value != 0 for value in g_upstream)):
            return False
        if (not np.isfinite(assignment_error) or assignment_error > 1e-6
                or len(trace_variance) != 4
                or not all(np.isfinite(value) for value in trace_variance)):
            return False
    seed = int(task["seed"])
    expected_ids_sha = hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
    thresholds = {"component_occupancy": [0.02, 0.98],
                  "minimum_mixed_unit_fraction_per_image_component": 0.10}
    return (record.get("status") == "passed_feasibility_screen"
            and record.get("experiment") == "SW0126" and record.get("seed") == seed
            and record.get("optimizer_updates") == 0
            and record.get("ground_truth_used_for_prediction_or_training") is False
            and record.get("implementation_fingerprint") == fingerprint
            and record.get("source_core_sha256") == source_sha
            and record.get("source_ids_sha256") == expected_ids_sha
            and record.get("matched_train_ids") == [int(v) for v in ids[:64]]
            and record.get("activity_thresholds") == thresholds
            and record.get("aggregate_activity", {}).get("pass") is True
            and batches[0].get("native_source_rollout_exact_first_batch") == {
                "theta": True, "component_membrane": True, "component_spikes": True}
            and all(row.get("theta_carrier_gate_exact") == {
                "theta": True, "carrier": True, "gate": True} for row in batches))


def valid_result(task):
    path = pathlib.Path(task["output"])
    if not path.is_file():
        return False
    try:
        ids = screen.sw125.source_contract(int(task["seed"]))[3]
        record = json.loads(path.read_text(encoding="utf-8"))
        return validate_screen_record(record, task, ids,
            screen.base.EXPECTED_SOURCE_SHAS[int(task["seed"])],
            screen.implementation_fingerprint())
    except (OSError, ValueError, TypeError, KeyError):
        return False


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
    foreign = []
    for pid in set(map(int, owners)) - set(map(int, owned_tree)):
        try:
            pathlib.Path(f"/proc/{pid}").stat()
        except FileNotFoundError:
            if os.name == "nt":
                foreign.append(pid)
        except OSError:
            foreign.append(pid)
        else:
            foreign.append(pid)
    return sorted(foreign)


class ScreenQueue:
    def __init__(self):
        if fcntl is None:
            raise RuntimeError("SW0126 screen queue requires Linux flock")
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        self.queue_lock = QUEUE_LOCK.open("a+")
        try:
            fcntl.flock(self.queue_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.queue_lock.close()
            raise RuntimeError("another SW0126 queue owns the lock") from exc
        if STATE_PATH.exists():
            self.queue_lock.close()
            raise FileExistsError(f"preserving existing queue state: {STATE_PATH}")
        self.lock = threading.RLock()
        self.tasks = task_plan()
        self.state = {"experiment": "SW0126", "status": "running",
                      "supervisor_pid": os.getpid(), "tasks": {
                          t["task_id"]: {"task": t, "status": "queued"} for t in self.tasks}}
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
    def _stop_own_child(proc):
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

    def _run_task(self, task):
        task_id = task["task_id"]
        if valid_result(task):
            self._update(task_id, status="reused_verified")
            return
        output = pathlib.Path(task["output"])
        if output.exists():
            self._update(task_id, status="failed_existing_output", reason="preserved")
            return
        gpu, uuid, memory, lease = self._acquire(task)
        proc = None
        log = RUNTIME_DIR / f"{task_id}.log"
        try:
            if log.exists():
                raise FileExistsError(f"preserving existing log {log}")
            triton = pathlib.Path(f"/tmp/kevinswk_sw0126_triton_{task_id}")
            triton.mkdir(mode=0o700, parents=True, exist_ok=False)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", TRITON_CACHE_DIR=str(triton))
            argv = [PYTHON, "-m", "collaborative_test.SW_0126_history_event_binding.screen",
                    "--seed", str(task["seed"]), "--device", "cuda:0", "--output", str(output)]
            with log.open("x", encoding="utf-8") as stream:
                proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                    pass_fds=(lease.fileno(),))
                self._update(task_id, status="running", child_pid=proc.pid, gpu=gpu,
                    gpu_uuid=uuid, memory_used_before_mib=memory, inherited_lease_fd=True,
                    triton_cache_dir=str(triton), argv=argv, log=str(log))
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
                        self._stop_own_child(proc)
                        break
                rc = proc.wait()
                accepted = rc == 0 and not foreign and valid_result(task)
                self._update(task_id, status="passed" if accepted else "failed",
                    returncode=rc, artifact_valid=valid_result(task), finished=time.time())
        except BaseException as exc:
            if proc is not None and proc.poll() is None:
                self._stop_own_child(proc)
            self._update(task_id, status="failed", error=repr(exc), finished=time.time())
        finally:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
            lease.close()

    def run(self):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                futures = [pool.submit(self._run_task, task) for task in self.tasks]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
            with self.lock:
                statuses = [row["status"] for row in self.state["tasks"].values()]
                self.state.update(status="complete" if all(
                    s in ("passed", "reused_verified") for s in statuses)
                    else "completed_with_failures", finished=time.time())
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
        ScreenQueue().run()
