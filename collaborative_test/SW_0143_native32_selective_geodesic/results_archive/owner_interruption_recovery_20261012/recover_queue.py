"""Private, create-once recovery wrapper for the interrupted SW0143 queue."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
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

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("PATCH_V2_REPO", HERE.parents[1] / "patch_v2_sw")).resolve()
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0143_native32_selective_geodesic import geodesic_queue as frozen
from collaborative_test.SW_0143_native32_selective_geodesic import run

QUEUE_ARCHIVE = ROOT / "collaborative_test/SW_0143_native32_selective_geodesic/results_archive"
ORIGINAL_STATE = QUEUE_ARCHIVE / "selective_geodesic_queue_state.json"
ORIGINAL_LOCK = QUEUE_ARCHIVE / "selective_geodesic_queue.lock"
RETIREMENT_DIR = QUEUE_ARCHIVE / "owner_interruption_20261012"
RETIREMENT_RECEIPT = RETIREMENT_DIR / "retirement_receipt.json"
ORIGINAL_STATE_COPY = RETIREMENT_DIR / "original_queue_state.json"
RECOVERY_ROOT = QUEUE_ARCHIVE / "recovery_v1_20261012"


def _pid_alive(pid):
    return Path(f"/proc/{int(pid)}").exists()


def _proc_cmdlines():
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        raise RuntimeError("recovery process audit requires readable /proc")
    result = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except FileNotFoundError:
            continue
        except PermissionError as exc:
            raise RuntimeError(f"cannot verify process argv for pid {entry.name}") from exc
        if raw:
            result.append((int(entry.name), raw.replace(b"\0", b" ").decode(errors="replace")))
    return result


def _is_live_original_queue(pid, argv):
    if int(pid) == os.getpid():
        return False
    normalized = str(argv).replace("\\", "/").lower()
    return ("/collaborative_test/sw_0143_native32_selective_geodesic/run.py" in normalized
            or "/collaborative_test/sw_0143_native32_selective_geodesic/geodesic_queue.py" in normalized
            or "collaborative_test.sw_0143_native32_selective_geodesic." in normalized)


def _decode_state_copy(original_bytes, copied_bytes):
    """Archive copies may differ in JSON whitespace; state content must match."""
    original = json.loads(original_bytes.decode("utf-8"))
    copied = json.loads(copied_bytes.decode("utf-8"))
    if copied != original:
        raise ValueError("archived original queue state differs semantically")
    return original


def validate_retirement(original_state_bytes, state, receipt, *, pid_alive, proc_cmdlines):
    """Fail closed unless root retired the old supervisor and all its children."""
    digest = __import__("hashlib").sha256(original_state_bytes).hexdigest()
    if state.get("experiment") != "SW0143_native32_selective_geodesic" or state.get("status") != "running":
        raise ValueError("original queue is not the recorded stalled SW0143 run")
    if state.get("implementation_fingerprint") != run.implementation_fingerprint():
        raise ValueError("original queue fingerprint differs from current frozen SW0143 dependencies")
    if state.get("queue_sha256") != run.sha256_file(Path(frozen.__file__)):
        raise ValueError("original queue implementation SHA changed")
    supervisor = int(state.get("supervisor_pid", 0))
    if not supervisor or pid_alive(supervisor):
        raise RuntimeError("original SW0143 supervisor must be retired before recovery")
    expected = {
        "status": "child_free_stalled_supervisor_retired",
        "supervisor_pid": supervisor,
        "verified_scientific_children_absent": True,
        "original_queue_state_preserved": True,
        "interrupted_task": "preflight_s2",
        "interrupted_child_pid": 378677,
        "foreign_owner_pids": [378607],
        "foreign_processes_modified": False,
        "scientific_files_modified": False,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("root retirement receipt does not match the registered SW0143 interruption")
    if receipt.get("original_queue_state_sha256") not in (None, digest):
        raise ValueError("retirement receipt binds a different original queue state")
    tasks = state.get("tasks")
    if not isinstance(tasks, dict) or "preflight_s2" not in tasks:
        raise ValueError("original SW0143 task state missing")
    interrupted = tasks["preflight_s2"]
    if (interrupted.get("status") != "interrupted_foreign_gpu_owner"
            or interrupted.get("child_pid") != 378677
            or interrupted.get("foreign_owner_pids") != [378607]
            or interrupted.get("returncode") != -15):
        raise ValueError("the recorded preflight_s2 interruption does not match the fixed recovery scope")
    for task_id, row in tasks.items():
        child = row.get("child_pid")
        if child and pid_alive(child):
            raise RuntimeError(f"original SW0143 child still alive: {task_id} pid={child}")
    live = [(pid, argv) for pid, argv in proc_cmdlines() if _is_live_original_queue(pid, argv)]
    if live:
        raise RuntimeError(f"SW0143 process argv still live: {live[0][0]} {live[0][1]}")
    return digest


def build_recovery_plan(state, *, artifact_exists, artifact_valid):
    """Return full DAG plus already-valid artifacts, refusing partial overwrites."""
    tasks = frozen.task_plan()
    original = state.get("tasks", {})
    if set(original) != {task["task_id"] for task in tasks}:
        raise ValueError("original queue task set differs from frozen SW0143 DAG")
    preserved, missing = [], []
    for task in tasks:
        task_id = task["task_id"]
        row = original[task_id]
        exists = artifact_exists(task)
        if exists:
            if not artifact_valid(task):
                raise FileExistsError(f"existing SW0143 artifact is invalid; preserve it: {task_id}")
            preserved.append(task_id)
            continue
        if row.get("status") == "passed":
            raise FileNotFoundError(f"original queue says passed but its artifact is missing: {task_id}")
        if row.get("status") == "interrupted_foreign_gpu_owner":
            if task_id != "preflight_s2" or row.get("foreign_owner_pids") != [378607]:
                raise ValueError(f"refusing unregistered interrupted task: {task_id}")
        elif row.get("status") not in {"queued", "waiting_for_exclusive_gpu"}:
            raise ValueError(f"refusing to retry task with unregistered terminal state: {task_id}")
        missing.append(task_id)
    return {"tasks": tasks, "preserved": preserved, "missing": missing}


def load_and_validate_inputs(*, original_state=ORIGINAL_STATE,
                             retirement_receipt=RETIREMENT_RECEIPT,
                             original_state_copy=ORIGINAL_STATE_COPY,
                             pid_alive=_pid_alive, proc_cmdlines=_proc_cmdlines):
    state_bytes = Path(original_state).read_bytes()
    copied = Path(original_state_copy).read_bytes()
    state = _decode_state_copy(state_bytes, copied)
    receipt = json.loads(Path(retirement_receipt).read_text(encoding="utf-8"))
    digest = validate_retirement(state_bytes, state, receipt,
                                 pid_alive=pid_alive, proc_cmdlines=proc_cmdlines)
    return state, receipt, digest


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class RecoveryQueue(frozen.SelectiveGeodesicQueue):
    def __init__(self, recovery_state, state_digest, plan):
        if fcntl is None:
            raise RuntimeError("SW0143 recovery requires Linux flock")
        if RECOVERY_ROOT.exists():
            raise FileExistsError(f"preserve existing SW0143 recovery output: {RECOVERY_ROOT}")
        # Keep the original experiment lock open for the full recovery lifetime.
        self.original_lock = ORIGINAL_LOCK.open("r+")
        try:
            fcntl.flock(self.original_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.original_lock.close()
            raise RuntimeError("original SW0143 experiment lock is still owned") from exc
        try:
            RECOVERY_ROOT.mkdir(parents=True)
            self.log_root = RECOVERY_ROOT / "logs"
            self.log_root.mkdir()
            self.state_path = RECOVERY_ROOT / "recovery_state.json"
            self.recovery_lock_path = RECOVERY_ROOT / "recovery.lock"
            self.queue_lock = self.recovery_lock_path.open("x+")
            fcntl.flock(self.queue_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            fcntl.flock(self.original_lock.fileno(), fcntl.LOCK_UN)
            self.original_lock.close()
            raise
        self.tasks = plan["tasks"]
        self.preserved = set(plan["preserved"])
        self.stop_event = threading.Event()
        self.guard = threading.RLock()
        self.state = {
            "experiment": "SW0143_native32_selective_geodesic_recovery_v1",
            "status": "running", "supervisor_pid": os.getpid(), "started": time.time(),
            "original_state_sha256": state_digest,
            "implementation_fingerprint": run.implementation_fingerprint(),
            "queue_sha256": run.sha256_file(Path(frozen.__file__)),
            "recovery_helper_sha256": run.sha256_file(Path(__file__)),
            "owner_dispatcher_sha256": run.sha256_file(Path(frozen.owner.__file__)),
            "preserved_artifacts": sorted(self.preserved),
            "tasks": {task["task_id"]: {"task": task,
                       "status": "preserved_valid_artifact" if task["task_id"] in self.preserved else "queued"}
                      for task in self.tasks},
        }
        self._save()

    def _save(self):
        with self.guard:
            _atomic_json(self.state_path, self.state)

    def _update(self, task_id, **values):
        with self.guard:
            self.state["tasks"][task_id].update(values)
            _atomic_json(self.state_path, self.state)

    def _reserve(self, task_id):
        while True:
            if self.stop_event.is_set():
                raise RuntimeError("recovery cancelled while waiting for an exclusive GPU")
            for gpu in range(4):
                if self.stop_event.is_set():
                    raise RuntimeError("recovery cancelled before GPU reservation")
                lease_path = frozen.GPU_LEASES / f"gpu{gpu}.lock"
                lease_path.parent.mkdir(parents=True, exist_ok=True)
                lease = lease_path.open("a+")
                try:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lease.close()
                    continue
                try:
                    uuid, memory, owners = frozen.owner._nvidia_gpu_info(gpu)
                    if frozen.gpu_candidate(memory, owners):
                        if self.stop_event.is_set():
                            raise RuntimeError("recovery cancelled after GPU lease acquisition")
                        self._update(task_id, status="reserved", physical_gpu=gpu,
                                     gpu_uuid=uuid, memory_before_mib=memory,
                                     lease_path=str(lease_path))
                        return gpu, lease, uuid
                except BaseException:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                    lease.close()
                    raise
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                lease.close()
            self._update(task_id, status="waiting_for_exclusive_gpu", last_wait=time.time())
            time.sleep(5)

    def _execute(self, task):
        task_id = task["task_id"]
        if task_id in self.preserved:
            self._update(task_id, status="preserved_valid_artifact", artifact_valid=True)
            return True
        if frozen.output_path(task).exists():
            if frozen.validate_result(task):
                self.preserved.add(task_id)
                self._update(task_id, status="preserved_valid_artifact", artifact_valid=True)
                return True
            self.stop_event.set()
            self._update(task_id, status="failed_existing_invalid_artifact")
            return False
        gpu = lease = proc = None
        try:
            if self.stop_event.is_set():
                raise RuntimeError("recovery stopped before task launch")
            if task["stage"] == "score":
                env = frozen.environment(task)
            else:
                gpu, lease, uuid = self._reserve(task_id)
                if self.stop_event.is_set():
                    raise RuntimeError("recovery stopped after GPU reservation")
                triton = Path("/tmp") / f"kevinswk_sw0143_recovery_{os.getpid()}_{task_id}_{int(time.time()*1000)}"
                triton.mkdir(exist_ok=False)
                env = frozen.environment(task, physical_gpu=gpu, triton_cache=triton)
                self._update(task_id, triton_cache=str(triton))
            argv = frozen.command(task, device="cuda:0")
            log = self.log_root / f"{task_id}.log"
            self._update(task_id, status="running", argv=argv, log_path=str(log),
                         environment={"CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"]}, started=time.time())
            with log.open("x", encoding="utf-8", newline="\n") as stream:
                if self.stop_event.is_set():
                    raise RuntimeError("recovery stopped before subprocess creation")
                options = {"cwd": str(frozen.ROOT), "env": env, "stdout": stream,
                           "stderr": subprocess.STDOUT, "start_new_session": True}
                if lease is not None:
                    options["pass_fds"] = (lease.fileno(),)
                proc = subprocess.Popen(argv, **options)
                self._update(task_id, child_pid=proc.pid)
                while proc.poll() is None:
                    time.sleep(5)
                    if proc.poll() is not None or gpu is None:
                        continue
                    _uuid, _memory, owners = frozen.owner._nvidia_gpu_info(gpu)
                    foreign = frozen.owner.foreign_owner_pids(owners, frozen.owner._process_tree(proc.pid))
                    if foreign:
                        frozen.SelectiveGeodesicQueue._terminate_owned(proc)
                        self._update(task_id, status="interrupted_foreign_gpu_owner",
                                     foreign_owner_pids=foreign, returncode=proc.returncode)
                        self.stop_event.set()
                        return False
            rc = int(proc.returncode)
            valid = rc == 0 and frozen.validate_result(task)
            self._update(task_id, status="passed" if valid else "failed", returncode=rc,
                         artifact_valid=bool(valid), finished=time.time())
            if not valid:
                self.stop_event.set()
            return bool(valid)
        except BaseException as exc:
            frozen.SelectiveGeodesicQueue._terminate_owned(proc)
            self._update(task_id, status="failed", error=f"{type(exc).__name__}: {exc}",
                         returncode=(proc.returncode if proc is not None else None), finished=time.time())
            self.stop_event.set()
            return False
        finally:
            if proc is not None and proc.poll() is None:
                frozen.SelectiveGeodesicQueue._terminate_owned(proc)
            if lease is not None:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                lease.close()

    def run(self):
        try:
            return super().run()
        finally:
            if not self.queue_lock.closed:
                self.queue_lock.close()
            if not self.original_lock.closed:
                self.original_lock.close()


def prepare_recovery(*, state_path=ORIGINAL_STATE, receipt_path=RETIREMENT_RECEIPT,
                     state_copy_path=ORIGINAL_STATE_COPY, recovery_root=RECOVERY_ROOT,
                     pid_alive=_pid_alive, proc_cmdlines=_proc_cmdlines,
                     artifact_exists=None, artifact_valid=None):
    global RECOVERY_ROOT
    if Path(recovery_root).exists():
        raise FileExistsError(f"preserve existing recovery path: {recovery_root}")
    state_bytes = Path(state_path).read_bytes()
    copied = Path(state_copy_path).read_bytes()
    state = _decode_state_copy(state_bytes, copied)
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    state_sha = validate_retirement(state_bytes, state, receipt,
                                    pid_alive=pid_alive, proc_cmdlines=proc_cmdlines)
    exists = artifact_exists or (lambda task: frozen.output_path(task).exists())
    valid = artifact_valid or frozen.validate_result
    plan = build_recovery_plan(state, artifact_exists=exists, artifact_valid=valid)
    RECOVERY_ROOT = Path(recovery_root)
    return state, receipt, state_sha, plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    state, _receipt, digest, plan = prepare_recovery()
    if args.dry_run:
        print(json.dumps({"status": "recovery_prechecked", "original_state_sha256": digest,
                          "preserved": plan["preserved"], "missing": plan["missing"],
                          "max_parallel": frozen.MAX_PARALLEL}, indent=2), flush=True)
        return 0
    queue = RecoveryQueue(state, digest, plan)
    return queue.run()


if __name__ == "__main__":
    raise SystemExit(main())
