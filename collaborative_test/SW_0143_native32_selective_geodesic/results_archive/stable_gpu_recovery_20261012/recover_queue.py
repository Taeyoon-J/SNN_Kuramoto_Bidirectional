"""Private v2 SW0143 recovery with a 60-second continuously idle GPU gate."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
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

V1_HELPER_SHA256 = "4ad493180d63a89e44bb9f6a27cd79aaaf9e520907a339958d16b4cec38499ed"
V1_SERVER_HELPER = (ROOT / "collaborative_test/SW_0143_native32_selective_geodesic"
                    / "results_archive/recovery_code_v2_20261012/recover_queue.py")
V1_LOCAL_HELPER = (HERE.parent / "SW0143_recovery_v1_20261012" / "recover_queue.py")


def _sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_v1_helper():
    path = V1_SERVER_HELPER if V1_SERVER_HELPER.is_file() else V1_LOCAL_HELPER
    if not path.is_file() or _sha256_path(path) != V1_HELPER_SHA256:
        raise RuntimeError("the reviewed SW0143 v1 recovery helper is unavailable or changed")
    spec = importlib.util.spec_from_file_location("sw0143_recovery_v1_frozen", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load reviewed recovery helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


v1 = _load_v1_helper()

frozen = v1.frozen
run = v1.run
QUEUE_ARCHIVE = v1.QUEUE_ARCHIVE
ORIGINAL_STATE = v1.ORIGINAL_STATE
ORIGINAL_LOCK = v1.ORIGINAL_LOCK
RETIREMENT_RECEIPT = v1.RETIREMENT_RECEIPT
ORIGINAL_STATE_COPY = v1.ORIGINAL_STATE_COPY
V1_ROOT = QUEUE_ARCHIVE / "recovery_v1_20261012"
V1_STATE = V1_ROOT / "recovery_state.json"
V1_STATE_SHA256 = "3889eda3ecbec37b23a87d2007d0cc089cab6b4ca1cb0756534ab6800472d9bc"
V2_ROOT = QUEUE_ARCHIVE / "recovery_v2_20261012"
STABLE_IDLE_SECONDS = 60.0
SAMPLE_SECONDS = 5.0


def stable_idle_window(sample, *, stop_event, monotonic, sleep,
                       duration=STABLE_IDLE_SECONDS, interval=SAMPLE_SECONDS,
                       memory_limit=frozen.MEMORY_LIMIT_MIB, on_sample=None):
    """Return success only after continuous idle samples on one UUID."""
    started, initial_uuid = None, None
    while True:
        if stop_event.is_set():
            return False, "cancelled"
        try:
            gpu_uuid, memory_mib, owners = sample()
        except Exception:
            return False, "gpu_query_error"
        now = monotonic()
        if owners or int(memory_mib) > memory_limit:
            return False, "owner_or_memory"
        if started is None:
            started, initial_uuid = now, gpu_uuid
        elif gpu_uuid != initial_uuid:
            return False, "uuid_changed"
        elif now - started >= duration:
            try:
                final_uuid, final_memory, final_owners = sample()
            except Exception:
                return False, "final_gpu_query_error"
            if final_uuid != initial_uuid or final_owners or int(final_memory) > memory_limit:
                return False, "final_sample_changed"
            return True, {"gpu_uuid": initial_uuid, "stable_seconds": now - started}
        if on_sample is not None:
            on_sample(max(0.0, now - started), gpu_uuid, int(memory_mib))
        sleep(min(interval, max(0.0, duration - (now - started))))


def validate_v1_terminal(original_bytes, original_state, receipt, v1_bytes, v1_state,
                         *, expected_v1_sha256=V1_STATE_SHA256, pid_alive, proc_cmdlines):
    if hashlib.sha256(v1_bytes).hexdigest() != expected_v1_sha256:
        raise ValueError("v1 terminal state SHA differs from the reviewed failed attempt")
    if _sha256_path(Path(v1.__file__)) != V1_HELPER_SHA256:
        raise ValueError("v1 recovery helper changed after the reviewed attempt")
    original_sha = v1.validate_retirement(
        original_bytes, original_state, receipt, pid_alive=pid_alive,
        proc_cmdlines=lambda: [row for row in proc_cmdlines() if v1._is_live_original_queue(*row)])
    if (v1_state.get("experiment") != "SW0143_native32_selective_geodesic_recovery_v1"
            or v1_state.get("status") != "failed"
            or v1_state.get("supervisor_pid") != 381401
            or v1_state.get("original_state_sha256") != original_sha
            or v1_state.get("implementation_fingerprint") != run.implementation_fingerprint()):
        raise ValueError("v1 terminal state is not the registered failed SW0143 attempt")
    tasks = v1_state.get("tasks", {})
    if set(tasks) != {task["task_id"] for task in frozen.task_plan()}:
        raise ValueError("v1 recovery task set differs from the frozen SW0143 DAG")
    interrupted = tasks["preflight_s1"]
    if (interrupted.get("status") != "interrupted_foreign_gpu_owner"
            or interrupted.get("child_pid") != 387322
            or interrupted.get("foreign_owner_pids") != [387302]
            or interrupted.get("returncode") != -15):
        raise ValueError("v1 failure does not match the observed preflight_s1 interruption")
    for seed in (0, 2):
        row = tasks[f"preflight_s{seed}"]
        if (row.get("status") != "failed"
                or "recovery cancelled while waiting for an exclusive GPU" not in row.get("error", "")):
            raise ValueError(f"v1 preflight_s{seed} is not the expected cancellation")
    for task_id, row in tasks.items():
        pid = row.get("child_pid")
        if pid and pid_alive(pid):
            raise RuntimeError(f"v1 child still alive: {task_id} pid={pid}")
    if pid_alive(381401) or pid_alive(387322):
        raise RuntimeError("v1 supervisor or interrupted child remains alive")
    live_v1 = [
        (pid, argv) for pid, argv in proc_cmdlines()
        if pid != os.getpid()
        and "/sw0143_recovery_v1_20261012/recover_queue.py"
        in argv.replace("\\", "/").lower()
    ]
    if live_v1:
        raise RuntimeError(f"v1 recovery process remains active: {live_v1[0]}")
    return {"original_state_sha256": original_sha,
            "v1_state_sha256": hashlib.sha256(v1_bytes).hexdigest()}


def load_inputs(*, original_state=ORIGINAL_STATE, original_copy=ORIGINAL_STATE_COPY,
                retirement_receipt=RETIREMENT_RECEIPT, v1_state_path=V1_STATE,
                expected_v1_sha256=V1_STATE_SHA256, pid_alive=v1._pid_alive,
                proc_cmdlines=v1._proc_cmdlines):
    original_bytes = Path(original_state).read_bytes()
    original = v1._decode_state_copy(original_bytes, Path(original_copy).read_bytes())
    receipt = json.loads(Path(retirement_receipt).read_text(encoding="utf-8"))
    v1_bytes = Path(v1_state_path).read_bytes()
    v1_state = json.loads(v1_bytes.decode("utf-8"))
    lineage = validate_v1_terminal(
        original_bytes, original, receipt, v1_bytes, v1_state,
        expected_v1_sha256=expected_v1_sha256, pid_alive=pid_alive,
        proc_cmdlines=proc_cmdlines)
    plan = v1.build_recovery_plan(
        original, artifact_exists=lambda task: frozen.output_path(task).exists(),
        artifact_valid=frozen.validate_result)
    return original, v1_state, lineage, plan


def _atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


class StableRecoveryQueue(frozen.SelectiveGeodesicQueue):
    def __init__(self, lineage, plan, *, sample=None, monotonic=None, sleep=None):
        if fcntl is None:
            raise RuntimeError("SW0143 recovery requires Linux flock")
        if V2_ROOT.exists():
            raise FileExistsError(f"preserve existing recovery v2 path: {V2_ROOT}")
        self.original_lock = ORIGINAL_LOCK.open("r+")
        try:
            fcntl.flock(self.original_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.original_lock.close()
            raise RuntimeError("original SW0143 experiment lock is still owned") from exc
        try:
            V2_ROOT.mkdir(parents=True)
            self.log_root = V2_ROOT / "logs"
            self.log_root.mkdir()
            self.state_path = V2_ROOT / "recovery_state.json"
            self.recovery_lock = (V2_ROOT / "recovery.lock").open("x+")
            fcntl.flock(self.recovery_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            fcntl.flock(self.original_lock.fileno(), fcntl.LOCK_UN)
            self.original_lock.close()
            raise
        self.tasks = plan["tasks"]
        self.preserved = set(plan["preserved"])
        self.stop_event = threading.Event()
        self.guard = threading.RLock()
        self.sample = sample or (lambda gpu: frozen.owner._nvidia_gpu_info(gpu))
        self.monotonic = monotonic or time.monotonic
        self.sleep = sleep or time.sleep
        self.state = {
            "experiment": "SW0143_native32_selective_geodesic_recovery_v2",
            "status": "running", "supervisor_pid": os.getpid(), "started": time.time(),
            **lineage, "v1_helper_sha256": V1_HELPER_SHA256,
            "implementation_fingerprint": run.implementation_fingerprint(),
            "queue_sha256": run.sha256_file(Path(frozen.__file__)),
            "recovery_helper_sha256": _sha256_path(Path(__file__)),
            "owner_dispatcher_sha256": run.sha256_file(Path(frozen.owner.__file__)),
            "stable_idle_seconds_required": STABLE_IDLE_SECONDS,
            "gpu_sample_seconds": SAMPLE_SECONDS,
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
        while not self.stop_event.is_set():
            for gpu in range(4):
                if self.stop_event.is_set():
                    break
                lease_path = frozen.GPU_LEASES / f"gpu{gpu}.lock"
                lease_path.parent.mkdir(parents=True, exist_ok=True)
                lease = lease_path.open("a+")
                try:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lease.close()
                    continue
                try:
                    sample = lambda: self.sample(gpu)
                    try:
                        first = sample()
                    except Exception as exc:
                        self._update(task_id, status="unstable_gpu_released",
                                     physical_gpu=gpu, stability_failure=f"gpu_query_error:{exc}")
                        fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                        lease.close()
                        continue
                    if not frozen.gpu_candidate(first[1], first[2]):
                        fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                        lease.close()
                        continue
                    stable, detail = stable_idle_window(
                        sample, stop_event=self.stop_event, monotonic=self.monotonic,
                        sleep=self.sleep,
                        on_sample=lambda elapsed, uuid, memory: self._update(
                            task_id, status="waiting_for_stable_idle_window",
                            physical_gpu=gpu, gpu_uuid=uuid, stable_seconds=elapsed,
                            memory_mib=memory, quiet_window_required_seconds=STABLE_IDLE_SECONDS))
                    if not stable:
                        self._update(task_id, status="unstable_gpu_released",
                                     physical_gpu=gpu, stability_failure=detail)
                        fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                        lease.close()
                        continue
                    self._update(task_id, status="reserved_stable_idle", physical_gpu=gpu,
                                 gpu_uuid=detail["gpu_uuid"], stable_seconds=detail["stable_seconds"],
                                 lease_path=str(lease_path))
                    return gpu, lease, detail["gpu_uuid"]
                except BaseException:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                    lease.close()
                    raise
            if self.stop_event.is_set():
                break
            self._update(task_id, status="waiting_for_stable_exclusive_gpu", last_wait=time.time())
            self.sleep(SAMPLE_SECONDS)
        raise RuntimeError("recovery cancelled while waiting for a stable exclusive GPU")

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
            if task["stage"] == "score":
                env = frozen.environment(task)
            else:
                while True:
                    gpu, lease, uuid = self._reserve(task_id)
                    triton = Path("/tmp") / f"kevinswk_sw0143_recovery_v2_{os.getpid()}_{task_id}_{int(time.time()*1000)}"
                    triton.mkdir(exist_ok=False)
                    env = frozen.environment(task, physical_gpu=gpu, triton_cache=triton)
                    break
            argv = frozen.command(task, device="cuda:0")
            attempt = 0
            while True:
                attempt += 1
                log = self.log_root / f"{task_id}.attempt{attempt:02d}.log"
                self._update(task_id, status="launching", argv=argv, log_path=str(log),
                             environment={"CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"]},
                             started=time.time(), launch_attempt=attempt)
                with log.open("x", encoding="utf-8", newline="\n") as stream:
                    options = {"cwd": str(frozen.ROOT), "env": env, "stdout": stream,
                               "stderr": subprocess.STDOUT, "start_new_session": True}
                    if lease is not None:
                        options["pass_fds"] = (lease.fileno(),)
                    # Last owner/UUID/memory check is immediately before Popen.
                    if lease is not None:
                        try:
                            now_uuid, now_memory, now_owners = self.sample(gpu)
                        except Exception as exc:
                            now_uuid, now_memory, now_owners = None, frozen.MEMORY_LIMIT_MIB + 1, ["query-error"]
                            failure = f"gpu_query_error:{exc}"
                        else:
                            failure = "uuid_or_ownership_changed"
                        if (self.stop_event.is_set() or now_uuid != uuid
                                or not frozen.gpu_candidate(now_memory, now_owners)):
                            self._update(task_id, status="unstable_gpu_released",
                                         stability_failure=failure, physical_gpu=gpu,
                                         last_stable_uuid=uuid)
                            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                            lease.close()
                            lease = gpu = None
                            if self.stop_event.is_set():
                                raise RuntimeError("recovery stopped before subprocess creation")
                            gpu, lease, uuid = self._reserve(task_id)
                            triton = Path("/tmp") / f"kevinswk_sw0143_recovery_v2_{os.getpid()}_{task_id}_{int(time.time()*1000)}"
                            triton.mkdir(exist_ok=False)
                            env = frozen.environment(task, physical_gpu=gpu, triton_cache=triton)
                            continue
                    if self.stop_event.is_set():
                        raise RuntimeError("recovery stopped before subprocess creation")
                    proc = subprocess.Popen(argv, **options)
                    self._update(task_id, status="running", child_pid=proc.pid)
                    break
            while proc.poll() is None:
                time.sleep(5)
                if proc.poll() is not None or gpu is None:
                    continue
                try:
                    _uuid, _memory, owners = frozen.owner._nvidia_gpu_info(gpu)
                except Exception as exc:
                    frozen.SelectiveGeodesicQueue._terminate_owned(proc)
                    self._update(task_id, status="failed_gpu_monitor_error",
                                 error=f"{type(exc).__name__}: {exc}", returncode=proc.returncode)
                    self.stop_event.set()
                    return False
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
            if not self.recovery_lock.closed:
                self.recovery_lock.close()
            if not self.original_lock.closed:
                self.original_lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    _original, _previous, lineage, plan = load_inputs()
    if args.dry_run:
        print(json.dumps({"status": "recovery_prechecked", **lineage,
                          "previous_state_sha256": lineage["v1_state_sha256"],
                          "preserved": plan["preserved"], "missing": plan["missing"],
                          "max_parallel": frozen.MAX_PARALLEL,
                          "quiet_window_seconds": STABLE_IDLE_SECONDS}, indent=2), flush=True)
        return 0
    return StableRecoveryQueue(lineage, plan).run()


if __name__ == "__main__":
    raise SystemExit(main())
