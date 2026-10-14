"""Owner-aware reservation for the four SW0163 matched training arms."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

try:
    import fcntl
except ImportError:  # pragma: no cover - execution queue is Linux-only
    fcntl = None

# The SW0148 owner module hides CUDA at import. Restore the environment for
# callers that import this module inside a GPU-assigned process.
_UNSET = object()
_VISIBLE = os.environ.get("CUDA_VISIBLE_DEVICES", _UNSET)
from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4 as owner
if _VISIBLE is _UNSET:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = _VISIBLE

from collaborative_test.SW_0163_relation_context_assignment import context_queue, train

EXPERIMENT = "SW0163_relation_context_assignment_training_queue"
RUNTIME = HERE / "results_archive" / "training_queue_20261014"
STATE, LOCK = RUNTIME / "state.json", RUNTIME / "queue.lock"
LOGS = {arm: RUNTIME / f"{arm}_attempt01.log" for arm in train.ARMS}
DEVICE = "cuda:0"
MAX_MEMORY_MIB = 512
MAX_PARALLEL = 4
POLL_SECONDS = 3.0
SAMPLE_SECONDS = 5.0
WATCHDOG_SECONDS = 48 * 60 * 60
PREFLIGHT_MODULE = "collaborative_test.SW_0163_relation_context_assignment.context_queue"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint():
    guard = Path(owner.OWNER_GUARD)
    if not guard.is_file() or sha256_file(guard) != owner.OWNER_GUARD_SHA256:
        raise ValueError("registered SW0148 GPU-owner guard changed")
    paths = [HERE / name for name in (
        "training_queue.py", "train.py", "model.py", "context_cache.py",
        "preflight.py", "protocol.json", "DESIGN_KO.md")]
    paths += [Path(train.durable.__file__), Path(train.source_cache.__file__),
              Path(train.objective.__file__), Path(train.source_train.__file__),
              Path(owner.__file__), guard]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("SW0163 training queue dependency is missing")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def arm_command(arm):
    if arm not in train.ARMS:
        raise ValueError("unregistered SW0163 training arm")
    return [sys.executable, "-m", "collaborative_test.SW_0163_relation_context_assignment.train",
            "--arm", arm, "--device", DEVICE]


def dry_run_plan():
    return {"experiment": EXPERIMENT, "arms": list(train.ARMS),
        "commands": {arm: arm_command(arm) for arm in train.ARMS},
        "max_parallel": MAX_PARALLEL, "runtime": str(RUNTIME),
        "gpu_candidates": [0, 1, 2, 3], "max_memory_mib": MAX_MEMORY_MIB,
        "stable_seconds": owner.STABLE_SECONDS, "watchdog_seconds": WATCHDOG_SECONDS,
        "preflight_receipt": str(train.PREFLIGHT_PATH),
        "queue_fingerprint": implementation_fingerprint()}


def _atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def _proc_module_live(pid, module):
    if not isinstance(pid, int) or pid <= 1 or not sys.platform.startswith("linux"):
        return False
    proc = Path("/proc") / str(pid)
    try:
        raw = (proc / "cmdline").read_bytes().split(b"\0")
        args = [item.decode("utf-8", errors="replace") for item in raw if item]
        return proc.exists() and "-m" in args and module in args
    except (OSError, ValueError):
        return False


def preflight_dependency(*, state_path=context_queue.STATE, report_path=train.PREFLIGHT_PATH,
                         process_live=_proc_module_live):
    """Return admitted receipt, wait only on its live registered supervisor, or fail closed."""
    state_path, report_path = Path(state_path), Path(report_path)
    if not state_path.is_file():
        raise ValueError("SW0163 preflight queue state is missing; refusing unowned training")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("queue_fingerprint") != context_queue.implementation_fingerprint():
        raise ValueError("SW0163 preflight queue code fingerprint is not current")
    status = state.get("status")
    if status in ("complete_rejected", "failed", "failed_context_cache_stage", "failed_preflight_stage"):
        raise ValueError(f"SW0163 prerequisite preflight is terminal: {status}")
    if status != "complete_admitted":
        pid = state.get("supervisor_pid")
        if not process_live(pid, PREFLIGHT_MODULE):
            raise ValueError("SW0163 preflight is nonterminal but its registered supervisor is not live")
        return None
    if not report_path.is_file():
        raise ValueError("SW0163 admitted preflight state has no receipt")
    report_sha = sha256_file(report_path)
    if state.get("preflight_sha256") != report_sha:
        raise ValueError("SW0163 preflight state/receipt SHA mismatch")
    prepared = train.prepare_training_inputs_cpu(preflight_path=report_path)
    if prepared["preflight_report_sha256"] != report_sha:
        raise ValueError("SW0163 preflight changed during validation")
    return {"prepared": prepared, "report_sha256": report_sha,
            "queue_state_sha256": sha256_file(state_path), "state": state}


def _final_gpu_admission(gpu, expected_uuid):
    try:
        actual_uuid, memory, owners = owner.gpu_guard._gpu_info(gpu)
    except Exception as exc:
        return False, {"reason": "gpu_query_error", "detail": repr(exc)}
    sample = {"gpu_uuid": actual_uuid, "memory_used_mib": int(memory),
              "owner_pids": list(owners)}
    if actual_uuid != expected_uuid:
        sample["reason"] = "gpu_uuid_changed"
    elif owners:
        sample["reason"] = "gpu_has_owners"
    elif int(memory) > MAX_MEMORY_MIB:
        sample["reason"] = "gpu_memory_over_limit"
    else:
        sample["reason"] = "idle"
        return True, sample
    return False, sample


def _foreign(gpu, expected_uuid, child_pid):
    gpu_uuid, _memory, owners = owner.gpu_guard._gpu_info(gpu)
    if gpu_uuid != expected_uuid:
        return ["gpu_uuid_changed"]
    return owner.gpu_guard._foreign_owners(owners, owner.gpu_guard._process_tree(child_pid))


class TrainingQueue:
    def __init__(self, *, runtime=RUNTIME, max_parallel=MAX_PARALLEL):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if fcntl is None:
            raise RuntimeError("SW0163 training reservation requires Linux flock")
        if max_parallel != MAX_PARALLEL:
            raise ValueError("SW0163 queue concurrency is registered at four arm workers")
        self.runtime = Path(runtime)
        self.state_path, self.lock_path = self.runtime / "state.json", self.runtime / "queue.lock"
        self.logs = {arm: self.runtime / f"{arm}_attempt01.log" for arm in train.ARMS}
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.lock_stream = self.lock_path.open("a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock_stream.close()
            raise RuntimeError("another SW0163 training supervisor owns this queue") from exc
        self.state_lock = threading.RLock()
        self.stop_event = threading.Event()
        self.queue_fp = implementation_fingerprint()
        self.children = {}
        self.leases = {}
        try:
            if self.state_path.exists() or any(path.exists() for path in self.logs.values()):
                raise FileExistsError("preserving existing SW0163 training queue state/logs")
            self._write_state({"experiment": EXPERIMENT, "status": "queued",
                "supervisor_pid": os.getpid(), "queue_fingerprint": self.queue_fp,
                "task_plan": dry_run_plan(), "arms": {}, "created": time.time()})
        except BaseException:
            self._unlock()
            raise

    def _unlock(self):
        try:
            if self.lock_stream and not self.lock_stream.closed:
                fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_UN)
                self.lock_stream.close()
        except OSError:
            pass

    def _write_state(self, state):
        with self.state_lock:
            _atomic_json(self.state_path, state)

    def _update(self, status, **fields):
        with self.state_lock:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            state.update(status=status, updated=time.time(), **fields)
            _atomic_json(self.state_path, state)

    def _arm_update(self, arm, status, **fields):
        with self.state_lock:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            arms = state.setdefault("arms", {})
            record = dict(arms.get(arm, {})); record.update(status=status, updated=time.time(), **fields)
            arms[arm] = record
            state.update(status=f"arms_{status}", updated=time.time())
            _atomic_json(self.state_path, state)

    def _verify_code(self):
        if implementation_fingerprint() != self.queue_fp:
            raise RuntimeError("SW0163 training code/protocol changed after queue registration")

    def _wait_for_admission(self):
        while not self.stop_event.is_set():
            self._verify_code()
            try:
                dependency = preflight_dependency()
            except ValueError as exc:
                self._update("failed_preflight_dependency", error=str(exc), finished=time.time())
                return None
            if dependency is not None:
                self._update("preflight_admitted", preflight_report_sha256=dependency["report_sha256"],
                    preflight_queue_state_sha256=dependency["queue_state_sha256"])
                return dependency
            self._update("waiting_for_preflight", predecessor_module=PREFLIGHT_MODULE,
                         predecessor_pid=json.loads(context_queue.STATE.read_text(encoding="utf-8")).get("supervisor_pid"))
            self.stop_event.wait(POLL_SECONDS)
        self._update("cancelled_before_training", finished=time.time())
        return None

    def _existing_arm(self, arm, prepared):
        output = train.TRAIN_ROOT / f"{arm}_seed1"
        if output.exists():
            result = train.validate_training_artifact(arm, directory=output, prepared=prepared)
            self._arm_update(arm, "adopted_complete", manifest_sha256=sha256_file(output / "manifest.json"),
                             completed_updates=result["completed_updates"])
            return True
        return False

    def _run_arm(self, arm, prepared):
        if self._existing_arm(arm, prepared):
            return True
        # Expensive source/cache/preflight validation runs before acquiring a GPU.
        current = train.prepare_training_inputs_cpu()
        if current["preflight_report_sha256"] != prepared["preflight_report_sha256"]:
            raise ValueError("SW0163 preflight changed while queue waited")
        self._arm_update(arm, "waiting_for_idle_gpu")
        while not self.stop_event.is_set():
            gpu, gpu_uuid, memory, lease = owner._acquire(self.stop_event)
            self.leases[arm] = lease
            proc = None
            try:
                stable = owner._stable_idle(gpu, gpu_uuid, self.stop_event,
                    lambda elapsed, used: self._arm_update(arm, "waiting_stable_gpu",
                        gpu=gpu, gpu_uuid=gpu_uuid, stable_seconds=elapsed, memory_used_mib=used))
                if not stable:
                    self._release_lease(arm)
                    if self.stop_event.is_set():
                        return False
                    self._arm_update(arm, "waiting_for_idle_gpu", reservation_released=True)
                    continue
                self._verify_code()
                current = train.prepare_training_inputs_cpu()
                if current["preflight_report_sha256"] != prepared["preflight_report_sha256"]:
                    raise ValueError("SW0163 preflight changed before training launch")
                if self._existing_arm(arm, prepared):
                    return True
                admitted, sample = _final_gpu_admission(gpu, gpu_uuid)
                if not admitted:
                    self._arm_update(arm, "waiting_for_idle_gpu", gpu_sample=sample,
                                     reservation_released=True)
                    self._release_lease(arm)
                    continue
                log = self.logs[arm]
                if log.exists():
                    raise FileExistsError(f"preserving existing SW0163 arm log: {log}")
                argv = arm_command(arm)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                started = time.monotonic()
                with log.open("x", encoding="utf-8") as stream:
                    proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                        pass_fds=(lease.fileno(),))
                    self.children[arm] = proc
                    self._arm_update(arm, "running", child_pid=proc.pid, child_pgid=proc.pid,
                        child_argv=argv, gpu=gpu, gpu_uuid=gpu_uuid,
                        initial_memory_mib=memory, preflight_report_sha256=prepared["preflight_report_sha256"],
                        log=str(log), inherited_lease_fd=True)
                    while proc.poll() is None and not self.stop_event.is_set():
                        if time.monotonic() - started > WATCHDOG_SECONDS:
                            owner._stop_owned(proc)
                            self._arm_update(arm, "failed_watchdog", child_pid=proc.pid,
                                             artifacts_preserved=True)
                            self.stop_event.set()
                            return False
                        time.sleep(SAMPLE_SECONDS)
                        if proc.poll() is not None:
                            break
                        try:
                            foreign = _foreign(gpu, gpu_uuid, proc.pid)
                        except Exception:
                            continue
                        if foreign:
                            owner._stop_owned(proc)
                            self._arm_update(arm, "failed_foreign_gpu_owner", child_pid=proc.pid,
                                foreign_owners=foreign, artifacts_preserved=True)
                            self.stop_event.set()
                            return False
                    if self.stop_event.is_set() and proc.poll() is None:
                        owner._stop_owned(proc)
                        self._arm_update(arm, "cancelled_owned_child_stopped",
                                         child_pid=proc.pid, artifacts_preserved=True)
                        return False
                    rc = int(proc.wait())
                if rc != 0:
                    self._arm_update(arm, "failed_child", child_pid=proc.pid,
                                     child_returncode=rc, artifacts_preserved=True)
                    self.stop_event.set()
                    return False
                final_inputs = train.prepare_training_inputs_cpu()
                manifest = train.validate_training_artifact(arm, prepared=final_inputs)
                self._arm_update(arm, "complete", child_pid=proc.pid, child_returncode=rc,
                    manifest_sha256=sha256_file(train.TRAIN_ROOT / f"{arm}_seed1" / "manifest.json"),
                    completed_updates=manifest["completed_updates"])
                return True
            finally:
                try:
                    if proc is not None and proc.poll() is None:
                        owner._stop_owned(proc)
                finally:
                    self.children.pop(arm, None)
                    self._release_lease(arm)
        return False

    def _release_lease(self, arm):
        lease = self.leases.pop(arm, None)
        if lease is not None:
            try:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
            finally:
                lease.close()

    def run(self):
        try:
            self._verify_code()
            dependency = self._wait_for_admission()
            if dependency is None:
                return 2
            prepared = dependency["prepared"]
            self._update("training_arms_queued", preflight_report_sha256=dependency["report_sha256"],
                         registered_arms=list(train.ARMS), max_parallel=MAX_PARALLEL)
            results = {}
            with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
                futures = {pool.submit(self._run_arm, arm, prepared): arm for arm in train.ARMS}
                for future in as_completed(futures):
                    arm = futures[future]
                    try:
                        results[arm] = bool(future.result())
                    except BaseException as exc:
                        results[arm] = False
                        self._arm_update(arm, "failed", error=repr(exc), artifacts_preserved=True)
                        self.stop_event.set()
            complete = all(results.get(arm) for arm in train.ARMS)
            self._update("complete" if complete else "failed_training", results=results,
                         finished=time.time(), artifacts_preserved=True)
            return 0 if complete else 1
        finally:
            self.stop_event.set()
            for arm, proc in list(self.children.items()):
                if proc.poll() is None:
                    owner._stop_owned(proc)
            for arm in list(self.leases):
                self._release_lease(arm)
            self._unlock()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"status": "planned", "plan": dry_run_plan(),
            "cuda_visible_devices_parent": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")},
            allow_nan=False), flush=True)
        return 0
    return TrainingQueue().run()


if __name__ == "__main__":
    raise SystemExit(main())
