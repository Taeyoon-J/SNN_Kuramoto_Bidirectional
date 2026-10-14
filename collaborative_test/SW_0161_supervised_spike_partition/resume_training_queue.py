"""Owner-safe SW0161 two-arm training reservation queue.

Each arm has an independent state/lock/log. The queue waits CPU-only for the
registered cache+disposable-preflight owner, then validates the exact receipt
before reserving a GPU and again before the final under-lease GPU admission.
"""
from __future__ import annotations

import argparse
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
for _path in (ROOT, ROOT / "collaborative_test"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np

try:
    import fcntl
except ImportError:
    fcntl = None

# The shared owner module hides CUDA at import; preserve an assigned child env.
_unset = object()
_visibility_before_owner = os.environ.get("CUDA_VISIBLE_DEVICES", _unset)
from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4 as owner
if _visibility_before_owner is _unset:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = _visibility_before_owner

from collaborative_test.SW_0161_supervised_spike_partition import cache, cache_queue, preflight, resume_cache_queue, train

EXPERIMENT = train.EXPERIMENT
RUNTIME = HERE / "results_archive" / "training_queue_retry_20261014"
ORIGINAL_RUNTIME = HERE / "results_archive" / "training_queue_20261014"
PREFLIGHT_STATE = resume_cache_queue.STATE
PREFLIGHT_MODULE = "collaborative_test.SW_0161_supervised_spike_partition.resume_cache_queue"
CHILD_MODULE = "collaborative_test.SW_0161_supervised_spike_partition.train"
DEVICE = "cuda:0"
MAX_MEMORY_MIB = cache_queue.MAX_MEMORY_MIB
POLL_SECONDS = 3.0
SAMPLE_SECONDS = 2.0
WATCHDOG_SECONDS = 24 * 60 * 60


class UpstreamPending(RuntimeError):
    """The only retryable state: the registered cache/preflight owner is live."""


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint():
    paths = (HERE / "resume_training_queue.py", HERE / "train.py", HERE / "cache.py",
             HERE / "preflight.py", HERE / "objective.py", HERE / "protocol.json",
             HERE / "cache_queue.py", HERE / "resume_cache_queue.py", Path(owner.__file__), owner.OWNER_GUARD)
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("SW0161 training queue dependency is missing")
    if sha256_file(owner.OWNER_GUARD) != owner.OWNER_GUARD_SHA256:
        raise ValueError("registered shared GPU owner guard changed")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def configure_arm(arm, runtime=RUNTIME):
    if arm not in train.ARMS:
        raise ValueError("unregistered SW0161 arm")
    root = Path(runtime) / arm
    return {"runtime": root, "state": root / "state.json",
            "lock": root / "queue.lock", "log": root / "attempt01.log",
            "output": Path(train.TRAIN_ROOT) / f"{arm}_seed1",
            "work": Path(train.WORK_ROOT) / f"{arm}_seed1" / "checkpoints"}


def command(arm):
    if arm not in train.ARMS:
        raise ValueError("unregistered SW0161 arm")
    return [sys.executable, "-m", "collaborative_test.SW_0161_supervised_spike_partition.train",
            "--arm", arm, "--device", DEVICE]


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    if os.name != "nt":
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _proc_live(pid, expected_module):
    if type(pid) is not int or pid <= 0:
        return False
    row = owner._proc_record(pid)
    if row is None or row.get("state") == "Z":
        return False
    argv = row.get("argv", [])
    if expected_module not in argv:
        raise RuntimeError(f"recorded SW0161 PID {pid} is live with unexpected identity")
    return True


_ACTIVE_UPSTREAM = {
    "waiting_for_sw0160_score", "cache_validated", "cache_running", "cache_adopted", "cache_complete",
    "cache_waiting_stable_gpu", "cache_waiting_for_exclusive_gpu",
    "preflight_running", "preflight_adopted", "preflight_complete_admitted",
    "preflight_complete_rejected", "preflight_waiting_stable_gpu",
    "preflight_waiting_for_exclusive_gpu",
}


def _upstream_owner_live(state):
    alive = _proc_live(state.get("supervisor_pid"), PREFLIGHT_MODULE)
    stages = state.get("stages", {})
    if not isinstance(stages, dict):
        raise ValueError("SW0161 upstream stage state is malformed")
    for stage, child_module in (("cache", "collaborative_test.SW_0161_supervised_spike_partition.cache"),
                                ("preflight", "collaborative_test.SW_0161_supervised_spike_partition.preflight")):
        item = stages.get(stage, {})
        if not isinstance(item, dict):
            raise ValueError("SW0161 upstream child state is malformed")
        alive = _proc_live(item.get("child_pid"), child_module) or alive
    return alive


def validate_original_failed_attempt(arm, *, state_path=None):
    """Admit only the known pre-GPU cache_validated failure, preserving its files."""
    if arm not in train.ARMS:
        raise ValueError("unregistered SW0161 arm")
    path = Path(state_path) if state_path is not None else ORIGINAL_RUNTIME / arm / "state.json"
    if not path.is_file():
        raise RuntimeError("original SW0161 training attempt receipt is absent")
    state = json.loads(path.read_text(encoding="utf-8"))
    error = state.get("error")
    known_failure = ((state.get("status") == "failed" and isinstance(error, str)
                      and "cache_validated" in error)
                     or (state.get("status") == "cache_validated" and isinstance(error, str)
                         and "cache_validated" in error))
    if (state.get("experiment") != EXPERIMENT or state.get("arm") != arm
            or not known_failure or state.get("child_pid") not in (None, 0)):
        raise RuntimeError("original SW0161 attempt is not the registered pre-GPU cache_validated failure")
    supervisor = state.get("supervisor_pid")
    if type(supervisor) is int and supervisor > 0 and _proc_live(
            supervisor, "collaborative_test.SW_0161_supervised_spike_partition.training_queue"):
        raise RuntimeError("original SW0161 failed supervisor is still live")
    output = Path(train.TRAIN_ROOT) / f"{arm}_seed1"
    work = Path(train.WORK_ROOT) / f"{arm}_seed1" / "checkpoints"
    if output.exists() or work.exists():
        raise RuntimeError("original SW0161 attempt left training output/work; preserving it")
    return state


def _validated_upstream(state_path=PREFLIGHT_STATE):
    """Deep-validate current cache and admitted preflight, or raise Pending/fail."""
    path = Path(state_path)
    if not path.is_file():
        raise RuntimeError("SW0161 registered cache/preflight queue state is absent")
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("experiment") != resume_cache_queue.EXPERIMENT:
        raise ValueError("SW0161 cache/preflight experiment mismatch")
    status = state.get("status")
    if status != "complete_admitted":
        if status == "complete_rejected":
            raise RuntimeError("SW0161 disposable preflight scientifically rejected this recipe")
        if status in _ACTIVE_UPSTREAM:
            if _upstream_owner_live(state):
                raise UpstreamPending("waiting for live registered SW0161 cache/preflight owner")
            raise RuntimeError("SW0161 cache/preflight owner is absent or terminal before admission")
        raise RuntimeError(f"SW0161 cache/preflight is not admitted (status={status!r})")
    if state.get("training_admission") is not True:
        raise RuntimeError("SW0161 terminal preflight state lacks positive admission")
    expected_queue_fp = cache_queue.implementation_fingerprint()
    if state.get("queue_fingerprint") != expected_queue_fp:
        raise ValueError("SW0161 cache/preflight queue fingerprint changed")
    if state.get("cache_fingerprint") != cache.implementation_fingerprint():
        raise ValueError("SW0161 source cache fingerprint changed")
    if state.get("preflight_fingerprint") != preflight.implementation_fingerprint():
        raise ValueError("SW0161 preflight fingerprint changed")
    cache_manifest = train.validate_cache()
    cache_manifest_path = Path(train.CACHE_ROOT) / "manifest.json"
    cache_sha = sha256_file(cache_manifest_path)
    if state.get("cache_manifest_sha256") != cache_sha:
        raise ValueError("SW0161 upstream state/cache manifest hash mismatch")
    if state.get("preflight_sha256") != sha256_file(train.PREFLIGHT_PATH):
        raise ValueError("SW0161 upstream state/preflight report hash mismatch")
    admission = train.validate_preflight_admission(cache_manifest=cache_manifest)
    # Re-read after the expensive hashes to reject a changed or replaced owner record.
    final_state = json.loads(path.read_text(encoding="utf-8"))
    if final_state != state or final_state.get("status") != "complete_admitted":
        raise RuntimeError("SW0161 upstream state changed during admission validation")
    return {"state": state, "cache_manifest": cache_manifest,
            "cache_manifest_sha256": cache_sha,
            "cache_bindings_sha256": cache_manifest["bindings_sha256"],
            "preflight_report_sha256": admission["report_sha256"],
            "preflight_implementation_fingerprint": admission["preflight_implementation_fingerprint"],
            "queue_fingerprint": expected_queue_fp}


def _wait_for_upstream(stop_event, on_wait=None, validator=None):
    validator = validator or _validated_upstream
    while not stop_event.is_set():
        try:
            return validator()
        except UpstreamPending as exc:
            if on_wait:
                on_wait(str(exc))
            stop_event.wait(POLL_SECONDS)
    raise RuntimeError("SW0161 training queue cancelled while waiting for preflight")


def preferred_resume_gpu(work_dir):
    """Pin resume to the physical GPU encoded by the last committed boundary."""
    boundary = Path(work_dir) / "boundary.ptx"
    if not boundary.is_file():
        return None
    from collaborative_test.SW_0148_checkpointable_boundary_q import checkpoint as durable
    metadata = durable.inspect_committed_metadata(Path(work_dir))
    completed = metadata.get("completed_updates")
    runtime = metadata.get("runtime")
    if (type(completed) is not int or not 0 <= completed <= train.UPDATES
            or not isinstance(runtime, dict) or runtime.get("device_type") != "cuda"
            or runtime.get("device_index") != 0):
        raise ValueError("SW0161 resume boundary has invalid cursor or CUDA runtime")
    visible = runtime.get("cuda_visible_devices")
    if not isinstance(visible, str) or not visible.isdigit():
        raise ValueError("SW0161 resume boundary lacks a physical GPU visibility binding")
    gpu = int(visible)
    if gpu not in (0, 1, 2, 3):
        raise ValueError("SW0161 resume boundary names an unregistered physical GPU")
    return gpu


def _final_gpu_admission(gpu, expected_uuid):
    try:
        actual_uuid, memory, owners = owner.gpu_guard._gpu_info(gpu)
    except Exception as exc:
        return False, {"reason": "gpu_query_error", "detail": repr(exc)}
    row = {"gpu_uuid": actual_uuid, "memory_used_mib": int(memory), "owner_pids": list(owners)}
    if actual_uuid != expected_uuid:
        row["reason"] = "gpu_uuid_changed"
    elif owners:
        row["reason"] = "gpu_has_owners"
    elif int(memory) > MAX_MEMORY_MIB:
        row["reason"] = "gpu_memory_over_limit"
    else:
        row["reason"] = "idle"
        return True, row
    return False, row


def _foreign_owners(gpu, expected_uuid, child_pid):
    current_uuid, _memory, owners = owner.gpu_guard._gpu_info(gpu)
    if current_uuid != expected_uuid:
        return ["gpu_uuid_changed"]
    return owner.gpu_guard._foreign_owners(owners, owner.gpu_guard._process_tree(child_pid))


class TrainingQueue:
    def __init__(self, arm, *, runtime=RUNTIME):
        if arm not in train.ARMS:
            raise ValueError("unregistered SW0161 arm")
        if fcntl is None:
            raise RuntimeError("SW0161 training queue requires Linux flock")
        self.arm = arm
        self.paths = configure_arm(arm, runtime)
        self.paths["runtime"].mkdir(parents=True, exist_ok=True)
        self.lock_stream = self.paths["lock"].open("a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock_stream.close()
            raise RuntimeError("another SW0161 arm queue owns this reservation") from exc
        self.stop_event = threading.Event()
        self.state_lock = threading.RLock()
        self.queue_fp = implementation_fingerprint()
        self._child = None
        self._lease = None
        try:
            validate_original_failed_attempt(arm)
            if self.paths["state"].exists() or self.paths["log"].exists():
                raise FileExistsError("preserving existing SW0161 arm queue state/log")
            _atomic_json(self.paths["state"], {"experiment": EXPERIMENT, "arm": arm,
                "status": "waiting_for_admitted_preflight", "supervisor_pid": os.getpid(),
                "queue_fingerprint": self.queue_fp, "task": command(arm), "created": time.time()})
        except BaseException:
            self._unlock()
            raise

    def _unlock(self):
        if self.lock_stream and not self.lock_stream.closed:
            try:
                fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_UN)
            finally:
                self.lock_stream.close()

    def _update(self, status, **fields):
        with self.state_lock:
            current = json.loads(self.paths["state"].read_text(encoding="utf-8"))
            current.update(status=status, updated=time.time(), **fields)
            _atomic_json(self.paths["state"], current)

    def _verify_code(self):
        if implementation_fingerprint() != self.queue_fp:
            raise RuntimeError("SW0161 training queue/dependencies changed")

    def _adopt_existing(self):
        output = self.paths["output"]
        if not output.exists():
            return False
        manifest = train.validate_training_artifact(self.arm, directory=output)
        manifest_sha = sha256_file(output / "manifest.json")
        self._update("complete_adopted", manifest_sha256=manifest_sha,
                     completed_updates=manifest.get("completed_updates"), adopted=True,
                     finished=time.time())
        return True

    def run(self):
        proc = lease = None
        try:
            upstream = _wait_for_upstream(self.stop_event,
                lambda detail: self._update("waiting_for_admitted_preflight", wait_reason=detail))
            if self._adopt_existing():
                return 0
            self._verify_code()
            preferred = preferred_resume_gpu(self.paths["work"])
            self._update("waiting_for_gpu", admitted_preflight_sha256=upstream["preflight_report_sha256"],
                         cache_manifest_sha256=upstream["cache_manifest_sha256"],
                         preferred_resume_gpu=preferred)
            while not self.stop_event.is_set():
                gpu, gpu_uuid, memory, lease = owner._acquire(self.stop_event,
                    preferred_gpu=preferred)
                self._lease = lease
                try:
                    stable = owner._stable_idle(gpu, gpu_uuid, self.stop_event,
                        lambda elapsed, used: self._update("waiting_stable_gpu", gpu=gpu,
                            gpu_uuid=gpu_uuid, stable_seconds=elapsed, memory_used_mib=used))
                    if not stable:
                        self._release_lease()
                        if self.stop_event.is_set():
                            self._update("cancelled_before_child"); return 1
                        self._update("waiting_for_gpu", reservation_released=True)
                        continue
                    self._verify_code()
                    fresh_upstream = _validated_upstream()
                    if fresh_upstream != upstream:
                        raise RuntimeError("SW0161 cache/preflight lineage changed before launch")
                    if self._adopt_existing():
                        return 0
                    fresh_preferred = preferred_resume_gpu(self.paths["work"])
                    if fresh_preferred != preferred:
                        raise RuntimeError("SW0161 checkpoint GPU affinity changed during admission")
                    admitted, gpu_check = _final_gpu_admission(gpu, gpu_uuid)
                    if not admitted:
                        self._release_lease()
                        self._update("waiting_for_gpu", reservation_released=True,
                                     final_gpu_check=gpu_check)
                        continue
                    argv = command(self.arm)
                    child_env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                                     OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                    started = time.monotonic()
                    with self.paths["log"].open("x", encoding="utf-8") as log:
                        proc = subprocess.Popen(argv, cwd=ROOT, env=child_env,
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True, pass_fds=(lease.fileno(),))
                        self._child = proc
                        self._update("running", child_pid=proc.pid, child_pgid=proc.pid,
                            child_argv=argv, physical_gpu=gpu, gpu_uuid=gpu_uuid,
                            memory_before_mib=memory, inherited_lease_fd=True,
                            upstream_sha256=hashlib.sha256(json.dumps(upstream, sort_keys=True,
                                separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
                            log=str(self.paths["log"]), watchdog_seconds=WATCHDOG_SECONDS)
                        while proc.poll() is None and not self.stop_event.is_set():
                            if time.monotonic() - started > WATCHDOG_SECONDS:
                                owner._stop_owned(proc)
                                self._update("failed_watchdog", child_pid=proc.pid,
                                             artifacts_preserved=True, finished=time.time())
                                return 1
                            self.stop_event.wait(SAMPLE_SECONDS)
                            if proc.poll() is not None:
                                break
                            try:
                                foreign = _foreign_owners(gpu, gpu_uuid, proc.pid)
                            except Exception as exc:
                                self._update("running", last_gpu_query_error=repr(exc)); continue
                            if foreign:
                                owner._stop_owned(proc)
                                self._update("failed_foreign_gpu_owner", child_pid=proc.pid,
                                    foreign_owner_pids=foreign, artifacts_preserved=True,
                                    finished=time.time())
                                return 1
                        if self.stop_event.is_set() and proc.poll() is None:
                            owner._stop_owned(proc)
                            self._update("cancelled_owned_child_stopped", child_pid=proc.pid,
                                         artifacts_preserved=True, finished=time.time())
                            return 1
                        rc = int(proc.wait())
                    if rc != 0:
                        self._update("failed_child", child_returncode=rc,
                                     artifacts_preserved=True, finished=time.time())
                        return rc
                    manifest = train.validate_training_artifact(self.arm, directory=self.paths["output"])
                    manifest_sha = sha256_file(self.paths["output"] / "manifest.json")
                    self._update("complete", child_returncode=rc, manifest_sha256=manifest_sha,
                                 completed_updates=manifest["completed_updates"],
                                 training_admission=True, finished=time.time())
                    return 0
                finally:
                    try:
                        if proc is not None and proc.poll() is None:
                            owner._stop_owned(proc)
                    finally:
                        self._child = None
                        self._release_lease()
            self._update("cancelled_before_gpu")
            return 1
        except BaseException as exc:
            self._update("failed", error=repr(exc), artifacts_preserved=True,
                         finished=time.time())
            raise
        finally:
            try:
                if proc is not None and proc.poll() is None:
                    owner._stop_owned(proc)
            finally:
                self._release_lease()
                self._unlock()

    def _release_lease(self):
        lease, self._lease = self._lease, None
        if lease is not None:
            try:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
            finally:
                lease.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=train.ARMS, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"experiment": EXPERIMENT, "arm": args.arm,
            "argv": command(args.arm), "state": str(configure_arm(args.arm)["state"]),
            "gpu_candidates": [0, 1, 2, 3], "requires_admitted_preflight": True,
            "training_updates": train.UPDATES, "queue_fingerprint": implementation_fingerprint()},
            sort_keys=True, allow_nan=False), flush=True)
        return 0
    return TrainingQueue(args.arm).run()


if __name__ == "__main__":
    raise SystemExit(main())

