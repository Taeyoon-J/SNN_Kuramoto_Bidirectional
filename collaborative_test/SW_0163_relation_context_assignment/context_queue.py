"""Owner-safe sequential reservation for SW0163 context creation and head preflight."""
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

try:
    import fcntl
except ImportError:  # pragma: no cover - server queue is Linux-only
    fcntl = None

# SW0148's shared owner module hides CUDA during import. Restore the caller's
# visibility so importing this library in an assigned GPU child is harmless.
_unset_visibility = object()
_saved_visibility = os.environ.get("CUDA_VISIBLE_DEVICES", _unset_visibility)
from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4 as owner
if _saved_visibility is _unset_visibility:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = _saved_visibility

from collaborative_test.SW_0163_relation_context_assignment import context_cache, preflight

EXPERIMENT = "SW0163_context_cache_and_head_preflight_queue"
RUNTIME = HERE / "results_archive" / "context_preflight_queue_20261014"
STATE, LOCK = RUNTIME / "state.json", RUNTIME / "queue.lock"
LOGS = {"context_cache": RUNTIME / "context_cache_attempt01.log",
        "preflight": RUNTIME / "head_preflight_attempt01.log"}
GPU_LEASE_DIR = owner.GPU_LEASE_DIR
DEVICE = "cuda:0"
MAX_MEMORY_MIB = 512
POLL_SECONDS = 3.0
SAMPLE_SECONDS = 2.0
WATCHDOG_SECONDS = 6 * 60 * 60


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
    paths = (HERE / "context_queue.py", HERE / "context_cache.py", HERE / "preflight.py",
             HERE / "model.py", HERE / "protocol.json", HERE / "DESIGN_KO.md",
             Path(context_cache.source_cache.__file__), Path(preflight.source_train.__file__),
             Path(owner.__file__), guard)
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("SW0163 queue dependency is missing")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def stage_command(stage):
    if stage == "context_cache":
        return [sys.executable, "-m", "collaborative_test.SW_0163_relation_context_assignment.context_cache",
                "--device", DEVICE, "--output", str(context_cache.CACHE_ROOT)]
    if stage == "preflight":
        return [sys.executable, "-m", "collaborative_test.SW_0163_relation_context_assignment.preflight",
                "--device", DEVICE, "--output", str(preflight.OUTPUT)]
    raise ValueError(f"unknown SW0163 stage: {stage}")


def dry_run_plan():
    return {"experiment": EXPERIMENT, "stages": ["context_cache", "preflight"],
        "commands": {stage: stage_command(stage) for stage in ("context_cache", "preflight")},
        "runtime": str(RUNTIME), "gpu_candidates": [0, 1, 2, 3],
        "device": DEVICE, "max_memory_mib": MAX_MEMORY_MIB,
        "stable_seconds": owner.STABLE_SECONDS, "watchdog_seconds": WATCHDOG_SECONDS,
        "queue_fingerprint": implementation_fingerprint()}


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _final_gpu_admission(gpu, expected_uuid):
    try:
        actual_uuid, memory, owners = owner.gpu_guard._gpu_info(gpu)
    except Exception as exc:
        return False, {"reason": "gpu_query_error", "detail": repr(exc)}
    result = {"gpu_uuid": actual_uuid, "memory_used_mib": int(memory),
              "owner_pids": list(owners)}
    if actual_uuid != expected_uuid:
        result["reason"] = "gpu_uuid_changed"
    elif owners:
        result["reason"] = "gpu_has_owners"
    elif int(memory) > MAX_MEMORY_MIB:
        result["reason"] = "gpu_memory_over_limit"
    else:
        result["reason"] = "idle"
        return True, result
    return False, result


def _foreign(gpu, expected_uuid, child_pid):
    current_uuid, _memory, owners = owner.gpu_guard._gpu_info(gpu)
    if current_uuid != expected_uuid:
        return ["gpu_uuid_changed"]
    return owner.gpu_guard._foreign_owners(owners, owner.gpu_guard._process_tree(child_pid))


class ContextQueue:
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if fcntl is None:
            raise RuntimeError("SW0163 owner queue requires Linux flock")
        RUNTIME.mkdir(parents=True, exist_ok=True)
        self.lock_stream = LOCK.open("a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock_stream.close()
            raise RuntimeError("another SW0163 context supervisor owns the queue") from exc
        self.state_lock = threading.RLock()
        self.stop_event = threading.Event()
        self.queue_fp = implementation_fingerprint()
        self.context_fp = context_cache._fingerprint()
        self.preflight_fp = preflight.implementation_fingerprint()
        self.child = None
        self.lease = None
        try:
            if STATE.exists():
                raise FileExistsError(f"preserving existing SW0163 queue state: {STATE}")
            if any(path.exists() for path in LOGS.values()):
                raise FileExistsError("preserving existing SW0163 queue attempt logs")
            _atomic_json(STATE, {"experiment": EXPERIMENT, "status": "queued",
                "supervisor_pid": os.getpid(), "queue_fingerprint": self.queue_fp,
                "context_cache_fingerprint": self.context_fp,
                "preflight_fingerprint": self.preflight_fp,
                "stages": {}, "task_plan": dry_run_plan(), "created": time.time()})
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

    def _update(self, status, **fields):
        with self.state_lock:
            record = json.loads(STATE.read_text(encoding="utf-8"))
            record.update(status=status, updated=time.time(), **fields)
            _atomic_json(STATE, record)

    def _stage_update(self, stage, status, **fields):
        with self.state_lock:
            record = json.loads(STATE.read_text(encoding="utf-8"))
            stages = record.setdefault("stages", {})
            item = dict(stages.get(stage, {})); item.update(status=status, updated=time.time(), **fields)
            stages[stage] = item
            record.update(status=f"{stage}_{status}", updated=time.time())
            _atomic_json(STATE, record)

    def _verify_frozen_code(self):
        if implementation_fingerprint() != self.queue_fp:
            raise RuntimeError("SW0163 queue or registered dependencies changed")
        if context_cache._fingerprint() != self.context_fp:
            raise RuntimeError("SW0163 context-cache implementation changed")
        if preflight.implementation_fingerprint() != self.preflight_fp:
            raise RuntimeError("SW0163 preflight implementation changed")

    def _upstream(self, stage):
        if stage == "context_cache":
            prepared = context_cache.prepare_context_inputs_cpu()
            return {"source_context_bindings_sha256": context_cache._json_sha(prepared["context_bindings"]),
                "source_trace_manifest_sha256": sha256_file(context_cache.source_cache.CACHE_ROOT / "manifest.json"),
                "source_checkpoint_sha256": prepared["context_bindings"]["source_checkpoint_sha256"],
                "context_fingerprint": self.context_fp}
        return preflight.validate_upstream()

    def _existing(self, stage, expected):
        if stage == "context_cache":
            if not (context_cache.CACHE_ROOT / "manifest.json").is_file():
                return False
            bindings = context_cache.prepare_context_inputs_cpu()["context_bindings"]
            result = context_cache.validate_final_cache(context_cache.CACHE_ROOT,
                                                        expected_bindings=bindings)
            self._stage_update(stage, "adopted", manifest_sha256=result["manifest_sha256"], count=result["count"])
            return True
        if not preflight.OUTPUT.is_file():
            return False
        current = preflight.validate_upstream()
        report = json.loads(preflight.OUTPUT.read_text(encoding="utf-8"))
        preflight.validate_report_record(report, expected=current)
        self._stage_update(stage, "adopted", report_sha256=sha256_file(preflight.OUTPUT),
                           training_admission=report["training_admission"])
        return True

    def _launch_stage(self, stage):
        while not self.stop_event.is_set():
            gpu, gpu_uuid, initial_memory, lease = owner._acquire(self.stop_event)
            self.lease = lease
            proc = None
            try:
                stable = owner._stable_idle(gpu, gpu_uuid, self.stop_event,
                    lambda elapsed, used: self._stage_update(stage, "waiting_stable_gpu",
                        gpu=gpu, gpu_uuid=gpu_uuid, stable_seconds=elapsed, memory_used_mib=used))
                if not stable:
                    self._release_lease()
                    if self.stop_event.is_set():
                        return False
                    continue
                self._verify_frozen_code()
                upstream = self._upstream(stage)
                upstream_digest = _digest(upstream)
                if self._existing(stage, upstream):
                    return True
                admitted, gpu_sample = _final_gpu_admission(gpu, gpu_uuid)
                if not admitted:
                    self._stage_update(stage, "waiting_for_exclusive_gpu",
                        gpu_check=gpu_sample, reservation_released=True)
                    self._release_lease()
                    continue
                log = LOGS[stage]
                log.parent.mkdir(parents=True, exist_ok=True)
                if log.exists():
                    raise FileExistsError(f"preserving existing SW0163 stage log: {log}")
                argv = stage_command(stage)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                start = time.monotonic()
                with log.open("x", encoding="utf-8") as stream:
                    proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                        pass_fds=(lease.fileno(),))
                    self.child = proc
                    self._stage_update(stage, "running", child_pid=proc.pid, child_pgid=proc.pid,
                        child_argv=argv, physical_gpu=gpu, gpu_uuid=gpu_uuid,
                        initial_memory_mib=initial_memory, upstream_sha256=upstream_digest,
                        log=str(log), inherited_lease_fd=True)
                    while proc.poll() is None and not self.stop_event.is_set():
                        if time.monotonic() - start > WATCHDOG_SECONDS:
                            owner._stop_owned(proc)
                            self._stage_update(stage, "failed_watchdog", child_pid=proc.pid,
                                               artifacts_preserved=True)
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
                            self._stage_update(stage, "failed_foreign_gpu_owner",
                                child_pid=proc.pid, foreign_owners=foreign, artifacts_preserved=True)
                            return False
                    if self.stop_event.is_set() and proc.poll() is None:
                        owner._stop_owned(proc)
                        self._stage_update(stage, "cancelled_owned_child_stopped",
                            child_pid=proc.pid, artifacts_preserved=True)
                        return False
                    rc = int(proc.wait())
                if stage == "context_cache" and rc == 0:
                    expected_bindings = context_cache.prepare_context_inputs_cpu()["context_bindings"]
                    result = context_cache.validate_final_cache(
                        context_cache.CACHE_ROOT, expected_bindings=expected_bindings)
                    valid = True
                    fields = {"child_returncode": rc, "manifest_sha256": result["manifest_sha256"],
                              "count": result["count"]}
                elif stage == "preflight" and rc in (0, 2):
                    current = preflight.validate_upstream()
                    report = json.loads(preflight.OUTPUT.read_text(encoding="utf-8"))
                    result = preflight.validate_report_record(report, expected=current)
                    valid = True
                    fields = {"child_returncode": rc, "report_sha256": sha256_file(preflight.OUTPUT),
                              "training_admission": result["training_admission"]}
                else:
                    valid = False; fields = {"child_returncode": rc, "artifacts_preserved": True}
                if not valid:
                    self._stage_update(stage, "failed_child_or_artifact_validation", **fields)
                    return False
                terminal = ("complete" if stage == "context_cache" else
                    ("complete_admitted" if result["training_admission"] else "complete_rejected"))
                self._stage_update(stage, terminal, **fields)
                return True
            finally:
                try:
                    if proc is not None and proc.poll() is None:
                        owner._stop_owned(proc)
                finally:
                    self.child = None
                    self._release_lease()
        return False

    def _release_lease(self):
        lease, self.lease = self.lease, None
        if lease is not None:
            try:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
            finally:
                lease.close()

    def run(self):
        try:
            self._update("validating_source_cache")
            # The first context-stage upstream check is the public CPU-only
            # source/cache admission, before CUDA use.
            self._verify_frozen_code()
            for stage in ("context_cache", "preflight"):
                upstream = self._upstream(stage)
                if self._existing(stage, upstream):
                    continue
                if not self._launch_stage(stage):
                    self._update(f"failed_{stage}_stage", artifacts_preserved=True,
                                 finished=time.time())
                    return 1
            report = json.loads(preflight.OUTPUT.read_text(encoding="utf-8"))
            expected = preflight.validate_upstream()
            preflight.validate_report_record(report, expected=expected)
            admitted = report["training_admission"] is True
            self._update("complete_admitted" if admitted else "complete_rejected",
                training_admission=admitted, preflight_sha256=sha256_file(preflight.OUTPUT),
                context_manifest_sha256=sha256_file(context_cache.CACHE_ROOT / "manifest.json"),
                finished=time.time())
            return 0 if admitted else 2
        except BaseException as exc:
            self._update("failed", error=repr(exc), artifacts_preserved=True, finished=time.time())
            raise
        finally:
            try:
                if self.child is not None and self.child.poll() is None:
                    owner._stop_owned(self.child)
            finally:
                self._release_lease()
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
    return ContextQueue().run()


if __name__ == "__main__":
    raise SystemExit(main())
