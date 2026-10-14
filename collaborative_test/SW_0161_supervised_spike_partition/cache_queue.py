"""Owner-safe coordinator for the SW161 source cache and disposable preflight."""
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

# The CLI is a CPU supervisor. Library imports preserve a caller's assigned GPU.
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
except ImportError:
    fcntl = None

# SW0148 hides CUDA at import. Restore the exact environment for GPU-child imports.
_unset = object()
_visibility_before_owner = os.environ.get("CUDA_VISIBLE_DEVICES", _unset)
from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4 as owner
if _visibility_before_owner is _unset:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = _visibility_before_owner

from collaborative_test.SW_0161_supervised_spike_partition import cache, preflight, train

EXPERIMENT = "SW0161_source_cache_and_head_preflight_queue"
RUNTIME = HERE / "results_archive" / "cache_preflight_queue_20261014"
STATE, LOCK = RUNTIME / "state.json", RUNTIME / "queue.lock"
LOGS = {"cache": RUNTIME / "cache_attempt01.log",
        "preflight": RUNTIME / "preflight_attempt01.log"}
CACHE_ROOT = cache.CACHE_ROOT
PREFLIGHT_OUTPUT = preflight.PREFLIGHT_OUTPUT
GPU_LEASE_DIR = owner.GPU_LEASE_DIR
DEVICE = "cuda:0"
MAX_MEMORY_MIB = 512
POLL_SECONDS = 3.0
SAMPLE_SECONDS = 2.0
WATCHDOG_SECONDS = 12 * 60 * 60


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint():
    paths = (HERE / "cache_queue.py", HERE / "cache.py", HERE / "preflight.py",
             HERE / "train.py", HERE / "objective.py", HERE / "protocol.json",
             Path(owner.__file__), owner.OWNER_GUARD)
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("SW161 queue dependency is missing")
    if sha256_file(owner.OWNER_GUARD) != owner.OWNER_GUARD_SHA256:
        raise ValueError("registered shared GPU owner guard changed")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def stage_command(stage):
    if stage == "cache":
        return [sys.executable, "-m", "collaborative_test.SW_0161_supervised_spike_partition.cache",
                "--device", DEVICE, "--output", str(CACHE_ROOT)]
    if stage == "preflight":
        return [sys.executable, "-m", "collaborative_test.SW_0161_supervised_spike_partition.preflight",
                "--device", DEVICE, "--cache-root", str(CACHE_ROOT),
                "--output", str(PREFLIGHT_OUTPUT)]
    raise ValueError(f"unknown SW161 queue stage: {stage}")


def dry_run_plan():
    return {"experiment": EXPERIMENT, "stages": ["cache", "preflight"],
            "cache_argv": stage_command("cache"),
            "preflight_argv": stage_command("preflight"),
            "cache_root": str(CACHE_ROOT), "preflight_output": str(PREFLIGHT_OUTPUT),
            "runtime": str(RUNTIME), "gpu_candidates": [0, 1, 2, 3],
            "max_memory_mib": MAX_MEMORY_MIB, "stable_seconds": owner.STABLE_SECONDS,
            "watchdog_seconds": WATCHDOG_SECONDS,
            "queue_fingerprint": implementation_fingerprint()}


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _assert_state_unwritten(path=STATE):
    if Path(path).exists():
        raise FileExistsError(f"preserve existing SW161 queue state: {path}")


def _score_dependency():
    """RuntimeError means a live upstream owner is still publishing the score."""
    return cache._current_score_dependency()


def _wait_score(stop_event, on_wait=None):
    while not stop_event.is_set():
        try:
            dependency = _score_dependency()
            return dependency
        except RuntimeError as exc:
            if on_wait:
                on_wait(str(exc))
            stop_event.wait(POLL_SECONDS)
    raise RuntimeError("queue cancelled while waiting for SW0160 terminal score")


def _queue_report(stage, cache_manifest=None):
    if stage != "preflight":
        raise ValueError("only the preflight stage has a report validator")
    upstream = preflight.validate_upstream(CACHE_ROOT)
    report = json.loads(Path(PREFLIGHT_OUTPUT).read_text(encoding="utf-8"))
    valid = preflight.validate_report_record(report, expected=upstream)
    if cache_manifest is not None and report.get("cache_manifest_sha256") != sha256_file(CACHE_ROOT / "manifest.json"):
        raise ValueError("preflight report does not bind the admitted cache manifest")
    return valid


def _validate_cache():
    return train.validate_cache(CACHE_ROOT)


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


def _foreign_gpu_owners(gpu, expected_uuid, child_pid):
    current_uuid, _memory, owners = owner.gpu_guard._gpu_info(gpu)
    if current_uuid != expected_uuid:
        return ["gpu_uuid_changed"]
    return owner.gpu_guard._foreign_owners(
        owners, owner.gpu_guard._process_tree(child_pid))


class CacheQueue:
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if fcntl is None:
            raise RuntimeError("SW161 cache queue requires Linux flock")
        RUNTIME.mkdir(parents=True, exist_ok=True)
        self.lock_stream = LOCK.open("a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock_stream.close()
            raise RuntimeError("another SW161 cache supervisor owns the queue") from exc
        self.stop_event = threading.Event()
        self.state_lock = threading.RLock()
        self.queue_fp = implementation_fingerprint()
        self.cache_fp = cache.implementation_fingerprint()
        self.preflight_fp = preflight.implementation_fingerprint()
        self.plan = dry_run_plan()
        self.existing_state = STATE.exists()
        self._child = None
        self._lease = None
        try:
            _assert_state_unwritten(STATE)
            _atomic_json(STATE, {"experiment": EXPERIMENT, "status": "waiting_for_sw0160_score",
                "supervisor_pid": os.getpid(), "queue_fingerprint": self.queue_fp,
                "cache_fingerprint": self.cache_fp, "preflight_fingerprint": self.preflight_fp,
                "stages": {}, "task_plan": self.plan, "created": time.time()})
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
            state = json.loads(STATE.read_text(encoding="utf-8"))
            state.update(status=status, updated=time.time(), **fields)
            _atomic_json(STATE, state)

    def _stage_update(self, stage, status, **fields):
        with self.state_lock:
            state = json.loads(STATE.read_text(encoding="utf-8"))
            stages = state.setdefault("stages", {})
            item = dict(stages.get(stage, {}))
            item.update(status=status, updated=time.time(), **fields)
            stages[stage] = item
            state.update(status=f"{stage}_{status}", updated=time.time())
            _atomic_json(STATE, state)

    def _verify_code(self):
        if implementation_fingerprint() != self.queue_fp:
            raise RuntimeError("SW161 queue or registered dependencies changed")
        if cache.implementation_fingerprint() != self.cache_fp:
            raise RuntimeError("SW161 cache implementation changed")
        if preflight.implementation_fingerprint() != self.preflight_fp:
            raise RuntimeError("SW161 preflight implementation changed")

    def _prepare_existing(self, stage, cache_manifest=None):
        if stage == "cache":
            manifest_path = Path(CACHE_ROOT) / "manifest.json"
            if manifest_path.is_file():
                manifest = _validate_cache()
                self._stage_update(stage, "adopted", manifest_sha256=sha256_file(manifest_path),
                                   cache_count=manifest.get("count"))
                return True
            # Partial cache remains in place; cache runner resumes verified rows.
            return False
        if Path(PREFLIGHT_OUTPUT).is_file():
            report = _queue_report("preflight", cache_manifest=(
                cache_manifest if cache_manifest is not None else _validate_cache()))
            admitted = report["training_admission"] is True
            self._stage_update(stage, "adopted", report_sha256=sha256_file(PREFLIGHT_OUTPUT),
                               training_admission=admitted,
                               receipt_status="complete_admitted" if admitted else "complete_rejected")
            return True
        return False

    def _acquire_and_launch(self, stage, upstream_digest):
        while not self.stop_event.is_set():
            gpu, gpu_uuid, memory, lease = owner._acquire(self.stop_event)
            self._lease = lease
            proc = None
            try:
                stable = owner._stable_idle(gpu, gpu_uuid, self.stop_event,
                    lambda elapsed, used: self._stage_update(stage, "waiting_stable_gpu",
                        gpu=gpu, gpu_uuid=gpu_uuid, stable_seconds=elapsed,
                        memory_used_mib=used))
                if not stable:
                    self._release_lease()
                    if self.stop_event.is_set():
                        return False
                    continue
                self._verify_code()
                # Expensive revalidation is deliberately before the final under-lease GPU sample.
                if stage == "cache":
                    current_upstream = _score_dependency()
                else:
                    current_upstream = preflight.validate_upstream(CACHE_ROOT)
                current_digest = hashlib.sha256(json.dumps(current_upstream, sort_keys=True,
                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
                if current_digest != upstream_digest:
                    raise RuntimeError(f"SW161 {stage} upstream changed while waiting for GPU")
                if self._prepare_existing(stage):
                    return True
                admitted, state = _final_gpu_admission(gpu, gpu_uuid)
                if not admitted:
                    self._stage_update(stage, "waiting_for_exclusive_gpu",
                        final_gpu_check=state, reservation_released=True)
                    self._release_lease()
                    continue
                argv = stage_command(stage)
                log_path = LOGS[stage]
                log_path.parent.mkdir(parents=True, exist_ok=True)
                if log_path.exists():
                    raise FileExistsError(f"preserve existing {stage} attempt log: {log_path}")
                child_env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                                 OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                started = time.monotonic()
                with log_path.open("x", encoding="utf-8") as log:
                    proc = subprocess.Popen(argv, cwd=ROOT, env=child_env,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True, pass_fds=(lease.fileno(),))
                    self._child = proc
                    self._stage_update(stage, "running", child_pid=proc.pid,
                        child_pgid=proc.pid, child_argv=argv, physical_gpu=gpu,
                        gpu_uuid=gpu_uuid, memory_before_mib=memory,
                        inherited_lease_fd=True, log=str(log_path), upstream_sha256=upstream_digest)
                    while proc.poll() is None and not self.stop_event.is_set():
                        if time.monotonic() - started > WATCHDOG_SECONDS:
                            owner._stop_owned(proc)
                            self._stage_update(stage, "failed_watchdog", child_pid=proc.pid,
                                               artifacts_preserved=True)
                            return False
                        time.sleep(SAMPLE_SECONDS)
                        if proc.poll() is not None:
                            break
                        try:
                            foreign = _foreign_gpu_owners(gpu, gpu_uuid, proc.pid)
                            owners = list(foreign)
                        except Exception:
                            continue
                        if foreign:
                            owner._stop_owned(proc)
                            self._stage_update(stage, "failed_foreign_gpu_owner",
                                child_pid=proc.pid, foreign_owner_pids=list(owners),
                                artifacts_preserved=True)
                            return False
                    if self.stop_event.is_set() and proc.poll() is None:
                        owner._stop_owned(proc)
                        self._stage_update(stage, "cancelled_owned_child_stopped",
                                           child_pid=proc.pid, artifacts_preserved=True)
                        return False
                    rc = int(proc.wait())
                if stage == "cache":
                    result = _validate_cache() if rc == 0 else None
                    valid = result is not None
                    fields = {"child_returncode": rc, "cache_valid": valid,
                              "cache_manifest_sha256": sha256_file(Path(CACHE_ROOT) / "manifest.json")
                                  if valid else None}
                else:
                    result = _queue_report("preflight", cache_manifest=_validate_cache()) if rc in (0, 2) else None
                    valid = result is not None
                    fields = {"child_returncode": rc, "report_valid": valid,
                              "training_admission": result.get("training_admission") if valid else None,
                              "report_sha256": sha256_file(PREFLIGHT_OUTPUT) if valid else None}
                if not valid:
                    self._stage_update(stage, "failed_child_or_artifact_validation", **fields)
                    return False
                status = "complete" if stage == "cache" else (
                    "complete_admitted" if result["training_admission"] else "complete_rejected")
                self._stage_update(stage, status, **fields)
                return True
            finally:
                try:
                    if proc is not None and proc.poll() is None:
                        owner._stop_owned(proc)
                finally:
                    self._child = None
                    self._release_lease()
        return False

    def _release_lease(self):
        lease, self._lease = self._lease, None
        if lease is not None:
            try:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
            finally:
                lease.close()

    def run(self):
        try:
            self._update("waiting_for_sw0160_score")
            score_dependency = _wait_score(self.stop_event,
                lambda detail: self._update("waiting_for_sw0160_score", wait_reason=detail))
            score_digest = hashlib.sha256(json.dumps(score_dependency, sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
            self._verify_code()
            if not self._prepare_existing("cache"):
                if not self._acquire_and_launch("cache", score_digest):
                    self._update("failed_cache_stage", artifacts_preserved=True)
                    return 1
            cache_manifest = _validate_cache()
            self._update("cache_validated", cache_manifest_sha256=sha256_file(CACHE_ROOT / "manifest.json"),
                         cache_bindings_sha256=cache_manifest["bindings_sha256"])
            if self._prepare_existing("preflight", cache_manifest=cache_manifest):
                report = _queue_report("preflight", cache_manifest=cache_manifest)
                status = "complete_admitted" if report["training_admission"] else "complete_rejected"
                self._update(status, preflight_sha256=sha256_file(PREFLIGHT_OUTPUT),
                             training_admission=report["training_admission"])
                return 0 if report["training_admission"] else 2
            upstream = preflight.validate_upstream(CACHE_ROOT)
            upstream_digest = hashlib.sha256(json.dumps(upstream, sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
            if not self._acquire_and_launch("preflight", upstream_digest):
                self._update("failed_preflight_stage", artifacts_preserved=True)
                return 1
            report = _queue_report("preflight", cache_manifest=_validate_cache())
            admitted = report["training_admission"] is True
            self._update("complete_admitted" if admitted else "complete_rejected",
                training_admission=admitted, preflight_sha256=sha256_file(PREFLIGHT_OUTPUT),
                finished=time.time())
            return 0 if admitted else 2
        except BaseException as exc:
            self._update("failed", error=repr(exc), artifacts_preserved=True, finished=time.time())
            raise
        finally:
            try:
                if self._child is not None and self._child.poll() is None:
                    owner._stop_owned(self._child)
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
    return CacheQueue().run()


if __name__ == "__main__":
    raise SystemExit(main())
