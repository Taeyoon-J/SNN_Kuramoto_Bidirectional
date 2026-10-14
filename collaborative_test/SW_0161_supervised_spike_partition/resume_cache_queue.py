"""Create-once retry wrapper for the pre-GPU-failed SW161 cache reservation.

The original cache queue and its failed state remain immutable. This wrapper
admits exactly that recorded pre-GPU code failure after proving its owners are
gone and no cache artifacts were created, then runs the unchanged queue logic
under a new runtime namespace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for item in (ROOT, ROOT / "collaborative_test"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

try:
    import fcntl
except ImportError:
    fcntl = None

from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4 as owner
from collaborative_test.SW_0161_supervised_spike_partition import cache, cache_queue as frozen_queue

EXPERIMENT = frozen_queue.EXPERIMENT
OLD_RUNTIME = HERE / "results_archive" / "cache_preflight_queue_20261014"
RUNTIME = HERE / "results_archive" / "cache_preflight_queue_retry_20261014"
STATE, LOCK = RUNTIME / "state.json", RUNTIME / "queue.lock"
LOGS = {name: RUNTIME / f"{name}_attempt01.log" for name in ("cache", "preflight")}
OLD_STATE = OLD_RUNTIME / "state.json"
OLD_LOCK = OLD_RUNTIME / "queue.lock"
EXPECTED_ERROR = "AttributeError: module '...objective' has no attribute 'SLOTS'"
EXPECTED_OLD_SUPERVISOR = 3169828
EXPECTED_OLD_CHILD = 3169874


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint():
    paths = (HERE / "resume_cache_queue.py", HERE / "cache.py", HERE / "cache_queue.py",
             HERE / "protocol.json", Path(owner.__file__), owner.OWNER_GUARD)
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("SW161 retry queue dependency missing")
    if sha256_file(owner.OWNER_GUARD) != owner.OWNER_GUARD_SHA256:
        raise ValueError("registered owner guard changed")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def _pid_dead(pid):
    if type(pid) is not int or pid <= 0:
        return True
    row = owner._proc_record(pid)
    return row is None or row.get("state") == "Z"


def validate_retry_prerequisites(old_state_path=OLD_STATE, cache_root=cache.CACHE_ROOT,
                                 runtime=RUNTIME):
    """Fail closed unless the exact pre-GPU failure left no cache artifacts."""
    old_path = Path(old_state_path)
    if not old_path.is_file():
        raise FileNotFoundError("original failed SW161 queue receipt is missing")
    state = json.loads(old_path.read_text(encoding="utf-8"))
    if (state.get("experiment") != frozen_queue.EXPERIMENT
            or state.get("status") != "failed_cache_stage"):
        raise ValueError("old queue state is not the registered cache-stage failure")
    stage = state.get("stages", {}).get("cache", {})
    if (state.get("supervisor_pid") != EXPECTED_OLD_SUPERVISOR
            or stage.get("child_pid") != EXPECTED_OLD_CHILD):
        raise ValueError("old queue owner identities do not match the reviewed attempt")
    if stage.get("child_returncode") != 1:
        raise ValueError("old cache child did not record the expected failed exit")
    log_path = old_path.parent / "cache_attempt01.log"
    if not log_path.is_file():
        raise FileNotFoundError("original failed cache log is missing")
    error = log_path.read_text(encoding="utf-8", errors="replace")
    # Pin the observed failure class and missing attribute from the preserved
    # child log, not a generalized retry-on-error policy.
    if "AttributeError" not in error or "SLOTS" not in error:
        raise ValueError("old queue failure is not the reviewed pre-GPU SLOTS defect")
    if not _pid_dead(state.get("supervisor_pid")) or not _pid_dead(stage.get("child_pid")):
        raise RuntimeError("old SW161 cache owner or child is still live")
    output = Path(cache_root)
    if output.exists():
        raise FileExistsError("preserve unexpected source cache artifacts; retry is not safe")
    if frozen_queue.PREFLIGHT_OUTPUT.exists():
        raise FileExistsError("preserve unexpected disposable preflight artifact")
    retry_root = Path(runtime)
    if retry_root.exists() and any(retry_root.iterdir()):
        raise FileExistsError("preserve existing SW161 retry runtime artifacts")
    return {"old_state_sha256": sha256_file(old_path), "old_state": state,
            "old_state_path": str(old_path), "cache_root_absent": True,
            "retry_runtime": str(retry_root), "error_class": "AttributeError", "missing_symbol": "SLOTS"}


def _exclusive_old_lock(lock_path=OLD_LOCK):
    if fcntl is None:
        raise RuntimeError("SW161 retry requires Linux flock")
    path = Path(lock_path)
    if not path.is_file():
        raise FileNotFoundError("original SW161 queue lock is missing")
    stream = path.open("a+")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        stream.close()
        raise RuntimeError("original SW161 queue still owns its lock") from exc
    return stream


def run():
    prereq = validate_retry_prerequisites()
    old_lease = _exclusive_old_lock()
    try:
        # Recheck the immutable failure and absence after acquiring the old lock.
        current = validate_retry_prerequisites()
        if current["old_state_sha256"] != prereq["old_state_sha256"]:
            raise RuntimeError("original SW161 failure receipt changed during retry admission")
        frozen_queue.RUNTIME = RUNTIME
        frozen_queue.STATE, frozen_queue.LOCK = STATE, LOCK
        frozen_queue.LOGS = LOGS
        frozen_queue.EXPERIMENT = EXPERIMENT
        return frozen_queue.CacheQueue().run()
    finally:
        fcntl.flock(old_lease.fileno(), fcntl.LOCK_UN)
        old_lease.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    prereq = validate_retry_prerequisites()
    if args.dry_run:
        print(json.dumps({"status": "planned", "experiment": EXPERIMENT,
                          "runtime": str(RUNTIME), "old_state_sha256": prereq["old_state_sha256"],
                          "cache_root_absent": prereq["cache_root_absent"],
                          "fingerprint": implementation_fingerprint()}, sort_keys=True), flush=True)
        return 0
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
