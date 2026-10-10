"""One-time no-retry recovery of the interrupted seed-1 preflight only."""
from __future__ import annotations

import json
import os
import pathlib
import signal
import subprocess
import sys
import time

try:
    import fcntl
except ImportError:
    fcntl = None

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0123_adaptive_temporal_assignment import dispatcher as owner
from collaborative_test.SW_0134_native_spike_binding import run, preflight_queue

ARCHIVE = HERE / "results_archive"
OLD_STATE = ARCHIVE / "preflight_queue_state.json"
STATE = ARCHIVE / "recovery_preflight_seed1_state.json"
LOCK = ARCHIVE / "recovery_preflight_seed1.lock"
LOG = ARCHIVE / "recovery_preflight_seed1.log"
LEASES = pathlib.Path("/tmp/kevinswk_sw0113_gpu_leases")
MAX_MEMORY_MIB = 512


def source_failure_is_recoverable(state, *, seed1_log, canonical_outputs):
    """Require the original queue to be terminal and the interrupted attempt empty."""
    if not isinstance(state, dict) or state.get("status") not in {
            "scientific_preflight_failed", "all_preflights_passed", "seed0_preflight_failed"}:
        return False
    supervisor = state.get("supervisor_pid")
    if not isinstance(supervisor, int) or supervisor <= 0:
        return False
    tasks = state.get("tasks", {})
    failed = [row for row in tasks.values()
              if row.get("task", {}).get("seed") == 1]
    if len(failed) != 1:
        return False
    row = failed[0]
    if row.get("status") != "interrupted_foreign_gpu_owner":
        return False
    foreign = row.get("foreign_owner_pids")
    if not isinstance(foreign, list) or not foreign:
        return False
    if row.get("artifact_valid") is True:
        return False
    child = row.get("child_pid")
    if not isinstance(child, int) or child <= 0:
        return False
    try:
        if pathlib.Path(f"/proc/{child}").exists() or pathlib.Path(f"/proc/{supervisor}").exists():
            return False
    except OSError:
        return False
    try:
        if pathlib.Path(seed1_log).stat().st_size != 0:
            return False
    except OSError:
        return False
    return not any(pathlib.Path(p).exists() for p in canonical_outputs)


def command(device="cuda:0"):
    return [sys.executable, str(HERE / "run.py"), "--stage", "preflight", "--seed", "1",
            "--device", device, "--output", str(ARCHIVE / "preflight_seed1.json")]


def _atomic(path, payload):
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    argv = command()
    if args.dry_run:
        print(json.dumps({"recovery": "seed1_preflight_only", "argv": argv,
                          "original_state_preserved": True, "no_retries": True},
                         indent=2, allow_nan=False))
        return 0
    if fcntl is None:
        raise RuntimeError("seed1 recovery requires Linux flock")
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    lock = LOCK.open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise RuntimeError("another seed1 recovery owns the lock") from exc
    try:
        if STATE.exists() or LOG.exists() or (ARCHIVE / "preflight_seed1.json").exists():
            raise FileExistsError("preserve existing SW0134 seed1 recovery/output artifacts")
        if not OLD_STATE.is_file():
            raise FileNotFoundError("original SW0134 queue state is required")
        old = json.loads(OLD_STATE.read_text(encoding="utf-8"))
        warm = [ARCHIVE / f"warm_{arm}_seed1.pt" for arm in ("actual_joint", "gate_joint")]
        failed_record = ARCHIVE / "preflight_seed1.failure.json"
        if not source_failure_is_recoverable(
                old, seed1_log=ARCHIVE / "sw0134_preflight_s1.log",
                canonical_outputs=[*warm, failed_record]):
            raise RuntimeError("original seed1 attempt is not safely recoverable yet")
        receipt = {"experiment": "SW0134_native_spike_binding",
                   "status": "running", "stage": "seed1_preflight_recovery_only",
                   "supervisor_pid": os.getpid(), "original_state_sha256": run.sha(OLD_STATE),
                   "original_seed1_log_sha256": run.sha(ARCHIVE / "sw0134_preflight_s1.log"),
                   "argv": argv, "no_training": True, "no_retries": True,
                   "started": time.time()}
        _atomic(STATE, receipt)
        gpu_lease = None
        proc = None
        try:
            while gpu_lease is None:
                for gpu in range(4):
                    lease_path = LEASES / f"gpu{gpu}.lock"
                    lease_path.parent.mkdir(parents=True, exist_ok=True)
                    candidate = lease_path.open("a+")
                    try:
                        fcntl.flock(candidate.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        candidate.close(); continue
                    try:
                        uuid, memory, owners = owner._nvidia_gpu_info(gpu)
                    except BaseException:
                        fcntl.flock(candidate.fileno(), fcntl.LOCK_UN)
                        candidate.close()
                        raise
                    if not owners and memory <= MAX_MEMORY_MIB:
                        gpu_lease = candidate
                        receipt.update(gpu=gpu, gpu_uuid=uuid, memory_before_mib=memory,
                                       lease_path=str(lease_path))
                        break
                    fcntl.flock(candidate.fileno(), fcntl.LOCK_UN); candidate.close()
                if gpu_lease is None:
                    time.sleep(5)
            triton = pathlib.Path(f"/tmp/kevinswk_sw0134_recovery_seed1_{int(time.time()*1000)}")
            triton.mkdir(parents=True, exist_ok=False)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(receipt["gpu"]),
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", TRITON_CACHE_DIR=str(triton))
            receipt.update(status="running", triton_cache=str(triton), launched=time.time())
            _atomic(STATE, receipt)
            with LOG.open("x", encoding="utf-8") as output:
                proc = subprocess.Popen(argv, cwd=str(ROOT), env=env, stdout=output,
                                        stderr=subprocess.STDOUT, start_new_session=True,
                                        pass_fds=(gpu_lease.fileno(),))
                receipt["child_pid"] = proc.pid
                _atomic(STATE, receipt)
                while proc.poll() is None:
                    time.sleep(5)
                    if proc.poll() is not None:
                        break
                    _uuid, _memory, owners = owner._nvidia_gpu_info(receipt["gpu"])
                    foreign = owner.foreign_owner_pids(owners, owner._process_tree(proc.pid))
                    if foreign:
                        preflight_queue.PreflightQueue._terminate_owned_child(proc)
                        receipt.update(status="interrupted_foreign_owner",
                                       foreign_owner_pids=foreign, finished=time.time())
                        _atomic(STATE, receipt)
                        return 1
            rc = int(proc.returncode)
            valid = rc == 0 and preflight_queue.valid_result(
                {"stage": "preflight", "seed": 1})
            receipt.update(status="passed" if valid else "failed", returncode=rc,
                           artifact_valid=bool(valid), finished=time.time(),
                           output_sha256=(run.sha(ARCHIVE / "preflight_seed1.json")
                                          if (ARCHIVE / "preflight_seed1.json").is_file() else None))
            _atomic(STATE, receipt)
            return 0 if valid else 1
        finally:
            if proc is not None and proc.poll() is None:
                preflight_queue.PreflightQueue._terminate_owned_child(proc)
            if gpu_lease is not None:
                fcntl.flock(gpu_lease.fileno(), fcntl.LOCK_UN)
                gpu_lease.close()
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
