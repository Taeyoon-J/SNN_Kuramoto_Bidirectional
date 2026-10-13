"""Owner-aware one-job GPU reservation for the SW0155 two-arm preflight."""
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

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for candidate in (ROOT, ROOT / "collaborative_test"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    import fcntl
except ImportError:
    fcntl = None

from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4 as owner
from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0155_supervised_spike_relations import preflight, train

EXPERIMENT = train.EXPERIMENT
RUNTIME = HERE / "results_archive" / "preflight_queue_20261013"
STATE, LOCK, LOG = RUNTIME / "state.json", RUNTIME / "queue.lock", RUNTIME / "attempt01.log"
OUTPUT = preflight.OUTPUT
GPU_LEASE_DIR = Path("/tmp/kevinswk_sw0113_gpu_leases")
MAX_MEMORY_MIB = 512
STABLE_SECONDS = 5.0
SAMPLE_SECONDS = 3.0
DEVICE = "cuda:0"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def report_fingerprint():
    return train.implementation_fingerprint()


def implementation_fingerprint():
    paths = (HERE / "preflight_queue.py", HERE / "test_preflight_queue.py",
        HERE / "preflight.py", HERE / "train.py", HERE / "objective.py",
        HERE / "protocol.json",
        ROOT / "collaborative_test/SW_0148_checkpointable_boundary_q/stage_queue_v4.py",
        ROOT / "collaborative_test/SW_0126_history_event_binding/pilot_queue.py")
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("SW0155 preflight queue dependency missing")
    if sha256_file(paths[-1]) != owner.OWNER_GUARD_SHA256:
        raise ValueError("registered shared GPU owner guard changed")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def implementation_sha256():
    raw = json.dumps(implementation_fingerprint(), sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def task_plan():
    return {"task_id": "source1_trainmask_two_arm_preflight", "stage": "actual_source_masked_two_update_preflight",
            "seed": 1, "device": DEVICE, "output": str(OUTPUT), "log": str(LOG), "depends_on": []}


def command(task=None):
    task = task or task_plan()
    registered = task_plan()
    for key in ("task_id", "stage", "seed", "device"):
        if task.get(key) != registered[key]:
            raise ValueError("only the registered SW0155 preflight may run")
    if Path(task.get("output", "")) != OUTPUT or Path(task.get("log", "")) != LOG:
        raise ValueError("SW0155 preflight output and log paths are fixed")
    return [sys.executable, "-m", "collaborative_test.SW_0155_supervised_spike_relations.preflight",
            "--device", DEVICE, "--output", str(OUTPUT)]


def validate_report(path=OUTPUT, *, expected_fingerprint=None):
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if expected_fingerprint is None:
            expected_fingerprint = report_fingerprint()
        source = train.source_contract()
        assets = sw130.validate_rgb_assets()
        stat = train.HDF5_PATH.stat()
        return preflight.validate_admission(report, source=source, assets=assets,
            hdf5_path=train.HDF5_PATH, hdf5_size=stat.st_size, hdf5_mtime_ns=stat.st_mtime_ns
        ) and report.get("implementation_fingerprint") == expected_fingerprint
    except (OSError, ValueError, TypeError, KeyError, AttributeError, json.JSONDecodeError):
        return False


def _check_unwritten(paths):
    present = [str(Path(path)) for path in paths if Path(path).exists()]
    if present:
        raise FileExistsError(f"preserve existing SW0155 preflight artifacts: {present}")


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


class PreflightQueue:
    def __init__(self):
        if fcntl is None:
            raise RuntimeError("SW0155 GPU reservation requires Linux flock")
        RUNTIME.mkdir(parents=True, exist_ok=True)
        self.lock_stream = LOCK.open("a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock_stream.close()
            raise RuntimeError("another SW0155 preflight supervisor owns the queue") from exc
        self.stop_event = threading.Event()
        self.task = task_plan()
        self.report_fp = report_fingerprint()
        self.queue_fp = implementation_fingerprint()
        self.existing_complete = False
        try:
            if OUTPUT.exists():
                if not validate_report(OUTPUT, expected_fingerprint=self.report_fp):
                    raise FileExistsError("preserve invalid or differently bound SW0155 preflight output")
                self.existing_complete = True
            else:
                _check_unwritten((STATE, LOG))
                _atomic_json(STATE, {"experiment": EXPERIMENT, "queue_version": 1,
                    "status": "waiting_for_exclusive_gpu", "supervisor_pid": os.getpid(),
                    "task": self.task, "report_fingerprint": self.report_fp,
                    "queue_fingerprint": self.queue_fp, "ground_truth_used": True,
                    "ground_truth_scope": "TRAIN masks only"})
        except BaseException:
            self._unlock()
            raise

    def _unlock(self):
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_UN)
        finally:
            self.lock_stream.close()

    def _update(self, status, **fields):
        state = json.loads(STATE.read_text(encoding="utf-8"))
        state.update(status=status, updated=time.time(), **fields)
        _atomic_json(STATE, state)

    def _verify_fingerprints(self):
        if report_fingerprint() != self.report_fp or implementation_fingerprint() != self.queue_fp:
            raise RuntimeError("SW0155 preflight dependencies changed after queue registration")

    def launch_once(self):
        if self.existing_complete:
            return "verified_existing"
        proc = lease = None
        self._verify_fingerprints()
        while not self.stop_event.is_set():
            gpu, gpu_uuid, memory, lease = owner._acquire(self.stop_event)
            try:
                stable = owner._stable_idle(gpu, gpu_uuid, self.stop_event,
                    lambda elapsed, used: self._update("waiting_stable_gpu", gpu=gpu,
                        gpu_uuid=gpu_uuid, stable_seconds=elapsed, memory_used_mib=used))
                if not stable:
                    if fcntl is not None:
                        fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                    lease.close()
                    lease = None
                    if self.stop_event.is_set():
                        return "cancelled_before_launch"
                    self._update("waiting_for_exclusive_gpu", reservation_released=True)
                    continue
                self._verify_fingerprints()
                _check_unwritten((OUTPUT, LOG))
                argv = command(self.task)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                with Path(LOG).open("x", encoding="utf-8") as log:
                    proc = subprocess.Popen(argv, cwd=ROOT, env=env,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True, pass_fds=(lease.fileno(),))
                    self._update("running", child_pid=proc.pid, child_argv=argv,
                        physical_gpu=gpu, gpu_uuid=gpu_uuid, memory_before_mib=memory,
                        inherited_lease_fd=True, log=str(LOG))
                    while proc.poll() is None and not self.stop_event.is_set():
                        time.sleep(SAMPLE_SECONDS)
                        if proc.poll() is not None:
                            break
                        try:
                            current_uuid, _, owners = owner.gpu_guard._gpu_info(gpu)
                            foreign = (["gpu_uuid_changed"] if current_uuid != gpu_uuid else
                                owner.gpu_guard._foreign_owners(owners,
                                    owner.gpu_guard._process_tree(proc.pid)))
                        except Exception as exc:
                            self._update("running", last_gpu_query_error=repr(exc))
                            continue
                        if foreign:
                            owner._stop_owned(proc)
                            self._update("failed_foreign_gpu_owner", foreign=foreign,
                                         output_preserved=OUTPUT.exists())
                            return "interrupted_foreign_gpu_owner"
                rc = int(proc.wait())
                passed = rc == 0 and validate_report(OUTPUT, expected_fingerprint=self.report_fp)
                self._update("complete" if passed else "failed_child_or_report_validation",
                    child_returncode=rc, output_valid=bool(passed), finished=time.time())
                return "passed" if passed else "failed_child_or_report_validation"
            except BaseException as exc:
                if proc is not None and proc.poll() is None:
                    owner._stop_owned(proc)
                self._update("failed", error=repr(exc), finished=time.time())
                raise
            finally:
                if lease is not None:
                    if fcntl is not None:
                        fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                    lease.close()
        return "cancelled_before_launch"

    def run(self):
        try:
            if self.existing_complete:
                return 0
            result = self.launch_once()
            return 0 if result in {"passed", "verified_existing"} else 1
        finally:
            self._unlock()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"status": "planned", "task": task_plan(), "argv": command(),
            "cuda_visible_devices_parent": "", "queue_fingerprint": implementation_sha256()},
            allow_nan=False), flush=True)
        return 0
    return PreflightQueue().run()


if __name__ == "__main__":
    raise SystemExit(main())
