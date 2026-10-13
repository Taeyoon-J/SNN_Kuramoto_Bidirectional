"""One-shot owner-aware reservation for the read-only SW0153 TRAIN probe."""
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

try:
    import fcntl
except ImportError:  # CPU plan/tests work on Windows; GPU execution is Linux-only.
    fcntl = None

# This is a CPU supervisor. Only the lease-owning child receives a physical GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for item in (ROOT, ROOT / "collaborative_test"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4 as owner
from collaborative_test.SW_0135_native32_spike_binding import foundation as sw135
from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0153_gamma_ordering_audit import audit_source

EXPERIMENT = "SW0153_gamma_ordering_audit"
SEED = 1
COUNT = audit_source.DEFAULT_COUNT  # fixed default prefix: four ordered TRAIN images
DEVICE = "cuda:0"
RUNTIME = HERE / "results_archive" / "probe_queue_20261013"
STATE = RUNTIME / "state.json"
LOCK = RUNTIME / "queue.lock"
OUTPUT = HERE / "results_archive" / "gamma_ordering_source1_first4.json"
LOG = RUNTIME / "attempt01.log"
GPU_LEASE_DIR = Path("/tmp/kevinswk_sw0113_gpu_leases")
MAX_MEMORY_MIB = 512
STABLE_SECONDS = 5.0
SAMPLE_SECONDS = 2.0


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_report_fingerprint():
    result = dict(sw135.implementation_fingerprint())
    for path in (HERE / "audit_source.py", HERE / "README.md", HERE / "test_ordering.py"):
        if not path.is_file():
            raise FileNotFoundError(f"SW0153 report dependency missing: {path}")
        result[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    return result


def implementation_fingerprint():
    result = {"source_report": source_report_fingerprint()}
    paths = (HERE / "probe_queue.py", HERE / "test_probe_queue.py",
             ROOT / "collaborative_test/SW_0148_checkpointable_boundary_q/stage_queue_v4.py",
             ROOT / "collaborative_test/SW_0126_history_event_binding/pilot_queue.py")
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("SW0153 owner-queue dependency is missing")
    if sha256_file(paths[-1]) != owner.OWNER_GUARD_SHA256:
        raise ValueError("registered physical-GPU owner helper SHA changed")
    result["queue_dependencies"] = {
        path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths
    }
    return result


def implementation_sha256():
    raw = json.dumps(implementation_fingerprint(), sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def task_plan():
    return {"task_id": "source1_train_prefix_first4", "stage": "audit_source",
            "seed": SEED, "count": COUNT, "device": DEVICE,
            "output": str(OUTPUT), "log": str(LOG), "depends_on": []}


def command(task=None):
    task = task or task_plan()
    if (task.get("task_id") != "source1_train_prefix_first4"
            or task.get("stage") != "audit_source" or task.get("seed") != SEED
            or task.get("count") != COUNT or task.get("device") != DEVICE):
        raise ValueError("only the registered source1 first-four TRAIN audit may run")
    return [sys.executable, "-m", "collaborative_test.SW_0153_gamma_ordering_audit.audit_source",
            "--count", str(COUNT), "--device", DEVICE, "--output", str(task["output"])]


def _finite(value):
    try:
        import math
        return not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def validate_report(path=OUTPUT, *, expected_fingerprint=None):
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        if expected_fingerprint is None:
            expected_fingerprint = source_report_fingerprint()
        checkpoint, manifest_path, _manifest, pool, train_ids, source_sha = sw130.source_contract(SEED)
        ids = [int(value) for value in train_ids[:COUNT]]
        rows = [int(value) for value in pool[:COUNT]]
        if (report.get("status") != "complete" or report.get("experiment") != EXPERIMENT
                or report.get("seed") != SEED or report.get("count") != COUNT
                or report.get("ids") != ids or report.get("pool_rows") != rows
                or report.get("source_core_sha256") != source_sha
                or report.get("source_manifest_sha256") != sha256_file(manifest_path)
                or report.get("ground_truth_used") is not False
                or report.get("optimizer_updates") != 0
                or report.get("implementation_fingerprint") != expected_fingerprint
                or not isinstance(report.get("per_image"), list)
                or len(report["per_image"]) != COUNT):
            return False
        state_hashes = report["model_state_sha256_before_after"]
        if (set(state_hashes) != {"foundation", "encoder"}
                or any(not isinstance(value, str) or len(value) != 64
                       for value in state_hashes.values())):
            return False
        for row in report["per_image"]:
            if row.get("image_id") not in ids or set(row.get("variants", {})) != {
                    "naive_fixed_channel_shuffle", "naive_train_mean_channel_order",
                    "consistent_global_channel_reindex", "per_image_patch_permutation"}:
                return False
            for variant in row["variants"].values():
                for key, value in variant.items():
                    if key.endswith(("_mse", "_fraction", "max_abs")) and not _finite(value):
                        return False
        summary = report.get("mean_change_summary")
        if not isinstance(summary, dict) or set(summary) != {
                "naive_fixed_channel_shuffle", "naive_train_mean_channel_order",
                "consistent_global_channel_reindex", "per_image_patch_permutation"}:
            return False
        return all(_finite(value) for row in summary.values() for value in row.values())
    except (OSError, ValueError, TypeError, KeyError, AttributeError, json.JSONDecodeError):
        return False


def check_unwritten(paths):
    present = [str(Path(path)) for path in paths if Path(path).exists()]
    if present:
        raise FileExistsError(f"preserve existing SW0153 probe outputs: {present}")


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


class ProbeQueue:
    def __init__(self):
        if fcntl is None:
            raise RuntimeError("SW0153 GPU reservation requires Linux flock")
        RUNTIME.mkdir(parents=True, exist_ok=True)
        self.lock_stream = LOCK.open("a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock_stream.close()
            raise RuntimeError("another SW0153 probe supervisor owns the queue") from exc
        self.stop_event = threading.Event()
        self.task = task_plan()
        self.report_fingerprint = source_report_fingerprint()
        self.queue_fingerprint = implementation_fingerprint()
        self.existing_complete = False
        try:
            if OUTPUT.exists():
                if not validate_report(OUTPUT, expected_fingerprint=self.report_fingerprint):
                    raise FileExistsError("preserve invalid or differently-bound existing SW0153 output")
                self.existing_complete = True
            else:
                check_unwritten((STATE, LOG))
            if not self.existing_complete:
                _atomic_json(STATE, {"experiment": EXPERIMENT, "queue_version": 1,
                                     "status": "waiting_for_exclusive_gpu",
                                     "supervisor_pid": os.getpid(), "task": self.task,
                                     "report_fingerprint": self.report_fingerprint,
                                     "queue_fingerprint": self.queue_fingerprint,
                                     "training_admission": False,
                                     "ground_truth_used": False, "optimizer_updates": 0})
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
        if (source_report_fingerprint() != self.report_fingerprint
                or implementation_fingerprint() != self.queue_fingerprint):
            raise RuntimeError("SW0153 code/source dependencies changed after queue registration")

    @staticmethod
    def _terminate_owned(proc):
        if proc is not None and proc.poll() is None:
            owner._stop_owned(proc)

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
                    fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                    if self.stop_event.is_set():
                        return "cancelled_before_launch"
                    self._update("waiting_for_exclusive_gpu", reservation_released=True)
                    continue
                self._verify_fingerprints()
                check_unwritten((OUTPUT, LOG))
                argv = command(self.task)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                with Path(LOG).open("x", encoding="utf-8") as log:
                    proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                        pass_fds=(lease.fileno(),))
                    self._update("running", child_pid=proc.pid, child_argv=argv,
                        physical_gpu=gpu, gpu_uuid=gpu_uuid, memory_before_mib=memory,
                        inherited_lease_fd=True, log=str(LOG))
                    while proc.poll() is None and not self.stop_event.is_set():
                        time.sleep(SAMPLE_SECONDS)
                        if proc.poll() is not None:
                            break
                        try:
                            current_uuid, _used, owners = owner.gpu_guard._gpu_info(gpu)
                            foreign = (["gpu_uuid_changed"] if current_uuid != gpu_uuid else
                                owner.gpu_guard._foreign_owners(owners, owner.gpu_guard._process_tree(proc.pid)))
                        except Exception as exc:
                            self._update("running", last_gpu_query_error=repr(exc))
                            continue
                        if foreign:
                            self._terminate_owned(proc)
                            self._update("failed_foreign_gpu_owner", foreign=foreign)
                            return "interrupted_foreign_gpu_owner"
                rc = int(proc.wait())
                passed = (rc == 0 and validate_report(OUTPUT, expected_fingerprint=self.report_fingerprint))
                self._update("complete" if passed else "failed_child_or_report_validation",
                             child_returncode=rc, output_valid=bool(passed), finished=time.time())
                return "passed" if passed else "failed_child_or_report_validation"
            except BaseException as exc:
                self._terminate_owned(proc)
                self._update("failed", error=repr(exc), finished=time.time())
                raise
            finally:
                if lease is not None:
                    try:
                        fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                    finally:
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
    return ProbeQueue().run()


if __name__ == "__main__":
    raise SystemExit(main())
