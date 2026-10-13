"""Owner-aware, one-image reservation for the SW0154 native64 resource probe."""
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
import numpy as np

try:
    import fcntl
except ImportError:  # CPU plan/tests work on Windows; real GPU reservation is Linux-only.
    fcntl = None

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for candidate in (ROOT, ROOT / "collaborative_test"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4 as owner
from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0154_native64_resolution_compare import resource_probe

EXPERIMENT = "SW0154_native64_resolution_compare"
SEED = 1
DEVICE = "cuda:0"
NODES = 4096
STEPS = 1024
SETTLE = 512
RUNTIME = HERE / "results_archive" / "resource_queue_20261013"
STATE = RUNTIME / "state.json"
LOCK = RUNTIME / "queue.lock"
OUTPUT = HERE / "results_archive" / resource_probe.REPORT_NAME
LOG = RUNTIME / "attempt01.log"
GPU_LEASE_DIR = Path("/tmp/kevinswk_sw0113_gpu_leases")
MAX_MEMORY_MIB = 512
STABLE_SECONDS = 5.0
SAMPLE_SECONDS = 3.0


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def report_fingerprint():
    return resource_probe.implementation_fingerprint()


def implementation_fingerprint():
    paths = (
        HERE / "resource_queue.py", HERE / "test_resource_queue.py",
        HERE / "resource_probe.py", HERE / "native64_model.py", HERE / "resolution64.py",
        HERE / "design.json",
        ROOT / "collaborative_test/SW_0148_checkpointable_boundary_q/stage_queue_v4.py",
        ROOT / "collaborative_test/SW_0126_history_event_binding/pilot_queue.py",
    )
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("SW0154 resource queue dependency missing")
    if sha256_file(paths[-1]) != owner.OWNER_GUARD_SHA256:
        raise ValueError("registered physical GPU owner helper SHA changed")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def implementation_sha256():
    raw = json.dumps(implementation_fingerprint(), sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def task_plan():
    return {"task_id": "source1_first_train_native64_resource",
            "stage": "source_only_resource_probe", "seed": SEED,
            "device": DEVICE, "output": str(OUTPUT), "log": str(LOG), "depends_on": []}


def command(task=None):
    task = task or task_plan()
    expected = task_plan()
    for key in ("task_id", "stage", "seed", "device"):
        if task.get(key) != expected[key]:
            raise ValueError("only the registered one-image native64 resource probe may run")
    if Path(task.get("output", "")) != OUTPUT or Path(task.get("log", "")) != LOG:
        raise ValueError("resource probe output/log path is fixed by registration")
    return [sys.executable, "-m", "collaborative_test.SW_0154_native64_resolution_compare.resource_probe",
            "--device", DEVICE, "--output", str(OUTPUT)]


def _finite(value):
    try:
        import math
        return not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def validate_report(path=OUTPUT, *, expected_fingerprint=None):
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        if expected_fingerprint is None:
            expected_fingerprint = report_fingerprint()
        _checkpoint, manifest_path, manifest, rows, ids, source_sha = sw130.source_contract(SEED)
        assets = sw130.validate_rgb_assets()
        parity_path = (ROOT / "collaborative_test/SW_0135_native32_spike_binding"
                       / "results_archive/native32_zero_adapter_parity_seed1_b4.json")
        expected_shapes = {
            "component_spikes": [1, 4, NODES, STEPS],
            "component_membrane": [1, 4, NODES, STEPS],
            "component_gates": [1, 4, NODES, STEPS],
            "spikes": [1, NODES, STEPS],
            "membrane": [1, NODES, STEPS],
            "theta": [1, STEPS, NODES, 4],
        }
        trace = record.get("rollout", {}).get("trace_fields", {})
        if (record.get("status") != "complete"
                or record.get("experiment") != EXPERIMENT
                or record.get("stage") != "source_only_resource_probe"
                or record.get("seed") != SEED
                or record.get("source_image_id") != int(ids[0])
                or record.get("source_pool_row") != int(rows[0])
                or record.get("source_training_ids_sha256") != hashlib.sha256(
                    np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
                or record.get("source_core_sha256") != source_sha
                or record.get("source_manifest_sha256") != sha256_file(manifest_path)
                or record.get("source_manifest_status") != manifest.get("status")
                or record.get("rgb_asset_validation") != assets
                or record.get("native32_parity_path") != str(parity_path)
                or record.get("native32_parity_sha256") != sha256_file(parity_path)
                or record.get("implementation_fingerprint") != expected_fingerprint
                or record.get("ground_truth_used") is not False
                or record.get("masks_read") is not False
                or record.get("rollout", {}).get("optimizer_updates") != 0
                or record.get("training_admission") is not False
                or record.get("native_grid") != [64, 64]
                or record.get("node_count") != NODES
                or record.get("gamma_shape") != [1, 8, NODES]
                or record.get("rollout", {}).get("total_steps") != STEPS
                or record.get("rollout", {}).get("settle") != SETTLE
                or record.get("rollout", {}).get("batch_size") != 1
                or record.get("readout", {}).get("q_shape") != [1, NODES, NODES]
                or record.get("readout", {}).get("labels_shape") != [1, 64, 64]
                or set(trace) != set(expected_shapes)
                or any(trace[key].get("shape") != shape or trace[key].get("finite") is not True
                       for key, shape in expected_shapes.items())):
            return False
        before, after = record.get("model_state_sha256_before"), record.get("model_state_sha256_after")
        if (not isinstance(before, dict) or set(before) != {"wrapped", "encoder"}
                or after != before
                or any(not isinstance(value, str) or len(value) != 64 for value in before.values())):
            return False
        resource = record.get("resource", {})
        return all(_finite(resource.get(key)) and float(resource[key]) >= 0 for key in
                   ("graph_seconds", "rollout_seconds", "q_qcc_seconds",
                    "peak_allocated_bytes", "peak_reserved_bytes"))
    except (OSError, ValueError, TypeError, KeyError, AttributeError, json.JSONDecodeError):
        return False


def _check_unwritten(paths):
    present = [str(Path(path)) for path in paths if Path(path).exists()]
    if present:
        raise FileExistsError(f"preserve existing SW0154 resource artifacts: {present}")


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


class ResourceQueue:
    def __init__(self):
        if fcntl is None:
            raise RuntimeError("SW0154 GPU reservation requires Linux flock")
        RUNTIME.mkdir(parents=True, exist_ok=True)
        self.lock_stream = LOCK.open("a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock_stream.close()
            raise RuntimeError("another SW0154 resource supervisor owns the queue") from exc
        self.stop_event = threading.Event()
        self.task = task_plan()
        self.report_fp = report_fingerprint()
        self.queue_fp = implementation_fingerprint()
        self.existing_complete = False
        try:
            if OUTPUT.exists():
                if not validate_report(OUTPUT, expected_fingerprint=self.report_fp):
                    raise FileExistsError("preserve invalid or differently-bound existing resource report")
                self.existing_complete = True
            else:
                _check_unwritten((STATE, LOG))
                _atomic_json(STATE, {"experiment": EXPERIMENT, "queue_version": 1,
                    "status": "waiting_for_exclusive_gpu", "supervisor_pid": os.getpid(),
                    "task": self.task, "report_fingerprint": self.report_fp,
                    "queue_fingerprint": self.queue_fp, "ground_truth_used": False,
                    "masks_read": False, "optimizer_updates": 0, "training_admission": False})
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
            raise RuntimeError("SW0154 code/source dependencies changed after queue registration")

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
                            current_uuid, _used, owners = owner.gpu_guard._gpu_info(gpu)
                            foreign = (["gpu_uuid_changed"] if current_uuid != gpu_uuid else
                                owner.gpu_guard._foreign_owners(owners,
                                    owner.gpu_guard._process_tree(proc.pid)))
                        except Exception as exc:
                            self._update("running", last_gpu_query_error=repr(exc))
                            continue
                        if foreign:
                            self._terminate_owned(proc)
                            self._update("failed_foreign_gpu_owner", foreign=foreign)
                            return "interrupted_foreign_gpu_owner"
                rc = int(proc.wait())
                passed = rc == 0 and validate_report(OUTPUT, expected_fingerprint=self.report_fp)
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
    return ResourceQueue().run()


if __name__ == "__main__":
    raise SystemExit(main())
