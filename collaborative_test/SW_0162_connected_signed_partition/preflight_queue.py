"""Owner-reserved single-GPU launcher for the SW0162 source-only preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
for path in (ROOT, ROOT / "collaborative_test"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import fcntl
except ImportError:
    fcntl = None

_VISIBILITY_MISSING = object()
_old_visibility = os.environ.get("CUDA_VISIBLE_DEVICES", _VISIBILITY_MISSING)
from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4 as owner
if _old_visibility is _VISIBILITY_MISSING:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = _old_visibility

from collaborative_test.SW_0162_connected_signed_partition import preflight

RUNTIME = HERE / "results_archive" / "preflight_queue_20261014"
STATE, LOCK, LOG = RUNTIME / "state.json", RUNTIME / "queue.lock", RUNTIME / "attempt01.log"
OUTPUT = preflight.OUTPUT
GPU_LEASE_DIR = Path("/tmp/kevinswk_sw0113_gpu_leases")
MAX_MEMORY_MIB = 512
STABLE_SECONDS = 5.0
SAMPLE_SECONDS = 2.0
POLL_SECONDS = 3.0
WATCHDOG_SECONDS = 1800
DEVICE = "cuda:0"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint():
    owner_guard = Path(owner.OWNER_GUARD)
    if not owner_guard.is_file() or sha256_file(owner_guard) != owner.OWNER_GUARD_SHA256:
        raise ValueError("registered shared GPU owner guard changed")
    return {str(path.relative_to(ROOT).as_posix()): sha256_file(path) for path in (
        HERE / "preflight_queue.py", HERE / "preflight.py", HERE / "connected_signed.py",
        HERE / "protocol.json", HERE / "DESIGN_KO.md", owner_guard,
    )}


def task_plan():
    return {"task_id": "source_T_connected_signed_eight_train_images", "device": DEVICE,
        "seed": preflight.SEED, "positions": list(preflight.POSITIONS),
        "steps": preflight.STEPS, "settle": preflight.SETTLE, "live_tail": 0,
        "optimizer_updates": 0, "ground_truth_used": False,
        "output": str(OUTPUT), "log": str(LOG),
        "depends_on": ["SW0130_source_contract_seed1", "SW0135_native32_source_seed1",
                       "SW0143_T_operator"]}


def command():
    return [sys.executable, "-m", "collaborative_test.SW_0162_connected_signed_partition.preflight",
            "--device", DEVICE, "--output", str(OUTPUT)]


def source_identity():
    prepared = preflight.prepare_preflight_cpu()
    return {"checkpoint_sha256": prepared["source_sha"],
        "manifest_sha256": sha256_file(prepared["manifest_path"]),
        "ordered_train_ids_sha256": hashlib.sha256(
            __import__("numpy").asarray(prepared["ordered_ids"], dtype="<i8").tobytes()).hexdigest(),
        "rgb_cache_size_bytes": prepared["cache_size_bytes"],
        "rgb_cache_mtime_ns": prepared["cache_mtime_ns"],
        "rgb_manifest_sha256": prepared["assets"]["train_manifest_sha256"],
        "source_positions": list(preflight.POSITIONS),
        "selected_pool_rows": prepared["selected_rows"],
        "selected_image_ids": prepared["selected_ids"],
        "rgb_batch_sha256": prepared["rgb_sha"],
        "encoder_source_sha256": prepared["assets"]["encoder_source_sha256"],
        "feature_stats_sha256": prepared["assets"]["feature_stats_sha256"],
        "feature_stats_mode": prepared["assets"]["feature_stats_mode"],
        "feature_clip": prepared["assets"]["feature_clip"],
        "rgb_image_sha256": [hashlib.sha256(item.tobytes()).hexdigest()
                              for item in prepared["rgb8"]]}


def validate_report(path=OUTPUT, *, expected_fingerprint=None, expected_source=None):
    path = Path(path)
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        source = record["source"]
        is_sha = lambda value: isinstance(value, str) and len(value) == 64 and all(
            char in "0123456789abcdef" for char in value)
        checks = (record["experiment"] == "SW0162_connected_signed_partition"
            and record["status"] == "source8_preflight_complete"
            and record["ground_truth_used"] is False and record["optimizer_updates"] == 0
            and record["source_operator"]["optimizer_updates"] == 0
            and record["source_operator"]["arm"] == "T"
            and record["source_operator"]["steps"] == 1024
            and record["source_operator"]["settle"] == 512
            and record["source_operator"]["live_tail"] == 0
            and record["source_operator"]["A_origin"] ==
                "untouched GraphVectorKuramoto.prepare_coupling tuple[0]"
            and record["source_operator"]["backend"] == "factorized"
            and record["seed"] == 1 and len(record["images"]) == 8
            and record["source"]["source_positions"] == list(preflight.POSITIONS)
            and source["source_state_sha256_before"] == source["source_state_sha256_after"]
            and is_sha(source["source_state_sha256_before"])
            and is_sha(source["checkpoint_sha256"]) and is_sha(source["manifest_sha256"])
            and is_sha(source["ordered_train_ids_sha256"])
            and is_sha(source["rgb_manifest_sha256"])
            and is_sha(source["encoder_source_sha256"])
            and is_sha(source["feature_stats_sha256"])
            and source["feature_stats_mode"] == "standardize"
            and source["rgb_asset_validation"]["scope"] ==
                "registered TRAIN RGB only; no validation images or masks opened"
            and all(item["trace_parity"] is True and item["Q_parity"] is True
                    and item["QCC_parity"] is True
                    and item["deterministic_solver_replay"] is True
                    and item["actual_A_symmetric_positive_finite"] is True
                    and set(item["trace_sha256"]) == set(preflight.TRACE_FIELDS)
                    and all(is_sha(value) for value in item["trace_sha256"].values())
                    and is_sha(item["Q_sha256"]) and is_sha(item["actual_A_sha256"])
                    and is_sha(item["qcc_labels_sha256"])
                    and isinstance(item["gain"], (int, float))
                    and math.isfinite(item["gain"]) and item["gain"] > 0
                    and is_sha(item["rgb_sha256"])
                    for item in record["images"]))
        if expected_fingerprint is not None:
            checks = checks and record["implementation"] == expected_fingerprint
        if expected_source is not None:
            for key, value in expected_source.items():
                if key != "rgb_image_sha256":
                    checks = checks and source.get(key) == value
            expected_rgbs = expected_source.get("rgb_image_sha256")
            if expected_rgbs is not None:
                images = record["images"]
                checks = checks and len(images) == len(expected_rgbs)
                for image, rgb_sha in zip(images, expected_rgbs):
                    checks = checks and image["rgb_sha256"] == rgb_sha
        return bool(checks)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


class PreflightQueue:
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if fcntl is None:
            raise RuntimeError("SW0162 GPU queue requires Linux flock")
        RUNTIME.mkdir(parents=True, exist_ok=True)
        self.lock_stream = LOCK.open("a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock_stream.close()
            raise RuntimeError("another SW0162 preflight supervisor owns the queue") from exc
        self.stop_event = threading.Event()
        self.queue_fp = implementation_fingerprint()
        self.report_fp = preflight.implementation_fingerprint()
        self.existing_complete = False
        try:
            if OUTPUT.exists():
                self.existing_complete = validate_report(OUTPUT,
                    expected_fingerprint=self.report_fp, expected_source=source_identity())
                if not self.existing_complete:
                    raise FileExistsError("preserve invalid or differently bound SW0162 report")
            else:
                for path in (STATE, LOG):
                    if path.exists():
                        raise FileExistsError(f"preserve existing SW0162 queue artifact: {path}")
                _atomic_json(STATE, {"experiment": "SW0162_connected_signed_partition",
                    "status": "waiting_for_idle_gpu", "supervisor_pid": os.getpid(),
                    "task": task_plan(), "queue_fingerprint": self.queue_fp,
                    "report_fingerprint": self.report_fp, "optimizer_updates": 0,
                    "ground_truth_used": False})
        except BaseException:
            self._unlock()
            raise

    def _unlock(self):
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_UN)
        finally:
            self.lock_stream.close()

    def _update(self, status, **extra):
        current = json.loads(STATE.read_text(encoding="utf-8"))
        current.update(status=status, updated=time.time(), **extra)
        _atomic_json(STATE, current)

    def run(self):
        if self.existing_complete:
            self._unlock()
            return 0
        proc = lease = None
        try:
            while not self.stop_event.is_set():
                source = source_identity()
                gpu, gpu_uuid, memory, lease = owner._acquire(self.stop_event)
                try:
                    stable = owner._stable_idle(gpu, gpu_uuid, self.stop_event,
                        lambda elapsed, used: self._update("waiting_stable_gpu", gpu=gpu,
                            gpu_uuid=gpu_uuid, stable_seconds=elapsed, memory_used_mib=used))
                    if not stable:
                        fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                        if self.stop_event.is_set():
                            return 1
                        self._update("waiting_for_exclusive_gpu", reservation_released=True)
                        continue
                    if implementation_fingerprint() != self.queue_fp or \
                            preflight.implementation_fingerprint() != self.report_fp:
                        raise RuntimeError("SW0162 code changed after queue registration")
                    if source_identity() != source:
                        raise RuntimeError("SW0162 source identity changed during reservation")
                    if OUTPUT.exists() or LOG.exists():
                        raise FileExistsError("preserve existing SW0162 report/log")
                    try:
                        actual_uuid, used, owners = owner.gpu_guard._gpu_info(gpu)
                    except Exception as exc:
                        self._update("waiting_gpu_probe_error", error=repr(exc))
                        fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                        continue
                    if actual_uuid != gpu_uuid or owners or int(used) > MAX_MEMORY_MIB:
                        self._update("waiting_for_exclusive_gpu", gpu=gpu,
                            gpu_uuid=gpu_uuid, final_gpu_check={"actual_uuid": actual_uuid,
                                "memory_mib": int(used), "owners": list(owners)})
                        fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                        continue
                    child_env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                    with LOG.open("x", encoding="utf-8") as log:
                        proc = subprocess.Popen(command(), cwd=ROOT, env=child_env,
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True, pass_fds=(lease.fileno(),))
                        self._update("running", child_pid=proc.pid, child_argv=command(),
                            gpu=gpu, gpu_uuid=gpu_uuid, inherited_lease_fd=True, log=str(LOG))
                        start = time.monotonic()
                        while proc.poll() is None and not self.stop_event.is_set():
                            if time.monotonic() - start > WATCHDOG_SECONDS:
                                owner._stop_owned(proc)
                                self._update("failed_watchdog_timeout", child_pid=proc.pid)
                                return 1
                            time.sleep(SAMPLE_SECONDS)
                    rc = int(proc.wait())
                    valid = rc == 0 and validate_report(OUTPUT,
                        expected_fingerprint=self.report_fp, expected_source=source)
                    self._update("complete" if valid else "failed_child_or_report_validation",
                        child_returncode=rc, report_valid=bool(valid),
                        report_sha256=sha256_file(OUTPUT) if OUTPUT.is_file() else None,
                        finished=time.time())
                    return 0 if valid else 1
                finally:
                    try:
                        if proc is not None and proc.poll() is None:
                            owner._stop_owned(proc)
                    finally:
                        if lease is not None:
                            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                            lease.close(); lease = None
            return 1
        except BaseException as exc:
            if proc is not None and proc.poll() is None:
                owner._stop_owned(proc)
            self._update("failed", error=repr(exc), finished=time.time())
            raise
        finally:
            self._unlock()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"status": "planned", "task": task_plan(), "argv": command(),
            "report_fingerprint": preflight.implementation_fingerprint(),
            "queue_fingerprint": implementation_fingerprint(),
            "parent_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")},
            allow_nan=False), flush=True)
        return 0
    return PreflightQueue().run()


if __name__ == "__main__":
    raise SystemExit(main())
