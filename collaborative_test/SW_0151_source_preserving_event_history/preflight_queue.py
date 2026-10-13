"""Owner-aware, one-shot reservation for the SW0151 actual-source preflight."""
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

import numpy as np

try:
    import fcntl
except ImportError:  # Windows supports dry-run and CPU tests only.
    fcntl = None

# The supervisor must not initialize CUDA; only its leased child sees a GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"),
                str(ROOT / "snn_kuramoto_bidirectional")]

from collaborative_test.SW_0126_history_event_binding import pilot_queue as gpu_guard
from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4 as owner_queue
from collaborative_test.SW_0150_spike_spatial_mixture import diagnosis_queue as sw150_diagnosis_queue
from collaborative_test.SW_0151_source_preserving_event_history import preflight

NAME = "SW0151_source_preserving_event_history"
RUNTIME = HERE / "results_archive" / "preflight_queue_20261013"
STATE = RUNTIME / "state.json"
LOCK = RUNTIME / "queue.lock"
OUTPUT = preflight.OUTPUT_DEFAULT
LOG = RUNTIME / "preflight_seed1.attempt01.log"
GPU_LEASE_DIR = Path("/tmp/kevinswk_sw0113_gpu_leases")
MAX_MEMORY_MIB = 512
SAMPLE_SECONDS = 5.0
OWNER_HELPER = ROOT / "collaborative_test/SW_0126_history_event_binding/pilot_queue.py"
OWNER_HELPER_SHA256 = owner_queue.OWNER_GUARD_SHA256


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dict_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode("utf-8")).hexdigest()


def implementation_fingerprint():
    """Bind the immutable preflight, local wrapper, SW0150 gates, and owner helpers."""
    fingerprint = dict(preflight.implementation_fingerprint())
    paths = (
        HERE / "preflight_queue.py",
        ROOT / "collaborative_test/SW_0150_spike_spatial_mixture/stage_queue.py",
        ROOT / "collaborative_test/SW_0150_spike_spatial_mixture/diagnosis_queue.py",
        ROOT / "collaborative_test/SW_0148_checkpointable_boundary_q/stage_queue_v4.py",
        OWNER_HELPER,
    )
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("an SW0151 preflight queue dependency is missing")
    if sha256_file(OWNER_HELPER) != OWNER_HELPER_SHA256:
        raise ValueError("registered shared GPU owner helper changed")
    fingerprint.update({path.relative_to(ROOT).as_posix(): sha256_file(path)
                        for path in paths})
    return fingerprint


def task_plan():
    return [{"task_id": "actual_source_preflight_seed1", "stage": "preflight",
             "seed": 1, "device": "cuda:0", "depends_on": [
                 "sw150_main_complete", "sw150_diagnostic_complete"],
             "output": str(OUTPUT), "log": str(LOG)}]


def command(task=None, *, device="cuda:0"):
    task = task or task_plan()[0]
    if (task.get("stage") != "preflight" or task.get("seed") != 1
            or task.get("task_id") != "actual_source_preflight_seed1"
            or task.get("depends_on") != ["sw150_main_complete", "sw150_diagnostic_complete"]):
        raise ValueError("only the registered SW0151 source1 preflight is allowed")
    return [sys.executable, "-m",
            "collaborative_test.SW_0151_source_preserving_event_history.preflight",
            "--device", device, "--output", str(task["output"])]


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _terminal_failure(status):
    return (status == "failed" or status.endswith("_failed")
            or status.startswith("failed_") or status.startswith("blocked_")
            or status.startswith("cancelled") or status.startswith("interrupted"))


def main_queue_readiness():
    """Use SW0150's strict terminal validator; never reserve while it is pending."""
    try:
        status, detail = sw150_diagnosis_queue.main_queue_state()
    except Exception as exc:
        return "failed", f"SW0150 main queue validation raised: {exc!r}"
    if status not in {"waiting", "ready", "failed"}:
        return "failed", "SW0150 main queue returned an unknown readiness state"
    return status, detail


def diagnostic_queue_readiness():
    """Require the separately registered SW0150 diagnostic task to be complete."""
    path = sw150_diagnosis_queue.STATE
    if not Path(path).is_file():
        return "waiting", None
    try:
        state = _read_json(path)
    except (OSError, ValueError, TypeError) as exc:
        return "failed", f"SW0150 diagnostic queue state is unreadable: {exc!r}"
    status = state.get("status")
    if not isinstance(status, str):
        return "failed", "SW0150 diagnostic queue has a malformed status"
    if _terminal_failure(status):
        return "failed", f"SW0150 diagnostic queue terminal status: {status}"
    # The registered SW0150 supervisor's terminal success label is "passed";
    # its task row is the authoritative per-stage success marker.
    if status not in {"complete", "passed", "verified_existing"}:
        return "waiting", None
    task_id = sw150_diagnosis_queue.diagnosis.DIAGNOSTIC
    tasks = state.get("tasks")
    row = tasks.get(task_id) if isinstance(tasks, dict) else None
    if not isinstance(row, dict) or row.get("status") not in {"passed", "verified_existing"}:
        return "failed", "completed SW0150 diagnostic queue lacks a passed diagnostic task"
    try:
        if not sw150_diagnosis_queue.validate_report():
            return "failed", "completed SW0150 diagnostic artifact failed frozen validation"
    except Exception as exc:
        return "failed", f"SW0150 diagnostic artifact validation raised: {exc!r}"
    return "ready", None


def upstream_readiness():
    main, main_detail = main_queue_readiness()
    if main == "failed":
        return "failed", main_detail
    diag, diag_detail = diagnostic_queue_readiness()
    if diag == "failed":
        return "failed", diag_detail
    if main != "ready" or diag != "ready":
        return "waiting", {"main": main, "diagnostic": diag}
    return "ready", None


def validate_report(path=OUTPUT, *, expected_fingerprint=None):
    """Validate exact source, parity, fixed first64 scope, and no-update receipt."""
    try:
        report = _read_json(path)
        expected_fingerprint = (preflight.implementation_fingerprint()
                                if expected_fingerprint is None else expected_fingerprint)
        ids = report["source"]["first64_train_ids"]
        rows = report["source"]["first64_pool_rows"]
        if (report.get("experiment") != NAME
                or report.get("stage") != "actual_source_native32_parity_and_scale_preflight"
                or report.get("status") != "passed_kappa_zero_actual_source_parity"
                or report.get("seed") != 1
                or report.get("training_admission") is not False
                or report.get("ground_truth_used") is not False
                or report.get("masks_read") is not False
                or report.get("validation_or_test_ids_read") is not False
                or report.get("optimizer_updates") != 0
                or report.get("implementation_fingerprint") != expected_fingerprint
                or not isinstance(ids, list) or len(ids) != 64 or len(set(ids)) != 64
                or len(rows) != 64 or len(set(rows)) != 64
                or report.get("parity", {}).get("batches") != 16
                or report.get("parity", {}).get("exact_trace_batches") != 16
                or report.get("parity", {}).get("exact_production_q_batches") != 16
                or report.get("parity", {}).get("all_six_trace_fields_exact") is not True
                or report.get("parity", {}).get("production_actual_spike_q_exact") is not True
                or report.get("calibration", {}).get("source_rollouts_reused_for_calibration") is not True
                or report.get("model", {}).get("registered_module_state_unchanged") is not True
                or report.get("model", {}).get("registered_module_state_sha256_before")
                   != report.get("model", {}).get("registered_module_state_sha256_after")):
            return False
        calibration = report["calibration"].get("fixed_per_component_scale")
        if (not isinstance(calibration, list) or len(calibration) != 4
                or any(not _finite_positive(value) for value in calibration)):
            return False
        source = report["source"]
        required_sha = ("core_checkpoint_sha256", "source_manifest_sha256",
                        "ordered_train_ids_sha256", "saved_native32_parity_sha256")
        if any(not _sha256_text(source.get(key)) for key in required_sha):
            return False
        if source.get("rgb_asset_validation") != preflight.sw130.validate_rgb_assets():
            return False
        checkpoint, manifest_path, _manifest, expected_rows, expected_ids, source_sha = (
            preflight.sw130.source_contract(1))
        expected_ids, expected_rows = preflight.validate_registered_train_order(
            expected_ids, expected_rows)
        if (ids != expected_ids[:64] or rows != expected_rows[:64].tolist()
                or source.get("core_checkpoint_sha256") != source_sha
                or source.get("source_manifest_sha256") != preflight.sw135.sha256_file(manifest_path)
                or source.get("ordered_train_ids_sha256") != hashlib.sha256(
                    np.asarray(expected_ids, dtype="<i8").tobytes()).hexdigest()
                or source.get("saved_native32_parity_path") != str(preflight.PARITY_PATH)
                or source.get("saved_native32_parity_sha256") != sha256_file(preflight.PARITY_PATH)
                or source.get("foundation_implementation_fingerprint")
                   != preflight.sw135.implementation_fingerprint()):
            return False
        model = report["model"]
        if (model.get("native32_grid") != [32, 32] or model.get("patches") != 1024
                or model.get("steps") != 1024 or model.get("settle") != 512
                or model.get("live_tail_steps") != 0 or model.get("new_parameters") != 8
                or model.get("kappa_initial") != 0.0):
            return False
        before = report["model"].get("registered_module_state_sha256_before")
        after = report["model"].get("registered_module_state_sha256_after")
        if (not isinstance(before, dict) or set(before) != {"encoder", "patcher", "wrapped", "event_history"}
                or before != after or any(not _sha256_text(value) for value in before.values())):
            return False
        diagnostics = report["descriptive_events"]
        for key in ("settled_event_on_fraction_by_component",
                    "settled_event_exact_zero_fraction_by_component",
                    "settled_event_exact_one_fraction_by_component"):
            values = diagnostics.get(key)
            if not isinstance(values, list) or len(values) != 4 or any(
                    not _finite_number(value) or not 0 <= float(value) <= 1 for value in values):
                return False
        variances = diagnostics.get("settled_history_variance_by_component")
        if (not isinstance(variances, list) or len(variances) != 4 or any(
                not _finite_number(value) or float(value) < 0 for value in variances)):
            return False
        return True
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError,
            json.JSONDecodeError, RuntimeError):
        return False


def _sha256_text(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value.lower()))


def _finite_number(value):
    try:
        return not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _finite_positive(value):
    return _finite_number(value) and float(value) > 0


def _atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


class PreflightQueue:
    def __init__(self):
        if fcntl is None:
            raise RuntimeError("SW0151 queue requires Linux flock")
        RUNTIME.mkdir(parents=True, exist_ok=True)
        self.lock_stream = LOCK.open("a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock_stream.close()
            raise RuntimeError("another SW0151 preflight supervisor owns this queue") from exc
        try:
            self.stop_event = threading.Event()
            self.task = task_plan()[0]
            self.fingerprint = implementation_fingerprint()
            self.preflight_fingerprint = preflight.implementation_fingerprint()
            self.fingerprint_sha256 = _dict_sha(self.fingerprint)
            self.preflight_fingerprint_sha256 = _dict_sha(self.preflight_fingerprint)
            if OUTPUT.exists():
                if not validate_report(OUTPUT, expected_fingerprint=self.preflight_fingerprint):
                    raise FileExistsError("preserve existing invalid/differently-bound preflight output")
                self.existing_complete = True
                return
            self.existing_complete = False
            existing = [str(path) for path in (STATE, LOG) if path.exists()]
            if existing:
                raise FileExistsError(f"preserve prior SW0151 queue attempt; no restart: {existing}")
            self.state = {"experiment": NAME, "queue_version": 1,
                          "status": "waiting_for_sw150_completion",
                          "supervisor_pid": os.getpid(), "training_admission": False,
                          "queue_fingerprint": self.fingerprint,
                          "queue_fingerprint_sha256": self.fingerprint_sha256,
                          "preflight_fingerprint": self.preflight_fingerprint,
                          "preflight_fingerprint_sha256": self.preflight_fingerprint_sha256,
                          "task": self.task, "attempt_policy": "one_shot_no_automatic_retry"}
            _atomic_json(STATE, self.state)
        except BaseException:
            self._unlock()
            raise

    def _unlock(self):
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_UN)
        finally:
            self.lock_stream.close()

    def _update(self, status, **fields):
        self.state.update(status=status, updated=time.time(), **fields)
        _atomic_json(STATE, self.state)

    def _verify_fingerprint(self):
        if (implementation_fingerprint() != self.fingerprint
                or preflight.implementation_fingerprint() != self.preflight_fingerprint):
            raise RuntimeError("registered SW0151 preflight or queue dependencies changed")

    def _launch_once(self):
        self._verify_fingerprint()
        while not self.stop_event.is_set():
            gpu, gpu_uuid, memory, lease = owner_queue._acquire(self.stop_event)
            proc = None
            retry_reservation = False
            try:
                stable = owner_queue._stable_idle(
                    gpu, gpu_uuid, self.stop_event,
                    lambda elapsed, used: self._update(
                        "waiting_for_stable_gpu", gpu=gpu, gpu_uuid=gpu_uuid,
                        stable_seconds=elapsed, memory_used_mib=used))
                if not stable:
                    if self.stop_event.is_set():
                        return "cancelled_before_launch"
                    self._update("waiting_for_exclusive_gpu", released_unstable_reservation=True)
                    retry_reservation = True
                    continue
                self._verify_fingerprint()
                if OUTPUT.exists() or LOG.exists():
                    self._update("failed_existing_artifact_preserved", finished=time.time())
                    return "failed_existing_artifact_preserved"
                argv = command(self.task)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                with LOG.open("x", encoding="utf-8") as log_stream:
                    proc = subprocess.Popen(
                        argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                        stdout=log_stream, stderr=subprocess.STDOUT,
                        start_new_session=True, pass_fds=(lease.fileno(),))
                    self._update("running", child_pid=proc.pid, child_argv=argv,
                                 physical_gpu=gpu, gpu_uuid=gpu_uuid,
                                 memory_before_mib=memory, inherited_lease_fd=True,
                                 log=str(LOG), started=time.time())
                    while proc.poll() is None and not self.stop_event.is_set():
                        time.sleep(SAMPLE_SECONDS)
                        if proc.poll() is not None:
                            break
                        try:
                            current_uuid, _used, owners = gpu_guard._gpu_info(gpu)
                            foreign = ("gpu_uuid_changed" if current_uuid != gpu_uuid else
                                       gpu_guard._foreign_owners(
                                           owners, gpu_guard._process_tree(proc.pid)))
                        except Exception as exc:
                            self._update("running", last_gpu_query_error=repr(exc))
                            continue
                        if foreign:
                            self._update("interrupted_foreign_gpu_owner",
                                         foreign_owner_pids=foreign,
                                         attempt_log_preserved=True)
                            owner_queue._stop_owned(proc)
                            return "interrupted_foreign_gpu_owner"
                    if self.stop_event.is_set() and proc.poll() is None:
                        owner_queue._stop_owned(proc)
                        return "cancelled_owned_child_stopped"
                code = int(proc.wait())
                valid = code == 0 and validate_report(
                    OUTPUT, expected_fingerprint=self.preflight_fingerprint)
                return "passed" if valid else "failed_preflight_or_artifact_validation"
            except BaseException as exc:
                if proc is not None and proc.poll() is None:
                    owner_queue._stop_owned(proc)
                self._update("failed", error=repr(exc), finished=time.time())
                raise
            finally:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                lease.close()
            if retry_reservation:
                continue
        return "cancelled_before_launch"

    def run(self, *, poll_seconds=SAMPLE_SECONDS):
        try:
            if self.existing_complete:
                return 0
            self._update("waiting_for_sw150_completion")
            while not self.stop_event.is_set():
                readiness, detail = upstream_readiness()
                if readiness == "ready":
                    self._update("sw150_complete_preflight_pending")
                    break
                if readiness == "failed":
                    self._update("blocked_sw150_terminal_failure", reason=detail,
                                 finished=time.time())
                    return 1
                self._update("waiting_for_sw150_completion", readiness=detail,
                             checked=time.time())
                time.sleep(poll_seconds)
            else:
                self._update("cancelled", finished=time.time())
                return 1
            outcome = self._launch_once()
            self._update(outcome,
                         output_sha256=sha256_file(OUTPUT) if OUTPUT.is_file() else None,
                         finished=time.time())
            return 0 if outcome == "passed" else 1
        except BaseException as exc:
            self._update("failed", error=repr(exc), finished=time.time())
            return 1
        finally:
            self._unlock()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"experiment": NAME, "tasks": task_plan(),
                          "wait_for": ["SW0150 main queue complete",
                                       "SW0150 TRAIN diagnostic complete"],
                          "exclusive_gpu_lease": True, "maximum_parallel_gpu_tasks": 1,
                          "ground_truth_used": False, "optimizer_updates": 0,
                          "attempt_policy": "one_shot_no_automatic_retry", "dry_run": True},
                         indent=2), flush=True)
        return 0
    return PreflightQueue().run()


if __name__ == "__main__":
    raise SystemExit(main())
