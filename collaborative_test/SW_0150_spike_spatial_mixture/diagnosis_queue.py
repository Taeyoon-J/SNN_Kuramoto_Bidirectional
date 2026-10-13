"""Owner-aware, single-run reservation for the registered SW0150 TRAIN diagnostic."""
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

try:
    import fcntl
except ImportError:  # Windows is for CPU tests/help only; execution requires Linux flock.
    fcntl = None

# The supervisor is CPU-only. Only its leased child can see the selected GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"),
               str(ROOT / "snn_kuramoto_bidirectional")]

from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4 as owner_queue
from collaborative_test.SW_0150_spike_spatial_mixture import diagnosis, evaluate, stage_queue

NAME = diagnosis.NAME
RUNTIME = HERE / "results_archive" / "diagnosis_queue_20261012"
STATE = RUNTIME / "state.json"
LOCK = RUNTIME / "queue.lock"
OUTPUT = diagnosis.OUTPUT_DEFAULT
LOG = RUNTIME / "diagnosis.attempt01.log"
MAIN_STATE = stage_queue.STATE
OWNER_HELPER = ROOT / "collaborative_test/SW_0126_history_event_binding/pilot_queue.py"
OWNER_HELPER_SHA256 = owner_queue.OWNER_GUARD_SHA256
GPU_LEASE_DIR = Path("/tmp/kevinswk_sw0113_gpu_leases")
MAX_MEMORY_MIB = 512
SAMPLE_SECONDS = 5.0


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_fingerprint():
    fingerprint = dict(diagnosis.implementation_fingerprint())
    paths = (HERE / "diagnosis_queue.py", HERE / "stage_queue.py", OWNER_HELPER,
             ROOT / "collaborative_test/SW_0148_checkpointable_boundary_q/stage_queue_v4.py")
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("an SW0150 diagnostic queue dependency is missing")
    if sha256_file(OWNER_HELPER) != OWNER_HELPER_SHA256:
        raise ValueError("the registered GPU owner guard changed")
    fingerprint.update({path.relative_to(ROOT).as_posix(): sha256_file(path)
                        for path in paths})
    return fingerprint


def implementation_sha256():
    payload = json.dumps(implementation_fingerprint(), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def task_plan():
    return [{"task_id": diagnosis.DIAGNOSTIC, "stage": "diagnosis", "seed": 1,
             "depends_on": ["main_queue_complete", "predict_encoder_graph",
                            "predict_joint", "predict_source_p"],
             "output": str(OUTPUT), "log": str(LOG)}]


def command(task=None, *, device="cuda:0"):
    task = task or task_plan()[0]
    if (task.get("stage") != "diagnosis" or task.get("seed") != 1
            or task.get("task_id") != diagnosis.DIAGNOSTIC):
        raise ValueError("only the registered source1 first64 diagnostic may run")
    return [sys.executable, "-m", "collaborative_test.SW_0150_spike_spatial_mixture.diagnosis",
            "--device", device, "--output", str(task["output"])]


def validate_report(path=OUTPUT, *, expected_fingerprint=None):
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        if expected_fingerprint is None:
            expected_fingerprint = diagnosis.implementation_fingerprint()
        protocol_sha = sha256_file(diagnosis.PROTOCOL_PATH)
        source = report["source"]
        ids = source["first64_image_ids"]
        actual_source = diagnosis.train._source_record()
        expected_ids = actual_source["ids"][:diagnosis.IMAGE_COUNT]
        expected_rows = [int(value) for value in
                         actual_source["rows"][:diagnosis.IMAGE_COUNT]]
        if (ids != expected_ids
                or source.get("first64_pool_rows") != expected_rows
                or source.get("seed") != diagnosis.train.SEED):
            return False
        if (report.get("status") != "complete"
                or report.get("experiment") != NAME
                or report.get("diagnostic") != diagnosis.DIAGNOSTIC
                or report.get("ground_truth_used") is not False
                or report.get("masks_read") is not False
                or report.get("optimizer_updates") != 0
                or report.get("protocol_sha256") != protocol_sha
                or report.get("implementation_fingerprint") != expected_fingerprint
                or report.get("implementation_sha256") != hashlib.sha256(json.dumps(
                    expected_fingerprint, sort_keys=True, separators=(",", ":"),
                    allow_nan=False).encode("utf-8")).hexdigest()
                or len(ids) != diagnosis.IMAGE_COUNT or len(set(ids)) != diagnosis.IMAGE_COUNT
                or set(report.get("arms", {})) != {"encoder_graph", "joint"}):
            return False
        training = report.get("training")
        if not isinstance(training, dict) or set(training) != {"encoder_graph", "joint"}:
            return False
        validated_training = {
            arm: diagnosis.train.validate_training(arm)
            for arm in ("encoder_graph", "joint")
        }
        source_lineage = None
        for arm, validated in validated_training.items():
            manifest = validated["manifest"]
            expected_training = {
                "manifest_sha256": validated["manifest_sha256"],
                "checkpoint_sha256": validated["checkpoint_sha256"],
                "optimizer_sha256": diagnosis.train.sha(validated["optimizer"]),
                "history_sha256": diagnosis.train.sha(validated["history"]),
                "implementation_sha256": manifest["implementation_sha256"],
            }
            if training[arm] != expected_training:
                return False
            row_source = {
                "core_sha256": manifest["source_core_sha256"],
                "manifest_sha256": manifest["source_manifest_sha256"],
                "training_ids_sha256": manifest["source_training_ids_sha256"],
                "rgb_asset_validation": manifest["rgb_asset_validation"],
                "preflight_sha256": manifest["preflight_sha256"],
                "warm_sha256": manifest["preflight_warm_sha256"],
            }
            if source_lineage is None:
                source_lineage = row_source
            elif source_lineage != row_source:
                return False
            if (source.get("core_sha256") != row_source["core_sha256"]
                    or source.get("manifest_sha256") != row_source["manifest_sha256"]
                    or source.get("training_ids_sha256") != row_source["training_ids_sha256"]
                    or source.get("rgb_asset_validation") != row_source["rgb_asset_validation"]
                    or source.get("preflight_sha256") != row_source["preflight_sha256"]
                    or source.get("warm_sha256") != row_source["warm_sha256"]):
                return False
        for arm, row in report["arms"].items():
            if (row.get("arm") != arm or row.get("rollouts") != diagnosis.IMAGE_COUNT
                    or row.get("live_tail_steps") != 64
                    or len(row.get("per_image", [])) != diagnosis.IMAGE_COUNT
                    or not _sha256_text(row.get("model_state_sha256_before"))
                    or row.get("model_state_sha256_before") != row.get("model_state_sha256_after")
                    or any(not _finite_number(row.get(key)) for key in (
                        "mean_changed_row_fraction", "mean_eligible_rows",
                        "mean_paired_delta_permuted_minus_original"))):
                return False
            if [int(item.get("image_id", -1)) for item in row["per_image"]] != ids:
                return False
            for index, item in enumerate(row["per_image"]):
                numeric = ("nll_original", "nll_color_bin_permuted",
                           "paired_delta_permuted_minus_original", "changed_row_fraction")
                if any(not _finite_number(item.get(key)) for key in numeric):
                    return False
                if (not 0.0 <= float(item["changed_row_fraction"]) <= 1.0
                        or isinstance(item.get("eligible_rows"), bool)
                        or not isinstance(item.get("eligible_rows"), int)
                        or not 0 <= item["eligible_rows"] <= 1024
                        or item.get("native_microbatch_slots") not in range(1, 12)
                        or item.get("padded_slots") not in range(1, 12)
                        or item["native_microbatch_slots"] > item["padded_slots"]):
                    return False
                if (isinstance(item.get("permuted_row_positions"), bool)
                        or not isinstance(item.get("permuted_row_positions"), int)
                        or not 0 <= item["permuted_row_positions"] <= 1024
                        or isinstance(item.get("changed_assignment_rows"), bool)
                        or not isinstance(item.get("changed_assignment_rows"), int)
                        or not 0 <= item["changed_assignment_rows"] <= 1024):
                    return False
                if float(item["paired_delta_permuted_minus_original"]) != (
                        float(item["nll_color_bin_permuted"])
                        - float(item["nll_original"])):
                    return False
                repeat_delta = item.get("nll_original_repeat_delta")
                if index == 0:
                    if not _finite_number(repeat_delta) or float(repeat_delta) != 0.0:
                        return False
                elif repeat_delta is not None:
                    return False
        return True
    except (OSError, ValueError, TypeError, KeyError, AttributeError,
            json.JSONDecodeError):
        return False


def _sha256_text(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value.lower()))


def _finite_number(value):
    try:
        return not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main_queue_state():
    """Return waiting, ready, or terminal-failed without reserving a GPU."""
    if not MAIN_STATE.is_file():
        return "waiting", None
    try:
        state = _read_json(MAIN_STATE)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return "failed", f"main queue state is unreadable: {exc!r}"
    status = state.get("status")
    if not isinstance(status, str):
        return "failed", "main SW0150 queue state has a malformed status"
    tasks = state.get("tasks")
    if (status == "failed" or status.endswith("_failed") or status.startswith("blocked_")
            or status.startswith("cancelled")
            or status == "score_blocked_missing_valid_predictions"):
        return "failed", f"main SW0150 queue terminal status: {status}"
    if status != "complete":
        return "waiting", None
    required = {"predict_encoder_graph", "predict_joint", "predict_source_p"}
    if not isinstance(tasks, dict) or not required.issubset(tasks):
        return "failed", "completed main queue lacks the three registered prediction tasks"
    for task_id in required:
        row = tasks[task_id]
        if (not isinstance(row, dict)
                or row.get("status") not in {"passed", "verified_existing"}):
            return "failed", f"main queue task {task_id} is not a passed prediction"
    try:
        for prediction in ("encoder_graph", "joint", "source_p"):
            evaluate.validate_prediction(prediction)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, AssertionError) as exc:
        return "failed", f"frozen prediction validation failed: {exc!r}"
    return "ready", None


def check_unwritten(paths):
    existing = [str(Path(path)) for path in paths if Path(path).exists()]
    if existing:
        raise FileExistsError(f"preserve existing SW0150 diagnostic artifacts: {existing}")


def _atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


class DiagnosisQueue:
    def __init__(self):
        if fcntl is None:
            raise RuntimeError("SW0150 diagnostic reservation requires Linux flock")
        RUNTIME.mkdir(parents=True, exist_ok=True)
        self.lock_stream = LOCK.open("a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock_stream.close()
            raise RuntimeError("another SW0150 diagnostic supervisor owns the queue") from exc
        try:
            self.stop_event = threading.Event()
            self.task = task_plan()[0]
            self.diagnostic_fingerprint = diagnosis.implementation_fingerprint()
            self.diagnostic_fingerprint_sha256 = hashlib.sha256(json.dumps(
                self.diagnostic_fingerprint, sort_keys=True, separators=(",", ":"),
                allow_nan=False).encode("utf-8")).hexdigest()
            self.queue_fingerprint = implementation_fingerprint()
            self.fingerprint_sha256 = hashlib.sha256(json.dumps(
                self.queue_fingerprint, sort_keys=True, separators=(",", ":"),
                allow_nan=False).encode("utf-8")).hexdigest()
            self.existing_complete = False
            if OUTPUT.exists():
                if not validate_report(OUTPUT,
                                       expected_fingerprint=self.diagnostic_fingerprint):
                    raise FileExistsError(
                        "preserve invalid or differently-bound existing diagnostic output")
                self.existing_complete = True
                return
            check_unwritten((STATE, LOG))
        except BaseException:
            self._unlock()
            raise
        self.state = {"experiment": NAME, "queue_version": 1,
                      "diagnostic": diagnosis.DIAGNOSTIC, "status": "waiting_for_main_queue",
                      "supervisor_pid": os.getpid(), "training_admission": False,
                      "diagnostic_fingerprint": self.diagnostic_fingerprint,
                      "diagnostic_fingerprint_sha256": self.diagnostic_fingerprint_sha256,
                      "queue_fingerprint": self.queue_fingerprint,
                      "implementation_sha256": self.fingerprint_sha256,
                      "main_queue_state": str(MAIN_STATE),
                      "tasks": {self.task["task_id"]: {"task": self.task,
                                                         "status": "waiting_for_main_queue"}}}
        _atomic_json(STATE, self.state)

    def _unlock(self):
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_UN)
        finally:
            self.lock_stream.close()

    def _update(self, status, **fields):
        row = self.state["tasks"][self.task["task_id"]]
        row.update(status=status, **fields)
        self.state.update(status=status, updated=time.time())
        _atomic_json(STATE, self.state)

    @staticmethod
    def _terminate_owned(proc):
        if proc is not None and proc.poll() is None:
            owner_queue._stop_owned(proc)

    def _verify_fingerprints(self):
        if (diagnosis.implementation_fingerprint() != self.diagnostic_fingerprint
                or implementation_fingerprint() != self.queue_fingerprint):
            raise RuntimeError("SW0150 diagnostic or reservation dependencies changed after registration")

    def _launch_once(self):
        attempt_log = Path(self.task["log"])
        proc = lease = None
        self._update("waiting_for_exclusive_gpu")
        self._verify_fingerprints()
        while not self.stop_event.is_set():
            gpu, gpu_uuid, memory, lease = owner_queue._acquire(self.stop_event)
            try:
                stable = owner_queue._stable_idle(
                    gpu, gpu_uuid, self.stop_event,
                    lambda elapsed, used: self._update(
                        "waiting_stable_gpu", gpu=gpu, gpu_uuid=gpu_uuid,
                        stable_seconds=elapsed, memory_used_mib=used))
                if not stable:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                    lease.close()
                    lease = None
                    if self.stop_event.is_set():
                        return "cancelled_before_launch"
                    self._update("waiting_for_exclusive_gpu",
                                 last_reservation_released=True)
                    continue
                self._verify_fingerprints()
                check_unwritten((OUTPUT, attempt_log))
                argv = command(self.task)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                with attempt_log.open("x", encoding="utf-8") as log_stream:
                    proc = subprocess.Popen(
                        argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                        stdout=log_stream, stderr=subprocess.STDOUT,
                        start_new_session=True, pass_fds=(lease.fileno(),))
                    self._update("running", child_pid=proc.pid, child_argv=argv,
                                 physical_gpu=gpu, gpu_uuid=gpu_uuid,
                                 memory_before_mib=memory, inherited_lease_fd=True,
                                 log=str(attempt_log), started=time.time())
                    while proc.poll() is None and not self.stop_event.is_set():
                        time.sleep(SAMPLE_SECONDS)
                        if proc.poll() is not None:
                            break
                        try:
                            current_uuid, _used, owners = owner_queue.gpu_guard._gpu_info(gpu)
                            foreign = (["gpu_uuid_changed"] if current_uuid != gpu_uuid else
                                       owner_queue.gpu_guard._foreign_owners(
                                           owners, owner_queue.gpu_guard._process_tree(proc.pid)))
                        except Exception as exc:
                            self._update("running", last_gpu_query_error=repr(exc))
                            continue
                        if foreign:
                            self._terminate_owned(proc)
                            return "interrupted_foreign_gpu_owner"
                    if self.stop_event.is_set() and proc.poll() is None:
                        self._terminate_owned(proc)
                        return "cancelled_owned_child_stopped"
                code = int(proc.wait())
                valid = code == 0 and validate_report(
                    OUTPUT, expected_fingerprint=self.diagnostic_fingerprint)
                return "passed" if valid else "failed_diagnostic_or_artifact_validation"
            except BaseException:
                self._terminate_owned(proc)
                raise
            finally:
                if lease is not None:
                    try:
                        fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                    finally:
                        lease.close()
        return "cancelled_before_launch"

    def run(self, *, poll_seconds=SAMPLE_SECONDS):
        try:
            if self.existing_complete:
                return 0
            while not self.stop_event.is_set():
                readiness, detail = main_queue_state()
                if readiness == "ready":
                    break
                if readiness == "failed":
                    self._update("blocked_main_queue_failure", reason=detail,
                                 finished=time.time())
                    return 1
                self._update("waiting_for_main_queue", checked=time.time())
                time.sleep(poll_seconds)
            else:
                self._update("cancelled", finished=time.time())
                return 1
            outcome = self._launch_once()
            self._update(outcome, returncode=(0 if outcome == "passed" else None),
                         output_sha256=sha256_file(OUTPUT) if OUTPUT.is_file() else None,
                         finished=time.time())
            return 0 if outcome == "passed" else 1
        except BaseException as exc:
            self._terminate_owned(locals().get("proc"))
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
                          "wait_for_main_queue_complete": True,
                          "predictions_revalidated_before_gpu": True,
                          "owner_aware_exclusive_gpu": True,
                          "maximum_parallel_gpu_tasks": 1,
                          "ground_truth_used": False,
                          "optimizer_updates": 0, "dry_run": True}, indent=2), flush=True)
        return 0
    return DiagnosisQueue().run()


if __name__ == "__main__":
    raise SystemExit(main())
