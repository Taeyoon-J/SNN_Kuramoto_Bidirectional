"""Owner-aware, seed0/2-only repair queue for the frozen SW0154 predictor."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
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
except ImportError:  # pragma: no cover - Linux is required to reserve GPUs.
    fcntl = None

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for item in (ROOT, ROOT / "collaborative_test", ROOT / "snn_kuramoto_bidirectional"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4 as owner
from collaborative_test.SW_0150_spike_spatial_mixture import preflight as legacy_preflight
from collaborative_test.SW_0154_native64_resolution_compare import (
    evaluate_native64 as evaluate, predict_native64 as predictor,
    prediction_queue as original_queue,
    resource_probe, resource_queue,
    predict_native64_recovery as recovery,
)

SEEDS = (0, 2)
EXPERIMENT = "SW0154_native64_resolution_compare"
RUNTIME = HERE / "results_archive" / "seed02_recovery_queue_20261013"
STATE = RUNTIME / "state.json"
LOCK = RUNTIME / "queue.lock"
OWNER_GUARD = owner.OWNER_GUARD
OWNER_GUARD_SHA256 = owner.OWNER_GUARD_SHA256
_VALIDATION_LOCK = threading.RLock()
FROZEN_VALIDATE_PREDICTION = evaluate.validate_native_prediction


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint():
    paths = (Path(__file__), HERE / "predict_native64_recovery.py",
        HERE / "predict_native64.py", HERE / "evaluate_native64.py",
        HERE / "resource_probe.py", HERE / "resource_queue.py",
        HERE / "native64_model.py", HERE / "resolution64.py", HERE / "score_slot64.py",
        ROOT / "collaborative_test/SW_0130_phase_state_integration/run.py",
        ROOT / "collaborative_test/SW_0135_native32_spike_binding/evaluation_contract32.py",
        ROOT / "collaborative_test/SW_0150_spike_spatial_mixture/preflight.py",
        ROOT / "collaborative_test/SW_0134_native_spike_binding/rollout.py",
        ROOT / "snn_kuramoto_bidirectional/evaluation.py",
        ROOT / "snn_kuramoto_bidirectional/spike_classifier.py", OWNER_GUARD)
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("SW0154 recovery dependency missing")
    if sha256_file(OWNER_GUARD) != OWNER_GUARD_SHA256:
        raise ValueError("shared GPU owner helper changed")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def task_plan():
    return [{"task_id": f"recover_seed{seed}", "seed": seed,
        "stage": "native64_fixed320_source_prediction_recovery",
        "output": str(predictor.PREDICTION_ROOT / f"seed{seed}"),
        "work": str(predictor.WORK_ROOT / f"seed{seed}")} for seed in SEEDS]


def command(task, *, physical_gpu):
    plan = {row["task_id"]: row for row in task_plan()}
    if task != plan.get(task.get("task_id")) or task.get("seed") not in SEEDS or physical_gpu not in range(4):
        raise ValueError("recovery is restricted to missing fixed320 seeds0/2 and physical GPUs0..3")
    return [sys.executable, "-m", "collaborative_test.SW_0154_native64_resolution_compare.predict_native64_recovery",
        "--seed", str(task["seed"]), "--device", "cuda:0"]


def validate_prediction(seed, *, prediction_root=predictor.PREDICTION_ROOT,
                       expected_fingerprint=None):
    if seed not in SEEDS:
        raise ValueError("recovery validator accepts only seed0 or seed2")
    folder = Path(prediction_root) / f"seed{seed}"
    protocol_path = folder / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    repair = protocol.get("seed_specific_parity_recovery")
    expected_repair = {
        "status": "seed_specific_legacy_validator_override",
        "recovery_entrypoint_sha256": recovery.implementation_sha256(),
        "frozen_predictor_sha256": recovery.EXPECTED_PREDICTOR_SHA256,
        "legacy_validator_module": "SW0150_spike_spatial_mixture.preflight",
        "legacy_default_seed": 1, "scoped_seed": seed,
        "source_checkpoint_sha256": recovery.EXPECTED_SOURCE_SHA256[seed],
    }
    if not isinstance(repair, dict) or any(repair.get(key) != value for key, value in expected_repair.items()):
        raise ValueError("seed-specific repair provenance is missing or mismatched")
    # Frozen evaluator performs every original asset, ID, prediction, ledger,
    # and fingerprint check. Only its legacy parity seed global is scoped here.
    with _VALIDATION_LOCK:
        old_seed = legacy_preflight.SEED
        try:
            legacy_preflight.SEED = seed
            return FROZEN_VALIDATE_PREDICTION(seed, prediction_root=prediction_root,
                expected_fingerprint=expected_fingerprint)
        finally:
            legacy_preflight.SEED = old_seed


def valid_prediction(seed):
    try:
        validate_prediction(seed)
        return True
    except (OSError, ValueError, TypeError, KeyError, AssertionError, RuntimeError,
            json.JSONDecodeError):
        return False


def wait_original_queue_clear(*, state_path=original_queue.STATE, process_reader=owner._proc_record,
                              sleep=time.sleep, on_wait=lambda _state: None):
    """Wait without a GPU lease until the original all-seed queue has exited."""
    terminal = {"predictions_failed", "complete"}
    while True:
        try:
            state = json.loads(Path(state_path).read_text(encoding="utf-8-sig"))
        except FileNotFoundError:
            on_wait({"reason": "original_queue_state_not_yet_present"})
            sleep(owner.SAMPLE_SECONDS)
            continue
        if not isinstance(state, dict) or state.get("experiment") != EXPERIMENT:
            raise ValueError("original SW0154 queue state is malformed or belongs to another experiment")
        if state.get("status") not in terminal:
            on_wait({"reason": "original_queue_not_terminal", "status": state.get("status")})
            sleep(owner.SAMPLE_SECONDS)
            continue
        tasks = state.get("tasks")
        if not isinstance(tasks, dict):
            raise ValueError("original SW0154 queue state lacks task records")
        live = []
        for row in tasks.values():
            child_pid = row.get("child_pid") if isinstance(row, dict) else None
            if isinstance(child_pid, int) and not isinstance(child_pid, bool):
                record = process_reader(child_pid)
                if record is not None and record.get("state") != "Z":
                    argv = record.get("argv", [])
                    if not any("predict_native64" in str(arg) for arg in argv):
                        raise RuntimeError(f"recorded old prediction PID {child_pid} identity changed")
                    live.append(child_pid)
        supervisor_pid = state.get("supervisor_pid")
        if isinstance(supervisor_pid, int) and not isinstance(supervisor_pid, bool):
            record = process_reader(supervisor_pid)
            if record is not None and record.get("state") != "Z":
                argv = record.get("argv", [])
                expected_tokens = {
                    "collaborative_test.SW_0154_native64_resolution_compare.prediction_queue",
                    str((HERE / "prediction_queue.py").resolve()),
                }
                if not any(str(arg) in expected_tokens for arg in argv):
                    raise RuntimeError("original SW0154 queue supervisor PID identity changed")
                live.append(supervisor_pid)
        if live:
            on_wait({"reason": "original_queue_processes_still_live", "pids": sorted(set(live))})
            sleep(owner.SAMPLE_SECONDS)
            continue
        return state


def valid_evaluation():
    return original_queue.valid_evaluation()


def score_after_predictions(*, dataset=None):
    """Use the frozen evaluator while scoping only the legacy seed validator."""
    for seed in SEEDS:
        validate_prediction(seed)
    output = original_queue.EVALUATION_OUTPUT
    if output.exists():
        if not valid_evaluation():
            raise FileExistsError("preserve existing invalid SW0154 evaluation output")
        return json.loads(output.read_text(encoding="utf-8"))
    original_validate = evaluate.validate_native_prediction

    def validate_scoped(seed, **kwargs):
        if seed in SEEDS:
            return validate_prediction(seed, **kwargs)
        return FROZEN_VALIDATE_PREDICTION(seed, **kwargs)

    evaluate.validate_native_prediction = validate_scoped
    try:
        report = evaluate.score_all() if dataset is None else evaluate.score_all(dataset=dataset)
    finally:
        evaluate.validate_native_prediction = original_validate
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + f".{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    if output.exists():
        raise FileExistsError("preserve concurrently-created SW0154 evaluation output")
    os.replace(temp, output)
    if not valid_evaluation():
        raise ValueError("frozen SW0154 score failed terminal validation")
    return report


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


class SeedRecoveryQueue:
    def __init__(self, max_parallel_gpu_tasks=2):
        if not isinstance(max_parallel_gpu_tasks, int) or isinstance(max_parallel_gpu_tasks, bool) or not 1 <= max_parallel_gpu_tasks <= 2:
            raise ValueError("seed recovery queue concurrency must be 1 or 2")
        if fcntl is None:
            raise RuntimeError("SW0154 recovery queue requires Linux flock")
        self.max_parallel_gpu_tasks = max_parallel_gpu_tasks
        RUNTIME.mkdir(parents=True, exist_ok=True)
        self.lock_stream = LOCK.open("a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock_stream.close()
            raise RuntimeError("another SW0154 seed recovery supervisor owns the queue") from exc
        self.guard = threading.RLock()
        self.stop_event = threading.Event()
        try:
            self.queue_fp = implementation_fingerprint()
            self.prediction_fp = predictor.implementation_fingerprint()
            self.resource_fp = resource_probe.implementation_fingerprint()
            report_path = resource_probe.HERE / "results_archive" / resource_probe.REPORT_NAME
            if not resource_queue.validate_report(report_path, expected_fingerprint=self.resource_fp):
                raise ValueError("actual-source native64 resource report must pass before recovery")
            self.tasks = task_plan()
            existing = {row["task_id"]: valid_prediction(row["seed"]) for row in self.tasks}
            partial = [row["output"] for row in self.tasks
                if Path(row["output"]).exists() and not existing[row["task_id"]]]
            if partial:
                raise FileExistsError(f"preserve invalid/partial recovery outputs: {partial}")
            if STATE.exists():
                raise FileExistsError(f"preserve existing recovery state: {STATE}")
            self.state = {"experiment": EXPERIMENT, "queue_version": 1,
                "status": "queued", "supervisor_pid": os.getpid(),
                "recovery_seed_scope": list(SEEDS), "queue_fingerprint": self.queue_fp,
                "frozen_prediction_fingerprint": self.prediction_fp,
                "resource_report_path": str(report_path.resolve()),
                "resource_report_sha256": sha256_file(report_path),
                "tasks": {row["task_id"]: {"task": row,
                    "status": "verified_existing" if existing[row["task_id"]] else "queued"}
                    for row in self.tasks}}
            _atomic_json(STATE, self.state)
        except BaseException:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_UN)
            self.lock_stream.close()
            raise

    def _update(self, task_id, status, **fields):
        with self.guard:
            self.state.update(status="recovery_running", updated=time.time())
            self.state["tasks"][task_id].update(status=status, updated=time.time(), **fields)
            _atomic_json(STATE, self.state)

    def _verify(self):
        if (implementation_fingerprint() != self.queue_fp
                or predictor.implementation_fingerprint() != self.prediction_fp
                or resource_probe.implementation_fingerprint() != self.resource_fp):
            raise RuntimeError("SW0154 recovery implementation changed after registration")
        report_path = resource_probe.HERE / "results_archive" / resource_probe.REPORT_NAME
        if (sha256_file(report_path) != self.state["resource_report_sha256"]
                or not resource_queue.validate_report(report_path, expected_fingerprint=self.resource_fp)):
            raise RuntimeError("SW0154 native64 resource admission report changed")

    def _execute_seed(self, task):
        task_id, seed = task["task_id"], task["seed"]
        if valid_prediction(seed):
            self._update(task_id, "verified_existing")
            return True
        out = Path(task["output"])
        if out.exists():
            self._update(task_id, "failed_existing_output_preserved")
            return False
        attempt, preferred_gpu, preferred_uuid = 1, None, None
        while not self.stop_event.is_set():
            self._verify()
            gpu, gpu_uuid, memory, lease = owner._acquire(self.stop_event, preferred_gpu)
            if preferred_uuid is not None and gpu_uuid != preferred_uuid:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close()
                self._update(task_id, "failed_resume_gpu_identity_changed",
                    expected_uuid=preferred_uuid, observed_uuid=gpu_uuid)
                return False
            child = None
            log = RUNTIME / f"{task_id}.attempt{attempt:02d}.log"
            retry = False
            try:
                if not owner._stable_idle(gpu, gpu_uuid, self.stop_event,
                    lambda elapsed, used: self._update(task_id, "waiting_stable_gpu",
                        gpu=gpu, gpu_uuid=gpu_uuid, stable_seconds=elapsed, memory_used_mib=used)):
                    if self.stop_event.is_set():
                        return False
                    self._update(task_id, "queued", reservation_released=True)
                    continue
                self._verify()
                if valid_prediction(seed):
                    self._update(task_id, "verified_existing")
                    return True
                if out.exists() or log.exists():
                    self._update(task_id, "failed_existing_output_preserved")
                    return False
                argv = command(task, physical_gpu=gpu)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
                    OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                with log.open("x", encoding="utf-8") as stream:
                    child = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                        pass_fds=(lease.fileno(),))
                    self._update(task_id, "running", child_pid=child.pid, argv=argv,
                        physical_gpu=gpu, gpu_uuid=gpu_uuid, memory_before_mib=memory,
                        inherited_lease_fd=True, log=str(log), attempt=attempt)
                    while child.poll() is None and not self.stop_event.is_set():
                        time.sleep(owner.SAMPLE_SECONDS)
                        if child.poll() is not None:
                            break
                        try:
                            current_uuid, _, owners = owner.gpu_guard._gpu_info(gpu)
                            foreign = (["gpu_uuid_changed"] if current_uuid != gpu_uuid else
                                owner.gpu_guard._foreign_owners(owners, owner.gpu_guard._process_tree(child.pid)))
                        except Exception as exc:
                            self._update(task_id, "running", last_gpu_query_error=repr(exc))
                            continue
                        if foreign:
                            self._update(task_id, "interrupted_foreign_gpu_owner", foreign_owner_pids=foreign)
                            owner._stop_owned(child); child.wait(); retry = True
                            preferred_gpu, preferred_uuid = gpu, gpu_uuid
                            break
                rc = int(child.wait())
                if retry:
                    self._update(task_id, "queued_after_foreign_owner", child_returncode=rc,
                        partial_work_preserved=Path(task["work"]).exists(), retry=attempt + 1)
                else:
                    valid = rc == 0 and valid_prediction(seed)
                    self._update(task_id, "passed" if valid else "failed_child_or_artifact_validation",
                        child_returncode=rc, artifact_valid=bool(valid), finished=time.time())
                    return bool(valid)
            except BaseException as exc:
                if child is not None and child.poll() is None:
                    owner._stop_owned(child)
                self._update(task_id, "failed", error=repr(exc), finished=time.time())
                return False
            finally:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close()
            if retry:
                if not valid_prediction(seed) and out.exists():
                    self._update(task_id, "failed_existing_output_preserved")
                    return False
                attempt += 1
        return False

    def run(self):
        try:
            prior = wait_original_queue_clear(on_wait=lambda detail: self.state.update(
                status="waiting_original_queue_terminal", wait_detail=detail))
            self.state["original_queue_terminal_status"] = prior["status"]
            _atomic_json(STATE, self.state)
            self._verify()
            with ThreadPoolExecutor(max_workers=self.max_parallel_gpu_tasks) as pool:
                futures = [pool.submit(self._execute_seed, task) for task in self.tasks]
                passed = [future.result() for future in futures]
            if all(passed):
                # The original seed1 run must also be complete. Scoring is GT-last
                # and is invoked only after all three native predictions validate.
                FROZEN_VALIDATE_PREDICTION(1)
                report = score_after_predictions()
                self.state.update(status="complete", evaluation_sha256=sha256_file(
                    original_queue.EVALUATION_OUTPUT), evaluation_status=report["status"], finished=time.time())
                _atomic_json(STATE, self.state)
                return 0
            self.state.update(status="predictions_failed", finished=time.time())
            _atomic_json(STATE, self.state)
            return 1
        finally:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_UN)
            self.lock_stream.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-parallel-gpu-tasks", type=int, choices=(1, 2), default=2)
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"status": "planned", "seed_scope": list(SEEDS),
            "tasks": task_plan(), "queue_fingerprint": implementation_fingerprint()},
            allow_nan=False), flush=True)
        return 0
    return SeedRecoveryQueue(args.max_parallel_gpu_tasks).run()


if __name__ == "__main__":
    raise SystemExit(main())
