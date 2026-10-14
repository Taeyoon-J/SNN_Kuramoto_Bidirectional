"""One-arm owner-aware, checkpoint-resumable SW0160 training reservation."""
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

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for item in (ROOT, ROOT / "collaborative_test"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

try:
    import fcntl
except ImportError:
    fcntl = None

_MISSING = object()
_saved_visibility = os.environ.get("CUDA_VISIBLE_DEVICES", _MISSING)
from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4 as owner
if _saved_visibility is _MISSING:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = _saved_visibility

from collaborative_test.SW_0160_end_to_end_source_encoder import preflight, train
from collaborative_test.SW_0160_end_to_end_source_encoder import preflight_queue
from collaborative_test.SW_0159_one_shot_event_regime import score_queue as upstream_score_queue

EXPERIMENT = train.EXPERIMENT
RUNTIME_ROOT = HERE / "results_archive" / "training_queues_20261014"
GPU_LEASE_DIR = Path("/tmp/kevinswk_sw0113_gpu_leases")
MAX_MEMORY_MIB = 512
STABLE_SECONDS = 5.0
SAMPLE_SECONDS = 2.0
POLL_SECONDS = 3.0
WATCHDOG_SECONDS = 21600.0
DEVICE = "cuda:0"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_arm(arm):
    if arm not in train.model.ARMS:
        raise ValueError("unregistered SW0160 arm")
    runtime = RUNTIME_ROOT / arm
    return {"arm": arm, "runtime": runtime, "state": runtime / "state.json",
        "lock": runtime / "queue.lock", "log": runtime / "attempt01.log",
        "output": train.TRAIN_ROOT / f"{arm}_seed1"}


def task_plan(arm):
    config = configure_arm(arm)
    return {"arm": arm, "seed": train.SEED, "updates": train.UPDATES,
        "logical_batch": train.BATCH, "microbatch": train.MICRO,
        "ordered_TRAIN_images": train.TRAIN_COUNT, "output": str(config["output"]),
        "checkpoint_resume": True, "watchdog_seconds": WATCHDOG_SECONDS}


def command(arm):
    return [sys.executable, "-m", "collaborative_test.SW_0160_end_to_end_source_encoder.train",
            "--arm", arm, "--device", DEVICE]


def _upstream_ready(arm, preflight_path=train.PREFLIGHT_PATH):
    record = train.validate_preflight_admission(preflight_path)
    if record.get("training_admission") is not True:
        raise ValueError("SW0160 scientific preflight rejected this fixed recipe")
    score_dependency = train._score_dependency()
    source = train.sw155.source_contract()
    assets = train.sw130.validate_rgb_assets()
    return {"arm": arm, "source_sha256": source["source_sha256"],
        "source_training_ids_sha256": train.sw155.ids_sha(source["image_ids"]),
        "preflight_sha256": sha256_file(preflight_path),
        "preflight_implementation_fingerprint": record["implementation_fingerprint"],
        "preflight_training_admission": True,
        "SW0159_score_dependency": score_dependency,
        "training_implementation_fingerprint": train.implementation_fingerprint(),
        "rgb_asset_validation": assets}


def _preflight_terminal_without_report(state_path=None):
    """Fail closed if the preflight queue is terminal but its report is absent."""
    path = Path(state_path) if state_path is not None else preflight_queue.STATE
    if not path.is_file():
        return False
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("experiment") != EXPERIMENT:
        raise ValueError("unexpected SW0160 preflight queue state")
    status = state.get("status")
    terminal = {
        "complete_admitted", "complete_rejected", "failed_child_or_report_validation",
        "failed_foreign_gpu_owner", "failed_exception", "cancelled_before_child",
        "cancelled", "blocked_scientific_preflight_rejection",
    }
    return status in terminal


def _preflight_queue_is_live(state_path=None):
    path = Path(state_path) if state_path is not None else preflight_queue.STATE
    if not path.is_file():
        return False
    state = json.loads(path.read_text(encoding="utf-8"))
    active = {"waiting_for_upstream", "waiting_stable_gpu", "waiting_for_exclusive_gpu", "running"}
    if state.get("experiment") != EXPERIMENT or state.get("status") not in active:
        return False
    return _registered_process_live(state.get("supervisor_pid"),
        "collaborative_test.SW_0160_end_to_end_source_encoder.preflight_queue")


def _registered_process_live(pid, module):
    if type(pid) is not int or pid <= 0:
        return False
    try:
        row = owner._proc_record(pid)
    except (OSError, RuntimeError):
        return False
    return bool(row and row.get("state") != "Z" and module in row.get("argv", []))


def _sw159_score_pipeline_live():
    """Wait only while a registered SW0159 owner is alive; never infer from timeout."""
    path = upstream_score_queue.STATE
    active = {"waiting_for_both_predictions", "waiting_for_prediction_owner", "running"}
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("experiment") != "SW0159_one_shot_event_regime":
            raise ValueError("unexpected SW0159 score dependency state")
        if state.get("status") in active and _registered_process_live(
                state.get("supervisor_pid"),
                "collaborative_test.SW_0159_one_shot_event_regime.score_queue"):
            return True
    # The score supervisor may not have been started yet. Keep waiting while
    # either registered arm's training/prediction owner is demonstrably alive.
    sw159 = upstream_score_queue.HERE
    checks = (
        (sw159 / "results_archive" / "training_queues_20261014", "training_queue", "train"),
        (sw159 / "results_archive" / "prediction_queues_20261014", "prediction_queue", "predict_evaluate"),
    )
    for root, queue_name, child_name in checks:
        for arm in ("control", "candidate"):
            state_path = root / arm / "state.json"
            if not state_path.is_file():
                continue
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("status") not in {"waiting_for_preflight", "waiting_for_training_artifact",
                    "waiting_stable_gpu", "waiting_for_exclusive_gpu", "running", "launching"}:
                continue
            module_prefix = "collaborative_test.SW_0159_one_shot_event_regime."
            if (_registered_process_live(state.get("supervisor_pid"), module_prefix + queue_name)
                    or _registered_process_live(state.get("child_pid"), module_prefix + child_name)):
                return True
    return False


def report_fingerprint():
    return train.implementation_fingerprint()


def implementation_fingerprint():
    owner_path = Path(owner.__file__)
    paths = (HERE / "training_queue.py", HERE / "train.py", HERE / "preflight.py",
        HERE / "preflight_queue.py", HERE / "model.py", HERE / "DESIGN_KO.md",
        ROOT / "collaborative_test" / "SW_0159_one_shot_event_regime" / "training_queue.py",
        ROOT / "collaborative_test" / "SW_0159_one_shot_event_regime" / "prediction_queue.py",
        ROOT / "collaborative_test" / "SW_0159_one_shot_event_regime" / "score_queue.py",
        owner_path, owner.OWNER_GUARD)
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("SW0160 training queue dependency missing")
    if sha256_file(owner.OWNER_GUARD) != owner.OWNER_GUARD_SHA256:
        raise ValueError("shared GPU owner guard checksum changed")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def validate_report(arm, path=None, *, expected_fingerprint=None):
    config = configure_arm(arm)
    output = Path(path) if path is not None else config["output"]
    return train.validate_training_artifact(arm, directory=output)


def _final_gpu_admission(gpu, expected_uuid):
    try:
        actual_uuid, memory, owners = owner.gpu_guard._gpu_info(gpu)
    except Exception as exc:
        return False, {"reason": "gpu_query_error", "detail": repr(exc)}
    state = {"gpu_uuid": actual_uuid, "memory_used_mib": int(memory), "owner_pids": list(owners)}
    if actual_uuid != expected_uuid:
        state["reason"] = "gpu_uuid_changed"
    elif owners:
        state["reason"] = "gpu_has_owners"
    elif int(memory) > MAX_MEMORY_MIB:
        state["reason"] = "gpu_memory_over_limit"
    else:
        state["reason"] = "idle"
        return True, state
    return False, state


def _atomic_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


class TrainingQueue:
    def __init__(self, arm):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if fcntl is None:
            raise RuntimeError("SW0160 training queue requires Linux flock")
        self.config = configure_arm(arm)
        self.arm = arm
        self.runtime, self.state_path = self.config["runtime"], self.config["state"]
        self.lock_path, self.log_path, self.output = (
            self.config["lock"], self.config["log"], self.config["output"])
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.lock_stream = self.lock_path.open("a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock_stream.close()
            raise RuntimeError(f"another SW0160 {arm} supervisor owns its queue") from exc
        self.state_lock = threading.RLock()
        self.stop_event = threading.Event()
        self.plan = task_plan(arm)
        self.report_fp = report_fingerprint()
        self.queue_fp = implementation_fingerprint()
        self.existing_complete = False
        try:
            if self.output.exists():
                if not validate_report(arm):
                    raise FileExistsError("preserve invalid/existing SW0160 training output")
                self.existing_complete = True
            else:
                if self.state_path.exists() or self.log_path.exists():
                    raise FileExistsError("preserve preexisting SW0160 queue state/log")
                _atomic_json(self.state_path, {"experiment": EXPERIMENT, "arm": arm,
                    "status": "waiting_for_preflight", "supervisor_pid": os.getpid(),
                    "task": self.plan, "report_fingerprint": self.report_fp,
                    "queue_fingerprint": self.queue_fp, "optimizer_updates": 0,
                    "ground_truth_scope": "registered TRAIN masks only"})
        except BaseException:
            self._unlock()
            raise

    def _unlock(self):
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_UN)
        finally:
            self.lock_stream.close()

    def _update(self, status, **fields):
        with self.state_lock:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            state.update(status=status, updated=time.time(), **fields)
            _atomic_json(self.state_path, state)

    def _verify(self):
        if report_fingerprint() != self.report_fp or implementation_fingerprint() != self.queue_fp:
            raise RuntimeError("SW0160 training dependencies changed after queue registration")

    def run(self):
        if self.existing_complete:
            self._unlock()
            return 0
        proc = lease = None
        try:
            while not self.stop_event.is_set():
                try:
                    upstream = _upstream_ready(self.arm)
                    break
                except FileNotFoundError as exc:
                    preflight_exists = Path(train.PREFLIGHT_PATH).is_file()
                    if not preflight_exists and _preflight_terminal_without_report():
                        self._update("blocked_preflight_terminal_without_report",
                                     error=repr(exc), finished=time.time())
                        return 1
                    if preflight_exists:
                        if not _sw159_score_pipeline_live():
                            self._update("blocked_upstream_score_missing", error=repr(exc), finished=time.time())
                            return 1
                        status = "waiting_for_sw0159_terminal_score"
                    else:
                        if not _preflight_queue_is_live():
                            self._update("blocked_preflight_owner_missing", error=repr(exc),
                                         finished=time.time())
                            return 1
                        status = "waiting_for_preflight"
                    self._update(status, last_validation=repr(exc))
                    self.stop_event.wait(POLL_SECONDS)
                except RuntimeError as exc:
                    # A missing SW0159 terminal score is pending only while a
                    # registered training/prediction/score owner is still live.
                    if not _sw159_score_pipeline_live():
                        self._update("blocked_upstream_score_missing", error=repr(exc), finished=time.time())
                        return 1
                    self._update("waiting_for_sw0159_terminal_score", last_validation=repr(exc))
                    self.stop_event.wait(POLL_SECONDS)
                except ValueError as exc:
                    record_path = train.PREFLIGHT_PATH
                    if record_path.is_file():
                        record = json.loads(record_path.read_text(encoding="utf-8"))
                        if record.get("status") == "complete" and record.get("training_admission") is False:
                            self._update("blocked_scientific_preflight_rejection", error=str(exc),
                                         preflight_sha256=sha256_file(record_path), finished=time.time())
                            return 2
                    raise
            if self.stop_event.is_set():
                self._update("cancelled_before_gpu_reservation")
                return 1
            self._verify()
            while not self.stop_event.is_set():
                gpu, gpu_uuid, memory, lease = owner._acquire(self.stop_event)
                try:
                    stable = owner._stable_idle(gpu, gpu_uuid, self.stop_event,
                        lambda elapsed, used: self._update("waiting_stable_gpu", gpu=gpu,
                            gpu_uuid=gpu_uuid, stable_seconds=elapsed, memory_used_mib=used))
                    if not stable:
                        fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                        if self.stop_event.is_set():
                            self._update("cancelled_before_child")
                            return 1
                        self._update("waiting_for_exclusive_gpu", reservation_released=True)
                        continue
                    self._verify()
                    current_upstream = _upstream_ready(self.arm)
                    if current_upstream != upstream:
                        raise RuntimeError("SW0160 source/preflight lineage changed while waiting for GPU")
                    if self.output.exists() or self.log_path.exists():
                        raise FileExistsError("preserve output/log appearing before SW0160 launch")
                    argv = command(self.arm)
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                               MKL_NUM_THREADS="1")
                    admitted, final_check = _final_gpu_admission(gpu, gpu_uuid)
                    if not admitted:
                        fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                        self._update("waiting_for_exclusive_gpu", reservation_released=True,
                                     final_gpu_check=final_check)
                        continue
                    start = time.monotonic()
                    with self.log_path.open("x", encoding="utf-8") as log:
                        proc = subprocess.Popen(argv, cwd=ROOT, env=env,
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True, pass_fds=(lease.fileno(),))
                        self._update("running", child_pid=proc.pid, child_argv=argv,
                            physical_gpu=gpu, gpu_uuid=gpu_uuid, memory_before_mib=memory,
                            inherited_lease_fd=True, log=str(self.log_path),
                            preflight_sha256=upstream["preflight_sha256"],
                            watchdog_seconds=WATCHDOG_SECONDS)
                        while proc.poll() is None and not self.stop_event.is_set():
                            if time.monotonic() - start > WATCHDOG_SECONDS:
                                owner._stop_owned(proc)
                                self._update("failed_watchdog", child_pid=proc.pid,
                                    work_checkpoint_preserved=True, finished=time.time())
                                return 1
                            self.stop_event.wait(SAMPLE_SECONDS)
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
                                owner._stop_owned(proc)
                                self._update("failed_foreign_gpu_owner", foreign_owner_pids=foreign,
                                    child_pid=proc.pid, work_checkpoint_preserved=True,
                                    output_preserved=self.output.exists(), finished=time.time())
                                return 1
                    rc = int(proc.wait())
                    valid = (rc == 0 and validate_report(self.arm))
                    self._update("complete" if valid else "failed_child_or_artifact_validation",
                        child_returncode=rc, artifact_valid=bool(valid), finished=time.time(),
                        manifest_sha256=sha256_file(self.output / "manifest.json")
                            if (self.output / "manifest.json").is_file() else None,
                        work_checkpoint_preserved=True)
                    return 0 if valid else 1
                finally:
                    try:
                        if proc is not None and proc.poll() is None:
                            owner._stop_owned(proc)
                    finally:
                        if lease is not None:
                            try:
                                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                            finally:
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
    parser.add_argument("--arm", choices=train.model.ARMS, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    plan = task_plan(args.arm)
    if args.dry_run:
        print(json.dumps({"status": "planned", "task": plan, "argv": command(args.arm),
            "cuda_visible_devices_parent": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
            "queue_fingerprint": implementation_fingerprint(),
            "training_fingerprint": report_fingerprint()}, allow_nan=False), flush=True)
        return 0
    return TrainingQueue(args.arm).run()


if __name__ == "__main__":
    raise SystemExit(main())
