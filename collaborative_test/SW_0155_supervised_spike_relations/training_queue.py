"""Owner-aware two-arm SW0155 training reservation queue (TRAIN masks only)."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import threading
import time

try:
    import fcntl
except ImportError:  # CPU contract tests run on Windows; GPU execution is Linux-only.
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
from collaborative_test.SW_0148_checkpointable_boundary_q.checkpoint import (
    inspect_committed_metadata, is_only_uncommitted_temp,
)
from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0155_supervised_spike_relations import (
    artifactvalidator, preflight, preflight_migration, train,
)

EXPERIMENT = "SW0155_supervised_spike_relations"
ARMS = train.ARMS
ORIGINAL_PREFLIGHT_PATH = HERE / "results_archive" / "preflight_completed_20261013" / "preflight_seed1.json"
PREFLIGHT_PATH = HERE / "results_archive" / preflight_migration.MIGRATED_REPORT_NAME
RUNTIME = HERE / "results_archive" / "training_queue_20261013"
STATE = RUNTIME / "state.json"
QUEUE_LOCK = RUNTIME / "queue.lock"
GPU_LEASE_DIR = Path("/tmp/kevinswk_sw0113_gpu_leases")
MAX_MEMORY_MIB = owner.MAX_MEMORY_MIB
STABLE_SECONDS = owner.STABLE_SECONDS
SAMPLE_SECONDS = owner.SAMPLE_SECONDS
OWNER_GUARD = owner.OWNER_GUARD
OWNER_GUARD_SHA256 = owner.OWNER_GUARD_SHA256


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint():
    paths = (HERE / "training_queue.py", HERE / "protocol.json", HERE / "train.py",
        HERE / "preflight.py", HERE / "objective.py", HERE / "artifactvalidator.py",
        HERE / "preflight_migration.py",
        ROOT / "collaborative_test/SW_0148_checkpointable_boundary_q/stage_queue_v4.py",
        OWNER_GUARD)
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("SW0155 training queue dependency missing")
    if sha256_file(OWNER_GUARD) != OWNER_GUARD_SHA256:
        raise ValueError("registered shared GPU owner helper changed")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def task_plan():
    return [{"task_id": f"train_{arm}", "stage": "train", "arm": arm,
             "seed": 1, "device": "cuda:0",
             "output": str(train.TRAIN_ROOT / f"{arm}_seed1"),
             "checkpoint_dir": str(train.WORK_ROOT / f"{arm}_seed1" / "checkpoints")}
            for arm in ARMS]


def command(task):
    expected = {row["task_id"]: row for row in task_plan()}
    reference = expected.get(task.get("task_id"))
    if reference is None or task != reference:
        raise ValueError("only the registered SW0155 seed1 two-arm training tasks may launch")
    return [sys.executable, "-m", "collaborative_test.SW_0155_supervised_spike_relations.train",
            "--arm", task["arm"], "--device", "cuda:0",
            "--preflight", str(PREFLIGHT_PATH)]


def preflight_admission(path=PREFLIGHT_PATH):
    path = Path(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    original = json.loads(ORIGINAL_PREFLIGHT_PATH.read_text(encoding="utf-8-sig"))
    if not preflight_migration.validate_migrated_report(original, report):
        raise ValueError("SW0155 serialization-migrated preflight is not an exact validated receipt clone")
    source, assets = train.source_contract(), sw130.validate_rgb_assets()
    stat = train.HDF5_PATH.stat()
    if not preflight.validate_admission(report, source=source, assets=assets,
            hdf5_path=train.HDF5_PATH, hdf5_size=stat.st_size,
            hdf5_mtime_ns=stat.st_mtime_ns):
        raise ValueError("actual SW0155 TRAIN-mask preflight is invalid or stale")
    return {"path": str(path.resolve()), "sha256": sha256_file(path),
            "parent_report_sha256": sha256_file(ORIGINAL_PREFLIGHT_PATH),
            "serialization_migration": report["serialization_migration"],
            "source_core_sha256": source["source_sha256"],
            "source_manifest_sha256": sha256_file(source["manifest_path"]),
            "source_training_ids_sha256": train.ids_sha(source["image_ids"]),
            "rgb_asset_validation": assets,
            "canonical_hdf5": {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
            "preflight_fingerprint": report["implementation_fingerprint"]}


def valid_training(task):
    return artifactvalidator.validate_training_artifact(task)


def _required_resume_gpu(task):
    checkpoint_dir = Path(task["checkpoint_dir"])
    boundary = checkpoint_dir / "boundary.ptx"
    if not boundary.exists():
        if checkpoint_dir.exists() and not is_only_uncommitted_temp(checkpoint_dir):
            raise ValueError("training work directory lacks a valid committed boundary")
        return None
    try:
        metadata = inspect_committed_metadata(checkpoint_dir)
    except (OSError, ValueError, TypeError, KeyError, EOFError,
            pickle.UnpicklingError, RuntimeError) as exc:
        raise ValueError("SW0155 checkpoint failed integrity validation") from exc
    runtime = metadata.get("runtime")
    visibility = runtime.get("cuda_visible_devices") if isinstance(runtime, dict) else None
    completed = metadata.get("completed_updates")
    if (not isinstance(visibility, str) or visibility not in {"0", "1", "2", "3"}
            or runtime.get("device_type") != "cuda" or runtime.get("device_index") != 0
            or not isinstance(completed, int) or isinstance(completed, bool)
            or not 0 <= completed <= train.UPDATES):
        raise ValueError("SW0155 checkpoint lacks valid physical-GPU affinity metadata")
    return int(visibility)


def _safe_foreign_retry(task):
    if valid_training(task):
        return False
    output = Path(task["output"])
    if output.exists():
        return False
    checkpoint_dir = Path(task["checkpoint_dir"])
    if (checkpoint_dir / "boundary.ptx").is_file():
        try:
            inspect_committed_metadata(checkpoint_dir)
            return True
        except (OSError, ValueError, TypeError, KeyError, EOFError,
                pickle.UnpicklingError, RuntimeError):
            return False
    return not checkpoint_dir.exists() or is_only_uncommitted_temp(checkpoint_dir)


def _check_unwritten(paths):
    found = [str(Path(path)) for path in paths if Path(path).exists()]
    if found:
        raise FileExistsError(f"preserve existing SW0155 queue artifacts: {found}")


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


class TrainingQueue:
    def __init__(self):
        if fcntl is None:
            raise RuntimeError("SW0155 training queue requires Linux flock")
        RUNTIME.mkdir(parents=True, exist_ok=True)
        self.lock_stream = QUEUE_LOCK.open("a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock_stream.close()
            raise RuntimeError("another SW0155 training supervisor owns the queue") from exc
        self.guard = threading.RLock()
        self.stop_event = threading.Event()
        self.tasks = task_plan()
        try:
            self.queue_fp = implementation_fingerprint()
            self.preflight = preflight_admission()
            _check_unwritten((STATE,))
            self.state = {"experiment": EXPERIMENT, "queue_version": 1,
                "status": "queued", "supervisor_pid": os.getpid(),
                "queue_fingerprint": self.queue_fp, "preflight": self.preflight,
                "ground_truth_scope": "TRAIN masks only", "tasks": {
                    task["task_id"]: {"status": "queued", "task": task} for task in self.tasks}}
            self._save()
        except BaseException:
            try:
                fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_UN)
            finally:
                self.lock_stream.close()
            raise

    def _save(self):
        with self.guard:
            _atomic_json(STATE, self.state)

    def _update(self, task_id, status, **fields):
        with self.guard:
            self.state["status"] = status
            self.state["updated"] = time.time()
            row = self.state["tasks"][task_id]
            row.update(status=status, updated=time.time(), **fields)
            _atomic_json(STATE, self.state)

    def _verify_fingerprint(self):
        if implementation_fingerprint() != self.queue_fp:
            raise RuntimeError("SW0155 training queue files changed after registration")

    def _execute(self, task):
        task_id = task["task_id"]
        attempt = 1
        while not self.stop_event.is_set():
            self._verify_fingerprint()
            if not preflight_admission().get("sha256") == self.preflight["sha256"]:
                self._update(task_id, "failed_preflight_changed")
                return False
            if valid_training(task):
                self._update(task_id, "verified_existing")
                return True
            if Path(task["output"]).exists():
                self._update(task_id, "failed_existing_output_preserved")
                return False
            try:
                preferred = _required_resume_gpu(task)
            except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
                self._update(task_id, "failed_checkpoint_admission", error=repr(exc))
                return False
            gpu, gpu_uuid, memory, lease = owner._acquire(self.stop_event, preferred)
            process = None
            retry_foreign = False
            log = RUNTIME / f"{task_id}.attempt{attempt:02d}.log"
            try:
                stable = owner._stable_idle(gpu, gpu_uuid, self.stop_event,
                    lambda elapsed, used: self._update(task_id, "waiting_stable_gpu",
                        gpu=gpu, gpu_uuid=gpu_uuid, stable_seconds=elapsed,
                        memory_used_mib=used))
                if not stable:
                    if fcntl is not None:
                        fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                    lease.close()
                    if self.stop_event.is_set():
                        return False
                    self._update(task_id, "queued", reservation_released=True)
                    continue
                self._verify_fingerprint()
                if valid_training(task):
                    self._update(task_id, "verified_existing")
                    return True
                if Path(task["output"]).exists() or log.exists():
                    self._update(task_id, "failed_existing_output_preserved")
                    return False
                argv = command(task)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                           CUDA_DEVICE_ORDER="PCI_BUS_ID", OMP_NUM_THREADS="1",
                           MKL_NUM_THREADS="1")
                with log.open("x", encoding="utf-8") as stream:
                    process = subprocess.Popen(argv, cwd=ROOT, env=env,
                        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                        start_new_session=True, pass_fds=(lease.fileno(),))
                    self._update(task_id, "running", child_pid=process.pid, argv=argv,
                        physical_gpu=gpu, gpu_uuid=gpu_uuid, memory_before_mib=memory,
                        inherited_lease_fd=True, log=str(log), attempt=attempt)
                    while process.poll() is None and not self.stop_event.is_set():
                        time.sleep(SAMPLE_SECONDS)
                        if process.poll() is not None:
                            break
                        try:
                            current_uuid, _, owners = owner.gpu_guard._gpu_info(gpu)
                            foreign = (["gpu_uuid_changed"] if current_uuid != gpu_uuid else
                                owner.gpu_guard._foreign_owners(owners,
                                    owner.gpu_guard._process_tree(process.pid)))
                        except Exception as exc:
                            self._update(task_id, "running", last_gpu_query_error=repr(exc))
                            continue
                        if foreign:
                            self._update(task_id, "interrupted_foreign_gpu_owner",
                                         foreign_owner_pids=foreign)
                            owner._stop_owned(process)
                            retry_foreign = True
                            break
                rc = int(process.wait())
                valid = rc == 0 and valid_training(task)
                self._update(task_id, "passed" if valid else "failed_child_or_artifact_validation",
                             child_returncode=rc, artifact_valid=bool(valid), finished=time.time())
                if valid:
                    return True
                if not retry_foreign:
                    return False
            except BaseException as exc:
                if process is not None and process.poll() is None:
                    owner._stop_owned(process)
                self._update(task_id, "failed", error=repr(exc), finished=time.time())
                return False
            finally:
                if lease is not None:
                    if fcntl is not None:
                        fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                    lease.close()
            if retry_foreign:
                if not _safe_foreign_retry(task):
                    self._update(task_id, "failed_foreign_owner_preserved", finished=time.time())
                    return False
                self._update(task_id, "queued_after_foreign_owner", retry=attempt + 1)
                attempt += 1
        return False

    def run(self):
        try:
            self._verify_fingerprint()
            with self.guard:
                self.state.update(status="training_running", updated=time.time())
                _atomic_json(STATE, self.state)
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(self._execute, task) for task in self.tasks]
                results = [future.result() for future in futures]
            complete = all(results) and all(valid_training(task) for task in self.tasks)
            self.state["status"] = "training_complete" if complete else "training_failed"
            self.state["finished"] = time.time()
            self._save()
            return 0 if complete else 1
        finally:
            try:
                fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_UN)
            finally:
                self.lock_stream.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        pre = preflight_admission()
        print(json.dumps({"status": "preflight_admitted", "tasks": task_plan(),
            "preflight_sha256": pre["sha256"], "queue_fingerprint": implementation_fingerprint()},
            allow_nan=False), flush=True)
        return 0
    return TrainingQueue().run()


if __name__ == "__main__":
    raise SystemExit(main())
