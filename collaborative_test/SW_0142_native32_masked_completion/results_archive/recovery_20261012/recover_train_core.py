"""One-shot recovery for the SW0142 interrupted core_trainable arm.

This reuses the frozen SW0142 stage queue's task execution and validators. It
never edits the experiment runner, retries a failed recovery, or kills foreign
GPU owners. Execute only after reviewing the original terminal queue record.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

# Keep the supervising Python process CPU-hidden. stage_queue.environment maps
# each GPU child to its leased physical device explicitly.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

EXPERIMENT = "SW0142_native32_masked_completion"
RECOVERY_NAME = "SW0142_core_trainable_recovery_20261012"


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_modules(repo: Path):
    repo = Path(repo).resolve(strict=True)
    if not (repo / "collaborative_test" / "SW_0142_native32_masked_completion" / "run.py").is_file():
        raise ValueError(f"not the expected SW0142 repository: {repo}")
    for item in (repo, repo / "collaborative_test", repo / "snn_kuramoto_bidirectional"):
        if str(item) not in sys.path:
            sys.path.insert(0, str(item))
    from collaborative_test.SW_0142_native32_masked_completion import run
    from collaborative_test.SW_0142_native32_masked_completion import stage_queue
    return run, stage_queue


def _pid_live(pid, proc_root: Path) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    return (Path(proc_root) / str(pid)).exists()


def _sw0142_processes(proc_root: Path, run_path: Path, queue_path: Path):
    """Return all live processes whose argv names either frozen SW0142 script."""
    found = []
    proc_root = Path(proc_root)
    if not proc_root.is_dir():
        raise RuntimeError(f"process table unavailable: {proc_root}")
    run_token = str(Path(run_path).resolve())
    queue_token = str(Path(queue_path).resolve())
    for entry in proc_root.iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as exc:
            raise RuntimeError(f"cannot inspect process command line: {entry}") from exc
        argv = raw.replace(b"\0", b" ").decode("utf-8", errors="replace")
        if run_token in argv and "--stage" in argv:
            found.append({"pid": int(entry.name), "argv": argv.strip()})
        elif queue_token in argv:
            found.append({"pid": int(entry.name), "argv": argv.strip()})
    return found


def validate_failed_queue(state: dict, *, expected_fingerprint: dict,
                          expected_queue_sha256: str, proc_root: Path,
                          run_path: Path, queue_path: Path) -> dict:
    """Fail closed unless the sole recorded failure is the known B interruption."""
    if (state.get("experiment") != EXPERIMENT or state.get("status") != "failed"
            or state.get("implementation_fingerprint") != expected_fingerprint
            or state.get("queue_sha256") != expected_queue_sha256):
        raise ValueError("original queue identity/status/fingerprint does not match frozen SW0142")
    rows = state.get("tasks")
    if not isinstance(rows, dict):
        raise ValueError("original queue has no task records")
    passed = ("cache_seed1", "warm_seed1", "preflight_encoder_graph",
              "preflight_core_trainable", "train_encoder_graph")
    for task_id in passed:
        row = rows.get(task_id, {})
        if row.get("status") != "passed" or row.get("artifact_valid") is not True:
            raise ValueError(f"required original stage did not pass: {task_id}")
    supervisor_pid = state.get("supervisor_pid")
    if not isinstance(supervisor_pid, int) or supervisor_pid <= 0:
        raise ValueError("original queue supervisor PID is missing")
    failed = rows.get("train_core_trainable", {})
    if (failed.get("status") != "interrupted_foreign_gpu_owner"
            or failed.get("artifact_valid") not in (None, False)
            or failed.get("returncode") != -15
            or not failed.get("foreign_owner_pids")
            or not isinstance(failed.get("child_pid"), int)
            or failed.get("child_pid") <= 0):
        raise ValueError("B arm was not interrupted by a recorded foreign GPU owner")
    if any(rows.get(task_id, {}).get("status") not in {"queued", "blocked"}
           for task_id in ("predict_encoder_graph", "predict_core_trainable", "score_seed1")):
        raise ValueError("original queue progressed into prediction/scoring; preserve and inspect it manually")

    recorded_pids = [state.get("supervisor_pid")]
    recorded_pids.extend(row.get("child_pid") for row in rows.values())
    live_recorded = sorted({pid for pid in recorded_pids if _pid_live(pid, proc_root)})
    if live_recorded:
        raise RuntimeError(f"original queue process IDs still exist: {live_recorded}")
    live_scripts = _sw0142_processes(proc_root, run_path, queue_path)
    if live_scripts:
        raise RuntimeError(f"live SW0142 runner/queue process found: {live_scripts}")
    return {"recorded_pid_count": len({pid for pid in recorded_pids if isinstance(pid, int)}),
            "foreign_owner_pids": list(failed["foreign_owner_pids"]),
            "interrupted_task_status": failed["status"],
            "interrupted_task_returncode": failed["returncode"]}


def recovery_tasks(run):
    """Only the interrupted arm, both predictions, and the paired CPU score."""
    return [
        {"task_id": "predict_encoder_graph_recovery", "stage": "predict",
         "arm": "encoder_graph", "deps": ["train_encoder_graph"]},
        {"task_id": "train_core_trainable_recovery", "stage": "train",
         "arm": "core_trainable", "deps": ["preflight_core_trainable"]},
        {"task_id": "predict_core_trainable_recovery", "stage": "predict",
         "arm": "core_trainable", "deps": ["train_core_trainable_recovery"]},
        {"task_id": "score_seed1_recovery", "stage": "score",
         "deps": ["predict_encoder_graph_recovery", "predict_core_trainable_recovery"]},
    ]


def recovery_groups():
    # Predict completed A first while GPU0 is available, then train B, predict
    # B, and score with CUDA hidden. A single GPU-0 worker avoids contention.
    return [("predict_encoder_graph_recovery",),
            ("train_core_trainable_recovery",),
            ("predict_core_trainable_recovery",),
            ("score_seed1_recovery",)]


def validate_inherited_artifacts(run, stage_queue):
    required = [
        {"task_id": "cache_seed1", "stage": "cache"},
        {"task_id": "warm_seed1", "stage": "warm"},
        {"task_id": "preflight_encoder_graph", "stage": "preflight", "arm": "encoder_graph"},
        {"task_id": "preflight_core_trainable", "stage": "preflight", "arm": "core_trainable"},
        {"task_id": "train_encoder_graph", "stage": "train", "arm": "encoder_graph"},
    ]
    bad = [task["task_id"] for task in required if not stage_queue.validate_result(task)]
    if bad:
        raise ValueError(f"completed original artifacts no longer validate: {bad}")
    # Validate exact existing optimizer/source/warm lineage, not just markers.
    run.validate_training("encoder_graph")
    return [row["task_id"] for row in required]


def check_outputs_clear(run):
    paths = [run.RESULTS / "encoder_graph", run.RESULTS / "core_trainable",
             run.HERE / "results_archive" / "score_seed1.json"]
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("preserve existing SW0142 prediction/score outputs: " + ", ".join(existing))


def require_new_recovery_archive(archive: Path):
    archive = Path(archive)
    if archive.exists():
        raise FileExistsError(f"preserve existing recovery attempt: {archive}")
    return archive


def _archive_partial_b(run, recovery_archive: Path):
    source = run.TRAIN / "core_trainable_seed1"
    if not source.exists():
        return None
    if source.is_symlink() or not source.is_dir() or source.parent.resolve() != run.TRAIN.resolve():
        raise ValueError("unexpected B training path type/location")
    completion = source / "TRAINING_COMPLETED.json"
    if completion.exists():
        raise FileExistsError("B completion marker exists; do not archive or rerun")
    manifest = source / "manifest.json"
    if manifest.exists():
        try:
            row = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("B manifest exists but is unreadable; preserve for inspection") from exc
        if row.get("status") == "training_complete":
            raise FileExistsError("B manifest claims completion; do not rerun")
    if any(source.iterdir()):
        raise ValueError("incomplete B directory is not empty; preserve for manual inspection")
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    destination = source.with_name(f"{source.name}.interrupted_{stamp}_{os.getpid()}")
    if destination.exists():
        raise FileExistsError(f"preserve existing interrupted archive: {destination}")
    os.replace(source, destination)
    return destination


def validate_interrupted_b_directory(source: Path):
    """Allow absent or empty B output; reject anything that may hold progress."""
    source = Path(source)
    if source.is_symlink():
        raise ValueError("unexpected B training path type/location")
    if not source.exists():
        return False
    if not source.is_dir():
        raise ValueError("unexpected B training path type/location")
    if (source / "TRAINING_COMPLETED.json").exists():
        raise FileExistsError("B completion marker exists; do not rerun")
    if any(source.iterdir()):
        raise ValueError("interrupted B directory is not empty; preserve for manual inspection")
    return True


def _precheck(run, stage_queue, state_path: Path, proc_root: Path):
    if not state_path.is_file():
        raise FileNotFoundError(f"original terminal queue state missing: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    queue_path = Path(stage_queue.__file__).resolve()
    queue_evidence = validate_failed_queue(
        state, expected_fingerprint=run.fingerprint(),
        expected_queue_sha256=run.sha256(queue_path), proc_root=proc_root,
        run_path=run.HERE / "run.py", queue_path=queue_path)
    inherited = validate_inherited_artifacts(run, stage_queue)
    check_outputs_clear(run)
    train_b = run.TRAIN / "core_trainable_seed1"
    validate_interrupted_b_directory(train_b)
    return state, queue_evidence, inherited


class RecoveryQueue:
    """Small supervisor that delegates actual stage children to frozen Queue._execute."""
    def __init__(self, run, stage_queue, archive: Path, state_path: Path, lock_path: Path):
        import fcntl
        self.experiment_run = run
        self.sq = stage_queue
        self.archive = archive
        self.state_path = state_path
        self.lock_path = lock_path
        self.guard = threading.RLock()
        self.tasks = recovery_tasks(run)
        self.state = {"status": "recovery_running", "experiment": EXPERIMENT,
                      "recovery_name": RECOVERY_NAME, "supervisor_pid": os.getpid(),
                      "started": time.time(), "implementation_fingerprint": run.fingerprint(),
                      "original_queue_state_sha256": None,
                      "queue_sha256": run.sha256(Path(__file__)),
                      "frozen_stage_queue_sha256": run.sha256(Path(stage_queue.__file__)),
                      "owner_dispatcher_sha256": run.sha256(Path(stage_queue.owner.__file__)),
                      "physical_gpu_policy": "GPU0 only", "tasks": {
                          row["task_id"]: {"task": row, "status": "queued"} for row in self.tasks}}
        self.lock_handle = self.lock_path.open("x+")
        fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        self._save()

    def _save(self):
        with self.guard:
            temp = self.state_path.with_name(self.state_path.name + f".{os.getpid()}.tmp")
            with temp.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(self.state, stream, indent=2, allow_nan=False)
                stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
            os.replace(temp, self.state_path)

    def _update(self, task_id, **fields):
        with self.guard:
            self.state["tasks"][task_id].update(fields)
            self._save()

    def _reserve_gpu0(self, task_id):
        import fcntl
        owner = self.sq.owner
        lease_path = self.sq.GPU_LEASE_DIR / "gpu0.lock"
        while True:
            lease_path.parent.mkdir(parents=True, exist_ok=True)
            lease = lease_path.open("a+")
            try:
                fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lease.close(); self._update(task_id, status="waiting_for_gpu0_lease", last_wait=time.time()); time.sleep(5); continue
            try:
                uuid, memory, owners = owner._nvidia_gpu_info(0)
            except BaseException:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); raise
            if not owners and memory <= self.sq.GPU_LIMIT_MIB:
                return 0, lease, uuid, memory
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close()
            self._update(task_id, status="waiting_for_exclusive_gpu0", last_wait=time.time(), memory_mib=memory)
            time.sleep(5)

    def _reserve(self, task_id):
        return self._reserve_gpu0(task_id)

    def _execute(self, task):
        # Frozen _execute supplies process ownership monitoring, per-task logs,
        # child termination only, and lease lifetime management.
        return self.sq.Queue._execute(self, task)

    def _terminate(self, proc):
        # Delegate exactly to the frozen queue's process-group termination
        # policy; it only signals the child process group this supervisor owns.
        return self.sq.Queue._terminate(proc)

    def run(self, original_state_sha, archived_b):
        self.state["original_queue_state_sha256"] = original_state_sha
        self.state["archived_interrupted_b_directory"] = str(archived_b) if archived_b else None
        self._save()
        failed = False
        by_id = {row["task_id"]: row for row in self.tasks}
        for group in recovery_groups():
            task = by_id[group[0]]
            for dep in task["deps"]:
                if dep in by_id:
                    if not self.sq.validate_result(by_id[dep]):
                        failed = True; self._update(task["task_id"], status="blocked_invalid_dependency", dependency=dep); break
                else:
                    dep_task = next((row for row in self.sq.task_plan() if row["task_id"] == dep), None)
                    if dep_task is None or not self.sq.validate_result(dep_task):
                        failed = True; self._update(task["task_id"], status="blocked_invalid_dependency", dependency=dep); break
            if failed:
                break
            if task["stage"] == "score":
                outcomes = [self._execute(task)]
            else:
                outcomes = [self._execute(task)]
            if not all(outcomes):
                failed = True; break
        self.state["status"] = "failed" if failed else "complete"
        self.state["completed"] = time.time(); self._save()
        return 1 if failed else 0


def run_recovery(repo: Path, original_state: Path, *, execute=False,
                 proc_root=Path("/proc")):
    run, stage_queue = _load_modules(repo)
    state, evidence, inherited = _precheck(run, stage_queue, Path(original_state), Path(proc_root))
    archive = require_new_recovery_archive(run.HERE / "queue_runtime_recovery_v3_20261012")
    prediction_paths = [run.RESULTS / "encoder_graph", run.RESULTS / "core_trainable"]
    plan = {"experiment": EXPERIMENT, "recovery_name": RECOVERY_NAME,
            "original_queue_state": str(original_state), "original_failed_task": "train_core_trainable",
            "gpu_policy": "GPU0 only; wait when owned/busy; never kill foreign processes",
            "validated_inherited_tasks": inherited,
            "stages": ["predict encoder_graph", "train core_trainable", "predict core_trainable", "CPU score"],
            "outputs_must_be_absent": [str(path) for path in prediction_paths] + [str(run.HERE / "results_archive" / "score_seed1.json")],
            "original_state_sha256": _sha256(Path(original_state)),
            "terminal_guard": evidence}
    if not execute:
        return {"status": "dry_run_validated", "plan": plan}

    import fcntl
    # Allocate an immutable new recovery namespace before moving only the
    # incomplete B directory. Original queue state/logs and A are read-only.
    archive.mkdir(parents=False, exist_ok=False)
    state_path = archive / "recovery_state.json"
    lock_path = archive / "recovery.lock"
    old_globals = {name: getattr(stage_queue, name) for name in ("ARCHIVE", "STATE", "LOCK")}
    stage_queue.ARCHIVE, stage_queue.STATE, stage_queue.LOCK = archive, state_path, lock_path
    q = RecoveryQueue(run, stage_queue, archive, state_path, lock_path)
    archived_b = None
    try:
        # Revalidate immediately before any filesystem mutation.
        _precheck(run, stage_queue, Path(original_state), Path(proc_root))
        archived_b = _archive_partial_b(run, archive)
        return q.run(plan["original_state_sha256"], archived_b)
    except BaseException as exc:
        q.state["status"] = "failed_before_recovery_stages"
        q.state["error"] = f"{type(exc).__name__}: {exc}"
        q.state["archived_interrupted_b_directory"] = str(archived_b) if archived_b else None
        q._save()
        raise
    finally:
        try:
            fcntl.flock(q.lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            q.lock_handle.close()
            for name, value in old_globals.items():
                setattr(stage_queue, name, value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--original-state", type=Path, required=True)
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    parser.add_argument("--execute", action="store_true", help="perform the guarded recovery; default is read-only preflight")
    args = parser.parse_args(argv)
    result = run_recovery(args.repo, args.original_state, execute=args.execute, proc_root=args.proc_root)
    if isinstance(result, int):
        return result
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
