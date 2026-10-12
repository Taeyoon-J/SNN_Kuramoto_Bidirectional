"""SW0148 v4 continues the active v3 run with identity-checked training handoff."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
try:
    import fcntl
except ImportError:  # Windows can validate plans/tests but cannot own Linux leases.
    fcntl = None
import json
import os
import pickle
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0126_history_event_binding import pilot_queue as gpu_guard
from collaborative_test.SW_0146_native32_boundary_q_pilot import run as sw146
from collaborative_test.SW_0148_checkpointable_boundary_q import run as experiment
from collaborative_test.SW_0148_checkpointable_boundary_q import train as trainer

RUNTIME = HERE / "results_archive" / "queue_v4_20261012"
STATE = RUNTIME / "state.json"
QUEUE_LOCK = RUNTIME / "queue.lock"
GPU_LEASE_DIR = Path("/tmp/kevinswk_sw0113_gpu_leases")
MAX_MEMORY_MIB = 512
STABLE_SECONDS = 5.0
SAMPLE_SECONDS = 5.0
OWNER_GUARD = ROOT / "collaborative_test/SW_0126_history_event_binding/pilot_queue.py"
OWNER_GUARD_SHA256 = "c59cdc39af20c824930d1926c8cec1418ee818577b8cc4240ad6eaae836764f6"
HANDOFF_IDENTITY_SHA256 = "684b6ddf7296b8e6a154f4bc46f0e1fbbc0fc1cf96e00bd109ac95b71077928d"
HANDOFF_IDENTITIES = {
    "supervisor": {"pid": 551842, "start_ticks": 398628819, "pgid": 551842,
                   "argv": ["/Data0/kevinswk/envs/snn/bin/python",
                            "/Data0/kevinswk/patch_v2_sw/collaborative_test/SW_0148_checkpointable_boundary_q/stage_queue_v3.py"]},
    "encoder_graph": {"pid": 769057, "start_ticks": 399259393, "pgid": 769057,
                      "argv": ["/Data0/kevinswk/envs/snn/bin/python", "-m",
                               "collaborative_test.SW_0148_checkpointable_boundary_q.run",
                               "--stage", "train", "--arm", "encoder_graph", "--device", "cuda:0"]},
    # Handoff observation recorded this PID and process group absent.
    "joint": {"pid": 769533, "start_ticks": None, "pgid": 769533, "missing": True},
}


def _proc_record(pid):
    """Return a small identity-bound /proc record; None means PID absent."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        tail = stat[stat.rfind(")") + 2:].split()
        argv = [part.decode("utf-8", errors="replace") for part in
                Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if part]
        return {"pid": int(pid), "state": tail[0], "ppid": int(tail[1]),
                "pgid": int(tail[2]), "start_ticks": int(tail[19]), "argv": argv}
    except (FileNotFoundError, ProcessLookupError):
        return None
    except PermissionError as exc:
        # An inaccessible captured PID is not evidence that its writer is gone.
        raise RuntimeError(f"cannot verify predecessor PID {pid}") from exc


def _process_group_rows(pgid):
    rows = []
    try:
        entries = list(os.scandir("/proc"))
    except OSError:
        raise RuntimeError("cannot verify predecessor process groups")
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as exc:
            raise RuntimeError(f"cannot verify process-group membership for PID {pid}") from exc
        tail = stat[stat.rfind(")") + 2:].split()
        if len(tail) < 20 or int(tail[2]) != pgid or tail[0] == "Z":
            continue
        # Membership in a captured predecessor process group establishes that
        # this is one of its still-running descendants (for example nvidia-smi).
        # Wait for the group to empty; never signal or classify it as foreign.
        try:
            argv = [part.decode("utf-8", errors="replace") for part in
                    Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if part]
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            argv = []
        rows.append({"pid": pid, "state": tail[0], "ppid": int(tail[1]),
                     "pgid": int(tail[2]), "start_ticks": int(tail[19]), "argv": argv})
    return rows


def predecessor_clear(arm, *, proc_reader=_proc_record, group_reader=_process_group_rows):
    """Fail closed on PID reuse/unexpected group members; never signals old jobs."""
    if arm == "joint":
        expected = [HANDOFF_IDENTITIES["joint"]]
    elif arm == "encoder_graph":
        expected = [HANDOFF_IDENTITIES["supervisor"], HANDOFF_IDENTITIES["encoder_graph"]]
    else:
        raise ValueError("unknown SW0148 handoff arm")
    active_expected = set()
    for identity in expected:
        try:
            row = proc_reader(identity["pid"])
        except PermissionError as exc:
            raise RuntimeError(f"cannot verify predecessor PID {identity['pid']}") from exc
        if row is None or row.get("state") == "Z":
            continue
        if (identity.get("missing") or row.get("start_ticks") != identity["start_ticks"]
                or row.get("pgid") != identity["pgid"]
                or row.get("argv") != identity.get("argv")):
            raise RuntimeError(f"predecessor PID {identity['pid']} was reused or command changed")
        active_expected.add(identity["pid"])
    live_group_members = set()
    for identity in expected:
        try:
            live_group_members.update(row.get("pid") for row in group_reader(identity["pgid"]))
        except PermissionError as exc:
            raise RuntimeError(f"cannot verify predecessor process group {identity['pgid']}") from exc
    # The process group can contain short-lived owned helpers not present in
    # the initial PID snapshot. Their presence means wait, not handoff or kill.
    return not active_expected and not live_group_members


def _handoff_manifest_sha():
    payload = json.dumps(HANDOFF_IDENTITIES, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_plan():
    tasks = []
    for arm in sw146.ARMS:
        tasks.append({"task_id": f"train_{arm}", "stage": "train", "arm": arm,
                      "depends_on": [], "output": str(trainer.TRAIN / f"{arm}_seed1")})
    for arm in sw146.ARMS:
        tasks.append({"task_id": f"predict_{arm}", "stage": "predict", "arm": arm,
                      "depends_on": [f"train_{arm}"],
                      "output": str(experiment._prediction_dir(arm))})
    tasks.append({"task_id": "score", "stage": "score", "arm": None,
                  "depends_on": [f"predict_{arm}" for arm in sw146.ARMS],
                  "output": str(experiment.ARCHIVE / "evaluation_seed1.json")})
    return tasks


def valid_result(task):
    try:
        if task["stage"] == "train":
            trainer.validate_training(task["arm"])
        elif task["stage"] == "predict":
            experiment.validate_prediction(task["arm"])
        else:
            result = json.loads(Path(task["output"]).read_text(encoding="utf-8"))
            if (result.get("status") != "complete" or result.get("experiment") != trainer.NAME
                    or result.get("seed") != sw146.SEED
                    or result.get("ground_truth_opened_after_all_predictions_validated") is not True
                    or result.get("ground_truth_used_for_prediction") is not False):
                return False
        return True
    except (OSError, ValueError, TypeError, KeyError, AssertionError):
        return False


def _acquire(stop_event, preferred_gpu=None):
    candidates = [preferred_gpu] if preferred_gpu is not None else list(range(4))
    while not stop_event.is_set():
        for gpu in candidates:
            lease_path = GPU_LEASE_DIR / f"gpu{gpu}.lock"
            lease_path.parent.mkdir(parents=True, exist_ok=True)
            lease = lease_path.open("a+")
            try:
                fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lease.close()
                continue
            try:
                uuid, memory, owners = gpu_guard._gpu_info(gpu)
            except BaseException:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                lease.close()
                continue
            if not owners and int(memory) <= MAX_MEMORY_MIB:
                return gpu, uuid, int(memory), lease
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
            lease.close()
        time.sleep(SAMPLE_SECONDS)
    raise RuntimeError("queue stopped while waiting for an exclusive GPU")


def required_resume_gpu(task):
    """Read an existing checkpoint's exact CUDA visibility binding, fail closed."""
    if task.get("stage") != "train":
        return None
    from collaborative_test.SW_0148_checkpointable_boundary_q.checkpoint import (
        inspect_committed_metadata, is_only_uncommitted_temp,
    )
    checkpoint_dir = trainer.WORK / f"{task['arm']}_seed1" / "checkpoints"
    boundary = checkpoint_dir / "boundary.ptx"
    if not boundary.exists():
        if checkpoint_dir.exists() and not is_only_uncommitted_temp(checkpoint_dir):
            raise ValueError("work directory without a committed boundary is not a known atomic temp")
        return None
    try:
        metadata = inspect_committed_metadata(checkpoint_dir)
    except (OSError, ValueError, TypeError, KeyError, EOFError,
            pickle.UnpicklingError, RuntimeError) as exc:
        raise ValueError("checkpoint container failed integrity validation") from exc
    runtime = metadata.get("runtime")
    visibility = runtime.get("cuda_visible_devices") if isinstance(runtime, dict) else None
    updates = metadata.get("completed_updates")
    if (not isinstance(visibility, str) or visibility not in {"0", "1", "2", "3"}
            or runtime.get("device_type") != "cuda"
            or runtime.get("device_index") != 0
            or not isinstance(updates, int) or isinstance(updates, bool)
            or not 0 <= updates <= sw146.UPDATES):
        raise ValueError("checkpoint has no valid single-physical-GPU resume binding")
    return int(visibility)


def _stable_idle(gpu, expected_uuid, stop_event, on_sample):
    started = None
    while not stop_event.is_set():
        try:
            uuid, memory, owners = gpu_guard._gpu_info(gpu)
        except Exception:
            return False
        if uuid != expected_uuid or owners or int(memory) > MAX_MEMORY_MIB:
            return False
        now = time.monotonic()
        if started is None:
            started = now
        elapsed = now - started
        on_sample(elapsed, int(memory))
        if elapsed >= STABLE_SECONDS:
            try:
                uuid, memory, owners = gpu_guard._gpu_info(gpu)
            except Exception:
                return False
            return uuid == expected_uuid and not owners and int(memory) <= MAX_MEMORY_MIB
        time.sleep(min(SAMPLE_SECONDS, STABLE_SECONDS - elapsed))
    return False


def safe_foreign_retry(task, output):
    """Permit only resumable training or absent-output prediction retries."""
    output = Path(output)
    if output.exists():
        return False
    if task["stage"] == "predict":
        return True
    if task["stage"] != "train":
        return False
    from collaborative_test.SW_0148_checkpointable_boundary_q.checkpoint import (
        inspect_committed_metadata, is_only_uncommitted_temp,
    )
    checkpoint_dir = trainer.WORK / f"{task['arm']}_seed1" / "checkpoints"
    boundary = checkpoint_dir / "boundary.ptx"
    if boundary.is_file():
        try:
            inspect_committed_metadata(checkpoint_dir)
            return True
        except (OSError, ValueError, TypeError, KeyError, EOFError, pickle.UnpicklingError, RuntimeError):
            return False
    return not checkpoint_dir.exists() or is_only_uncommitted_temp(checkpoint_dir)


def _argv(task):
    if task["stage"] == "train":
        return [sys.executable, "-m", "collaborative_test.SW_0148_checkpointable_boundary_q.recovery_v2",
                "--arm", task["arm"], "--device", "cuda:0"]
    return [sys.executable, "-m", "collaborative_test.SW_0148_checkpointable_boundary_q.run",
            "--stage", task["stage"], *( ["--arm", task["arm"]] if task.get("arm") else [] ),
            *( ["--device", "cuda:0"] if task["stage"] != "score" else [] )]


def _stop_owned(proc):
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


class Queue:
    def __init__(self):
        if fcntl is None:
            raise RuntimeError("SW0148 queue requires Linux flock")
        if fcntl is None:
            raise RuntimeError("SW0148 queue requires Linux flock")
        RUNTIME.mkdir(parents=True, exist_ok=True)
        self.lock_stream = QUEUE_LOCK.open("a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock_stream.close()
            raise RuntimeError("another SW0148 supervisor owns the queue") from exc
        if STATE.exists():
            self.lock_stream.close()
            raise FileExistsError(f"preserving previous SW0148 queue state: {STATE}")
        if _sha(OWNER_GUARD) != OWNER_GUARD_SHA256:
            self.lock_stream.close()
            raise RuntimeError("reviewed SW0148 GPU owner guard changed")
        self.stop_event = threading.Event()
        self.state_lock = threading.RLock()
        self.tasks = task_plan()
        self.state = {"experiment": trainer.NAME, "status": "running",
                      "queue_version": 4, "supervisor_pid": os.getpid(), "no_overwrite": True,
                      "owner_dispatcher_sha256": OWNER_GUARD_SHA256,
                      "predecessor_handoff": {"identity_file_sha256": HANDOFF_IDENTITY_SHA256,
                                               "identities": HANDOFF_IDENTITIES,
                                               "manifest_sha256": _handoff_manifest_sha()},
                      "tasks": {row["task_id"]: {"task": row, "status": "queued"}
                                for row in self.tasks}}
        self._save()

    def _save(self):
        with self.state_lock:
            tmp = STATE.with_name(STATE.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
            with tmp.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(self.state, stream, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, STATE)

    def _update(self, task_id, **fields):
        with self.state_lock:
            self.state["tasks"][task_id].update(fields)
            self._save()

    def _wait_for_training_handoff(self, task):
        arm = task["arm"]
        while not self.stop_event.is_set():
            if predecessor_clear(arm):
                self._update(task["task_id"], status="predecessor_handoff_clear",
                             predecessor_manifest_sha256=_handoff_manifest_sha())
                return True
            self._update(task["task_id"], status="waiting_for_predecessor_handoff",
                         predecessor_manifest_sha256=_handoff_manifest_sha())
            time.sleep(SAMPLE_SECONDS)
        self._update(task["task_id"], status="cancelled_waiting_for_predecessor")
        return False

    def _execute_gpu(self, task):
        task_id = task["task_id"]
        if task["stage"] == "train":
            try:
                if not self._wait_for_training_handoff(task):
                    return False
            except (OSError, RuntimeError, ValueError) as exc:
                self._update(task_id, status="failed_predecessor_handoff",
                             error=repr(exc), finished=time.time())
                return False
        if valid_result(task):
            self._update(task_id, status="verified_existing")
            return True
        output = Path(task["output"])
        if output.exists():
            self._update(task_id, status="failed_existing_output_preserved")
            return False
        proc = None
        attempt = 1
        while not self.stop_event.is_set():
            if valid_result(task):
                self._update(task_id, status="verified_existing")
                return True
            if output.exists():
                self._update(task_id, status="failed_existing_output_preserved")
                return False
            try:
                preferred_gpu = required_resume_gpu(task)
            except (OSError, ValueError, TypeError, KeyError, EOFError,
                    pickle.UnpicklingError, RuntimeError) as exc:
                self._update(task_id, status="failed_checkpoint_admission",
                             error=repr(exc), finished=time.time())
                return False
            gpu, gpu_uuid, memory, lease = _acquire(self.stop_event, preferred_gpu)
            retry_foreign = False
            log = RUNTIME / f"{task_id}.attempt{attempt:02d}.log"
            try:
                if not _stable_idle(gpu, gpu_uuid, self.stop_event,
                                    lambda elapsed, used: self._update(
                                        task_id, status="waiting_stable_gpu", gpu=gpu,
                                        gpu_uuid=gpu_uuid, stable_seconds=elapsed,
                                        memory_used_mib=used)):
                    continue
                if output.exists() or log.exists():
                    self._update(task_id, status="failed_existing_output_preserved")
                    return False
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                           MKL_NUM_THREADS="1")
                argv = _argv(task)
                with log.open("x", encoding="utf-8") as stream:
                    proc = subprocess.Popen(argv, cwd=ROOT, env=env,
                        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                        start_new_session=True, pass_fds=(lease.fileno(),))
                    self._update(task_id, status="running", child_pid=proc.pid,
                                 gpu=gpu, gpu_uuid=gpu_uuid, memory_used_before_mib=memory,
                                 argv=argv, log=str(log), inherited_lease_fd=True)
                    foreign = []
                    while proc.poll() is None:
                        time.sleep(5)
                        if proc.poll() is not None:
                            break
                        try:
                            current_uuid, _used, owners = gpu_guard._gpu_info(gpu)
                        except Exception:
                            continue
                        if current_uuid != gpu_uuid:
                            foreign = ["gpu_uuid_changed"]
                        else:
                            foreign = gpu_guard._foreign_owners(
                                owners, gpu_guard._process_tree(proc.pid))
                        if foreign:
                            self._update(task_id, status="interrupted_foreign_gpu_owner",
                                         foreign_owner_pids=foreign)
                            _stop_owned(proc)
                            retry_foreign = True
                            break
                    code = proc.wait()
                    good = code == 0 and valid_result(task)
                    self._update(task_id, status="passed" if good else "failed",
                                 returncode=code, artifact_valid=bool(good),
                                 finished=time.time())
                    if good:
                        return True
                    if not retry_foreign:
                        return False
            except BaseException as exc:
                if proc is not None and proc.poll() is None:
                    _stop_owned(proc)
                self._update(task_id, status="failed", error=repr(exc), finished=time.time())
                return False
            finally:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                lease.close()
            if retry_foreign:
                # train.py resumes only from a self-verifying committed logical boundary.
                if not safe_foreign_retry(task, output):
                    return False
                # This is an explicit foreign-owner recovery, never a numerical retry.
                self._update(task_id, status="queued_after_foreign_owner",
                             prior_foreign_owner_pids=self.state["tasks"][task_id].get("foreign_owner_pids", []),
                             retry_log=str(RUNTIME / f"{task_id}.attempt{attempt:02d}.log"))
                attempt += 1
                continue
        self._update(task_id, status="cancelled")
        return False

    def _execute_cpu(self, task):
        if valid_result(task):
            self._update(task["task_id"], status="verified_existing")
            return True
        if Path(task["output"]).exists():
            self._update(task["task_id"], status="failed_existing_output_preserved")
            return False
        log = RUNTIME / f"{task['task_id']}.log"
        if log.exists():
            self._update(task["task_id"], status="failed_existing_log_preserved")
            return False
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1")
        argv = _argv(task)
        try:
            with log.open("x", encoding="utf-8") as stream:
                proc = subprocess.Popen(argv, cwd=ROOT, env=env,
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=True)
                self._update(task["task_id"], status="running", child_pid=proc.pid,
                             argv=argv, log=str(log), cuda_hidden=True)
                code = proc.wait()
            good = code == 0 and valid_result(task)
            self._update(task["task_id"], status="passed" if good else "failed",
                         returncode=code, artifact_valid=bool(good), finished=time.time())
            return good
        except BaseException as exc:
            self._update(task["task_id"], status="failed", error=repr(exc), finished=time.time())
            return False

    def run(self):
        try:
            train_tasks = [task for task in self.tasks if task["stage"] == "train"]
            with ThreadPoolExecutor(max_workers=2) as pool:
                outcomes = list(pool.map(self._execute_gpu, train_tasks))
            if not all(outcomes):
                self.state["status"] = "training_failed_or_interrupted"
                self._save()
                return 1
            prediction_tasks = [task for task in self.tasks if task["stage"] == "predict"]
            with ThreadPoolExecutor(max_workers=2) as pool:
                outcomes = list(pool.map(self._execute_gpu, prediction_tasks))
            if not all(outcomes):
                self.state["status"] = "prediction_failed"
                self._save()
                return 1
            score = next(task for task in self.tasks if task["stage"] == "score")
            if not all(valid_result(task) for task in prediction_tasks):
                self.state["status"] = "score_blocked_invalid_predictions"
                self._save()
                return 1
            result = self._execute_cpu(score)
            self.state["status"] = "complete" if result else "score_failed"
            self._save()
            return 0 if result else 1
        finally:
            try:
                fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_UN)
                self.lock_stream.close()
            except (AttributeError, OSError):
                pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"experiment": trainer.NAME, "tasks": task_plan(),
                          "stable_idle_seconds": STABLE_SECONDS,
                          "owner_aware": True, "training_resume": "committed logical update only",
                          "dry_run": True}, indent=2), flush=True)
        return 0
    return Queue().run()


if __name__ == "__main__":
    raise SystemExit(main())
