"""Owner-aware one-task queue for the SW0149 actual-source preflight only."""
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
except ImportError:  # Windows supports dry-run and CPU validators only.
    fcntl = None

# The supervisor is CPU-only. The child receives the selected physical GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4 as owner_queue
from collaborative_test.SW_0149_spike_graph_modularity import preflight

RUNTIME = HERE / "results_archive" / "preflight_queue_20261012"
STATE = RUNTIME / "state.json"
LOCK = RUNTIME / "queue.lock"
OUTPUT = HERE / "results_archive" / "preflight_seed1.json"
LOG = RUNTIME / "preflight_seed1.log"
OWNER_HELPER = ROOT / "collaborative_test/SW_0148_checkpointable_boundary_q/stage_queue_v4.py"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_plan():
    return [{"task_id": "preflight_seed1", "stage": "preflight", "seed": 1,
             "device": "cuda:0", "output": str(OUTPUT), "log": str(LOG),
             "depends_on": []}]


def valid_result(path=OUTPUT):
    try:
        row = json.loads(Path(path).read_text(encoding="utf-8"))
        if (row.get("status") != "preflight_complete"
                or row.get("experiment") != preflight.NAME
                or row.get("seed") != 1
                or row.get("training_admission") is not False
                or row.get("ground_truth_used") is not False
                or row.get("implementation_sha256") != preflight.implementation_sha()
                or row.get("implementation_fingerprint") != preflight.fingerprint()
                or row.get("disposable_optimizer_updates_per_arm") != preflight.UPDATES):
            return False
        parity = row.get("existing_native32_parity", {})
        if not parity.get("sha256") or len(parity.get("exact_arm_horizons", [])) != 4:
            return False
        credit = row.get("first64_modularity_credit", {})
        if (credit.get("images") != 64 or credit.get("positive_M_offdiagonal_entries", 0) <= 0
                or credit.get("negative_M_offdiagonal_entries", 0) <= 0
                or len(credit.get("per_b4_batch", [])) != 16):
            return False
        arms = row.get("arms", [])
        if {arm.get("arm") for arm in arms} != {"encoder_graph", "joint"}:
            return False
        for arm in arms:
            if (not arm.get("fresh_source_parity", {}).get("actual_spike_q_exact")
                    or not arm.get("fresh_source_parity", {}).get("source_graph_exact")
                    or len(arm.get("updates", [])) != preflight.UPDATES):
                return False
        return True
    except (OSError, ValueError, TypeError, KeyError):
        return False


class PreflightQueue:
    def __init__(self):
        if fcntl is None:
            raise RuntimeError("SW0149 preflight queue requires Linux flock")
        RUNTIME.mkdir(parents=True, exist_ok=True)
        self.lock_stream = LOCK.open("a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock_stream.close()
            raise RuntimeError("another SW0149 preflight supervisor owns the queue") from exc
        if STATE.exists() or OUTPUT.exists() or LOG.exists():
            self.lock_stream.close()
            raise FileExistsError("preserving existing SW0149 preflight queue artifacts")
        self.tasks = task_plan()
        self.state_lock = threading.RLock()
        self.state = {"experiment": preflight.NAME, "queue_version": 1,
                      "status": "running", "supervisor_pid": os.getpid(),
                      "owner_helper_sha256": sha(OWNER_HELPER),
                      "owner_queue_sha256": sha(owner_queue.__file__),
                      "owner_dispatcher_sha256": owner_queue.OWNER_GUARD_SHA256,
                      "implementation_sha256": preflight.implementation_sha(),
                      "tasks": {row["task_id"]: {"task": row, "status": "queued"}
                                for row in self.tasks}}
        self.stop_event = threading.Event()
        self._save()

    def _save(self):
        with self.state_lock:
            tmp = STATE.with_name(STATE.name + f".{os.getpid()}.tmp")
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

    def run(self):
        task = self.tasks[0]
        task_id = task["task_id"]
        proc = None
        attempt = 1
        try:
            while not self.stop_event.is_set():
                if valid_result():
                    self._update(task_id, status="verified_existing")
                    self.state["status"] = "complete"
                    self._save()
                    return 0
                if OUTPUT.exists() or LOG.exists():
                    self._update(task_id, status="failed_existing_artifact_preserved")
                    self.state["status"] = "failed_existing_artifact_preserved"
                    self._save()
                    return 1
                gpu, gpu_uuid, memory, lease = owner_queue._acquire(self.stop_event)
                retry_foreign = False
                attempt_log = RUNTIME / f"preflight_seed1.attempt{attempt:02d}.log"
                try:
                    if not owner_queue._stable_idle(
                            gpu, gpu_uuid, self.stop_event,
                            lambda elapsed, used: self._update(
                                task_id, status="waiting_stable_gpu", gpu=gpu,
                                gpu_uuid=gpu_uuid, stable_seconds=elapsed,
                                memory_used_mib=used)):
                        continue
                    if OUTPUT.exists() or attempt_log.exists():
                        self._update(task_id, status="failed_existing_artifact_preserved")
                        self.state["status"] = "failed_existing_artifact_preserved"
                        self._save()
                        return 1
                    argv = [sys.executable, "-m",
                            "collaborative_test.SW_0149_spike_graph_modularity.preflight",
                            "--device", "cuda:0", "--output", str(OUTPUT)]
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                    with attempt_log.open("x", encoding="utf-8") as stream:
                        proc = subprocess.Popen(
                            argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                            stdout=stream, stderr=subprocess.STDOUT,
                            start_new_session=True, pass_fds=(lease.fileno(),))
                        self._update(task_id, status="running", attempt=attempt,
                                     child_pid=proc.pid, child_argv=argv, gpu=gpu,
                                     gpu_uuid=gpu_uuid, memory_before_mib=memory,
                                     inherited_lease_fd=True, log=str(attempt_log))
                        foreign = []
                        while proc.poll() is None and not self.stop_event.is_set():
                            time.sleep(5)
                            if proc.poll() is not None:
                                break
                            try:
                                current_uuid, _used, owners = owner_queue.gpu_guard._gpu_info(gpu)
                            except Exception:
                                continue
                            if current_uuid != gpu_uuid:
                                foreign = ["gpu_uuid_changed"]
                            else:
                                foreign = owner_queue.gpu_guard._foreign_owners(
                                    owners, owner_queue.gpu_guard._process_tree(proc.pid))
                            if foreign:
                                owner_queue._stop_owned(proc)
                                retry_foreign = True
                                self._update(task_id, status="interrupted_foreign_gpu_owner",
                                             foreign_owner_pids=foreign)
                                break
                    code = proc.wait()
                    good = code == 0 and valid_result()
                    self._update(task_id, status="passed" if good else "failed",
                                 returncode=code, artifact_valid=bool(good),
                                 finished=time.time())
                    if good:
                        self.state["status"] = "complete"
                        self._save()
                        return 0
                    if not retry_foreign:
                        self.state["status"] = "failed_preflight"
                        self._save()
                        return 1
                except BaseException as exc:
                    if proc is not None and proc.poll() is None:
                        owner_queue._stop_owned(proc)
                    self._update(task_id, status="failed", error=repr(exc),
                                 finished=time.time())
                    self.state["status"] = "failed_preflight"
                    self._save()
                    return 1
                finally:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                    lease.close()
                # A retry is permitted only when foreign ownership interrupted
                # the disposable preflight and no result was committed.
                if not retry_foreign or OUTPUT.exists():
                    self.state["status"] = "failed_preflight_preserved"
                    self._save()
                    return 1
                self._update(task_id, status="queued_after_foreign_owner",
                             prior_log=str(attempt_log))
                attempt += 1
            self.state["status"] = "cancelled"
            self._save()
            return 1
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
        print(json.dumps({"experiment": preflight.NAME, "tasks": task_plan(),
                          "owner_aware": True, "maximum_parallel_gpu_tasks": 1,
                          "ground_truth": False, "dry_run": True}, indent=2), flush=True)
        return 0
    return PreflightQueue().run()


if __name__ == "__main__":
    raise SystemExit(main())
