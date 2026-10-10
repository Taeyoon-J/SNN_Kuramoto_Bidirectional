"""Owner-aware exclusive-GPU queue for the two frozen native32 transfer evaluations."""
from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

try:
    import fcntl
except ImportError:  # Windows supports dry-run/tests only.
    fcntl = None

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "results_archive"
PREDICTION_ROOT = ARCHIVE / "transfer_predictions_seed1"
STATE = ARCHIVE / "transfer_queue_state.json"
QUEUE_LOCK = ARCHIVE / "transfer_queue.lock"
GPU_LEASES = Path("/tmp/kevinswk_sw0113_gpu_leases")
MEMORY_LIMIT_MIB = 512
MAX_PARALLEL = 2
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]
from collaborative_test.SW_0123_adaptive_temporal_assignment import dispatcher as owner
from collaborative_test.SW_0135_native32_spike_binding.foundation import sha256_file
from collaborative_test.SW_0136_native32_transfer import evaluate


def task_plan():
    return [{"task_id": f"sw0136_{arm}_seed1", "arm": arm} for arm in evaluate.ARMS]


def result_dir(task):
    arm = task.get("arm")
    if arm not in evaluate.ARMS:
        raise ValueError("unregistered SW0136 queue arm")
    return PREDICTION_ROOT / arm


def log_path(task):
    return ARCHIVE / f"{task['task_id']}.log"


def command(task, device="cuda:0"):
    arm = task.get("arm")
    if arm not in evaluate.ARMS:
        raise ValueError("unregistered SW0136 queue arm")
    return [sys.executable, "-m", "collaborative_test.SW_0136_native32_transfer.evaluate",
            "--stage", "predict", "--arm", arm, "--device", device,
            "--output-root", str(PREDICTION_ROOT)]


def gpu_is_exclusive_candidate(memory_mib, compute_owners):
    return not compute_owners and int(memory_mib) <= MEMORY_LIMIT_MIB


def validate_result(task):
    arm = task.get("arm")
    if arm not in evaluate.ARMS:
        return False
    try:
        evaluate.validate_prediction(result_dir(task), arm=arm)
        if arm == "actual_joint":
            source = evaluate.validate_prediction(PREDICTION_ROOT / "source97_qcc_seed1",
                                                  arm="source97_qcc")
            transfer = source["protocol"].get("transfer", {})
            return (source["protocol"].get("primary") == "mapped source97 QCC reference only"
                    and transfer.get("reference_role") == "mapped SW0097 native32 QCC only"
                    and transfer.get("source_core_sha256") == evaluate.sw130.source97.EXPECTED_SOURCE_SHAS[1]
                    and bool(transfer.get("converted_state_sha256")))
        return True
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def score_mean_view(report):
    """Reduce the actual production score nesting while retaining readout names."""
    scores = report.get("scores")
    if not isinstance(scores, dict) or not scores:
        raise ValueError("scoring report missing metrics")
    result = {}
    for name, readouts in scores.items():
        if not isinstance(readouts, dict) or not readouts:
            raise ValueError(f"score row {name} missing readouts")
        result[name] = {}
        for readout, metrics in readouts.items():
            if not isinstance(metrics, dict) or set(metrics) != set(evaluate.METRICS):
                raise ValueError(f"score row {name}/{readout} has unexpected metric schema")
            result[name][readout] = {metric: float(value["mean"])
                                     for metric, value in metrics.items()}
    return result


def _atomic_json(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


class TransferQueue:
    def __init__(self):
        if fcntl is None:
            raise RuntimeError("SW0136 GPU queue requires Linux flock")
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        self.queue_lock = QUEUE_LOCK.open("a+")
        try:
            fcntl.flock(self.queue_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.queue_lock.close()
            raise RuntimeError("another SW0136 transfer queue owns the lock") from exc
        conflicts = [STATE, ARCHIVE / "transfer_evaluation_seed1"]
        for task in task_plan():
            conflicts.extend((result_dir(task), log_path(task)))
        conflicts.append(PREDICTION_ROOT / "source97_qcc_seed1")
        existing = [str(path) for path in conflicts if path.exists()]
        if existing:
            fcntl.flock(self.queue_lock.fileno(), fcntl.LOCK_UN); self.queue_lock.close()
            raise FileExistsError("preserve existing SW0136 attempt paths: " + ", ".join(existing))
        self.lock = threading.RLock()
        self.tasks = task_plan()
        self.state = {
            "experiment": "SW0136_native32_transfer", "stage": "two_arm_transfer_screen",
            "status": "running", "supervisor_pid": os.getpid(),
            "started": time.time(), "max_parallel": MAX_PARALLEL,
            "implementation_fingerprint": evaluate.implementation_fingerprint(),
            "queue_sha256": sha256_file(HERE / "transfer_queue.py"),
            "owner_dispatcher_sha256": sha256_file(Path(owner.__file__)),
            "ground_truth_used_for_prediction": False, "optimizer_updates": 0,
            "tasks": {row["task_id"]: {"task": row, "status": "queued"} for row in self.tasks},
        }
        self.state["tasks"]["score"] = {"task": {"task_id": "score", "stage": "gt_last_score"},
                                          "status": "blocked_on_predictions"}
        self._save()

    def _save(self):
        with self.lock:
            _atomic_json(STATE, self.state)

    def _update(self, task_id, **values):
        with self.lock:
            self.state["tasks"][task_id].update(values)
            _atomic_json(STATE, self.state)

    def _gpu_reserve(self, task_id):
        while True:
            for gpu in range(4):
                lease_path = GPU_LEASES / f"gpu{gpu}.lock"
                lease_path.parent.mkdir(parents=True, exist_ok=True)
                lease = lease_path.open("a+")
                try:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lease.close(); continue
                try:
                    gpu_uuid, memory, owners = owner._nvidia_gpu_info(gpu)
                    if gpu_is_exclusive_candidate(memory, owners):
                        self._update(task_id, status="reserved", physical_gpu=gpu,
                                     gpu_uuid=gpu_uuid, memory_before_mib=memory,
                                     lease_path=str(lease_path), reservation_pid=os.getpid())
                        return gpu, lease, gpu_uuid
                except BaseException:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); raise
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close()
            self._update(task_id, status="waiting_for_exclusive_gpu", last_wait=time.time())
            time.sleep(5)

    @staticmethod
    def _terminate_owned(proc):
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()

    def _execute(self, task):
        task_id = task["task_id"]
        output, log = result_dir(task), log_path(task)
        lease = proc = None
        try:
            gpu, lease, gpu_uuid = self._gpu_reserve(task_id)
            triton = Path(f"/tmp/kevinswk_sw0136_{task_id}_{int(time.time()*1000)}")
            triton.mkdir(parents=True, exist_ok=False)
            argv = command(task)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", TRITON_CACHE_DIR=str(triton))
            self._update(task_id, status="launching", physical_gpu=gpu, gpu_uuid=gpu_uuid,
                         argv=argv, output=str(output), log_path=str(log),
                         triton_cache=str(triton),
                         environment={"CUDA_VISIBLE_DEVICES": str(gpu), "TRITON_CACHE_DIR": str(triton)},
                         launched=time.time())
            with log.open("x", encoding="utf-8") as stream:
                proc = subprocess.Popen(argv, cwd=str(ROOT), env=env,
                                        stdout=stream, stderr=subprocess.STDOUT,
                                        start_new_session=True, pass_fds=(lease.fileno(),))
                self._update(task_id, status="running", child_pid=proc.pid, started=time.time())
                while proc.poll() is None:
                    time.sleep(5)
                    if proc.poll() is not None:
                        break
                    _uuid, _memory, owners = owner._nvidia_gpu_info(gpu)
                    foreign = owner.foreign_owner_pids(owners, owner._process_tree(proc.pid))
                    if foreign:
                        self._terminate_owned(proc)
                        self._update(task_id, status="interrupted_foreign_gpu_owner",
                                     foreign_owner_pids=foreign, finished=time.time())
                        return "interrupted_foreign_gpu_owner"
            rc = int(proc.returncode)
            valid = rc == 0 and validate_result(task)
            status = "passed" if valid else "failed"
            self._update(task_id, status=status, returncode=rc, artifact_valid=bool(valid),
                         result_sha256=(sha256_file(output / "predictions.npz")
                                        if (output / "predictions.npz").is_file() else None),
                         finished=time.time())
            return status
        except BaseException as exc:
            self._terminate_owned(proc)
            self._update(task_id, status="failed", error=repr(exc), finished=time.time())
            return "failed"
        finally:
            if lease is not None:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close()

    def run(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
            futures = {pool.submit(self._execute, task): task for task in self.tasks}
            outcomes = {futures[future]["task_id"]: future.result() for future in futures}
        if all(value == "passed" for value in outcomes.values()):
            try:
                self._update("score", status="running", started=time.time(),
                             argv=[sys.executable, "-m",
                                   "collaborative_test.SW_0136_native32_transfer.evaluate",
                                   "--stage", "score"])
                report = evaluate.score_transfer()
                self._update("score", status="passed", artifact_valid=True,
                             arm_metrics=score_mean_view(report),
                             finished=time.time())
                final = "transfer_evaluation_complete"
            except BaseException as exc:
                self._update("score", status="failed", error=repr(exc), finished=time.time())
                final = "transfer_evaluation_failed"
        else:
            final = "prediction_tasks_failed_or_interrupted"
        with self.lock:
            self.state.update(status=final, task_outcomes=outcomes, finished=time.time())
            _atomic_json(STATE, self.state)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"max_parallel": MAX_PARALLEL,
                          "tasks": [{**task, "argv": command(task)} for task in task_plan()],
                          "score_after_all_predictions": True,
                          "ground_truth_used_for_prediction": False}, indent=2))
        return
    TransferQueue().run()


if __name__ == "__main__":
    main()

