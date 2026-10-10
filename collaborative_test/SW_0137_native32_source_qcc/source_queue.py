"""Exclusive owner-aware queue for three source predictions and GT-last scoring."""
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
except ImportError:  # Windows is limited to dry-run/contract tests.
    fcntl = None

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "results_archive"
OUTPUT_ROOT = HERE / "results_archive" / "source_predictions"
STATE = ARCHIVE / "source_qcc_queue_state.json"
LOCK = ARCHIVE / "source_qcc_queue.lock"
GPU_LEASES = Path("/tmp/kevinswk_sw0113_gpu_leases")
MEMORY_LIMIT_MIB = 512
MAX_PARALLEL = 2
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]
from collaborative_test.SW_0123_adaptive_temporal_assignment import dispatcher as owner
from collaborative_test.SW_0135_native32_spike_binding.foundation import sha256_file
from collaborative_test.SW_0137_native32_source_qcc import run


def task_plan():
    # Start independent seeds0/2 together; seed1 checks/reuses a live SW0136 source result.
    return [{"task_id": f"sw0137_source_qcc_seed{seed}", "seed": seed}
            for seed in (0, 2, 1)]


def output_path(task):
    seed = int(task.get("seed", -1))
    if seed not in run.SEEDS:
        raise ValueError("unregistered SW0137 source seed")
    return run._prediction_dir(seed, OUTPUT_ROOT)


def log_path(task):
    return ARCHIVE / f"{task['task_id']}.log"


def command(task, device="cuda:0"):
    seed = int(task.get("seed", -1))
    if seed not in run.SEEDS:
        raise ValueError("unregistered SW0137 source seed")
    argv = [sys.executable, "-m", "collaborative_test.SW_0137_native32_source_qcc.run",
            "--stage", "predict", "--seed", str(seed), "--device", device,
            "--output-root", str(OUTPUT_ROOT)]
    if seed == 1:
        argv.append("--no-reuse-seed1")
    return argv


def gpu_is_exclusive_candidate(memory_mib, compute_owners):
    return not compute_owners and int(memory_mib) <= MEMORY_LIMIT_MIB


def validate_result(task):
    try:
        run.validate_prediction(output_path(task), seed=int(task["seed"]))
        return True
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def ensure_absent(paths):
    existing = [str(Path(path)) for path in paths if Path(path).exists()]
    if existing:
        raise FileExistsError("preserve existing SW0137 attempt paths: " + ", ".join(existing))


def _atomic_json(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


class SourceQCCQueue:
    def __init__(self):
        if fcntl is None:
            raise RuntimeError("SW0137 GPU queue requires Linux flock")
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        self.queue_lock = LOCK.open("a+")
        try:
            fcntl.flock(self.queue_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.queue_lock.close(); raise RuntimeError("another SW0137 queue owns the lock") from exc
        paths = [STATE, ARCHIVE / "source_qcc_evaluation"]
        for task in task_plan():
            paths.extend((output_path(task), log_path(task)))
        ensure_absent(paths)
        self.lock = threading.RLock()
        self.tasks = task_plan()
        self.state = {
            "experiment": "SW0137_native32_source_qcc", "stage": "frozen_source_qcc",
            "status": "running", "supervisor_pid": os.getpid(), "started": time.time(),
            "max_parallel": MAX_PARALLEL, "implementation_fingerprint": run.implementation_fingerprint(),
            "queue_sha256": sha256_file(HERE / "source_queue.py"),
            "owner_dispatcher_sha256": sha256_file(Path(owner.__file__)),
            "ground_truth_used_for_prediction": False, "optimizer_updates": 0,
            "tasks": {str(t["seed"]): {"task": t, "status": "queued"} for t in self.tasks},
        }
        self.state["score"] = {"status": "blocked_on_all_three_predictions"}
        self._save()

    def _save(self):
        with self.lock:
            _atomic_json(STATE, self.state)

    def _update(self, key, **values):
        with self.lock:
            self.state["tasks"][str(key)].update(values)
            _atomic_json(STATE, self.state)

    def _reserve(self, seed):
        while True:
            for gpu in range(4):
                path = GPU_LEASES / f"gpu{gpu}.lock"
                path.parent.mkdir(parents=True, exist_ok=True)
                lease = path.open("a+")
                try:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lease.close(); continue
                try:
                    uuid, memory, owners = owner._nvidia_gpu_info(gpu)
                    if gpu_is_exclusive_candidate(memory, owners):
                        self._update(seed, status="reserved", physical_gpu=gpu, gpu_uuid=uuid,
                                     memory_before_mib=memory, lease_path=str(path))
                        return gpu, lease, uuid
                except BaseException:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); raise
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close()
            self._update(seed, status="waiting_for_exclusive_gpu", last_wait=time.time())
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
            try: os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            proc.wait()

    def _execute(self, task):
        seed = int(task["seed"])
        output, log = output_path(task), log_path(task)
        lease = proc = None
        try:
            if seed == 1:
                self._update(seed, status="checking_sw0136_seed1_reuse")
                reused = run.wait_and_reuse_sw136_seed1(output)
                if reused is not None:
                    valid = validate_result(task)
                    self._update(seed, status="passed" if valid else "failed",
                                 reused_sw0136=True, artifact_valid=valid,
                                 result_sha256=(sha256_file(output / "predictions.npz") if valid else None),
                                 finished=time.time())
                    return "passed" if valid else "failed"
            gpu, lease, uuid = self._reserve(seed)
            triton = Path(f"/tmp/kevinswk_sw0137_source_s{seed}_{int(time.time()*1000)}")
            triton.mkdir(parents=True, exist_ok=False)
            argv = command(task)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", TRITON_CACHE_DIR=str(triton))
            self._update(seed, status="launching", physical_gpu=gpu, gpu_uuid=uuid,
                         argv=argv, output=str(output), log_path=str(log), triton_cache=str(triton),
                         environment={"CUDA_VISIBLE_DEVICES": str(gpu), "TRITON_CACHE_DIR": str(triton)},
                         launched=time.time())
            with log.open("x", encoding="utf-8") as stream:
                proc = subprocess.Popen(argv, cwd=str(ROOT), env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, start_new_session=True,
                                        pass_fds=(lease.fileno(),))
                self._update(seed, status="running", child_pid=proc.pid, started=time.time())
                while proc.poll() is None:
                    time.sleep(5)
                    if proc.poll() is not None:
                        break
                    _u, _m, owners = owner._nvidia_gpu_info(gpu)
                    foreign = owner.foreign_owner_pids(owners, owner._process_tree(proc.pid))
                    if foreign:
                        self._terminate_owned(proc)
                        self._update(seed, status="interrupted_foreign_gpu_owner",
                                     foreign_owner_pids=foreign, finished=time.time())
                        return "interrupted_foreign_gpu_owner"
            rc = int(proc.returncode)
            valid = rc == 0 and validate_result(task)
            status = "passed" if valid else "failed"
            self._update(seed, status=status, returncode=rc, artifact_valid=bool(valid),
                         result_sha256=(sha256_file(output / "predictions.npz")
                                        if (output / "predictions.npz").is_file() else None),
                         finished=time.time())
            return status
        except BaseException as exc:
            self._terminate_owned(proc)
            self._update(seed, status="failed", error=repr(exc), finished=time.time())
            return "failed"
        finally:
            if lease is not None:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close()

    def run(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
            futures = {pool.submit(self._execute, task): task for task in self.tasks}
            outcomes = {int(futures[future]["seed"]): future.result() for future in futures}
        if set(outcomes) == set(run.SEEDS) and all(value == "passed" for value in outcomes.values()):
            self.state["score"].update(status="running", started=time.time())
            self._save()
            try:
                report = run.score_sources()
                with self.lock:
                    self.state["score"].update(status="passed", artifact_valid=True,
                                               output=str(run.SUMMARY_DIR),
                                               metrics=report["source97_three_seed_mean"],
                                               finished=time.time())
                    _atomic_json(STATE, self.state)
                final = "source_qcc_evaluation_complete"
            except BaseException as exc:
                with self.lock:
                    self.state["score"].update(status="failed", error=repr(exc), finished=time.time())
                    _atomic_json(STATE, self.state)
                final = "source_qcc_evaluation_failed"
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
        print(json.dumps({"max_parallel": MAX_PARALLEL, "tasks": [
            {**task, "argv": command(task)} for task in task_plan()],
            "source_seed1_reuse_requires_livevalidated_sw0136_artifact": True,
            "score_after_all_three_valid_predictions": True,
            "ground_truth_used_for_prediction": False}, indent=2))
        return
    SourceQCCQueue().run()


if __name__ == "__main__":
    main()
