"""Exclusive queue for frozen SW0138 seed predictions and GT-last scoring."""
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
OUTPUT_ROOT = HERE / "results_archive" / "event_predictions"
STATE = ARCHIVE / "event_information_queue_state.json"
LOCK = ARCHIVE / "event_information_queue.lock"
GPU_LEASES = Path("/tmp/kevinswk_sw0113_gpu_leases")
MEMORY_LIMIT_MIB = 512
MAX_PARALLEL = 2
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]
from collaborative_test.SW_0123_adaptive_temporal_assignment import dispatcher as owner
from collaborative_test.SW_0135_native32_spike_binding.foundation import sha256_file
from collaborative_test.SW_0137_native32_source_qcc import run as sw137
from collaborative_test.SW_0138_native32_event_information import run


def task_plan():
    return [{"task_id": f"sw0138_event_information_seed{seed}", "seed": seed}
            for seed in (0, 2, 1)]


def output_path(task):
    return run.prediction_dir(int(task.get("seed", -1)), OUTPUT_ROOT)


def log_path(task):
    return ARCHIVE / f"{task['task_id']}.log"


def command(task, device="cuda:0"):
    seed = int(task.get("seed", -1))
    if seed not in run.SEEDS:
        raise ValueError("unregistered SW0138 seed")
    return [sys.executable, "-m", "collaborative_test.SW_0138_native32_event_information.run",
            "--stage", "predict", "--seed", str(seed), "--device", device,
            "--output-root", str(OUTPUT_ROOT)]


def gpu_is_exclusive_candidate(memory_mib, compute_owners):
    return not compute_owners and int(memory_mib) <= MEMORY_LIMIT_MIB


def validate_result(task):
    try:
        run.validate_prediction(output_path(task), seed=int(task["seed"]))
        return True
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def validate_sw137_dependencies(root=sw137.OUTPUT_ROOT):
    rows = {seed: sw137.validate_prediction(Path(root) / f"seed{seed}", seed=seed)
            for seed in run.SEEDS}
    if any(not (rows[s]["ids"] == rows[0]["ids"]).all() for s in (1, 2)):
        raise ValueError("SW0137 dependencies do not share the fixed ordered320 image IDs")
    return {seed: {"prediction_sha256": rows[seed]["prediction_sha256"],
                   "protocol_sha256": rows[seed]["protocol_sha256"]}
            for seed in run.SEEDS}


def ensure_absent(paths):
    existing = [str(Path(path)) for path in paths if Path(path).exists()]
    if existing:
        raise FileExistsError("preserve existing SW0138 attempt paths: " + ", ".join(existing))


def _atomic_json(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


class EventInformationQueue:
    def __init__(self, *, sw137_root=sw137.OUTPUT_ROOT):
        if fcntl is None:
            raise RuntimeError("SW0138 queue requires Linux flock")
        self.sw137_root = Path(sw137_root)
        # Dependency check precedes any SW0138 attempt output or GPU reservation.
        self.dependencies = validate_sw137_dependencies(self.sw137_root)
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        self.queue_lock = LOCK.open("a+")
        try:
            fcntl.flock(self.queue_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.queue_lock.close()
            raise RuntimeError("another SW0138 queue owns the lock") from exc
        paths = [STATE, ARCHIVE / "evaluation"]
        for task in task_plan():
            paths.extend((output_path(task), log_path(task)))
        try:
            ensure_absent(paths)
        except BaseException:
            self.queue_lock.close()
            raise
        self.lock = threading.RLock()
        self.tasks = task_plan()
        self.state = {
            "experiment": "SW0138_native32_event_information", "stage": "frozen_event_readout",
            "status": "running", "supervisor_pid": os.getpid(), "started": time.time(),
            "max_parallel": MAX_PARALLEL,
            "implementation_fingerprint": run.implementation_fingerprint(),
            "queue_sha256": sha256_file(HERE / "event_queue.py"),
            "owner_dispatcher_sha256": sha256_file(Path(owner.__file__)),
            "sw0137_dependencies": self.dependencies,
            "ground_truth_used_for_prediction": False, "optimizer_updates": 0,
            "tasks": {str(t["seed"]): {"task": t, "status": "queued"} for t in self.tasks},
            "score": {"status": "blocked_until_all_six_predictions_validate"},
        }
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
                    gpu_uuid, memory, owners = owner._nvidia_gpu_info(gpu)
                    if gpu_is_exclusive_candidate(memory, owners):
                        self._update(seed, status="reserved", physical_gpu=gpu,
                                     gpu_uuid=gpu_uuid, memory_before_mib=memory,
                                     lease_path=str(path))
                        return gpu, lease, gpu_uuid
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
            gpu, lease, uuid = self._reserve(seed)
            triton = Path(f"/tmp/kevinswk_sw0138_event_s{seed}_{int(time.time()*1000)}")
            triton.mkdir(parents=True, exist_ok=False)
            argv = command(task)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", TRITON_CACHE_DIR=str(triton))
            self._update(seed, status="launching", physical_gpu=gpu, gpu_uuid=uuid,
                         argv=argv, output=str(output), log_path=str(log),
                         triton_cache=str(triton),
                         environment={"CUDA_VISIBLE_DEVICES": str(gpu),
                                      "TRITON_CACHE_DIR": str(triton)}, launched=time.time())
            with log.open("x", encoding="utf-8") as stream:
                proc = subprocess.Popen(argv, cwd=str(ROOT), env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, start_new_session=True,
                                        pass_fds=(lease.fileno(),))
                self._update(seed, status="running", child_pid=proc.pid, started=time.time())
                while proc.poll() is None:
                    time.sleep(5)
                    if proc.poll() is not None:
                        break
                    _uuid, _memory, owners = owner._nvidia_gpu_info(gpu)
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
            outcomes = {int(futures[f]["seed"]): f.result() for f in futures}
        if set(outcomes) == set(run.SEEDS) and all(v == "passed" for v in outcomes.values()):
            self.state["score"].update(status="running", started=time.time()); self._save()
            try:
                report = run.score(sw137_root=self.sw137_root)
                with self.lock:
                    self.state["score"].update(status="passed", artifact_valid=True,
                        output=str(run.SUMMARY_DIR), gate_minus_actual=report["paired_bootstrap_gate_minus_actual"],
                        finished=time.time())
                    _atomic_json(STATE, self.state)
                terminal = "event_information_evaluation_complete"
            except BaseException as exc:
                with self.lock:
                    self.state["score"].update(status="failed", error=repr(exc), finished=time.time())
                    _atomic_json(STATE, self.state)
                terminal = "event_information_scoring_failed"
        else:
            terminal = "prediction_tasks_failed_or_interrupted"
        with self.lock:
            self.state.update(status=terminal, task_outcomes=outcomes, finished=time.time())
            _atomic_json(STATE, self.state)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"max_parallel": MAX_PARALLEL,
            "tasks": [{**task, "argv": command(task)} for task in task_plan()],
            "requires_all_three_validated_sw0137_predictions": True,
            "gt_scoring_after_six_prediction_arrays": True,
            "ground_truth_used_for_prediction": False}, indent=2))
        return
    EventInformationQueue().run()


if __name__ == "__main__":
    main()
