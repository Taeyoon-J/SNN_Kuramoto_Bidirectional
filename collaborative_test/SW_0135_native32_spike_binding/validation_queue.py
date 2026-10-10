"""Exclusive-GPU queue for the three native32 parity checks and B1/B4 diagnostic."""
from __future__ import annotations

import concurrent.futures
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

try:
    import fcntl
except ImportError:  # Windows permits dry-run/tests only; no GPU lease can be acquired.
    fcntl = None

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0123_adaptive_temporal_assignment import dispatcher as owner
from collaborative_test.SW_0135_native32_spike_binding import parity
from collaborative_test.SW_0135_native32_spike_binding import diagnostic
from collaborative_test.SW_0135_native32_spike_binding.foundation import sha256_file

ARCHIVE = HERE / "results_archive"
STATE = ARCHIVE / "validation_queue_state.json"
LOCK = ARCHIVE / "validation_queue.lock"
GPU_LEASES = Path("/tmp/kevinswk_sw0113_gpu_leases")
MEMORY_LIMIT_MIB = 512
MAX_PARALLEL = 3


def task_plan():
    return ([{"task_id": f"sw0135_parity_s{seed}_b4", "kind": "parity", "seed": seed}
             for seed in range(3)] +
            [{"task_id": "sw0135_batching_diagnostic_s0_b4", "kind": "diagnostic", "seed": 0}])


def result_path(task):
    if task["kind"] == "parity" and task["seed"] in (0, 1, 2):
        return ARCHIVE / f"native32_zero_adapter_parity_seed{task['seed']}_b4.json"
    if task["kind"] == "diagnostic" and task["seed"] == 0:
        return ARCHIVE / "batching_diagnostic_seed0_b4.json"
    raise ValueError("unregistered SW0135 validation task")


def command(task, device="cuda:0"):
    if task["kind"] == "parity":
        return [sys.executable, "-m", "collaborative_test.SW_0135_native32_spike_binding.parity",
                "--seed", str(task["seed"]), "--device", device,
                "--microbatch-size", "4", "--output", str(result_path(task))]
    if task["kind"] == "diagnostic":
        return [sys.executable, "-m", "collaborative_test.SW_0135_native32_spike_binding.diagnostic",
                "--seed", "0", "--device", device, "--output", str(result_path(task))]
    raise ValueError("unknown task kind")


def gpu_is_exclusive_candidate(memory_mib, compute_owners):
    return not compute_owners and int(memory_mib) <= MEMORY_LIMIT_MIB


def validate_record(report, task, fingerprint):
    try:
        expected_fp = (parity.implementation_fingerprint() if task["kind"] == "parity"
                       else diagnostic.implementation_fingerprint())
        if (fingerprint != expected_fp
                or report.get("experiment") != "SW0135_native32_spike_binding"
                or report.get("seed", report.get("source_seed")) != task["seed"]
                or report.get("ground_truth_used") is not False
                or report.get("masks_read") is not False
                or report.get("optimizer_updates") != 0
                or report.get("training_admission") is not False
                or report.get("implementation_fingerprint") != fingerprint):
            return False
        ids = report.get("image_ids", report.get("training_image_ids"))
        if task["kind"] == "parity":
            if (report.get("stage") != "native32_zero_adapter_parity"
                    or report.get("status") != "native32_zero_adapter_parity_passed"
                    or report.get("microbatch_size") != 4
                    or report.get("horizons") != [{"steps": 64, "settle": 32},
                                                   {"steps": 1024, "settle": 512}]
                    or len(ids) != 4 or len(set(ids)) != 4):
                return False
            rows = report.get("arms", {})
            if set(rows) != {"phase", "constant"}:
                return False
            expected = {(64, 32), (1024, 512)}
            for armrows in rows.values():
                if {(r.get("steps"), r.get("settle")) for r in armrows} != expected:
                    return False
                if len(armrows) != 2 or any(r.get("exact_parity") is not True for r in armrows):
                    return False
            return True
        if (report.get("stage") != "batching_diagnostic"
                or report.get("status") != "diagnostic_complete"
                or report.get("batch_size") != 4 or report.get("steps") != 1024
                or report.get("settle") != 512 or report.get("live_tail_steps") != 64
                or len(ids) != 4 or len(set(ids)) != 4):
            return False
        # A diagnostic is accepted for completion based on finite measurements, not equality.
        comparison = report["true_rollout_comparison"]
        scalars = [comparison["old_loss_b4"], comparison["old_loss_mean_b1"],
                   comparison["old_loss_abs_difference"],
                   report["registered_reduction"]["batch_vs_mean_loss_abs"],
                   comparison["gamma_b4_vs_independent_b1"]["max_abs"],
                   comparison["prepared_graph_b4_vs_b1"]["max_abs"],
                   comparison["q_b4_vs_concatenated_b1"]["max_abs"]]
        if not all(math.isfinite(float(v)) for v in scalars):
            return False
        return all(math.isfinite(float(v)) for v in
                   comparison["old_gradient_b4_vs_accumulated_b1"].values()
                   if isinstance(v, (float, int)))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def valid_result(task):
    path = result_path(task)
    if not path.is_file():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        fingerprint = (parity.implementation_fingerprint() if task["kind"] == "parity"
                       else diagnostic.implementation_fingerprint())
        return validate_record(report, task, fingerprint)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _atomic_json(path, payload):
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False); stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


class ValidationQueue:
    def __init__(self):
        if fcntl is None:
            raise RuntimeError("SW0135 GPU queue requires Linux flock")
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        self.queue_lock = LOCK.open("a+")
        try:
            fcntl.flock(self.queue_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.queue_lock.close(); raise RuntimeError("validation queue already active") from exc
        if STATE.exists():
            fcntl.flock(self.queue_lock.fileno(), fcntl.LOCK_UN); self.queue_lock.close()
            raise FileExistsError(f"preserve existing queue state: {STATE}")
        conflicts = [result_path(t) for t in task_plan()]
        conflicts += [ARCHIVE / f"{t['task_id']}.log" for t in task_plan()]
        existing = [str(p) for p in conflicts if p.exists()]
        if existing:
            fcntl.flock(self.queue_lock.fileno(), fcntl.LOCK_UN)
            self.queue_lock.close()
            raise FileExistsError(f"preserve existing task outputs: {existing}")
        self.lock = threading.RLock()
        self.tasks = task_plan()
        self.state = {"experiment": "SW0135_native32_spike_binding",
                      "stage": "parity_and_batching_diagnostics", "status": "running",
                      "supervisor_pid": os.getpid(), "max_parallel": MAX_PARALLEL,
                      "runner_fingerprints": {
                          "parity": parity.implementation_fingerprint(),
                          "diagnostic": diagnostic.implementation_fingerprint()},
                      "queue_sha256": sha256_file(HERE / "validation_queue.py"),
                      "owner_dispatcher_sha256": sha256_file(owner.__file__),
                      "training_admission": False, "tasks": {
                          t["task_id"]: {"task": t, "status": "queued"} for t in self.tasks}}
        self._save()

    def _update(self, task_id, **values):
        with self.lock:
            self.state["tasks"][task_id].update(values); _atomic_json(STATE, self.state)

    def _finish(self, status):
        with self.lock:
            self.state.update(status=status, finished=time.time()); _atomic_json(STATE, self.state)

    def _reserve(self, task_id):
        while True:
            for gpu in range(4):
                lease_path = GPU_LEASES / f"gpu{gpu}.lock"
                lease_path.parent.mkdir(parents=True, exist_ok=True)
                handle = lease_path.open("a+")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close(); continue
                try:
                    gpu_uuid, memory, owners = owner._nvidia_gpu_info(gpu)
                    if gpu_is_exclusive_candidate(memory, owners):
                        self._update(task_id, status="reserved", physical_gpu=gpu,
                                     gpu_uuid=gpu_uuid, memory_before_mib=memory,
                                     lease_path=str(lease_path))
                        return gpu, handle, gpu_uuid
                except BaseException:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN); handle.close(); raise
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN); handle.close()
            self._update(task_id, status="waiting_for_exclusive_gpu", last_wait=time.time())
            time.sleep(5)

    @staticmethod
    def _terminate(proc):
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try: os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            proc.wait()
        except ProcessLookupError:
            pass

    def _execute(self, task):
        task_id = task["task_id"]
        out, log = result_path(task), ARCHIVE / f"{task_id}.log"
        lease = proc = None
        try:
            gpu, lease, gpu_uuid = self._reserve(task_id)
            triton = Path(f"/tmp/kevinswk_sw0135_{task_id}_{int(time.time()*1000)}")
            triton.mkdir(parents=True, exist_ok=False)
            argv = command(task, "cuda:0")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", TRITON_CACHE_DIR=str(triton))
            self._update(task_id, status="launching", physical_gpu=gpu,
                         gpu_uuid=gpu_uuid, argv=argv, log_path=str(log),
                         triton_cache=str(triton), launched=time.time())
            with log.open("x", encoding="utf-8") as stream:
                proc = subprocess.Popen(argv, cwd=str(ROOT), env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, start_new_session=True,
                                        pass_fds=(lease.fileno(),))
                self._update(task_id, status="running", child_pid=proc.pid, started=time.time())
                while proc.poll() is None:
                    time.sleep(5)
                    if proc.poll() is not None:
                        break
                    _u, _m, owners = owner._nvidia_gpu_info(gpu)
                    foreign = owner.foreign_owner_pids(owners, owner._process_tree(proc.pid))
                    if foreign:
                        self._terminate(proc)
                        self._update(task_id, status="interrupted_foreign_gpu_owner",
                                     foreign_owner_pids=foreign, finished=time.time())
                        return "interrupted_foreign_gpu_owner"
            rc = int(proc.returncode)
            passed = rc == 0 and valid_result(task)
            status = "passed" if passed else "failed"
            self._update(task_id, status=status, returncode=rc,
                         artifact_valid=bool(passed), result_sha256=(sha256_file(out) if out.is_file() else None),
                         finished=time.time())
            return status
        except BaseException as exc:
            self._terminate(proc)
            self._update(task_id, status="failed", error=repr(exc), finished=time.time())
            return "failed"
        finally:
            if lease is not None:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close()

    def run(self):
        # All three parity checks and the diagnostic are independent read-only tasks.
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
            futures = {pool.submit(self._execute, t): t for t in self.tasks}
            outcomes = {futures[f]["task_id"]: f.result() for f in futures}
        self._finish("all_validation_tasks_complete" if all(v == "passed" for v in outcomes.values())
                     else "validation_tasks_incomplete_or_failed")

    def _save(self):
        with self.lock:
            _atomic_json(STATE, self.state)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"max_parallel": MAX_PARALLEL, "training_admission": False,
                          "tasks": [{**t, "argv": command(t)} for t in task_plan()]}, indent=2))
        return
    ValidationQueue().run()


if __name__ == "__main__":
    main()
