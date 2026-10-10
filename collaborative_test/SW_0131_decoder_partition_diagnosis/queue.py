"""Exclusive, no-retry launcher for the three bounded SW0131 TRAIN diagnostics."""
from __future__ import annotations

import concurrent.futures
import json
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time

try:
    import fcntl
except ImportError:  # only Linux executes GPU jobs; local tests exercise pure contracts
    fcntl = None

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]
# The filename intentionally documents the queue stage, but must not shadow
# Python's standard-library `queue` when Torch imports its data utilities.
sys.path[:] = [entry for entry in sys.path
               if not entry or pathlib.Path(entry).resolve() != HERE]
from collaborative_test.SW_0131_decoder_partition_diagnosis import run
from collaborative_test.SW_0123_adaptive_temporal_assignment import dispatcher as owner
sys.path.insert(0, str(HERE))

ARCHIVE = HERE / "results_archive"
STATE = ARCHIVE / "decoder_partition_queue_20261009.json"
QUEUE_LOCK = ARCHIVE / "decoder_partition_queue.lock"
GPU_LEASE_DIR = pathlib.Path("/tmp/kevinswk_sw0113_gpu_leases")
GPU_MEMORY_LIMIT_MIB = 512
MAX_PARALLEL = 3


def task_plan():
    return [{"task_id": f"sw0131_decoder_diagnostic_s{seed}", "seed": seed,
             "stage": "read_only_diagnostic"} for seed in (0, 1, 2)]


def artifact_path(task):
    if task.get("stage") != "read_only_diagnostic" or task.get("seed") not in (0, 1, 2):
        raise ValueError("SW0131 queue accepts only its three registered diagnostic tasks")
    return ARCHIVE / f"decoder_partition_seed{int(task['seed'])}_20261009.json"


def command(task, device="cuda:0"):
    return [sys.executable, str(HERE / "run.py"), "--seed", str(int(task["seed"])),
            "--device", device, "--output", str(artifact_path(task))]


def gpu_is_exclusive_candidate(memory_mib, compute_owners):
    return not compute_owners and int(memory_mib) <= GPU_MEMORY_LIMIT_MIB


def validate_report(report, seed, *, expected_ids, source_sha, fingerprint, assets,
                    warm_sha):
    import math
    if (report.get("status") != "complete"
            or report.get("experiment") != "SW0131_decoder_partition_diagnosis"
            or report.get("seed") != seed or report.get("images") != 64
            or report.get("image_ids") != expected_ids
            or report.get("source_core_sha256") != source_sha
            or report.get("implementation_fingerprint") != fingerprint
            or report.get("asset_hashes") != assets
            or report.get("warm_decoder_sha256") != warm_sha
            or report.get("optimizer_updates") != 0
            or report.get("ground_truth_used") is not False
            or report.get("source_checkpoint_modified") is not False
            or report.get("decoder_artifact_modified") is not False
            or report.get("time_steps") != 64 or report.get("settle") != 32
            or report.get("batch_size") != 16
            or report.get("decoder_states") != ["initial_random", "warmed32"]
            or report.get("conditions") != ["native", "row_shuffled", "image_mean_content"]):
        return False
    rows = report.get("per_image")
    if not isinstance(rows, list) or len(rows) != 64:
        return False
    if [row.get("image_id") for row in rows] != expected_ids:
        return False
    for row in rows:
        if not isinstance(row.get("K"), int) or row["K"] < 1:
            return False
        sizes = row.get("native_group_sizes")
        if (not isinstance(sizes, list) or len(sizes) != row["K"]
                or any(not isinstance(value, int) or value <= 0 for value in sizes)
                or sum(sizes) != 256):
            return False
        for state in ("initial_random", "warmed32"):
            losses = row.get("mse", {}).get(state)
            if not isinstance(losses, dict) or set(losses) != {
                    "native", "row_shuffled", "image_mean_content"}:
                return False
            if any(not math.isfinite(float(value)) for value in losses.values()):
                return False
    deltas = report.get("paired64image_deltas")
    expected_delta_names = {
        f"{state}_{condition}_minus_native"
        for state in ("initial_random", "warmed32")
        for condition in ("row_shuffled", "image_mean_content")}
    if not isinstance(deltas, dict) or set(deltas) != expected_delta_names:
        return False
    for key, row in deltas.items():
        values = row.get("per_image")
        stem = key.removesuffix("_minus_native")
        state = next(name for name in ("initial_random", "warmed32")
                     if stem.startswith(name + "_"))
        comparison = stem[len(state) + 1:]
        if (not isinstance(values, list) or len(values) != 64
                or not all(math.isfinite(float(value)) for value in values)
                or not math.isfinite(float(row.get("mean", float("nan"))))):
            return False
        expected_values = [
            image["mse"][state][comparison] - image["mse"][state]["native"]
            for image in rows]
        if any(not math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12)
               for actual, expected in zip(values, expected_values)):
            return False
        if abs(sum(map(float, values)) / 64. - float(row["mean"])) > 1e-10:
            return False
    gradients = report.get("batch_gradient_diagnostics")
    if not isinstance(gradients, list) or len(gradients) != 8:
        return False
    gradient_pairs = set()
    for row in gradients:
        if (row.get("condition") != "native" or row.get("decoder_state") not in
                ("initial_random", "warmed32") or row.get("batch_index") not in range(4)):
            return False
        gradient_pairs.add((row["decoder_state"], row["batch_index"]))
        if not math.isfinite(float(row.get("q_gradient_norm", float("nan")))):
            return False
        norms = row.get("rgb_gradient_norms_by_family")
        if not isinstance(norms, dict) or not norms:
            return False
        if any(not math.isfinite(float(value)) or float(value) < 0 for value in norms.values()):
            return False
    if gradient_pairs != {(state, batch) for state in ("initial_random", "warmed32")
                          for batch in range(4)}:
        return False
    return True


def valid_result(task, assets=None):
    path = artifact_path(task)
    if not path.is_file():
        return False
    try:
        seed = int(task["seed"])
        _checkpoint, _manifest_path, _manifest, _pool, ids, source_sha = run.sw130.source_contract(seed)
        if assets is None:
            assets = run.sw130.validate_rgb_assets()
        _report_path, _preflight, _warm_path, warm_sha, _warm, _audit_path, _audit_sha = \
            run._warmup_inputs(seed)
        report = json.loads(path.read_text(encoding="utf-8"))
        return validate_report(report, seed, expected_ids=ids[:64], source_sha=source_sha,
                               fingerprint=run.implementation_fingerprint(), assets=assets,
                               warm_sha=warm_sha)
    except (OSError, ValueError, KeyError, TypeError, AssertionError, RuntimeError):
        return False


def _atomic_json(path, value):
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


class DiagnosticQueue:
    def __init__(self):
        if fcntl is None:
            raise RuntimeError("SW0131 diagnostic queue requires Linux flock")
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        self.lock_file = QUEUE_LOCK.open("a+")
        try:
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock_file.close()
            raise RuntimeError("another SW0131 diagnostic queue owns its lock") from exc
        self.tasks = task_plan()
        collisions = [path for task in self.tasks for path in
                      (artifact_path(task), ARCHIVE / f"{task['task_id']}.log")]
        if STATE.exists() or any(path.exists() for path in collisions):
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
            self.lock_file.close()
            raise FileExistsError("preserving existing SW0131 queue state, output, or log")
        self.mutex = threading.RLock()
        self.assets = run.sw130.validate_rgb_assets()
        self.state = {"experiment": "SW0131_decoder_partition_diagnosis",
                      "status": "running", "stage": "read_only_diagnostic",
                      "supervisor_pid": os.getpid(), "task_count": len(self.tasks),
                      "max_parallel": MAX_PARALLEL, "started": time.time(),
                      "implementation_fingerprint": run.implementation_fingerprint(),
                      "queue_sha256": run.sw130.sha(HERE / "queue.py"),
                      "owner_dispatcher_path": str(pathlib.Path(owner.__file__).resolve()),
                      "owner_dispatcher_sha256": run.sw130.sha(owner.__file__),
                      "asset_hashes": self.assets,
                      "no_training_or_optimizer_updates": True,
                      "tasks": {task["task_id"]: {"task": task, "status": "queued"}
                                for task in self.tasks}}
        self._save()

    def _save(self):
        with self.mutex:
            _atomic_json(STATE, self.state)

    def _update(self, task, **fields):
        with self.mutex:
            self.state["tasks"][task["task_id"]].update(fields)
            _atomic_json(STATE, self.state)

    def _reserve(self, task):
        while True:
            for gpu in range(4):
                lease_path = GPU_LEASE_DIR / f"gpu{gpu}.lock"
                lease_path.parent.mkdir(parents=True, exist_ok=True)
                lease = lease_path.open("a+")
                try:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lease.close(); continue
                try:
                    uuid, memory, owners = owner._nvidia_gpu_info(gpu)
                    if gpu_is_exclusive_candidate(memory, owners):
                        self._update(task, status="reserved", gpu=gpu, gpu_uuid=uuid,
                                     memory_before_mib=memory, lease_path=str(lease_path))
                        return gpu, uuid, lease
                except BaseException:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); raise
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close()
            self._update(task, status="waiting_for_exclusive_gpu", last_wait=time.time())
            time.sleep(5)

    @staticmethod
    def _terminate_owned(proc):
        if proc.poll() is not None:
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
        output = artifact_path(task)
        if output.exists():
            status = "reused_verified" if valid_result(task, self.assets) else "failed_existing_artifact"
            self._update(task, status=status, artifact=str(output))
            return status
        gpu, gpu_uuid, lease = self._reserve(task)
        task_id = task["task_id"]
        log_path = ARCHIVE / f"{task_id}.log"
        proc = None
        try:
            if log_path.exists():
                raise FileExistsError(f"preserving existing diagnostic log: {log_path}")
            triton_path = pathlib.Path(f"/tmp/kevinswk_sw0131_triton_{task_id}_{int(time.time()*1000)}")
            triton_path.mkdir(parents=True, exist_ok=False)
            argv = command(task)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", TRITON_CACHE_DIR=str(triton_path))
            self._update(task, status="launching", gpu=gpu, gpu_uuid=gpu_uuid,
                         argv=argv, log_path=str(log_path), triton_cache=str(triton_path))
            with log_path.open("x", encoding="utf-8") as log:
                proc = subprocess.Popen(argv, cwd=str(ROOT), env=env, stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True,
                                        pass_fds=(lease.fileno(),))
                self._update(task, status="running", child_pid=proc.pid, started=time.time())
                while proc.poll() is None:
                    time.sleep(5)
                    if proc.poll() is not None:
                        break
                    _uuid, _memory, owners = owner._nvidia_gpu_info(gpu)
                    descendants = owner._process_tree(proc.pid)
                    foreign = owner.foreign_owner_pids(owners, descendants)
                    if foreign:
                        self._terminate_owned(proc)
                        self._update(task, status="interrupted_foreign_gpu_owner",
                                     foreign_owner_pids=foreign, finished=time.time())
                        return "interrupted_foreign_gpu_owner"
            code = int(proc.returncode)
            ok = code == 0 and valid_result(task, self.assets)
            status = "passed" if ok else "failed"
            self._update(task, status=status, returncode=code, artifact_valid=bool(ok),
                         artifact=str(output), artifact_sha256=run.sw130.sha(output)
                         if output.is_file() else None, finished=time.time())
            return status
        except BaseException as exc:
            if proc is not None:
                self._terminate_owned(proc)
            self._update(task, status="failed", error=repr(exc), finished=time.time())
            return "failed"
        finally:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
            lease.close()

    def run(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
            futures = {pool.submit(self._execute, task): task for task in self.tasks}
            results = {futures[future]["task_id"]: future.result() for future in futures}
        success = all(value in ("passed", "reused_verified") for value in results.values())
        with self.mutex:
            self.state.update(status="complete" if success else "diagnostic_failed",
                              task_results=results, finished=time.time())
            _atomic_json(STATE, self.state)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"experiment": "SW0131_decoder_partition_diagnosis",
                          "max_parallel": MAX_PARALLEL,
                          "tasks": [{**task, "argv": command(task)} for task in task_plan()]},
                         indent=2, allow_nan=False))
        return
    DiagnosticQueue().run()


if __name__ == "__main__":
    main()
