"""Exclusive-GPU, no-retry launcher for the three SW0130 preflights only."""
from __future__ import annotations

import concurrent.futures
import json
import os
import pathlib
import signal
import statistics
import subprocess
import sys
import threading
import time

try:
    import fcntl
except ImportError:  # Windows supports dry-run and contract tests only.
    fcntl = None

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0130_phase_state_integration import run
from collaborative_test.SW_0123_adaptive_temporal_assignment import dispatcher as owner

ARCHIVE = HERE / "results_archive"
STATE = ARCHIVE / "preflight_queue_state.json"
QUEUE_LOCK = ARCHIVE / "preflight_queue.lock"
GPU_LEASE_DIR = pathlib.Path("/tmp/kevinswk_sw0113_gpu_leases")
GPU_MEMORY_LIMIT_MIB = 512
MAX_PARALLEL = 2
REQUIRED_FAMILIES = ("encoder", "graph", "oscillator_drive", "kuramoto",
                     "dendrite", "membrane", "a_d", "a_m", "b")
REQUIRED_ARMS = ("phase", "constant")


def task_plan():
    return [
        {"task_id": "sw0130_preflight_s0", "stage": "preflight", "seed": 0,
         "depends_on": [], "arm": "phase"},
        {"task_id": "sw0130_preflight_s1", "stage": "preflight", "seed": 1,
         "depends_on": ["sw0130_preflight_s0"], "arm": "phase"},
        {"task_id": "sw0130_preflight_s2", "stage": "preflight", "seed": 2,
         "depends_on": ["sw0130_preflight_s0"], "arm": "phase"},
    ]


def artifact_path(task):
    if task.get("stage") != "preflight" or task.get("seed") not in (0, 1, 2):
        raise ValueError("queue accepts only the three registered preflight tasks")
    return ARCHIVE / f"preflight_seed{int(task['seed'])}.json"


def command(task, device="cuda:0"):
    return [sys.executable, str(HERE / "run.py"), "--stage", "preflight",
            "--seed", str(int(task["seed"])), "--device", device,
            "--output", str(artifact_path(task))]


def gpu_is_exclusive_candidate(memory_mib, compute_owners):
    return not compute_owners and int(memory_mib) <= GPU_MEMORY_LIMIT_MIB


def _atomic_json(path, value):
    path = pathlib.Path(path)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def validate_preflight_record(report, seed, *, fingerprint, source_sha, source_manifest_sha,
                              training_ids, pool_sha, asset_hashes, warmup_path,
                              seed0_report=None, seed0_report_sha=None, warmup_sha_fn=None):
    """Validate the scientific and artifact evidence emitted by run.preflight."""
    import hashlib
    import math

    if (report.get("status") != "passed"
            or report.get("experiment") != "SW0130_phase_state_integration"
            or report.get("seed") != seed or report.get("arm") != "phase"
            or report.get("implementation_fingerprint") != fingerprint
            or report.get("source_core_sha256") != source_sha
            or report.get("source_manifest_sha256") != source_manifest_sha
            or report.get("source_updates") != 256
            or report.get("matched_training_ids") != training_ids
            or report.get("training_ids_sha256") != hashlib.sha256(
                __import__("numpy").asarray(training_ids, dtype="<i8").tobytes()).hexdigest()
            or report.get("pool_indices_sha256") != pool_sha
            or report.get("shuffle_seed") != 117 + seed
            or report.get("batch_size") != 16 or report.get("updates") != 256
            or report.get("train_time_steps") != 64 or report.get("settle_steps") != 32
            or report.get("asset_hashes") != asset_hashes
            or report.get("ground_truth_used") is not False
            or report.get("original_core_checkpoint_modified") is not False):
        return False
    try:
        if not math.isfinite(float(report["registered_gamma_cache_max_abs_diff_first_batch"])):
            return False
        if float(report["registered_gamma_cache_max_abs_diff_first_batch"]) > 2e-5:
            return False
        if report.get("row_scramble_count") != 64 or report.get("row_scramble_positive_count", 0) < 1:
            return False
        if not math.isfinite(float(report["row_scramble_mean_excess"])) or report["row_scramble_mean_excess"] <= 0:
            return False
        warm_path = pathlib.Path(report["decoder_warmup_artifact"])
        if warm_path.resolve() != pathlib.Path(warmup_path).resolve() or not warm_path.is_file():
            return False
        if warmup_sha_fn is None:
            warmup_sha_fn = run.sha
        if warmup_sha_fn(warm_path) != report.get("decoder_warmup_artifact_sha256"):
            return False
        parity = report["native_zero_initial_parity"]
        expected = {(arm, steps, settle) for arm in REQUIRED_ARMS
                    for steps, settle in ((64, 32), (1024, 512))}
        actual = {(row.get("arm"), row.get("time_steps"), row.get("settle"))
                  for row in parity["checks"]
                  if row.get("theta_component_membrane_spikes_q_h_primary_oldloss_exact") is True}
        if parity.get("source_core_sha256") != source_sha or actual != expected:
            return False
        if report.get("disposable_update_finite") is not True:
            return False
        disposable = report["disposable_updates"]
        if {row.get("arm") for row in disposable} != set(REQUIRED_ARMS):
            return False
        for row in disposable:
            norms = row["rgb_gradient_norms_by_family"]
            if row.get("optimizer_updates") != 1 or row.get("throwaway_only") is not True:
                return False
            if any(not math.isfinite(float(norms.get(family, 0))) or float(norms[family]) <= 0
                   for family in REQUIRED_FAMILIES):
                return False
            if (row.get("joint_gradient_norm", 0) <= 0
                    or row.get("decoder_gradient_norm", 0) <= 0
                    or row.get("changed_native_core_parameter_count", 0) <= 0
                    or row.get("changed_encoder_parameter_count", 0) <= 0
                    or row.get("changed_decoder_parameter_count", 0) <= 0
                    or "b" not in row.get("changed_integration_parameters", [])):
                return False
        value = float(report["lambda"])
        if not math.isfinite(value) or value <= 0 or report.get("lambda_source_seed") != 0:
            return False
        if seed == 0:
            if report.get("seed0_lambda_record_sha256") is not None:
                return False
            calibration = report.get("lambda_seed0_calibration")
            if not isinstance(calibration, list) or len(calibration) != 4:
                return False
            ratios = []
            for batch_index, row in enumerate(calibration):
                old_norm = float(row["old_joint_gradient_norm"])
                rgb_norm = float(row["rgb_joint_gradient_norm"])
                ratio = float(row["ratio"])
                if (row.get("batch") != batch_index or not math.isfinite(old_norm)
                        or not math.isfinite(rgb_norm) or not math.isfinite(ratio)
                        or old_norm <= 0 or rgb_norm <= 0
                        or not math.isclose(ratio, .25 * old_norm / rgb_norm,
                                            rel_tol=1e-12, abs_tol=0.0)):
                    return False
                ratios.append(ratio)
            if statistics.median(ratios) != value:
                return False
        else:
            if not isinstance(seed0_report, dict):
                return False
            if (seed0_report.get("status") != "passed" or seed0_report.get("seed") != 0
                    or seed0_report.get("implementation_fingerprint") != fingerprint
                    or report.get("seed0_lambda_record_sha256") != seed0_report_sha
                    or report.get("lambda_seed0_calibration") !=
                        "reused immutable seed0 preflight calibration"
                    or value != float(seed0_report["lambda"])):
                return False
    except (KeyError, TypeError, ValueError, OSError, OverflowError):
        return False
    return True


def valid_result(task, assets=None):
    path = artifact_path(task)
    seed = int(task["seed"])
    if not path.is_file():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        checkpoint, manifest_path, manifest, pool, ids, source_sha = run.source_contract(seed)
        if assets is None:
            assets = run.validate_rgb_assets()
        seed0_report = None
        seed0_report_sha = None
        if seed:
            seed0_path = artifact_path({"stage": "preflight", "seed": 0})
            if (not seed0_path.is_file()
                    or not valid_result({"stage": "preflight", "seed": 0}, assets)):
                return False
            seed0_report = json.loads(seed0_path.read_text(encoding="utf-8"))
            seed0_report_sha = run.sha(seed0_path)
        expected_pool_sha = __import__("hashlib").sha256(
            __import__("numpy").asarray(pool, dtype="<i8").tobytes()).hexdigest()
        return validate_preflight_record(
            report, seed, fingerprint=run.implementation_fingerprint(), source_sha=source_sha,
            source_manifest_sha=run.sha(manifest_path), training_ids=ids,
            pool_sha=expected_pool_sha, asset_hashes=assets,
            warmup_path=ARCHIVE / f"preflight_decoder_seed{seed}.pt",
            seed0_report=seed0_report, seed0_report_sha=seed0_report_sha)
    except (OSError, ValueError, KeyError, TypeError, AssertionError):
        return False


class PreflightQueue:
    def __init__(self):
        if fcntl is None:
            raise RuntimeError("SW0130 GPU preflight queue requires Linux flock")
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        self.queue_lock = QUEUE_LOCK.open("a+")
        try:
            fcntl.flock(self.queue_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.queue_lock.close()
            raise RuntimeError("another SW0130 preflight queue owns its lock") from exc
        if STATE.exists():
            fcntl.flock(self.queue_lock.fileno(), fcntl.LOCK_UN)
            self.queue_lock.close()
            raise FileExistsError(f"preserve existing queue state: {STATE}")
        self.lock = threading.RLock()
        self.tasks = task_plan()
        self.assets = run.validate_rgb_assets()
        self.state = {
            "experiment": "SW0130_phase_state_integration",
            "stage": "preflight_only", "status": "running",
            "supervisor_pid": os.getpid(), "started": time.time(),
            "runner_fingerprint": run.implementation_fingerprint(),
            "task_count": len(self.tasks), "no_training_stage": True,
            "tasks": {task["task_id"]: {"task": task, "status": "queued"}
                      for task in self.tasks},
        }
        self._save()

    def _save(self):
        with self.lock:
            _atomic_json(STATE, self.state)

    def _update(self, task_id, **fields):
        with self.lock:
            self.state["tasks"][task_id].update(fields)
            _atomic_json(STATE, self.state)

    def _reserve_gpu(self, task_id):
        while True:
            for gpu in range(4):
                lease_path = GPU_LEASE_DIR / f"gpu{gpu}.lock"
                lease_path.parent.mkdir(parents=True, exist_ok=True)
                lease = lease_path.open("a+")
                try:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lease.close()
                    continue
                try:
                    gpu_uuid, memory_mib, owners = owner._nvidia_gpu_info(gpu)
                    if gpu_is_exclusive_candidate(memory_mib, owners):
                        self._update(task_id, status="reserved", gpu=gpu, gpu_uuid=gpu_uuid,
                                     memory_before_mib=memory_mib, lease_path=str(lease_path),
                                     reservation_pid=os.getpid(), reserved=time.time())
                        return gpu, lease, gpu_uuid
                except BaseException:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                    lease.close()
                    raise
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                lease.close()
            self._update(task_id, status="waiting_for_exclusive_gpu", last_wait=time.time())
            time.sleep(5)

    @staticmethod
    def _terminate_owned_child(proc):
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
        task_id = task["task_id"]
        output = artifact_path(task)
        if output.exists():
            if valid_result(task, self.assets):
                self._update(task_id, status="reused_verified", artifact=str(output),
                             verified_at=time.time())
                return "reused_verified"
            self._update(task_id, status="failed_existing_artifact",
                         existing_artifact=str(output), finished=time.time())
            return "failed_existing_artifact"
        warmup = ARCHIVE / f"preflight_decoder_seed{task['seed']}.pt"
        if warmup.exists():
            self._update(task_id, status="failed_existing_warmup",
                         existing_warmup=str(warmup), finished=time.time())
            return "failed_existing_warmup"
        gpu, lease, gpu_uuid = self._reserve_gpu(task_id)
        log_path = ARCHIVE / f"{task_id}.log"
        proc = None
        try:
            if log_path.exists():
                raise FileExistsError(f"preserve existing task log: {log_path}")
            triton_dir = pathlib.Path(
                f"/tmp/kevinswk_sw0130_triton_{task_id}_{int(time.time() * 1000)}")
            triton_dir.mkdir(parents=True, exist_ok=False)
            argv = command(task, device="cuda:0")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", TRITON_CACHE_DIR=str(triton_dir))
            self._update(task_id, status="launching", gpu=gpu, gpu_uuid=gpu_uuid,
                         argv=argv, log_path=str(log_path), triton_cache=str(triton_dir),
                         launched=time.time())
            with log_path.open("x", encoding="utf-8") as log:
                proc = subprocess.Popen(argv, cwd=str(ROOT), env=env, stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True,
                                        pass_fds=(lease.fileno(),))
                self._update(task_id, status="running", child_pid=proc.pid, started=time.time())
                while proc.poll() is None:
                    time.sleep(5)
                    if proc.poll() is not None:
                        break
                    _uuid, _memory, owners = owner._nvidia_gpu_info(gpu)
                    descendants = owner._process_tree(proc.pid)
                    foreign = owner.foreign_owner_pids(owners, descendants)
                    if foreign:
                        self._terminate_owned_child(proc)
                        self._update(task_id, status="interrupted_foreign_gpu_owner",
                                     foreign_owner_pids=foreign, finished=time.time())
                        return "interrupted_foreign_gpu_owner"
            rc = int(proc.returncode)
            is_valid = rc == 0 and valid_result(task, self.assets)
            status = "passed" if is_valid else "failed"
            self._update(task_id, status=status, returncode=rc,
                         artifact_valid=bool(is_valid), finished=time.time())
            return status
        except BaseException as exc:
            if proc is not None:
                self._terminate_owned_child(proc)
            self._update(task_id, status="failed", error=repr(exc), finished=time.time())
            return "failed"
        finally:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
            lease.close()

    def _run_tasks(self, rows):
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
            futures = {executor.submit(self._execute, task): task for task in rows}
            return {futures[future]["task_id"]: future.result() for future in futures}

    def run(self):
        seed0 = self.tasks[0]
        result0 = self._execute(seed0)
        if result0 not in ("passed", "reused_verified"):
            for task in self.tasks[1:]:
                self._update(task["task_id"], status="blocked_seed0_preflight_failure")
            self._finish("seed0_preflight_failed")
            return
        result_later = self._run_tasks(self.tasks[1:])
        passed = all(value in ("passed", "reused_verified") for value in result_later.values())
        self._finish("all_preflights_passed" if passed else "scientific_preflight_failed")

    def _finish(self, status):
        with self.lock:
            self.state.update(status=status, finished=time.time())
            _atomic_json(STATE, self.state)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the fixed three-task plan only")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"experiment": "SW0130_phase_state_integration",
                          "stage": "preflight_only", "max_parallel": MAX_PARALLEL,
                          "tasks": [{**task, "argv": command(task)} for task in task_plan()]},
                         indent=2, allow_nan=False))
        return
    PreflightQueue().run()


if __name__ == "__main__":
    main()
