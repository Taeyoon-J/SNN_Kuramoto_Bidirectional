"""Exclusive-GPU, no-retry dispatcher for the three SW0132 TRAIN preflights."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import pathlib
import signal
import statistics
import subprocess
import sys
import threading
import time

import numpy as np

try:
    import fcntl
except ImportError:  # Windows supports dry-run and validation tests only.
    fcntl = None

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0130_phase_state_integration import run
from collaborative_test.SW_0123_adaptive_temporal_assignment import dispatcher as owner
from collaborative_test.SW_0132_partition_relative_rgb import run as sw132

ARCHIVE = HERE / "results_archive"
STATE = ARCHIVE / "preflight_queue_state.json"
QUEUE_LOCK = ARCHIVE / "preflight_queue.lock"
GPU_LEASE_DIR = pathlib.Path("/tmp/kevinswk_sw0113_gpu_leases")
GPU_MEMORY_LIMIT_MIB = 512
MAX_PARALLEL = 2
FAMILIES = ("encoder", "graph", "oscillator_drive", "kuramoto", "dendrite",
            "membrane", "a_d", "a_m", "b")


def task_plan():
    return [
        {"task_id": f"sw0132_preflight_s{seed}", "stage": "preflight", "seed": seed,
         "depends_on": [] if seed == 0 else ["sw0132_preflight_s0"]}
        for seed in (0, 1, 2)
    ]


def artifact_path(task):
    if task.get("stage") != "preflight" or task.get("seed") not in (0, 1, 2):
        raise ValueError("queue only supports the three registered preflight seeds")
    return ARCHIVE / f"preflight_seed{int(task['seed'])}.json"


def command(task, device="cuda:0"):
    return [sys.executable, str(HERE / "run.py"), "--stage", "preflight",
            "--seed", str(int(task["seed"])), "--device", device,
            "--output", str(artifact_path(task))]


def gpu_is_exclusive_candidate(memory_mib, owners):
    return not owners and int(memory_mib) <= GPU_MEMORY_LIMIT_MIB


def validate_preflight_record(report, seed, *, fingerprint, source_sha,
                              source_manifest_sha, training_ids, pool_sha,
                              asset_hashes, warmup_path, seed0_report=None,
                              seed0_report_sha=None, warmup_sha_fn=None):
    import math
    import numpy as np

    if (report.get("status") != "passed"
            or report.get("experiment") != "SW0132_partition_relative_rgb"
            or report.get("seed") != seed
            or report.get("implementation_fingerprint") != fingerprint
            or report.get("source_core_sha256") != source_sha
            or report.get("source_manifest_sha256") != source_manifest_sha
            or report.get("source_steps") != 256
            or report.get("training_ids") != training_ids
            or report.get("training_ids_sha256") != hashlib.sha256(
                np.asarray(training_ids, dtype="<i8").tobytes()).hexdigest()
            or report.get("pool_indices_sha256") != pool_sha
            or report.get("shuffle_seed") != 117 + seed
            or report.get("batch_size") != 16
            or report.get("train_time_steps") != 64
            or report.get("settle_steps") != 32
            or report.get("asset_hashes") != asset_hashes
            or report.get("ground_truth_used") is not False
            or report.get("optimizer_updates") != 0
            or report.get("source_core_checkpoint_modified") is not False
            or report.get("source_encoder_modified") is not False):
        return False
    try:
        gamma_diff = float(report["live_source_gamma_max_abs_diff_first_batch"])
        if not math.isfinite(gamma_diff) or gamma_diff > 2e-5:
            return False
        if report.get("row_scramble_count_per_arm") != 64:
            return False
        scramble = report["row_scramble_by_arm"]
        if set(scramble) != {"phase", "constant"}:
            return False
        for row in scramble.values():
            values = row["per_image_excess"]
            if (len(values) != 64 or not all(math.isfinite(float(v)) for v in values)
                    or not math.isclose(float(row["mean_excess"]),
                                        float(np.mean(values)), rel_tol=1e-12, abs_tol=1e-12)
                    or row["mean_excess"] <= 0):
                return False
        paired = report["paired_zero_initial_arm_check"]
        if not all(paired.get(key) is True for key in (
                "initial_decoder_equal", "initial_gamma_q_hard_traces_equal",
                "warm_decoder_optimizer_shared_across_arms")):
            return False
        warm_path = pathlib.Path(report["decoder_warmup_artifact"])
        if warm_path.resolve() != pathlib.Path(warmup_path).resolve() or not warm_path.is_file():
            return False
        if warmup_sha_fn is None:
            warmup_sha_fn = sw132.sha
        if warmup_sha_fn(warm_path) != report.get("decoder_warmup_artifact_sha256"):
            return False
        warm = torch_load(warm_path)
        if warm.get("warmup_updates") != 32 or warm.get("source_core_sha256") != source_sha:
            return False
        if warm.get("training_ids_sha256") != report["training_ids_sha256"] \
                or warm.get("asset_hashes") != asset_hashes:
            return False
        steps = []
        for state in warm["optimizer_state_dict"].get("state", {}).values():
            value = state.get("step")
            steps.append(int(value.item() if hasattr(value, "item") else value))
        if len(steps) != 6 or set(steps) != {32}:
            return False
        parity = report["native_zero_initial_parity"]
        expected_checks = {(arm, step, settle) for arm in ("phase", "constant")
                           for step, settle in ((64, 32), (1024, 512))}
        actual_checks = {(row.get("arm"), row.get("time_steps"), row.get("settle"))
                         for row in parity["checks"]
                         if row.get("theta_component_membrane_spikes_q_h_primary_oldloss_exact") is True}
        if parity.get("source_core_sha256") != source_sha or actual_checks != expected_checks:
            return False
        required_disposables = {"phase", "constant"}
        disposable = report["disposable_updates"]
        if {row.get("arm") for row in disposable} != required_disposables:
            return False
        for row in disposable:
            if row.get("optimizer_updates") != 1 or row.get("throwaway_only") is not True:
                return False
            norms = row["rgb_gradient_norms_by_family"]
            if any(not math.isfinite(float(norms.get(key, 0.0))) or float(norms[key]) <= 0
                   for key in FAMILIES):
                return False
            changed_families = row.get("changed_core_families", {})
            if (set(changed_families) != {"graph", "oscillator_drive", "kuramoto",
                                         "dendrite", "membrane"}
                    or not all(value is True for value in changed_families.values())
                    or not row.get("joint_and_decoder_changed")
                    or row.get("changed_encoder_parameter_count", 0) <= 0
                    or row.get("changed_decoder_parameter_count", 0) <= 0
                    or row.get("changed_integration_parameters", []).count("b") != 1):
                return False
            by_component = row.get("integration_parameter_gradient_abs_by_component", {})
            if any(len(by_component.get(name, [])) != 4
                   or any(not math.isfinite(float(x)) or float(x) <= 0
                          for x in by_component[name])
                   for name in ("core.a_d", "core.a_m", "core.b")):
                return False
        value = float(report["lambda"])
        if not math.isfinite(value) or value <= 0 or report.get("lambda_source_seed") != 0:
            return False
        if seed == 0:
            if report.get("seed0_lambda_record_sha256") is not None:
                return False
            ratios = report.get("lambda_batch_ratios")
            rows = report.get("lambda_seed0_calibration")
            if not isinstance(ratios, list) or len(ratios) != 4 or not isinstance(rows, list) or len(rows) != 4:
                return False
            if not all(math.isfinite(float(v)) and float(v) > 0 for v in ratios):
                return False
            if not math.isclose(value, statistics.median(ratios), rel_tol=0, abs_tol=0):
                return False
            if [row.get("lambda_ratio") for row in rows] != ratios:
                return False
            for index, row in enumerate(rows):
                old_norm = float(row.get("old_joint_norm", float("nan")))
                rgb_norm = float(row.get("rgb_joint_norm", float("nan")))
                if (row.get("batch_index") != index or not math.isfinite(old_norm)
                        or not math.isfinite(rgb_norm) or old_norm <= 0 or rgb_norm <= 0
                        or not math.isclose(float(ratios[index]), .25 * old_norm / rgb_norm,
                                            rel_tol=1e-12, abs_tol=0)):
                    return False
        else:
            if not isinstance(seed0_report, dict) or seed0_report.get("status") != "passed":
                return False
            if (seed0_report.get("seed") != 0
                    or seed0_report.get("implementation_fingerprint") != fingerprint
                    or report.get("seed0_lambda_record_sha256") != seed0_report_sha
                    or report.get("lambda_seed0_calibration") != "reused frozen seed0 preflight"
                    or value != float(seed0_report["lambda"])):
                return False
    except (KeyError, TypeError, ValueError, OSError, OverflowError, RuntimeError):
        return False
    return True


def torch_load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def valid_result(task, assets=None):
    path = artifact_path(task)
    seed = int(task["seed"])
    if not path.is_file():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        checkpoint, manifest_path, manifest, pool, ids, source_sha = sw132.source_contract(seed)
        if assets is None:
            assets = sw132.sw130.validate_rgb_assets()
        seed0_report = seed0_sha = None
        if seed:
            seed0_path = artifact_path({"stage": "preflight", "seed": 0})
            if not seed0_path.is_file() or not valid_result({"stage": "preflight", "seed": 0}, assets):
                return False
            seed0_report = json.loads(seed0_path.read_text(encoding="utf-8"))
            seed0_sha = sw132.sha(seed0_path)
        pool_sha = hashlib.sha256(np.asarray(pool, dtype="<i8").tobytes()).hexdigest()
        return validate_preflight_record(
            report, seed, fingerprint=sw132.implementation_fingerprint(),
            source_sha=source_sha, source_manifest_sha=sw132.sha(manifest_path),
            training_ids=ids, pool_sha=pool_sha, asset_hashes=assets,
            warmup_path=ARCHIVE / f"preflight_decoder_seed{seed}.pt",
            seed0_report=seed0_report, seed0_report_sha=seed0_sha)
    except (OSError, ValueError, KeyError, TypeError, AssertionError):
        return False


def _atomic_json(path, value):
    path = pathlib.Path(path)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


class PreflightQueue:
    def __init__(self):
        if fcntl is None:
            raise RuntimeError("SW0132 queue requires Linux flock")
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        self.queue_lock = QUEUE_LOCK.open("a+")
        try:
            fcntl.flock(self.queue_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.queue_lock.close()
            raise RuntimeError("another SW0132 queue owns its lock") from exc
        if STATE.exists():
            fcntl.flock(self.queue_lock.fileno(), fcntl.LOCK_UN)
            self.queue_lock.close()
            raise FileExistsError(f"preserve existing SW0132 queue state: {STATE}")
        occupied = [path for task in task_plan() for path in (
            artifact_path(task), ARCHIVE / f"preflight_decoder_seed{task['seed']}.pt",
            ARCHIVE / f"{task['task_id']}.log") if path.exists()]
        if occupied:
            fcntl.flock(self.queue_lock.fileno(), fcntl.LOCK_UN)
            self.queue_lock.close()
            raise FileExistsError(f"preserve prior SW0132 task artifacts: {occupied}")
        self.lock = threading.RLock()
        self.tasks = task_plan()
        self.assets = sw132.sw130.validate_rgb_assets()
        self.state = {"experiment": "SW0132_partition_relative_rgb", "stage": "preflight_only",
                      "status": "running", "supervisor_pid": os.getpid(),
                      "started": time.time(), "runner_fingerprint": sw132.implementation_fingerprint(),
                      "queue_sha256": sw132.sha(HERE / "preflight_queue.py"),
                      "owner_dispatcher_sha256": sw132.sha(pathlib.Path(owner.__file__)),
                      "task_count": 3, "no_training_stage": True,
                      "tasks": {row["task_id"]: {"task": row, "status": "queued"}
                                for row in self.tasks}}
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
                    lease.close(); continue
                try:
                    gpu_uuid, memory, owners = owner._nvidia_gpu_info(gpu)
                    if gpu_is_exclusive_candidate(memory, owners):
                        self._update(task_id, status="reserved", gpu=gpu, gpu_uuid=gpu_uuid,
                                     memory_before_mib=memory, lease_path=str(lease_path),
                                     reservation_pid=os.getpid(), reserved=time.time())
                        return gpu, lease, gpu_uuid
                except BaseException:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); raise
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close()
            self._update(task_id, status="waiting_for_exclusive_gpu", last_wait=time.time())
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
        tid, output = task["task_id"], artifact_path(task)
        if output.exists():
            if valid_result(task, self.assets):
                self._update(tid, status="reused_verified", artifact=str(output), verified_at=time.time())
                return "reused_verified"
            self._update(tid, status="failed_existing_artifact", artifact=str(output), finished=time.time())
            return "failed_existing_artifact"
        warm_path = ARCHIVE / f"preflight_decoder_seed{task['seed']}.pt"
        if warm_path.exists():
            self._update(tid, status="failed_existing_warmup", artifact=str(warm_path), finished=time.time())
            return "failed_existing_warmup"
        gpu, lease, gpu_uuid = self._reserve_gpu(tid)
        log_path = ARCHIVE / f"{tid}.log"
        proc = None
        try:
            if log_path.exists():
                raise FileExistsError(f"preserve existing log: {log_path}")
            triton = pathlib.Path(f"/tmp/kevinswk_sw0132_triton_{tid}_{int(time.time()*1000)}")
            triton.mkdir(parents=True, exist_ok=False)
            argv = command(task, "cuda:0")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", TRITON_CACHE_DIR=str(triton))
            self._update(tid, status="launching", gpu=gpu, gpu_uuid=gpu_uuid, argv=argv,
                         log_path=str(log_path), triton_cache=str(triton), launched=time.time())
            with log_path.open("x", encoding="utf-8") as log:
                proc = subprocess.Popen(argv, cwd=str(ROOT), env=env, stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True,
                                        pass_fds=(lease.fileno(),))
                self._update(tid, status="running", child_pid=proc.pid, started=time.time())
                while proc.poll() is None:
                    time.sleep(5)
                    if proc.poll() is not None:
                        break
                    _uuid, _memory, owners = owner._nvidia_gpu_info(gpu)
                    foreign = owner.foreign_owner_pids(owners, owner._process_tree(proc.pid))
                    if foreign:
                        self._terminate_owned(proc)
                        self._update(tid, status="interrupted_foreign_gpu_owner",
                                     foreign_owner_pids=foreign, finished=time.time())
                        return "interrupted_foreign_gpu_owner"
            rc = int(proc.returncode)
            passed = rc == 0 and valid_result(task, self.assets)
            status = "passed" if passed else "failed"
            self._update(tid, status=status, returncode=rc, artifact_valid=bool(passed), finished=time.time())
            return status
        except BaseException as exc:
            if proc is not None:
                self._terminate_owned(proc)
            self._update(tid, status="failed", error=repr(exc), finished=time.time())
            return "failed"
        finally:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close()

    def run(self):
        first = self._execute(self.tasks[0])
        if first not in ("passed", "reused_verified"):
            for task in self.tasks[1:]:
                self._update(task["task_id"], status="blocked_seed0_failure")
            self._finish("seed0_preflight_failed")
            return
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
            futures = {executor.submit(self._execute, task): task for task in self.tasks[1:]}
            results = {futures[future]["task_id"]: future.result() for future in futures}
        passed = all(value in ("passed", "reused_verified") for value in results.values())
        self._finish("all_preflights_passed" if passed else "scientific_preflight_failed")

    def _finish(self, status):
        with self.lock:
            self.state.update(status=status, finished=time.time())
            _atomic_json(STATE, self.state)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"experiment": "SW0132_partition_relative_rgb",
                          "stage": "preflight_only", "max_parallel": MAX_PARALLEL,
                          "tasks": [{**task, "argv": command(task)} for task in task_plan()]},
                         indent=2, allow_nan=False))
        return
    PreflightQueue().run()


if __name__ == "__main__":
    main()
