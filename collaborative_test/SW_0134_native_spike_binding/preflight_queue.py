"""Exclusive-GPU, no-retry queue for SW0134's three source preflights."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
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
except ImportError:  # Windows supports contracts and dry-run only.
    fcntl = None

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

import numpy as np
import torch

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0123_adaptive_temporal_assignment import dispatcher as owner
from collaborative_test.SW_0134_native_spike_binding import run

ARCHIVE = HERE / "results_archive"
STATE = ARCHIVE / "preflight_queue_state.json"
QUEUE_LOCK = ARCHIVE / "preflight_queue.lock"
GPU_LEASE_DIR = pathlib.Path("/tmp/kevinswk_sw0113_gpu_leases")
GPU_MEMORY_LIMIT_MIB = 512
MAX_PARALLEL = 2
LIVE_FAMILIES = ("encoder", "graph", "oscillator_drive", "kuramoto",
                 "dendrite", "membrane", "a_d", "a_m", "b")
RGB_SOURCE_FAMILIES = ("encoder", "graph", "oscillator_drive", "kuramoto",
                       "dendrite", "membrane", "a_d", "a_m", "b")
ARMS = ("actual_joint", "gate_joint", "actual_frozen")


def task_plan():
    return [
        {"task_id": "sw0134_preflight_s0", "stage": "preflight", "seed": 0,
         "depends_on": [], "arm": "all_three"},
        {"task_id": "sw0134_preflight_s1", "stage": "preflight", "seed": 1,
         "depends_on": ["sw0134_preflight_s0"], "arm": "all_three"},
        {"task_id": "sw0134_preflight_s2", "stage": "preflight", "seed": 2,
         "depends_on": ["sw0134_preflight_s0"], "arm": "all_three"},
    ]


def artifact_path(task):
    if task.get("stage") != "preflight" or task.get("seed") not in (0, 1, 2):
        raise ValueError("SW0134 queue accepts only its three registered preflights")
    return ARCHIVE / f"preflight_seed{int(task['seed'])}.json"


def command(task, device="cuda:0"):
    return [sys.executable, str(HERE / "run.py"), "--stage", "preflight",
            "--seed", str(int(task["seed"])), "--device", device,
            "--output", str(artifact_path(task))]


def gpu_is_exclusive_candidate(memory_mib, compute_owners):
    return not compute_owners and int(memory_mib) <= GPU_MEMORY_LIMIT_MIB


def _finite_positive(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(value) and value > 0


def validate_preflight_record(report, seed, *, fingerprint, source_sha,
                              source_manifest_sha, training_ids, pool_sha,
                              asset_hashes, archive, seed0_report=None,
                              seed0_report_sha=None):
    """Strictly validate the canonical SW0134 preflight evidence."""
    ids_sha = hashlib.sha256(np.asarray(training_ids, dtype="<i8").tobytes()).hexdigest()
    expected = {
        "status": "passed", "experiment": "SW0134_native_spike_binding", "seed": seed,
        "shuffle_seed": 117 + seed, "batch_size": 16, "train_time_steps": 1024,
        "settle_steps": 512, "live_tail_steps": 64,
        "source_core_sha256": source_sha, "source_manifest_sha256": source_manifest_sha,
        "source_updates": 256, "training_ids": training_ids,
        "training_ids_sha256": ids_sha, "pool_indices_sha256": pool_sha,
        "asset_hashes": asset_hashes, "ground_truth_used": False,
        "source_checkpoint_modified": False, "joint_training_updates": 0,
        "warmup_optimizer_updates_per_head": 32,
        "throwaway_optimizer_updates_per_arm": 1,
        "implementation_fingerprint": fingerprint,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        return False
    try:
        gamma_diff = float(report["live_gamma_cache_max_abs_diff_first_batch"])
        if not math.isfinite(gamma_diff) or gamma_diff > 2e-5:
            return False
        expected_parity = {("phase", 64, 32), ("constant", 64, 32),
                           ("phase", 1024, 512), ("constant", 1024, 512)}
        parity = report["native_zero_init_late_parity"]
        parity_rows = {(row["arm"], row["steps"], row["settle"])
                       for row in parity if row.get("full_component_trace_gate_q_h_oldloss_exact") is True
                       and row.get("production_adapter_gate_trace_exact") is True}
        if parity_rows != expected_parity:
            return False

        expected_warm_params = (sum(1 for _ in run.NativeSpikeSlotBinder().parameters()) +
                                sum(1 for _ in run.RelativeSlotRGBDecoder().parameters()))
        warm = report["warm_artifacts"]
        if set(warm) != {"actual_joint", "gate_joint"} or report.get("warm_actual_arms_shared") is not True:
            return False
        for arm, info in warm.items():
            path = pathlib.Path(info["path"])
            expected_path = pathlib.Path(archive) / f"warm_{arm}_seed{seed}.pt"
            if (int(info["updates"]) != 32 or path.resolve() != expected_path.resolve()
                    or not path.is_file() or run.sha(path) != info["sha256"]):
                return False
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if (payload.get("experiment") != "SW0134_native_spike_binding"
                    or payload.get("seed") != seed or payload.get("arm") != arm
                    or payload.get("updates") != 32 or payload.get("batch_size") != 16
                    or payload.get("training_ids") != [int(x) for x in training_ids[:512]]
                    or payload.get("training_ids_sha256") != hashlib.sha256(
                        np.asarray(training_ids[:512], dtype="<i8").tobytes()).hexdigest()
                    or payload.get("all_training_ids_sha256") != ids_sha
                    or payload.get("source_core_sha256") != source_sha
                    or payload.get("source_manifest_sha256") != source_manifest_sha
                    or payload.get("asset_hashes") != asset_hashes
                    or payload.get("implementation_fingerprint") != fingerprint):
                return False
            for name in ("binder_state_dict", "decoder_state_dict"):
                state = payload.get(name)
                if not isinstance(state, dict) or not state or any(
                        torch.is_tensor(v) and not bool(torch.isfinite(v).all())
                        for v in state.values()):
                    return False
            opt_states = payload["optimizer_state_dict"].get("state", {})
            steps = [int((s["step"].item() if torch.is_tensor(s["step"]) else s["step"]))
                     for s in opt_states.values() if "step" in s]
            if len(steps) != expected_warm_params or set(steps) != {32}:
                return False
            losses = info.get("loss_first_last", [])
            if len(losses) != 2 or not all(math.isfinite(float(x)) for x in losses):
                return False

        updates = report["disposable_updates"]
        rows = {row["arm"]: row for row in updates}
        if set(rows) != set(ARMS) or len(updates) != len(ARMS):
            return False
        for arm, row in rows.items():
            if (row.get("throwaway_only") is not True
                    or not math.isfinite(float(row["old_loss"]))
                    or not math.isfinite(float(row["rgb_loss"]))
                    or not _finite_positive(row["head_gradient_norm"])
                    or int(row["head_changed_parameter_count"]) <= 0):
                return False
            norms = row["rgb_gradient_norms_by_source_family"]
            if any(not math.isfinite(float(v)) or float(v) < 0 for v in norms.values()):
                return False
            if arm == "actual_joint":
                if (not _finite_positive(row["joint_gradient_norm"])
                        or int(row["joint_changed_parameter_count"]) <= 0
                        or any(not _finite_positive(norms.get(k)) for k in RGB_SOURCE_FAMILIES)
                        or row.get("source_frozen") is not False):
                    return False
            elif arm == "gate_joint":
                if (not _finite_positive(row["joint_gradient_norm"])
                        or int(row["joint_changed_parameter_count"]) <= 0
                        or row.get("source_frozen") is not False
                        or any(float(norms.get(k, 0.0)) != 0.0 for k in
                               ("dendrite", "membrane", "a_d", "a_m", "b"))):
                    return False
            elif (float(row["joint_gradient_norm"]) != 0.0
                  or int(row["joint_changed_parameter_count"]) != 0
                  or row.get("source_frozen") is not True
                  or row.get("source_training_mode") != "eval"
                  or row.get("source_parameters_unchanged") is not True
                  or norms != {}):
                return False

        value = float(report["lambda"])
        if not _finite_positive(value) or report.get("lambda_source_seed") != 0:
            return False
        if seed == 0:
            calibration = report.get("lambda_calibration")
            if report.get("lambda_seed0_record_sha256") is not None or len(calibration) != 4:
                return False
            ratios = []
            for i, row in enumerate(calibration):
                old, rgb, ratio = (float(row[k]) for k in
                                   ("old_joint_norm", "rgb_joint_norm", "lambda_ratio"))
                if (row.get("batch_index") != i or not all(math.isfinite(x) and x > 0
                        for x in (old, rgb, ratio))
                        or not math.isclose(ratio, .25 * old / rgb, rel_tol=1e-12, abs_tol=0.0)):
                    return False
                ratios.append(ratio)
            if value != float(statistics.median(ratios)):
                return False
        elif (not isinstance(seed0_report, dict) or not seed0_report_sha
              or seed0_report.get("status") != "passed" or seed0_report.get("seed") != 0
              or seed0_report.get("implementation_fingerprint") != fingerprint
              or report.get("lambda_seed0_record_sha256") != seed0_report_sha
              or value != float(seed0_report.get("lambda", float("nan")))):
            return False
    except (KeyError, TypeError, ValueError, OSError, OverflowError, RuntimeError):
        return False
    return True


def valid_result(task, assets=None):
    path = artifact_path(task)
    seed = int(task["seed"])
    if not path.is_file():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        _checkpoint, manifest_path, _manifest, pool, ids, source_sha = run.source_contract(seed)
        assets = run.sw130.validate_rgb_assets() if assets is None else assets
        seed0_report = seed0_sha = None
        if seed:
            seed0_path = artifact_path({"stage": "preflight", "seed": 0})
            if not seed0_path.is_file():
                return False
            seed0_report = json.loads(seed0_path.read_text(encoding="utf-8"))
            seed0_sha = run.sha(seed0_path)
            if not valid_result({"stage": "preflight", "seed": 0}, assets):
                return False
        pool_sha = hashlib.sha256(np.asarray(pool, dtype="<i8").tobytes()).hexdigest()
        return validate_preflight_record(
            report, seed, fingerprint=run.implementation_fingerprint(), source_sha=source_sha,
            source_manifest_sha=run.sha(manifest_path), training_ids=ids,
            pool_sha=pool_sha, asset_hashes=assets, archive=ARCHIVE,
            seed0_report=seed0_report, seed0_report_sha=seed0_sha)
    except (OSError, ValueError, KeyError, TypeError, AssertionError,
            RuntimeError, EOFError, json.JSONDecodeError):
        return False


def _atomic_json(path, value):
    path = pathlib.Path(path)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


class PreflightQueue:
    def __init__(self):
        if fcntl is None:
            raise RuntimeError("SW0134 GPU queue requires Linux flock")
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        self.queue_lock = QUEUE_LOCK.open("a+")
        try:
            fcntl.flock(self.queue_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.queue_lock.close()
            raise RuntimeError("another SW0134 preflight queue owns its lock") from exc
        if STATE.exists():
            fcntl.flock(self.queue_lock.fileno(), fcntl.LOCK_UN)
            self.queue_lock.close()
            raise FileExistsError(f"preserve existing SW0134 queue state: {STATE}")
        self.lock = threading.RLock()
        self.tasks = task_plan()
        self.assets = run.sw130.validate_rgb_assets()
        self.state = {
            "experiment": "SW0134_native_spike_binding", "stage": "preflight_only",
            "status": "running", "supervisor_pid": os.getpid(), "started": time.time(),
            "runner_fingerprint": run.implementation_fingerprint(),
            "queue_sha256": run.sha(HERE / "preflight_queue.py"),
            "owner_dispatcher_sha256": run.sha(owner.__file__),
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

    def _finish(self, status):
        with self.lock:
            self.state.update(status=status, finished=time.time())
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
                    uuid, memory, owners = owner._nvidia_gpu_info(gpu)
                    if gpu_is_exclusive_candidate(memory, owners):
                        self._update(task_id, status="reserved", gpu=gpu, gpu_uuid=uuid,
                                     memory_before_mib=memory, lease_path=str(lease_path),
                                     reservation_pid=os.getpid(), reserved=time.time())
                        return gpu, lease, uuid
                except BaseException:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); raise
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close()
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
        artifact = artifact_path(task)
        warmups = [ARCHIVE / f"warm_{arm}_seed{task['seed']}.pt"
                   for arm in ("actual_joint", "gate_joint")]
        log_path = ARCHIVE / f"{task_id}.log"
        if artifact.exists() or log_path.exists() or any(p.exists() for p in warmups):
            self._update(task_id, status="failed_existing_artifact", finished=time.time(),
                         preserved=[str(p) for p in [artifact, log_path, *warmups] if p.exists()])
            return "failed_existing_artifact"
        lease = None
        proc = None
        try:
            gpu, lease, uuid = self._reserve_gpu(task_id)
            triton = pathlib.Path(f"/tmp/kevinswk_sw0134_triton_{task_id}_{int(time.time()*1000)}")
            triton.mkdir(parents=True, exist_ok=False)
            argv = command(task, "cuda:0")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", TRITON_CACHE_DIR=str(triton))
            self._update(task_id, status="launching", gpu=gpu, gpu_uuid=uuid,
                         argv=argv, log_path=str(log_path), triton_cache=str(triton),
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
                    _uuid, _mem, owners = owner._nvidia_gpu_info(gpu)
                    foreign = owner.foreign_owner_pids(owners, owner._process_tree(proc.pid))
                    if foreign:
                        self._terminate_owned_child(proc)
                        self._update(task_id, status="interrupted_foreign_gpu_owner",
                                     foreign_owner_pids=foreign, finished=time.time())
                        return "interrupted_foreign_gpu_owner"
            rc = int(proc.returncode)
            passed = rc == 0 and valid_result(task, self.assets)
            status = "passed" if passed else "failed"
            self._update(task_id, status=status, returncode=rc, artifact_valid=bool(passed),
                         finished=time.time())
            return status
        except BaseException as exc:
            if proc is not None:
                self._terminate_owned_child(proc)
            self._update(task_id, status="failed", error=repr(exc), finished=time.time())
            return "failed"
        finally:
            if lease is not None:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                lease.close()

    def _run_later(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
            futures = {pool.submit(self._execute, task): task for task in self.tasks[1:]}
            return {futures[f]["task_id"]: f.result() for f in futures}

    def run(self):
        first = self._execute(self.tasks[0])
        if first not in ("passed", "reused_verified"):
            for task in self.tasks[1:]:
                self._update(task["task_id"], status="blocked_seed0_preflight_failure")
            self._finish("seed0_preflight_failed")
            return
        outcomes = self._run_later()
        self._finish("all_preflights_passed" if all(
            value in ("passed", "reused_verified") for value in outcomes.values())
            else "scientific_preflight_failed")


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"experiment": "SW0134_native_spike_binding",
                          "stage": "preflight_only", "max_parallel": MAX_PARALLEL,
                          "tasks": [{**task, "argv": command(task)} for task in task_plan()]},
                         indent=2, allow_nan=False))
        return
    PreflightQueue().run()


if __name__ == "__main__":
    main()
