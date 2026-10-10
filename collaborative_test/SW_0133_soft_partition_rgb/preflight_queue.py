"""Exclusive-GPU, no-retry queue for SW0133's three source preflights."""
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

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0123_adaptive_temporal_assignment import dispatcher as owner
from collaborative_test.SW_0133_soft_partition_rgb import run

ARCHIVE = HERE / "results_archive"
STATE = ARCHIVE / "preflight_queue_state.json"
QUEUE_LOCK = ARCHIVE / "preflight_queue.lock"
GPU_LEASE_DIR = pathlib.Path("/tmp/kevinswk_sw0113_gpu_leases")
GPU_MEMORY_LIMIT_MIB = 512
MAX_PARALLEL = 2
LIVE_FAMILIES = ("encoder", "graph", "oscillator_drive", "kuramoto",
                 "dendrite", "membrane", "a_d", "a_m", "b")
NATIVE_CORE_FAMILIES = ("graph", "oscillator_drive", "kuramoto", "dendrite", "membrane")
ARMS = ("phase_live", "constant_live", "phase_detached")


def task_plan():
    return [
        {"task_id": "sw0133_preflight_s0", "stage": "preflight", "seed": 0,
         "depends_on": [], "arm": "all_three"},
        {"task_id": "sw0133_preflight_s1", "stage": "preflight", "seed": 1,
         "depends_on": ["sw0133_preflight_s0"], "arm": "all_three"},
        {"task_id": "sw0133_preflight_s2", "stage": "preflight", "seed": 2,
         "depends_on": ["sw0133_preflight_s0"], "arm": "all_three"},
    ]


def artifact_path(task):
    if task.get("stage") != "preflight" or task.get("seed") not in (0, 1, 2):
        raise ValueError("SW0133 queue accepts only its three registered preflights")
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
                              asset_hashes, warmup_path, seed0_report=None,
                              seed0_report_sha=None, warmup_sha_fn=None):
    """Validate all emitted preflight evidence; scientific screens fail closed."""
    expected = {
        "status": "passed", "experiment": "SW0133_soft_partition_rgb", "seed": seed,
        "shuffle_seed": 117 + seed, "batch_size": 16, "train_time_steps": 64,
        "settle_steps": 32, "source_core_sha256": source_sha,
        "source_manifest_sha256": source_manifest_sha,
        "source_steps": 256, "training_ids": training_ids,
        "pool_indices_sha256": pool_sha, "asset_hashes": asset_hashes,
        "ground_truth_used": False, "source_core_checkpoint_modified": False,
        "source_encoder_modified": False, "optimizer_updates": 0,
        "implementation_fingerprint": fingerprint,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        return False
    try:
        ids_sha = hashlib.sha256(np.asarray(training_ids, dtype="<i8").tobytes()).hexdigest()
        if report.get("training_ids_sha256") != ids_sha:
            return False
        gamma_diff = float(report["live_source_gamma_max_abs_diff_first_batch"])
        if not math.isfinite(gamma_diff) or gamma_diff > 2e-5:
            return False
        parity = report["native_zero_initial_parity"]
        expected_checks = {("phase", 64, 32), ("constant", 64, 32),
                           ("phase", 1024, 512), ("constant", 1024, 512)}
        actual_checks = {(row.get("arm"), row.get("time_steps"), row.get("settle"))
                         for row in parity["checks"]
                         if row.get("theta_component_membrane_spikes_q_h_primary_oldloss_exact") is True}
        if parity.get("source_core_sha256") != source_sha or actual_checks != expected_checks:
            return False
        arm_check = report["paired_zero_initial_arm_check"]
        if not all(arm_check.get(key) is True for key in (
                "initial_decoder_equal", "initial_gamma_q_hard_traces_equal",
                "initial_soft_p_rgb_equal", "warm_decoder_optimizer_shared_across_arms")):
            return False
        warm = pathlib.Path(report["decoder_warmup_artifact"])
        if warm.resolve() != pathlib.Path(warmup_path).resolve() or not warm.is_file():
            return False
        if warmup_sha_fn is None:
            warmup_sha_fn = run.sha
        if warmup_sha_fn(warm) != report.get("decoder_warmup_artifact_sha256"):
            return False
        if report.get("decoder_warmup_updates") != 32:
            return False
        warm_payload = __import__("torch").load(warm, map_location="cpu", weights_only=True)
        if (warm_payload.get("source_core_sha256") != source_sha
                or warm_payload.get("training_ids_sha256") != ids_sha
                or warm_payload.get("asset_hashes") != asset_hashes
                or warm_payload.get("warmup_updates") != 32
                or not isinstance(warm_payload.get("decoder_state_dict"), dict)
                or not warm_payload["decoder_state_dict"]):
            return False
        if any(__import__("torch").is_tensor(value)
               and not __import__("torch").isfinite(value).all()
               for value in warm_payload["decoder_state_dict"].values()):
            return False
        run._validate_warm_optimizer(warm_payload)
        warm_loss = report["decoder_warmup_loss_first_last"]
        if len(warm_loss) != 2 or any(not math.isfinite(float(x)) for x in warm_loss):
            return False
        if report.get("row_scramble_count_per_arm") != 64:
            return False
        scramble = report["row_scramble_by_arm"].get("phase_live")
        if (scramble is None or len(scramble["per_image_excess"]) != 64
                or not all(math.isfinite(float(x)) for x in scramble["per_image_excess"])
                or not math.isfinite(float(scramble["mean_excess"]))
                or scramble.get("positive_count") != sum(
                    float(x) > 0 for x in scramble["per_image_excess"])
                or not math.isclose(float(scramble["mean_excess"]),
                                    float(np.mean(scramble["per_image_excess"])),
                                    rel_tol=1e-12, abs_tol=1e-15)):
            return False
        if not isinstance(report.get("disposable_updates"), list):
            return False
        updates = {row.get("arm"): row for row in report["disposable_updates"]}
        if set(updates) != set(ARMS):
            return False
        for arm, row in updates.items():
            if row.get("optimizer_updates") != 1 or row.get("throwaway_only") is not True:
                return False
            if not _finite_positive(row.get("joint_preclip_norm")) or not _finite_positive(
                    row.get("decoder_preclip_norm")):
                return False
            if not math.isfinite(float(row["old_loss"])) or not math.isfinite(float(row["rgb_loss"])):
                return False
            q_norm = float(row["rgb_to_q_gradient_norm"])
            if not math.isfinite(q_norm):
                return False
            family_norms = row["rgb_gradient_norms_by_family"]
            if arm == "phase_detached":
                if q_norm != 0.0 or any(float(x) != 0.0 for x in family_norms.values()):
                    return False
                if not row.get("joint_and_decoder_changed"):
                    return False
            else:
                if q_norm <= 0 or any(not _finite_positive(family_norms.get(name))
                                       for name in LIVE_FAMILIES):
                    return False
                changed_core = row["changed_core_families"]
                if set(changed_core) != set(NATIVE_CORE_FAMILIES) or not all(changed_core.values()):
                    return False
                if (row.get("changed_encoder_parameter_count", 0) <= 0
                        or row.get("changed_decoder_parameter_count", 0) <= 0
                        or "b" not in row.get("changed_integration_parameters", [])):
                    return False
            integration = row["integration_parameter_gradient_abs_by_component"]
            if not all(name in integration and len(integration[name]) == 4
                       and all(math.isfinite(float(v)) and float(v) >= 0
                               for v in integration[name])
                       for name in ("core.a_d", "core.a_m", "core.b")):
                return False
            unused = row["rgb_gradient_unused_integration"]
            if arm != "phase_detached":
                if set(unused) != {"core.a_d", "core.a_m", "core.b"} or any(unused.values()):
                    return False
            else:
                if (set(unused) != {"core.a_d", "core.a_m", "core.b"}
                        or not all(unused.values())
                        or any(any(float(v) != 0.0 for v in integration[name])
                               for name in integration)):
                    return False
        value = float(report["lambda"])
        if not _finite_positive(value) or report.get("lambda_source_seed") != 0:
            return False
        if seed == 0:
            if report.get("seed0_lambda_record_sha256") is not None:
                return False
            rows = report.get("lambda_seed0_calibration")
            ratios = report.get("lambda_batch_ratios")
            if not isinstance(rows, list) or len(rows) != 4 or not isinstance(ratios, list) or len(ratios) != 4:
                return False
            recomputed = []
            for batch_index, row in enumerate(rows):
                old_norm = float(row["old_joint_norm"])
                rgb_norm = float(row["rgb_joint_norm"])
                ratio = float(row["lambda_ratio"])
                if (row.get("batch_index") != batch_index
                        or not all(math.isfinite(x) and x > 0 for x in (old_norm, rgb_norm, ratio))
                        or not math.isclose(ratio, 0.25 * old_norm / rgb_norm,
                                            rel_tol=1e-12, abs_tol=0.0)
                        or ratio != float(ratios[batch_index])):
                    return False
                recomputed.append(ratio)
            if value != float(statistics.median(recomputed)):
                return False
        else:
            if not isinstance(seed0_report, dict) or not seed0_report_sha:
                return False
            if (seed0_report.get("status") != "passed" or seed0_report.get("seed") != 0
                    or seed0_report.get("implementation_fingerprint") != fingerprint
                    or report.get("seed0_lambda_record_sha256") != seed0_report_sha
                    or report.get("lambda_seed0_calibration") != "reused frozen seed0 preflight"
                    or value != float(seed0_report["lambda"])):
                return False
            if report.get("lambda_batch_ratios") is not None:
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
            report, seed, fingerprint=run.implementation_fingerprint(),
            source_sha=source_sha, source_manifest_sha=run.sha(manifest_path),
            training_ids=ids, pool_sha=pool_sha, asset_hashes=assets,
            warmup_path=ARCHIVE / f"preflight_decoder_seed{seed}.pt",
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
            raise RuntimeError("SW0133 GPU queue requires Linux flock")
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        self.queue_lock = QUEUE_LOCK.open("a+")
        try:
            fcntl.flock(self.queue_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.queue_lock.close()
            raise RuntimeError("another SW0133 preflight queue owns its lock") from exc
        if STATE.exists():
            fcntl.flock(self.queue_lock.fileno(), fcntl.LOCK_UN)
            self.queue_lock.close()
            raise FileExistsError(f"preserve existing SW0133 queue state: {STATE}")
        self.lock = threading.RLock()
        self.tasks = task_plan()
        self.assets = run.sw130.validate_rgb_assets()
        self.state = {
            "experiment": "SW0133_soft_partition_rgb", "stage": "preflight_only",
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
        warmup = ARCHIVE / f"preflight_decoder_seed{task['seed']}.pt"
        log_path = ARCHIVE / f"{task_id}.log"
        if artifact.exists() or warmup.exists() or log_path.exists():
            self._update(task_id, status="failed_existing_artifact", finished=time.time(),
                         preserved=[str(p) for p in (artifact, warmup, log_path) if p.exists()])
            return "failed_existing_artifact"
        lease = None
        proc = None
        try:
            gpu, lease, uuid = self._reserve_gpu(task_id)
            triton = pathlib.Path(f"/tmp/kevinswk_sw0133_triton_{task_id}_{int(time.time()*1000)}")
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
        print(json.dumps({"experiment": "SW0133_soft_partition_rgb",
                          "stage": "preflight_only", "max_parallel": MAX_PARALLEL,
                          "tasks": [{**task, "argv": command(task)} for task in task_plan()]},
                         indent=2, allow_nan=False))
        return
    PreflightQueue().run()


if __name__ == "__main__":
    main()
