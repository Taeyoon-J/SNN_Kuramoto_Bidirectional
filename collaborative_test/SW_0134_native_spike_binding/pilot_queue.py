"""Owner-aware, no-retry SW0134 seed-1 train-then-evaluate queue."""
from __future__ import annotations

import concurrent.futures
import json
import math
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time

try:
    import fcntl
except ImportError:  # Windows supports dry-run and validation only.
    fcntl = None

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

import numpy as np

from collaborative_test.SW_0123_adaptive_temporal_assignment import dispatcher as owner
from collaborative_test.SW_0134_native_spike_binding import run, train, evaluate

ARCHIVE = HERE / "results_archive"
STATE = ARCHIVE / "pilot_queue_state.json"
LOCK = ARCHIVE / "pilot_queue.lock"
LEASES = pathlib.Path("/tmp/kevinswk_sw0113_gpu_leases")
MAX_MEMORY_MIB = 512
MAX_PARALLEL_TRAIN = 3
ARMS = tuple(run.ARMS)
METRICS = tuple(evaluate.METRICS)
REPORT_METRICS = ("patch_fg_ari", "patch_foreground_iou", "patch_matched_object_iou")


def registered_pilot_gate(report, *, bootstrap_samples=10000):
    """Apply the preregistered seed-1 paired comparisons to primary predictions."""
    if report.get("status") != "complete" or report.get("seed") != 1:
        raise ValueError("complete seed-1 evaluation is required for pilot gate")
    scores = report["scores"]
    refs = ("source97_native", "gate_joint", "actual_frozen")
    candidates = scores["actual_joint"]["primary"]
    reference = {name: scores[name]["primary"] for name in refs}
    rng = np.random.RandomState(134)
    indices = rng.randint(0, 320, size=(bootstrap_samples, 320))
    comparisons = {}
    for ref_name in refs:
        result = {}
        for metric in REPORT_METRICS:
            cand = np.asarray(candidates[metric]["per_image"], dtype=np.float64)
            ref = np.asarray(reference[ref_name][metric]["per_image"], dtype=np.float64)
            if cand.shape != (320,) or ref.shape != (320,) or not np.isfinite(cand).all() or not np.isfinite(ref).all():
                raise ValueError("paired endpoint metric arrays must be finite 320-vectors")
            diff = cand - ref
            low, high = np.quantile(diff[indices].mean(axis=1), [0.025, 0.975])
            result[metric] = {"mean_difference": float(diff.mean()),
                              "ci95": [float(low), float(high)]}
        comparisons[ref_name] = result
    strict_means = all(float(candidates[m]["mean"]) >
                       float(reference[ref][m]["mean"])
                       for ref in refs for m in ("patch_fg_ari",))
    positive_fg_ci = all(comparisons[ref]["patch_fg_ari"]["ci95"][0] > 0
                         for ref in refs)
    passed = strict_means and positive_fg_ci
    return {"experiment": "SW0134_native_spike_binding", "seed": 1,
            "status": "seed1_pilot_gate_passed" if passed else "seed1_pilot_gate_failed",
            "bootstrap": {"method": "paired_image", "samples": bootstrap_samples,
                          "random_state": 134, "common_indices_across_comparisons": True},
            "actual_joint_strictly_above_all_references_fg_ari": strict_means,
            "all_three_registered_fg_ari_ci_lower_bounds_positive": positive_fg_ci,
            "comparisons_actual_joint_minus_reference": comparisons,
            "interpretation": "Seed-1 pilot only; no multi-seed promotion or scaling claim."}


def task_plan():
    tasks = [{"task_id": f"sw0134_train_seed1_{arm}", "stage": "train",
              "seed": 1, "arm": arm, "depends_on": [f"preflight_seed{s}" for s in run.SEEDS]}
             for arm in ARMS]
    tasks.append({"task_id": "sw0134_train_diagnostic_seed1", "stage": "diagnostic",
                  "seed": 1, "arm": "all_three", "depends_on": [t["task_id"] for t in tasks]})
    tasks.append({"task_id": "sw0134_evaluate_seed1", "stage": "evaluate", "seed": 1,
                  "arm": "all_three", "depends_on": ["sw0134_train_diagnostic_seed1"]})
    return tasks


def artifact_path(task):
    if task.get("seed") != 1:
        raise ValueError("SW0134 paired pilot is registered for seed1 only")
    if task.get("stage") == "train" and task.get("arm") in ARMS:
        return ROOT / "trained_models/SW0134_native_spike_binding" / f"{task['arm']}_seed1"
    if task.get("stage") == "evaluate" and task.get("arm") == "all_three":
        return ARCHIVE / "evaluation_seed1"
    if task.get("stage") == "diagnostic" and task.get("arm") == "all_three":
        return ARCHIVE / "train_diagnostic_seed1.json"
    raise ValueError("unregistered SW0134 pilot task")


def command(task, device="cuda:0"):
    if task["stage"] == "train":
        return [sys.executable, "-m", "collaborative_test.SW_0134_native_spike_binding.train", "--stage", "train",
                "--seed", "1", "--arm", task["arm"], "--device", device,
                "--output-root", str(ROOT / "trained_models/SW0134_native_spike_binding"),
                "--archive", str(ARCHIVE)]
    if task["stage"] == "diagnostic":
        return [sys.executable, "-m", "collaborative_test.SW_0134_native_spike_binding.train_diagnostic", "--seed", "1",
                "--device", device, "--output", str(artifact_path(task))]
    return [sys.executable, "-m", "collaborative_test.SW_0134_native_spike_binding.evaluate", "--seed", "1",
            "--device", device, "--output", str(artifact_path(task)),
            "--training-root", str(ROOT / "trained_models/SW0134_native_spike_binding")]


def valid_result(task):
    """Validate completed artifacts from the actual train/evaluation producers."""
    try:
        if task["stage"] == "train":
            folder = artifact_path(task)
            manifest, _state, _hashes = train.validate_completed_training(
                1, task["arm"], folder)
            return (manifest.get("status") == "training_complete"
                    and manifest.get("trainer_sha256") == run.sha(HERE / "train.py"))
        if task["stage"] == "diagnostic":
            path = artifact_path(task)
            report = json.loads(path.read_text(encoding="utf-8"))
            assets = run.sw130.validate_rgb_assets()
            pool, ids = run.source_contract(1)[3:5]
            expected_code = {str((HERE / name).relative_to(ROOT).as_posix()):
                             run.sha(HERE / name)
                             for name in ("run.py", "binder.py", "rollout.py",
                                          "train.py", "train_diagnostic.py")}
            if (report.get("status") != "complete" or report.get("seed") != 1
                    or report.get("stage") != "post_training_train_only_readout_diagnostic"
                    or report.get("ground_truth_used") is not False
                    or report.get("optimizer_updates") != 0
                    or set(report.get("diagnostics", {})) != set(ARMS)
                    or report.get("image_ids") != [int(v) for v in ids[:run.BATCH]]
                    or report.get("train_cache_sha256") != assets["train_cache_sha256"]
                    or report.get("implementation") != expected_code):
                return False
            for arm, row in report["diagnostics"].items():
                manifest, _state, checkpoint = train.validate_completed_training(
                    1, arm, ROOT / "trained_models/SW0134_native_spike_binding" / f"{arm}_seed1",
                    assets=assets)
                if (row.get("ground_truth_used") is not False
                        or row.get("optimizer_updates") != 0
                        or row.get("checkpoint_sha256") != run.sha(checkpoint)
                        or row.get("source_core_sha256") != manifest.get("source_core_sha256")
                        or len(row.get("per_image_rgb_mse", [])) != run.BATCH
                        or len(row.get("per_image_slot_column_scramble_rgb_mse", [])) != run.BATCH
                        or not all(math.isfinite(float(x)) for x in row["per_image_rgb_mse"]
                                   + row["per_image_slot_column_scramble_rgb_mse"])
                        or len(row.get("mean_slot_probability_occupancy", [])) != 11
                        or not all(math.isfinite(float(x)) for x in
                                   row["mean_slot_probability_occupancy"])):
                    return False
            return True
        path = artifact_path(task) / "evaluation.json"
        if not path.is_file():
            return False
        report = json.loads(path.read_text(encoding="utf-8"))
        if (report.get("status") != "complete"
                or report.get("experiment") != "SW0134_native_spike_binding"
                or report.get("seed") != 1 or report.get("count") != 320
                or report.get("image_ids") != [1320, 1639]
                or report.get("ground_truth_used_for_prediction") is not False
                or report.get("ground_truth_used_for_scoring") is not True):
            return False
        predictions = pathlib.Path(report["prediction_path"])
        manifest_path = predictions.parent / "prediction_manifest.json"
        evaluation_manifest_path = path.parent / "evaluation_manifest.json"
        if (not predictions.is_file() or not manifest_path.is_file()
                or not evaluation_manifest_path.is_file()
                or run.sha(predictions) != report.get("prediction_sha256")
                or run.sha(manifest_path) != report.get("prediction_manifest_sha256")
                or run.sha(path) != json.loads(evaluation_manifest_path.read_text(
                    encoding="utf-8")).get("evaluation_sha256")):
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        evaluation_manifest = json.loads(evaluation_manifest_path.read_text(encoding="utf-8"))
        expected_impl = {**run.implementation_fingerprint(),
                         (HERE / "train.py").relative_to(ROOT).as_posix(): run.sha(HERE / "train.py"),
                         (HERE / "evaluate.py").relative_to(ROOT).as_posix(): run.sha(HERE / "evaluate.py")}
        source_path, _source_values = evaluate._source_reference(1)
        if (manifest.get("seed") != 1
                or manifest.get("prediction_sha256") != report["prediction_sha256"]):
            return False
        if (evaluation_manifest.get("experiment") != "SW0134_native_spike_binding"
                or evaluation_manifest.get("seed") != 1
                or evaluation_manifest.get("prediction_sha256") != report["prediction_sha256"]
                or evaluation_manifest.get("source97_evaluation_sha256") != run.sha(source_path)
                or evaluation_manifest.get("implementation_fingerprint") != expected_impl
                or report.get("implementation_fingerprint") != expected_impl
                or manifest.get("image_ids") != [1320, 1639] or manifest.get("count") != 320
                or manifest.get("arms") != ["source97_native", *ARMS]
                or manifest.get("ground_truth_used_for_prediction") is not False
                or report.get("source97_evaluation_sha256") != run.sha(source_path)):
            return False
        stored = __import__("torch").load(predictions, map_location="cpu", weights_only=True)
        if (stored.get("image_ids") != list(range(1320, 1640))
                or set(stored.get("predictions", {})) != {"source97_native", *ARMS}
                or any(tuple(value.shape) != (320, 16, 16)
                       for bundle in stored["predictions"].values()
                       for value in bundle.values())):
            return False
        provenance = report.get("arm_provenance", {})
        if set(provenance) != {"source97_native", *ARMS}:
            return False
        if provenance["source97_native"].get("checkpoint_sha256") != run.source97.EXPECTED_SOURCE_SHAS[1]:
            return False
        for arm in ARMS:
            folder = ROOT / "trained_models/SW0134_native_spike_binding" / f"{arm}_seed1"
            training_manifest, _state, checkpoint = train.validate_completed_training(1, arm, folder)
            item = provenance[arm]
            if (item.get("checkpoint_sha256") != run.sha(checkpoint)
                    or item.get("source_core_sha256") != training_manifest.get("source_core_sha256")
                    or item.get("ground_truth_used_for_prediction") is not False):
                return False
        scores = report["scores"]
        expected_arms = {"source97_native", *ARMS}
        if set(scores) != expected_arms:
            return False
        for arm in expected_arms:
            for readout in ("primary", "qcc"):
                for metric in REPORT_METRICS:
                    row = scores[arm][readout][metric]
                    values = row.get("per_image")
                    mean = float(row.get("mean", float("nan")))
                    if (row.get("valid_count") != 320 or not isinstance(values, list)
                            or len(values) != 320
                            or not all(math.isfinite(float(x)) for x in values)
                            or not math.isfinite(mean)
                            or abs(float(np.mean(values)) - mean) > 1e-10):
                        return False
        if report.get("source97_qcc_per_image_reproduction", {}).get("passed") is not True:
            return False
        return True
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, AssertionError,
            OverflowError, json.JSONDecodeError):
        return False


def _atomic(path, value):
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


class PilotQueue:
    def __init__(self):
        if fcntl is None:
            raise RuntimeError("SW0134 GPU pilot queue requires Linux flock")
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        self.lock_file = LOCK.open("a+")
        try:
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock_file.close()
            raise RuntimeError("another SW0134 pilot queue owns its lock") from exc
        try:
            if STATE.exists():
                raise FileExistsError(f"preserve existing pilot state: {STATE}")
            assets = run.sw130.validate_rgb_assets()
            for seed in run.SEEDS:
                if not train.preflight_queue.valid_result(
                        {"stage": "preflight", "seed": seed}, assets):
                    raise RuntimeError(f"SW0134 seed{seed} preflight is missing or invalid")
            # Refuse every destination before creating queue state or launching a child.
            for task in task_plan():
                target = artifact_path(task)
                if target.exists() or ARCHIVE.joinpath(task["task_id"] + ".log").exists():
                    raise FileExistsError(f"preserve existing pilot artifact/log: {target}")
        except BaseException:
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
            self.lock_file.close()
            raise
        self.tasks = task_plan()
        self.guard = threading.RLock()
        self.state = {"experiment": "SW0134_native_spike_binding", "stage": "paired_seed1_pilot",
                      "status": "running", "supervisor_pid": os.getpid(),
                      "runner_fingerprint": run.implementation_fingerprint(),
                      "queue_sha256": run.sha(HERE / "pilot_queue.py"),
                      "trainer_sha256": run.sha(HERE / "train.py"),
                      "evaluator_sha256": run.sha(HERE / "evaluate.py"),
                      "owner_dispatcher_sha256": run.sha(owner.__file__),
                      "max_parallel_train": MAX_PARALLEL_TRAIN,
                      "no_retries": True,
                      "tasks": {t["task_id"]: {"task": t, "status": "queued"}
                                for t in self.tasks}, "started": time.time()}
        _atomic(STATE, self.state)

    def _update(self, task_id, **fields):
        with self.guard:
            self.state["tasks"][task_id].update(fields)
            _atomic(STATE, self.state)

    def _reserve(self, task_id):
        while True:
            for gpu in range(4):
                path = LEASES / f"gpu{gpu}.lock"
                path.parent.mkdir(parents=True, exist_ok=True)
                lease = path.open("a+")
                try:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lease.close()
                    continue
                try:
                    uuid, memory, owners = owner._nvidia_gpu_info(gpu)
                    if not owners and memory <= MAX_MEMORY_MIB:
                        self._update(task_id, status="reserved", gpu=gpu, gpu_uuid=uuid,
                                     memory_before_mib=memory, lease_path=str(path),
                                     reservation_pid=os.getpid())
                        return gpu, lease, uuid
                except BaseException:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                    lease.close()
                    raise
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                lease.close()
            self._update(task_id, status="waiting_for_exclusive_gpu", last_wait=time.time())
            time.sleep(5)

    @staticmethod
    def _stop_owned(proc):
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
        key = task["task_id"]
        artifact = artifact_path(task)
        log_path = ARCHIVE / f"{key}.log"
        if artifact.exists() or log_path.exists():
            self._update(key, status="failed_existing_artifact", finished=time.time())
            return False
        lease = proc = None
        try:
            gpu, lease, uuid = self._reserve(key)
            triton = pathlib.Path(f"/tmp/kevinswk_sw0134_{key}_{int(time.time()*1000)}")
            triton.mkdir(parents=True, exist_ok=False)
            argv = command(task)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", TRITON_CACHE_DIR=str(triton))
            self._update(key, status="launching", argv=argv, gpu=gpu, gpu_uuid=uuid,
                         log_path=str(log_path), triton_cache=str(triton), launched=time.time())
            with log_path.open("x", encoding="utf-8") as stream:
                proc = subprocess.Popen(argv, cwd=str(ROOT), env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, start_new_session=True,
                                        pass_fds=(lease.fileno(),))
                self._update(key, status="running", child_pid=proc.pid, started=time.time())
                while proc.poll() is None:
                    time.sleep(5)
                    if proc.poll() is not None:
                        break
                    _uuid, _mem, owners = owner._nvidia_gpu_info(gpu)
                    foreign = owner.foreign_owner_pids(owners, owner._process_tree(proc.pid))
                    if foreign:
                        self._stop_owned(proc)
                        self._update(key, status="interrupted_foreign_gpu_owner",
                                     foreign_owner_pids=foreign, finished=time.time())
                        return False
            rc = int(proc.returncode)
            passed = rc == 0 and valid_result(task)
            self._update(key, status="passed" if passed else "failed", returncode=rc,
                         artifact_valid=bool(passed), finished=time.time())
            return passed
        except BaseException as exc:
            if proc is not None:
                self._stop_owned(proc)
            self._update(key, status="failed", error=repr(exc), finished=time.time())
            return False
        finally:
            if lease is not None:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                lease.close()

    def _finish(self, status):
        with self.guard:
            self.state.update(status=status, finished=time.time())
            _atomic(STATE, self.state)

    def run(self):
        train_tasks = self.tasks[:3]
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_TRAIN) as pool:
            futures = {pool.submit(self._execute, task): task for task in train_tasks}
            outcomes = {futures[f]["task_id"]: f.result() for f in futures}
        if not all(outcomes.values()):
            self._update(self.tasks[3]["task_id"], status="blocked_training_failure")
            self._finish("pilot_training_failed")
            return False
        if not all(valid_result(task) for task in train_tasks):
            self._update(self.tasks[3]["task_id"], status="blocked_training_validation")
            self._finish("pilot_training_artifact_invalid")
            return False
        diagnostic = self.tasks[3]
        if not self._execute(diagnostic):
            self._update(self.tasks[4]["task_id"], status="blocked_diagnostic_failure")
            self._finish("pilot_diagnostic_failed")
            return False
        evaluation = self.tasks[4]
        if not self._execute(evaluation):
            self._finish("pilot_evaluation_failed")
            return False
        report_path = artifact_path(evaluation) / "evaluation.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        gate = registered_pilot_gate(report)
        gate.update({"evaluation_path": str(report_path.resolve()),
                     "evaluation_sha256": run.sha(report_path),
                     "prediction_sha256": report["prediction_sha256"]})
        gate_path = ARCHIVE / "pilot_gate_seed1.json"
        run.write_once(gate_path, gate)
        self.state.update(status=gate["status"], pilot_gate_path=str(gate_path.resolve()),
                          pilot_gate_sha256=run.sha(gate_path), finished=time.time())
        _atomic(STATE, self.state)
        return True


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"experiment": "SW0134_native_spike_binding",
                          "stage": "paired_seed1_pilot", "max_parallel_train": MAX_PARALLEL_TRAIN,
                          "no_retries": True,
                          "tasks": [{**task, "argv": command(task)} for task in task_plan()]},
                         indent=2, allow_nan=False))
        return
    PilotQueue().run()


if __name__ == "__main__":
    main()
