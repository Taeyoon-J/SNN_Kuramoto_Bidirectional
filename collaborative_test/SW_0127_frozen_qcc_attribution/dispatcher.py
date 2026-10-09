"""Owner-aware, no-retry preflight then frozen-evaluation dispatcher."""
from __future__ import annotations

import json
import os
import pathlib
import signal
import subprocess
import sys
import time

try:
    import fcntl
except ImportError:  # Unit tests can inspect the queue plan on Windows.
    fcntl = None

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
RUNTIME = pathlib.Path("/tmp/kevinswk_sw0127_qcc_attribution")
LEASE_DIR = pathlib.Path("/tmp/kevinswk_sw0113_gpu_leases")
QUEUE_LOCK = RUNTIME / "queue.lock"
STATE = RUNTIME / "state.json"
MAX_MEMORY_MIB = 512
PYTHON = "/Data0/kevinswk/envs/snn/bin/python"
PREFLIGHT_OUTPUT = HERE / "results_archive" / "preflight_seed1_20261009.json"
EVALUATION_OUTPUT = HERE / "results_archive" / "qcc_attribution_seed1_20261009"

sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]
from collaborative_test.SW_0126_history_event_binding import pilot_queue as resource_guard
from collaborative_test.SW_0127_frozen_qcc_attribution import run as evaluator


def task_plan():
    return [
        {"task_id": "sw0127_seed1_factory_preflight", "stage": "preflight",
         "output": str(PREFLIGHT_OUTPUT), "depends_on": []},
        {"task_id": "sw0127_seed1_frozen_qcc", "stage": "evaluate",
         "output": str(EVALUATION_OUTPUT), "depends_on": ["sw0127_seed1_factory_preflight"]},
    ]


def gpu_is_exclusive(memory_mib, owner_pids):
    return int(memory_mib) <= MAX_MEMORY_MIB and not owner_pids


def _finite_scores(scores):
    metrics = ("fg_ari", "foreground_iou", "matched_object_iou")
    if not isinstance(scores, dict):
        return False
    for arm in ("source97_native", "source97_untrained_history_adapter",
                "sw0126_history_event", "sw0126_gate_only"):
        score = scores.get(arm)
        if not isinstance(score, dict):
            return False
        for metric in metrics:
            row = score.get(metric, {})
            values = row.get("per_image")
            if (row.get("valid_count") != 320 or not isinstance(values, list)
                    or len(values) != 320):
                return False
            try:
                if (not __import__("math").isfinite(float(row.get("mean", float("nan"))))
                        or not all(__import__("math").isfinite(float(value)) for value in values)):
                    return False
            except (TypeError, ValueError):
                return False
    return True


def valid_result(task):
    path = pathlib.Path(task["output"])
    try:
        if task["stage"] == "preflight":
            report = json.loads(path.read_text(encoding="utf-8"))
            parity = report.get("native_rollout_parity", {})
            return (report.get("status") == "passed"
                    and report.get("experiment") == "SW0127"
                    and report.get("ground_truth_used") is False
                    and report.get("optimizer_updates") == 0
                    and report.get("implementation_fingerprint") == evaluator._implementation_fingerprint()
                    and len(parity.get("batches", [])) == 2
                    and len(parity.get("four_arm_validation_smokes", [])) == 4
                    and all(all(row.get("traces_exact", {}).values())
                            and row.get("qcc_labels_exact") is True
                            for row in parity["batches"])
                    and all(row.get("count") == 2
                            and row.get("actual_spikes_finite") is True
                            and tuple(row.get("qcc_labels_shape", ())) == (2, 16, 16)
                            for row in parity["four_arm_validation_smokes"]))
        if task["stage"] == "evaluate":
            report_path = path / "evaluation.json"
            manifest_path = path / "evaluation_manifest.json"
            prediction_path = path / "frozen_predictions.pt"
            prediction_manifest_path = path / "prediction_manifest.json"
            if not all(item.is_file() for item in (report_path, manifest_path,
                                                   prediction_path, prediction_manifest_path)):
                return False
            report = json.loads(report_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            prediction = json.loads(prediction_manifest_path.read_text(encoding="utf-8"))
            digest = resource_guard.pilot_run.sha(prediction_path)
            return (report.get("status") == "complete"
                    and report.get("experiment") == "SW0127"
                    and report.get("image_ids") == [1320, 1639]
                    and report.get("count") == 320
                    and report.get("ground_truth_used_for_prediction") is False
                    and report.get("source97_per_image_reproduction", {}).get("passed") is True
                    and _finite_scores(report.get("scores"))
                    and report.get("implementation_fingerprint") == evaluator._implementation_fingerprint()
                    and manifest.get("evaluation_sha256") == resource_guard.pilot_run.sha(report_path)
                    and manifest.get("prediction_sha256") == digest
                    and prediction.get("prediction_sha256") == digest
                    and prediction.get("ground_truth_used_for_prediction") is False
                    and prediction.get("image_ids") == [1320, 1639]
                    and prediction.get("count") == 320)
    except (OSError, ValueError, TypeError, KeyError):
        return False
    return False


def _process_tree(root_pid):
    return resource_guard._process_tree(root_pid)


def _foreign_owners(owners, owned_tree):
    return resource_guard._foreign_owners(owners, owned_tree)


class FrozenQCCDispatcher:
    def __init__(self):
        if fcntl is None:
            raise RuntimeError("SW0127 dispatcher requires Linux flock")
        RUNTIME.mkdir(parents=True, exist_ok=True)
        self.lock_stream = QUEUE_LOCK.open("a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock_stream.close()
            raise RuntimeError("another SW0127 dispatcher owns the queue lock") from exc
        if STATE.exists():
            self.lock_stream.close()
            raise FileExistsError(f"preserving prior SW0127 queue state: {STATE}")
        self.tasks = task_plan()
        self.state = {"experiment": "SW0127", "status": "running",
                      "supervisor_pid": os.getpid(), "no_automatic_retry": True,
                      "tasks": {task["task_id"]: {"task": task, "status": "queued"}
                                for task in self.tasks}}
        self._save()

    def _save(self):
        tmp = STATE.with_suffix(".json.tmp")
        with tmp.open("x", encoding="utf-8") as stream:
            json.dump(self.state, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, STATE)

    def _update(self, task_id, **fields):
        self.state["tasks"][task_id].update(fields)
        self._save()

    def _acquire(self):
        while True:
            for gpu in range(4):
                lease_path = LEASE_DIR / f"gpu{gpu}.lock"
                lease_path.parent.mkdir(parents=True, exist_ok=True)
                lease = lease_path.open("a+")
                try:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lease.close()
                    continue
                try:
                    gpu_uuid, memory, owners = resource_guard._gpu_info(gpu)
                    if gpu_is_exclusive(memory, owners):
                        return gpu, gpu_uuid, memory, lease
                except BaseException:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                    lease.close()
                    raise
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                lease.close()
            time.sleep(5)

    @staticmethod
    def _argv(task):
        command = ["--output", task["output"], "--device", "cuda:0"]
        if task["stage"] == "preflight":
            command.append("--preflight-only")
        return [PYTHON, "-m", "collaborative_test.SW_0127_frozen_qcc_attribution.run", *command]

    @staticmethod
    def _stop_owned(proc):
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()

    def _execute(self, task):
        task_id = task["task_id"]
        if valid_result(task):
            self._update(task_id, status="verified_existing")
            return True
        output = pathlib.Path(task["output"])
        if output.exists():
            self._update(task_id, status="failed_existing_output_preserved")
            return False
        gpu, gpu_uuid, memory, lease = self._acquire()
        process = None
        log = RUNTIME / f"{task_id}.log"
        triton = pathlib.Path(f"/tmp/kevinswk_sw0127_triton_{task_id}")
        try:
            if log.exists() or triton.exists():
                raise FileExistsError("preserving prior SW0127 log/Triton cache")
            triton.mkdir(mode=0o700, parents=False, exist_ok=False)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", TRITON_CACHE_DIR=str(triton))
            argv = self._argv(task)
            with log.open("x", encoding="utf-8") as stream:
                process = subprocess.Popen(argv, cwd=ROOT, env=env,
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=True, pass_fds=(lease.fileno(),))
                self._update(task_id, status="running", child_pid=process.pid,
                    gpu=gpu, gpu_uuid=gpu_uuid, memory_used_before_mib=memory,
                    argv=argv, log=str(log), triton_cache_dir=str(triton),
                    inherited_lease_fd=True)
                foreign = []
                while process.poll() is None:
                    time.sleep(5)
                    if process.poll() is not None:
                        break
                    _uuid, _used, owners = resource_guard._gpu_info(gpu)
                    foreign = _foreign_owners(owners, _process_tree(process.pid))
                    if foreign:
                        self._update(task_id, status="interrupted_foreign_gpu_owner",
                                     foreign_owner_pids=foreign)
                        self._stop_owned(process)
                        break
                returncode = process.wait()
                passed = returncode == 0 and not foreign and valid_result(task)
                self._update(task_id, status="passed" if passed else "failed",
                             returncode=returncode, artifact_valid=valid_result(task),
                             finished=time.time())
                return passed
        except BaseException as exc:
            if process is not None and process.poll() is None:
                self._stop_owned(process)
            self._update(task_id, status="failed", error=repr(exc), finished=time.time())
            return False
        finally:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
            lease.close()

    def run(self):
        success = True
        for task in self.tasks:
            if any(not self.state["tasks"][dep]["status"] in
                   {"passed", "verified_existing"} for dep in task["depends_on"]):
                self._update(task["task_id"], status="blocked_failed_dependency")
                success = False
                break
            if not self._execute(task):
                success = False
                break
        self.state["status"] = "complete" if success else "failed"
        self.state["finished"] = time.time()
        self._save()
        return success


def main():
    queue = FrozenQCCDispatcher()
    raise SystemExit(0 if queue.run() else 1)


if __name__ == "__main__":
    main()
