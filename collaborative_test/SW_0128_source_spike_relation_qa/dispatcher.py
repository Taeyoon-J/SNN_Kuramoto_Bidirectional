"""Small owner-aware two-stage dispatcher for SW0128 frozen QA."""
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
except ImportError:  # Queue mechanics are Linux-only; tests inspect the plan on Windows.
    fcntl = None

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
RUNTIME = pathlib.Path("/tmp/kevinswk_sw0128_source_spike_relation_qa")
LEASE_DIR = pathlib.Path("/tmp/kevinswk_sw0113_gpu_leases")
QUEUE_LOCK = RUNTIME / "queue.lock"
STATE = RUNTIME / "state.json"
OUT = HERE / "results_archive" / "source_relation_qa_20261009"
MAX_MEMORY_MIB = 512
PYTHON = "/Data0/kevinswk/envs/snn/bin/python"

sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]
from collaborative_test.SW_0126_history_event_binding import pilot_queue as resource_guard
from collaborative_test.SW_0128_source_spike_relation_qa import run as qa


def task_plan():
    return [
        {"task_id": "sw0128_source_qcc_generate_all3", "stage": "generate",
         "output": str(OUT), "depends_on": []},
        {"task_id": "sw0128_source_qcc_score_frozen", "stage": "score",
         "output": str(OUT), "depends_on": ["sw0128_source_qcc_generate_all3"]},
    ]


def gpu_is_exclusive(memory_mib, owner_pids):
    return int(memory_mib) <= MAX_MEMORY_MIB and not owner_pids


def valid_result(task):
    output = pathlib.Path(task["output"])
    try:
        if task["stage"] == "generate":
            pred = output / "frozen_predictions.pt"
            manifest_path = output / "prediction_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            return (manifest.get("status") == "predictions_complete"
                    and manifest.get("experiment") == "SW0128"
                    and manifest.get("image_ids") == [1320, 1335]
                    and manifest.get("ground_truth_used_for_prediction") is False
                    and manifest.get("optimizer_updates") == 0
                    and manifest.get("implementation_fingerprint") == qa._implementation_fingerprint()
                    and manifest.get("prediction_sha256") == qa.sha256_file(pred)
                    and set(manifest.get("native_rollout_parity", {})) == {"0", "1", "2"}
                    and all(all(row.get("trace_exact", {}).values())
                            and row.get("qcc_labels_exact") is True
                            for row in manifest["native_rollout_parity"].values())
                    and set(manifest.get("source_provenance", {})) == {"0", "1", "2"})
        if task["stage"] == "score":
            report_path = output / "qa_results.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            manifest = json.loads((output / "prediction_manifest.json").read_text(encoding="utf-8"))
            return (report.get("status") == "complete"
                    and report.get("experiment") == "SW0128"
                    and report.get("image_ids") == [1320, 1335]
                    and report.get("count") == 16
                    and report.get("ground_truth_used_for_prediction") is False
                    and report.get("ground_truth_used_for_scoring") is True
                    and report.get("all_three_prediction_bundles_sha_verified_before_ground_truth") is True
                    and report.get("prediction_sha256") == manifest.get("prediction_sha256")
                    and report.get("implementation_fingerprint") == qa._implementation_fingerprint()
                    and set(report.get("per_seed", {})) == {"0", "1", "2"})
    except (OSError, ValueError, TypeError, KeyError):
        return False
    return False


class SourceRelationDispatcher:
    def __init__(self):
        if fcntl is None:
            raise RuntimeError("SW0128 dispatcher requires Linux flock")
        RUNTIME.mkdir(parents=True, exist_ok=True)
        self.lock = QUEUE_LOCK.open("a+")
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock.close()
            raise RuntimeError("another SW0128 dispatcher owns the queue") from exc
        if STATE.exists():
            self.lock.close()
            raise FileExistsError(f"preserving prior SW0128 state: {STATE}")
        self.tasks = task_plan()
        self.state = {"experiment": "SW0128", "status": "running",
            "supervisor_pid": os.getpid(), "no_automatic_retry": True,
            "tasks": {row["task_id"]: {"task": row, "status": "queued"} for row in self.tasks}}
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

    def _acquire_gpu(self):
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
    def _argv(task, device="cpu"):
        args = ["--stage", task["stage"], "--output", task["output"]]
        if task["stage"] == "generate":
            args.extend(("--device", device))
        return [PYTHON, "-m", "collaborative_test.SW_0128_source_spike_relation_qa.run", *args]

    @staticmethod
    def _stop_owned(process):
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()

    def _execute(self, task):
        task_id, output = task["task_id"], pathlib.Path(task["output"])
        if valid_result(task):
            self._update(task_id, status="verified_existing")
            return True
        if task["stage"] == "score" and not valid_result(self.tasks[0]):
            self._update(task_id, status="blocked_invalid_predictions")
            return False
        if task["stage"] == "generate" and output.exists():
            self._update(task_id, status="failed_existing_output_preserved")
            return False
        gpu = gpu_uuid = memory = lease = None
        if task["stage"] == "generate":
            gpu, gpu_uuid, memory, lease = self._acquire_gpu()
        process = None
        log = RUNTIME / f"{task_id}.log"
        triton = pathlib.Path(f"/tmp/kevinswk_sw0128_triton_{task_id}")
        try:
            if log.exists():
                raise FileExistsError(f"preserving prior SW0128 task log: {log}")
            if task["stage"] == "generate":
                if triton.exists():
                    raise FileExistsError(f"preserving prior SW0128 Triton cache: {triton}")
                triton.mkdir(mode=0o700, parents=False, exist_ok=False)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                           MKL_NUM_THREADS="1", TRITON_CACHE_DIR=str(triton))
                device = "cuda:0"
            else:
                env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                           MKL_NUM_THREADS="1")
                device = "cpu"
            argv = self._argv(task, device)
            with log.open("x", encoding="utf-8") as stream:
                process = subprocess.Popen(argv, cwd=ROOT, env=env,
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=True,
                    pass_fds=(() if lease is None else (lease.fileno(),)))
                self._update(task_id, status="running", child_pid=process.pid, argv=argv,
                    log=str(log), gpu=gpu, gpu_uuid=gpu_uuid,
                    memory_used_before_mib=memory, inherited_lease_fd=lease is not None,
                    triton_cache_dir=str(triton) if task["stage"] == "generate" else None)
                foreign = []
                while process.poll() is None:
                    time.sleep(5)
                    if process.poll() is not None or lease is None:
                        break
                    _uuid, _used, owners = resource_guard._gpu_info(gpu)
                    foreign = resource_guard._foreign_owners(owners, resource_guard._process_tree(process.pid))
                    if foreign:
                        self._update(task_id, status="interrupted_foreign_gpu_owner",
                                     foreign_owner_pids=foreign)
                        self._stop_owned(process)
                        break
                rc = process.wait()
                passed = rc == 0 and not foreign and valid_result(task)
                self._update(task_id, status="passed" if passed else "failed",
                             returncode=rc, artifact_valid=valid_result(task), finished=time.time())
                return passed
        except BaseException as exc:
            if process is not None and process.poll() is None:
                self._stop_owned(process)
            self._update(task_id, status="failed", error=repr(exc), finished=time.time())
            return False
        finally:
            if lease is not None:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                lease.close()

    def run(self):
        success = True
        for task in self.tasks:
            if not success:
                break
            success = self._execute(task)
        self.state["status"] = "complete" if success else "failed"
        self.state["finished"] = time.time()
        self._save()
        self.lock.close()
        return success


if __name__ == "__main__":
    raise SystemExit(0 if SourceRelationDispatcher().run() else 1)
