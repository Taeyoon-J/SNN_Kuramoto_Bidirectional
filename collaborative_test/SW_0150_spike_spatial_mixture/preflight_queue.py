"""One-task owner-aware SW0150 source preflight queue; never starts training."""
from __future__ import annotations

import argparse
import hashlib
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
except ImportError:  # Windows is for CPU tests/help only; server must own Linux flock.
    fcntl = None

# Keep this supervisor CPU-only. The child sees only the reserved physical GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4 as owner_queue
from collaborative_test.SW_0150_spike_spatial_mixture import preflight


RUNTIME = HERE / "results_archive" / "preflight_queue_20261012"
STATE = RUNTIME / "state.json"
LOCK = RUNTIME / "queue.lock"
OUTPUT = HERE / "results_archive" / "preflight_seed1.json"
LOG = RUNTIME / "preflight_seed1.log"
PRIVATE_RUNTIME = Path(os.environ.get(
    "SW0150_PRIVATE_RUNTIME",
    "/Data0/kevinswk/collaborative_private_runtime_20261008/SW0150_preflight_20261012"))
WARM_ARTIFACT = PRIVATE_RUNTIME / "decoder_warm_seed1.pt"
PARITY = preflight.PARITY_PATH
MEMORY_LIMIT_MIB = 512
OWNER_HELPER = ROOT / "collaborative_test/SW_0126_history_event_binding/pilot_queue.py"
OWNER_HELPER_SHA256 = owner_queue.OWNER_GUARD_SHA256


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint():
    """Bind queue and every local/upstream execution dependency by source bytes."""
    paths = (
        HERE / "preflight_queue.py", HERE / "preflight.py", HERE / "protocol.json",
        HERE / "integration.py", HERE / "spatial_diffusion.py",
        HERE / "seed_selection.py", HERE / "likelihood.py",
        ROOT / "collaborative_test/SW_0135_native32_spike_binding/foundation.py",
        ROOT / "collaborative_test/SW_0135_native32_spike_binding/parity.py",
        ROOT / "collaborative_test/SW_0136_native32_transfer/evaluate.py",
        ROOT / "collaborative_test/SW_0145_native32_boundary_credit_probe/run.py",
        ROOT / "collaborative_test/SW_0148_checkpointable_boundary_q/stage_queue_v4.py",
        OWNER_HELPER,
    )
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("an SW0150 queue/preflight dependency is missing")
    if sha256_file(OWNER_HELPER) != OWNER_HELPER_SHA256:
        raise ValueError("the registered GPU ownership guard changed")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def task_plan():
    return [{"task_id": "preflight_seed1", "stage": "preflight", "seed": 1,
             "device": "cuda:0", "depends_on": [],
             "output": str(OUTPUT), "warm_output": str(WARM_ARTIFACT),
             "parity": str(PARITY), "log": str(LOG)}]


def command(task, device="cuda:0"):
    if (task.get("stage") != "preflight" or task.get("seed") != 1
            or task.get("depends_on") != []):
        raise ValueError("only the registered source1 preflight stage is allowed")
    return [sys.executable, "-m", "collaborative_test.SW_0150_spike_spatial_mixture.preflight",
            "--device", device, "--output", str(task["output"]),
            "--warm-output", str(task["warm_output"]),
            "--parity", str(task["parity"])]


def check_no_existing_artifacts(paths):
    existing = [str(Path(path)) for path in paths if Path(path).exists()]
    if existing:
        raise FileExistsError(f"preserve existing SW0150 preflight markers; no restart: {existing}")


def _finite_number(value):
    try:
        import math
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def validate_report(report, *, warm_artifact=WARM_ARTIFACT,
                    warm_digest=None, expected_fingerprint=None):
    """Validate complete preflight and warm artifact before marking queue passed."""
    try:
        if expected_fingerprint is None:
            expected_fingerprint = preflight.implementation_fingerprint()
        warm_artifact = Path(warm_artifact)
        warm = report["warm_checkpoint"]
        source = report["source"]
        parity = report["saved_source_parity"]
        if warm_digest is None:
            warm_digest = sha256_file(warm_artifact)
        required_asset_fields = {"train_cache_sha256", "train_manifest_sha256",
                                 "validation_cache_sha256", "validation_manifest_sha256",
                                 "encoder_source_sha256", "feature_preprocessing_sha256"}
        assets = source["rgb_asset_validation"]
        ids = source["training_ids"]
        if (report.get("experiment") != preflight.EXPERIMENT
                or report.get("stage") != "source1_shared_warm_and_disposable_preflight"
                or report.get("status") != "source_preflight_complete"
                or report.get("seed") != 1
                or report.get("training_admission") is not False
                or report.get("ground_truth_used") is not False
                or report.get("masks_read") is not False
                or report.get("optimizer_updates") != {
                    "shared_decoder_warmup": 32, "encoder_graph": 2, "joint": 2}
                or report.get("exposures") != {
                    "warmup_images": 512, "each_disposable_arm_images": 32,
                    "full_native32_70000_image_training": False}
                or report.get("implementation_fingerprint") != expected_fingerprint
                or len(ids) != 4096 or len(set(ids)) != 4096
                or source.get("training_ids_sha256") != preflight._ids_sha(ids)
                or not isinstance(assets, dict) or not required_asset_fields.issubset(assets)
                or not isinstance(source.get("checkpoint_sha256"), str)
                or len(source["checkpoint_sha256"]) != 64
                or not isinstance(source.get("manifest_sha256"), str)
                or len(source["manifest_sha256"]) != 64
                or parity.get("path") != str(PARITY)
                or parity.get("sha256") != sha256_file(PARITY)
                or parity.get("status") != "native32_zero_adapter_parity_passed"
                or parity.get("source1_actual_source_exact_T1024_settle512") is not True
                or warm.get("path") != str(warm_artifact)
                or warm.get("sha256") != warm_digest
                or not warm_artifact.is_file()
                or warm.get("shared_exactly_across_disposable_arms") is not True
                or warm.get("optimizer_state_steps", {}).get("maximum") != 32
                or not warm.get("optimizer_state_steps", {}).get("count", 0)):
            return False
        if report.get("likelihood", {}).get("sigma2", 0) <= 0 or not _finite_number(
                report["likelihood"]["sigma2"]):
            return False
        arms = report.get("arms")
        if not isinstance(arms, dict) or set(arms) != {"encoder_graph", "joint"}:
            return False
        expected_groups = {
            "encoder_graph": {"encoder", "graph", "native_core", "integration_scalars", "decoder"},
            "joint": {"encoder", "graph", "native_core", "integration_scalars", "decoder"},
        }
        for name, arm in arms.items():
            if (arm.get("status") != "disposable_two_updates_complete"
                    or arm.get("optimizer_updates") != 2
                    or len(arm.get("history", [])) != 2
                    or set(arm.get("initial_parameter_group_sha256", {})) != expected_groups[name]
                    or set(arm.get("final_parameter_group_sha256", {})) != expected_groups[name]
                    or set(arm.get("changed_parameter_groups", {})) != expected_groups[name]
                    or len(arm.get("image_ids", [])) != 32
                    or not _finite_number(arm.get("final_mean_nll"))):
                return False
            for row in arm["history"]:
                if (not _finite_number(row.get("mean_nll"))
                        or set(row.get("family_gradient_norms", {})) != expected_groups[name]
                        or any(not _finite_number(value)
                               for value in row["family_gradient_norms"].values())):
                    return False
        return True
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False


def valid_result(task=None):
    task = task or task_plan()[0]
    output, warm = Path(task["output"]), Path(task["warm_output"])
    if not output.is_file() or not warm.is_file():
        return False
    try:
        report = json.loads(output.read_text(encoding="utf-8"))
        return validate_report(report, warm_artifact=warm)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _atomic_json(path, payload):
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


class PreflightQueue:
    def __init__(self):
        if fcntl is None:
            raise RuntimeError("SW0150 GPU queue requires Linux flock")
        RUNTIME.mkdir(parents=True, exist_ok=True)
        self.queue_lock = LOCK.open("a+")
        try:
            fcntl.flock(self.queue_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.queue_lock.close()
            raise RuntimeError("another SW0150 preflight supervisor owns the queue") from exc
        self.tasks = task_plan()
        try:
            check_no_existing_artifacts([STATE, OUTPUT, LOG, WARM_ARTIFACT])
        except BaseException:
            fcntl.flock(self.queue_lock.fileno(), fcntl.LOCK_UN)
            self.queue_lock.close()
            raise
        self.state_lock = threading.RLock()
        self.stop_event = threading.Event()
        self.state = {
            "experiment": preflight.EXPERIMENT,
            "stage": "source1_shared_warm_and_disposable_preflight",
            "queue_version": 1,
            "status": "running",
            "supervisor_pid": os.getpid(),
            "training_admission": False,
            "tasks": {task["task_id"]: {"task": task, "status": "queued"}
                      for task in self.tasks},
            "implementation_fingerprint": implementation_fingerprint(),
            "preflight_fingerprint": preflight.implementation_fingerprint(),
            "owner_queue_sha256": sha256_file(owner_queue.__file__),
            "owner_dispatcher_sha256": OWNER_HELPER_SHA256,
        }
        self._save()

    def _save(self):
        with self.state_lock:
            tmp = STATE.with_name(STATE.name + f".{os.getpid()}.tmp")
            with tmp.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(self.state, stream, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, STATE)

    def _update(self, task_id, **fields):
        with self.state_lock:
            self.state["tasks"][task_id].update(fields)
            self._save()

    def _finish(self, status):
        with self.state_lock:
            self.state.update(status=status, finished=time.time())
            self._save()

    @staticmethod
    def _terminate_owned(proc):
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        except ProcessLookupError:
            pass

    def _execute(self, task):
        task_id = task["task_id"]
        proc = lease = None
        attempt_log = RUNTIME / f"{task_id}.attempt01.log"
        self._update(task_id, status="waiting_for_exclusive_gpu")
        try:
            while not self.stop_event.is_set():
                gpu, gpu_uuid, memory, lease = owner_queue._acquire(self.stop_event)
                stable = owner_queue._stable_idle(
                    gpu, gpu_uuid, self.stop_event,
                    lambda elapsed, used: self._update(
                        task_id, status="waiting_stable_gpu", gpu=gpu,
                        gpu_uuid=gpu_uuid, stable_seconds=elapsed,
                        memory_used_mib=used))
                if stable:
                    break
                if fcntl is not None:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                lease.close()
                lease = None
                if self.stop_event.is_set():
                    self._update(task_id, status="cancelled_before_launch", finished=time.time())
                    return "cancelled_before_launch"
                self._update(task_id, status="waiting_for_exclusive_gpu",
                             last_reservation_released=True)
            else:
                self._update(task_id, status="cancelled_before_launch", finished=time.time())
                return "cancelled_before_launch"
            # Recheck create-once markers after reservation/stability admission.
            check_no_existing_artifacts([attempt_log, OUTPUT, LOG, WARM_ARTIFACT])
            argv = command(task, "cuda:0")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            with attempt_log.open("x", encoding="utf-8") as stream:
                proc = subprocess.Popen(
                    argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=True, pass_fds=(lease.fileno(),))
                self._update(task_id, status="running", child_pid=proc.pid,
                             child_argv=argv, physical_gpu=gpu, gpu_uuid=gpu_uuid,
                             memory_before_mib=memory, inherited_lease_fd=True,
                             log=str(attempt_log), started=time.time())
                while proc.poll() is None and not self.stop_event.is_set():
                    time.sleep(5)
                    if proc.poll() is not None:
                        break
                    try:
                        current_uuid, _used, owners = owner_queue.gpu_guard._gpu_info(gpu)
                        if current_uuid != gpu_uuid:
                            foreign = ["gpu_uuid_changed"]
                        else:
                            foreign = owner_queue.gpu_guard._foreign_owners(
                                owners, owner_queue.gpu_guard._process_tree(proc.pid))
                    except Exception as exc:
                        self._update(task_id, last_gpu_query_error=repr(exc))
                        continue
                    if foreign:
                        self._terminate_owned(proc)
                        self._update(task_id, status="interrupted_foreign_gpu_owner",
                                     foreign_owner_pids=foreign, finished=time.time())
                        return "interrupted_foreign_gpu_owner"
            returncode = int(proc.wait())
            good = returncode == 0 and valid_result(task)
            status = "passed" if good else "failed_preflight_or_artifact_validation"
            self._update(task_id, status=status, returncode=returncode,
                         artifact_valid=bool(good),
                         output_sha256=sha256_file(OUTPUT) if OUTPUT.is_file() else None,
                         warm_sha256=sha256_file(WARM_ARTIFACT) if WARM_ARTIFACT.is_file() else None,
                         finished=time.time())
            return status
        except BaseException as exc:
            self._terminate_owned(proc)
            self._update(task_id, status="failed", error=repr(exc), finished=time.time())
            return "failed"
        finally:
            if lease is not None:
                try:
                    if fcntl is not None:
                        fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                finally:
                    lease.close()

    def run(self):
        try:
            task = self.tasks[0]
            outcome = self._execute(task)
            self._finish("preflight_complete" if outcome == "passed" else "preflight_failed")
            return 0 if outcome == "passed" else 1
        finally:
            try:
                fcntl.flock(self.queue_lock.fileno(), fcntl.LOCK_UN)
                self.queue_lock.close()
            except (AttributeError, OSError):
                pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"experiment": preflight.EXPERIMENT, "tasks": task_plan(),
                          "owner_aware": True, "maximum_parallel_gpu_tasks": 1,
                          "training_stages": 0, "ground_truth": False,
                          "warm_artifact_outside_repository": True,
                          "dry_run": True}, indent=2), flush=True)
        return 0
    return PreflightQueue().run()


if __name__ == "__main__":
    raise SystemExit(main())
