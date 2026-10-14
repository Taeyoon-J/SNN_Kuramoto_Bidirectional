"""Create-once, owner-aware reservation for the SW0156 source-only screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np

try:
    import fcntl
except ImportError:
    fcntl = None

# Parent is a CPU supervisor; CUDA is exposed only to the lease-owning child.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for item in (ROOT, ROOT / "collaborative_test"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4 as owner
from collaborative_test.SW_0143_native32_selective_geodesic import run as sw143
from collaborative_test.SW_0156_signed_coupling import screen

EXPERIMENT = "SW0156_native32_signed_coupling"
SEED = 1
DEVICE = "cuda:0"
STEPS, SETTLE = 1024, 512
RUNTIME = HERE / "results_archive" / "source_screen_queue_20261013"
STATE = RUNTIME / "state.json"
LOCK = RUNTIME / "queue.lock"
OUTPUT = HERE / "results_archive" / "source_screen_seed1.json"
LOG = RUNTIME / "attempt01.log"
GPU_LEASE_DIR = Path("/tmp/kevinswk_sw0113_gpu_leases")
MAX_MEMORY_MIB = 512
STABLE_SECONDS = 5.0
SAMPLE_SECONDS = 2.0
UPSTREAM_POLL_SECONDS = 10.0


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def implementation_fingerprint():
    result = {"report_fingerprint": screen.implementation_fingerprint()}
    paths = (HERE / "source_screen_queue.py", HERE / "test_source_screen_queue.py",
             ROOT / "collaborative_test/SW_0148_checkpointable_boundary_q/stage_queue_v4.py",
             ROOT / "collaborative_test/SW_0126_history_event_binding/pilot_queue.py")
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("SW0156 queue dependency missing")
    if sha256_file(paths[-1]) != owner.OWNER_GUARD_SHA256:
        raise ValueError("registered GPU owner guard fingerprint changed")
    result["queue_dependencies"] = {
        path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths
    }
    return result


def implementation_sha256():
    payload = json.dumps(implementation_fingerprint(), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def task_plan():
    return {"task_id": "seed1_train_row0_source_screen", "stage": "source_screen",
            "seed": SEED, "device": DEVICE, "output": str(OUTPUT), "log": str(LOG),
            "steps": 1024, "settle": 512, "depends_on": ["SW0143_T_seed1_complete"]}


def command(task=None):
    task = task or task_plan()
    if task != task_plan():
        raise ValueError("only the registered SW0156 seed1 source screen may launch")
    return [sys.executable, "-m", "collaborative_test.SW_0156_signed_coupling.screen",
            "--device", DEVICE, "--output", str(OUTPUT)]


def _finite(value):
    try:
        import math
        return not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def upstream_state():
    """Return pending for absent artifacts; reject present but invalid lineage."""
    preflight = sw143._preflight_path(SEED)
    prediction_dir = sw143._arm_dir("T", SEED)
    if not preflight.is_file() or not (prediction_dir / "predictions.npz").is_file() \
            or not (prediction_dir / "protocol.json").is_file():
        return False, "waiting_for_sw0143_T_seed1"
    record = json.loads(preflight.read_text(encoding="utf-8"))
    if (record.get("status") != "preflight_complete" or record.get("seed") != SEED
            or record.get("implementation") != sw143.implementation_fingerprint()
            or record.get("ground_truth_used") is not False
            or record.get("optimizer_updates") != 0):
        raise ValueError("existing SW0143 source preflight is invalid/stale")
    t = sw143.validate_arm_prediction(prediction_dir, seed=SEED)
    if t.get("arm") != "T" or t.get("protocol", {}).get("seed") != SEED:
        raise ValueError("existing SW0143 T reference is invalid")
    return True, {"preflight_sha256": sha256_file(preflight),
                  "prediction_sha256": t["prediction_sha256"],
                  "protocol_sha256": t["protocol_sha256"],
                  "source_lineage": t["protocol"].get("source")}


def validate_report(path=OUTPUT, *, expected_fingerprint=None, upstream=None):
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        if expected_fingerprint is None:
            expected_fingerprint = screen.implementation_fingerprint()
        ready, ref = upstream_state() if upstream is None else (True, upstream)
        if not ready:
            return False
        source_checkpoint, manifest_path, _manifest, pool, ids, source_sha = \
            sw143.foundation.sw130.source_contract(SEED)
        source_id_sha = hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
        first_row, first_id = int(pool[0]), int(ids[0])
        rgb = np.load(sw143.foundation.sw130.TRAIN_RGB, mmap_mode="r", allow_pickle=False)
        if rgb.dtype != np.uint8 or rgb.shape != (70000, 128, 128, 3):
            return False
        rgb_sha = hashlib.sha256(np.asarray(rgb[first_row]).tobytes()).hexdigest()
        source = record.get("source_lineage")
        if (record.get("experiment") != EXPERIMENT
                or record.get("status") != "source_screen_complete"
                or record.get("seed") != SEED
                or record.get("training_cache_row") != first_row
                or record.get("training_image_id") != first_id
                or record.get("sw0143_preflight_sha256") != ref["preflight_sha256"]
                or record.get("sw0143_T_prediction_sha256") != ref["prediction_sha256"]
                or record.get("sw0143_T_protocol_sha256") != ref["protocol_sha256"]
                or record.get("implementation") != expected_fingerprint
                or record.get("training_rgb_sha256") != rgb_sha
                or not isinstance(source, dict)
                or source.get("source_core_sha256") != source_sha
                or source.get("source_manifest_sha256") != sha256_file(manifest_path)
                or source.get("source_training_ids_sha256") != source_id_sha
                or source != ref.get("source_lineage")
                or record.get("source_model_unchanged") is not True
                or record.get("source_model_state_before_sha256") != record.get("source_model_state_after_sha256")
                or not isinstance(record.get("source_model_state_before_sha256"), str)
                or len(record["source_model_state_before_sha256"]) != 64
                or record.get("ground_truth_used") is not False
                or record.get("optimizer_updates") != 0):
            return False
        parity = record.get("zero_intervention_exact_parity")
        if (not isinstance(parity, dict)
                or set(parity.get("trace_fields", {})) != set(screen.TRACE_FIELDS)
                or not all(parity["trace_fields"].values())
                or parity.get("q_equal") is not True or parity.get("qcc_equal") is not True):
            return False
        arms = record.get("arms")
        if not isinstance(arms, dict) or set(arms) != {
                "source_T", "beta0_exact_control", "signed_candidate", "matched_control"}:
            return False
        for name in ("source_T", "beta0_exact_control", "signed_candidate", "matched_control"):
            arm = arms[name]
            expected_shapes = {
                "theta": [1, 1024, 1024, 4],
                "membrane": [1, 1024, 1024],
                "spikes": [1, 1024, 1024],
                "component_membrane": [1, 4, 1024, 1024],
                "component_spikes": [1, 4, 1024, 1024],
                "component_gates": [1, 4, 1024, 1024],
            }
            if (not isinstance(arm, dict) or not _finite(arm.get("seconds"))
                    or float(arm["seconds"]) <= 0
                    or not isinstance(arm.get("trace_schema"), dict)
                    or set(arm["trace_schema"]) != set(screen.TRACE_FIELDS)
                    or any(arm["trace_schema"][key].get("shape") != shape
                           or arm["trace_schema"][key].get("dtype") != "torch.float32"
                           for key, shape in expected_shapes.items())):
                return False
            dynamics = arm.get("dynamics")
            components = dynamics.get("per_component") if isinstance(dynamics, dict) else None
            q_stats = dynamics.get("actual_spike_q_offdiagonal") if isinstance(dynamics, dict) else None
            if (dynamics.get("settled_steps") != 512 or not isinstance(components, list)
                    or len(components) != 4 or not isinstance(q_stats, dict)
                    or any(not isinstance(component, dict)
                           or any(not _finite(component.get(key)) for key in (
                               "weighted_spike_mean", "spike_variance", "gate_mean",
                               "gate_active_fraction"))
                           or (component.get("event_on_fraction_when_gate_positive") is not None
                               and (not _finite(component.get("event_on_fraction_when_gate_positive"))
                                    or not 0 <= float(component[
                                        "event_on_fraction_when_gate_positive"]) <= 1))
                           or type(component.get("gate_positive_support_count")) is not int
                           or component["gate_positive_support_count"] < 0
                           or not 0 <= float(component["weighted_spike_mean"]) <= 1
                           or float(component["spike_variance"]) < 0
                           or not 0 <= float(component["gate_mean"]) <= 1
                           or not 0 <= float(component["gate_active_fraction"]) <= 1
                           for component in components)
                    or any(not _finite(q_stats.get(key)) for key in ("mean", "min", "max"))
                    or float(q_stats["min"]) > float(q_stats["mean"])
                    or float(q_stats["mean"]) > float(q_stats["max"])):
                return False
        signed = arms["signed_candidate"]
        control = arms["matched_control"]
        if arms["beta0_exact_control"].get("dynamics") != arms["source_T"].get("dynamics"):
            return False
        if (signed.get("finite_q") is not True or control.get("finite_q") is not True
                or not _finite(signed.get("peak_cuda_allocated_bytes"))
                or not _finite(control.get("peak_cuda_allocated_bytes"))
                or int(signed["peak_cuda_allocated_bytes"]) <= 0
                or int(control["peak_cuda_allocated_bytes"]) <= 0
                or not isinstance(signed.get("qcc_label_counts"), dict)
                or not isinstance(control.get("qcc_label_counts"), dict)
                or sum(int(v) for v in signed["qcc_label_counts"].values()) != 1024
                or sum(int(v) for v in control["qcc_label_counts"].values()) != 1024):
            return False
        params = record.get("registered_parameters")
        if (not isinstance(params, dict) or params.get("steps") != STEPS
                or params.get("settle") != 512 or params.get("beta") != .5
                or params.get("source_top_k") != 128
                or params.get("coupling_backend") != "factorized"
                or not _finite(params.get("gain")) or float(params["gain"]) <= 0):
            return False
        diagnostic = record.get("signed_interaction_screen")
        return (isinstance(diagnostic, dict)
                and _finite(diagnostic.get("candidate_negative_offdiagonal_abs_mass_fraction"))
                and float(diagnostic["candidate_negative_offdiagonal_abs_mass_fraction"]) > 0
                and diagnostic.get("ground_truth_used") is False
                and diagnostic.get("optimizer_updates") == 0
                and diagnostic.get("full320_authorized_by_this_record") is False)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, json.JSONDecodeError):
        return False


def _atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


class SourceScreenQueue:
    def __init__(self):
        if fcntl is None:
            raise RuntimeError("SW0156 GPU reservation requires Linux flock")
        RUNTIME.mkdir(parents=True, exist_ok=True)
        self.lock_stream = LOCK.open("a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock_stream.close()
            raise RuntimeError("another SW0156 source-screen supervisor is active") from exc
        self.stop_event = threading.Event()
        self.task = task_plan()
        self.queue_fingerprint = implementation_fingerprint()
        self.report_fingerprint = screen.implementation_fingerprint()
        self.completed = False
        try:
            if OUTPUT.exists():
                ready, ref = upstream_state()
                if not validate_report(OUTPUT, expected_fingerprint=self.report_fingerprint,
                                       upstream=ref if ready else None):
                    raise FileExistsError("preserve invalid/stale SW0156 screen output")
                self.completed = True
            else:
                existing = [p for p in (STATE, LOG) if p.exists()]
                if existing:
                    raise FileExistsError(f"preserve existing SW0156 runtime artifacts: {existing}")
                _atomic_json(STATE, {"experiment": EXPERIMENT, "queue_version": 1,
                    "status": "waiting_for_sw0143_T_seed1", "supervisor_pid": os.getpid(),
                    "task": self.task, "queue_fingerprint": self.queue_fingerprint,
                    "report_fingerprint": self.report_fingerprint,
                    "ground_truth_used": False, "optimizer_updates": 0})
        except BaseException:
            self._unlock(); raise

    def _unlock(self):
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_UN)
        finally:
            self.lock_stream.close()

    def _update(self, status, **fields):
        state = json.loads(STATE.read_text(encoding="utf-8"))
        state.update(status=status, updated=time.time(), **fields)
        _atomic_json(STATE, state)

    def _verify(self):
        if implementation_fingerprint() != self.queue_fingerprint:
            raise RuntimeError("SW0156 source-screen queue dependencies changed")

    def run(self):
        proc = lease = None
        try:
            if self.completed:
                return 0
            while not self.stop_event.is_set():
                try:
                    ready, ref = upstream_state()
                except Exception as exc:
                    self._update("failed_invalid_upstream", error=repr(exc), finished=time.time())
                    return 1
                if not ready:
                    self._update("waiting_for_sw0143_T_seed1")
                    self.stop_event.wait(UPSTREAM_POLL_SECONDS)
                    continue
                self._verify()
                while not self.stop_event.is_set():
                    gpu, gpu_uuid, memory, lease = owner._acquire(self.stop_event)
                    try:
                        stable = owner._stable_idle(gpu, gpu_uuid, self.stop_event,
                            lambda elapsed, used: self._update("waiting_stable_gpu", gpu=gpu,
                                gpu_uuid=gpu_uuid, stable_seconds=elapsed, memory_used_mib=used))
                        if not stable:
                            fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                            if self.stop_event.is_set():
                                return 1
                            self._update("waiting_for_exclusive_gpu", reservation_released=True)
                            continue
                        self._verify()
                        upstream_now, ref_now = upstream_state()
                        if not upstream_now or ref_now != ref:
                            raise RuntimeError("SW0143 upstream changed before screen launch")
                        if OUTPUT.exists() or LOG.exists():
                            raise FileExistsError("preserve SW0156 output/log collision before launch")
                        argv = command(self.task)
                        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                        with LOG.open("x", encoding="utf-8") as log:
                            proc = subprocess.Popen(argv, cwd=ROOT, env=env,
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, pass_fds=(lease.fileno(),))
                            self._update("running", child_pid=proc.pid, child_argv=argv,
                                physical_gpu=gpu, gpu_uuid=gpu_uuid, memory_before_mib=memory,
                                inherited_lease_fd=True, log=str(LOG), upstream=ref)
                            while proc.poll() is None and not self.stop_event.is_set():
                                time.sleep(SAMPLE_SECONDS)
                                if proc.poll() is not None:
                                    break
                                try:
                                    current_uuid, _used, owners = owner.gpu_guard._gpu_info(gpu)
                                    foreign = (["gpu_uuid_changed"] if current_uuid != gpu_uuid else
                                        owner.gpu_guard._foreign_owners(owners,
                                            owner.gpu_guard._process_tree(proc.pid)))
                                except Exception as exc:
                                    self._update("running", last_gpu_query_error=repr(exc))
                                    continue
                                if foreign:
                                    owner._stop_owned(proc)
                                    self._update("failed_foreign_gpu_owner", foreign=foreign,
                                                 child_pid=proc.pid, finished=time.time())
                                    return 1
                        rc = int(proc.wait())
                        valid = (rc == 0 and validate_report(OUTPUT,
                            expected_fingerprint=self.report_fingerprint, upstream=ref))
                        self._update("complete" if valid else "failed_child_or_report_validation",
                            child_returncode=rc, output_valid=bool(valid), finished=time.time())
                        return 0 if valid else 1
                    finally:
                        if lease is not None:
                            try:
                                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                            finally:
                                lease.close(); lease = None
            return 1
        except BaseException as exc:
            if proc is not None and proc.poll() is None:
                owner._stop_owned(proc)
            self._update("failed", error=repr(exc), finished=time.time())
            raise
        finally:
            self._unlock()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"status": "planned", "task": task_plan(), "argv": command(),
            "cuda_visible_devices_parent": "", "queue_fingerprint": implementation_sha256()},
            allow_nan=False), flush=True)
        return 0
    return SourceScreenQueue().run()


if __name__ == "__main__":
    raise SystemExit(main())
