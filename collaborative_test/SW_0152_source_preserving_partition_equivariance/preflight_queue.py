"""One GPU-owned SW0152 parity/feasibility preflight; it never starts training."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, threading, time
from pathlib import Path
try:
    import fcntl
except ImportError:
    fcntl = None

# Keep the supervising process CPU-only before importing any Torch-backed module.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4 as owner
from collaborative_test.SW_0150_spike_spatial_mixture import preflight as sw150_preflight
from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0152_source_preserving_partition_equivariance import preflight

RUNTIME = HERE / "results_archive" / "preflight_queue_20261013"
STATE = RUNTIME / "state.json"
QUEUE_LOCK = RUNTIME / "queue.lock"
OUTPUT = HERE / "results_archive" / "preflight_seed1.json"
LOG = RUNTIME / "preflight_seed1.attempt01.log"
PARITY = sw150_preflight.PARITY_PATH
OWNER_GUARD = owner.OWNER_GUARD
OWNER_GUARD_SHA256 = owner.OWNER_GUARD_SHA256


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def implementation_fingerprint():
    paths = (HERE / "preflight_queue.py", HERE / "preflight.py", HERE / "training.py",
             HERE / "deep_encoder.py", HERE / "view_augmentation.py",
             HERE / "partition_objective.py", HERE / "protocol.json",
             ROOT / "collaborative_test/SW_0150_spike_spatial_mixture/preflight_queue.py",
             ROOT / "collaborative_test/SW_0148_checkpointable_boundary_q/stage_queue_v4.py",
             OWNER_GUARD)
    missing = [p for p in paths if not p.is_file()]
    if missing: raise FileNotFoundError("missing SW0152 queue dependency: " + str(missing[0]))
    if sha256_file(OWNER_GUARD) != OWNER_GUARD_SHA256:
        raise ValueError("registered SW0148 GPU owner guard changed")
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}


def task_plan():
    return [{"task_id": "preflight_seed1", "stage": "preflight", "seed": 1,
             "device": "cuda:0", "depends_on": [], "output": str(OUTPUT),
             "log": str(LOG), "parity": str(PARITY)}]


def command(task, device="cuda:0"):
    if task.get("stage") != "preflight" or task.get("seed") != 1 or task.get("depends_on") != []:
        raise ValueError("only the registered SW0152 source1 preflight may run")
    return [sys.executable, "-m", "collaborative_test.SW_0152_source_preserving_partition_equivariance.preflight",
            "--device", device, "--output", str(task["output"])]


def _finite(value):
    try:
        import math
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def validate_report(report, *, expected_ids, expected_source, expected_assets,
                    expected_parity_sha, expected_fingerprint):
    """Strict, no-GT preflight validation; this never admits training by itself."""
    try:
        if (report.get("experiment") != preflight.EXPERIMENT
                or report.get("stage") != "source1_two_update_preflight"
                or report.get("status") != "source_preflight_complete"
                or report.get("seed") != 1 or report.get("training_admission") is not False
                or report.get("ground_truth_used") is not False or report.get("masks_read") is not False
                or report.get("implementation_fingerprint") != expected_fingerprint):
            return False
        source, parity = report["source"], report["saved_source_parity"]
        ids = [int(v) for v in expected_ids]
        if (len(ids) != 4096 or len(set(ids)) != 4096
                or source.get("training_ids") != ids
                or source.get("training_ids_sha256") != preflight.ids_sha(ids)
                or source.get("checkpoint_sha256") != expected_source["checkpoint_sha256"]
                or source.get("manifest_sha256") != expected_source["manifest_sha256"]
                or source.get("source_rgb_asset_validation") != expected_assets
                or parity.get("path") != str(PARITY) or parity.get("sha256") != expected_parity_sha
                or parity.get("status") != "native32_zero_adapter_parity_passed"
                or parity.get("source1_actual_source_exact_T1024_settle512") is not True):
            return False
        reg = report["registered"]
        if (reg.get("logical_batch") != 16 or reg.get("microbatch") != 4
                or reg.get("preflight_updates") != 2 or reg.get("total_steps") != 1024
                or reg.get("settle") != 512 or reg.get("live_tail") != 64):
            return False
        diag = report["within_color_permutation_diagnostic"]
        delta = float(diag["within_color_permuted_alignment_js_mean"]) - float(diag["baseline_alignment_js_mean"])
        if (diag.get("training_ids") != ids[:64] or diag.get("ground_truth_used") is not False
                or diag.get("diagnostic_only") is not True or not _finite(delta)
                or not _finite(diag.get("moved_fraction"))
                or not 0 <= float(diag["moved_fraction"]) <= 1
                or int(diag.get("eligible_patch_count", 0)) < 0
                or int(diag.get("changed_hard_assignment_count", 0)) < 0):
            return False
        records = diag.get("augmentation_records")
        if not isinstance(records, list) or len(records) != 64:
            return False
        for image_id, record in zip(ids[:64], records):
            gain, offset = record.get("gain"), record.get("offset")
            if (record.get("image_id") != image_id or record.get("source_seed") != 1
                    or not isinstance(gain, list) or len(gain) != 3
                    or not isinstance(offset, list) or len(offset) != 3
                    or any(not _finite(v) or not 0.6 <= float(v) <= 1.4 for v in gain)
                    or any(not _finite(v) or not -0.1 <= float(v) <= 0.1 for v in offset)):
                return False
        rates = report["registered"].get("optimizer_learning_rates")
        if rates != {"spatial_residual": 3e-6, "graph": 3e-5,
                     "native_core": 3e-5, "integration_scalars": 1e-3}:
            return False
        names = {"spatial_residual", "graph", "native_core", "integration_scalars"}
        arms = report["arms"]
        if set(arms) != {"equivariance", "mass_control"}: return False
        for arm in arms.values():
            if (arm.get("status") != "two_actual_updates_complete" or arm.get("image_ids") != ids[:32]
                    or arm.get("source_zero_head_parity_exact") is not True
                    or len(arm.get("history", [])) != 2
                    or set(arm.get("changed_parameter_families", {})) != names
                    or not all(arm["changed_parameter_families"].values())): return False
            before, after = arm["parameter_sha256_before"], arm["parameter_sha256_after"]
            if set(before) != names or set(after) != names: return False
            if any(not _sha256(before[n]) or not _sha256(after[n]) or before[n] == after[n]
                   for n in names): return False
            if (not _sha256(arm.get("frozen_base_encoder_sha256_before_after"))
                    or arm["frozen_base_encoder_sha256_before_after"] != source.get("source_encoder_sha256")):
                return False
            for row in arm["history"]:
                norms = row.get("family_gradient_norms", {})
                if set(norms) != names or any(not _finite(norms[n]) or float(norms[n]) <= 0 for n in names): return False
                if not _finite(row.get("mean_loss")): return False
        return True
    except (KeyError, TypeError, ValueError, OSError, AttributeError):
        return False


def _atomic_json(path, value):
    path = Path(path); tmp = path.with_name(path.name + ".tmp")
    with tmp.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def stop_owned_child(proc):
    """Delegate termination to the audited SW0148 owner helper."""
    return owner._stop_owned(proc)


def child_failure_record(returncode, output_path):
    """Represent a nonzero child exit without parsing a possibly absent output."""
    output_path = Path(output_path)
    return {"status": "failed_child", "returncode": int(returncode),
            "output_sha256": sha256_file(output_path) if output_path.is_file() else None}


class PreflightQueue:
    def __init__(self):
        if fcntl is None: raise RuntimeError("SW0152 GPU queue requires Linux flock")
        RUNTIME.mkdir(parents=True, exist_ok=True)
        self.lock_file = QUEUE_LOCK.open("a+")
        try: fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock_file.close(); raise RuntimeError("another SW0152 preflight queue owns this runtime") from exc
        for path in (STATE, OUTPUT, LOG):
            if Path(path).exists():
                self._release(); raise FileExistsError("preserving existing SW0152 preflight state/output/log")
        self.state_lock = threading.RLock(); self.stop_event = threading.Event()
        self.task = task_plan()[0]
        self.fingerprint = implementation_fingerprint()
        self.preflight_fingerprint = preflight.implementation_fingerprint()
        self.state = {"experiment": preflight.EXPERIMENT, "queue_version": 1,
            "stage": "preflight_only", "status": "waiting_for_exclusive_gpu",
            "supervisor_pid": os.getpid(), "training_admission": False,
            "tasks": {self.task["task_id"]: {"task": self.task, "status": "queued"}},
            "implementation_fingerprint": self.fingerprint,
            "preflight_fingerprint": self.preflight_fingerprint,
            "owner_queue_sha256": sha256_file(owner.__file__), "owner_guard_sha256": OWNER_GUARD_SHA256}
        _atomic_json(STATE, self.state)

    def _release(self):
        try: fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
        except (AttributeError, OSError): pass
        try: self.lock_file.close()
        except (AttributeError, OSError): pass

    def _update(self, status, **fields):
        with self.state_lock:
            self.state["status"] = status
            self.state["tasks"][self.task["task_id"]].update(status=status, **fields)
            _atomic_json(STATE, self.state)

    def _execute(self):
        proc = lease = None
        try:
            self._update("waiting_for_exclusive_gpu")
            while not self.stop_event.is_set():
                gpu, gpu_uuid, memory, lease = owner._acquire(self.stop_event)
                stable = owner._stable_idle(gpu, gpu_uuid, self.stop_event,
                    lambda elapsed, used: self._update("waiting_stable_gpu", gpu=gpu,
                        gpu_uuid=gpu_uuid, stable_seconds=elapsed, memory_used_mib=used))
                if stable: break
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                if self.stop_event.is_set():
                    self._update("cancelled_before_launch", finished=time.time()); return 1
                self._update("waiting_for_exclusive_gpu", reservation_released=True)
            if self.stop_event.is_set(): return 1
            for path in (OUTPUT, LOG):
                if path.exists(): raise FileExistsError("preflight output/log exists; no restart")
            argv = command(self.task, "cuda:0")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            with LOG.open("x", encoding="utf-8") as stream:
                proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                    pass_fds=(lease.fileno(),))
                self._update("running", child_pid=proc.pid, child_argv=argv, physical_gpu=gpu,
                    gpu_uuid=gpu_uuid, memory_before_mib=memory, inherited_lease_fd=True,
                    log=str(LOG), started=time.time())
                while proc.poll() is None and not self.stop_event.is_set():
                    time.sleep(5)
                    if proc.poll() is not None: break
                    try:
                        current_uuid, _used, owners = owner.gpu_guard._gpu_info(gpu)
                        foreign = (["gpu_uuid_changed"] if current_uuid != gpu_uuid else
                            owner.gpu_guard._foreign_owners(owners, owner.gpu_guard._process_tree(proc.pid)))
                    except Exception as exc:
                        self._update("running", last_gpu_query_error=repr(exc)); continue
                    if foreign:
                        stop_owned_child(proc)
                        self._update("interrupted_foreign_gpu_owner", foreign_owner_pids=foreign,
                                     finished=time.time()); return 1
            rc = int(proc.wait())
            if rc != 0:
                failure = child_failure_record(rc, OUTPUT)
                self._update(failure["status"], returncode=failure["returncode"],
                    output_sha256=failure["output_sha256"], log=str(LOG), finished=time.time())
                return 1
            report = json.loads(OUTPUT.read_text(encoding="utf-8"))
            src = source_contract_for_validation()
            assets = sw130.validate_rgb_assets()
            good = rc == 0 and validate_report(report, expected_ids=src["ids"],
                expected_source={"checkpoint_sha256": src["source_sha256"],
                                 "manifest_sha256": sha256_file(src["manifest_path"])},
                expected_assets=assets, expected_parity_sha=sha256_file(PARITY),
                expected_fingerprint=self.preflight_fingerprint)
            diag = report.get("within_color_permutation_diagnostic", {})
            delta = float(diag.get("within_color_permuted_alignment_js_mean", float("nan"))
                          - diag.get("baseline_alignment_js_mean", float("nan")))
            self._update("passed" if good else "failed_preflight_or_artifact_validation",
                returncode=rc, artifact_valid=bool(good),
                output_sha256=sha256_file(OUTPUT) if OUTPUT.is_file() else None,
                within_color_alignment_delta=delta if _finite(delta) else None,
                descriptive_permutation_gate=bool(_finite(delta) and delta > 0),
                training_admission=False, finished=time.time())
            return 0 if good else 1
        except BaseException as exc:
            if proc is not None: stop_owned_child(proc)
            self._update("failed", error=repr(exc), finished=time.time()); return 1
        finally:
            if lease is not None:
                try: fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                finally: lease.close()

    def run(self):
        try: return self._execute()
        finally: self._release()


def source_contract_for_validation():
    checkpoint, manifest, _record, _rows, ids, source_sha = sw130.source_contract(1)
    return {"checkpoint": checkpoint, "manifest_path": Path(manifest),
            "ids": [int(x) for x in ids], "source_sha256": source_sha}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"experiment": preflight.EXPERIMENT, "tasks": task_plan(),
            "owner_aware": True, "maximum_parallel_gpu_tasks": 1, "training_stages": 0,
            "ground_truth": False, "dry_run": True}, indent=2), flush=True)
        return 0
    return PreflightQueue().run()


if __name__ == "__main__": raise SystemExit(main())
