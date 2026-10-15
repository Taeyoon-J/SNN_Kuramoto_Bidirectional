"""Single-child owner-safe launcher for the bounded SW0167 preflight."""
from __future__ import annotations

import os
import sys

_WORKER = "--worker" in sys.argv
if not _WORKER:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import hashlib
import importlib
import json
import math
import signal
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"),
                str(ROOT / "snn_kuramoto_bidirectional")]
RUNTIME = HERE / "results_archive" / "split_feature_admission_v2_20261015"
STATE = RUNTIME / "state.json"
REPORT = RUNTIME / "preflight.json"
LOG = RUNTIME / "worker.log"
LOCK = RUNTIME / "queue.lock"
OWNER_MODULE = "collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4"
EXPERIMENT = "SW0167_split_feature_preflight_queue"
_stop = threading.Event()


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _load_owner_preserving_visibility():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    owner = importlib.import_module(OWNER_MODULE)
    if visible is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
    return owner


def _load_preflight():
    return importlib.import_module(
        "collaborative_test.SW_0167_structured_relation_energy.preflight_v2")


def _source_proof(prepared):
    return {
        "ordered_image_ids": [int(value) for value in prepared["selected_ids"]],
        "source_checkpoint_sha256": prepared["source_checkpoint_sha256"],
        "source_manifest_sha256": prepared["source_manifest_sha256"],
        "source_cache_manifest_sha256": prepared["source_cache_manifest_sha256"],
        "context_cache_manifest_sha256": prepared["context_cache_manifest_sha256"],
        "source_state_sha256": prepared["source_state_sha256"],
        "context_state_sha256": prepared["context_state_sha256"],
        "gain_receipt_sha256": prepared["gain_receipt_sha256"],
        "row_bindings": [
            {key: row[key] for key in ("index", "image_id", "rgb_sha256", "source_file_sha256",
                                      "trace_sha256", "context_file_sha256", "q_sha256",
                                      "adjacency_sha256")}
            for row in prepared["rows"]],
    }


def _source_admission():
    prep = _load_preflight()
    audit = prep._import_preserving_cuda_visibility(
        "collaborative_test.SW_0167_structured_relation_energy.audit_train_cache")
    prepared = audit._validate_upstream_and_rows()
    supports, targets = prep.prepare_train_targets(prepared)
    if len(prepared["selected_ids"]) != 8 or len(targets) != 8:
        raise ValueError("source admission did not resolve the exact first eight TRAIN rows")
    return audit, prep, prepared, supports, targets, _source_proof(prepared)


def implementation_fingerprint():
    prep = _load_preflight()
    owner = _load_owner_preserving_visibility()
    values = prep.implementation_fingerprint()
    values["collaborative_test/SW_0167_structured_relation_energy/preflight_queue_v2.py"] = _sha_file(__file__)
    values["collaborative_test/SW_0148_checkpointable_boundary_q/stage_queue_v4.py"] = _sha_file(owner.__file__)
    return values


def fingerprint_sha256():
    raw = json.dumps(implementation_fingerprint(), sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_report(report, *, source_proof, fingerprint):
    if (not isinstance(report, dict)
            or report.get("experiment") != "SW0167_split_feature_two_update_preflight"
            or report.get("status") != "complete_admitted"
            or report.get("optimizer_updates") != 2
            or report.get("feature_mode") != "rho_from_cached_S_B1; context_Q_and_A_from_registered_original_B4"
            or report.get("source_F_mode") != "production CPU float32 Q on original TRAIN B4 groups, then production QCC readout"
            or report.get("same_feature_bytes_for_both_arms") is not True
            or report.get("validation_data_read") is not False
            or report.get("implementation_fingerprint_sha256") != fingerprint
            or report.get("source_proof") != source_proof
            or report.get("selected_image_ids") != source_proof["ordered_image_ids"]
            or report.get("source_checkpoint_sha256") != source_proof["source_checkpoint_sha256"]
            or report.get("source_manifest_sha256") != source_proof["source_manifest_sha256"]
            or report.get("source_cache_manifest_sha256") != source_proof["source_cache_manifest_sha256"]
            or report.get("context_cache_manifest_sha256") != source_proof["context_cache_manifest_sha256"]
            or report.get("source_state_sha256") != source_proof["source_state_sha256"]
            or report.get("context_state_sha256") != source_proof["context_state_sha256"]):
        raise ValueError("SW0167 preflight report provenance/schema mismatch")
    rows = report.get("rows")
    parity = report.get("source_foreground_and_zero_residual_parity")
    arms = report.get("arms")
    if (not isinstance(rows, list) or len(rows) != 8
            or not isinstance(parity, list) or len(parity) != 8
            or not isinstance(arms, dict) or set(arms) != {"structured_hinge", "local_margin"}):
        raise ValueError("SW0167 preflight row/arm schema mismatch")
    for index, row in enumerate(rows):
        if (row.get("cache_index") != index
                or row.get("image_id") != source_proof["ordered_image_ids"][index]
                or row.get("q_sha256") != source_proof["row_bindings"][index]["q_sha256"]
                or row.get("source_foreground_exact") is not True
                or row.get("source_zero_head_labels_exact") is not True
                or row.get("source_zero_head_merge_log_exact") is not True
                or not isinstance(row.get("cpu_b4_source_F_sha256"), str)
                or len(row["cpu_b4_source_F_sha256"]) != 64
                or not isinstance(row.get("feature_sha256"), str)
                or len(row["feature_sha256"]) != 64):
            raise ValueError(f"SW0167 source parity proof invalid at row {index}")
    if report.get("feature_sha256_by_row") != [row["feature_sha256"] for row in rows]:
        raise ValueError("SW0167 shared feature bytes do not match row receipts")
    initial_hashes = {record.get("initial_state_sha256") for record in arms.values()}
    if len(initial_hashes) != 1 or any(not isinstance(value, str) or len(value) != 64
                                       for value in initial_hashes):
        raise ValueError("SW0167 candidate/control head initialization differs")
    for arm, record in arms.items():
        steps = record.get("steps")
        if (not record.get("both_steps_finite") or not record.get("output_changed")
                or not record.get("hidden_changed_after_second_step")
                or not isinstance(steps, list) or len(steps) != 2
                or [step.get("step") for step in steps] != [1, 2]):
            raise ValueError(f"SW0167 disposable-update proof invalid for {arm}")
        for step in steps:
            if (not math.isfinite(float(step.get("mean_loss")))
                    or any(not math.isfinite(float(value)) or float(value) < 0
                           for value in step.get("gradient_norms", {}).values())):
                raise ValueError(f"SW0167 nonfinite/invalid gradient record for {arm}")
        if (sum(float(steps[0]["gradient_norms"].get(name, 0.0))
                for name in ("output.weight", "output.bias")) <= 0
                or sum(float(steps[1]["gradient_norms"].get(name, 0.0))
                       for name in ("hidden.weight", "hidden.bias")) <= 0):
            raise ValueError(f"SW0167 required output/hidden credit missing for {arm}")
    return True


def _write_state(status, **fields):
    current = {}
    if STATE.is_file():
        current = json.loads(STATE.read_text(encoding="utf-8"))
    current.update({"experiment": EXPERIMENT, "status": status,
                    "supervisor_pid": os.getpid(), "updated": time.time(), **fields})
    _atomic_json(STATE, current)


def _worker(physical_gpu, output):
    assigned = os.environ.get("CUDA_VISIBLE_DEVICES")
    if assigned != str(physical_gpu):
        raise RuntimeError("worker visibility differs from leased physical GPU")
    owner = _load_owner_preserving_visibility()
    del owner
    prep = _load_preflight()
    import torch
    if torch.cuda.is_initialized() or torch.cuda.device_count() != 1:
        raise RuntimeError("worker CUDA initialization/visibility is invalid")
    audit, _prep, prepared, supports, targets, proof_before = _source_admission()
    expected_fp = fingerprint_sha256()
    torch.set_num_threads(1)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    report = prep.run_two_update_preflight(prepared, supports, targets, device=device)
    after = audit._validate_upstream_and_rows()
    proof_after = _source_proof(after)
    if proof_before != proof_after:
        raise RuntimeError("source/cache bindings or TRAIN rows changed during preflight")
    report.update({"physical_gpu": physical_gpu, "cuda_visible_devices": assigned,
                   "implementation_fingerprint_sha256": expected_fp,
                   "source_proof": proof_before,
                   "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device))})
    prep._atomic_json(output, report)
    return report


def _worker_command(physical_gpu):
    return [sys.executable, "-m",
            "collaborative_test.SW_0167_structured_relation_energy.preflight_queue_v2",
            "--worker", "--physical-gpu", str(int(physical_gpu)), "--output", str(REPORT)]


def _supervisor():
    import fcntl
    signal.signal(signal.SIGTERM, lambda *_: _stop.set())
    signal.signal(signal.SIGINT, lambda *_: _stop.set())
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("supervisor must remain CPU-only with CUDA hidden")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    if any(path.exists() for path in (STATE, REPORT, LOG)):
        raise FileExistsError("preserving previous SW0167 preflight artifacts")
    lock = LOCK.open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = None
    lease = None
    owner = None
    try:
        fp = fingerprint_sha256()
        _write_state("validating_source_cpu", implementation_fingerprint_sha256=fp)
        audit, prep, prepared, supports, targets, proof = _source_admission()
        del audit, prep, prepared, supports, targets
        _write_state("waiting_for_exclusive_gpu", implementation_fingerprint_sha256=fp,
                     source_proof=proof, ground_truth_scope="TRAIN first eight only")
        owner = _load_owner_preserving_visibility()
        while not _stop.is_set():
            gpu, uuid, memory, lease = owner._acquire(_stop)
            stable = owner._stable_idle(gpu, uuid, _stop,
                lambda elapsed, used: _write_state("waiting_stable_gpu", physical_gpu=gpu,
                    gpu_uuid=uuid, stable_seconds=elapsed, memory_used_mib=int(used),
                    source_proof=proof, implementation_fingerprint_sha256=fp))
            if not stable:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                continue
            # Revalidate the expensive source/mask admission under the lease,
            # then perform the final UUID/owner/memory check immediately before launch.
            audit, prep, current, supports, targets, current_proof = _source_admission()
            if current_proof != proof or fingerprint_sha256() != fp:
                raise RuntimeError("preflight source or implementation changed before launch")
            actual_uuid, used, owners = owner.gpu_guard._gpu_info(gpu)
            if actual_uuid != uuid or owners or int(used) > owner.MAX_MEMORY_MIB:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                _write_state("waiting_for_exclusive_gpu", source_proof=proof,
                             gpu_check={"uuid": actual_uuid, "memory_mib": int(used),
                                        "owners": list(owners)})
                continue
            del audit, prep, current, supports, targets
            break
        else:
            raise RuntimeError("preflight stopped before GPU admission")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1")
        command = _worker_command(gpu)
        with LOG.open("x", encoding="utf-8", newline="\n") as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                pass_fds=(lease.fileno(),))
            _write_state("running_preflight", child_pid=child.pid, child_pgid=child.pid,
                         child_argv=command, physical_gpu=gpu, gpu_uuid=uuid,
                         memory_used_before_mib=memory, source_proof=proof,
                         implementation_fingerprint_sha256=fp)
            deadline = time.monotonic() + 1800.0
            while child.poll() is None and not _stop.is_set():
                if time.monotonic() > deadline:
                    owner._stop_owned(child)
                    raise TimeoutError("SW0167 bounded preflight exceeded 1800 seconds")
                now_uuid, now_memory, gpu_owners = owner.gpu_guard._gpu_info(gpu)
                if now_uuid != uuid or any(pid != child.pid for pid in gpu_owners):
                    owner._stop_owned(child)
                    raise RuntimeError("GPU identity/ownership changed; stopped only owned child")
                proc = owner._proc_record(child.pid)
                if child.poll() is None and (not proc or proc.get("state") == "Z"
                                             or command[1] not in proc.get("argv", [])
                                             or command[2] not in proc.get("argv", [])):
                    # A normal exit can race /proc disappearance.
                    if child.poll() is None:
                        owner._stop_owned(child)
                        raise RuntimeError("leased preflight child identity changed")
                _write_state("running_preflight", child_pid=child.pid,
                             physical_gpu=gpu, gpu_uuid=uuid,
                             memory_used_mib=int(now_memory), source_proof=proof,
                             implementation_fingerprint_sha256=fp)
                time.sleep(2.0)
            rc = child.wait()
        if rc != 0:
            raise RuntimeError(f"SW0167 preflight child failed with exit code {rc}; see {LOG}")
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        validate_report(report, source_proof=proof, fingerprint=fp)
        _write_state("complete_admitted", report_sha256=_sha_file(REPORT),
                     implementation_fingerprint_sha256=fp, source_proof=proof,
                     child_pid=child.pid, physical_gpu=gpu, gpu_uuid=uuid,
                     ground_truth_scope="TRAIN first eight only", validation_data_read=False,
                     finished=time.time())
        print(json.dumps({"status": "complete_admitted", "report_sha256": _sha_file(REPORT),
                          "physical_gpu": gpu}, sort_keys=True))
        return 0
    except BaseException as exc:
        if child is not None and child.poll() is None:
            owner._stop_owned(child)
        try:
            if STATE.parent.exists():
                _write_state("failed", error=repr(exc), finished=time.time())
        except BaseException:
            pass
        raise
    finally:
        if child is not None and child.poll() is None:
            owner._stop_owned(child)
        if lease is not None:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
            lease.close()
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.physical_gpu not in range(4) or args.output is None:
            raise SystemExit("worker requires its leased physical GPU and output path")
        report = _worker(args.physical_gpu, args.output)
        print(json.dumps({"status": "complete_admitted", "rows": len(report["rows"])}))
        return 0
    return _supervisor()


if __name__ == "__main__":
    raise SystemExit(main())
