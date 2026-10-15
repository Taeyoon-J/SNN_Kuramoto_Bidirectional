"""Versioned owner-safe launcher for the corrected SW0168 support-mask admission."""
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
RUNTIME = HERE / "results_archive" / "source_admission_v4_20261015"
STATE = RUNTIME / "state.json"
REPORT = RUNTIME / "preflight.json"
LOG = RUNTIME / "worker.log"
LOCK = RUNTIME / "queue.lock"
OWNER_MODULE = "collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4"
EXPERIMENT = "SW0168_actual_spike_preflight_queue_v4"
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
    return importlib.import_module("collaborative_test.SW_0168_end_to_end_connectivity.preflight_v3")


def _source_proof(prepared):
    return _load_preflight()._source_identity(prepared)


def _source_admission():
    prep = _load_preflight()
    prepared = prep.prepare_preflight_cpu()
    if len(prepared["selected_ids"]) != 8 or len(prepared["targets"]) != 8:
        raise ValueError("source admission did not resolve the exact first eight TRAIN rows")
    return prep, prepared, _source_proof(prepared)


def implementation_fingerprint():
    prep = _load_preflight()
    owner = _load_owner_preserving_visibility()
    values = prep.implementation_fingerprint()
    values["collaborative_test/SW_0168_end_to_end_connectivity/preflight_queue_v4.py"] = _sha_file(__file__)
    values["collaborative_test/SW_0148_checkpointable_boundary_q/stage_queue_v4.py"] = _sha_file(owner.__file__)
    return values


def fingerprint_sha256():
    raw = json.dumps(implementation_fingerprint(), sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_report(report, *, source_proof, fingerprint):
    prep = _load_preflight()
    if report.get("queue_fingerprint_sha256") != fingerprint:
        raise ValueError("SW0168 queue fingerprint mismatch")
    current_preflight = prep.implementation_fingerprint()
    if report.get("implementation_fingerprint") != current_preflight:
        raise ValueError("SW0168 preflight implementation fingerprint mismatch")
    return prep.validate_report(report, expected_source=source_proof,
                                expected_fingerprint=current_preflight)


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
    if torch.cuda.is_initialized():
        raise RuntimeError("worker initialized CUDA before registered TRAIN source admission")
    prep, prepared, proof_before = _source_admission()
    if torch.cuda.is_initialized() or torch.cuda.device_count() != 1:
        raise RuntimeError("worker CUDA initialization/visibility is invalid after CPU admission")
    expected_fp = fingerprint_sha256()
    torch.set_num_threads(1)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    report = prep.run_two_update_preflight(prepared, device=device)
    report.update({"physical_gpu": physical_gpu, "cuda_visible_devices": assigned,
                   "source": proof_before,
                   "implementation_fingerprint": prep.implementation_fingerprint(),
                   "queue_fingerprint_sha256": expected_fp})
    # Preserve the completed GPU evidence before any post-run CPU validation.
    prep._atomic_json(output, report)
    command = [sys.executable, "-c",
        "import json; import numpy; import torch; "
        "from collaborative_test.SW_0168_end_to_end_connectivity import preflight_v3 as p; "
        "print(json.dumps(p._source_identity(p.prepare_preflight_cpu()), sort_keys=True))"]
    cpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    checked = subprocess.run(command, cwd=ROOT, env=cpu_env, stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             timeout=900, check=True)
    proof_after = json.loads(checked.stdout.strip().splitlines()[-1])
    if proof_before != proof_after:
        raise RuntimeError("source/cache bindings or TRAIN rows changed during preflight")
    prep.validate_report(report, expected_source=proof_before,
                         expected_fingerprint=report["implementation_fingerprint"])
    return report


def _worker_command(physical_gpu):
    return [sys.executable, "-m",
            "collaborative_test.SW_0168_end_to_end_connectivity.preflight_queue_v4",
            "--worker", "--physical-gpu", str(int(physical_gpu)), "--output", str(REPORT)]


def _check_deadline(child, deadline, owner, now):
    if now > deadline:
        owner._stop_owned(child)
        raise TimeoutError("SW0168 bounded preflight exceeded 1800 seconds")


def _supervisor():
    import fcntl
    signal.signal(signal.SIGTERM, lambda *_: _stop.set())
    signal.signal(signal.SIGINT, lambda *_: _stop.set())
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("supervisor must remain CPU-only with CUDA hidden")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    if any(path.exists() for path in (STATE, REPORT, LOG)):
        raise FileExistsError("preserving previous SW0168 prior preflight artifacts")
    lock = LOCK.open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = None
    lease = None
    owner = None
    try:
        fp = fingerprint_sha256()
        _write_state("validating_source_cpu", implementation_fingerprint_sha256=fp)
        prep, prepared, proof = _source_admission()
        del prep, prepared
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
            prep, current, current_proof = _source_admission()
            if current_proof != proof or fingerprint_sha256() != fp:
                raise RuntimeError("preflight source or implementation changed before launch")
            actual_uuid, used, owners = owner.gpu_guard._gpu_info(gpu)
            if actual_uuid != uuid or owners or int(used) > owner.MAX_MEMORY_MIB:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                _write_state("waiting_for_exclusive_gpu", source_proof=proof,
                             gpu_check={"uuid": actual_uuid, "memory_mib": int(used),
                                        "owners": list(owners)})
                continue
            del prep, current
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
                _check_deadline(child, deadline, owner, time.monotonic())
                now_uuid, now_memory, gpu_owners = owner.gpu_guard._gpu_info(gpu)
                if now_uuid != uuid or any(pid != child.pid for pid in gpu_owners):
                    owner._stop_owned(child)
                    raise RuntimeError("GPU identity/ownership changed; stopped only owned child")
                proc = owner._proc_record(child.pid)
                identity_bad = (not proc or proc.get("state") == "Z"
                                or command[1] not in proc.get("argv", [])
                                or command[2] not in proc.get("argv", []))
                if identity_bad and child.poll() is None:
                    time.sleep(0.05)
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
            raise RuntimeError(f"SW0168 preflight child failed with exit code {rc}; see {LOG}")
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
        print(json.dumps({"status": "complete_admitted",
                          "rows": len(report["source_b4_reference_rows"])}))
        return 0
    return _supervisor()


if __name__ == "__main__":
    raise SystemExit(main())
