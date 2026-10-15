"""Exclusive-GPU supervisor for the bounded SW0170 TRAIN8 admission."""
from __future__ import annotations

import os
import sys

_WORKER = "--worker" in sys.argv
if not _WORKER:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    import fcntl
except ImportError:  # CPU-only contract tests may run on Windows.
    fcntl = None
import hashlib
import importlib
import json
import signal
import subprocess
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"),
                str(ROOT / "snn_kuramoto_bidirectional")]
EXPERIMENT = "SW0170_causal_synaptic_readout_admission"
RUNTIME = HERE / "results_archive" / "admission_20261015"
STATE, REPORT = RUNTIME / "state.json", RUNTIME / "preflight.json"
RAW_REPORT = RUNTIME / "preflight_raw.json"
LOG, LOCK = RUNTIME / "worker.log", RUNTIME / "queue.lock"
PREFLIGHT = "collaborative_test.SW_0170_causal_synaptic_readout.preflight"
OWNER = "collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4"
_stop = threading.Event()


def _restore_visibility(visible):
    if visible is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _preflight():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        return importlib.import_module(PREFLIGHT)
    finally:
        if visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible


def _owner_preserving_visibility():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        return importlib.import_module(OWNER)
    finally:
        if visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible


def implementation_fingerprint():
    prep = _preflight()
    owner = _owner_preserving_visibility()
    values = prep.implementation_fingerprint()
    values["collaborative_test/SW_0170_causal_synaptic_readout/admission_queue.py"] = _sha(__file__)
    values["collaborative_test/SW_0148_checkpointable_boundary_q/stage_queue_v4.py"] = _sha(owner.__file__)
    return values


def fingerprint_sha256():
    data = json.dumps(implementation_fingerprint(), sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def source_admission():
    prepared = _preflight().prepare_inputs_cpu()
    proof = prepared.get("source_proof")
    if (not isinstance(proof, dict) or prepared.get("positions") != list(range(8))
            or len(prepared.get("selected_ids", ())) != 8
            or proof.get("ground_truth_read") is not False
            or proof.get("validation_data_read") is not False):
        raise ValueError("SW0170 CPU source admission is incomplete or out of scope")
    return prepared, proof


def validate_report(report, *, source_proof, queue_fingerprint, preflight_fingerprint):
    if report.get("experiment") != "SW0170_causal_synaptic_readout":
        raise ValueError("SW0170 report experiment mismatch")
    if report.get("source", {}).get("source_proof") != source_proof:
        raise ValueError("SW0170 report source proof mismatch")
    if report.get("queue_fingerprint_sha256") != queue_fingerprint:
        raise ValueError("SW0170 queue fingerprint mismatch")
    if report.get("implementation_fingerprint") != preflight_fingerprint:
        raise ValueError("SW0170 preflight fingerprint mismatch")
    status = report.get("status")
    admitted = report.get("training_admission")
    if status == "complete_admitted":
        if admitted is not True or report.get("stage1_gate_failures"):
            raise ValueError("admitted report has inconsistent scientific gate fields")
    elif status == "scientific_rejection":
        if admitted is not False or not isinstance(report.get("stage1_gate_failures"), list):
            raise ValueError("rejected report has inconsistent scientific gate fields")
    else:
        raise ValueError(f"nonterminal/unknown SW0170 report status: {status!r}")
    if report.get("validation_data_read") is not False:
        raise ValueError("SW0170 admission report claims validation data access")
    return status


def _atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with tmp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _write_state(status, **fields):
    state = {}
    if STATE.exists():
        state = json.loads(STATE.read_text(encoding="utf-8"))
    state.update({"experiment": EXPERIMENT, "status": status,
                  "supervisor_pid": os.getpid(), "updated": time.time(), **fields})
    _atomic_json(STATE, state)


def worker_command(gpu):
    return [sys.executable, "-m", PREFLIGHT.replace(".preflight", ".admission_queue"),
            "--worker", "--physical-gpu", str(int(gpu)), "--output", str(REPORT)]


def _stable_idle(owner, gpu, uuid, stop_event, on_sample):
    """Use the registered SW0148 four-argument stability callback contract."""
    return owner._stable_idle(gpu, uuid, stop_event, on_sample)


def _worker(gpu, output):
    assigned = os.environ.get("CUDA_VISIBLE_DEVICES")
    if assigned != str(gpu):
        raise RuntimeError("worker CUDA visibility does not match leased physical GPU")
    owner = _owner_preserving_visibility()
    del owner
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized before SW0170 CPU source admission")
    prepared, proof = source_admission()
    _restore_visibility(assigned)
    if torch.cuda.is_initialized() or torch.cuda.device_count() != 1:
        raise RuntimeError("CUDA visibility changed during SW0170 CPU source admission")
    prep = _preflight()
    prep_fp = prep.implementation_fingerprint()
    queue_fp = fingerprint_sha256()
    _restore_visibility(assigned)
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0170 CPU fingerprinting initialized CUDA before admission")
    if Path(output).exists():
        raise FileExistsError("preserving existing SW0170 admission report")
    torch.set_num_threads(1)
    report = _invoke_preflight(prep, prepared, RAW_REPORT, torch)
    report.update({"source": {**report.get("source", {}), "source_proof": proof},
                   "implementation_fingerprint": prep_fp,
                   "queue_fingerprint_sha256": queue_fp,
                   "physical_gpu": int(gpu), "cuda_visible_devices": assigned})
    _atomic_json(output, report)
    return report


def _worker_main(gpu, output):
    _worker(gpu, output)
    return 0


def _invoke_preflight(prep, prepared, output, torch_module):
    """Enter the frozen preflight before any explicit CUDA context selection."""
    if torch_module.cuda.is_initialized():
        raise RuntimeError("SW0170 worker entered preflight with CUDA already initialized")
    return prep.run_preflight(device="cuda:0", output_path=output, prepared=prepared)


def _proc_identity(owner, child, command):
    row = owner._proc_record(child.pid)
    if row is None or row.get("state") == "Z":
        return False
    argv = row.get("argv", [])
    return argv == command


def _supervise():
    if fcntl is None:
        raise RuntimeError("SW0170 GPU ownership requires Linux flock")
    signal.signal(signal.SIGTERM, lambda *_: _stop.set())
    signal.signal(signal.SIGINT, lambda *_: _stop.set())
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("SW0170 supervisor must hide CUDA")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    if any(path.exists() for path in (STATE, REPORT, RAW_REPORT, LOG)):
        raise FileExistsError("preserving existing SW0170 admission artifacts")
    lock = LOCK.open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = lease = owner = None
    try:
        fp = fingerprint_sha256()
        _write_state("validating_source_cpu", queue_fingerprint_sha256=fp)
        prepared, proof = source_admission()
        del prepared
        prep_fp = _preflight().implementation_fingerprint()
        _write_state("waiting_for_exclusive_gpu", queue_fingerprint_sha256=fp,
                     implementation_fingerprint=prep_fp, source_proof=proof,
                     validation_data_read=False)
        owner = _owner_preserving_visibility()
        while not _stop.is_set():
            gpu, uuid, before_memory, lease = owner._acquire(_stop)
            stable = _stable_idle(owner,
                gpu, uuid, _stop,
                lambda elapsed, used: _write_state("waiting_stable_gpu", physical_gpu=gpu,
                    gpu_uuid=uuid, stable_seconds=float(elapsed), memory_used_mib=int(used),
                    queue_fingerprint_sha256=fp, source_proof=proof))
            if not stable:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                lease.close()
                lease = None
                continue
            # Repeat source admission under lease, before final UUID/owner/memory check.
            current, current_proof = source_admission()
            del current
            if current_proof != proof or fingerprint_sha256() != fp:
                raise RuntimeError("SW0170 source or implementation changed before launch")
            actual_uuid, used, owners = owner.gpu_guard._gpu_info(gpu)
            if actual_uuid != uuid or owners or int(used) > owner.MAX_MEMORY_MIB:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                lease.close()
                lease = None
                _write_state("waiting_for_exclusive_gpu", physical_gpu=gpu,
                    gpu_uuid=uuid, gpu_check={"uuid": actual_uuid, "used_mib": int(used),
                                              "owners": list(owners)}, source_proof=proof)
                continue
            break
        else:
            raise RuntimeError("SW0170 admission stopped before GPU launch")

        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        command = worker_command(gpu)
        with LOG.open("x", encoding="utf-8", newline="\n") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                pass_fds=(lease.fileno(),))
            _write_state("running_preflight", child_pid=child.pid, child_pgid=child.pid,
                         child_argv=command, physical_gpu=gpu, gpu_uuid=uuid,
                         memory_used_before_mib=int(before_memory), source_proof=proof,
                         queue_fingerprint_sha256=fp)
            deadline = time.monotonic() + 1800
            while child.poll() is None and not _stop.is_set():
                if time.monotonic() > deadline:
                    owner._stop_owned(child)
                    raise TimeoutError("SW0170 bounded admission exceeded 1800 seconds")
                live_uuid, used_now, gpu_owners = owner.gpu_guard._gpu_info(gpu)
                if child.poll() is not None:
                    break
                if live_uuid != uuid or any(int(pid) != child.pid for pid in gpu_owners):
                    owner._stop_owned(child)
                    raise RuntimeError("leased GPU UUID/ownership changed; stopped owned child")
                proc = owner._proc_record(child.pid)
                if proc is not None and proc.get("state") != "Z" and not _proc_identity(owner, child, command):
                    owner._stop_owned(child)
                    raise RuntimeError("leased worker command identity changed")
                if proc is None or proc.get("state") == "Z":
                    try:
                        child.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        pass
                    if child.poll() is not None:
                        break
                    continue
                _write_state("running_preflight", child_pid=child.pid, physical_gpu=gpu,
                             gpu_uuid=uuid, memory_used_mib=int(used_now), source_proof=proof,
                             queue_fingerprint_sha256=fp)
                time.sleep(2)
            if _stop.is_set() and child.poll() is None:
                owner._stop_owned(child)
            rc = child.wait()
        if rc != 0:
            raise RuntimeError(f"SW0170 worker failed with exit code {rc}; see {LOG}")
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        status = validate_report(report, source_proof=proof, queue_fingerprint=fp,
                                 preflight_fingerprint=prep_fp)
        terminal = "complete_admitted" if status == "complete_admitted" else "complete_rejected"
        _write_state(terminal, report_sha256=_sha(REPORT), queue_fingerprint_sha256=fp,
                     implementation_fingerprint=prep_fp, source_proof=proof,
                     child_pid=child.pid, physical_gpu=gpu, gpu_uuid=uuid,
                     validation_data_read=False, finished=time.time())
        print(json.dumps({"status": terminal, "report_sha256": _sha(REPORT),
                          "physical_gpu": gpu}, sort_keys=True))
        return 0
    except BaseException as exc:
        if child is not None and child.poll() is None and owner is not None:
            owner._stop_owned(child)
        if STATE.exists():
            _write_state("failed", error=repr(exc), finished=time.time())
        raise
    finally:
        if child is not None and child.poll() is None and owner is not None:
            owner._stop_owned(child)
        if lease is not None:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
            lease.close()
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.worker:
        if args.physical_gpu not in range(4) or args.output is None:
            parser.error("worker requires a leased physical GPU and report path")
        _worker_main(args.physical_gpu, args.output)
        return 0
    return _supervise()


if __name__ == "__main__":
    raise SystemExit(main())
