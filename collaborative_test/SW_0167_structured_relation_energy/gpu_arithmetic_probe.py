"""Owner-safe, no-rollout GPU arithmetic check for SW0167's first TRAIN B4."""
from __future__ import annotations

import os
import sys

# The supervisor remains CPU-only. The child keeps the assigned physical GPU
# visible; importing audit_train_cache temporarily hides CUDA, so it restores
# visibility before the first CUDA runtime call.
_WORKER = "--worker" in sys.argv
if not _WORKER:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
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


def _repo_root():
    override = os.environ.get("SW0167_AUDIT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "collaborative_test").is_dir() and (parent / "snn_kuramoto_bidirectional").is_dir():
            return parent
    raise RuntimeError("cannot locate patch_v2_sw; set SW0167_AUDIT_ROOT")


ROOT = _repo_root()
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"),
                str(ROOT / "snn_kuramoto_bidirectional")]


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _delta(reference, actual):
    a = np.asarray(reference, dtype=np.float32)
    b = np.asarray(actual, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError("Q comparison shape mismatch")
    return {
        "max_abs": float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))),
        "exact_entry_differences": int(np.count_nonzero(a != b)),
        "threshold_edge_disagreements_at_0_5": int(np.count_nonzero((a >= 0.5) != (b >= 0.5))),
    }


def _source_proof(prepared):
    """Normalize the real audit producer's selected_ids into receipt schema."""
    return {
        "ordered_image_ids": list(prepared["selected_ids"]),
        "source_checkpoint_sha256": prepared["source_checkpoint_sha256"],
        "source_manifest_sha256": prepared["source_manifest_sha256"],
        "source_cache_manifest_sha256": prepared["source_cache_manifest_sha256"],
        "context_cache_manifest_sha256": prepared["context_cache_manifest_sha256"],
        "source_state_sha256": prepared["source_state_sha256"],
        "gain": prepared["gain"],
        "gain_receipt_sha256": prepared["gain_receipt_sha256"],
    }


def _q_modes(q_cached, q_gpu_b4, q_gpu_b1):
    """Keep cached and both independently recomputed GPU Q paths explicit."""
    return (("cached_gpu_b4", q_cached), ("gpu_b4", q_gpu_b4), ("gpu_b1", q_gpu_b1))


def _atomic_create_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _audit_module():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    audit = importlib.import_module(
        "collaborative_test.SW_0167_structured_relation_energy.audit_train_cache")
    if visible is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
    return audit


def _run_worker(args):
    assigned = os.environ.get("CUDA_VISIBLE_DEVICES")
    if assigned != str(args.physical_gpu):
        raise RuntimeError("worker CUDA visibility does not match leased physical GPU")
    audit = _audit_module()
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("unexpected CUDA initialization before restoring assigned visibility")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("worker must see exactly its leased physical GPU")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    # This validates manifests, selected TRAIN RGB bytes, source checkpoint,
    # row ledgers, and saved S/Q/A before arithmetic; no masks/VAL are loaded.
    prepared = audit._validate_upstream_and_rows()
    if len(prepared["selected_ids"]) != 8:
        raise AssertionError("expected exactly the registered first eight TRAIN rows")
    from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity
    from collaborative_test.SW_0167_structured_relation_energy import structured_energy
    from collaborative_test.SW_0167_structured_relation_energy import qcc_readout

    rows = []
    for group_start in (0, 4):
        components_np = np.stack(prepared["component_spikes"][group_start:group_start + 4])
        components = torch.from_numpy(components_np).to(device=device, dtype=torch.float32)
        q_b4 = spike_synchrony_affinity(
            components.mean(dim=1), components, settle=0, affinity_mode="spike")
        if q_b4.shape != (4, 1024, 1024) or q_b4.dtype != torch.float32:
            raise ValueError("production GPU B4 Q returned unexpected geometry/dtype")
        for offset in range(4):
            index = group_start + offset
            q_cached_np, adjacency_np = prepared["contexts"][index]
            q_cached = torch.from_numpy(q_cached_np).to(device=device)
            adjacency = torch.from_numpy(adjacency_np).to(device=device)
            q4 = q_b4[offset]
            if not torch.isfinite(q4).all() or not torch.isfinite(adjacency).all():
                raise FloatingPointError(f"nonfinite GPU B4 Q/A at cache row {index}")
            delta_b4 = _delta(q_cached_np, q4.detach().cpu().numpy())

            single = components[offset:offset + 1]
            q_b1 = spike_synchrony_affinity(
                single.mean(dim=1), single, settle=0, affinity_mode="spike")[0]
            if not torch.isfinite(q_b1).all():
                raise FloatingPointError(f"nonfinite GPU B1 Q at cache row {index}")
            delta_b1 = _delta(q_cached_np, q_b1.detach().cpu().numpy())
            cached_e0, _ = structured_energy.energy_components(q_cached, adjacency, prepared["gain"])

            feature_results = {}
            energy_results = {}
            for name, q in _q_modes(q_cached, q4, q_b1):
                try:
                    features = structured_energy.pair_features(
                        torch.from_numpy(prepared["component_spikes"][index]).to(device),
                        q, adjacency, prepared["gain"])
                    e0, weight = structured_energy.energy_components(q, adjacency, prepared["gain"])
                    rng_state = torch.random.get_rng_state()
                    try:
                        torch.manual_seed(167001)
                        head = structured_energy.PairRelationMLP().to(device)
                    finally:
                        torch.random.set_rng_state(rng_state)
                    e_theta, residual = head(features, e0, weight)
                    feature_results[name] = {"passed": True, "error": None}
                    energy_results[name] = {
                        "zero_residual_exact": bool(torch.equal(e_theta, e0)
                                                     and torch.count_nonzero(residual).item() == 0),
                        "base_energy_exact_vs_cached_q": bool(torch.equal(e0, cached_e0)),
                        "base_energy_max_abs_vs_cached_q": float((e0 - cached_e0).abs().max().item()),
                        "zero_residual_source_parity": bool(
                            torch.equal(e_theta, cached_e0) and torch.count_nonzero(residual).item() == 0),
                        "energy_sha256": hashlib.sha256(
                            e_theta.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
                        "residual_max_abs": float(residual.detach().abs().max().item()),
                    }
                except ValueError as exc:
                    feature_results[name] = {
                        "passed": False, "error": f"{type(exc).__name__}: {exc}"}
                    energy_results[name] = None

            labels_b4 = qcc_readout.qcc_labels_from_affinity(
                q4.detach().cpu()[None])[0].numpy().astype(np.int32)
            labels_b1 = qcc_readout.qcc_labels_from_affinity(
                q_b1.detach().cpu()[None])[0].numpy().astype(np.int32)
            rows.append({
                "cache_index": index,
                "image_id": prepared["selected_ids"][index],
                "cached_q_sha256": prepared["rows"][index]["q_sha256"],
                "gpu_b4_vs_cached_q": delta_b4,
                "gpu_b1_vs_cached_q": delta_b1,
                "gpu_b4_qcc_foreground_patch_count": int(np.count_nonzero(labels_b4)),
                "gpu_b1_qcc_foreground_patch_count": int(np.count_nonzero(labels_b1)),
                "gpu_b4_vs_b1_q_max_abs": float(torch.max(torch.abs(q4 - q_b1)).item()),
                "pair_features_guard": feature_results,
                "zero_mlp_energy": energy_results,
            })
        del components, q_b4
        torch.cuda.synchronize(device)

    source_state_before = prepared["source_state_sha256"]
    # Rehash the consumed immutable files after arithmetic. No model rollout or
    # model construction occurred in this process.
    for index, row in enumerate(prepared["rows"]):
        if _sha_file(audit.source_cache.CACHE_ROOT / "rows" / f"{index:04d}.npz") != row["source_file_sha256"]:
            raise RuntimeError("source S cache row changed during arithmetic probe")
        if _sha_file(audit.context_cache.CACHE_ROOT / "rows" / f"{index:04d}.npz") != row["context_file_sha256"]:
            raise RuntimeError("context Q/A cache row changed during arithmetic probe")
    if (_sha_file(audit.source_cache.CACHE_ROOT / "manifest.json") != prepared["source_cache_manifest_sha256"]
            or _sha_file(audit.context_cache.CACHE_ROOT / "manifest.json") != prepared["context_cache_manifest_sha256"]):
        raise RuntimeError("source/context cache manifest changed during arithmetic probe")

    all_b4 = all(item["pair_features_guard"]["gpu_b4"]["passed"] for item in rows)
    all_b1 = all(item["pair_features_guard"]["gpu_b1"]["passed"] for item in rows)
    report = {
        "experiment": "SW0167_gpu_b4_b1_arithmetic_probe",
        "status": "complete",
        "probe_fingerprint_sha256": os.environ.get("SW0167_GPU_PROBE_FP"),
        "audit_script_sha256": _sha_file(Path(audit.__file__)),
        "physical_gpu": args.physical_gpu,
        "gpu_visible_device_count": 1,
        "source_seed": 1,
        "ordered_image_ids": prepared["selected_ids"],
        "source_checkpoint_sha256": prepared["source_checkpoint_sha256"],
        "source_cache_manifest_sha256": prepared["source_cache_manifest_sha256"],
        "context_cache_manifest_sha256": prepared["context_cache_manifest_sha256"],
        "source_model_state_sha256": source_state_before,
        "source_proof": _source_proof(prepared),
        "gain": prepared["gain"],
        "rows": rows,
        "gpu_b4_pair_features_all_passed": all_b4,
        "gpu_b1_pair_features_all_passed": all_b1,
        "ground_truth_or_mask_data_read": False,
        "validation_data_read": False,
        "model_rollouts": 0,
        "optimizer_updates": 0,
        "source_and_cache_rows_unchanged": True,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "interpretation": (
            "Arithmetic-only diagnostic. B4 and B1 Q are separately compared to the registered cached GPU-B4 Q; "
            "the 2e-6 production pair_features guard is unchanged. This is not a rollout, training, or performance claim."
        ),
    }
    if not math.isfinite(report["gain"]):
        raise ValueError("source gain is nonfinite")
    _atomic_create_json(args.output, report)
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "gpu_b4_pair_features_all_passed": all_b4,
                      "gpu_b1_pair_features_all_passed": all_b1}, sort_keys=True), flush=True)
    return 0


def _supervisor_main():
    # Import the established owner/lease implementation only after the
    # supervisor has hidden CUDA. These helpers are shared infrastructure;
    # this queue never signals a process other than its own child.
    try:
        import fcntl
    except ImportError as exc:
        raise RuntimeError("GPU arithmetic probe requires Linux flock") from exc
    from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4 as owner

    audit = _audit_module()
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU supervisor initialized CUDA")
    runtime = ROOT / "collaborative_test/SW_0167_structured_relation_energy/results_archive/gpu_arithmetic_probe_20261015"
    state_path, lock_path = runtime / "state.json", runtime / "queue.lock"
    log_path = runtime / "child.log"
    report_path = runtime / "gpu_arithmetic_report.json"
    worker_path = Path(__file__).resolve()
    child = None
    lease = None
    queue_lock = None
    stop_event = threading.Event()
    state_mutex = threading.RLock()

    def fingerprint():
        guard = Path(owner.OWNER_GUARD)
        if _sha_file(guard) != owner.OWNER_GUARD_SHA256:
            raise ValueError("shared GPU owner guard checksum mismatch")
        files = [worker_path, Path(audit.__file__), guard,
                 ROOT / "collaborative_test/SW_0164_foreground_preserving_partition/foreground_signed.py",
                 ROOT / "collaborative_test/SW_0167_structured_relation_energy/structured_energy.py",
                 ROOT / "collaborative_test/SW_0167_structured_relation_energy/qcc_readout.py"]
        dep_fps = audit.source_cache.implementation_fingerprint()
        dep_fps.update(audit.context_cache._fingerprint())
        dep_fps.update(audit.sw164_eval.implementation_fingerprint())
        result = {path.relative_to(ROOT).as_posix(): _sha_file(path) for path in files}
        result.update(dep_fps)
        return result

    code_fp = fingerprint()
    code_fp_sha = hashlib.sha256(json.dumps(code_fp, sort_keys=True, separators=(",", ":"),
                                  allow_nan=False).encode("utf-8")).hexdigest()

    def write_state(status, **fields):
        with state_mutex:
            current = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
            current.update(experiment="SW0167_gpu_arithmetic_probe", status=status,
                           supervisor_pid=os.getpid(), updated=time.time(),
                           implementation_fingerprint=code_fp, **fields)
            temp = state_path.with_name(state_path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
            with temp.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(current, stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, state_path)

    def unlock_lease():
        nonlocal lease
        if lease is not None:
            try:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
            finally:
                lease.close()
                lease = None

    def verify_code():
        if fingerprint() != code_fp:
            raise RuntimeError("GPU arithmetic probe dependencies changed after registration")

    try:
        if torch.cuda.is_initialized() or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("supervisor must remain CPU-only with CUDA hidden")
        runtime.mkdir(parents=True, exist_ok=True)
        queue_lock = lock_path.open("a+")
        try:
            fcntl.flock(queue_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("GPU arithmetic probe already has an owner") from exc
        if state_path.exists() or log_path.exists() or report_path.exists():
            raise FileExistsError("preserving pre-existing arithmetic-probe runtime artifacts")
        write_state("validating_source_cache", ground_truth_or_mask_data_read=False)
        prepared = audit._validate_upstream_and_rows()
        if len(prepared["selected_ids"]) != 8:
            raise ValueError("upstream did not validate exactly eight TRAIN rows")
        source_proof = _source_proof(prepared)
        del prepared

        while not stop_event.is_set():
            gpu, expected_uuid, initial_memory, lease = owner._acquire(stop_event)
            write_state("waiting_stable_gpu", physical_gpu=gpu, gpu_uuid=expected_uuid,
                        initial_memory_mib=initial_memory, source_proof=source_proof)
            stable = owner._stable_idle(
                gpu, expected_uuid, stop_event,
                lambda elapsed, memory: write_state("waiting_stable_gpu", physical_gpu=gpu,
                    gpu_uuid=expected_uuid, stable_seconds=elapsed, memory_used_mib=memory,
                    source_proof=source_proof))
            if not stable:
                unlock_lease()
                if stop_event.is_set():
                    raise RuntimeError("probe stopped while waiting for a stable GPU")
                write_state("waiting_for_exclusive_gpu", reservation_released=True,
                            source_proof=source_proof)
                continue

            # Revalidate all upstream identities after stabilization, then do
            # a final UUID/owner/memory check while holding the lease.
            write_state("revalidating_source_under_lease", physical_gpu=gpu, gpu_uuid=expected_uuid,
                        source_proof=source_proof)
            current = audit._validate_upstream_and_rows()
            current_proof = _source_proof(current)
            if current_proof != source_proof:
                raise RuntimeError("source/cache identity changed between queue admission and launch")
            del current
            verify_code()
            actual_uuid, memory, owners = owner.gpu_guard._gpu_info(gpu)
            if actual_uuid != expected_uuid or owners or int(memory) > owner.MAX_MEMORY_MIB:
                unlock_lease()
                write_state("waiting_for_exclusive_gpu", reservation_released=True,
                            gpu_check={"gpu_uuid": actual_uuid, "memory_used_mib": int(memory),
                                       "owner_pids": list(owners)}, source_proof=source_proof)
                continue
            break
        else:
            raise RuntimeError("probe stopped before acquiring an exclusive GPU")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["SW0167_GPU_PROBE_FP"] = code_fp_sha
        command = [sys.executable, str(worker_path), "--worker", "--physical-gpu", str(gpu),
                   "--output", str(report_path)]
        with log_path.open("x", encoding="utf-8", newline="\n") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                     pass_fds=(lease.fileno(),))
            write_state("running_gpu_arithmetic", child_pid=child.pid, child_pgid=child.pid,
                        physical_gpu=gpu, gpu_uuid=expected_uuid, child_argv=command,
                        source_proof=source_proof, started=time.time())
            deadline = time.monotonic() + 600.0
            while child.poll() is None:
                if time.monotonic() >= deadline:
                    owner._stop_owned(child)
                    raise TimeoutError("GPU arithmetic child exceeded 600 seconds")
                try:
                    uuid_now, used, gpu_owners = owner.gpu_guard._gpu_info(gpu)
                except Exception as exc:
                    owner._stop_owned(child)
                    raise RuntimeError("could not verify leased GPU ownership") from exc
                if uuid_now != expected_uuid or any(pid != child.pid for pid in gpu_owners):
                    owner._stop_owned(child)
                    raise RuntimeError("GPU UUID changed or foreign GPU owner appeared; stopped only owned child")
                proc = owner._proc_record(child.pid)
                if not proc or proc.get("state") == "Z" or str(worker_path) not in proc.get("argv", []):
                    owner._stop_owned(child)
                    raise RuntimeError("owned GPU worker process identity changed")
                write_state("running_gpu_arithmetic", child_pid=child.pid, child_pgid=child.pid,
                            physical_gpu=gpu, gpu_uuid=expected_uuid, memory_used_mib=int(used),
                            source_proof=source_proof, started=time.time())
                time.sleep(2.0)
            returncode = child.wait()
        if returncode != 0:
            raise RuntimeError(f"GPU arithmetic worker failed with exit code {returncode}; see {log_path}")
        verify_code()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (report.get("status") != "complete" or report.get("experiment") != "SW0167_gpu_b4_b1_arithmetic_probe"
                or report.get("probe_fingerprint_sha256") != code_fp_sha
                or report.get("audit_script_sha256") != _sha_file(Path(audit.__file__))
                or report.get("physical_gpu") != gpu or report.get("source_proof") != source_proof
                or len(report["rows"]) != 8
                or report.get("ground_truth_or_mask_data_read") is not False
                or report.get("validation_data_read") is not False
                or report.get("model_rollouts") != 0 or report.get("optimizer_updates") != 0):
            raise ValueError("GPU arithmetic worker report does not match the registered diagnostic scope")
        for index, row in enumerate(report["rows"]):
            if (row.get("cache_index") != index or row.get("image_id") != source_proof["ordered_image_ids"][index]
                    or not isinstance(row.get("gpu_b4_vs_cached_q"), dict)
                    or not isinstance(row.get("gpu_b1_vs_cached_q"), dict)
                    or not isinstance(row.get("pair_features_guard"), dict)
                    or not isinstance(row.get("zero_mlp_energy"), dict)
                    or set(row["pair_features_guard"]) != {"cached_gpu_b4", "gpu_b4", "gpu_b1"}
                    or set(row["zero_mlp_energy"]) != {"cached_gpu_b4", "gpu_b4", "gpu_b1"}):
                raise ValueError(f"GPU arithmetic row {index} failed structural validation")
        write_state("complete", child_pid=child.pid, physical_gpu=gpu, gpu_uuid=expected_uuid,
                    report_sha256=_sha_file(report_path), source_proof=source_proof,
                    gpu_b4_pair_features_all_passed=report["gpu_b4_pair_features_all_passed"],
                    gpu_b1_pair_features_all_passed=report["gpu_b1_pair_features_all_passed"],
                    ground_truth_or_mask_data_read=False, validation_data_read=False,
                    model_rollouts=0, optimizer_updates=0, finished=time.time())
        print(json.dumps({"status": "complete", "report": str(report_path),
                          "report_sha256": _sha_file(report_path), "physical_gpu": gpu}, sort_keys=True))
        return 0
    except BaseException as exc:
        if child is not None and child.poll() is None:
            owner._stop_owned(child)
        try:
            if queue_lock is not None and state_path.parent.exists():
                write_state("failed", error=repr(exc), finished=time.time())
        except BaseException:
            pass
        raise
    finally:
        # Stop only our captured child before releasing the shared lease.
        if child is not None and child.poll() is None:
            owner._stop_owned(child)
        unlock_lease()
        if queue_lock is not None:
            try:
                fcntl.flock(queue_lock.fileno(), fcntl.LOCK_UN)
            finally:
                queue_lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.physical_gpu not in (0, 1, 2, 3) or args.output is None:
            raise SystemExit("worker requires a leased physical GPU and output path")
        raise SystemExit(_run_worker(args))
    raise SystemExit(_supervisor_main())
