"""CPU-only validator and immutable recovery receipt for completed SW0168 arms."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import torch
from collaborative_test.SW_0168_end_to_end_connectivity import train

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
QUEUE_ROOT = HERE / "results_archive" / "training_queue_20261015"
ARTIFACT_ROOT = train.TRAIN_ROOT
RUNTIME = HERE / "results_archive" / "training_recovery_20261015"
RECEIPT = RUNTIME / "receipt.json"
EXPERIMENT = "SW0168_completed_training_artifact_recovery"
IDENTITY_ERROR = "RuntimeError('owned training child identity changed')"


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_create(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with tmp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        if path.exists(): raise FileExistsError(path)
        os.link(tmp, path)
    finally:
        if tmp.exists(): tmp.unlink()


def _atomic_replace_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with tmp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists(): tmp.unlink()


def _proc_argv(pid):
    proc = Path("/proc") / str(pid)
    try:
        state = (proc / "stat").read_text().split(") ", 1)[1].split()[0]
        argv = [item.decode(errors="replace") for item in (proc / "cmdline").read_bytes().split(b"\0") if item]
        return None if state == "Z" else argv
    except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError):
        return None


def _active_training_processes():
    procroot = Path("/proc")
    if not procroot.is_dir(): return []
    active = []
    module = "collaborative_test.SW_0168_end_to_end_connectivity.training_queue"
    for path in procroot.iterdir():
        if not path.name.isdigit(): continue
        argv = _proc_argv(int(path.name))
        if not argv or "-m" not in argv: continue
        idx = argv.index("-m")
        if idx + 1 < len(argv) and argv[idx + 1] == module:
            active.append({"pid": int(path.name), "argv": argv})
    return active


def _assert_old_lock_free(arm):
    import fcntl
    path = QUEUE_ROOT / f"{arm}.lock"
    if not path.is_file(): raise FileNotFoundError(f"original SW0168 queue lock missing for {arm}")
    with path.open("a+") as stream:
        try: fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc: raise RuntimeError(f"original SW0168 queue lock still held for {arm}") from exc
        finally:
            try: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            except OSError: pass


def _queue_snapshot(arm):
    path = QUEUE_ROOT / f"{arm}_state.json"
    if not path.is_file(): raise FileNotFoundError(f"original SW0168 {arm} state missing")
    raw = path.read_bytes(); state = json.loads(raw.decode("utf-8"))
    if state.get("experiment") != "SW0168_training_queue" or state.get("arm") != arm:
        raise ValueError("original SW0168 training state identity mismatch")
    if state.get("status") == "failed":
        if state.get("error") != IDENTITY_ERROR:
            raise ValueError("only the known post-completion identity-poll failure can be recovered")
    elif state.get("status") == "complete":
        if not isinstance(state.get("manifest_sha256"), str):
            raise ValueError("completed original state lacks manifest binding")
    else:
        raise RuntimeError(f"original training arm is not terminal: {arm}/{state.get('status')}")
    pid = state.get("supervisor_pid")
    if type(pid) is not int or _proc_argv(pid) is not None:
        raise RuntimeError(f"original training supervisor still appears live for {arm}")
    return state, hashlib.sha256(raw).hexdigest()


def _successful_worker_logs(arm):
    paths = sorted(QUEUE_ROOT.glob(f"{arm}_worker_attempt*.log"))
    if not paths: raise FileNotFoundError(f"SW0168 {arm} worker log missing")
    successful = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        if ('"status": "training_complete"' in text and '"completed_updates": 64' in text
                and f'"arm": "{arm}"' in text):
            successful.append(path)
    if len(successful) != 1:
        raise ValueError(f"expected one successful 64-update worker log for {arm}, got {len(successful)}")
    return successful[0]


def validate_completed_training():
    if torch.cuda.is_initialized() or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("SW0168 training recovery must remain CPU-only")
    live = _active_training_processes()
    if live: raise RuntimeError(f"old SW0168 training owner process is still active: {live}")
    fingerprints = train.implementation_fingerprint()
    arms = {}
    common_bindings = None
    for arm in train.ARMS:
        _assert_old_lock_free(arm)
        state, state_sha = _queue_snapshot(arm)
        artifact_dir = ARTIFACT_ROOT / f"{arm}_seed1"
        manifest_path = artifact_dir / "manifest.json"
        if not manifest_path.is_file(): raise FileNotFoundError(f"completed artifact missing for {arm}")
        manifest_sha = _sha(manifest_path)
        if state.get("status") == "complete" and state.get("manifest_sha256") != manifest_sha:
            raise ValueError("original successful queue state does not bind current manifest")
        validated = train.validate_training_artifact(arm, artifact_dir)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("status") != "training_complete" or manifest.get("completed_updates") != 64
                or manifest.get("arm") != arm or manifest.get("seed") != 1):
            raise ValueError(f"completed SW0168 manifest schema/cursor invalid for {arm}")
        bindings = manifest.get("bindings")
        if not isinstance(bindings, dict) or bindings.get("experiment") != "SW0168_actual_spike_connectivity_training":
            raise ValueError("recovered arm lacks the registered SW0168 training bindings")
        if common_bindings is None:
            common_bindings = {k: v for k, v in bindings.items() if k != "arm"}
        elif {k: v for k, v in bindings.items() if k != "arm"} != common_bindings:
            raise ValueError("completed arms do not share identical source/data/training recipe bindings")
        worker_log = _successful_worker_logs(arm)
        log_hashes = {}
        for path in sorted(QUEUE_ROOT.glob(f"{arm}*.log")):
            log_hashes[path.name] = _sha(path)
        arms[arm] = {
            "original_state": state, "original_state_sha256": state_sha,
            "manifest_sha256": manifest_sha,
            "history_sha256": _sha(artifact_dir / "history.json"),
            "optimizer_sha256": _sha(artifact_dir / "optimizer.pt"),
            "checkpoint_sha256": _sha(artifact_dir / "checkpoint.pt"),
            "boundary_sha256": _sha(artifact_dir / "boundary.ptx"),
            "successful_worker_log": worker_log.name,
            "successful_worker_log_sha256": _sha(worker_log), "queue_log_sha256": log_hashes,
            "validated_completed_updates": int(validated["completed_updates"]),
            "optimizer_step_summary": manifest["optimizer_step_summary"],
            "bindings_sha256": train._json_sha(bindings),
        }
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise RuntimeError("SW0168 recovery validator unexpectedly initialized CUDA")
    return {"experiment": EXPERIMENT, "status": "both_arms_validated_complete",
        "created_unix": time.time(), "source_training_fingerprint": fingerprints,
        "source_training_fingerprint_sha256": train._json_sha(fingerprints),
        "common_bindings_sha256": train._json_sha(common_bindings),
        "arms": arms, "gpu_initialized": False, "ground_truth_validation_read": False,
        "recovery_scope": "CPU validation and lineage receipt only; original states/artifacts preserved"}


def publish_receipt():
    if RECEIPT.exists(): raise FileExistsError("preserving existing SW0168 training recovery receipt")
    payload = validate_completed_training()
    _atomic_create(RECEIPT, payload)
    return payload


def validate_receipt(*, deep=True):
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    if receipt.get("experiment") != EXPERIMENT or receipt.get("status") != "both_arms_validated_complete":
        raise ValueError("SW0168 completed-training recovery receipt identity invalid")
    if receipt.get("source_training_fingerprint") != train.implementation_fingerprint():
        raise ValueError("SW0168 training fingerprint changed after recovery validation")
    for arm in train.ARMS:
        record = receipt.get("arms", {}).get(arm)
        if not isinstance(record, dict): raise ValueError(f"SW0168 recovery receipt lacks {arm}")
        state_path = QUEUE_ROOT / f"{arm}_state.json"
        if _sha(state_path) != record.get("original_state_sha256"):
            raise ValueError(f"SW0168 original training state changed after recovery for {arm}")
        manifest_path = ARTIFACT_ROOT / f"{arm}_seed1" / "manifest.json"
        artifact = manifest_path.parent
        for key, name in (("manifest_sha256", "manifest.json"), ("history_sha256", "history.json"),
                          ("optimizer_sha256", "optimizer.pt"), ("checkpoint_sha256", "checkpoint.pt"),
                          ("boundary_sha256", "boundary.ptx")):
            if _sha(artifact / name) != record.get(key):
                raise ValueError(f"SW0168 recovered training artifact changed for {arm}/{name}")
        if deep:
            validated = train.validate_training_artifact(arm, artifact)
            if validated.get("completed_updates") != 64:
                raise ValueError("SW0168 deep training validation did not confirm 64 updates")
    return receipt


if __name__ == "__main__":
    result = publish_receipt()
    print(json.dumps({"status": result["status"], "receipt": str(RECEIPT),
                      "manifest_sha256": {arm: result["arms"][arm]["manifest_sha256"] for arm in train.ARMS}},
                     sort_keys=True))
