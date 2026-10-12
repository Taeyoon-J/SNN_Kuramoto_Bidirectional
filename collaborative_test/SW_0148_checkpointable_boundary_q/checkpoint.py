"""Transactional logical-update checkpoints for the SW0148 local successor.

One self-verifying container is atomically replaced after each update, so a
crash during serialization leaves the previous boundary usable and never
restores partial microbatch gradients.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import platform
from pathlib import Path
import random
import tempfile
from typing import Any

import numpy as np
import torch


FORMAT_VERSION = 1


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def runtime_signature(device: str | torch.device) -> dict[str, Any]:
    """Capture the environment needed for a same-runtime deterministic resume."""
    device = torch.device(device)
    row: dict[str, Any] = {
        "device_type": device.type,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("cannot bind a CUDA resume checkpoint without CUDA")
        index = torch.cuda.current_device() if device.index is None else device.index
        props = torch.cuda.get_device_properties(index)
        row.update({"device_index": int(index), "device_name": props.name,
                    "device_capability": list(torch.cuda.get_device_capability(index)),
                    "device_total_memory": int(props.total_memory)})
    return row


def capture_rng(device: str | torch.device) -> dict[str, Any]:
    """Use safe primitive/tensor RNG state (no NumPy pickle objects)."""
    dev = torch.device(device)
    state: dict[str, Any] = {
        "python_state": random.getstate(),
        "numpy_state": {
            "algorithm": str(np.random.get_state()[0]),
            "keys": torch.from_numpy(np.random.get_state()[1].copy()),
            "position": int(np.random.get_state()[2]),
            "has_gauss": int(np.random.get_state()[3]),
            "cached_gaussian": float(np.random.get_state()[4]),
        },
        "torch_cpu": torch.get_rng_state().clone(),
    }
    if dev.type == "cuda":
        index = torch.cuda.current_device() if dev.index is None else dev.index
        state["torch_cuda"] = {str(index): torch.cuda.get_rng_state(index).cpu().clone()}
    else:
        state["torch_cuda"] = {}
    return state


def restore_rng(state: dict[str, Any], device: str | torch.device) -> None:
    dev = torch.device(device)
    random.setstate(state["python_state"])
    ns = state["numpy_state"]
    np.random.set_state((ns["algorithm"], ns["keys"].cpu().numpy().astype(np.uint32),
                         int(ns["position"]), int(ns["has_gauss"]),
                         float(ns["cached_gaussian"])))
    torch.set_rng_state(state["torch_cpu"].cpu())
    saved_cuda = state.get("torch_cuda", {})
    if dev.type == "cuda":
        index = torch.cuda.current_device() if dev.index is None else dev.index
        key = str(index)
        if set(saved_cuda) != {key}:
            raise ValueError("checkpoint CUDA RNG device does not match selected device")
        torch.cuda.set_rng_state(saved_cuda[key].cpu(), index)
    elif saved_cuda:
        raise ValueError("CUDA checkpoint RNG cannot resume on CPU")


def _torch_bytes(obj: Any) -> bytes:
    stream = io.BytesIO()
    torch.save(obj, stream)
    return stream.getvalue()


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _write_container(path: Path, payload_bytes: bytes, metadata: dict[str, Any]) -> str:
    """Atomically replace one self-verifying checkpoint container."""
    outer = {"format_version": FORMAT_VERSION,
             "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
             "metadata": metadata,
             "payload_bytes": payload_bytes}
    data = _torch_bytes(outer)
    _atomic_bytes(path, data)
    return hashlib.sha256(data).hexdigest()


def save_boundary(root: str | Path, *, completed_updates: int,
                  wrapped: torch.nn.Module, encoder: torch.nn.Module,
                  optimizer: torch.optim.Optimizer, history: list[dict],
                  bindings: dict[str, Any], runtime: dict[str, Any],
                  device: str | torch.device) -> dict[str, Any]:
    """Atomically commit one fully completed optimizer boundary."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    completed_updates = int(completed_updates)
    if completed_updates < 0 or len(history) != completed_updates:
        raise ValueError("history length must equal completed logical updates")
    payload = {
        "format_version": FORMAT_VERSION,
        "completed_updates": completed_updates,
        "wrapped_state": {k: v.detach().cpu().clone() for k, v in wrapped.state_dict().items()},
        "encoder_state": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "optimizer_schema": optimizer_schema(optimizer),
        "history": history,
        "bindings": bindings,
        "bindings_sha256": canonical_sha(bindings),
        "runtime": runtime,
        "runtime_sha256": canonical_sha(runtime),
        "rng": capture_rng(device),
    }
    data = _torch_bytes(payload)
    checkpoint_path = root / "boundary.ptx"
    metadata = {"completed_updates": completed_updates,
                "bindings": bindings, "bindings_sha256": canonical_sha(bindings),
                "runtime": runtime, "runtime_sha256": canonical_sha(runtime)}
    checkpoint_sha = _write_container(checkpoint_path, data, metadata)
    return {"format_version": FORMAT_VERSION,
            "completed_updates": completed_updates,
            "checkpoint_name": checkpoint_path.name,
            "checkpoint_sha256": checkpoint_sha,
            "payload_sha256": hashlib.sha256(data).hexdigest(),
            "bindings_sha256": canonical_sha(bindings),
            "runtime_sha256": canonical_sha(runtime)}


def optimizer_schema(optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
    return [{"name": group.get("name"), "lr": float(group["lr"]),
             "parameter_shapes": [list(p.shape) for p in group["params"]],
             "parameter_dtypes": [str(p.dtype) for p in group["params"]]}
            for group in optimizer.param_groups]


def load_latest(root: str | Path, *, wrapped: torch.nn.Module,
                encoder: torch.nn.Module, optimizer: torch.optim.Optimizer,
                expected_bindings: dict[str, Any], expected_runtime: dict[str, Any],
                device: str | torch.device, max_updates: int) -> dict[str, Any]:
    """Validate commit, content, lineage and runtime before restoring any state."""
    root = Path(root)
    path = root / "boundary.ptx"
    if not path.is_file():
        raise ValueError("committed boundary checkpoint is missing")
    raw = path.read_bytes()
    checkpoint_sha = hashlib.sha256(raw).hexdigest()
    outer = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    if outer.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported checkpoint container version")
    data = outer.get("payload_bytes")
    if not isinstance(data, bytes) or hashlib.sha256(data).hexdigest() != outer.get("payload_sha256"):
        raise ValueError("checkpoint container payload is corrupt")
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    if outer.get("metadata") != {
        "completed_updates": payload.get("completed_updates"),
        "bindings": payload.get("bindings"),
        "bindings_sha256": payload.get("bindings_sha256"),
        "runtime": payload.get("runtime"),
        "runtime_sha256": payload.get("runtime_sha256"),
    }:
        raise ValueError("checkpoint envelope metadata differs from payload")
    step = payload.get("completed_updates")
    if not isinstance(step, int) or not 0 <= step <= max_updates:
        raise ValueError("invalid committed update count")
    if payload.get("completed_updates") != step or len(payload.get("history", [])) != step:
        raise ValueError("checkpoint update/history boundary mismatch")
    bindings = payload.get("bindings")
    runtime = payload.get("runtime")
    if (payload.get("bindings_sha256") != canonical_sha(bindings)
            or bindings != expected_bindings):
        raise ValueError("checkpoint source/science bindings mismatch")
    if (payload.get("runtime_sha256") != canonical_sha(runtime)
            or runtime != expected_runtime):
        raise ValueError("checkpoint runtime/device settings mismatch")
    if payload.get("optimizer_schema") != optimizer_schema(optimizer):
        raise ValueError("checkpoint optimizer parameter-group schema mismatch")
    # Restore only after every pointer, byte, lineage and schema check passed.
    wrapped.load_state_dict(payload["wrapped_state"], strict=True)
    encoder.load_state_dict(payload["encoder_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    restore_rng(payload["rng"], device)
    return {"completed_updates": step, "history": payload["history"],
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_path": path}


def inspect_committed_metadata(root: str | Path) -> dict[str, Any]:
    """Verify the atomic container and expose small lineage metadata only."""
    path = Path(root) / "boundary.ptx"
    raw = path.read_bytes()
    outer = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    if outer.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported checkpoint container version")
    data = outer.get("payload_bytes")
    if not isinstance(data, bytes) or hashlib.sha256(data).hexdigest() != outer.get("payload_sha256"):
        raise ValueError("checkpoint container payload is corrupt")
    metadata = outer.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("bindings_sha256") != canonical_sha(metadata.get("bindings")):
        raise ValueError("checkpoint metadata lineage digest is invalid")
    if metadata.get("runtime_sha256") != canonical_sha(metadata.get("runtime")):
        raise ValueError("checkpoint metadata runtime digest is invalid")
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    if (metadata.get("completed_updates") != payload.get("completed_updates")
            or metadata.get("bindings") != payload.get("bindings")
            or metadata.get("runtime") != payload.get("runtime")):
        raise ValueError("checkpoint envelope metadata differs from payload")
    return metadata


def is_only_uncommitted_temp(root: str | Path) -> bool:
    """True only for an empty work dir or temp files from our atomic writer."""
    root = Path(root)
    if not root.is_dir() or (root / "boundary.ptx").exists():
        return False
    children = list(root.iterdir())
    return all(child.is_file() and child.name.startswith(".boundary.ptx.")
               and child.name.endswith(".tmp") for child in children)
