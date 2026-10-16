"""Exact model/optimizer/RNG snapshots for SW0181 accept-or-restore proposals."""
from __future__ import annotations

import copy
import random
from collections import OrderedDict
from collections.abc import Mapping

import numpy as np
import torch


def _clone_tree(value, *, device=None):
    if isinstance(value, torch.Tensor):
        result = value.detach().clone()
        return result.to(device) if device is not None else result
    if isinstance(value, dict):
        return {key: _clone_tree(item, device=device) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item, device=device) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item, device=device) for item in value)
    return copy.deepcopy(value)


def capture_rng_state():
    """Capture RNG streams without initializing CUDA."""
    return {
        "python": random.getstate(),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda_initialized": bool(torch.cuda.is_initialized()),
        "torch_cuda_all": ([state.clone() for state in torch.cuda.get_rng_state_all()]
                            if torch.cuda.is_initialized() else None),
    }


def restore_rng_state(state):
    if not isinstance(state, dict) or set(state) != {
            "python", "numpy", "torch_cpu", "torch_cuda_initialized", "torch_cuda_all"}:
        raise ValueError("SW0181 RNG snapshot schema mismatch")
    random.setstate(state["python"])
    np.random.set_state(copy.deepcopy(state["numpy"]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state["torch_cuda_all"]
    if state["torch_cuda_initialized"]:
        if cuda_states is None or not torch.cuda.is_initialized():
            raise RuntimeError("cannot restore CUDA RNG without the original initialized CUDA runtime")
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError("SW0181 CUDA RNG device count changed during a proposal")
        torch.cuda.set_rng_state_all(cuda_states)
    elif cuda_states is not None:
        raise ValueError("SW0181 CPU-only RNG snapshot unexpectedly contains CUDA states")


def _module_map(modules):
    if isinstance(modules, Mapping):
        rows = dict(modules)
    else:
        rows = {str(index): module for index, module in enumerate(modules)}
    if not rows or any(not isinstance(name, str) or not name for name in rows):
        raise ValueError("SW0181 requires a named nonempty module collection")
    if any(not isinstance(module, torch.nn.Module) for module in rows.values()):
        raise TypeError("SW0181 transaction entries must be torch modules")
    return rows


def _clone_module_state(module, *, device=None):
    original = module.state_dict()
    cloned = OrderedDict((key, _clone_tree(value, device=device))
                         for key, value in original.items())
    if hasattr(original, "_metadata"):
        cloned._metadata = copy.deepcopy(original._metadata)
    return cloned


def capture_transaction(modules, optimizer, *, offload_to_cpu=True):
    """Deep snapshot model parameters/buffers, Adam moments, and RNG state.

    State tensors are copied (optionally to CPU) so later optimizer writes cannot
    alias or corrupt the rollback point. Gradients are intentionally not stored:
    restoring a rejected proposal clears all parameter gradients.
    """
    if optimizer is None or not hasattr(optimizer, "state_dict"):
        raise TypeError("SW0181 transaction needs a stateful optimizer")
    module_rows = _module_map(modules)
    destination = "cpu" if offload_to_cpu else None
    return {
        "schema": "SW0181_state_transaction_v1",
        "modules": {name: _clone_module_state(module, device=destination)
                    for name, module in module_rows.items()},
        "optimizer": _clone_tree(optimizer.state_dict(), device=destination),
        "rng": capture_rng_state(),
        "offload_to_cpu": bool(offload_to_cpu),
    }


def restore_transaction(snapshot, modules, optimizer, *, restore_rng=True, clear_grad=True):
    """Restore a proposal snapshot strictly; reject changed module/Adam schemas."""
    if not isinstance(snapshot, dict) or snapshot.get("schema") != "SW0181_state_transaction_v1":
        raise ValueError("SW0181 transaction snapshot schema mismatch")
    module_rows = _module_map(modules)
    saved_modules = snapshot.get("modules")
    if not isinstance(saved_modules, dict) or set(saved_modules) != set(module_rows):
        raise ValueError("SW0181 transaction module names changed")
    for name, module in module_rows.items():
        current = module.state_dict()
        saved = saved_modules[name]
        if set(current) != set(saved):
            raise ValueError(f"SW0181 module state schema changed for {name}")
        for key, value in current.items():
            if value.shape != saved[key].shape or value.dtype != saved[key].dtype:
                raise ValueError(f"SW0181 module tensor schema changed for {name}.{key}")
        module.load_state_dict(saved, strict=True)
    saved_optimizer = snapshot.get("optimizer")
    if not isinstance(saved_optimizer, dict):
        raise ValueError("SW0181 optimizer snapshot is malformed")
    try:
        optimizer.load_state_dict(_clone_tree(saved_optimizer))
    except (KeyError, ValueError, RuntimeError, TypeError) as exc:
        raise ValueError("SW0181 optimizer parameter/state schema changed") from exc
    if clear_grad:
        for module in module_rows.values():
            for parameter in module.parameters():
                parameter.grad = None
    if restore_rng:
        restore_rng_state(snapshot["rng"])


def transaction_digest(snapshot):
    """Stable digest over serialized module/optimizer tensors (not RNG secrets)."""
    import hashlib

    digest = hashlib.sha256()
    for section in ("modules", "optimizer"):
        root = snapshot[section]

        def visit(prefix, value):
            if isinstance(value, torch.Tensor):
                digest.update(prefix.encode("utf-8"))
                digest.update(str(value.dtype).encode("ascii"))
                digest.update(str(tuple(value.shape)).encode("ascii"))
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            elif isinstance(value, dict):
                for key in sorted(value, key=str):
                    visit(f"{prefix}/{key}", value[key])
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    visit(f"{prefix}/{index}", item)
            else:
                digest.update(prefix.encode("utf-8"))
                digest.update(repr(value).encode("utf-8"))

        visit(section, root)
    return digest.hexdigest()
