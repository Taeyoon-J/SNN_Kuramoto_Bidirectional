"""Forward-preserving local phase-gradient policy for SW0176."""
from __future__ import annotations

from contextlib import contextmanager

import torch


@contextmanager
def detach_previous_theta(kuramoto):
    """Detach only theta_prev at each Kuramoto call, preserving forward and all else.

    The registered module, its forward method, parameters, drive, A, coupling,
    spike input, and returned theta tensors are untouched. The hook replaces
    only the first positional input (or the named ``theta_prev`` keyword) with
    a detached view. Removal is guaranteed even when the rollout raises.
    """
    if not isinstance(kuramoto, torch.nn.Module):
        raise TypeError("SW0176 policy requires the registered Kuramoto module")
    calls = {"count": 0}

    def _detach(module, args, kwargs):
        if args:
            theta = args[0]
            if not isinstance(theta, torch.Tensor):
                raise TypeError("Kuramoto theta_prev must be a tensor")
            changed = (theta.detach(), *args[1:])
            calls["count"] += 1
            return changed, kwargs
        if "theta_prev" not in kwargs or not isinstance(kwargs["theta_prev"], torch.Tensor):
            raise TypeError("Kuramoto call must provide tensor theta_prev")
        changed_kwargs = dict(kwargs)
        changed_kwargs["theta_prev"] = changed_kwargs["theta_prev"].detach()
        calls["count"] += 1
        return args, changed_kwargs

    handle = kuramoto.register_forward_pre_hook(_detach, with_kwargs=True)
    try:
        yield calls
    finally:
        handle.remove()
