"""Scoped SW0174 intervention: stop gradients at native event outputs only."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass
class EventGradientScope:
    module: object
    original: object
    replacement: object
    calls: int = 0

    def install_original(self):
        self.module.act_fun_adp = self.original

    def install_intervention(self):
        self.module.act_fun_adp = self.replacement


@contextmanager
def detach_event_outputs(module):
    """Temporarily detach only `act_fun_adp(x)` output; always restore alias."""
    if not hasattr(module, "act_fun_adp"):
        raise AttributeError("native rollout module has no act_fun_adp alias")
    original = module.act_fun_adp
    if not callable(original):
        raise TypeError("native act_fun_adp alias must be callable")
    calls = {"count": 0}

    def detached_output(value):
        result = original(value)
        if not torch.is_tensor(result):
            raise TypeError("native event function must return a tensor")
        calls["count"] += 1
        return result.detach()

    scope = EventGradientScope(module, original, detached_output)
    module.act_fun_adp = detached_output
    try:
        yield scope
    finally:
        module.act_fun_adp = original
        scope.calls = calls["count"]


def snapshot_forward(value, trace_fields):
    """Copy native forward outputs to CPU for exact paired comparisons."""
    gamma, trace, adjacency, projection, q = value
    return {
        "gamma": gamma.detach().cpu().contiguous().clone(),
        "adjacency": adjacency.detach().cpu().contiguous().clone(),
        "projection": projection.detach().cpu().contiguous().clone(),
        "q": q.detach().cpu().contiguous().clone(),
        "trace": {name: trace[name].detach().cpu().contiguous().clone()
                  for name in trace_fields},
    }


def compare_forward_snapshots(native, isolated):
    trace_exact = {name: torch.equal(native["trace"][name], isolated["trace"][name])
                   for name in native["trace"]}
    checks = {"trace_fields_exact": trace_exact,
              "gamma_exact": torch.equal(native["gamma"], isolated["gamma"]),
              "q_exact": torch.equal(native["q"], isolated["q"]),
              "actual_A_exact": torch.equal(native["adjacency"], isolated["adjacency"]),
              "projection_exact": torch.equal(native["projection"], isolated["projection"])}
    checks["all_trace_fields_exact"] = all(trace_exact.values())
    checks["all_forward_values_exact"] = all(checks[key] for key in
        ("all_trace_fields_exact", "gamma_exact", "q_exact", "actual_A_exact", "projection_exact"))
    return checks
