from __future__ import annotations
import torch

def capture_source_coupling(graph, invoke_rollout):
    """Capture original prepare_coupling tuple[0] and return its tuple unchanged.

    The instance method is restored exactly even if the rollout raises. This is
    instance-local instrumentation; it does not alter the graph class or tuple.
    """
    original = graph.prepare_coupling
    had_instance_method = "prepare_coupling" in graph.__dict__
    instance_method = graph.__dict__.get("prepare_coupling")
    captured = []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        if len(captured):
            raise RuntimeError("SW0162 expected one source coupling preparation per static rollout")
        if not isinstance(result, tuple) or len(result) != 3 or not torch.is_tensor(result[0]):
            raise TypeError("native prepare_coupling must return its registered three-tensor tuple")
        captured.append(result[0].detach().clone())
        return result

    graph.prepare_coupling = capture
    try:
        rollout_output = invoke_rollout()
    finally:
        if had_instance_method:
            graph.prepare_coupling = instance_method
        else:
            delattr(graph, "prepare_coupling")
    if len(captured) != 1:
        raise RuntimeError("SW0162 source rollout did not prepare exactly one coupling tuple")
    return rollout_output, captured[0]
