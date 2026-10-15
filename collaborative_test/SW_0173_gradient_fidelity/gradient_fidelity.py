"""Fixed-direction gradient versus symmetric finite-difference accounting.

This is a measurement helper, not a step-size search: callers provide the
registered directions and epsilon values, and every requested epsilon is
reported without selecting a winner. Module state and RNG streams are restored
even when a perturbed evaluation raises.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from collections.abc import Mapping

import torch


@dataclass(frozen=True)
class ParameterGroup:
    """One unique Parameter and every qualified name that aliases it."""
    name: str
    aliases: tuple[str, ...]
    parameter: torch.nn.Parameter


def parameter_union(named_modules: Mapping[str, torch.nn.Module]) -> tuple[ParameterGroup, ...]:
    """Return a deterministic, alias-aware union of trainable module parameters."""
    if not isinstance(named_modules, Mapping) or not named_modules:
        raise ValueError("named_modules must be a nonempty ordered mapping")
    groups: dict[int, dict] = {}
    for module_name, module in named_modules.items():
        if not isinstance(module_name, str) or not module_name:
            raise ValueError("module names must be nonempty strings")
        if not isinstance(module, torch.nn.Module):
            raise TypeError(f"{module_name} is not a torch module")
        for local_name, parameter in module.named_parameters(recurse=True, remove_duplicate=False):
            if not parameter.requires_grad:
                continue
            qualified = f"{module_name}.{local_name}"
            entry = groups.setdefault(id(parameter), {"parameter": parameter, "names": []})
            if qualified not in entry["names"]:
                entry["names"].append(qualified)
    result = []
    for entry in groups.values():
        aliases = tuple(entry["names"])
        result.append(ParameterGroup(aliases[0], aliases, entry["parameter"]))
    return tuple(result)


def _state_tensors(named_modules):
    """Enumerate state including buffers and aliased state-dict names."""
    tensors = []
    for module_name, module in named_modules.items():
        for key, value in module.state_dict(keep_vars=True).items():
            if not torch.is_tensor(value):
                raise TypeError(f"non-tensor module state at {module_name}.{key}")
            tensors.append((f"{module_name}.{key}", value))
    return tensors


def state_sha256(named_modules) -> str:
    """Hash exact named tensor state on CPU, preserving aliases as names."""
    digest = hashlib.sha256()
    for name, tensor in sorted(_state_tensors(named_modules), key=lambda pair: pair[0]):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(repr(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _rng_snapshot():
    state = {"cpu": torch.random.get_rng_state().clone(), "cuda": None}
    if torch.cuda.is_initialized():
        state["cuda"] = [item.clone() for item in torch.cuda.get_rng_state_all()]
    return state


def _rng_restore(state):
    torch.random.set_rng_state(state["cpu"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _scalar_loss(closure, *, gradients):
    value = closure()
    if not torch.is_tensor(value) or value.numel() != 1:
        raise ValueError("loss closure must return one scalar tensor")
    if not bool(torch.isfinite(value.detach()).all()):
        raise ValueError("loss closure returned a non-finite value")
    if gradients and not value.requires_grad:
        raise ValueError("baseline loss must retain an autograd graph")
    return value.reshape(())


def directional_gradient_check(*, named_modules, directions, epsilons, closure):
    """Compare ``grad(loss)·direction`` with central differences per epsilon.

    ``directions`` maps each group's canonical first qualified name to one
    finite tensor matching that Parameter. Epsilon values are consumed exactly
    as supplied; the result deliberately performs no best-epsilon selection.
    ``closure`` should be a deterministic scalar loss function. CPU/CUDA RNG
    is reset to a common baseline for every evaluation and restored afterward.
    """
    groups = parameter_union(named_modules)
    if not groups:
        raise ValueError("no trainable parameters are present in the module union")
    expected = {group.name for group in groups}
    if not isinstance(directions, Mapping) or set(directions) != expected:
        raise ValueError("direction keys must exactly match the unique parameter union")
    epsilons = tuple(float(value) for value in epsilons)
    if not epsilons or any(not math.isfinite(value) or value <= 0 for value in epsilons):
        raise ValueError("epsilons must be a nonempty sequence of positive finite values")

    parameters = [group.parameter for group in groups]
    direction_values = []
    for group in groups:
        direction = directions[group.name]
        parameter = group.parameter
        if not torch.is_tensor(direction) or direction.shape != parameter.shape:
            raise ValueError(f"direction for {group.name} has the wrong shape")
        if direction.device != parameter.device or direction.dtype != parameter.dtype:
            raise ValueError(f"direction for {group.name} must match parameter device and dtype")
        if not bool(torch.isfinite(direction).all()):
            raise ValueError(f"direction for {group.name} is non-finite")
        direction_values.append(direction.detach().clone())

    state_entries = _state_tensors(named_modules)
    snapshots = [(tensor, tensor.detach().clone()) for _, tensor in state_entries]
    before_sha = state_sha256(named_modules)
    rng = _rng_snapshot()
    try:
        _rng_restore(rng)
        baseline = _scalar_loss(closure, gradients=True)
        grads = torch.autograd.grad(baseline, parameters, allow_unused=True)
        directional = torch.zeros((), device=parameters[0].device, dtype=torch.float64)
        for grad, direction in zip(grads, direction_values):
            if grad is not None:
                directional = directional + (grad.detach().to(torch.float64)
                                              * direction.to(torch.float64)).sum()
        if not bool(torch.isfinite(directional)):
            raise ValueError("autograd directional derivative is non-finite")
        baseline_value = float(baseline.detach().cpu())
        autograd_value = float(directional.cpu())
        # The baseline closure may update buffers. Every perturbation must start
        # from the identical raw state used to define the baseline objective.
        _restore_state(snapshots)

        rows = []
        for epsilon in epsilons:
            with torch.no_grad():
                for (parameter, _), direction in zip(snapshots_for_parameters(parameters), direction_values):
                    parameter.copy_(parameter + epsilon * direction)
            try:
                _rng_restore(rng)
                with torch.no_grad():
                    plus = float(_scalar_loss(closure, gradients=False).cpu())
            finally:
                _restore_state(snapshots)

            with torch.no_grad():
                for (parameter, _), direction in zip(snapshots_for_parameters(parameters), direction_values):
                    parameter.copy_(parameter - epsilon * direction)
            try:
                _rng_restore(rng)
                with torch.no_grad():
                    minus = float(_scalar_loss(closure, gradients=False).cpu())
            finally:
                _restore_state(snapshots)
            estimate = (plus - minus) / (2.0 * epsilon)
            absolute_error = abs(estimate - autograd_value)
            scale = max(abs(estimate), abs(autograd_value), 1e-12)
            rows.append({"epsilon": epsilon, "loss_plus": plus, "loss_minus": minus,
                         "symmetric_difference": estimate,
                         "autograd_dot_direction": autograd_value,
                         "absolute_error": absolute_error,
                         "relative_error": absolute_error / scale})
        return {"baseline_loss": baseline_value,
                "autograd_dot_direction": autograd_value,
                "state_before_sha256": before_sha,
                "state_after_sha256": state_sha256(named_modules),
                "state_restored": before_sha == state_sha256(named_modules),
                "evaluations": rows,
                "epsilon_selection": "none; all supplied values reported in input order"}
    finally:
        _restore_state(snapshots)
        _rng_restore(rng)


def directional_forward_check(*, named_modules, directions, scales, closure):
    """Compare a gradient prediction with one-sided feasible ``theta+s*delta``.

    ``directions`` are the actual caller-supplied projected update vectors.
    The helper never constructs or projects them. All scales are evaluated and
    reported in order, with the same baseline state/RNG; no epsilon is selected.
    """
    groups = parameter_union(named_modules)
    if not groups:
        raise ValueError("no trainable parameters are present in the module union")
    expected = {group.name for group in groups}
    if not isinstance(directions, Mapping) or set(directions) != expected:
        raise ValueError("direction keys must exactly match the unique parameter union")
    scales = tuple(float(value) for value in scales)
    if not scales or any(not math.isfinite(value) or value <= 0 for value in scales):
        raise ValueError("scales must be a nonempty sequence of positive finite values")
    parameters, direction_values = [], []
    for group in groups:
        direction, parameter = directions[group.name], group.parameter
        if not torch.is_tensor(direction) or direction.shape != parameter.shape:
            raise ValueError(f"direction for {group.name} has the wrong shape")
        if direction.device != parameter.device or direction.dtype != parameter.dtype:
            raise ValueError(f"direction for {group.name} must match parameter device and dtype")
        if not bool(torch.isfinite(direction).all()):
            raise ValueError(f"direction for {group.name} is non-finite")
        parameters.append(parameter)
        direction_values.append(direction.detach().clone())

    snapshots = [(tensor, tensor.detach().clone())
                 for _, tensor in _state_tensors(named_modules)]
    before_sha = state_sha256(named_modules)
    rng = _rng_snapshot()
    try:
        _rng_restore(rng)
        baseline = _scalar_loss(closure, gradients=True)
        grads = torch.autograd.grad(baseline, parameters, allow_unused=True)
        directional = torch.zeros((), device=parameters[0].device, dtype=torch.float64)
        for grad, direction in zip(grads, direction_values):
            if grad is not None:
                directional += (grad.detach().to(torch.float64)
                                * direction.to(torch.float64)).sum()
        if not bool(torch.isfinite(directional)):
            raise ValueError("autograd directional derivative is non-finite")
        predicted_derivative = float(directional.cpu())
        baseline_value = float(baseline.detach().cpu())
        _restore_state(snapshots)

        rows = []
        for scale in scales:
            _restore_state(snapshots)
            with torch.no_grad():
                for parameter, direction in zip(parameters, direction_values):
                    parameter.copy_(parameter + scale * direction)
            try:
                _rng_restore(rng)
                with torch.no_grad():
                    actual_value = float(_scalar_loss(closure, gradients=False).cpu())
            finally:
                _restore_state(snapshots)
            predicted_change = scale * predicted_derivative
            actual_change = actual_value - baseline_value
            rows.append({"scale": scale, "loss_at_projected_step": actual_value,
                         "predicted_change": predicted_change,
                         "actual_change": actual_change,
                         "difference": actual_change - predicted_change})
        after_sha = state_sha256(named_modules)
        return {"baseline_loss": baseline_value,
                "autograd_dot_direction": predicted_derivative,
                "state_before_sha256": before_sha, "state_after_sha256": after_sha,
                "state_restored": before_sha == after_sha, "evaluations": rows,
                "difference_scheme": "one-sided feasible forward difference",
                "scale_selection": "none; all supplied values reported in input order"}
    finally:
        _restore_state(snapshots)
        _rng_restore(rng)


def snapshots_for_parameters(parameters):
    """Yield distinct parameter slots; aliases appear once in the direction dot."""
    seen = set()
    for parameter in parameters:
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        yield parameter, None


def _restore_state(snapshots):
    with torch.no_grad():
        for tensor, value in snapshots:
            tensor.copy_(value)
