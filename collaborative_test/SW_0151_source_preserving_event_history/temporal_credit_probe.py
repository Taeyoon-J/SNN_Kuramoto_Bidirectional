"""GT-free diagnostic for adaptive event-history temporal credit.

This is a disposable analysis helper, not a training objective or admission
test. It compares the sealed native rollout, zero-gain parity, and one caller-
declared positive-kappa rollout with live versus detached auxiliary-state
gradients. The detached control has identical forward values and differs only
in backward credit through the recurrent event/history state.
"""
from __future__ import annotations

import copy
import hashlib
from typing import Any

import torch

from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
from collaborative_test.SW_0151_source_preserving_event_history.adaptive_rollout import adaptive_rollout
from collaborative_test.SW_0151_source_preserving_event_history.event_history import AdaptiveEventHistory
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity


TRACE_KEYS = (
    "component_membrane", "component_spikes", "component_gates",
    "membrane", "spikes", "theta",
)


class DetachedAuxiliaryState(AdaptiveEventHistory):
    """Same forward recurrence, but detach only auxiliary states each step."""

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        result = super().forward(*args, **kwargs)
        result["history"] = result["history"].detach()
        result["previous_event"] = result["previous_event"].detach()
        return result


def _state_digest(module: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().contiguous().cpu()
        h.update(name.encode("utf-8"))
        h.update(str(value.dtype).encode("ascii"))
        h.update(str(tuple(value.shape)).encode("ascii"))
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _q_and_objective(trace: dict[str, torch.Tensor], settle_steps: int):
    q = spike_synchrony_affinity(
        trace["spikes"], trace["component_spikes"],
        settle=settle_steps, affinity_mode="spike")
    n = q.shape[-1]
    offdiag = ~torch.eye(n, device=q.device, dtype=torch.bool)
    objective = q[..., offdiag].mean()
    return q, objective


def _gradient_record(objective: torch.Tensor, helper: AdaptiveEventHistory):
    params = (helper.lambda_logit, helper.kappa_raw)
    grads = torch.autograd.grad(objective, params, allow_unused=True)
    record = {}
    for name, grad in zip(("lambda_logit", "kappa_raw"), grads):
        if grad is None:
            record[name] = {"finite": True, "norm": 0.0, "unused": True}
        else:
            record[name] = {
                "finite": bool(torch.isfinite(grad).all().detach().cpu()),
                "norm": float(torch.linalg.vector_norm(grad).detach().cpu()),
                "unused": False,
            }
    return record


def _configure(helper: AdaptiveEventHistory, kappa: float | torch.Tensor) -> None:
    value = torch.as_tensor(kappa, dtype=helper.kappa_raw.dtype)
    if value.numel() not in (1, 4) or not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
        raise ValueError("kappa must be a finite nonnegative scalar or four-vector")
    if not bool((value > 0).any()):
        raise ValueError("positive-kappa trial requires at least one positive component")
    with torch.no_grad():
        helper.kappa_raw.copy_(value.expand_as(helper.kappa_raw))


def run_temporal_credit_probe(
    wrapped: torch.nn.Module,
    gamma: torch.Tensor,
    *,
    scale: float | torch.Tensor,
    positive_kappa: float | torch.Tensor,
    total_steps: int = 1024,
    live_tail_steps: int = 64,
    settle_steps: int = 512,
) -> dict[str, Any]:
    """Run one deterministic matched-state temporal-credit comparison.

    No gradients are applied and the caller's model, gamma, and RNG state are
    restored/preserved. ``scale`` and ``positive_kappa`` are fixed inputs; this
    function does not calibrate or tune them.
    """
    if gamma.ndim != 3 or gamma.dtype != torch.float32 or not bool(torch.isfinite(gamma).all()):
        raise ValueError("gamma must be finite float32 [B,features,N]")
    if gamma.shape[-1] != wrapped.core.in_dim or gamma.shape[-1] < 2:
        raise ValueError(f"gamma node dimension must equal wrapped.core.in_dim={wrapped.core.in_dim} and be >=2")
    if not 0 <= settle_steps < total_steps:
        raise ValueError("settle_steps must be in [0,total_steps)")
    scale_t = torch.as_tensor(scale)
    if scale_t.requires_grad or not bool(torch.isfinite(scale_t).all()) or bool((scale_t <= 0).any()):
        raise ValueError("scale must be fixed, finite, and positive")

    original_model_digest = _state_digest(wrapped)
    original_gamma = gamma.detach().clone()
    diagnostic_gamma = gamma.detach()
    cpu_rng = torch.random.get_rng_state()
    cuda_devices = [gamma.device.index if gamma.device.index is not None else torch.cuda.current_device()] if gamma.is_cuda else []
    cuda_rng = torch.cuda.get_rng_state(cuda_devices[0]) if cuda_devices else None

    def reset_rng() -> None:
        torch.random.set_rng_state(cpu_rng)
        if cuda_devices:
            torch.cuda.set_rng_state(cuda_rng, cuda_devices[0])

    try:
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            # Native reference and helper-zero rollout share weights, input,
            # seed stream, and source rollout implementation.
            reset_rng()
            with torch.no_grad():
                native = late_rollout(copy.deepcopy(wrapped), diagnostic_gamma, total_steps=total_steps,
                                      live_tail_steps=0)
            zero_helper = AdaptiveEventHistory().to(device=gamma.device)
            reset_rng()
            with torch.no_grad():
                zero = adaptive_rollout(copy.deepcopy(wrapped), diagnostic_gamma, zero_helper,
                                        total_steps=total_steps, live_tail_steps=0,
                                        scale=scale)

            native_q, _ = _q_and_objective(native, settle_steps)
            zero_q, _ = _q_and_objective(zero, settle_steps)
            zero_parity = all(torch.equal(zero[key], native[key]) for key in TRACE_KEYS)
            zero_q_parity = torch.equal(zero_q, native_q)
            zero_events = zero["component_events"].detach().clone()
            del native, zero

            full_helper = AdaptiveEventHistory().to(device=gamma.device)
            detached_helper = DetachedAuxiliaryState().to(device=gamma.device)
            _configure(full_helper, positive_kappa)
            _configure(detached_helper, positive_kappa)

            reset_rng()
            full = adaptive_rollout(copy.deepcopy(wrapped), diagnostic_gamma, full_helper,
                                    total_steps=total_steps, live_tail_steps=live_tail_steps,
                                    scale=scale)
            full_q, full_objective = _q_and_objective(full, settle_steps)
            full_grad = _gradient_record(full_objective, full_helper)
            full_snapshot = {key: full[key].detach().clone() for key in TRACE_KEYS}
            full_q_snapshot = full_q.detach().clone()
            full_events_snapshot = full["component_events"].detach().clone()
            full_history_variance = float(full["component_event_history"].var(unbiased=False).detach().cpu())
            del full_objective, full_q, full, full_helper

            reset_rng()
            detached = adaptive_rollout(copy.deepcopy(wrapped), diagnostic_gamma, detached_helper,
                                        total_steps=total_steps, live_tail_steps=live_tail_steps,
                                        scale=scale)
            detached_q, detached_objective = _q_and_objective(detached, settle_steps)
            detached_grad = _gradient_record(detached_objective, detached_helper)
            control_forward_parity = all(torch.equal(full_snapshot[key], detached[key]) for key in TRACE_KEYS)
            control_q_parity = torch.equal(full_q_snapshot, detached_q)
            flips = int((full_events_snapshot != zero_events).sum().cpu())
            # Measure the preservation contract at the return boundary.
            torch.random.set_rng_state(cpu_rng)
            if cuda_devices:
                torch.cuda.set_rng_state(cuda_rng, cuda_devices[0])
            report = {
                "status": "probe_complete",
                "training_admission": False,
                "total_steps": total_steps,
                "live_tail_steps": live_tail_steps,
                "settle_steps_for_q": settle_steps,
                "scale_fixed_input": torch.as_tensor(scale).detach().cpu().reshape(-1).tolist(),
                "positive_kappa_fixed_input": torch.as_tensor(positive_kappa).detach().cpu().reshape(-1).tolist(),
                "zero_kappa_trace_exact_native": zero_parity,
                "zero_kappa_actual_spike_q_exact_native": zero_q_parity,
                "positive_live_vs_detached_forward_exact": control_forward_parity,
                "positive_live_vs_detached_q_exact": control_q_parity,
                "positive_kappa_event_flips_vs_zero": flips,
                "positive_history_variance": full_history_variance,
                "reference_rollouts_no_grad": True,
                "actual_spike_q_offdiag_mean": {
                    "native": float(native_q[..., ~torch.eye(native_q.shape[-1], device=native_q.device, dtype=torch.bool)].mean().detach().cpu()),
                    "zero_kappa": float(zero_q[..., ~torch.eye(zero_q.shape[-1], device=zero_q.device, dtype=torch.bool)].mean().detach().cpu()),
                    "positive_live": float(full_q_snapshot[..., ~torch.eye(full_q_snapshot.shape[-1], device=full_q_snapshot.device, dtype=torch.bool)].mean().cpu()),
                    "positive_detached_control": float(detached_q[..., ~torch.eye(detached_q.shape[-1], device=detached_q.device, dtype=torch.bool)].mean().detach().cpu()),
                },
                "positive_live_q_gradient": full_grad,
                "positive_detached_control_q_gradient": detached_grad,
                "source_model_state_unchanged": _state_digest(wrapped) == original_model_digest,
                "gamma_unchanged": torch.equal(gamma, original_gamma),
                "caller_rng_restored": torch.equal(torch.random.get_rng_state(), cpu_rng),
            }
            del detached_objective, detached, detached_q, detached_helper
            if cuda_devices:
                report["caller_rng_restored"] = report["caller_rng_restored"] and torch.equal(
                    torch.cuda.get_rng_state(cuda_devices[0]), cuda_rng)
            return report
    finally:
        # fork_rng also restores these; explicit restoration makes this
        # function's preservation contract clear if an exception is raised.
        torch.random.set_rng_state(cpu_rng)
        if cuda_devices:
            torch.cuda.set_rng_state(cuda_rng, cuda_devices[0])
