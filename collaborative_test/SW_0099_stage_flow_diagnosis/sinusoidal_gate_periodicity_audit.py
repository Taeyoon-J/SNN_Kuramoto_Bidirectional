"""Synthetic representation-sensitivity audit for the SW0099 raw gate."""
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

# Permit running this small script from a temporary directory on the server.
REPO_ROOT = (Path(os.environ["SW0099_REPO_ROOT"]) if "SW0099_REPO_ROOT" in os.environ
             else Path(__file__).resolve().parents[2])
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "snn_kuramoto_bidirectional"),
                str(REPO_ROOT / "collaborative_test/SW_0094_aligned_joint_pilot"),
                str(REPO_ROOT / "collaborative_test")]
from diagnose import sinusoidal_raw_gate
from snn_kuramoto_bidirectional.sinusoidal_gating import sinusoidal_gating


def max_abs(a, b):
    return float((a - b).abs().max())


def actual_gate_and_drive(theta):
    """Evaluate the repository's real single-step function over a history."""
    drives, gates = [], []
    for t in range(theta.shape[1]):
        history = theta[:, :t + 1].transpose(0, 1)
        drive_t, gate_t = sinusoidal_gating(history, t, 2, gate_mode="raw")
        drives.append(drive_t)
        gates.append(gate_t)
    return torch.stack(gates, dim=1), torch.stack(drives, dim=1)


def main():
    # Fixed non-symmetric finite phases: [batch,time,node,component].
    x = torch.arange(1 * 5 * 3 * 4, dtype=torch.float64).reshape(1, 5, 3, 4)
    theta = .17 * x + .31 * torch.sin(.7 * x) - .4
    if not torch.isfinite(theta).all():
        raise AssertionError("synthetic phase history must be finite")

    helper_base_mask, helper_base_drive = sinusoidal_raw_gate(theta)
    base_mask, base_drive = actual_gate_and_drive(theta)

    one_component = theta.clone()
    one_component[..., 0] += 2 * torch.pi
    one_mask, one_drive = actual_gate_and_drive(one_component)

    all_components = theta + 2 * torch.pi
    all_mask, all_drive = actual_gate_and_drive(all_components)

    permutation = torch.tensor([2, 0, 3, 1])
    perm_mask, perm_drive = actual_gate_and_drive(theta[..., permutation])

    helper_deltas = {}
    for name, case in (("baseline", theta), ("single_component_plus_2pi", one_component),
                       ("all_components_plus_2pi", all_components),
                       ("component_permutation", theta[..., permutation])):
        actual_mask, actual_drive = actual_gate_and_drive(case)
        helper_mask, helper_drive = sinusoidal_raw_gate(case)
        helper_deltas[name] = {
            "gate_max_abs_delta": max_abs(actual_mask, helper_mask),
            "drive_max_abs_delta": max_abs(actual_drive, helper_drive),
        }

    result = {
        "audit": "synthetic_raw_sinusoidal_gate_representation_sensitivity",
        "shape_batch_time_nodes_components": list(theta.shape),
        "dtype": str(theta.dtype),
        "delay_steps": 2,
        "phase_values_finite": True,
        "single_component_plus_2pi": {
            "carrier_max_abs_delta": max_abs(theta.sin(), one_component.sin()),
            "gate_max_abs_delta": max_abs(base_mask, one_mask),
            "gated_drive_max_abs_delta": max_abs(base_drive, one_drive),
            "gate_delta_nonzero": bool((base_mask != one_mask).any()),
            "drive_delta_nonzero": bool((base_drive != one_drive).any()),
        },
        "all_components_plus_2pi": {
            "gate_max_abs_delta": max_abs(base_mask, all_mask),
            "gated_drive_max_abs_delta": max_abs(base_drive, all_drive),
        },
        "component_permutation": {
            "gate_max_abs_delta": max_abs(base_mask, perm_mask),
            "gated_drive_after_inverse_permutation_max_abs_delta": max_abs(
                base_drive, perm_drive[..., torch.argsort(permutation)]),
        },
        "helper_vs_actual_function_max_abs_delta": helper_deltas,
        "interpretation_limit": (
            "synthetic mathematical representation-sensitivity only; no checkpoint, "
            "image, or model-performance inference"),
        "gate_helper_sha256": hashlib.sha256(
            Path(__file__).with_name("diagnose.py").read_bytes()).hexdigest(),
        "actual_sinusoidal_gating_source_sha256": hashlib.sha256(
            (REPO_ROOT / "snn_kuramoto_bidirectional/sinusoidal_gating.py").read_bytes()
        ).hexdigest(),
        "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if result["single_component_plus_2pi"]["carrier_max_abs_delta"] > 1e-12:
        raise AssertionError("sin(theta) carrier should be 2pi-periodic")
    if not result["single_component_plus_2pi"]["gate_delta_nonzero"]:
        raise AssertionError("chosen single-component shift should change this raw gate")
    for key in ("gate_max_abs_delta", "gated_drive_max_abs_delta"):
        if result["all_components_plus_2pi"][key] > 1e-12:
            raise AssertionError(f"common 2pi shift should preserve {key}")
    if result["component_permutation"]["gate_max_abs_delta"] > 1e-12:
        raise AssertionError("component permutation should preserve the gate")
    if result["component_permutation"][
            "gated_drive_after_inverse_permutation_max_abs_delta"] > 1e-12:
        raise AssertionError("component permutation should permute the drive equivariantly")
    if max(v["gate_max_abs_delta"] for v in helper_deltas.values()) > 1e-12:
        raise AssertionError("vectorized helper gate differs from actual per-step function")
    if max(v["drive_max_abs_delta"] for v in helper_deltas.values()) > 1e-12:
        raise AssertionError("vectorized helper drive differs from actual per-step function")
    out = Path(__file__).with_name("sinusoidal_gate_periodicity_audit.json")
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
