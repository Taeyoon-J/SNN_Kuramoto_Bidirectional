"""Pure tensor interventions for SW0054 gate-causality controls."""
import torch


def permute_gate_regions(gamma_wave, membrane_gate, permutation):
    """Permute nodes without changing values within any node/time trajectory."""
    permutation = torch.as_tensor(permutation, dtype=torch.long, device=gamma_wave.device)
    n = gamma_wave.shape[1]
    if gamma_wave.ndim != 3 or membrane_gate.ndim != 2:
        raise ValueError("expected gamma [B,N,D] and membrane gate [B,N]")
    if membrane_gate.shape != gamma_wave.shape[:2] or permutation.shape != (n,):
        raise ValueError("gate/permutation dimensions do not match")
    if not torch.equal(torch.sort(permutation).values, torch.arange(n, device=permutation.device)):
        raise ValueError("permutation must contain every region index exactly once")
    return gamma_wave.index_select(1, permutation), membrane_gate.index_select(1, permutation)


def temporal_mean_gates(gamma_history, membrane_history):
    """Return per-image, per-region temporal means for both gate sites."""
    if gamma_history.ndim != 4 or membrane_history.ndim != 3:
        raise ValueError("expected gamma history [B,T,N,D] and gate history [B,T,N]")
    if gamma_history.shape[:3] != membrane_history.shape:
        raise ValueError("gamma and membrane gate histories must align in B,T,N")
    return gamma_history.mean(dim=1), membrane_history.mean(dim=1)
