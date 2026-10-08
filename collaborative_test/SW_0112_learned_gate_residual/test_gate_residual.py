import unittest

import torch
from torch import nn

from gate_residual import SharedGateResidual, actual_gate_binding


def sinusoidal_gating(theta_hist, t, phase_delay_steps, gate_mode="sigmoid"):
    theta = theta_hist[t]
    delayed = theta_hist[max(0, t - phase_delay_steps)]
    gate = 0.5 * (1.0 + torch.sin(delayed.mean(dim=-1)))
    if gate_mode == "raw":
        return torch.sin(theta) * gate.unsqueeze(-1), gate
    return torch.sin(theta) * torch.sigmoid(gate).unsqueeze(-1), torch.sigmoid(gate)


class FakeCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_residual = SharedGateResidual()


class GateResidualTests(unittest.TestCase):
    def test_zero_initialization_is_bitwise_identity(self):
        gate = SharedGateResidual()
        delayed = torch.randn(3, 7, 4)
        base = torch.rand(3, 7)
        self.assertTrue(torch.equal(gate(delayed, base), base))

    def test_residual_is_bounded_and_has_finite_parameter_gradients(self):
        gate = SharedGateResidual()
        with torch.no_grad():
            gate.w.copy_(torch.linspace(-1.0, 1.0, 8))
            gate.b.fill_(0.3)
        delayed = torch.randn(2, 5, 4)
        base = torch.rand(2, 5)
        corrected = gate(delayed, base)
        self.assertTrue(torch.all(corrected >= base.square() - 1e-7))
        self.assertTrue(torch.all(corrected <= 2 * base - base.square() + 1e-7))
        grad = torch.autograd.grad(corrected.sum(), (gate.w, gate.b))
        self.assertTrue(all(torch.isfinite(g).all() for g in grad))

    def test_actual_binding_uses_delayed_history_and_restores_on_error(self):
        core = FakeCore()
        original = sinusoidal_gating
        history = [torch.full((2, 3, 4), value) for value in (0.1, 0.5, 1.0)]
        with actual_gate_binding(core):
            drive, corrected = sinusoidal_gating(history, 2, 2, gate_mode="raw")
        self.assertIs(sinusoidal_gating, original)
        base = 0.5 * (1 + torch.sin(history[0].mean(-1)))
        self.assertTrue(torch.equal(corrected, base))
        self.assertTrue(torch.equal(drive, torch.sin(history[2]) * base.unsqueeze(-1)))
        with self.assertRaises(RuntimeError):
            with actual_gate_binding(core):
                raise RuntimeError("sentinel")
        self.assertIs(sinusoidal_gating, original)

    def test_rejects_wrong_fold_or_component_contract(self):
        gate = SharedGateResidual()
        with self.assertRaises(ValueError):
            gate(torch.randn(2, 3, 5), torch.rand(2, 3))


if __name__ == "__main__":
    unittest.main()
