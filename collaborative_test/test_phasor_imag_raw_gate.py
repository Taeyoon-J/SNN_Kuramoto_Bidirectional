import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "snn_kuramoto_bidirectional"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from snn_kuramoto_bidirectional.error_bound import (
    validate_sinusoidal_gating_sinusoidal_gating,
)
from snn_kuramoto_bidirectional.sinusoidal_gating import sinusoidal_gating
from snn_kuramoto_bidirectional.hyperparameter import S2NetHyperparameters


class PhasorImagRawGateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4100)
        self.history = [torch.randn(2, 7, 4, dtype=torch.float64) for _ in range(4)]
        self.t = 3
        self.delay = 2

    def evaluate(self, history, mode="phasor_imag_raw"):
        return sinusoidal_gating(history, self.t, self.delay, mode)

    def test_independent_component_phase_lifts_leave_drive_and_gate_unchanged(self):
        base_drive, base_gate = self.evaluate(self.history)
        shifted = [x.clone() for x in self.history]
        generator = torch.Generator().manual_seed(4101)
        lifts = torch.randint(-3, 4, shifted[0].shape, generator=generator)
        for x in shifted:
            x.add_(2 * torch.pi * lifts.to(dtype=x.dtype))
        drive, gate = self.evaluate(shifted)
        self.assertTrue(torch.allclose(drive, base_drive, atol=2e-14, rtol=0))
        self.assertTrue(torch.allclose(gate, base_gate, atol=2e-14, rtol=0))

    def test_component_permutation_preserves_gate_and_permutes_drive(self):
        base_drive, base_gate = self.evaluate(self.history)
        permutation = torch.tensor([2, 0, 3, 1])
        permuted = [x[..., permutation] for x in self.history]
        drive, gate = self.evaluate(permuted)
        self.assertTrue(torch.allclose(gate, base_gate, atol=5e-15, rtol=0))
        self.assertTrue(torch.allclose(drive, base_drive[..., permutation], atol=5e-15, rtol=0))

    def test_single_component_and_exact_cancellation_are_finite(self):
        one_component = [x[..., :1].clone() for x in self.history]
        one_drive, one_gate = self.evaluate(one_component)
        delayed = one_component[self.t - self.delay]
        current = one_component[self.t]
        expected_gate = 0.5 * (1.0 + torch.sin(delayed[..., 0]))
        self.assertTrue(torch.equal(one_gate, expected_gate))
        self.assertTrue(torch.equal(one_drive, torch.sin(current) * expected_gate.unsqueeze(-1)))

        cancelled = [torch.cat((torch.zeros_like(x[..., :1]),
                                torch.full_like(x[..., :1], torch.pi)), dim=-1)
                     for x in self.history]
        cancel_drive, cancel_gate = self.evaluate(cancelled)
        self.assertTrue(torch.isfinite(cancel_drive).all())
        self.assertTrue(torch.isfinite(cancel_gate).all())
        self.assertTrue(torch.allclose(cancel_gate, torch.full_like(cancel_gate, 0.5), atol=1e-16, rtol=0))

    def test_equal_components_reduce_to_legacy_raw_gate(self):
        equal = [x[..., :1].expand(-1, -1, 4).clone() for x in self.history]
        actual_drive, actual_gate = self.evaluate(equal)
        legacy_drive, legacy_gate = self.evaluate(equal, mode="raw")
        self.assertTrue(torch.equal(actual_gate, legacy_gate))
        self.assertTrue(torch.equal(actual_drive, legacy_drive))

    def test_formula_and_raw_membrane_gate(self):
        drive, gate = self.evaluate(self.history)
        delayed = self.history[self.t - self.delay]
        current = self.history[self.t]
        expected_gate = 0.5 * (1.0 + torch.sin(delayed).mean(dim=-1))
        expected_drive = torch.sin(current) * expected_gate.unsqueeze(-1)
        self.assertTrue(torch.equal(gate, expected_gate))
        self.assertTrue(torch.equal(drive, expected_drive))
        self.assertGreaterEqual(float(gate.min()), 0.0)
        self.assertLessEqual(float(gate.max()), 1.0)

    def test_finite_nonzero_gradients_through_current_and_delayed_phases(self):
        history = [x.clone().requires_grad_(True) for x in self.history]
        drive, gate = self.evaluate(history)
        objective = drive.square().mean() + gate.mean()
        current_i = self.t
        delayed_i = max(0, self.t - self.delay)
        grads = torch.autograd.grad(objective, (history[current_i], history[delayed_i]))
        self.assertTrue(torch.isfinite(drive).all())
        self.assertTrue(torch.isfinite(gate).all())
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertGreater(float(torch.stack([g.norm() for g in grads]).norm()), 0.0)
        self.assertLess(max(float(g.abs().max()) for g in grads), 1.0)

    def test_legacy_modes_remain_exact_and_new_mode_is_opt_in(self):
        delayed = self.history[self.t - self.delay]
        current = self.history[self.t]
        raw_mask = 0.5 * (1.0 + torch.sin(delayed.mean(dim=-1)))
        for mode in ("raw", "sigmoid"):
            drive, gate = self.evaluate(self.history, mode)
            self.assertTrue(torch.equal(drive, torch.sin(current) * raw_mask.unsqueeze(-1)))
            expected_gate = raw_mask if mode == "raw" else torch.sigmoid(raw_mask)
            self.assertTrue(torch.equal(gate, expected_gate))
        validate_sinusoidal_gating_sinusoidal_gating("phasor_imag_raw")
        with self.assertRaises(ValueError):
            validate_sinusoidal_gating_sinusoidal_gating("phasor_imag")
        hp = S2NetHyperparameters(gate_mode="phasor_imag_raw", spike_per_component=True)
        self.assertIs(hp.validate(), hp)

    def test_model_state_dict_shape_is_unchanged(self):
        from snn_kuramoto_bidirectional.s2net_cls import S2NetCore

        common = dict(num_feature_maps=8, num_regions=16, num_time_steps=4,
                      osc_dim=4, sc=torch.eye(16), graph_mode="static",
                      spike_per_component=True, dendritic_projection="shared")
        raw = S2NetCore(S2NetHyperparameters(**common, gate_mode="raw"), device="cpu")
        candidate = S2NetCore(
            S2NetHyperparameters(**common, gate_mode="phasor_imag_raw"), device="cpu")
        self.assertEqual(tuple(raw.state_dict()), tuple(candidate.state_dict()))
        candidate.load_state_dict(raw.state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
