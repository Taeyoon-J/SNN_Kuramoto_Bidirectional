import unittest
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SNN_DIR = ROOT / "snn_kuramoto_bidirectional"
for path in (ROOT, SNN_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from snn_kuramoto_bidirectional.error_bound import validate_sinusoidal_gating_sinusoidal_gating
from snn_kuramoto_bidirectional.sinusoidal_gating import sinusoidal_gating


class CenteredDelayedGateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.theta0 = torch.randn(2, 7, 4, requires_grad=True)
        self.theta1 = torch.randn(2, 7, 4, requires_grad=True)
        self.history = [self.theta0, self.theta1]

    def test_legacy_raw_and_sigmoid_are_exactly_unchanged(self):
        delayed_mask = 0.5 * (1.0 + torch.sin(self.theta0.mean(dim=-1)))
        expected_drive = torch.sin(self.theta1) * delayed_mask.unsqueeze(-1)
        expected_raw_gate = delayed_mask
        actual_drive, actual_gate = sinusoidal_gating(self.history, 1, 1, "raw")
        self.assertTrue(torch.equal(actual_drive, expected_drive))
        self.assertTrue(torch.equal(actual_gate, expected_raw_gate))

        sigmoid_drive, sigmoid_gate = sinusoidal_gating(self.history, 1, 1, "sigmoid")
        self.assertTrue(torch.equal(sigmoid_drive, expected_drive))
        self.assertTrue(torch.equal(sigmoid_gate, torch.sigmoid(delayed_mask)))

    def test_centered_mode_is_exact_delayed_centered_mask_and_raw_membrane_gate(self):
        delayed_mask = 0.5 * (1.0 + torch.sin(self.theta0.mean(dim=-1)))
        expected_drive = (2.0 * delayed_mask - 1.0).unsqueeze(-1).expand_as(self.theta1)
        actual_drive, actual_membrane_gate = sinusoidal_gating(
            self.history, 1, 1, "centered_raw"
        )
        self.assertEqual(tuple(actual_drive.shape), (2, 7, 4))
        self.assertEqual(tuple(actual_membrane_gate.shape), (2, 7))
        self.assertTrue(torch.equal(actual_drive, expected_drive))
        self.assertTrue(torch.equal(actual_membrane_gate, delayed_mask))

    def test_centered_delayed_gradient_and_finite_values(self):
        drive, membrane_gate = sinusoidal_gating(self.history, 1, 1, "centered_raw")
        objective = drive.square().mean() + membrane_gate.mean()
        grad = torch.autograd.grad(objective, self.theta0)[0]
        self.assertTrue(torch.isfinite(drive).all())
        self.assertTrue(torch.isfinite(membrane_gate).all())
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.norm()), 0.0)

    def test_signed_mask_preserves_carrier_and_centers_only_modulation(self):
        delayed_mask = 0.5 * (1.0 + torch.sin(self.theta0.mean(dim=-1)))
        expected_drive = torch.sin(self.theta1) * (2.0 * delayed_mask - 1.0).unsqueeze(-1)
        actual_drive, actual_membrane_gate = sinusoidal_gating(
            self.history, 1, 1, "signed_mask"
        )
        self.assertTrue(torch.equal(actual_drive, expected_drive))
        self.assertTrue(torch.equal(actual_membrane_gate, delayed_mask))
        grad = torch.autograd.grad(actual_drive.square().mean(), self.theta0)[0]
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.norm()), 0.0)

    def test_mode_validation_accepts_opt_in_and_rejects_typo(self):
        validate_sinusoidal_gating_sinusoidal_gating("centered_raw")
        validate_sinusoidal_gating_sinusoidal_gating("signed_mask")
        with self.assertRaises(ValueError):
            validate_sinusoidal_gating_sinusoidal_gating("centered")

    def test_model_state_dict_strict_compatibility(self):
        from snn_kuramoto_bidirectional.hyperparameter import S2NetHyperparameters
        from snn_kuramoto_bidirectional.s2net_cls import S2NetCore

        base = dict(num_feature_maps=8, num_regions=256, num_time_steps=4,
                    osc_dim=4, sc=torch.eye(256), graph_mode="static",
                    spike_per_component=True, gate_mode="raw")
        source = S2NetCore(S2NetHyperparameters(**base), device="cpu")
        centered = dict(base, gate_mode="centered_raw")
        target = S2NetCore(S2NetHyperparameters(**centered), device="cpu")
        self.assertEqual(tuple(source.state_dict().keys()), tuple(target.state_dict().keys()))
        target.load_state_dict(source.state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
