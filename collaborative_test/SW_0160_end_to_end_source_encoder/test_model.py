from __future__ import annotations

import unittest

import numpy  # initialize the local BLAS runtime before importing Torch
import torch
from torch import nn

from collaborative_test.SW_0160_end_to_end_source_encoder import model


class _Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3)

    def forward(self, x):
        return self.conv(x)


class _Graph(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 2)
        self.geodesic_contrast = nn.Parameter(torch.tensor(0.25))
        self._sw0143_geodesic_arm = "T"


class _Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.graph_generator = _Graph()
        self.membrane_weight = nn.Parameter(torch.tensor(0.5))


class _Wrapped(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = _Core()
        self.a_d = nn.Parameter(torch.zeros(4))
        self.a_m = nn.Parameter(torch.zeros(4))
        self.b = nn.Parameter(torch.zeros(4))

    def project_integrations_(self):
        return None


class _Foundation:
    def __init__(self):
        self.encoder = _Base()
        self.wrapped = _Wrapped()


class SW0160ModelTests(unittest.TestCase):
    def test_zero_residual_preserves_features_and_candidate_base_receives_credit(self):
        torch.manual_seed(10)
        candidate = _Foundation()
        reference = _Foundation()
        reference.encoder.load_state_dict(candidate.encoder.state_dict())
        x = torch.rand(2, 3, 128, 128)
        expected = reference.encoder(x)
        wrapper = model.install_encoder(candidate, "full_encoder_joint", residual_seed=61)
        actual = wrapper(x)
        self.assertTrue(torch.equal(actual, expected))
        actual.square().mean().backward()
        self.assertIsNotNone(wrapper.base_encoder.conv.weight.grad)
        self.assertGreater(float(wrapper.base_encoder.conv.weight.grad.norm()), 0)
        self.assertGreater(float(wrapper.residual[-1].weight.grad.norm()), 0)

    def test_matched_arm_families_freeze_only_control_base_and_geodesic_contrast(self):
        torch.manual_seed(12)
        outputs = {}
        for arm in model.ARMS:
            foundation = _Foundation()
            model.install_encoder(foundation, arm, residual_seed=91)
            groups = model.trainable_families(foundation, arm)
            self.assertEqual(set(groups), set(model.LEARNING_RATES))
            self.assertFalse(foundation.wrapped.core.graph_generator.geodesic_contrast.requires_grad)
            self.assertEqual(bool(foundation.encoder.base_encoder.conv.weight.requires_grad),
                             arm == "full_encoder_joint")
            self.assertEqual(bool(groups["source_base"]), arm == "full_encoder_joint")
            optimizer = model.make_optimizer(groups)
            outputs[arm] = {group["name"]: group["lr"] for group in optimizer.param_groups}
        self.assertEqual(outputs["frozen_base_joint"], {
            key: value for key, value in outputs["full_encoder_joint"].items() if key != "source_base"})
        self.assertEqual(outputs["full_encoder_joint"]["source_base"], model.LEARNING_RATES["source_base"])
        self.assertNotIn("source_base", outputs["frozen_base_joint"])

    def test_unregistered_arm_rejected(self):
        with self.assertRaisesRegex(ValueError, "unregistered"):
            model.install_encoder(_Foundation(), "source", residual_seed=1)


if __name__ == "__main__":
    unittest.main()
