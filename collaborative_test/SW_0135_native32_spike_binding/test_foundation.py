from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for item in (str(ROOT), str(ROOT / "collaborative_test"), str(HERE)):
    if item not in sys.path:
        sys.path.insert(0, item)

from collaborative_test.SW_0094_aligned_joint_pilot.run import hparams
from collaborative_test.SW_0135_native32_spike_binding.foundation import (
    encode_native32, make_criterion32, wrap_mapped_source,
)
from snn_kuramoto_bidirectional.gamma_initializer import FeaturePatchGammaInitializer
from snn_kuramoto_bidirectional.s2net_cls import S2NetCore


class TinyRegisteredEncoder(nn.Module):
    """Small deterministic stand-in for the registered eight-map encoder."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=1, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(torch.arange(24, dtype=torch.float32).reshape(8, 3, 1, 1) / 24)

    def forward(self, images):
        return self.conv(images)


class Native32FoundationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(13550)

    def test_uint8_rgb_is_scaled_once_and_adaptive_pooled_to_32(self):
        encoder = TinyRegisteredEncoder()
        patcher = FeaturePatchGammaInitializer(grid_size=32)
        rgb = torch.arange(2 * 128 * 128 * 3, dtype=torch.int64).remainder(256)
        rgb = rgb.to(torch.uint8).reshape(2, 128, 128, 3)
        mean = torch.zeros(1, 8, 1, 1)
        std = torch.ones(1, 8, 1, 1)

        gamma = encode_native32(encoder, patcher, mean, std, 3.0, rgb)
        scaled = rgb.permute(0, 3, 1, 2).float() / 255.0
        expected = patcher(encoder(scaled))
        self.assertEqual(tuple(gamma.shape), (2, 8, 1024))
        self.assertTrue(torch.equal(gamma, expected))
        gamma.square().mean().backward()
        self.assertIsNotNone(encoder.conv.weight.grad)
        self.assertTrue(torch.isfinite(encoder.conv.weight.grad).all())
        self.assertGreater(float(encoder.conv.weight.grad.abs().sum()), 0.0)

    def test_registered_statistics_and_rgb_contract_fail_closed(self):
        encoder = TinyRegisteredEncoder()
        patcher = FeaturePatchGammaInitializer(grid_size=32)
        rgb = torch.zeros(1, 128, 128, 3, dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError, "uint8 RGB"):
            encode_native32(encoder, patcher, torch.zeros(8), torch.ones(8), 3., rgb.float())
        with self.assertRaisesRegex(ValueError, "statistics"):
            encode_native32(encoder, patcher, torch.zeros(8), torch.zeros(8), 3., rgb)
        with self.assertRaisesRegex(ValueError, "clipping"):
            encode_native32(encoder, patcher, torch.zeros(1, 8, 1, 1),
                            torch.ones(1, 8, 1, 1), float("nan"), rgb)

    def test_source_core_is_mapped_then_wrapped_with_trainable_native32_graph(self):
        torch.manual_seed(13551)
        source = S2NetCore(hparams("raw"), device="cpu")
        source_state = {key: value.detach().clone() for key, value in source.state_dict().items()}
        wrapped = wrap_mapped_source(source_state, device="cpu")
        core = wrapped.core
        self.assertEqual(core.in_dim, 1024)
        self.assertEqual(core.spike_spatial_grid_size, 32)
        self.assertEqual(core.graph_generator.top_k, 128)
        self.assertEqual(wrapped.arm, "phase")
        self.assertTrue(torch.equal(core.sc, torch.eye(1024)))
        self.assertTrue(any(parameter.requires_grad for parameter in core.graph_generator.parameters()))
        self.assertEqual(tuple(wrapped.a_d.shape), (4,))
        self.assertEqual(tuple(wrapped.a_m.shape), (4,))
        self.assertEqual(tuple(wrapped.b.shape), (4,))
        parent = torch.arange(32).div(2, rounding_mode="floor")
        parent = (parent[:, None] * 16 + parent[None, :]).reshape(-1)
        self.assertTrue(torch.equal(core.kuramoto.omega, source_state["kuramoto.omega"][parent]))
        self.assertTrue(torch.equal(
            core.kuramoto.direction_learner,
            source_state["kuramoto.direction_learner"][parent][:, parent]))
        self.assertTrue(torch.equal(source.state_dict()["kuramoto.omega"], source_state["kuramoto.omega"]))

    def test_native32_criterion_changes_only_registered_coherence_weight(self):
        criterion = make_criterion32()
        self.assertEqual(criterion.plv_coherence_weight, 1.0)
        self.assertEqual(criterion.plv_bimodality_weight, 6.0)
        self.assertEqual(criterion.plv_balance_weight, 10.0)
        self.assertEqual(criterion.plv_collapse_weight, 1.0)
        self.assertEqual(criterion.plv_target_density, .867)
        self.assertEqual(criterion.patch_grid_size, (32, 32))


if __name__ == "__main__":
    unittest.main()
