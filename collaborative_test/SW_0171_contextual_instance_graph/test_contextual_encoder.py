from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np  # Load NumPy before Torch on Windows to avoid duplicate OpenMP runtimes.
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from collaborative_test.SW_0171_contextual_instance_graph.contextual_encoder import ContextualFeatureEncoder
from collaborative_test.SW_0135_native32_spike_binding.foundation import encode_native32
from snn_kuramoto_bidirectional.gamma_initializer import FeaturePatchGammaInitializer
from snn_kuramoto_bidirectional.input_layer_generator import CNNFeatureEncoder


class ContextualEncoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(81)

    def test_actual_source_kernel_encoder_native32_zero_residual_parity(self):
        base = CNNFeatureEncoder(8, 3, in_channels=3)
        patcher = FeaturePatchGammaInitializer(grid_size=32)
        mean = torch.zeros((1, 8, 1, 1), dtype=torch.float32)
        std = torch.ones_like(mean)
        rgb = torch.randint(0, 256, (2, 128, 128, 3), dtype=torch.uint8)
        source_gamma = encode_native32(base, patcher, mean, std, 3.0, rgb)
        model = ContextualFeatureEncoder(base, seed=171001)
        context_gamma = encode_native32(model, patcher, mean, std, 3.0, rgb)
        self.assertEqual(tuple(context_gamma.shape), (2, 8, 1024))
        self.assertTrue(torch.equal(source_gamma, context_gamma))
        self.assertTrue(all(not p.requires_grad for p in model.base_encoder.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.context.parameters()))
        self.assertEqual(tuple(model.base_encoder.kernels.shape), (8, 3, 3, 3))

    def test_context_initialization_preserves_caller_rng_and_is_reproducible(self):
        base_a = CNNFeatureEncoder(8, 3, in_channels=3)
        base_b = CNNFeatureEncoder(8, 3, in_channels=3)
        base_b.load_state_dict(base_a.state_dict())
        before = torch.random.get_rng_state().clone()
        a = ContextualFeatureEncoder(base_a, seed=171001)
        after = torch.random.get_rng_state()
        b = ContextualFeatureEncoder(base_b, seed=171001)
        self.assertTrue(torch.equal(before, after))
        self.assertEqual(a.context.state_dict().keys(), b.context.state_dict().keys())
        for key in a.context.state_dict():
            self.assertTrue(torch.equal(a.context.state_dict()[key], b.context.state_dict()[key]))

    def test_zero_last_layer_delays_hidden_credit_until_second_update(self):
        model = ContextualFeatureEncoder(CNNFeatureEncoder(8, 3, in_channels=3), seed=171001)
        x = torch.rand((1, 3, 128, 128), dtype=torch.float32)
        optimizer = torch.optim.SGD(model.context.parameters(), lr=0.01)
        first_loss = -model(x).mean()
        first_loss.backward()
        self.assertGreater(float(model.context[-1].weight.grad.norm()), 0.0)
        self.assertTrue(all(p.grad is None or torch.count_nonzero(p.grad) == 0
                            for p in model.context[:8].parameters()))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        second_loss = -model(x).mean()
        second_loss.backward()
        hidden_grads = [p.grad for p in model.context[:8].parameters() if p.grad is not None]
        self.assertTrue(hidden_grads)
        self.assertGreater(sum(float(g.norm()) for g in hidden_grads), 0.0)

    def test_rejects_wrong_rgb_contract(self):
        model = ContextualFeatureEncoder(CNNFeatureEncoder(8, 3, in_channels=3))
        with self.assertRaises(ValueError):
            model(torch.rand((1, 3, 127, 128)))
        with self.assertRaises(TypeError):
            model(torch.rand((1, 3, 128, 128), dtype=torch.float64))
        with self.assertRaises(ValueError):
            model(torch.full((1, 3, 128, 128), 1.1))


if __name__ == "__main__":
    unittest.main()
