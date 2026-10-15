import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from causal_synapse import DECAY, causal_synapse, production_affinity
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity


class CausalSynapseTests(unittest.TestCase):
    def test_recurrence_is_causal_and_matches_scalar_oracle(self):
        s = torch.zeros((1, 4, 2, 6), dtype=torch.float32)
        s[0, 0, 0, 0] = 1.0
        s[0, 0, 0, 3] = 0.5
        z = causal_synapse(s, live_start=6)
        expected = []
        state = 0.0
        for value in (1.0, 0.0, 0.0, 0.5, 0.0, 0.0):
            state = DECAY * state + (1.0 - DECAY) * value
            expected.append(state)
        torch.testing.assert_close(z[0, 0, 0], torch.tensor(expected), rtol=1e-6, atol=1e-7)
        self.assertGreater(float(z[0, 0, 0, 4]), 0.0)  # zero input decays, not holds

        changed_future = s.clone()
        changed_future[..., 4:] = 7.0
        z_changed = causal_synapse(changed_future, live_start=6)
        torch.testing.assert_close(z[..., :4], z_changed[..., :4], rtol=0, atol=0)

    def test_live_tail_credit_starts_at_boundary_and_carry_is_detached(self):
        s = torch.ones((1, 4, 1, 6), requires_grad=True)
        z = causal_synapse(s, live_start=3)
        z[..., 5].sum().backward()
        grad = s.grad[0, 0, 0]
        self.assertTrue(torch.equal(grad[:3], torch.zeros(3)))
        self.assertGreater(float(grad[3]), 0.0)
        self.assertGreater(float(grad[5]), 0.0)
        # The actual recurrence remains differentiable from S in the live tail.
        self.assertTrue(torch.isfinite(grad).all())

    def test_identity_returns_exact_input_and_native_affinity(self):
        torch.manual_seed(170)
        s = torch.rand((2, 4, 5, 12), dtype=torch.float32)
        identity = causal_synapse(s, identity=True)
        self.assertIs(identity, s)
        q_native = spike_synchrony_affinity(
            s.mean(dim=1), components=s, settle=4, affinity_mode="spike")
        q_identity = production_affinity(identity, settle=4)
        self.assertTrue(torch.equal(q_native, q_identity))

    def test_filtered_affinity_uses_existing_production_function(self):
        torch.manual_seed(171)
        s = torch.rand((2, 4, 7, 16), dtype=torch.float32)
        z = causal_synapse(s, live_start=12)
        actual = production_affinity(z, settle=5)
        expected = spike_synchrony_affinity(
            z.mean(dim=1), components=z, settle=5, affinity_mode="spike")
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.isfinite(actual).all())
        self.assertEqual(tuple(actual.shape), (2, 7, 7))

    def test_production_affinity_credit_reaches_live_actual_spike_input(self):
        torch.manual_seed(172)
        s = torch.rand((1, 4, 4, 10), dtype=torch.float32, requires_grad=True)
        z = causal_synapse(s, live_start=6)
        q = production_affinity(z, settle=0)
        # A non-diagonal objective avoids the constant self-correlation terms.
        (q[0, 0, 1] + q[0, 1, 2]).backward()
        self.assertIsNotNone(s.grad)
        self.assertTrue(torch.isfinite(s.grad).all())
        self.assertGreater(float(s.grad[..., 6:].norm()), 0.0)
        self.assertTrue(torch.equal(s.grad[..., :6], torch.zeros_like(s.grad[..., :6])))

    def test_rejects_invalid_geometry_and_boundary(self):
        with self.assertRaises(ValueError):
            causal_synapse(torch.zeros(1, 3, 2, 4))
        with self.assertRaises(ValueError):
            causal_synapse(torch.zeros(1, 4, 2, 4), live_start=5)
        with self.assertRaises(TypeError):
            causal_synapse(torch.zeros(1, 4, 2, 4, dtype=torch.int64))


if __name__ == "__main__":
    unittest.main()
