from __future__ import annotations

import unittest

import numpy as np
import torch

from collaborative_test.SW_0167_structured_relation_energy import feature_modes
from collaborative_test.SW_0167_structured_relation_energy import structured_energy


class SplitFeatureModeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(167)
        # Nonuniform signals avoid degenerate all-constant correlation rows.
        self.spikes = torch.tensor(rng.normal(size=(4, 7, 31)).astype(np.float32))
        self.rho = feature_modes.component_rho(self.spikes)
        self.q = self.rho.clamp_min(0).prod(dim=0)
        raw = torch.tensor(rng.uniform(size=(7, 7)).astype(np.float32))
        self.a = 0.5 * (raw + raw.T)
        self.a.fill_diagonal_(0.25)
        self.gain = 7.623262882232666

    def test_same_mode_is_exactly_the_existing_strict_wrapper(self):
        old = structured_energy.pair_features(self.spikes, self.q, self.a, self.gain)
        split = feature_modes.features_from_rho_and_context_q(self.rho, self.q, self.a, self.gain)
        self.assertTrue(torch.equal(old, split))

    def test_b4_context_q_is_explicitly_accepted_without_cross_mode_claim(self):
        q_context = self.q.clone()
        q_context[0, 1] = q_context[1, 0] = 0.125
        split = feature_modes.features_from_rho_and_context_q(self.rho, q_context, self.a, self.gain)
        self.assertEqual(tuple(split.shape), (7, 7, 10))
        expected_rho = self.rho.permute(1, 2, 0)
        expected_rho = 0.5 * (expected_rho + expected_rho.transpose(0, 1))
        self.assertTrue(torch.equal(split[:, :, :4], expected_rho))
        with self.assertRaisesRegex(ValueError, "does not reconstruct"):
            structured_energy.pair_features(self.spikes, q_context, self.a, self.gain)

    def test_split_builder_rejects_bad_dtype_shape_and_symmetry(self):
        with self.assertRaisesRegex(ValueError, "float32"):
            feature_modes.features_from_rho_and_context_q(self.rho.double(), self.q, self.a, self.gain)
        with self.assertRaisesRegex(ValueError, "geometry"):
            feature_modes.features_from_rho_and_context_q(self.rho[:, :6, :6], self.q, self.a, self.gain)
        bad_q = self.q.clone()
        bad_q[0, 1] += 0.1
        with self.assertRaisesRegex(ValueError, "symmetric"):
            feature_modes.features_from_rho_and_context_q(self.rho, bad_q, self.a, self.gain)
        bad_rho = self.rho.clone()
        bad_rho[0, 0, 1] += 0.1
        with self.assertRaisesRegex(ValueError, "rho matrices must be symmetric"):
            feature_modes.features_from_rho_and_context_q(bad_rho, self.q, self.a, self.gain)


if __name__ == "__main__":
    unittest.main()
