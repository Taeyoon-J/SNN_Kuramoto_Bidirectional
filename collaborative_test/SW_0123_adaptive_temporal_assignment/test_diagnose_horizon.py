"""Focused contract tests for the paired horizon diagnostic."""
import numpy as np  # Import before torch on the Windows development host.
import unittest

import torch

from collaborative_test.SW_0123_adaptive_temporal_assignment import diagnose_horizon as diagnostic
from collaborative_test.SW_0123_adaptive_temporal_assignment.model import (
    AdaptiveTemporalRGBModel,
    normalized_patch_centers,
)


class TinyCore:
    def __init__(self, horizon):
        self.num_time_steps = horizon
        self.last_component_spikes = None

    def __call__(self, gamma, return_core_out=False, return_theta=False):
        b = gamma.shape[0]
        self.last_component_spikes = (
            torch.rand((b, 4, 256, self.num_time_steps), device=gamma.device) > .82
        ).to(dtype=gamma.dtype)
        return None


class HorizonDiagnosticTests(unittest.TestCase):
    def test_static_gamma_uses_core_horizon_not_patch_axis(self):
        # The real cache is [B,8,256]; 256 is the patch count, never T.
        gamma = torch.zeros((2, 8, 256))
        for horizon in (64, 1024):
            core = TinyCore(horizon)
            traces = diagnostic._rollout_components(core, gamma, min(32, horizon - 1))
            self.assertEqual(tuple(traces.shape), (2, 4, 256, horizon))

    def test_rejects_time_shaped_gamma_and_invalid_settle(self):
        with self.assertRaisesRegex(ValueError, r"static \[B,8,256\]"):
            diagnostic._validate_static_gamma_horizon(torch.zeros((2, 8, 256, 64)), 64, 32)
        with self.assertRaisesRegex(ValueError, "must exceed"):
            diagnostic._validate_static_gamma_horizon(torch.zeros((2, 8, 256)), 32, 32)

    def test_real_model_diagnostic_runs_temporal_and_affinity_paths(self):
        torch.manual_seed(91)
        gamma = torch.zeros((1, 8, 256))
        core = TinyCore(64)
        # Vary channels, patches, and time so the affinity/temporal summaries
        # exercise nonconstant traces rather than only their zero guards.
        model = AdaptiveTemporalRGBModel().eval()
        rgb = torch.rand((1, 3, 128, 128))
        result = diagnostic._one_horizon_batch(
            core, model, gamma, rgb, "legacy_full", 32,
            normalized_patch_centers(), torch.arange(256))
        self.assertEqual(len(result["foreground_fraction_per_image"]), 1)
        self.assertEqual(len(result["foreground_group_count_per_image"]), 1)
        temporal = result["temporal_and_affinity"]
        self.assertTrue(np.isfinite(temporal["settled_temporal_conv_feature_variance"]["median"]))
        self.assertTrue(np.isfinite(temporal["actual_four_component_positive_product_affinity"]
                                   ["offdiagonal_zero_fraction"]))


if __name__ == "__main__":
    unittest.main()
