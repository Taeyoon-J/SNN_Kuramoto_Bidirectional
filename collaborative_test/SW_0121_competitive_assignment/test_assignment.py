"""Synthetic CPU contract checks for SW0121's assignment head and losses."""
import unittest

import torch

from collaborative_test.SW_0121_competitive_assignment.assignment import (
    CompetitiveAssignmentHead, analytic_rgb_terms, fit_channel_rms, patch_features,
)


class CompetitiveAssignmentTests(unittest.TestCase):
    def test_feature_order_is_component_then_settled_time_per_patch(self):
        x = torch.arange(4 * 256 * 32, dtype=torch.float32).reshape(1, 4, 256, 32)
        got = patch_features(x)
        expected = torch.cat([x[0, d, 7] for d in range(4)]).reshape(1, 1, 128)
        self.assertTrue(torch.equal(got[:, 7:8], expected))

    def test_rms_is_train_fitted_with_neutral_scaling_for_inactive_channels(self):
        features = torch.ones(3, 256, 128)
        features[:, :, 0] = 0
        rms = fit_channel_rms(features)
        self.assertTrue(torch.equal(rms[1:], torch.ones_like(rms[1:])))
        self.assertEqual(float(rms[0]), 1.0)
        head = CompetitiveAssignmentHead(rms)
        self.assertFalse(head.channel_rms.requires_grad)
        with torch.no_grad():
            p = head(torch.ones(2, 4, 256, 32))
        self.assertEqual(tuple(p.shape), (2, 256, 11))
        self.assertTrue(torch.allclose(p.sum(-1), torch.ones(2, 256), atol=1e-6))

    def test_reconstruction_and_consistency_match_registered_equations_and_credit(self):
        torch.manual_seed(121)
        logits = torch.randn(1, 256, 11, requires_grad=True)
        p = torch.softmax(logits, dim=-1)
        q = torch.rand(1, 256, 256, requires_grad=True)
        rgb = torch.rand(1, 256, 3, requires_grad=True)
        r, c, details = analytic_rgb_terms(q, p, rgb)
        x = rgb.detach()
        mu = torch.bmm(p.transpose(1, 2), x) / p.sum(dim=1).unsqueeze(-1).clamp_min(1e-8)
        pred = torch.bmm(p, mu)
        variance = x.var(dim=1, unbiased=False).mean(dim=-1).clamp_min(1e-6)
        r_manual = ((pred - x).square().mean(dim=(1, 2)) / variance).mean()
        eye = torch.eye(256, dtype=torch.bool)
        target = torch.bmm(p, p.transpose(1, 2)).detach()
        c_manual = (q[:, ~eye] - target[:, ~eye]).square().mean()
        self.assertTrue(torch.allclose(r, r_manual))
        self.assertTrue(torch.allclose(c, c_manual))
        self.assertTrue(torch.equal(details["affinity_target_detached"], target))
        g_r_p, = torch.autograd.grad(r, p, retain_graph=True)
        g_c_q, = torch.autograd.grad(c, q, retain_graph=True)
        g_c_p = torch.autograd.grad(c, p, allow_unused=True, retain_graph=True)[0]
        g_r_q = torch.autograd.grad(r, q, allow_unused=True, retain_graph=True)[0]
        self.assertTrue(torch.isfinite(g_r_p).all() and float(g_r_p.abs().sum()) > 0)
        self.assertTrue(torch.isfinite(g_c_q).all() and float(g_c_q.abs().sum()) > 0)
        self.assertIsNone(g_c_p)
        self.assertIsNone(g_r_q)
        self.assertIsNone(rgb.grad)  # RGB target is detached.

    def test_diagonal_q_values_do_not_change_consistency(self):
        torch.manual_seed(1211)
        p = torch.softmax(torch.randn(1, 256, 11), dim=-1)
        q = torch.rand(1, 256, 256)
        x = torch.rand(1, 256, 3)
        r1, c1, _ = analytic_rgb_terms(q, p, x)
        changed = q.clone()
        changed[:, torch.arange(256), torch.arange(256)] = 1e6
        r2, c2, _ = analytic_rgb_terms(changed, p, x)
        self.assertTrue(torch.equal(c1, c2))
        self.assertTrue(torch.equal(r1, r2))

    def test_shape_and_probability_contracts_fail_closed(self):
        head = CompetitiveAssignmentHead(torch.ones(128))
        with self.assertRaises(ValueError):
            head(torch.zeros(1, 4, 256, 31))
        with self.assertRaises(ValueError):
            analytic_rgb_terms(torch.zeros(1, 256, 256), torch.full((1, 256, 11), 0.1),
                               torch.zeros(1, 256, 3))
        with self.assertRaises(ValueError):
            analytic_rgb_terms(torch.zeros(1, 256, 256), torch.full((1, 256, 10), 0.1),
                               torch.zeros(1, 256, 3))


if __name__ == "__main__":
    unittest.main()
