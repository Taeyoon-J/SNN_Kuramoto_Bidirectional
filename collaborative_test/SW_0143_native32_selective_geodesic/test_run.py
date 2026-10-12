from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]
import torch
import torch.nn as nn
import torch.nn.functional as F

from collaborative_test.SW_0135_native32_spike_binding import resolution
from collaborative_test.SW_0143_native32_selective_geodesic import geodesic_operator as op
from collaborative_test.SW_0143_native32_selective_geodesic import run


def tiny_graph():
    return SimpleNamespace(spatial_rate=nn.Parameter(torch.tensor(0.2)),
        geodesic_contrast=nn.Parameter(torch.tensor(0.7)), geodesic_steps=2,
        geodesic_radius=1.5, geodesic_temperature=.5, geodesic_cap=16.)


def dense_power2(z, e, contrast, steps=2, temperature=.5, radius=1.5, cap=16.):
    sim = torch.bmm(z, z.transpose(1, 2)).clamp(-1., 1.)
    step = e * (1. + F.softplus(contrast) * (1. - sim).square())
    d = torch.where(e <= radius, step, torch.full_like(step, cap))
    corr = temperature * torch.log(torch.tensor(4., dtype=z.dtype, device=z.device))
    for _ in range(steps):
        candidates = d.unsqueeze(-1) + d.unsqueeze(1)
        relaxed = -temperature * torch.logsumexp(-candidates / temperature, dim=2) + corr
        d = torch.minimum(d, relaxed)
    return d.clamp(max=cap)


class SelectiveGeodesicOperatorTests(unittest.TestCase):
    def test_same_batch_reference_is_exact_but_cross_batch_difference_is_descriptive(self):
        torch.manual_seed(1403)
        one = {key: torch.randn(1, 3, dtype=torch.float64) for key in run.TRACE_FIELDS}
        same = {key: value.clone() for key, value in one.items()}
        self.assertEqual(run.assert_same_batch_trace(one, same, "fixture"),
                         {key: True for key in run.TRACE_FIELDS})
        four = {key: torch.cat([value + i * 1e-4 for i in range(4)], dim=0)
                for key, value in same.items()}
        batch_one = {key: value + 2e-4 for key, value in one.items()}
        cross = run.trace_max_abs({key: value[:1] for key, value in four.items()}, batch_one)
        self.assertTrue(all(np.isfinite(value) and value > 0 for value in cross.values()))
        # B1-vs-B4 is not supplied as a pass/fail threshold; exact same-batch
        # parity remains the actual source-preservation guard.
        changed = {key: value.clone() for key, value in one.items()}
        changed["theta"][0, 0] += 1e-8
        with self.assertRaisesRegex(AssertionError, "differs"):
            run.assert_same_batch_trace(one, changed, "R B1")

    def test_R_is_bit_exact_native135_value_and_gradient(self):
        torch.manual_seed(143)
        z = torch.randn(2, 7, 5, dtype=torch.float64, requires_grad=True)
        e = torch.rand(7, 7, dtype=torch.float64)
        e = (e + e.T) / 2
        graph = tiny_graph()
        graph.geodesic_contrast = nn.Parameter(graph.geodesic_contrast.detach().double())
        graph.spatial_rate = nn.Parameter(graph.spatial_rate.detach().double())
        original_contrast = graph.geodesic_contrast
        expected = resolution.geodesic_distance_chunked(z, e, graph.geodesic_contrast,
            steps=2, radius=1.5, temperature=.5, cap=16., chunk=32, log4=True)
        expected_grad = torch.autograd.grad(expected.square().sum(), (z, graph.geodesic_contrast), retain_graph=True)
        op.install_operator(graph, "R")
        actual = graph._geodesic_distance(z, e)
        actual_grad = torch.autograd.grad(actual.square().sum(), (z, graph.geodesic_contrast))
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual_grad[0], expected_grad[0]))
        self.assertTrue(torch.equal(actual_grad[1], expected_grad[1]))
        self.assertIs(graph.geodesic_contrast, original_contrast)

    def test_C_changes_only_effective_contrast_and_matches_linear_operator(self):
        torch.manual_seed(17)
        z = torch.randn(1, 6, 4)
        e = torch.rand(6, 6); e = (e + e.T) / 2
        graph = tiny_graph()
        source_raw = graph.geodesic_contrast.detach().clone()
        op.install_operator(graph, "C")
        got = graph._geodesic_distance(z, e)
        expected = resolution.geodesic_distance_chunked(z, e, torch.tensor(op.inverse_softplus(8.)),
            steps=2, radius=1.5, temperature=.5, cap=16., chunk=32, log4=True)
        self.assertTrue(torch.equal(got, expected))
        self.assertAlmostEqual(float(F.softplus(torch.tensor(op.inverse_softplus(8.)))), 8., places=6)
        self.assertTrue(torch.equal(graph.geodesic_contrast.detach(), source_raw))

    def test_T_matches_dense_squared_dissimilarity_and_has_finite_feature_gradient(self):
        torch.manual_seed(1430)
        z = torch.randn(1, 5, 3, dtype=torch.float64, requires_grad=True)
        e = torch.rand(5, 5, dtype=torch.float64); e = (e + e.T) / 2
        raw = torch.tensor(op.inverse_softplus(8.), dtype=torch.float64)
        got = op.geodesic_distance_power2(z, e, raw, steps=2, radius=1.5,
                                         temperature=.5, cap=16., chunk=2, log4=True)
        expected = dense_power2(z, e, raw, steps=2)
        self.assertTrue(torch.allclose(got, expected, atol=1e-12, rtol=1e-12))
        grad = torch.autograd.grad(got.square().sum(), z)[0]
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.norm()), 0.)

    def test_arm_specs_are_registered_and_invalid_graph_refuses_silent_noop(self):
        self.assertEqual(run._arm_spec("C"), {"effective_contrast": 8.0, "dissimilarity_power": 1})
        self.assertEqual(run._arm_spec("T"), {"effective_contrast": 8.0, "dissimilarity_power": 2})
        with self.assertRaises(ValueError):
            op.install_operator(tiny_graph(), "X")
        graph = tiny_graph(); graph.geodesic_steps = 0
        with self.assertRaises(ValueError):
            op.install_operator(graph, "T")


if __name__ == "__main__":
    unittest.main()
