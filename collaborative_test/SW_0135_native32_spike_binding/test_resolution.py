from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for value in (str(ROOT), str(ROOT / "collaborative_test"), str(HERE)):
    if value not in sys.path:
        sys.path.insert(0, value)

from resolution import (
    GRID, convert_state_dict, geodesic_distance_chunked, grid_distance,
    parent_index, build_core,
)


def dense_log4_reference(z, euclidean, contrast, *, steps, radius, temperature, cap):
    similarity = torch.bmm(z, z.transpose(1, 2)).clamp(-1., 1.)
    distance = euclidean * (1 + F.softplus(contrast) * (1 - similarity))
    distance = torch.where(euclidean <= radius, distance, torch.full_like(distance, cap))
    for _ in range(steps):
        # [B,i,k,1] + [B,1,k,j], reduce the full k axis.
        candidates = distance.unsqueeze(-1) + distance.unsqueeze(1)
        relaxed = -temperature * torch.logsumexp(-candidates / temperature, dim=2)
        distance = torch.minimum(distance, relaxed + temperature * math_log4(z))
    return distance.clamp(max=cap)


def math_log4(reference):
    return torch.log(torch.tensor(4., dtype=reference.dtype, device=reference.device))


class Native32ResolutionTests(unittest.TestCase):
    def test_spatial_parent_map_and_strict_declared_tensor_conversion(self):
        parent = parent_index()
        self.assertEqual(tuple(parent.shape), (1024,))
        self.assertEqual(parent[:4].tolist(), [0, 0, 1, 1])
        self.assertEqual(parent[32:36].tolist(), [0, 0, 1, 1])
        self.assertEqual(parent[64:68].tolist(), [16, 16, 17, 17])
        source = {
            "sc": torch.eye(256),
            "graph_generator.grid_distance": grid_distance(16, 1.),
            "kuramoto.omega": torch.arange(256 * 4).reshape(256, 4).float(),
            "kuramoto.kappa": torch.arange(256 * 4).reshape(256, 4).float(),
            "kuramoto.direction_learner": torch.arange(256 * 256).reshape(256, 256).float(),
            "dendric_layer.tau_n": torch.arange(256 * 4).reshape(256, 4).float(),
            "membrane_layer.tau_m": torch.arange(256).float(),
            "shared.weight": torch.tensor([2.5]),
        }
        target = {
            "sc": torch.empty(1024, 1024),
            "graph_generator.grid_distance": torch.empty(1024, 1024),
            "kuramoto.omega": torch.empty(1024, 4),
            "kuramoto.kappa": torch.empty(1024, 4),
            "kuramoto.direction_learner": torch.empty(1024, 1024),
            "dendric_layer.tau_n": torch.empty(1024, 4),
            "membrane_layer.tau_m": torch.empty(1024),
            "shared.weight": torch.empty(1),
        }
        mapped = convert_state_dict(source, target)
        self.assertTrue(torch.equal(mapped["kuramoto.omega"], source["kuramoto.omega"][parent]))
        self.assertTrue(torch.equal(mapped["kuramoto.direction_learner"],
                                     source["kuramoto.direction_learner"][parent][:, parent]))
        self.assertTrue(torch.equal(mapped["dendric_layer.tau_n"], source["dendric_layer.tau_n"][parent]))
        self.assertTrue(torch.equal(mapped["membrane_layer.tau_m"], source["membrane_layer.tau_m"][parent]))
        self.assertTrue(torch.equal(mapped["shared.weight"], source["shared.weight"]))
        self.assertTrue(torch.equal(mapped["sc"], torch.eye(1024)))
        self.assertEqual(float(mapped["graph_generator.grid_distance"][0, 1]), .5)
        with self.assertRaisesRegex(ValueError, "keys differ"):
            convert_state_dict({**source, "unexpected": torch.ones(1)}, target)

    def test_chunked_log4_operator_matches_dense_values_and_all_gradients(self):
        torch.manual_seed(13511)
        base_z = F.normalize(torch.randn(1, 16, 5, dtype=torch.float64), dim=-1)
        euclidean = grid_distance(4, .5, dtype=torch.float64).unsqueeze(0)
        contrast0 = torch.tensor(.37, dtype=torch.float64)
        weights = torch.randn(1, 16, 16, dtype=torch.float64)
        z_chunk = base_z.clone().requires_grad_()
        contrast_chunk = contrast0.clone().requires_grad_()
        z_dense = base_z.clone().requires_grad_()
        contrast_dense = contrast0.clone().requires_grad_()
        actual = geodesic_distance_chunked(
            z_chunk, euclidean, contrast_chunk, steps=3, radius=1.5,
            temperature=.5, cap=16., chunk=3, log4=True)
        expected = dense_log4_reference(
            z_dense, euclidean, contrast_dense, steps=3, radius=1.5,
            temperature=.5, cap=16.)
        loss_actual = (actual * weights).sum()
        loss_expected = (expected * weights).sum()
        grad_actual = torch.autograd.grad(loss_actual, (z_chunk, contrast_chunk))
        grad_expected = torch.autograd.grad(loss_expected, (z_dense, contrast_dense))
        self.assertTrue(torch.allclose(actual, expected, atol=1e-11, rtol=1e-10))
        for got, want in zip(grad_actual, grad_expected):
            self.assertTrue(torch.allclose(got, want, atol=2e-10, rtol=2e-9))
            self.assertTrue(torch.isfinite(got).all())
        self.assertGreater(float(grad_actual[0].abs().sum()), 0.)
        self.assertGreater(float(grad_actual[1].abs()), 0.)

    def test_native_core_strict_load_maps_source_and_leaves_graph_trainable(self):
        from SW_0094_aligned_joint_pilot.run import hparams
        from snn_kuramoto_bidirectional.s2net_cls import S2NetCore
        torch.manual_seed(13512)
        source = S2NetCore(hparams("raw"), device="cpu").state_dict()
        candidate = build_core(source, device="cpu")
        self.assertEqual(candidate.in_dim, 1024)
        self.assertEqual(candidate.graph_generator.top_k, 128)
        self.assertTrue(any(p.requires_grad for p in candidate.graph_generator.parameters()))
        self.assertTrue(torch.equal(candidate.sc, torch.eye(1024)))
        expected_parent = source["kuramoto.omega"][parent_index()]
        self.assertTrue(torch.equal(candidate.kuramoto.omega, expected_parent))
        expected_dist = grid_distance(32, .5)
        self.assertTrue(torch.equal(candidate.graph_generator.grid_distance.cpu(), expected_dist))
        self.assertEqual(candidate.graph_generator.geodesic_steps, 3)


if __name__ == "__main__":
    unittest.main()
