import unittest

import torch

from resolution import convert_state_dict, geodesic_distance_chunked, grid_distance, parent_index


class ResolutionTests(unittest.TestCase):
    def test_parent_map_and_pair_axis_replication(self):
        p = parent_index()
        self.assertEqual(tuple(p.shape), (1024,))
        self.assertEqual(p[0].item(), 0)
        self.assertEqual(p[1].item(), 0)
        self.assertEqual(p[2].item(), 1)
        self.assertEqual(p[64].item(), 16)
        src = {
            "sc": torch.eye(256),
            "graph_generator.grid_distance": grid_distance(16, 1.),
            "kuramoto.omega": torch.arange(1024.).view(256, 4),
            "kuramoto.kappa": torch.arange(1024.).view(256, 4),
            "kuramoto.direction_learner": torch.arange(65536.).view(256, 256),
            "dendric_layer.tau_n": torch.arange(1024.).view(256, 4),
            "membrane_layer.tau_m": torch.arange(256.),
            "shared.weight": torch.ones(3, 3),
        }
        tgt = {
            "sc": torch.empty(1024, 1024),
            "graph_generator.grid_distance": torch.empty(1024, 1024),
            "kuramoto.omega": torch.empty(1024, 4),
            "kuramoto.kappa": torch.empty(1024, 4),
            "kuramoto.direction_learner": torch.empty(1024, 1024),
            "dendric_layer.tau_n": torch.empty(1024, 4),
            "membrane_layer.tau_m": torch.empty(1024),
            "shared.weight": torch.empty(3, 3),
        }
        out = convert_state_dict(src, tgt, candidate=True)
        self.assertTrue(torch.equal(out["kuramoto.omega"], src["kuramoto.omega"][p]))
        self.assertTrue(torch.equal(out["kuramoto.direction_learner"], src["kuramoto.direction_learner"][p][:, p]))
        self.assertTrue(torch.equal(out["shared.weight"], src["shared.weight"]))
        self.assertTrue(torch.equal(out["sc"], torch.eye(1024)))
        self.assertTrue(torch.allclose(out["graph_generator.grid_distance"][32, 33], torch.tensor(.5)))

    def test_conversion_rejects_unregistered_shape_change(self):
        src = {"ordinary": torch.ones(2)}
        tgt = {"ordinary": torch.ones(3)}
        with self.assertRaisesRegex(ValueError, "unregistered shape"):
            convert_state_dict(src, tgt, candidate=True)

    def test_chunked_operator_matches_dense_rows(self):
        torch.manual_seed(11)
        z = torch.nn.functional.normalize(torch.randn(1, 16, 3), dim=-1)
        d = grid_distance(4, 1.).unsqueeze(0)
        s = torch.bmm(z, z.transpose(1, 2)).clamp(-1, 1)
        step = d * (1 + torch.nn.functional.softplus(torch.tensor(1.)) * (1 - s))
        step = torch.where(d <= 1.5, step, torch.full_like(step, 16.))
        expected = step
        for _ in range(2):
            through = expected.unsqueeze(2) + expected.unsqueeze(1)
            relaxed = -.5 * torch.logsumexp(-through / .5, dim=-1)
            expected = torch.minimum(expected, relaxed)
        actual = geodesic_distance_chunked(z, d, torch.tensor(1.), steps=2, chunk=3)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_log4_corrects_fourfold_duplicate_sampling(self):
        torch.manual_seed(22)
        z16 = torch.nn.functional.normalize(torch.randn(1, 9, 3), dim=-1)
        d16 = grid_distance(3, 1.).unsqueeze(0)
        coarse = geodesic_distance_chunked(z16, d16, 1., steps=1, chunk=2, log4=False)
        idx = (torch.arange(3)[:, None] * 3 + torch.arange(3)[None, :]).repeat_interleave(2, 0).repeat_interleave(2, 1).reshape(-1)
        z32 = z16[:, idx]
        d32 = d16[:, idx][:, :, idx]
        fine = geodesic_distance_chunked(z32, d32, 1., steps=1, chunk=5, log4=True)
        self.assertTrue(torch.allclose(fine, coarse[:, idx][:, :, idx], atol=2e-6, rtol=2e-6))


if __name__ == "__main__":
    unittest.main()
