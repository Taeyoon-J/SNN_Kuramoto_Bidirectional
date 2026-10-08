import unittest

import torch

from resolution import (convert_state_dict, downsample_component_labels,
                        geodesic_distance_chunked, grid_distance, parent_index)


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

    def test_registered_core_strict_loads_both_grids(self):
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        sys.path[:0] = [str(root), str(root / "collaborative_test")]
        from SW_0094_aligned_joint_pilot.run import hparams
        from snn_kuramoto_bidirectional.s2net_cls import S2NetCore
        from resolution import build_core
        source = S2NetCore(hparams("raw"), device="cpu").state_dict()
        control = build_core(source, grid_size=16)
        candidate = build_core(source, grid_size=32)
        self.assertEqual(control.kuramoto.N, 256)
        self.assertEqual(candidate.kuramoto.N, 1024)
        self.assertEqual(candidate.graph_generator.top_k, 128)
        self.assertFalse(any(p.requires_grad for p in candidate.graph_generator.parameters()))
        self.assertTrue(torch.equal(candidate.kuramoto.omega, source["kuramoto.omega"][parent_index()]))

    def test_downsample_vote_ties_and_component_renaming(self):
        labels = torch.zeros((1, 32, 32), dtype=torch.int64)
        labels[0, 0, 0] = 1
        labels[0, 0, 1] = 1
        labels[0, 0, 2] = 2
        labels[0, 0, 3] = 2
        # 2-vs-2 foreground tie: larger full component wins (label 2).
        groups = [[(9,), tuple(range(20))]]
        pooled = downsample_component_labels(labels, groups)
        self.assertEqual(pooled[0, 0, 0].item(), 2)
        # Background wins a 2-vs-2 tie against any foreground label.
        labels[0, 0, 4:6] = 1
        labels[0, 1, 4:6] = 0
        pooled = downsample_component_labels(labels, groups)
        self.assertEqual(pooled[0, 0, 2].item(), 0)
        renamed = torch.where(labels == 1, 2, torch.where(labels == 2, 1, 0))
        renamed_groups = [[groups[0][1], groups[0][0]]]
        renamed_pooled = downsample_component_labels(renamed, renamed_groups)
        self.assertTrue(torch.equal((pooled != 0), (renamed_pooled != 0)))
        self.assertTrue(torch.equal(pooled == 2, renamed_pooled == 1))


if __name__ == "__main__":
    unittest.main()
