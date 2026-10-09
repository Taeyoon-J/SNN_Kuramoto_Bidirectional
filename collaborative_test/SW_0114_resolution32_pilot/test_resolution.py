import numpy as np  # Import NumPy before torch on the Windows development runtime.
import unittest

import torch

from resolution import (convert_state_dict, downsample_component_labels,
                        geodesic_distance_chunked, grid_distance, parent_index)
from gamma_cache import cache_rows, hdf5_indices


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

    def test_candidate_core_short_forward_backward_cpu(self):
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        sys.path[:0] = [str(root), str(root / "collaborative_test")]
        from SW_0094_aligned_joint_pilot.run import hparams
        from snn_kuramoto_bidirectional.s2net_cls import S2NetCore
        from resolution import build_core
        torch.set_num_threads(2)
        source = S2NetCore(hparams("raw"), device="cpu").state_dict()
        candidate = build_core(source, grid_size=32, device="cpu")
        gamma = torch.randn(1, 8, 1024)
        groups, spikes, core_out = candidate(gamma, return_core_out=True, num_time_steps=2)
        component_spikes = candidate.last_component_spikes
        self.assertEqual(tuple(component_spikes.shape), (1, 4, 1024, 2))
        loss = spikes.float().mean() + core_out.square().mean()
        loss.backward()
        grads = [p.grad for p in candidate.parameters() if p.requires_grad and p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        from run import predict_frozen
        labels, groups, diag = predict_frozen(candidate, gamma.detach(), 32,
                                               time_steps=2, settle=1)
        self.assertEqual(tuple(labels.shape), (1, 32, 32))
        self.assertEqual(len(groups), 1)
        self.assertIn("empty_images", diag)

    def test_predict_frozen_hands_detached_cpu_traces_to_production_classifier(self):
        import sys
        from pathlib import Path
        from unittest import mock
        root = Path(__file__).resolve().parents[2]
        sys.path[:0] = [str(root), str(root / "collaborative_test"), str(root / "collaborative_test/SW_0114_resolution32_pilot")]
        import run as resolution_run
        from snn_kuramoto_bidirectional import spike_classifier
        from snn_kuramoto_bidirectional.evaluation import spatial_components_to_patch_labels

        pattern = torch.tensor([1., 0., 1., 0.])
        components = torch.zeros((1, 4, 1024, 4), requires_grad=True)
        with torch.no_grad():
            components[:, :, :8, :] = pattern.view(1, 1, 1, 4)
        spikes = components.detach().mean(dim=1).requires_grad_(True)

        class FakeCore:
            last_component_spikes = components

            def __call__(self, gamma, *, return_core_out, num_time_steps):
                self.last_component_spikes = components
                return None, spikes, torch.zeros_like(spikes)

        original = spike_classifier.spike_synchrony_components
        calls = []

        def checked_classifier(activity, *args, **kwargs):
            self.assertEqual(activity.device.type, "cpu")
            self.assertFalse(activity.requires_grad)
            self.assertEqual(kwargs["components"].device.type, "cpu")
            self.assertFalse(kwargs["components"].requires_grad)
            calls.append(True)
            return original(activity, *args, **kwargs)

        expected_groups = original(
            spikes.detach().cpu(), synchrony_threshold=.5, min_group_size=8,
            settle=1, components=components.detach().cpu(),
            background="largest_component", spatial_grid_size=32, affinity_mode="spike")
        expected_labels = spatial_components_to_patch_labels(expected_groups, 32, device="cpu")
        with mock.patch.object(spike_classifier, "spike_synchrony_components", side_effect=checked_classifier):
            labels, groups, _ = resolution_run.predict_frozen(
                FakeCore(), torch.zeros((1, 8, 1024)), 32, time_steps=4, settle=1)
        self.assertTrue(calls)
        self.assertEqual(groups, expected_groups)
        self.assertTrue(torch.equal(labels, expected_labels))

    def test_downsample_vote_ties_and_component_renaming(self):
        labels = torch.zeros((1, 32, 32), dtype=torch.int64)
        labels[0, 0, 0] = 1
        labels[0, 0, 1] = 1
        labels[0, 0, 2] = 2
        labels[0, 0, 3] = 2
        labels[0, 1, 0:2] = 1
        labels[0, 1, 2:4] = 2
        # 2-vs-2 foreground tie: larger full component wins (label 1).
        groups = [[tuple(range(20)), (9,)]]
        pooled = downsample_component_labels(labels, groups)
        self.assertEqual(pooled[0, 0, 0].item(), 1)
        # Background wins a 2-vs-2 tie against any foreground label.
        labels[0, 0, 4:6] = 1
        labels[0, 1, 4:6] = 0
        pooled = downsample_component_labels(labels, groups)
        self.assertEqual(pooled[0, 0, 2].item(), 0)
        renamed = torch.where(labels == 1, 2, torch.where(labels == 2, 1, 0))
        renamed_groups = [[groups[0][1], groups[0][0]]]
        renamed_pooled = downsample_component_labels(renamed, renamed_groups)
        self.assertTrue(torch.equal((pooled != 0), (renamed_pooled != 0)))
        self.assertTrue(torch.equal(pooled == 1, renamed_pooled == 2))

    def test_rgb_indices_are_global_but_cache_rows_are_packed(self):
        ids = [999, 1640, 70639]
        self.assertEqual(hdf5_indices(ids).tolist(), ids)
        self.assertEqual(cache_rows(ids).tolist(), [999, 1000, 69999])
        with self.assertRaisesRegex(ValueError, "reserved validation gap"):
            hdf5_indices([1000])
        for forbidden in (70640, 90640):
            with self.assertRaisesRegex(ValueError, "outside the available non-reserved range"):
                hdf5_indices([forbidden])
        self.assertEqual(hdf5_indices([1320, 1639], allow_validation=True).tolist(), [1320, 1639])

    def test_immutable_sw0097_batch8_reference_artifacts(self):
        from run import registered_source_evaluation
        for seed in (0, 1, 2):
            path, digest, scores = registered_source_evaluation(seed)
            self.assertEqual(digest, __import__("run").SOURCE_EVAL_SHAS[seed])
            for metric in ("fg_ari", "foreground_iou", "matched_object_iou"):
                self.assertEqual(scores["valid_count"][metric], 320)
                self.assertEqual(len(scores["per_image"][metric]), 320)


if __name__ == "__main__":
    unittest.main()
